"""Mesh preview.

Two independent views live here:
  - a 2-D matplotlib "station / custom-polygon" plot, shown before any mesh
    preview exists. Custom-polygon picking (Mesh tab, Refine Survey Area)
    needs a literal 2-D plan-view click target, so this stays matplotlib
    regardless of the 3-D mesh view below -- there's no clean way to pick
    ground-plane points by clicking a perspective 3-D render.
  - a 3-D PyVista view of the built mesh (``model_step<N>.vtu``), shown once
    a preview/build exists. This replaces the old flat z=0/y=0 cross-section
    plots: those couldn't usefully show receiver-level refinement (thousands
    of small cells packed around each station) without either hiding most of
    them behind a hard cap or looking sparse/patchy regardless of how the cap
    was chosen. A real 3-D volume, clipped to a box around the stations and
    cut open to expose the interior, doesn't have that problem -- PyVista/VTK
    render hundreds of thousands of cells natively, so you only lose detail
    by zooming out past it, not by hitting a fixed on-screen budget.
"""
from __future__ import annotations

import math

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton,
    QStackedWidget, QVBoxLayout, QWidget,
)

from .mpl_canvas import HAVE_MPL, MplCanvas

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAVE_PYVISTA = True
except Exception:  # pragma: no cover - environment guard
    HAVE_PYVISTA = False


def cells_in_box(mesh, bounds):
    """The whole cells that overlap an axis-aligned box (x0, x1, y0, y1, z0, z1): what
    ``clip_box(crinkle=True)`` keeps, without its cost.  That took 84 s on a 400,000-cell
    model, whose 3.2 million points (deal.II gives each cell its own) it tests one by one."""
    import numpy as np

    if mesh.n_cells == 0:
        return mesh
    corners = mesh.points[mesh.cell_connectivity]
    starts = mesh.offset[:-1]
    lo, hi = np.minimum.reduceat(corners, starts, axis=0), np.maximum.reduceat(corners, starts, axis=0)
    box_lo, box_hi = np.asarray(bounds[0::2], dtype="float64"), np.asarray(bounds[1::2], dtype="float64")
    keep = np.all((hi > box_lo) & (lo < box_hi), axis=1)      # touching a face isn't overlapping
    return mesh.extract_cells(np.flatnonzero(keep))


if HAVE_PYVISTA:
    class _Interactor(QtInteractor):
        """A 3-D view that is the only native window in its tab.  By default its parents all
        turn native too, right up to the tab page, and those left stale pieces of panels on
        screen when flipping tabs under WSLg.  VTK asks for the window in its constructor,
        so the attribute has to be set at that moment."""

        def winId(self):
            self.setAttribute(Qt.WA_DontCreateNativeAncestors)
            return super().winId()

        # Qt unmaps a native child window the first time it's hidden and never maps it again
        # (measured on Qt 5.15.6 / xcb, with a plain QWidget too), so VTK does both.
        def showEvent(self, event):
            super().showEvent(event)
            self.GetRenderWindow().SetShowWindow(True)

        def hideEvent(self, event):
            super().hideEvent(event)
            self.GetRenderWindow().SetShowWindow(False)


def _nice_ticks(lo: float, hi: float, target_n: int = 4) -> list:
    """Tick positions spanning [lo, hi], step size 1/2/5 x 10**k -- "rounded
    to the nearest 10^N, N chosen for the number" (the same spirit as
    ``mesh_design.round_nice``, generalised to a whole tick sequence rather
    than a single value)."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw_step = span / target_n
    exp = math.floor(math.log10(raw_step))
    base = raw_step / (10 ** exp)
    mult = 1 if base < 1.5 else 2 if base < 3.5 else 5 if base < 7.5 else 10
    step = mult * (10 ** exp)
    start = math.ceil(lo / step) * step
    ticks, v, guard = [], start, 0
    while v <= hi + step * 1e-6 and guard < 200:
        ticks.append(v)
        v += step
        guard += 1
    return ticks or [lo]


def _fmt_km(v_m: float, step_m: float) -> str:
    """Axis tick text: metres -> km, with just enough decimals for the step."""
    step_km = max(abs(step_m), 1e-9) / 1000.0
    decimals = 0 if step_km >= 1 else 1 if step_km >= 0.1 else 2
    return f"{v_m / 1000.0:.{decimals}f}"


class Mesh3DView(QWidget):
    """PyVista render of one ``model_step<N>.vtu``.

    deal.II's octree mesh is solid, so its outer surface alone hides all the
    refinement structure. This clips to a box around the stations (or shows
    the whole model) and, optionally, additionally cuts that box open along
    one axis through its centre so the interior is visible -- either a fixed
    cutaway (an opaque cut face, plus an optional translucent "ghost" of the
    full clipped box for context) or an "Interactive Slice": a single thin
    cut surface you drag through the mesh by hand (PyVista's plane-widget
    slice, not a clip -- computing just the cut surface stays cheap to
    recompute on every drag, unlike re-clipping the whole solid volume).
    """

    #: colour scheme when "colour by refinement level" is on -- reversed
    #: Yellow-Orange-Red (so it reads red -> orange -> yellow, lightest at
    #: the coarsest/least-refined end), discretised to exactly one colour per
    #: integer level actually present so the colourbar has that many entries.
    _LEVEL_CMAP = "YlOrRd_r"

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._plotter = None
        if not HAVE_PYVISTA:
            msg = QLabel(
                "3-D mesh view needs pyvista + pyvistaqt -- see environment.yml "
                "(conda env update -f environment.yml)")
            msg.setWordWrap(True)
            lay.addWidget(msg)
        self._mesh = None            # pv.UnstructuredGrid, full/unclipped
        self._bounds = None          # (xmin,xmax,ymin,ymax,zmin,zmax) or None -> whole model
        self._cutaway = True
        self._ghost = False
        self._axis = "y"
        self._interactive = False
        self._color_by_level = False
        self._hide_air = False
        self._stations: list = []    # [[x_m, y_m], ...], drawn at z=0
        self._has_rendered = False   # only auto-frame the camera on the very first render
        self._framed_bounds = "__unset__"   # re-frame whenever the *configured* extent actually
                                            # changes (Extent/N/Max Depth) -- but not for style-only
                                            # toggles (cutaway/ghost/colour/hide-air), which keep
                                            # whatever camera the user has now

    @property
    def plotter(self):
        """The 3-D view, created on first use.  Panels are built before they join the main
        window, and that step would destroy a native window made any earlier."""
        if self._plotter is None and HAVE_PYVISTA:
            # no auto_update: its timer re-renders 5 times a second for ever, and a render of a
            # 400,000-cell model takes 0.55 s under WSLg, which starved the GUI after a Build.
            # Everything that changes the scene calls render() itself.
            self._plotter = _Interactor(self, auto_update=False)
            self.layout().addWidget(self._plotter.interactor)
        return self._plotter

    def load(self, vtu_path: str):
        if not HAVE_PYVISTA:
            return
        self._mesh = pv.read(vtu_path)

    def clear(self):
        self._mesh = None
        if self._plotter is not None:
            self._plotter.clear()
            self._plotter.clear_plane_widgets()

    def configure(self, *, bounds=None, cutaway: bool = True, ghost: bool = False,
                 axis: str = "y", interactive: bool = False,
                 color_by_level: bool = False, hide_air: bool = False,
                 stations: list | None = None):
        """Set view options and (re)render in one go."""
        self._bounds = bounds
        self._cutaway = cutaway
        self._ghost = ghost
        self._axis = axis
        self._interactive = interactive
        self._color_by_level = color_by_level
        self._hide_air = hide_air
        if stations is not None:
            self._stations = stations
        self.render()

    def _normal(self):
        return {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}[self._axis]

    def _style_kwargs(self, m) -> dict:
        """Surface/edge/colour kwargs for ``add_mesh``/``add_mesh_slice``.

        Off: plain light-grey surface, black edges -- the mesh shape reads
        clearly with nothing competing for attention. On: discretised so the
        colourbar has exactly one entry per integer refinement level present
        in *m* (not a smooth 0-255 gradient), computed from *m* itself so it
        reflects whatever's actually in view after clipping/hiding air.
        """
        if not self._color_by_level:
            return dict(color="lightgray", edge_color="black", show_edges=True, style="surface")
        import numpy as np

        levels = np.asarray(m.point_data["level"])
        lo, hi = int(round(levels.min())), int(round(levels.max()))
        n = max(hi - lo + 1, 1)
        return dict(scalars="level", cmap=self._LEVEL_CMAP, n_colors=n,
                   clim=[lo - 0.5, hi + 0.5], show_edges=True, style="surface")

    def render(self):
        if not HAVE_PYVISTA or self._mesh is None:
            return
        self.plotter.clear()
        self.plotter.clear_plane_widgets()   # else a disabled slice stays draggable, just invisible
        m = self._mesh
        if self._hide_air:
            m = m.threshold(value=(1.5, 1.0e9), scalars="material")
        if self._bounds is not None:
            m = cells_in_box(m, self._bounds)
        if m.n_cells == 0:
            self.plotter.add_text("nothing in this extent", font_size=10, render=False)
        else:
            style = self._style_kwargs(m)
            if self._interactive:
                self.plotter.add_mesh_slice(m, normal=self._normal(), **style)
            elif self._cutaway:
                b = m.bounds
                origin = (0.5 * (b[0] + b[1]), 0.5 * (b[2] + b[3]), 0.5 * (b[4] + b[5]))
                if self._ghost:
                    self.plotter.add_mesh(m, opacity=0.12, show_scalar_bar=False, render=False, **style)
                cut = m.clip(normal=self._normal(), origin=origin, invert=True, crinkle=True)
                self.plotter.add_mesh(cut, render=False, **style)
            else:
                self.plotter.add_mesh(m, render=False, **style)
            self._add_axes(m.bounds)
        self._add_stations(m.bounds[4] if m.n_cells else self._mesh.bounds[4])
        if not self._has_rendered or self._bounds != self._framed_bounds:
            self._orient_camera(m.bounds if m.n_cells else self._mesh.bounds)
            self._has_rendered = True
            self._framed_bounds = self._bounds
        self.plotter.render()

    def _add_stations(self, z_top: float):
        """Station locations as black-edged circles, just above *z_top* (the
        shallow/upper bound of whatever's currently displayed, z is +down).

        Sitting exactly at the true z=0 surface would embed them inside the
        mesh's own solid cells -- air is opaque too unless "Hide Air" is on --
        so any marker not at the outer boundary of the displayed volume is
        occluded, however slightly it's offset. Floating them just above the
        very top of the current extent keeps them outside all solid geometry
        regardless of extent/cutaway/hide-air settings.
        """
        if not self._stations:
            return
        z = z_top - 0.03 * max(abs(z_top), 1000.0)
        pts = [[float(x), float(y), z] for x, y in self._stations]
        poly = pv.PolyData(pts)
        # two passes on the same points, black behind + a smaller red on top,
        # to fake an "edged circle" marker -- distinct names, else the second
        # add_mesh silently replaces the first as the same auto-named actor
        self.plotter.add_mesh(poly, color="black", point_size=14, name="stations_edge",
                              render_points_as_spheres=False, style="points", render=False)
        self.plotter.add_mesh(poly, color="red", point_size=9, name="stations_fill",
                              render_points_as_spheres=False, style="points", render=False)

    def _add_axes(self, bounds):
        """A bounding-box outline with tick labels rounded to a nice 1/2/5 x
        10**k step per axis (``_nice_ticks``) -- North/East/Down in km, off
        the corner nearest the default camera so they read without digging
        through the mesh."""
        xlo, xhi, ylo, yhi, zlo, zhi = bounds
        outline = pv.Box(bounds=bounds).outline()
        self.plotter.add_mesh(outline, color="black", line_width=1, render=False)
        corner = [xlo, yhi, zlo]   # south / east / shallow -- nearest the default camera
        pts, labels = [], []
        for lo, hi, idx in ((xlo, xhi, 0), (ylo, yhi, 1), (zlo, zhi, 2)):
            ticks = _nice_ticks(lo, hi, target_n=4)
            step = (ticks[1] - ticks[0]) if len(ticks) > 1 else (hi - lo)
            for t in ticks:
                p = list(corner)
                p[idx] = t
                pts.append(p)
                labels.append(_fmt_km(t, step))
        self.plotter.add_point_labels(
            pts, labels, font_size=12, text_color="black", shape=None,
            always_visible=True, show_points=True, point_size=4, point_color="black", render=False)

    def _orient_camera(self, bounds):
        """North (+X) up, East (+Y) right, Z+ down -- tilted from above, and
        angled toward whichever cutaway/slice axis is active so its open face
        stays in view (the axis with no cutaway falls back to the same angle
        the default Y-axis cutaway uses)."""
        xlo, xhi, ylo, yhi, zlo, zhi = bounds
        cx, cy, cz = 0.5 * (xlo + xhi), 0.5 * (ylo + yhi), 0.5 * (zlo + zhi)
        diag = max(xhi - xlo, yhi - ylo, zhi - zlo) or 1.0
        axis = self._axis if (self._cutaway or self._interactive) else "y"
        if axis == "x":
            offset = (1.6 * diag, -0.8 * diag, -1.6 * diag)
        else:   # "y" default and "z" fallback
            offset = (-0.8 * diag, 1.6 * diag, -1.6 * diag)
        pos = (cx + offset[0], cy + offset[1], cz + offset[2])
        self.plotter.camera_position = [pos, (cx, cy, cz), (1.0, 0.0, 0.0)]
        self.plotter.reset_camera(render=False)


class MeshViewer(QWidget):
    #: emitted on a left-click over the plan view while ``set_picking(True)``
    point_picked = pyqtSignal(float, float)   # north_m, east_m (mesh frame)
    #: emitted on a mouse-wheel step over the plan view: -1 up, +1 down
    plan_scrolled = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: dict | None = None
        self._scales: dict = {}
        self._stations: list = []          # [[x_m, y_m], ...]
        self._preview_polygon: list = []   # [[x_m, y_m], ...] -- custom-polygon-in-progress
        self._shell_outlines: list = []    # one [[x_m, y_m], ...] per shell around the polygon
        self._picking = False
        self._plan_ax = None
        self._overlay_artists: list = []
        self._slice = None                 # (rects, log10 rho, title), see set_model_slice
        self._slice_artist = None
        self._slice_colorbar = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        ctl1 = QHBoxLayout()
        self.extent = QComboBox()
        self.extent.addItems(["Within N Skin Depths", "Tight to Survey", "Whole Model"])
        self.nskin = QDoubleSpinBox()
        self.nskin.setRange(0.25, 20.0)
        self.nskin.setSingleStep(0.5)
        self.nskin.setValue(1.0)
        self.nskin.setPrefix("N = ")
        self.max_depth_km = QDoubleSpinBox()
        self.max_depth_km.setRange(0.1, 10000.0)
        self.max_depth_km.setSingleStep(10.0)
        self.max_depth_km.setValue(1000.0)
        self.max_depth_km.setPrefix("Max Depth = ")
        self.max_depth_km.setSuffix(" km")
        self.max_depth_km.setToolTip(
            "Caps how deep the clipped extent goes, for 'Within N Skin Depths' and "
            "'Tight to Survey' (ignored for 'Whole Model').")
        self.view_toggle_btn = QPushButton("Show 3-D Mesh")
        self.view_toggle_btn.setEnabled(False)
        self.view_toggle_btn.clicked.connect(self._toggle_view)
        self.extent.currentIndexChanged.connect(self._on_extent_changed)
        self.nskin.valueChanged.connect(self._update_3d)
        self.max_depth_km.valueChanged.connect(self._update_3d)
        ctl1.addWidget(QLabel("Extent:"))
        ctl1.addWidget(self.extent)
        ctl1.addWidget(self.nskin)
        ctl1.addWidget(self.max_depth_km)
        ctl1.addStretch(1)
        ctl1.addWidget(self.view_toggle_btn)
        lay.addLayout(ctl1)

        ctl2 = QHBoxLayout()
        self.cutaway = QCheckBox("Cutaway (expose interior)")
        self.cutaway.setChecked(True)
        self.show_ghost = QCheckBox("Show Ghost")
        self.show_ghost.setToolTip(
            "Translucent outline of the full clipped extent around the cutaway face.")
        self.show_ghost.setChecked(False)
        self.cut_axis = QComboBox()
        self.cut_axis.addItems(["X", "Y", "Z"])
        self.cut_axis.setCurrentText("Y")
        self.interactive_slice = QCheckBox("Interactive Slice")
        self.interactive_slice.setToolTip(
            "Drag a live slice plane through the mesh, in place of the fixed cutaway.")
        self.color_by_level = QCheckBox("Colour by Refinement Level")
        self.color_by_level.setChecked(False)
        self.hide_air = QCheckBox("Hide Air Layer")
        self.hide_air.setChecked(False)
        for cb in (self.cutaway, self.show_ghost, self.interactive_slice,
                  self.color_by_level, self.hide_air):
            cb.toggled.connect(self._update_3d)
        self.cut_axis.currentIndexChanged.connect(self._update_3d)
        self.interactive_slice.toggled.connect(self._on_interactive_toggled)
        self.cutaway.toggled.connect(
            lambda on: self.show_ghost.setEnabled(on and not self.interactive_slice.isChecked()))
        ctl2.addWidget(self.cutaway)
        ctl2.addWidget(self.show_ghost)
        ctl2.addWidget(QLabel("Cut Axis:"))
        ctl2.addWidget(self.cut_axis)
        ctl2.addWidget(self.interactive_slice)
        ctl2.addWidget(self.color_by_level)
        ctl2.addWidget(self.hide_air)
        ctl2.addStretch(1)
        lay.addLayout(ctl2)

        self.info = QLabel("no mesh built yet")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("font-size: 11px;")
        lay.addWidget(self.info)

        # page 0: 2-D station / custom-polygon plot (before any mesh exists)
        self.canvas = MplCanvas(figsize=(6, 7))
        if HAVE_MPL:
            self.canvas.canvas.mpl_connect("button_press_event", self._on_click)
            self.canvas.canvas.mpl_connect("scroll_event", self._on_scroll)
        # page 1: 3-D mesh view (once a preview/build exists)
        self.mesh3d = Mesh3DView()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.canvas)
        self.stack.addWidget(self.mesh3d)
        lay.addWidget(self.stack, 1)
        self._set_3d_controls_enabled(False)

    # ------------------------------------------------------------------ #
    def set_scales(self, scales: dict | None):
        self._scales = dict(scales or {})

    def set_stations(self, xy_m: list | None):
        """Station positions (local frame, metres) -- shown even pre-mesh."""
        self._stations = list(xy_m or [])
        if not self._result:
            self.info.setText(f"no mesh built yet  --  {len(self._stations)} stations")
        else:
            self._update_3d()
        self._render_2d()

    def set_preview_polygon(self, xy_m: list | None, shells: list | None = None):
        """Overlay a custom-polygon-in-progress (local frame, metres), and optionally
        dashed *shells* around it, widening the view if one falls outside it.

        Updates just the overlay artists on the existing axes (no full
        figure rebuild) so repeated calls -- one per picked point -- don't
        reset the user's current pan/zoom, and don't mutate the figure out
        from under matplotlib's own in-progress click handling.
        """
        self._preview_polygon = list(xy_m or [])
        self._shell_outlines = [s for s in (shells or []) if len(s)]
        if self._shell_outlines and self._plan_ax is not None:
            self._widen_view_to(self._shell_outlines)
        self.update_overlay()

    def _widen_view_to(self, outlines: list) -> None:
        """Grow the map's limits to hold these outlines; never shrinks them."""
        import numpy as np

        pts = np.concatenate([np.asarray(o, dtype="float64") for o in outlines]) / 1e3
        ax = self._plan_ax
        margin = 0.05 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
        (e0, e1), (n0, n1) = ax.get_xlim(), ax.get_ylim()
        if pts[:, 1].min() < e0 or pts[:, 1].max() > e1:
            ax.set_xlim(min(e0, pts[:, 1].min() - margin), max(e1, pts[:, 1].max() + margin))
        if pts[:, 0].min() < n0 or pts[:, 0].max() > n1:
            ax.set_ylim(min(n0, pts[:, 0].min() - margin), max(n1, pts[:, 0].max() + margin))

    def set_model_slice(self, rects=None, log10_rho=None, title: str = ""):
        """Draw a model slice under the stations, in place, keeping the zoom.  *rects* is
        ``(n, 4)`` north0, north1, east0, east1 in metres; ``None`` removes the slice."""
        self._slice = None if rects is None else (rects, log10_rho, title)
        if self._plan_ax is None:
            self._render_2d()
            return
        self._draw_slice(self._plan_ax)
        self._draw_overlay(self._plan_ax)      # its legend's placement depends on the slice
        self.canvas.redraw()

    def set_picking(self, on: bool):
        """Enable/disable emitting :attr:`point_picked` on plan-view clicks.

        Also disengages the navigation toolbar's Pan/Zoom tool, if active --
        left on, it intercepts plain clicks as (tiny, often near-invisible)
        zoom/pan drags, which both eats picks and creeps the view in on every
        click. And switches to the 2-D page, if the 3-D mesh view is
        currently showing -- picking needs a literal plan-view click target,
        so there'd otherwise be no way back to it once any mesh preview
        exists.
        """
        self._picking = bool(on)
        if on:
            self._deactivate_toolbar_mode()
            self._show_2d()

    def _deactivate_toolbar_mode(self):
        tb = getattr(self.canvas, "toolbar", None)
        name = getattr(getattr(tb, "mode", None), "name", "")
        if name == "PAN":
            tb.pan()    # toggles back to no-tool
        elif name == "ZOOM":
            tb.zoom()   # toggles back to no-tool

    def _toolbar_mode_active(self) -> bool:
        tb = getattr(self.canvas, "toolbar", None)
        return bool(getattr(tb, "mode", ""))

    def show_result(self, result: dict):
        self._result = result
        self.info.setText(
            f"step {result['step']} / {result['n_total_steps']}   |   "
            f"{result['n_cells']:,} cells   |   smallest "
            f"{result['smallest_cell_m']:.0f} m   largest "
            f"{result['largest_cell_m']:.0f} m   |   {result['vtu']}")
        self.mesh3d.load(result["vtu"])
        self._set_3d_controls_enabled(True)   # also triggers the first render
        self.view_toggle_btn.setEnabled(True)
        self.stack.setCurrentWidget(self.mesh3d)
        self.view_toggle_btn.setText("Show 2-D Station / Polygon Map")

    def _set_3d_controls_enabled(self, on: bool):
        for w in (self.extent, self.nskin, self.max_depth_km, self.cutaway, self.show_ghost,
                  self.cut_axis, self.interactive_slice, self.color_by_level, self.hide_air):
            w.setEnabled(on)
        if on:
            self._on_interactive_toggled(self.interactive_slice.isChecked())
            self._on_extent_changed()

    def _toggle_view(self):
        if self.stack.currentWidget() is self.canvas:
            if not self._result:
                return
            self.stack.setCurrentWidget(self.mesh3d)
            self.view_toggle_btn.setText("Show 2-D Station / Polygon Map")
        else:
            self._show_2d()

    def _show_2d(self):
        self.stack.setCurrentWidget(self.canvas)
        if self._result:
            self.view_toggle_btn.setText("Show 3-D Mesh")

    def _on_interactive_toggled(self, on: bool):
        # cutaway/ghost have no effect while the interactive slice is active
        self.cutaway.setEnabled(not on)
        self.show_ghost.setEnabled(not on and self.cutaway.isChecked())

    def _on_click(self, event):
        if not self._picking or event.button != 1 or event.xdata is None:
            return
        if event.inaxes is not self._plan_ax:
            return
        if self._toolbar_mode_active():
            return   # a pan/zoom drag is in progress -- let the toolbar have this click
        # plot axes are east(x)/north(y); emit in the mesh frame (north, east)
        self.point_picked.emit(float(event.ydata) * 1e3, float(event.xdata) * 1e3)

    def _on_scroll(self, event):
        if event.inaxes is self._plan_ax and event.inaxes is not None:
            self.plan_scrolled.emit(-1 if event.step > 0 else 1)

    # ------------------------------------------------------------------ #
    def _delta_max(self) -> float:
        rho = self._scales.get("rho_deep")
        t = self._scales.get("period_max_s")
        if rho and t:
            return 503.0 * math.sqrt(rho * t)
        return 0.0

    def _on_extent_changed(self):
        mode = self.extent.currentText()
        self.nskin.setEnabled(mode == "Within N Skin Depths")
        self.max_depth_km.setEnabled(mode != "Whole Model")
        self._update_3d()

    def _update_3d(self):
        """Recompute the clip box from Extent/N/Max Depth and push all 3-D view options."""
        if not self._result:
            return
        r = self._result
        mode = self.extent.currentText()
        max_depth_m = self.max_depth_km.value() * 1000.0
        if mode == "Whole Model":
            bounds = None
        elif mode == "Tight to Survey":
            xspan = r["station_x_max"] - r["station_x_min"]
            yspan = r["station_y_max"] - r["station_y_min"]
            pad = 0.10 * max(xspan, yspan, 1.0)
            bounds = (r["station_x_min"] - pad, r["station_x_max"] + pad,
                     r["station_y_min"] - pad, r["station_y_max"] + pad,
                     -0.15 * max_depth_m, max_depth_m)
        elif self._delta_max() > 0:
            m = self.nskin.value() * self._delta_max()             # lateral padding, uncapped
            z_hi = min(m, max_depth_m)                              # depth, capped
            bounds = (r["station_x_min"] - m, r["station_x_max"] + m,
                     r["station_y_min"] - m, r["station_y_max"] + m,
                     -0.15 * z_hi, z_hi)                # surface up, capped depth down (z is +down)
        else:
            bounds = None   # no scales yet (period/rho) -- fall back to whole model
        self.mesh3d.configure(
            bounds=bounds, cutaway=self.cutaway.isChecked(),
            ghost=self.show_ghost.isChecked(),
            axis=self.cut_axis.currentText().lower(),
            interactive=self.interactive_slice.isChecked(),
            color_by_level=self.color_by_level.isChecked(),
            hide_air=self.hide_air.isChecked(),
            stations=self._stations)

    # ------------------------------------------------------------------ #
    def _render_2d(self):
        if not HAVE_MPL:
            return
        if not self._stations and not self._preview_polygon and self._slice is None:
            return
        fig = self.canvas.figure
        fig.clear()
        self._slice_artist = self._slice_colorbar = None
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_title("station layout")
        # stations/polygon are stored in the mesh frame (col 0 = north = mesh x,
        # col 1 = east = mesh y); plot it as a map -- east right, north up.
        ax1.set_xlabel("east (km)")
        ax1.set_ylabel("north (km)")
        ax1.set_aspect("equal", adjustable="box")
        # only stations to show (no in-progress polygon) -- pad the view so
        # markers aren't flush against the axes; once picking starts, leave
        # it to autoscale so newly-clicked points stay visible.
        if self._stations and not self._preview_polygon:
            lims = self._station_limits()
            if lims:
                (east_l, north_l) = lims
                ax1.set_xlim(east_l[0] / 1e3, east_l[1] / 1e3)
                ax1.set_ylim(north_l[0] / 1e3, north_l[1] / 1e3)
        self._plan_ax = ax1
        self._draw_slice(ax1)
        if self._slice is not None and not self._stations and not self._preview_polygon:
            rects = self._slice[0]
            if len(rects):
                ax1.set_xlim(rects[:, 2].min() / 1e3, rects[:, 3].max() / 1e3)
                ax1.set_ylim(rects[:, 0].min() / 1e3, rects[:, 1].max() / 1e3)
        self._draw_overlay(ax1)
        self.canvas.redraw()

    def update_overlay(self):
        """Redraw just the station/polygon overlay in place, preserving zoom/pan.

        Falls back to a full :meth:`_render_2d` if there is no live plan axes
        yet (nothing built so far this session).
        """
        if not HAVE_MPL or self._plan_ax is None:
            self._render_2d()
            return
        self._draw_overlay(self._plan_ax)
        self.canvas.redraw()

    def _station_limits(self):
        """Station bbox padded by 20% of the survey footprint's larger span."""
        import numpy as np

        if not self._stations:
            return None
        s = np.asarray(self._stations, dtype="float64")
        north0, north1 = float(s[:, 0].min()), float(s[:, 0].max())
        east0, east1 = float(s[:, 1].min()), float(s[:, 1].max())
        span = max(north1 - north0, east1 - east0)
        pad = 0.2 * span if span > 0 else 1000.0   # fallback for a single/collinear station
        # returned as (east-limits, north-limits) to match the map axes
        return (east0 - pad, east1 + pad), (north0 - pad, north1 + pad)

    def _draw_slice(self, ax):
        """(Re)draw the model slice on *ax* without touching its axis limits."""
        import copy

        import matplotlib
        import numpy as np
        from matplotlib.collections import PolyCollection
        from matplotlib.colors import Normalize

        if self._slice is None:
            if self._slice_artist is not None:
                self._slice_colorbar.remove()       # needs the artist still in place
                self._slice_artist.remove()
                self._slice_artist = self._slice_colorbar = None
            ax.set_title("station layout")
            return
        rects, values, title = self._slice
        n0, n1, e0, e1 = (rects[:, k] / 1e3 for k in range(4))
        verts = np.stack([np.c_[e0, n0], np.c_[e1, n0], np.c_[e1, n1], np.c_[e0, n1]], axis=1)
        values = np.ma.masked_invalid(values)
        if self._slice_artist is not None:
            # the same artist throughout: a colorbar can only be removed with the one it was made for
            self._slice_artist.set_verts(verts)
            self._slice_artist.set_array(values)
        else:
            cmap = copy.copy(matplotlib.colormaps["turbo_r"])
            cmap.set_bad("lightgrey")                    # no .materials: cell outlines still help
            coll = PolyCollection(verts, array=values, cmap=cmap, norm=Normalize(0.0, 4.0),
                                  edgecolors=(0, 0, 0, 0.3), linewidths=0.3, zorder=1)
            ax.add_collection(coll, autolim=False)
            self._slice_artist = coll
            self._slice_colorbar = ax.figure.colorbar(coll, ax=ax, extend="both", shrink=0.85,
                                                      label="log10 resistivity (ohm m)")
        ax.set_title(title or "station layout")

    def _draw_overlay(self, ax):
        """(Re)draw the station/polygon overlay on *ax* without touching its
        axis limits -- callable either right after a full rebuild (fresh axes)
        or standalone for an in-place update (existing axes, zoom preserved)."""
        import numpy as np

        for art in self._overlay_artists:
            try:
                art.remove()
            except (ValueError, NotImplementedError):
                pass   # already gone (e.g. its axes was torn down by a full rebuild)
        self._overlay_artists = []

        # mesh frame -> map axes: x = east = col 1, y = north = col 0
        drew = False
        if self._stations:
            s = np.asarray(self._stations, dtype="float64")
            sc = ax.scatter(s[:, 1] / 1e3, s[:, 0] / 1e3, marker="^", c="red", s=25,
                            zorder=5, label="stations")
            self._overlay_artists.append(sc)
            drew = True
        if self._preview_polygon:
            pts = self._preview_polygon + [self._preview_polygon[0]] \
                if len(self._preview_polygon) > 1 else self._preview_polygon
            p = np.asarray(pts, dtype="float64")
            (line,) = ax.plot(p[:, 1] / 1e3, p[:, 0] / 1e3, "o-", c="orange", zorder=6,
                              label="custom polygon")
            self._overlay_artists.append(line)
            drew = True
        for k, outline in enumerate(self._shell_outlines):
            p = np.asarray(outline, dtype="float64")
            (line,) = ax.plot(p[:, 1] / 1e3, p[:, 0] / 1e3, "--", c="orange", lw=1.2, zorder=6,
                              label="shells" if k == 0 else None)
            self._overlay_artists.append(line)
            drew = True
        if drew:
            # "best" tests every polygon of a slice at each candidate position: 20 s per draw
            leg = ax.legend(loc="upper right" if self._slice is not None else "best", fontsize=7)
            if leg is not None:
                self._overlay_artists.append(leg)
