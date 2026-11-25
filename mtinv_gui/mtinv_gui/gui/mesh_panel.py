"""Step 5 -- Mesh.

A guided, biggest-to-smallest flow (QToolBox accordion): model size -> ocean ->
base cells -> refine the survey area -> local refinement -> topography &
materials -> build.  The 3-D preview on the right rebuilds the mesh up to any step.
"""
from __future__ import annotations

import dataclasses
import math
import os
import traceback

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QSplitter, QToolBox,
    QVBoxLayout, QWidget,
)

from ..backends import mesh
from ..backends import mesh_design as MD
from .mesh_viewer import MeshViewer
from .panel_base import BasePanel
from .worker import run_async

_KM = 1000.0


def _dspin(lo, hi, step, dec, val, suffix=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(dec)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


def _fmt_list(vals) -> str:
    return ", ".join(f"{v:g}" for v in vals)


def _parse_list(text: str) -> list[float]:
    txt = text.replace(";", ",")
    return [float(x) for x in txt.split(",") if x.strip()]


def _fmt_xy_list(pts) -> str:
    return "; ".join(f"{x:g},{y:g}" for x, y in pts)


def _parse_xy_list(text: str) -> list[list[float]]:
    pts = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        xy = [float(v) for v in chunk.replace(",", " ").split()]
        if len(xy) != 2:
            raise ValueError(f"bad polygon point {chunk!r} -- expected 'x,y'")
        pts.append(xy)
    return pts


def _shape_key(label: str) -> str:
    """Combo display text -> stored shape key, e.g. 'Convex Hull' -> 'convex_hull'."""
    return label.strip().lower().replace(" ", "_")


def _shape_label(key: str) -> str:
    """Stored shape key -> combo display text, e.g. 'convex_hull' -> 'Convex Hull'."""
    return " ".join(w.capitalize() for w in str(key).split("_")) or "Ellipse"


class MeshPanel(BasePanel):
    title = "5 - Mesh"

    # ------------------------------------------------------------------ #
    def build(self) -> None:
        self._loading = True   # widgets fire signals as they're constructed below
        self._mm: dict | None = None   # last suggest_mesh() meta, for the survey-area preview
        root = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        self.recompute_btn = QPushButton("Recompute From Data")
        self.recompute_btn.clicked.connect(self._recompute)
        top.addWidget(self.recompute_btn)
        top.addStretch(1)
        lv.addLayout(top)
        self.scales_lbl = QLabel("press 'Recompute from data' to size the mesh from the survey")
        self.scales_lbl.setWordWrap(True)
        self.scales_lbl.setStyleSheet("color: gray; font-size: 11px;")
        lv.addWidget(self.scales_lbl)

        self.tb = QToolBox()
        lv.addWidget(self.tb, 1)
        self._build_pages()

        lv.addWidget(self._build_buildbar())
        split.addWidget(left)

        self.viewer = MeshViewer()
        self.viewer.point_picked.connect(self._on_plot_point_picked)
        split.addWidget(self.viewer)
        split.setSizes([440, 640])

        self._loading = False
        self.refresh()

    # ---- pages --------------------------------------------------------- #
    def _build_pages(self):
        d = MD.DesignRules()

        # 1 - model size
        p1 = QWidget()
        f1 = QFormLayout(p1)
        self.depth_mult = _dspin(2, 8, 0.5, 1, d.depth_skin_multiple)
        self.pad_mult = _dspin(2, 8, 0.5, 1, d.pad_skin_multiple)
        self.air_mode = QComboBox()
        self.air_mode.addItems(["Match Earth Thickness"])
        self.mantle = QCheckBox("20 Ω·m Half-Space Below 410 km")
        self.mantle.setToolTip(
            "Start the model with a resistive-mantle transition: 20 Ω·m below "
            "410 km, fixed (not inverted). It limits the penetration depth, so "
            "the model bottom is capped at 1000 km.")
        for w in (self.depth_mult, self.pad_mult):
            w.valueChanged.connect(self._regen)
        self.mantle.toggled.connect(self._regen)
        f1.addRow("Model Depth = δ_max ×", self.depth_mult)
        f1.addRow("Edge Padding = δ_max ×", self.pad_mult)
        f1.addRow("Air Column", self.air_mode)
        f1.addRow(self.mantle)
        self.size_lbl = QLabel("-")
        self.size_lbl.setWordWrap(True)
        f1.addRow(self.size_lbl)
        self.tb.addItem(p1, "1 · Model Size")

        # 2 - ocean
        self._build_ocean_page()

        # 3 - base cells  (NOT sized to fit between stations -- see the survey-area page)
        p2 = QWidget()
        f2 = QFormLayout(p2)

        self.base_auto = QCheckBox("Auto: Span the Padding in N Cells")
        self.base_auto.setChecked(True)
        self.base_count = QSpinBox()
        self.base_count.setRange(2, 200)
        self.base_count.setValue(int(d.base_cell_count))
        self.base_size = _dspin(0.1, 5000, 1, 2, 100.0, " km")
        self.base_auto.toggled.connect(self._on_base_auto_toggled)
        self.base_count.valueChanged.connect(self._regen)
        self.base_size.valueChanged.connect(self._regen)
        f2.addRow(self.base_auto)
        f2.addRow("N (Cells Across the Padding)", self.base_count)
        f2.addRow("Base Cell Size", self.base_size)

        self.vgrowth = _dspin(1.05, 2.0, 0.05, 2, d.vertical_growth)
        self.uniform_dx = QCheckBox("Uniform dx Half-Blocks")
        self.uniform_dx.setChecked(d.constant_dx)
        self.round_grid = QCheckBox("Snap Sizes to 1e3 / 1e4 / 1e5")
        self.round_grid.setChecked(d.round_grid)
        self.vgrowth.valueChanged.connect(self._regen)
        for w in (self.uniform_dx, self.round_grid):
            w.toggled.connect(self._regen)
        f2.addRow("Vertical Growth Factor", self.vgrowth)
        f2.addRow(self.uniform_dx)
        f2.addRow(self.round_grid)

        adv = QGroupBox("Advanced")
        adv.setCheckable(True)
        adv.setChecked(False)
        af = QFormLayout(adv)
        self.dz_floor = _dspin(1, 5000, 10, 0, d.dz_floor_m, " m")
        self.max_dz_frac = _dspin(0.05, 0.5, 0.05, 2, d.max_dz_fraction)
        for w in (self.dz_floor, self.max_dz_frac):
            w.valueChanged.connect(self._regen)
        af.addRow("Hard Minimum Cell", self.dz_floor)
        af.addRow("Max Layer / Model Depth", self.max_dz_frac)
        f2.addRow(adv)

        self.base_lbl = QLabel("-")
        self.base_lbl.setWordWrap(True)
        f2.addRow(self.base_lbl)

        self.override_lists = QCheckBox("Override Cell Lists")
        self.override_lists.toggled.connect(self._on_override_toggled)
        f2.addRow(self.override_lists)
        self.dx_text = QLineEdit()
        self.dz_text = QLineEdit()
        self.dx_text.editingFinished.connect(self._regen)
        self.dz_text.editingFinished.connect(self._regen)
        f2.addRow("dx Half-Blocks (m)", self.dx_text)
        f2.addRow("dz Earth (m)", self.dz_text)

        sug_box = QGroupBox("Suggest Survey-Area Refinement Levels")
        sf = QFormLayout(sug_box)
        self.between_spin = QSpinBox()
        self.between_spin.setRange(1, 20)
        self.between_spin.setValue(int(d.cells_between_sites))
        self.spacing_choice = QComboBox()
        self.spacing_choice.addItems(["Minimum", "Median"])
        self.spacing_choice.setCurrentIndex(1 if d.suggestion_use_median else 0)
        self.between_spin.valueChanged.connect(self._regen)
        self.spacing_choice.currentIndexChanged.connect(self._regen)
        sf.addRow("Desired Cells Between Sites", self.between_spin)
        sf.addRow("Site Spacing to Use", self.spacing_choice)
        self.suggestion_lbl = QLabel("-")
        self.suggestion_lbl.setWordWrap(True)
        sf.addRow(self.suggestion_lbl)
        f2.addRow(sug_box)

        self.tb.addItem(p2, "3 · Base Cells")

        # 3 - refine survey area  (edits stage 0)
        # Shells are numbered outer (shell 0, the "Outer Buffer" / "Refine
        # Down to Depth" values) -> inner. Each further shell shrinks the
        # previous one geometrically by the matching shrink factor, or the
        # whole sequence can be replaced with an explicit outer->inner list.
        p3 = QWidget()
        f3 = QFormLayout(p3)
        self.sa_levels = QSpinBox()
        self.sa_levels.setRange(0, 12)

        self.sa_buffer = _dspin(0, 99999, 5, 1, 20.0, " km")
        self.sa_buffer_shrink = _dspin(0.25, 0.75, 0.05, 2, 0.5)
        self.sa_buffer_override = QCheckBox("Override Shell Buffers")
        self.sa_buffer_list = QLineEdit()
        self.sa_buffer_list.setPlaceholderText("outer → inner, km, e.g. 20, 8, 3")

        self.sa_depth = _dspin(1, 2000, 10, 0, 100.0, " km")
        self.sa_taper = QCheckBox("Taper Refined Depth Each Level")
        self.sa_taper.setChecked(True)
        self.sa_depth_shrink = _dspin(0.25, 0.75, 0.05, 2, 0.55)
        self.sa_depth_override = QCheckBox("Override Shell Depths")
        self.sa_depth_list = QLineEdit()
        self.sa_depth_list.setPlaceholderText("outer → inner, km, e.g. 100, 55, 30")

        self.sa_shape = QComboBox()
        self.sa_shape.addItems(["Ellipse", "Rectangle", "Convex Hull", "Custom Polygon"])
        self.sa_polygon_text = QLineEdit()
        self.sa_polygon_text.setPlaceholderText(
            "north,east km per point, e.g. 0,0; 10,0; 10,8  (replaces the station hull)")
        self.sa_polygon_pick = QPushButton("Pick On Plot")
        self.sa_polygon_pick.setCheckable(True)
        self.sa_polygon_clear = QPushButton("Clear Points")
        self._picked_pts: list = []   # metres, while sa_polygon_pick is checked

        self.sa_polygon_pnorm = _dspin(1.0, 2.0, 0.1, 2, 1.0)
        self.sa_polygon_pnorm.setToolTip(
            "refine_within_polygon pnorm, for Convex Hull / Custom Polygon.\n"
            "1 = exact inside-the-boundary test (recommended).\n"
            ">1 (up to 2) = gofem approximates with a bounding circle sized off\n"
            "the footprint's diagonal (rounder, faster, less precise).")

        for w in (self.sa_buffer, self.sa_buffer_shrink, self.sa_depth, self.sa_depth_shrink):
            w.valueChanged.connect(self._stages_changed)
        self.sa_levels.valueChanged.connect(self._on_sa_levels_edited)
        self.sa_taper.toggled.connect(self._on_sa_taper_toggled)
        self.sa_shape.currentIndexChanged.connect(self._on_sa_shape_changed)
        self.sa_buffer_override.toggled.connect(self._on_sa_buffer_override_toggled)
        self.sa_depth_override.toggled.connect(self._on_sa_depth_override_toggled)
        self.sa_buffer_list.editingFinished.connect(self._stages_changed)
        self.sa_depth_list.editingFinished.connect(self._stages_changed)
        self.sa_polygon_text.editingFinished.connect(self._stages_changed)
        self.sa_polygon_pick.toggled.connect(self._on_polygon_pick_toggled)
        self.sa_polygon_clear.clicked.connect(self._on_polygon_clear_clicked)
        self.sa_polygon_pnorm.valueChanged.connect(self._stages_changed)

        f3.addRow("Refinement Levels", self.sa_levels)
        f3.addRow("Outer Buffer (Shell 0)", self.sa_buffer)
        f3.addRow("Shell Shrink Factor (Lateral)", self.sa_buffer_shrink)
        f3.addRow(self.sa_buffer_override)
        f3.addRow("Shell Buffers", self.sa_buffer_list)
        f3.addRow("Refine Down to Depth (Shell 0)", self.sa_depth)
        f3.addRow(self.sa_taper)
        f3.addRow("Shell Shrink Factor (Depth)", self.sa_depth_shrink)
        f3.addRow(self.sa_depth_override)
        f3.addRow("Shell Depths", self.sa_depth_list)
        f3.addRow("Footprint Shape", self.sa_shape)
        poly_row = QHBoxLayout()
        poly_row.addWidget(self.sa_polygon_text, 1)
        poly_row.addWidget(self.sa_polygon_pick)
        poly_row.addWidget(self.sa_polygon_clear)
        poly_row_w = QWidget()
        poly_row_w.setLayout(poly_row)
        f3.addRow("Custom Polygon", poly_row_w)
        f3.addRow("Hull / Polygon pnorm", self.sa_polygon_pnorm)
        self.sa_pick_status_lbl = QLabel()
        self.sa_pick_status_lbl.setStyleSheet("font-weight: bold;")
        f3.addRow(self.sa_pick_status_lbl)
        self.sa_lbl = QLabel(
            "Shell 0 is the outer buffer around the convex hull of the stations "
            "(Ellipse / Rectangle / Convex Hull) or a hand-typed / plot-picked "
            "polygon (Custom Polygon). Each inner shell's buffer/depth shrinks "
            "by its shrink factor, outer → inner, unless "
            "overridden with an explicit list.")
        self.sa_lbl.setWordWrap(True)
        self.sa_lbl.setStyleSheet("color: gray; font-size: 11px;")
        f3.addRow(self.sa_lbl)
        self.sa_info_lbl = QLabel("-")
        self.sa_info_lbl.setWordWrap(True)
        f3.addRow(self.sa_info_lbl)
        self.tb.addItem(p3, "4 · Refine Survey Area")

        # 5 - local refinement (extra stages)
        p4 = QWidget()
        v4 = QVBoxLayout(p4)
        addrow = QHBoxLayout()
        for text, kind in (("+ Around Stations", "stations"), ("+ Zone", "zone"),
                           ("+ Along Coastline", "coastline"), ("+ Interface", "interface")):
            b = QPushButton(text)
            b.setToolTip(MD.STAGE_KINDS[kind])
            b.clicked.connect(lambda _=False, k=kind: self._add_stage(k))
            addrow.addWidget(b)
        addrow.addStretch(1)
        v4.addLayout(addrow)
        self.extra_box = QVBoxLayout()
        v4.addLayout(self.extra_box)
        note4 = QLabel(
            "Forward-solve accuracy right at the receivers is handled by GoFEM at "
            "run time (GoFEM Files tab) -- you usually do not need station "
            "refinement in the mesh itself.")
        note4.setWordWrap(True)
        v4.addWidget(note4)
        v4.addStretch(1)
        self.tb.addItem(p4, "5 · Local Refinement")

        # 6 - topography & materials
        p5 = QWidget()
        f5 = QFormLayout(p5)
        self.use_topo = QCheckBox("Fit Mesh to the Projected DEM")
        self.use_topo.setChecked(True)
        self.use_topo.toggled.connect(self._store_misc)
        self.bg_auto = QCheckBox("Background ρ = SSQ Log-Mean")
        self.bg_auto.setChecked(True)
        self.bg_res = _dspin(0.1, 1e5, 10, 1, 150.0, " Ω·m")
        self.bg_auto.toggled.connect(lambda on: (self.bg_res.setDisabled(on), self._store_misc()))
        self.bg_res.valueChanged.connect(self._store_misc)
        f5.addRow(self.use_topo)
        f5.addRow(self.bg_auto)
        f5.addRow("Background Resistivity", self.bg_res)
        self.tb.addItem(p5, "6 · Topography & Materials")

        # initial enabled/disabled state (signals are quiet while _loading)
        self._on_base_auto_toggled(self.base_auto.isChecked())
        self._on_override_toggled(self.override_lists.isChecked())
        self._on_sa_taper_toggled(self.sa_taper.isChecked())
        self._on_sa_buffer_override_toggled(self.sa_buffer_override.isChecked())
        self._on_sa_depth_override_toggled(self.sa_depth_override.isChecked())
        self._on_sa_shape_changed()
        self._update_pick_status()
        self.bg_res.setDisabled(self.bg_auto.isChecked())

    # ---- ocean ------------------------------------------------------------ #
    _OCEAN_MODES = (("none", "None (Flat Sea Level)"),
                    ("thin_sheet", "Thin Sheet"),
                    ("real", "Real Ocean (Bathymetry)"))

    def _build_ocean_page(self):
        p = QWidget()
        f = QFormLayout(p)
        self.ocean_mode = QComboBox()
        for _key, label in self._OCEAN_MODES:
            self.ocean_mode.addItem(label)
        self.ocean_res = _dspin(0.01, 100.0, 0.05, 4, 1.0 / 3.2, " Ω·m")
        self.ocean_res.setToolTip("Sea-water resistivity (3.2 S/m = 0.3125 Ω·m).")
        self.ocean_sheet = _dspin(0.1, 100.0, 0.5, 2, 1.5, " km")
        self.ocean_sheet.setToolTip(
            "Thin sheet only: the top X km over the ocean carries the conductance "
            "(depth × sea conductivity) of the real water column.")
        self.ocean_mode.currentIndexChanged.connect(self._store_ocean)
        self.ocean_res.valueChanged.connect(self._store_ocean)
        self.ocean_sheet.valueChanged.connect(self._store_ocean)
        from ..backends.mesh_dem import SEA_RULES

        self.sea_rule = QComboBox()
        for _key, label in SEA_RULES.items():
            self.sea_rule.addItem(label)
        self.sea_rule.setToolTip(
            "Which DEM points below sea level are sea. 'Connected' leaves land below sea level\n"
            "(Death Valley, polders, lake beds below sea level) as land, at its own elevation:\n"
            "only areas joined to the open sea (the edge of the DEM) count. The polygon files\n"
            "below override it either way.")
        self.land_polys = QLineEdit()
        self.land_polys.setPlaceholderText("(optional) polygons that are never sea")
        self.sea_polys = QLineEdit()
        self.sea_polys.setPlaceholderText("(optional) polygons that are sea where below sea level")
        poly_tip = ("A shapefile / GeoPackage / GeoJSON / KML of polygons, in any CRS (longitude /\n"
                    "latitude if the file has none). Use it where the rule gets it wrong: a polder\n"
                    "joined to the sea by a channel (land), or an enclosed sea or lake (sea).")
        rows = {}
        for key, edit in (("land", self.land_polys), ("sea", self.sea_polys)):
            edit.setToolTip(poly_tip)
            edit.editingFinished.connect(self._store_ocean)
            b = QPushButton("Browse…")
            b.clicked.connect(lambda _c=False, e=edit: self._browse_polygons(e))
            row = QHBoxLayout()
            row.addWidget(edit, 1)
            row.addWidget(b)
            rows[key] = row
        self.sea_gap = _dspin(0.0, 200.0, 1.0, 1, 10.0, " km")
        self.sea_gap.setToolTip(
            "'Connected' only: below-sea-level areas this close to the sea count as joined to it.\n"
            "A DEM doesn't resolve narrow straits or fjord mouths (San Francisco Bay looks\n"
            "enclosed at 3 km), while true land below sea level lies much further from the sea.")
        self.sea_rule.currentIndexChanged.connect(self._store_ocean)
        self.sea_gap.valueChanged.connect(self._store_ocean)
        f.addRow("Ocean Model", self.ocean_mode)
        f.addRow("Sea Resistivity", self.ocean_res)
        f.addRow("Sheet Thickness X", self.ocean_sheet)
        f.addRow("Sea Is", self.sea_rule)
        f.addRow("Joined Across Gaps Up To", self.sea_gap)
        f.addRow("Always Land", rows["land"])
        f.addRow("Always Sea", rows["sea"])
        self.ocean_note = QLabel("-")
        self.ocean_note.setWordWrap(True)
        f.addRow(self.ocean_note)
        self.ocean_info = QLabel("-")
        self.ocean_info.setWordWrap(True)
        self.ocean_info.setStyleSheet("color: gray; font-size: 11px;")
        f.addRow(self.ocean_info)
        self._ocean_cache: tuple | None = None    # (key, stats) for the DEM summary
        self.tb.addItem(p, "2 · Ocean")

    def _ocean_key(self) -> str:
        return self._OCEAN_MODES[max(self.ocean_mode.currentIndex(), 0)][0]

    def _sea_rule_key(self) -> str:
        from ..backends.mesh_dem import SEA_RULES

        return list(SEA_RULES)[max(self.sea_rule.currentIndex(), 0)]

    def _browse_polygons(self, edit):
        f, _ = QFileDialog.getOpenFileName(
            self, "Polygons", edit.text() or self.project.workdir or "",
            "vector files (*.shp *.gpkg *.geojson *.json *.kml);;all files (*)")
        if f:
            edit.setText(f)
            self._store_ocean()

    def _store_ocean(self, *_):
        if self._loading:
            return
        m = self.project.mesh
        m.ocean_mode = self._ocean_key()
        m.ocean_resistivity_ohm_m = self.ocean_res.value()
        m.ocean_thickness_m = self.ocean_sheet.value() * _KM
        m.sea_rule = self._sea_rule_key()
        m.sea_gap_km = self.sea_gap.value()
        m.land_polygons_path = self.land_polys.text().strip()
        m.sea_polygons_path = self.sea_polys.text().strip()
        self._update_ocean_info()
        self._update_dem_cover()
        self.notify_changed()

    def _mesh_dem(self):
        """The projected DEM with this project's sea settings (``mesh_dem.MeshDem``), cached
        by file + settings; ``None`` if there is no DEM yet.  Raises on a bad polygon file."""
        path = self.project.projection.projected_dem_path
        if not (path and os.path.isfile(path)):
            return None
        m = self.project.mesh

        def stamp(f):
            return (f, os.path.getmtime(f)) if f and os.path.isfile(f) else (f, None)

        key = (stamp(path), m.sea_rule, m.sea_gap_km, stamp(m.land_polygons_path),
               stamp(m.sea_polygons_path))
        if getattr(self, "_dem_cache", None) is None or self._dem_cache[0] != key:
            self._dem_cache = (key, None)
            frame = None
            if m.land_polygons_path or m.sea_polygons_path:
                from ..backends import projection as P

                frame = P.frame_from_projection(self.project.projection)
            dem = mesh.read_mesh_dem(path, sea_rule=m.sea_rule or "connected", gap_km=m.sea_gap_km,
                                     land_polygons_path=m.land_polygons_path,
                                     sea_polygons_path=m.sea_polygons_path,
                                     frame=frame, log=lambda *_a: None)
            self._dem_cache = (key, dem)
        return self._dem_cache[1]

    def _dem_ocean_stats(self):
        """Ocean coverage of the projected DEM inside the model footprint
        (cached by file + footprint + sea settings); ``None`` if there is no DEM yet."""
        path = self.project.projection.projected_dem_path
        if not (path and os.path.isfile(path)):
            return None
        m = self.project.mesh
        half = sum(m.dx_blocks or [])
        key = (path, os.path.getmtime(path), half, m.sea_rule, m.sea_gap_km,
               m.land_polygons_path, m.sea_polygons_path)
        if self._ocean_cache is None or self._ocean_cache[0] != key:
            try:
                bounds = (-half, half, -half, half) if half > 0 else None
                self._ocean_cache = (key, self._mesh_dem().ocean_stats(bounds))
            except Exception as exc:  # noqa: BLE001 -- unreadable DEM / polygons: just no summary
                self.log(f"could not read the projected DEM for the ocean summary: {exc}")
                self._ocean_cache = (key, None)
        return self._ocean_cache[1]

    def _update_ocean_info(self):
        mode = self._ocean_key()
        self.ocean_res.setEnabled(mode != "none")
        self.ocean_sheet.setEnabled(mode == "thin_sheet")
        for w in (self.sea_rule, self.land_polys, self.sea_polys):
            w.setEnabled(mode != "none" or self.project.mesh.use_topography)
        self.sea_gap.setEnabled(self.sea_rule.isEnabled() and self._sea_rule_key() == "connected")
        self.ocean_note.setText({
            "none": "Bathymetry is ignored: the sea surface stays flat at z = 0 and "
                    "offshore ground is treated as earth.",
            "thin_sheet": "The top X km over the ocean gets the conductance of the real "
                          "water column (depth × sea conductivity ÷ X, never below the "
                          "background). The surface stays flat; sheet cells are fixed, "
                          "not inverted.",
            "real": "The seafloor is part of the fitted surface; cells between sea level "
                    "and the seabed become water (fixed, not inverted). Resolution is "
                    "limited by the cell size near the surface -- add 'Interface' and "
                    "'Along Coastline' stages on page 5.",
        }[mode])
        stats = self._dem_ocean_stats()
        if stats is None:
            self.ocean_info.setText("No projected DEM yet -- run the Project DEM step; the "
                                    "ocean options need it.")
            return
        if not stats["has_ocean"]:
            self.ocean_info.setText(
                "The projected DEM has no ocean inside the model footprint."
                + ("  %.0f km² of land below sea level stays land." % stats["inland_km2"]
                   if stats.get("inland_km2") else ""))
            return
        txt = ("DEM: %.0f%% of the model footprint is ocean, mean depth %.0f m, "
               "max %.0f m." % (100.0 * stats["ocean_frac"], stats["mean_depth_m"],
                                stats["max_depth_m"]))
        if stats.get("inland_km2"):
            txt += ("  %.0f km² of land below sea level stays land, at its own elevation."
                    % stats["inland_km2"])
        if mode == "thin_sheet":
            sig = 1.0 / max(self.ocean_res.value(), 1e-6)
            x_m = max(self.ocean_sheet.value() * _KM, 1.0)
            txt += ("  Sheet conductivity %.2f S/m at the mean depth, %.2f S/m at the "
                    "maximum." % (stats["mean_depth_m"] * sig / x_m,
                                  stats["max_depth_m"] * sig / x_m))
        self.ocean_info.setText(txt)

    def _update_dem_cover(self):
        """Warn when the projected DEM doesn't reach the mesh's edges: past them the build
        repeats the DEM's edge values (with an ocean, the sea floor fills the padding)."""
        from ..backends import mesh_dem

        if not hasattr(self, "dem_cover_lbl"):
            return
        m = self.project.mesh
        path = self.project.projection.projected_dem_path
        half = sum(m.dx_blocks or [])
        text = ""
        if half > 0 and path and os.path.isfile(path) and (m.use_topography or m.ocean_mode != "none"):
            key = (path, os.path.getmtime(path))
            if getattr(self, "_extent_cache", None) is None or self._extent_cache[0] != key:
                try:
                    self._extent_cache = (key, mesh_dem.file_extent(path))
                except Exception:  # noqa: BLE001 -- unreadable: the build will say so
                    self._extent_cache = (key, None)
            ext = self._extent_cache[1]
            if ext is not None:
                short = mesh_dem.shortfall(ext, (-half, half, -half, half))
                if any(v > 0 for v in short.values()):
                    text = ("⚠ The projected DEM stops short of the mesh by "
                            + mesh_dem.shortfall_text(short) + ". Past its edge the mesh repeats the "
                            "DEM's edge values" + (": with an ocean model the sea floor at the edge "
                                                   "fills the padding" if m.ocean_mode != "none" else "")
                            + ". Add a regional DEM on the Topography tab (pad ≥ the mesh's reach), "
                            "then re-run Project DEM.")
        self.dem_cover_lbl.setText(text)
        self.dem_cover_lbl.setVisible(bool(text))

    def _build_buildbar(self) -> QWidget:
        box = QGroupBox("Preview / Build (needs PyDealII + pyGoFEM)")
        v = QVBoxLayout(box)
        r1 = QHBoxLayout()
        self.stack_lbl = QLabel("deal.II / pyGoFEM: not checked")
        self.stack_lbl.setWordWrap(True)
        chk = QPushButton("Check")
        chk.clicked.connect(self._check_stack)
        r1.addWidget(self.stack_lbl, 1)
        r1.addWidget(chk)
        v.addLayout(r1)

        r_out = QHBoxLayout()
        r_out.addWidget(QLabel("Output Folder"))
        self.out_dir_edit = QLineEdit()
        self.out_dir_edit.setPlaceholderText("(blank = the project folder)")
        self.out_dir_edit.setToolTip(
            "Where 'Build Full Mesh' writes model.tria / .vtu / .materials / "
            ".mask and receivers_projected.csv.\nThe GoFEM Files step reads the "
            "mesh from here too.\nBlank = the project's working folder (where "
            "the .mtinv.json lives).")
        self.out_dir_edit.editingFinished.connect(self._on_out_dir_edited)
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._browse_out_dir)
        r_out.addWidget(self.out_dir_edit, 1)
        r_out.addWidget(out_browse)
        v.addLayout(r_out)
        self.out_dir_lbl = QLabel("-")
        self.out_dir_lbl.setWordWrap(True)
        self.out_dir_lbl.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(self.out_dir_lbl)

        self.dem_cover_lbl = QLabel("")
        self.dem_cover_lbl.setWordWrap(True)
        self.dem_cover_lbl.setStyleSheet("color: #b35c00;")
        self.dem_cover_lbl.hide()
        v.addWidget(self.dem_cover_lbl)

        r_fmt = QHBoxLayout()
        r_fmt.addWidget(QLabel("Mesh For"))
        self.out_format = QComboBox()
        for key, fmt in mesh.OUTPUT_FORMATS.items():
            self.out_format.addItem(fmt["label"], key)
        self.out_format.setToolTip(
            "The mesh design is the same for every code; this only changes what the build "
            "writes.\nGoFEM: model.tria / .vtu / .materials / .mask.\n"
            "GoFEM + FEMTIC: the same, plus mesh.dat and resistivity_block_iter0.dat "
            "(FEMTIC's DHEXA mesh: the same cells).\n"
            "GoFEM + SimPEG: the same, plus an OcTree (discretize TreeMesh, UBC files) as fine as\n"
            "the GoFEM cells horizontally, with earth / air / sea cells from the DEM.\n"
            "The thin-sheet ocean isn't available for FEMTIC or SimPEG (no ocean or a real one).")
        self.out_format.currentIndexChanged.connect(self._store_out_format)
        r_fmt.addWidget(self.out_format)
        self.simpeg_ratio_lbl = QLabel("cell width / height")
        self.simpeg_ratio = QDoubleSpinBox()
        self.simpeg_ratio.setRange(1.0, 64.0)
        self.simpeg_ratio.setDecimals(1)
        self.simpeg_ratio.setSingleStep(1.0)
        self.simpeg_ratio.setToolTip(
            "SimPEG octree cells are all the same shape: the finest is as wide as the smallest "
            "GoFEM cell and this many times flatter.\n1 = cubes. Raise it to resolve the topography "
            "and shallow sea more finely (more cells).")
        self.simpeg_ratio.valueChanged.connect(self._store_simpeg_ratio)
        r_fmt.addWidget(self.simpeg_ratio_lbl)
        r_fmt.addWidget(self.simpeg_ratio)
        r_fmt.addStretch(1)
        v.addLayout(r_fmt)

        r2 = QHBoxLayout()
        step_lbl = QLabel("Preview to Refinement Step #")
        step_lbl.setToolTip(
            "Each step is one deal.II refinement pass -- e.g. one survey-area "
            "shell, or one 'around stations' level -- not a tab/page number. "
            "See the breakdown below for which steps belong to which stage.")
        r2.addWidget(step_lbl)
        self.step = QSpinBox()
        self.step.setRange(0, 0)
        r2.addWidget(self.step)
        self.preview_btn = QPushButton("Preview")
        self.preview_btn.clicked.connect(self._preview_step)
        self.build_btn = QPushButton("Build Full Mesh → Files")
        self.build_btn.clicked.connect(self._build_full)
        r2.addWidget(self.preview_btn)
        r2.addWidget(self.build_btn)
        r2.addStretch(1)
        v.addLayout(r2)
        self.step_breakdown_lbl = QLabel("-")
        self.step_breakdown_lbl.setWordWrap(True)
        self.step_breakdown_lbl.setStyleSheet("color: gray; font-size: 11px;")
        v.addWidget(self.step_breakdown_lbl)
        self.result_lbl = QLabel("")
        self.result_lbl.setWordWrap(True)
        self.result_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.result_lbl)
        return box

    # ---- base-cells page helpers --------------------------------------- #
    def _on_base_auto_toggled(self, on: bool):
        self.base_count.setEnabled(on)
        self.base_size.setEnabled(not on)
        self._regen()

    def _on_override_toggled(self, on: bool):
        for w in (self.base_auto, self.vgrowth, self.uniform_dx, self.round_grid,
                  self.dz_floor, self.max_dz_frac):
            w.setEnabled(not on)
        self.dx_text.setEnabled(on)
        self.dz_text.setEnabled(on)
        if on:
            self.base_count.setEnabled(False)
            self.base_size.setEnabled(False)
        else:
            self._on_base_auto_toggled(self.base_auto.isChecked())
            return  # _on_base_auto_toggled already called _regen()
        self._regen()

    # ---- rules <-> widgets ------------------------------------------- #
    def _rules(self) -> MD.DesignRules:
        base = dataclasses.asdict(MD.DesignRules())
        base.update(dict(
            depth_skin_multiple=self.depth_mult.value(),
            pad_skin_multiple=self.pad_mult.value(),
            base_cell_count=self.base_count.value(),
            base_cell_override_m=(0.0 if self.base_auto.isChecked()
                                  else self.base_size.value() * _KM),
            vertical_growth=self.vgrowth.value(),
            constant_dx=self.uniform_dx.isChecked(),
            round_grid=self.round_grid.isChecked(),
            dz_floor_m=self.dz_floor.value(),
            max_dz_fraction=self.max_dz_frac.value(),
            mantle_transition=self.mantle.isChecked(),
            cells_between_sites=self.between_spin.value(),
            suggestion_use_median=(self.spacing_choice.currentText() == "Median"),
            override_cell_lists=self.override_lists.isChecked(),
        ))
        return MD.DesignRules(**base)

    def _set_rule_widgets(self, dr: dict):
        d = MD.DesignRules(**{**dataclasses.asdict(MD.DesignRules()),
                              **{k: v for k, v in dr.items()
                                 if k in dataclasses.asdict(MD.DesignRules())}})
        self.depth_mult.setValue(d.depth_skin_multiple)
        self.pad_mult.setValue(d.pad_skin_multiple)
        self.base_count.setValue(int(d.base_cell_count))
        self.base_auto.setChecked(d.base_cell_override_m <= 0)
        if d.base_cell_override_m > 0:
            self.base_size.setValue(d.base_cell_override_m / _KM)
        self.vgrowth.setValue(d.vertical_growth)
        self.uniform_dx.setChecked(d.constant_dx)
        self.round_grid.setChecked(d.round_grid)
        self.dz_floor.setValue(d.dz_floor_m)
        self.max_dz_frac.setValue(d.max_dz_fraction)
        self.mantle.setChecked(d.mantle_transition)
        self.between_spin.setValue(int(d.cells_between_sites))
        self.spacing_choice.setCurrentIndex(1 if d.suggestion_use_median else 0)
        self.override_lists.setChecked(d.override_cell_lists)
        self._on_base_auto_toggled(self.base_auto.isChecked())
        self._on_override_toggled(d.override_cell_lists)

    # ---- data / regen --------------------------------------------- #
    def _recompute(self):
        self.project.mesh.suggest_locked = False   # a fresh recompute re-seeds the suggestion
        self.log("collecting survey scales (footprint, spacing, SSQ resistivity, relief)")

        def job():
            return MD.collect_survey_scales(self.project)

        run_async(self, job, self._after_recompute, self._error)

    def _after_recompute(self, scales):
        curve = scales.pop("curve", None)
        if curve is not None:
            t_min = self.project.gofem.period_min_s or curve.period_min_s
            scales["c_response_depth_km_at_tmin"] = curve.c_response_depth_km(t_min)
            scales["c_response_tmin_s"] = float(t_min)
            t_max = self.project.gofem.period_max_s or curve.period_max_s
            scales["c_response_depth_km_at_tmax"] = curve.c_response_depth_km(t_max)
            scales["c_response_tmax_s"] = float(t_max)
        self.project.mesh.scales = scales
        self.viewer.set_stations(scales.get("station_xy_m") or [])
        if curve is not None and curve.rho_geomean == curve.rho_geomean and self.bg_auto.isChecked():
            self.bg_res.setValue(float(curve.rho_geomean))
            self.project.mesh.background_resistivity_ohm_m = float(curve.rho_geomean)
        self.scales_lbl.setText(
            f"footprint {scales['footprint_x_m']/1e3:.0f}×{scales['footprint_y_m']/1e3:.0f} km  |  "
            f"spacing min/med {scales['spacing_min_m']:.0f}/{scales['spacing_median_m']:.0f} m  |  "
            f"periods {scales['period_min_s']:.3g}–{scales['period_max_s']:.3g} s  |  "
            f"ρ {scales['rho_shallow']:.0f}→{scales['rho_deep']:.0f} Ω·m  |  "
            f"δ_min {scales['delta_min_m']:.0f} m  δ_max {scales['delta_max_m']/1e3:.0f} km  |  "
            f"relief {scales['relief_m']:.0f} m")
        # fresh stages from the suggestion
        self._regen(fresh_stages=True)
        self.log("mesh sized from survey data")

    def _regen(self, *_, fresh_stages: bool = False):
        if self._loading:
            return
        m = self.project.mesh
        if not m.scales:
            return
        self._loading = True   # re-entrancy lock; restored in the finally below
        try:
            self._regen_inner(m, fresh_stages)
        finally:
            self._loading = False
        self.notify_changed()

    def _regen_inner(self, m, fresh_stages: bool):
        s = m.scales
        try:
            sugg = MD.suggest_mesh(
                footprint_x_m=s["footprint_x_m"], footprint_y_m=s["footprint_y_m"],
                spacing_min_m=s["spacing_min_m"], spacing_median_m=s["spacing_median_m"],
                period_min_s=s["period_min_s"], period_max_s=s["period_max_s"],
                rho_shallow=s["rho_shallow"], rho_deep=s["rho_deep"],
                relief_m=s.get("relief_m", 0.0),
                c_response_max_km=s.get("c_response_depth_km_at_tmax", 0.0),
                rules=self._rules())
        except Exception as exc:  # noqa: BLE001
            return self._error(exc)

        if self.override_lists.isChecked():
            try:
                dx = _parse_list(self.dx_text.text()) or sugg.dx_blocks
                dz = _parse_list(self.dz_text.text()) or sugg.dz_earth
            except ValueError as exc:
                return self._error(exc)
            m.dx_blocks, m.dz_earth, m.dz_air = dx, dz, list(dz)
        else:
            m.dx_blocks, m.dz_air, m.dz_earth = sugg.dx_blocks, sugg.dz_air, sugg.dz_earth
            self.dx_text.setText(_fmt_list(sugg.dx_blocks))
            self.dz_text.setText(_fmt_list(sugg.dz_earth))
        m.design_rules = dataclasses.asdict(self._rules())

        mm = sugg.meta
        self._mm = mm
        if fresh_stages or not m.refinement_stages:
            m.refinement_stages = sugg.stages
            self._load_stage0()
            self._rebuild_extra_rows()
        elif not m.suggest_locked:
            lv = int(mm["suggested_survey_levels"])
            if m.refinement_stages[0].get("levels") != lv:
                m.refinement_stages[0]["levels"] = lv
                self._load_stage0()

        if self.base_auto.isChecked():
            self.base_size.blockSignals(True)
            self.base_size.setValue(mm["dx_base_m"] / _KM)
            self.base_size.blockSignals(False)

        cap = "  (depth capped at 1000 km by the mantle transition)" \
            if mm.get("mantle_depth_capped") else ""
        self.size_lbl.setText(
            f"δ_max {mm['delta_max_m']/1e3:.0f} km  →  model "
            f"{mm['x_half_m']*2/1e3:.0f} × {mm['x_half_m']*2/1e3:.0f} × "
            f"{mm['model_depth_m']/1e3:.0f} km   (edge pad {mm['edge_pad_m']/1e3:.0f} km)"
            + cap)
        dz0 = mm["dz_base_1_m"]
        c_km = m.scales.get("c_response_depth_km_at_tmin", 0.0)
        c_tmin = m.scales.get("c_response_tmin_s", 0.0)
        dz_part = ""
        if c_km > 0:
            need = max(math.ceil(math.log2(dz0 / (c_km * _KM))), 0) if dz0 > c_km * _KM else 0
            dz_part = (f"   |   surface dz {dz0/1e3:.1f} km  vs  C-response depth at "
                       f"T_min {c_tmin:.3g} s = {c_km:.2f} km  ({need} refine levels to match)")
        self.base_lbl.setText(
            f"base cell {mm['dx_base_m']/1e3:.1f} km  ·  dz {mm['dz_base_1_m']/1e3:.1f}→"
            f"{max(m.dz_earth)/1e3:.0f} km ({len(m.dz_earth)} layers, first = 0.5x base "
            f"cell)  ·  {sugg.n_base_cells:,} base cells" + dz_part)
        self.suggestion_lbl.setText(
            f"→ {mm['suggested_survey_levels']:.0f} survey-area refinement levels for "
            f"~{mm['suggested_cell_m']:.0f} m horizontal cells "
            f"(target {mm['suggestion_target_m']:.0f} m)")
        self._update_sa_info()
        self._update_steps()
        self._update_dem_cover()
        self.viewer.set_scales(m.scales)

    # ---- survey-area stage (stage 0) ----------------------------- #
    def _load_stage0(self):
        m = self.project.mesh
        if not m.refinement_stages:
            return
        st = m.refinement_stages[0]
        prev, self._loading = self._loading, True
        self.sa_levels.setValue(int(st.get("levels", 3)))
        self.sa_buffer.setValue(float(st.get("buffer_km", 20.0)))
        self.sa_buffer_shrink.setValue(float(st.get("buffer_shrink", 0.5)))
        self.sa_buffer_override.setChecked(bool(st.get("buffer_override", False)))
        self.sa_buffer_list.setText(_fmt_list(st.get("buffer_km_list") or []))
        self.sa_depth.setValue(float(st.get("max_depth_km", 100.0)))
        self.sa_taper.setChecked(bool(st.get("taper", True)))
        self.sa_depth_shrink.setValue(float(st.get("depth_shrink", 0.55)))
        self.sa_depth_override.setChecked(bool(st.get("depth_override", False)))
        self.sa_depth_list.setText(_fmt_list(st.get("depth_km_list") or []))
        self.sa_shape.setCurrentText(_shape_label(st.get("shape", "ellipse")))
        self.sa_polygon_pnorm.setValue(float(st.get("polygon_pnorm", 1.0)))
        self.sa_polygon_text.setText(_fmt_xy_list(st.get("polygon_xy_km") or []))
        self.sa_polygon_pick.setChecked(False)
        self.sa_polygon_pick.setStyleSheet("")
        self.sa_polygon_pick.setText("Pick On Plot")
        self._picked_pts = []
        self._on_sa_taper_toggled(self.sa_taper.isChecked())
        self._on_sa_buffer_override_toggled(self.sa_buffer_override.isChecked())
        self._on_sa_depth_override_toggled(self.sa_depth_override.isChecked())
        self._on_sa_shape_changed()
        self._loading = prev
        self._update_sa_info()
        self._update_pick_status()
        self._sync_polygon_preview()

    def _on_sa_levels_edited(self, *_):
        if not self._loading:
            self.project.mesh.suggest_locked = True   # stop auto-syncing to the base-cells page
        self._stages_changed()

    def _on_sa_taper_toggled(self, on: bool):
        self.sa_depth_shrink.setEnabled(on and not self.sa_depth_override.isChecked())
        self._stages_changed()

    def _on_sa_buffer_override_toggled(self, on: bool):
        self.sa_levels.setEnabled(not on)
        self.sa_buffer.setEnabled(not on)
        self.sa_buffer_shrink.setEnabled(not on)
        self.sa_buffer_list.setEnabled(on)
        self._stages_changed()

    def _on_sa_depth_override_toggled(self, on: bool):
        self.sa_depth.setEnabled(not on)
        self.sa_taper.setEnabled(not on)
        self.sa_depth_shrink.setEnabled(not on and self.sa_taper.isChecked())
        self.sa_depth_list.setEnabled(on)
        self._stages_changed()

    def _on_sa_shape_changed(self, *_):
        shape_key = _shape_key(self.sa_shape.currentText())
        is_custom = shape_key == "custom_polygon"
        self.sa_polygon_text.setEnabled(is_custom)
        self.sa_polygon_pick.setEnabled(is_custom)
        self.sa_polygon_clear.setEnabled(is_custom)
        self.sa_polygon_pnorm.setEnabled(shape_key in ("convex_hull", "custom_polygon"))
        if not is_custom and self.sa_polygon_pick.isChecked():
            self.sa_polygon_pick.setChecked(False)   # also stops picking, via its own toggled handler
        self._stages_changed()

    _PICKING_STYLE = ("background-color: #2e7d32; color: white; font-weight: bold;")

    def _on_polygon_pick_toggled(self, on: bool):
        self.viewer.set_picking(on)
        if on:
            try:
                self._picked_pts = [[x * _KM, y * _KM]
                                    for x, y in _parse_xy_list(self.sa_polygon_text.text())]
            except ValueError:
                self._picked_pts = []
            self.sa_polygon_pick.setText("STOP Picking")
            self.sa_polygon_pick.setStyleSheet(self._PICKING_STYLE)
        else:
            self.sa_polygon_pick.setText("Pick On Plot")
            self.sa_polygon_pick.setStyleSheet("")
        self._update_pick_status()
        self._sync_polygon_preview()   # (re)draw current points either way -- picking off keeps them visible

    def _on_plot_point_picked(self, x_m: float, y_m: float):
        if not self.sa_polygon_pick.isChecked():
            return
        self._picked_pts.append([x_m, y_m])
        self.sa_polygon_text.setText(_fmt_xy_list([[x / _KM, y / _KM] for x, y in self._picked_pts]))
        self._update_pick_status()
        self._stages_changed()   # persists + redraws the map via _sync_polygon_preview

    def _on_polygon_clear_clicked(self):
        self._picked_pts = []
        self.sa_polygon_text.setText("")
        self._update_pick_status()
        self._stages_changed()   # persists the now-empty polygon + clears the map overlay

    def _update_pick_status(self):
        if self.sa_polygon_pick.isChecked():
            self.sa_pick_status_lbl.setText(
                f"\U0001F534 PICKING -- click the plot to add points ({len(self._picked_pts)} so far)")
            self.sa_pick_status_lbl.setStyleSheet("font-weight: bold; color: #2e7d32;")
        else:
            n = len(self._picked_pts)
            self.sa_pick_status_lbl.setText(
                f"picking off ({n} point{'s' if n != 1 else ''} saved)" if n else "picking off")
            self.sa_pick_status_lbl.setStyleSheet("font-weight: bold; color: gray;")

    def _sync_polygon_preview(self):
        """Push the current custom-polygon text to the viewer overlay (or clear it)."""
        if _shape_key(self.sa_shape.currentText()) != "custom_polygon":
            self.viewer.set_preview_polygon([])
            return
        try:
            pts_km = _parse_xy_list(self.sa_polygon_text.text())
        except ValueError:
            return   # leave whatever was last valid on screen
        self.viewer.set_preview_polygon([[x * _KM, y * _KM] for x, y in pts_km])

    def _stages_changed(self, *_):
        if self._loading:
            return
        m = self.project.mesh
        if not m.refinement_stages:
            m.refinement_stages = [MD.RefinementStage(kind="survey_area").asdict()]
        shape_key = _shape_key(self.sa_shape.currentText())
        try:
            buffer_list = _parse_list(self.sa_buffer_list.text())
            depth_list = _parse_list(self.sa_depth_list.text())
            polygon_xy = (_parse_xy_list(self.sa_polygon_text.text())
                         if shape_key == "custom_polygon" else [])
        except ValueError as exc:
            return self._error(exc)
        m.refinement_stages[0].update(dict(
            kind="survey_area", enabled=True,
            levels=self.sa_levels.value(),
            buffer_km=self.sa_buffer.value(),
            buffer_shrink=self.sa_buffer_shrink.value(),
            buffer_override=self.sa_buffer_override.isChecked(),
            buffer_km_list=buffer_list,
            max_depth_km=self.sa_depth.value(),
            depth_shrink=self.sa_depth_shrink.value(),
            depth_override=self.sa_depth_override.isChecked(),
            depth_km_list=depth_list,
            taper=self.sa_taper.isChecked(),
            shape=shape_key,
            polygon_pnorm=self.sa_polygon_pnorm.value(),
            polygon_xy_km=polygon_xy))
        self._update_steps()
        self._update_sa_info()
        self._sync_polygon_preview()
        self.notify_changed()

    def _update_sa_info(self):
        mm = self._mm
        m = self.project.mesh
        if not (mm and m.refinement_stages):
            self.sa_info_lbl.setText("-")
            return
        buffers_km, _depths_km = MD.survey_area_shells(
            m.refinement_stages[0], model_depth_m=mm.get("model_depth_m", 0.0))
        n = len(buffers_km)
        dx_cur = mm["dx_base_m"] / (2 ** n)
        dz_cur = mm["dz_base_1_m"] / (2 ** n)
        parts = [f"after {n} level{'s' if n != 1 else ''}: dx {dx_cur:.0f} m  ·  "
                f"min dz {dz_cur:.0f} m"]
        target_dx = mm.get("suggestion_target_m", 0.0)
        if target_dx:
            parts.append(f"target dx {target_dx:.0f} m (for the base-cells step's "
                        f"desired cells between sites)")
        c_km = m.scales.get("c_response_depth_km_at_tmin", 0.0)
        c_tmin = m.scales.get("c_response_tmin_s", 0.0)
        if c_km > 0:
            parts.append(f"C-response min depth at T_min {c_tmin:.3g} s = {c_km:.2f} km")
        self.sa_info_lbl.setText("  |  ".join(parts))

    # ---- extra local stages ------------------------------------ #
    def _add_stage(self, kind: str):
        m = self.project.mesh
        st = MD.RefinementStage(kind=kind, enabled=True, levels=1,
                                taper=False, shape="ellipse")
        if kind == "coastline":
            # clean the contour on the scale of the refinement: wiggles under a
            # quarter of the radius, and loops smaller than one radius-square
            st.radius_km, st.depth_km = 10.0, 20.0
            st.simplify_km = st.radius_km / 4.0
            st.min_area_km2 = st.radius_km ** 2
        m.refinement_stages.append(st.asdict())
        self._rebuild_extra_rows()
        self._update_steps()
        self.notify_changed()

    def _rebuild_extra_rows(self):
        while self.extra_box.count():
            item = self.extra_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        m = self.project.mesh
        for idx, st in enumerate(m.refinement_stages):
            if idx == 0:
                continue
            self.extra_box.addWidget(self._extra_row(idx, st))

    def _extra_row(self, idx: int, st: dict) -> QWidget:
        box = QGroupBox(f"{idx}. {self._STAGE_TITLES.get(st['kind'], st['kind'].title())}")
        box.setCheckable(True)
        box.setChecked(bool(st.get("enabled", True)))
        f = QFormLayout(box)
        widgets: dict[str, object] = {}

        lv = QSpinBox()
        lv.setRange(1, 6)
        lv.setValue(int(st.get("levels", 1)))
        f.addRow("Levels", lv)
        widgets["levels"] = lv

        if st["kind"] == "interface":
            note = QLabel("Splits the cells along the air/earth interface, so the "
                          "topography (and, in real-ocean mode, the seafloor) is "
                          "resolved. Levels = how many times.")
            note.setWordWrap(True)
            f.addRow(note)
        elif st["kind"] == "coastline":
            rad = _dspin(0.1, 500, 1, 2, float(st.get("radius_km", 10.0)), " km")
            dep = _dspin(0.5, 2000, 5, 1, float(st.get("depth_km", 20.0)), " km")
            simp = _dspin(0.0, 200, 0.5, 2, float(st.get("simplify_km", 0.0)), " km")
            simp.setSpecialValueText("Off")
            simp.setToolTip("Douglas-Peucker tolerance: coastline wiggles smaller than "
                            "this are straightened before refining.")
            area = _dspin(0.0, 1e6, 10, 0, float(st.get("min_area_km2", 0.0)), " km²")
            area.setSpecialValueText("Off")
            area.setToolTip("Islands / lakes enclosing less than this area are ignored.")
            f.addRow("Radius", rad)
            f.addRow("Depth", dep)
            f.addRow("Simplify Tolerance", simp)
            f.addRow("Drop Loops Smaller Than", area)
            widgets.update(radius_km=rad, depth_km=dep, simplify_km=simp, min_area_km2=area)
            prev = QPushButton("Preview Coastline…")
            prev.clicked.connect(lambda: self._preview_coast(idx, widgets))
            f.addRow(prev)
        elif st["kind"] == "stations":
            rad = _dspin(0.1, 500, 1, 2, float(st.get("radius_km", 5.0)), " km")
            dep = _dspin(0.5, 2000, 5, 1, float(st.get("depth_km", 20.0)), " km")
            f.addRow("Radius", rad)
            f.addRow("Depth", dep)
            widgets.update(radius_km=rad, depth_km=dep)
        else:  # zone
            cx = _dspin(-2000, 2000, 5, 1, float(st.get("center_x_km", 0.0)), " km")
            cy = _dspin(-2000, 2000, 5, 1, float(st.get("center_y_km", 0.0)), " km")
            hx = _dspin(1, 2000, 5, 1, float(st.get("half_x_km", 50.0)), " km")
            hy = _dspin(1, 2000, 5, 1, float(st.get("half_y_km", 50.0)), " km")
            dep = _dspin(0.5, 2000, 5, 1, float(st.get("depth_km", 50.0)), " km")
            for lab, w in (("Centre X", cx), ("Centre Y", cy),
                           ("Half-Size X", hx), ("Half-Size Y", hy), ("Depth", dep)):
                f.addRow(lab, w)
            widgets.update(center_x_km=cx, center_y_km=cy,
                           half_x_km=hx, half_y_km=hy, depth_km=dep)

        if st["kind"] != "interface":
            shape = QComboBox()
            shape.addItems(["Ellipse", "Rectangle"])
            shape.setCurrentText(str(st.get("shape", "ellipse")).capitalize())
            f.addRow("Shape", shape)
            widgets["shape"] = shape

        rm = QPushButton("Remove")
        rm.clicked.connect(lambda: self._remove_stage(idx))
        f.addRow(rm)

        def commit(*_):
            if self._loading:
                return
            s = self.project.mesh.refinement_stages[idx]
            s["enabled"] = box.isChecked()
            for key, w in widgets.items():
                s[key] = (w.currentText().lower() if isinstance(w, QComboBox)
                          else w.value())
            self._update_steps()
            self.notify_changed()

        box.toggled.connect(commit)
        for w in widgets.values():
            (w.currentIndexChanged if isinstance(w, QComboBox) else w.valueChanged).connect(commit)
        return box

    def _preview_coast(self, idx: int, widgets: dict):
        """Open the coastline preview for stage *idx*; 'Apply' writes the dialog's
        simplify / small-loop / radius values back into the stage's widgets."""
        from .coast_preview import CoastPreviewDialog

        path = self.project.projection.projected_dem_path
        if not (path and os.path.isfile(path)):
            self.log("coastline preview needs the projected DEM -- run the Project DEM step")
            return
        m = self.project.mesh
        st = dict(m.refinement_stages[idx])
        st.update({k: w.value() for k, w in widgets.items() if hasattr(w, "value")})
        half = sum(m.dx_blocks or [])
        try:
            dem = self._mesh_dem()
        except Exception as exc:  # noqa: BLE001 -- a bad polygon file, say
            self.log(f"coastline preview: {exc}")
            return
        dlg = CoastPreviewDialog(
            dem, st, stations_xy=m.scales.get("station_xy_m"),
            half_extent_m=(half, half) if half > 0 else None, parent=self)
        if dlg.exec_():
            for key, val in dlg.values().items():
                if key in widgets:
                    widgets[key].setValue(val)

    def _remove_stage(self, idx: int):
        m = self.project.mesh
        if 0 < idx < len(m.refinement_stages):
            m.refinement_stages.pop(idx)
            self._rebuild_extra_rows()
            self._update_steps()
            self.notify_changed()

    # ---- misc / topo ------------------------------------------- #
    def _store_misc(self, *_):
        if self._loading:
            return
        m = self.project.mesh
        m.use_topography = self.use_topo.isChecked()
        if not self.bg_auto.isChecked():
            m.background_resistivity_ohm_m = self.bg_res.value()
        self._update_dem_cover()
        self._update_ocean_info()
        self.notify_changed()

    # ---- steps / build --------------------------------------- #
    def _n_ops(self) -> int:
        return sum(MD.stage_op_count(s) for s in self.project.mesh.refinement_stages)

    _STAGE_TITLES = {"survey_area": "Refine Survey Area", "stations": "Around Stations",
                     "zone": "Zone", "interface": "Interface", "coastline": "Along Coastline"}

    def _step_breakdown(self) -> str:
        """'step 1-5 = Refine Survey Area, 6 = Around Stations' -- each step is
        one deal.II refinement op, not a QToolBox page; this spells out which
        stage's ops which step numbers belong to."""
        parts, n = [], 0
        for st in self.project.mesh.refinement_stages:
            c = MD.stage_op_count(st)
            if c <= 0:
                continue
            lo, hi = n + 1, n + c
            rng = f"{lo}" if lo == hi else f"{lo}-{hi}"
            parts.append(f"{rng} = {self._STAGE_TITLES.get(st.get('kind'), st.get('kind', '?'))}")
            n += c
        return "step:  " + "   ·   ".join(parts) if parts else "-"

    def _update_steps(self):
        n = self._n_ops()
        self.step.setRange(0, n)
        self.step.setValue(n)   # always default to "everything enabled", after any edit
        self.step_breakdown_lbl.setText(self._step_breakdown())

    def _check_stack(self):
        rep = mesh.check_stack()
        self.stack_lbl.setText("  |  ".join(f"{k.split('.')[-1]}: {v}"
                                            for k, v in rep.items()))
        self.log("checked deal.II / pyGoFEM imports")

    def _spec(self):
        return mesh.mesh_spec_from_project(self.project)

    def _store_out_format(self):
        if self._loading:
            return
        key = self.out_format.currentData() or "gofem"
        self._show_simpeg_ratio(key)
        if key != self.project.mesh.output_format:
            self.project.mesh.output_format = key
            self.notify_changed()

    def _show_simpeg_ratio(self, key):
        self.simpeg_ratio.setVisible(key == "simpeg")
        self.simpeg_ratio_lbl.setVisible(key == "simpeg")

    def _store_simpeg_ratio(self):
        if self._loading:
            return
        if self.simpeg_ratio.value() != self.project.mesh.simpeg_vertical_ratio:
            self.project.mesh.simpeg_vertical_ratio = float(self.simpeg_ratio.value())
            self.notify_changed()

    # ---- output folder ------------------------------------------------- #
    def _browse_out_dir(self):
        start = mesh.mesh_output_dir(self.project)
        d = QFileDialog.getExistingDirectory(self, "Mesh output folder", start)
        if d:
            self.out_dir_edit.setText(d)
            self._on_out_dir_edited()

    def _on_out_dir_edited(self):
        if self._loading:
            return
        self.project.mesh.output_dir = self.out_dir_edit.text().strip()
        self._refresh_out_dir_lbl()
        self.notify_changed()

    def _refresh_out_dir_lbl(self):
        resolved = mesh.mesh_output_dir(self.project)
        names = ", ".join((mesh.MESH_TRIA, mesh.MESH_VTU, mesh.MESH_MATERIALS,
                           mesh.MESH_MASK, mesh.RECEIVERS_CSV))
        exists = os.path.isfile(os.path.join(resolved, mesh.MESH_TRIA))
        self.out_dir_lbl.setText(
            f"→ {resolved}\nwrites: {names}"
            + ("   ·   model.tria already present here" if exists else ""))

    def _busy_build(self, on: bool):
        self.build_btn.setEnabled(not on)
        self.preview_btn.setEnabled(not on)
        wd = getattr(self, "_build_watchdog", None)
        if not on and wd is not None:
            wd.stop()

    def _build_busy(self) -> bool:
        """True only while a build/preview worker thread is actually alive.
        Used to reject a second concurrent build *and* to recover the buttons
        (see :meth:`_recover_build_buttons`) if a worker never reported back."""
        t = getattr(self, "_build_thread", None)
        try:
            return bool(t is not None and t.isRunning())
        except RuntimeError:      # Qt object already deleted
            return False

    def _recover_build_buttons(self):
        """Re-enable Preview / Build if nothing is actually running -- the
        escape hatch if a deal.II crash killed the worker without it ever
        emitting done/failed."""
        if not self._build_busy():
            self._busy_build(False)

    def on_shown(self) -> None:
        self._recover_build_buttons()

    def _start_build(self, fn, on_done) -> bool:
        if self._build_busy():
            self.log("a mesh build is already running -- wait for it to finish")
            return False
        self._busy_build(True)
        thread = run_async(self, fn, on_done, self._error)
        self._build_thread = thread
        # backstop: fires when the thread's event loop exits, even if the
        # done/failed handler path was somehow skipped
        thread.finished.connect(lambda: self._busy_build(False))
        # watchdog: if the worker thread dies without ever reporting back
        # (e.g. a hard deal.II crash), re-enable the buttons on our own.
        wd = getattr(self, "_build_watchdog", None)
        if wd is None:
            wd = self._build_watchdog = QTimer(self)
            wd.setInterval(3000)
            wd.timeout.connect(self._build_watchdog_tick)
        wd.start()
        return True

    def _build_watchdog_tick(self):
        if not self._build_busy():
            self._busy_build(False)
            self._build_watchdog.stop()

    def _preflight(self) -> str | None:
        """Human-readable reason the build can't run, or None if it's good."""
        m = self.project.mesh
        if not (m.dx_blocks and m.dz_earth):
            return "run 'Recompute From Data' first (no base cell lists yet)."
        try:
            mesh.output_format(self._spec())
        except ValueError as exc:
            return str(exc) + " -- change 'Mesh For' or the ocean model on page 2."
        pj = self.project.projection
        if not (pj.origin_lon or pj.origin_lat):
            return ("no local frame -- set the projection / origin on the "
                    "Survey tab (Apply), then 'Recompute From Data'.")
        rep = mesh.check_stack()
        broken = [k for k in ("PyDealII.Release", "gofem", "gofem.mesh_utils")
                  if rep.get(k, "").lower() != "ok"]
        if broken:
            self.stack_lbl.setText("  |  ".join(f"{k.split('.')[-1]}: {v}"
                                                for k, v in rep.items()))
            return ("deal.II / pyGoFEM not importable (" + ", ".join(broken)
                    + ") -- set the paths in Project ▸ Settings; see the Check line above.")
        if m.use_topography and not (
                self.project.projection.projected_dem_path
                and os.path.isfile(self.project.projection.projected_dem_path)):
            return ("'Fit Mesh to the Projected DEM' is on but no projected DEM "
                    "exists -- run the Project DEM step, or turn topography off "
                    "on page 6.")
        if m.ocean_mode != "none" and not (
                self.project.projection.projected_dem_path
                and os.path.isfile(self.project.projection.projected_dem_path)):
            return ("the ocean model needs the projected DEM -- run the Project DEM "
                    "step, or set the Ocean model to None on page 2.")
        return None

    def _preview_step(self):
        if not (self.project.mesh.dx_blocks and self.project.mesh.dz_earth):
            self.log("run 'Recompute From Data' first")
            return
        spec = self._spec()
        step = self.step.value()
        out_dir = mesh.mesh_output_dir(self.project)
        if self._start_build(
                lambda: mesh.run(spec, out_dir, self.log, stop_after_refinement=step),
                self._after_preview):
            self.result_lbl.setText(f"building to step {step}…")

    def _after_preview(self, result):
        self._busy_build(False)
        self.viewer.set_scales(self.project.mesh.scales)
        self.viewer.show_result(result)
        self.result_lbl.setText("step %(step)d / %(n_total_steps)d — %(n_cells)s cells "
                                "(smallest %(smallest_cell_m).0f m)" % result)

    def _build_full(self):
        problem = self._preflight()
        if problem:
            self.log("cannot build mesh: " + problem)
            self.result_lbl.setText("<b>cannot build:</b> " + problem)
            return
        spec = self._spec()
        out_dir = mesh.mesh_output_dir(self.project)
        if self._start_build(lambda: mesh.run(spec, out_dir, self.log),
                             self._after_full):
            self.result_lbl.setText(
                f"building full mesh into {out_dir} … a fine recipe can take "
                "many minutes.<br>If Preview / Build stay greyed after it "
                "finishes, switch to another tab and back.")
            self.log(f"building full mesh into {out_dir}")

    def _after_full(self, result):
        self._busy_build(False)
        m = self.project.mesh
        m.mesh_tria_path = result["tria"]
        m.mesh_vtu_path = result["vtu"]
        out_dir = os.path.dirname(result["tria"])
        def _sz(n):
            return f"{n/1048576:.1f} MB" if n >= 1048576 else f"{n/1024:.0f} KB"
        written = []
        for key in ("tria", "vtu", "materials", "mask", "receivers",
                    "femtic_mesh", "femtic_blocks", "simpeg_mesh", "simpeg_model", "simpeg_receivers"):
            p = result.get(key)
            if p and os.path.isfile(p):
                written.append(f"{os.path.basename(p)} ({_sz(os.path.getsize(p))})")
        self.result_lbl.setText(
            "<b>mesh built</b> — %(n_cells)s cells, smallest %(smallest_cell_m).0f m, "
            "largest %(largest_cell_m).0f m<br>" % result
            + f"written to <b>{out_dir}</b>:<br>" + "<br>".join(written)
            + "<br>ready for the GoFEM Files step.")
        self.log(f"mesh built: {result['n_cells']} cells → {out_dir}")
        self._refresh_out_dir_lbl()
        self.notify_changed()

    def _error(self, exc: Exception):
        self._busy_build(False)
        self.log(f"ERROR: {exc}")
        self.result_lbl.setText(
            "<b>BUILD FAILED:</b> " + str(exc) + "<pre>" + "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)[-4:])
            + "</pre>")

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        """Repopulate widgets from the project.  Never recomputes or emits --
        that only happens on an explicit user action (avoids a refresh <-> regen
        signal loop with the sibling panels)."""
        m = self.project.mesh
        prev, self._loading = self._loading, True
        try:
            if m.design_rules:
                self._set_rule_widgets(m.design_rules)
            if m.dx_blocks:
                self.dx_text.setText(_fmt_list(m.dx_blocks))
            if m.dz_earth:
                self.dz_text.setText(_fmt_list(m.dz_earth))
            self.use_topo.setChecked(m.use_topography)
            keys = [k for k, _label in self._OCEAN_MODES]
            self.ocean_mode.setCurrentIndex(
                keys.index(m.ocean_mode) if m.ocean_mode in keys else 0)
            self.out_format.setCurrentIndex(max(self.out_format.findData(m.output_format), 0))
            self.simpeg_ratio.setValue(m.simpeg_vertical_ratio)
            self._show_simpeg_ratio(m.output_format)
            self.ocean_res.setValue(m.ocean_resistivity_ohm_m)
            self.ocean_sheet.setValue(m.ocean_thickness_m / _KM)
            from ..backends.mesh_dem import SEA_RULES

            rules = list(SEA_RULES)
            self.sea_rule.setCurrentIndex(rules.index(m.sea_rule) if m.sea_rule in rules else 0)
            self.sea_gap.setValue(m.sea_gap_km)
            self.land_polys.setText(m.land_polygons_path)
            self.sea_polys.setText(m.sea_polygons_path)
            self._update_ocean_info()
            self._update_dem_cover()
            self.bg_res.setValue(m.background_resistivity_ohm_m)
            self.bg_res.setDisabled(self.bg_auto.isChecked())
            self.out_dir_edit.setText(m.output_dir)
            self._refresh_out_dir_lbl()
            if m.refinement_stages:
                self._load_stage0()
                self._rebuild_extra_rows()
            self._update_steps()
            self._recover_build_buttons()
            self.viewer.set_scales(m.scales)
            self.viewer.set_stations(m.scales.get("station_xy_m") or [])
        finally:
            self._loading = prev
