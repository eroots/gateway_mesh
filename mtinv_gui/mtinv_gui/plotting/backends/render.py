"""Figure recipes -> matplotlib figures.  Pure functions, no Qt.

The GUI preview, single-figure export and batch export all call the same
``render_*`` function on a fresh :class:`matplotlib.figure.Figure`, so what
you see is what gets written, and a saved project redraws a figure exactly.
"""
from __future__ import annotations

import os

import numpy as np

from . import slicing

RHO_LABEL = "Resistivity (Ω·m)"

#: offered in the colour-map combo (any matplotlib / ``cmc.<name>`` name works)
COLORMAPS = ["turbo_r", "jet_r", "rainbow_r", "Spectral", "RdYlBu", "viridis_r",
             "cmc.roma", "cmc.vik_r", "cmc.lajolla", "cmc.batlow_r"]


# --------------------------------------------------------------------------- #
def make_cmap(name: str, n_colors: int = 0):
    """Colour map by name; ``cmc.<name>`` -> Crameri (cmcrameri).  ``n_colors``
    > 0 makes it discrete (e.g. ``turbo_r`` with 32)."""
    import matplotlib as mpl

    if name.startswith("cmc."):
        import cmcrameri.cm as cmc  # noqa: F401 -- registers "cmc.*" names

    cmap = mpl.colormaps[name]
    return cmap.resampled(int(n_colors)) if n_colors and n_colors > 0 else cmap


def rho_norm(color):
    from matplotlib.colors import LogNorm

    return LogNorm(vmin=float(color.rho_min), vmax=float(color.rho_max))


def km_formatter(absolute: bool = False):
    from matplotlib.ticker import FuncFormatter

    if absolute:
        return FuncFormatter(lambda v, _pos: f"{abs(v) / 1000.0:g}")
    return FuncFormatter(lambda v, _pos: f"{v / 1000.0:g}")


def format_template(template: str, **values) -> str:
    """``str.format`` that leaves unknown ``{keys}`` alone instead of raising."""
    class _Keep(dict):
        def __missing__(self, key):
            return "{" + key + "}"

    try:
        return template.format_map(_Keep(values))
    except (ValueError, IndexError):
        return template


# --------------------------------------------------------------------------- #
# house style + panels (layouts draw each panel into a matplotlib SubFigure)
# --------------------------------------------------------------------------- #
def is_panel(fig) -> bool:
    from matplotlib.figure import SubFigure

    return isinstance(fig, SubFigure)


def begin(fig) -> None:
    """Clear *fig*; a whole figure gets constrained layout (a layout panel is
    laid out by its parent)."""
    fig.clear()
    if not is_panel(fig):
        fig.set_layout_engine("constrained")


def house_of(session):
    h = getattr(session, "house", None)
    return h if (h is not None and h.enabled) else None


FONT_DIRS = ["/mnt/c/Windows/Fonts", os.path.expanduser("~/.fonts"),
             os.path.expanduser("~/.local/share/fonts"), "/usr/share/fonts", "/Library/Fonts"]
#: family -> file-name prefix, for the usual suspects (anything else: the
#: family name without spaces, lower case)
_FONT_PREFIX = {"times new roman": "times", "helvetica": "helvetica", "arial": "arial",
                "calibri": "calibri", "liberation sans": "liberationsans"}
_fonts_tried: set = set()


def font_families() -> list[str]:
    import matplotlib.font_manager as fm

    return sorted(set(fm.get_font_names()))


def ensure_font(family: str) -> bool:
    """Make *family* available to matplotlib, registering font files found in
    :data:`FONT_DIRS` (e.g. the Windows fonts from WSL) the first time."""
    import glob

    import matplotlib.font_manager as fm

    if not family or family in fm.get_font_names():
        return True
    key = family.lower()
    if key in _fonts_tried:
        return family in fm.get_font_names()
    _fonts_tried.add(key)
    prefix = _FONT_PREFIX.get(key, key.replace(" ", ""))
    for d in FONT_DIRS:
        if not os.path.isdir(d):
            continue
        flat = d.lower().endswith(os.path.join("windows", "fonts"))     # 600+ files, no subfolders
        pattern = os.path.join(d, "*") if flat else os.path.join(d, "**", "*")
        for f in glob.glob(pattern, recursive=not flat):
            base = os.path.basename(f).lower()
            if not base.startswith(prefix) or not base.endswith((".ttf", ".otf")):
                continue
            try:
                if fm.FontProperties(fname=f).get_name().lower() == key:
                    fm.fontManager.addfont(f)
            except Exception:  # noqa: BLE001 -- unreadable font file
                continue
    return family in fm.get_font_names()


def style_rc(session, font_size: float) -> dict:
    """rcParams for a figure: its own font size, or the house style."""
    h = house_of(session)
    if h is None:
        return {"font.size": float(font_size)}
    ensure_font(h.font_family)
    rc = {"font.size": float(h.font_size), "font.family": h.font_family,
          "axes.linewidth": float(h.line_width), "xtick.major.width": float(h.line_width),
          "ytick.major.width": float(h.line_width), "xtick.minor.width": 0.6 * float(h.line_width),
          "ytick.minor.width": 0.6 * float(h.line_width), "patch.linewidth": float(h.line_width)}
    if h.editable_text:
        rc.update({"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    return rc


def font_size_of(session, own: float) -> float:
    h = house_of(session)
    return float(h.font_size) if h is not None else float(own)


def color_of(session, own):
    """The figure's own colour scale, or the house style's shared one."""
    h = house_of(session)
    return h.color if (h is not None and h.link_colors) else own


def width_of(session, own_in: float) -> float:
    h = house_of(session)
    return float(h.width_mm) / 25.4 if h is not None else float(own_in)


def dpi_of(session, own: int) -> int:
    h = house_of(session)
    return int(h.dpi) if h is not None else int(own)


def save_figure(fig, path: str, session, dpi: int) -> str:
    """``savefig`` under the same rcParams the figure was drawn with (PDF /
    SVG font embedding is decided at save time)."""
    import matplotlib as mpl

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with mpl.rc_context(style_rc(session, mpl.rcParams["font.size"])):
        fig.savefig(path, dpi=dpi_of(session, dpi), bbox_inches="tight")
    return path


# --------------------------------------------------------------------------- #
def add_polys(ax, sl: slicing.PolySlice, cmap, norm, zorder: float = 1):
    """Draw cell polygons coloured by resistivity; returns the collection."""
    from matplotlib.collections import PolyCollection

    pc = PolyCollection(sl.polys, cmap=cmap, norm=norm, zorder=zorder)
    pc.set_array(10.0 ** np.asarray(sl.values))
    # edge = face colour hides the hairline seams between cells in PDF viewers
    pc.set_edgecolor("face")
    pc.set_linewidth(0.3)
    ax.add_collection(pc)
    return pc


def marker_edge(marker: str, size: float) -> dict:
    """Black edge only where it helps: filled markers big enough that the edge
    does not swamp the colour."""
    from matplotlib.markers import MarkerStyle

    if MarkerStyle(marker).is_filled() and float(size) >= 10.0:
        return dict(edgecolors="k", linewidths=0.4)
    return {}


def draw_stations(ax, xy: np.ndarray, names=None, *, marker="v", size=12.0,
                  color="white", labels=False, label_size=6.0, zorder=10):
    if not len(xy):
        return
    ax.scatter(xy[:, 0], xy[:, 1], s=size, marker=marker, c=color,
               edgecolors="k", linewidths=0.5, zorder=zorder)
    if labels and names:
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        for (x, y), name in zip(xy, names):
            if x0 <= x <= x1 and y0 <= y <= y1:
                ax.annotate(name, (x, y), xytext=(3, 3), textcoords="offset points",
                            fontsize=label_size, zorder=zorder + 1)


def cbar_extend(color) -> str:
    """Colour-bar ends: square by default, pointed if the scale asks for it."""
    return "both" if getattr(color, "extend_arrows", False) else "neither"


def add_colorbar(fig, ax, mappable, where: str = "right", label: str = RHO_LABEL, **kw):
    if where == "none":
        return None
    orient = "horizontal" if where == "bottom" else "vertical"
    opts = dict(shrink=0.8 if orient == "vertical" else 0.6,
                pad=0.02 if orient == "vertical" else 0.1, extend="neither")
    opts.update(kw)
    cb = fig.colorbar(mappable, ax=ax, orientation=orient, **opts)
    cb.set_label(label)
    return cb


def _nice_step(span: float) -> float:
    """1, 2, 5 x 10^k degree step giving ~4 lines over *span* degrees."""
    raw = max(span / 4.0, 1e-6)
    mag = 10.0 ** np.floor(np.log10(raw))
    for m in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= m * mag:
            return float(m * mag)
    return float(10.0 * mag)


def _deg_label(v: float, step: float, pos: str, neg: str) -> str:
    dec = max(0, int(np.ceil(-np.log10(step) + 1e-9))) if step < 1 else 0
    return f"{abs(v):.{dec}f}°{pos if v >= 0 else neg}"


def draw_graticule(ax, maptr, step_deg: float = 0.0, color: str = "0.25",
                   lw: float = 0.5):
    """Lon/lat lines over a projected-km map, labelled on the top/right axes."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    t = np.linspace(0.0, 1.0, 60)
    bx = np.r_[x0 + (x1 - x0) * t, np.full_like(t, x1), x1 - (x1 - x0) * t, np.full_like(t, x0)]
    by = np.r_[np.full_like(t, y0), y0 + (y1 - y0) * t, np.full_like(t, y1), y1 - (y1 - y0) * t]
    lon, lat = maptr.to_lonlat(bx, by)
    lo0, lo1, la0, la1 = lon.min(), lon.max(), lat.min(), lat.max()
    step = float(step_deg) or _nice_step(max(lo1 - lo0, la1 - la0))
    pad_lo, pad_la = 0.5 * (lo1 - lo0), 0.5 * (la1 - la0)
    lons = np.arange(np.ceil(lo0 / step) * step, lo1 + 1e-9, step)
    lats = np.arange(np.ceil(la0 / step) * step, la1 + 1e-9, step)
    style = dict(color=color, lw=lw, ls=":", zorder=5)

    top_pos, top_lab, right_pos, right_lab = [], [], [], []
    for lo in lons:
        la = np.linspace(la0 - pad_la, la1 + pad_la, 300)
        E, N = maptr.from_lonlat(np.full_like(la, lo), la)
        ax.plot(E, N, **style)
        k = np.where(np.diff(np.sign(N - y1)) != 0)[0]
        if len(k):
            i = k[0]
            f = (y1 - N[i]) / (N[i + 1] - N[i])
            e = E[i] + f * (E[i + 1] - E[i])
            if x0 <= e <= x1:
                top_pos.append(e)
                top_lab.append(_deg_label(lo, step, "E", "W"))
    for la in lats:
        lo = np.linspace(lo0 - pad_lo, lo1 + pad_lo, 300)
        E, N = maptr.from_lonlat(lo, np.full_like(lo, la))
        ax.plot(E, N, **style)
        k = np.where(np.diff(np.sign(E - x1)) != 0)[0]
        if len(k):
            i = k[0]
            f = (x1 - E[i]) / (E[i + 1] - E[i])
            n = N[i] + f * (N[i + 1] - N[i])
            if y0 <= n <= y1:
                right_pos.append(n)
                right_lab.append(_deg_label(la, step, "N", "S"))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    top = ax.secondary_xaxis("top")
    top.set_xticks(top_pos, top_lab)
    right = ax.secondary_yaxis("right")
    right.set_yticks(right_pos, right_lab)
    for sec in (top, right):
        sec.tick_params(labelsize="small")


def _map_axes(ax, extent):
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_formatter(km_formatter())
    ax.yaxis.set_major_formatter(km_formatter())
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")


# --------------------------------------------------------------------------- #
# other datasets on maps (backends/layers.py)
# --------------------------------------------------------------------------- #
def draw_raster_image(ax, session, info, loaded, extent, zorder: float = 2):
    """A raster layer as an image over *extent*; returns the mappable."""
    from . import layers as L

    g = L.grid_on_extent(loaded, session.maptr, extent)
    vmin, vmax = L.value_range(info, loaded)
    return ax.pcolormesh(g.E, g.N, np.ma.masked_invalid(g.values), cmap=make_cmap(info.cmap),
                         vmin=vmin, vmax=vmax, alpha=float(info.alpha), shading="auto",
                         rasterized=True, zorder=zorder)


def draw_layers_on_map(ax, session, layer_infos, extent, skip: str = "") -> list:
    """Every enabled layer per its ``map_mode``; returns ``[(info, error)]``."""
    from . import layers as L

    ok, bad = session.active_layers(layer_infos or [])
    for info, loaded in ok:
        if info.name == skip or info.map_mode == "none":
            continue
        try:
            if isinstance(loaded, L.LoadedRaster):
                if info.map_mode == "image":
                    draw_raster_image(ax, session, info, loaded, extent, zorder=2)
                elif info.map_mode == "contours":
                    g = L.grid_on_extent(loaded, session.maptr, extent)
                    if np.isfinite(g.values).sum() > 4:
                        cs = ax.contour(g.E, g.N, g.values,
                                        levels=L.contour_levels(info, L.value_range(info, loaded)),
                                        colors=info.color, linewidths=float(info.linewidth),
                                        alpha=float(info.alpha), zorder=4)
                        if info.contour_labels:
                            ax.clabel(cs, fontsize=float(info.label_size), fmt="%g")
            elif isinstance(loaded, L.LoadedVector):
                _draw_vector(ax, info, loaded, extent)
            elif isinstance(loaded, L.LoadedPoints):
                _draw_points(ax, info, loaded, extent)
        except Exception as exc:  # noqa: BLE001
            bad.append((info, exc))
    return bad


def _gpd_plot(gdf, **kwargs):
    """``GeoDataFrame.plot`` onto ``kwargs["ax"]``.  geopandas ends with ``plt.draw()``,
    which makes an empty pyplot figure when none is current (a notebook then shows it,
    and the GUI keeps it); close any figure it made."""
    import matplotlib.pyplot as plt

    before = set(plt.get_fignums())
    gdf.plot(**kwargs)
    for num in set(plt.get_fignums()) - before:
        plt.close(num)


def _draw_vector(ax, info, vec, extent):
    x0, x1, y0, y1 = extent
    gdf = vec.gdf
    sub = gdf.cx[x0:x1, y0:y1]
    if not len(sub):
        return
    gt = sub.geom_type
    pts = sub[gt.isin(["Point", "MultiPoint"])]
    rest = sub[~gt.isin(["Point", "MultiPoint"])]
    if len(rest):
        _gpd_plot(rest, ax=ax, facecolor="none", edgecolor=info.color, linewidth=float(info.linewidth),
                  alpha=float(info.alpha), zorder=6)
    if len(pts):
        edge = marker_edge(info.marker, info.size)
        _gpd_plot(pts, ax=ax, color=info.color, marker=info.marker, markersize=float(info.size),
                  edgecolor=edge.get("edgecolors", info.color), linewidth=edge.get("linewidths", 0.0),
                  alpha=float(info.alpha), zorder=7)
    if vec.labels:
        idx = sub.index[:400]               # keep the map readable
        for i in idx:
            lab = vec.labels[i]
            if lab:
                p = gdf.geometry.iloc[i].representative_point()
                ax.text(p.x, p.y, lab, fontsize=float(info.label_size), zorder=7, clip_on=True)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)


def _draw_points(ax, info, pts, extent):
    from . import layers as L

    x0, x1, y0, y1 = extent
    keep = (pts.E >= x0) & (pts.E <= x1) & (pts.N >= y0) & (pts.N <= y1)
    if not keep.any():
        return
    kw = dict(s=float(info.size), marker=info.marker, alpha=float(info.alpha), zorder=7,
              **marker_edge(info.marker, info.size))
    if pts.values is not None:
        vmin, vmax = L.value_range(info, pts)
        ax.scatter(pts.E[keep], pts.N[keep], c=pts.values[keep], cmap=make_cmap(info.cmap),
                   vmin=vmin, vmax=vmax, **kw)
    else:
        ax.scatter(pts.E[keep], pts.N[keep], c=info.color, **kw)
    if pts.labels:
        for i in np.where(keep)[0][:400]:
            ax.text(pts.E[i], pts.N[i], pts.labels[i], fontsize=float(info.label_size), zorder=7)


def find_layer(layer_infos, name: str):
    for info in layer_infos or []:
        if info.name == name and info.enabled and info.kind in ("raster", "surface"):
            return info
    return None


# --------------------------------------------------------------------------- #
def plan_title(plan, depth_m: float, session) -> str:
    model = os.path.basename(getattr(session.model, "source_path", "") or "")
    return format_template(
        plan.title, depth_label=slicing.depth_label(depth_m, plan.reference),
        depth_km=f"{depth_m / 1000.0:g}", depth_m=f"{depth_m:g}",
        depth_tag=f"{depth_m:g}".replace(".", "p"), model=model,
        reference=slicing.REFERENCES.get(plan.reference, plan.reference))


def render_plan(fig, session, plan, depth_m: float | None = None, layers=None):
    """Draw one plan-view slice onto *fig* (cleared first); returns the axes.

    With ``plan.main_layer`` naming a raster layer, that layer is the main
    image instead of the model (a magnetics map with the same stations,
    extent and overlays).  *layers* (``project.layers``) are drawn per their
    map settings.
    """
    import matplotlib as mpl

    from . import layers as L

    depth = float(plan.depth_m if depth_m is None else depth_m)
    color = color_of(session, plan.color)
    fs = font_size_of(session, plan.font_size)
    with mpl.rc_context(style_rc(session, plan.font_size)):
        begin(fig)
        ax = fig.add_subplot(111)
        extent = session.extent(plan.extent)
        main = find_layer(layers, plan.main_layer) if plan.main_layer else None
        cb_label = RHO_LABEL
        if main is not None:
            loaded_main = session.layer(main)
            pc = draw_raster_image(ax, session, main, loaded_main, extent, zorder=1)
            cb_label = L.label(main, loaded_main)
        else:
            sl = slicing.plan_slice(session.model, session.maptr, depth, plan.reference, plan.crinkle)
            cmap = make_cmap(color.cmap, color.n_colors)
            norm = rho_norm(color)
            if sl.n:
                pc = add_polys(ax, sl, cmap, norm)
            else:
                from matplotlib.cm import ScalarMappable

                pc = ScalarMappable(norm=norm, cmap=cmap)
                ax.text(0.5, 0.5, "slice is empty at this depth", transform=ax.transAxes,
                        ha="center", va="center", color="0.4")
        _map_axes(ax, extent)
        draw_layers_on_map(ax, session, layers, extent, skip=main.name if main else "")
        _map_axes(ax, extent)
        if plan.show_stations:
            draw_stations(ax, session.stations_display(), session.receivers.names,
                          marker=plan.station_marker, size=plan.station_size,
                          color=plan.station_color, labels=plan.station_labels,
                          label_size=0.6 * fs)
        if plan.graticule:
            draw_graticule(ax, session.maptr, plan.graticule_step_deg)
        title = plan_title(plan, depth, session)
        if main is not None:
            title = format_template(plan.title, depth_label=L.label(main), layer=L.label(main),
                                    depth_km="", depth_m="", model="", reference="")
        if title:
            ax.set_title(title)
        add_colorbar(fig, ax, pc, plan.colorbar, label=cb_label, extend=cbar_extend(color))
        ax._mt_mappable, ax._mt_cb_label, ax._mt_extend = pc, cb_label, cbar_extend(color)
    return ax


def plan_size(plan, session) -> tuple[float, float]:
    """Plan figure size; the house width keeps the plan's own aspect."""
    w = width_of(session, plan.width_in)
    return w, float(plan.height_in) * w / max(float(plan.width_in), 1e-6)


def plan_file_name(plan, depth_m: float, session) -> str:
    model = os.path.splitext(os.path.basename(getattr(session.model, "source_path", "") or ""))[0]
    return format_template(plan.file_template, depth_tag=f"{depth_m:g}".replace(".", "p"),
                           depth_m=f"{depth_m:g}", reference=plan.reference, model=model)


def new_figure(width_in: float, height_in: float):
    from matplotlib.figure import Figure

    return Figure(figsize=(float(width_in), float(height_in)))


def export_plan(path: str, session, plan, depth_m: float | None = None, layers=None) -> str:
    """Render and save one plan view; the format follows the file extension."""
    fig = new_figure(*plan_size(plan, session))
    render_plan(fig, session, plan, depth_m, layers)
    return save_figure(fig, path, session, plan.dpi)


def export_plan_all(out_dir: str, session, plan, progress=None, layers=None) -> list[str]:
    """One file per depth in ``plan.depths_m``, named by ``plan.file_template``."""
    out = []
    depths = list(plan.depths_m)
    for i, d in enumerate(depths):
        if progress:
            progress(i, len(depths), f"{d:g} m")
        out.append(export_plan(os.path.join(out_dir, plan_file_name(plan, d, session)),
                               session, plan, d, layers))
    if progress:
        progress(len(depths), len(depths), "done")
    return out


# --------------------------------------------------------------------------- #
def render_station_map(fig, session, extent=None, show_surface: bool = True,
                       labels: bool = False):
    """Overview: the model's surface elevation under the stations, in the
    display CRS -- a quick visual check that the georeferencing is right."""
    fig.clear()
    fig.set_layout_engine("constrained")
    ax = fig.add_subplot(111)
    if not session.ready:
        ax.text(0.5, 0.5, "load + georeference a model", transform=ax.transAxes,
                ha="center", va="center", color="0.4")
        ax.set_axis_off()
        return ax
    from ..project import ExtentInfo

    extent = extent or session.extent(ExtentInfo())
    if show_surface:
        nx = 300
        ny = max(int(nx * (extent[3] - extent[2]) / max(extent[1] - extent[0], 1.0)), 2)
        E, N = np.meshgrid(np.linspace(extent[0], extent[1], nx),
                           np.linspace(extent[2], extent[3], ny))
        x, y = session.maptr.inverse(E, N)
        z = session.model.surface()(x, y)
        im = ax.pcolormesh(E, N, z, cmap="terrain", shading="auto", zorder=0)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Model surface elevation (m)")
    _map_axes(ax, extent)
    draw_stations(ax, session.stations_display(), session.receivers.names,
                  labels=labels, size=10)
    ax.set_title(session.maptr.display_label, fontsize="small")
    return ax


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
X_AXES = {"distance": "Distance (km)", "easting": "Easting (km)", "northing": "Northing (km)"}


def section_description(sec) -> str:
    if sec.kind == "ns":
        return f"N-S section, Easting {sec.coord_m / 1000.0:g} km"
    if sec.kind == "ew":
        return f"E-W section, Northing {sec.coord_m / 1000.0:g} km"
    if sec.kind == "stations":
        return f"Section along {len(sec.stations)} stations"
    return "Section along path"


def allowed_x_axes(sec) -> list[str]:
    """An N-S line has one easting: plotting against it makes no sense."""
    if sec.kind == "ns":
        return ["distance", "northing"]
    if sec.kind == "ew":
        return ["distance", "easting"]
    return list(X_AXES)


def _x_mode(sec) -> str:
    return sec.x_axis if sec.x_axis in allowed_x_axes(sec) else "distance"


def _x_from_s(mode, prof, s):
    if mode == "distance":
        return np.asarray(s)
    _x, _y, E, N = prof.at(s)
    return E if mode == "easting" else N


def section_size(sec, style, data, session=None) -> tuple[float, float]:
    """Figure size; with ``height_in == 0`` the height follows the VE so the
    section fills the width.  The house style sets the width."""
    w = width_of(session, style.width_in)
    if style.height_in > 0:
        return w, float(style.height_in) * (w / max(float(style.width_in), 1e-6))
    span_x = max(data.profile.length, 1.0)
    span_z = data.z_hi - data.z_lo
    axes_w = max(w - (1.9 if style.colorbar == "right" else 1.0), 1.0)
    h = axes_w * span_z * float(sec.ve) / span_x + 1.3 + (0.9 if style.colorbar == "bottom" else 0.0)
    if any(it.kind == "profile" for it in data.layer_items):
        h += float(style.profile_height_in) + 0.25
    return w, float(min(max(h, 2.0), 30.0))


def section_title(sec, style) -> str:
    ve_text = f" (VE = {sec.ve:g})" if abs(sec.ve - 1.0) > 1e-9 else ""
    return format_template(style.title, description=section_description(sec), ve_text=ve_text,
                           label=sec.label, ve=f"{sec.ve:g}", kind=sec.kind)


def render_section(fig, session, sec, style, data=None, others=None):
    """Draw one section onto *fig* (cleared first); returns the axes.

    With ``style.inset_map`` a small index map goes in a corner of the section
    (not in layout panels): *others* (normally the sections of *sec*'s group,
    :func:`sections.group_members`) in grey, *sec* in red."""
    import matplotlib as mpl
    from matplotlib.ticker import FuncFormatter
    from matplotlib.transforms import offset_copy

    from . import sections as SEC

    data = data or SEC.prepare(session, sec, style)
    prof = data.profile
    mode = _x_mode(sec)
    color = color_of(session, style.color)
    fs = font_size_of(session, style.font_size)
    profiles = [it for it in data.layer_items if it.kind == "profile"]
    title = section_title(sec, style)
    gap_in = 0.12
    if style.show_stations and style.station_labels or any(
            it.kind == "markers" and any(it.labels) for it in data.layer_items):
        gap_in += 0.55                          # rotated labels stand above the surface
    if profiles and style.colorbar == "right":
        gap_in = max(gap_in, 0.35)              # colour-bar arrow + top label vs the panel's right axis
    reserve = float(style.profile_height_in) + gap_in + (0.45 if (title or style.end_labels) else 0.1)
    panel = is_panel(fig)
    with mpl.rc_context(style_rc(session, style.font_size)):
        begin(fig)
        holder = None
        if panel and profiles:
            # a layout panel can't use the figure-wide layout rect: reserve the
            # profile panel's room with a placeholder row instead
            H = max(fig.bbox.height / fig.dpi, reserve + 0.5)
            gs = fig.add_gridspec(2, 1, height_ratios=[reserve, H - reserve], hspace=0.0)
            holder = fig.add_subplot(gs[0])
            holder.set_axis_off()
            ax = fig.add_subplot(gs[1])
        else:
            ax = fig.add_subplot(111)
        cmap = make_cmap(color.cmap, color.n_colors)
        norm = rho_norm(color)
        if style.show_interp and data.grid is not None:
            g = data.grid
            X = np.broadcast_to(_x_from_s(mode, prof, g.s), (len(g.z), len(g.s)))
            Z = np.broadcast_to(g.z[:, None], X.shape)
            V = np.ma.masked_invalid(10.0 ** g.values)
            mappable = ax.pcolormesh(X, Z, V, cmap=cmap, norm=norm, shading="nearest",
                                     rasterized=True, zorder=1)
        else:
            cut = data.cut
            if mode == "distance":
                polys = cut.polys
            else:
                col = 0 if mode == "easting" else 1
                polys = [np.c_[en[:, col], p[:, 1]] for p, en in zip(cut.polys, cut.en)]
            mappable = add_polys(ax, slicing.PolySlice(polys, cut.values), cmap, norm)
        if data.surface is not None:
            s, z = data.surface
            ax.plot(_x_from_s(mode, prof, s), z, color="k", lw=0.6, zorder=3)
        st = data.stations
        if style.show_stations and len(st.names):
            xs = _x_from_s(mode, prof, st.s)
            size = float(style.station_size)
            tr = offset_copy(ax.transData, fig=fig, y=0.5 * np.sqrt(size), units="points")
            ax.scatter(xs, st.z, s=size, marker=style.station_marker, c=style.station_color,
                       edgecolors="k", linewidths=0.5, transform=tr, clip_on=False, zorder=10)
            if style.station_labels:
                tr2 = offset_copy(ax.transData, fig=fig, y=np.sqrt(size) + 1.0, units="points")
                for x, z, name in zip(xs, st.z, st.names):
                    ax.text(x, z, name, transform=tr2, rotation=90, ha="center", va="bottom",
                            fontsize=0.6 * fs, clip_on=False, zorder=11)
        drew_legend_items = _section_layer_items(ax, fig, data, prof, mode, style)
        x0, x1 = (float(v) for v in _x_from_s(mode, prof, np.array([0.0, prof.length])))
        ax.set_xlim(x0, x1)
        ax.set_ylim(data.z_lo, data.z_hi)
        ax.set_aspect(float(sec.ve), adjustable="box")
        ax.xaxis.set_major_formatter(km_formatter())
        if style.vaxis == "depth":
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{-v / 1000.0 + 0.0:g}"))
            ax.set_ylabel("Depth (km)")
        else:
            ax.yaxis.set_major_formatter(km_formatter())
            ax.set_ylabel("Elevation (km)")
        ax.set_xlabel(X_AXES[mode])
        if drew_legend_items and style.legend:
            ax.legend(loc="lower right", fontsize="small", framealpha=0.85)
        if profiles and not panel:
            W, H = fig.get_size_inches()
            fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, max(1.0 - reserve / H, 0.3)))
        add_colorbar(fig, ax, mappable, style.colorbar, extend=cbar_extend(color))
        ax._mt_mappable, ax._mt_cb_label, ax._mt_extend = mappable, RHO_LABEL, cbar_extend(color)
        if getattr(style, "inset_map", False) and not panel:
            draw_inset_map(ax, session, sec, style, others or [])

        def finish(head, top):
            if style.end_labels:
                head.text(0.0, top, sec.label, transform=head.transAxes, ha="left", va="bottom",
                          fontweight="bold", fontsize=fs)
                head.text(1.0, top, sec.label + "′", transform=head.transAxes, ha="right",
                          va="bottom", fontweight="bold", fontsize=fs)
            if title:
                head.set_title(title, pad=18 if style.end_labels else 6, fontsize=1.2 * fs)
            ax._mt_head = head

        if profiles and panel:
            root = fig.get_figure(root=True)
            root.__dict__.setdefault("_mt_post_layout", []).append(
                lambda: finish(_panel_profile(root, ax, holder, profiles, prof, mode, style, gap_in), 1.02))
        elif profiles:
            finish(_profile_panel(fig, ax, profiles, prof, mode, style, gap_in), 1.02)
        else:
            finish(ax, 1.08 if (style.show_stations and style.station_labels) else 1.02)
    return ax


def _panel_profile(root, ax, holder, profiles, prof, mode, style, gap_in):
    """Layout panels: once the parent layout has run, put the profile panel
    above the section box (in root-figure coordinates) where the placeholder
    row reserved room for it."""
    renderer = root._get_renderer()
    inv = root.transFigure.inverted()
    ax.apply_aspect()
    box = ax.get_window_extent(renderer).transformed(inv)
    room = holder.get_window_extent(renderer).transformed(inv)
    W, H = root.get_size_inches()
    h = min(float(style.profile_height_in) / H, max(room.y1 - box.y1 - gap_in / H - 0.25 / H, 0.02))
    holder.remove()
    pax = root.add_axes([box.x0, box.y1 + gap_in / H, box.width, h])
    ax._mt_profile_axes = [pax] + _fill_profile_axes(pax, ax, profiles, prof, mode)
    return pax


def _section_layer_items(ax, fig, data, prof, mode, style) -> bool:
    """Depth-surface lines and vector / point markers on the section."""
    from matplotlib.transforms import offset_copy

    from . import layers as L

    any_legend = False
    for it in data.layer_items:
        info = it.info
        if it.kind == "line":
            ax.plot(_x_from_s(mode, prof, it.s), it.values, color=info.color,
                    lw=max(float(info.linewidth), 0.5), ls="--", zorder=4, label=L.label(info))
            any_legend = True
        elif it.kind == "markers" and len(it.s):
            xs = _x_from_s(mode, prof, it.s)
            ax.scatter(xs, it.values, s=float(info.size), marker=info.marker, c=info.color,
                       zorder=9, clip_on=False, label=L.label(info),
                       **marker_edge(info.marker, info.size))
            any_legend = True
            if it.labels:
                tr = offset_copy(ax.transData, fig=fig, y=np.sqrt(float(info.size)) + 1.0,
                                 units="points")
                for x, z, lab in zip(xs, it.values, it.labels):
                    if lab:
                        ax.text(x, z, lab, transform=tr, rotation=90, ha="center", va="bottom",
                                fontsize=float(info.label_size), clip_on=False, zorder=9)
    return any_legend


def _profile_panel(fig, ax, profiles, prof, mode, style, gap_in: float = 0.12):
    """Raster profiles in a panel exactly as wide as the (VE-shrunk) section
    box.  The panel is placed after the constrained layout has run; if its
    tick labels / axis labels stick out of the figure (they are often wider
    than the section's), the layout is re-run with that much more margin."""
    eng = fig.get_layout_engine()
    rect = list(eng.get().get("rect", (0.0, 0.0, 1.0, 1.0)))
    renderer = fig.canvas.get_renderer() if hasattr(fig.canvas, "get_renderer") else fig._get_renderer()
    pax = None
    for attempt in range(3):
        eng.set(rect=tuple(rect))
        eng.execute(fig)
        pax, extra = _place_profile_axes(fig, ax, profiles, prof, mode, style, gap_in)
        bb = pax.get_tightbbox(renderer)
        for a in extra:
            bb = bb.union([bb, a.get_tightbbox(renderer)])
        bb = bb.transformed(fig.transFigure.inverted())
        over_l, over_r = max(0.0, -bb.x0), max(0.0, bb.x1 - 1.0)
        if over_l + over_r < 0.002 or attempt == 2:
            break
        pax.remove()
        for a in extra:
            a.remove()
        rect[0] += over_l + 0.005
        rect[2] -= over_l + over_r + 0.01
    fig.set_layout_engine("none")
    return pax


def _place_profile_axes(fig, ax, profiles, prof, mode, style, gap_in):
    from . import layers as L

    ax.apply_aspect()
    pos = ax.get_position()
    W, H = fig.get_size_inches()
    pax = fig.add_axes([pos.x0, pos.y1 + gap_in / H, pos.width, float(style.profile_height_in) / H])
    return pax, _fill_profile_axes(pax, ax, profiles, prof, mode)


#: colours for profiles whose own colour is already taken in the same panel
PROFILE_COLORS = ["black", "tab:orange", "tab:blue", "tab:green", "tab:red", "tab:purple",
                  "tab:brown", "tab:pink"]
PROFILE_AXIS_OFFSET_PT = 48.0          # each extra axis on a side sits this far further out


def profile_axes_plan(profiles) -> list[tuple[str, int, str]]:
    """``[(side, slot, colour)]`` per profile.

    Every profile gets its own y-axis (curves with different units never
    share one).  ``auto`` takes the left axis if it is free, else the right;
    a second curve on a side gets slot 1 (an extra axis offset outward), etc.
    Colours already used by an earlier curve are swapped for the next free
    one, so two default-black profiles don't look the same."""
    from matplotlib.colors import to_hex

    def key(c):
        try:
            return to_hex(c)
        except ValueError:
            return str(c)

    slots = {"left": 0, "right": 0}
    used: set = set()
    plan = []
    for it in profiles:
        side = it.info.profile_axis
        if side not in ("left", "right"):
            side = "left" if slots["left"] == 0 else "right"
        colour = it.info.color or "black"
        if key(colour) in used:
            colour = next((c for c in PROFILE_COLORS if key(c) not in used), colour)
        used.add(key(colour))
        plan.append((side, slots[side], colour))
        slots[side] += 1
    return plan


def _fill_profile_axes(pax, ax, profiles, prof, mode) -> list:
    """Draw the profiles into *pax*, one colour-coded y-axis per profile (e.g.
    magnetics left in black, gravity right in orange);
    returns the extra (twin) axes."""
    from . import layers as L

    pax.set_xlim(ax.get_xlim())
    pax.tick_params(labelbottom=False)
    extra = []
    left_taken = False
    for it, (side, slot, colour) in zip(profiles, profile_axes_plan(profiles)):
        if side == "left" and slot == 0:
            a = pax
            left_taken = True
        else:
            a = pax.twinx()
            extra.append(a)
            if side == "left":
                a.yaxis.set_label_position("left")
                a.yaxis.set_ticks_position("left")
                a.spines["left"].set_visible(True)
            if slot:
                a.spines[side].set_position(("outward", PROFILE_AXIS_OFFSET_PT * slot))
            a.spines["right" if side == "left" else "left"].set_visible(False)
            a.spines["top"].set_visible(False)
            a.spines["bottom"].set_visible(False)
        a.plot(_x_from_s(mode, prof, it.s), it.values, color=colour,
               lw=max(float(it.info.linewidth), 0.6))
        lo, hi = it.vrange
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            pad = 0.05 * (hi - lo)
            a.set_ylim(lo - pad, hi + pad)
        a.set_ylabel(L.label(it.info), fontsize="small", color=colour)
        a.tick_params(axis="y", labelsize="small", colors=colour)
        a.spines[side].set_color(colour)
    if not left_taken:
        pax.set_yticks([])
    return extra


def section_file_name(sec, style, session) -> str:
    model = os.path.splitext(os.path.basename(getattr(session.model, "source_path", "") or ""))[0]
    return format_template(style.file_template, label=sec.label, kind=sec.kind, model=model)


def export_section(path: str, session, sec, style, data=None, layers=None, others=None) -> str:
    from . import sections as SEC

    data = data or SEC.prepare(session, sec, style, layers)
    fig = new_figure(*section_size(sec, style, data, session))
    render_section(fig, session, sec, style, data, others=others)
    return save_figure(fig, path, session, style.dpi)


def export_sections_all(out_dir: str, session, sections_list, style, progress=None,
                        layers=None, all_sections=None) -> list[str]:
    """*all_sections* (default *sections_list*): where each one's inset map
    finds the other sections of its group."""
    from . import sections as SEC

    pool = sections_list if all_sections is None else all_sections
    out = []
    n = len(sections_list)
    for i, sec in enumerate(sections_list):
        if progress:
            progress(i, n, sec.label)
        out.append(export_section(os.path.join(out_dir, section_file_name(sec, style, session)),
                                  session, sec, style, layers=layers,
                                  others=SEC.group_members(pool, sec)))
    if progress:
        progress(n, n, "done")
    return out


def render_index_map(fig, session, sections_list, style, plan=None, current: int = -1,
                     layers=None):
    """Plan-view map of where the sections run, labelled A ... A'.

    Background (``style.map_background``): the model surface, the plan view at
    ``plan.depth_m`` with the plan colours, or nothing.
    """
    import matplotlib as mpl
    from matplotlib import patheffects as pe

    with mpl.rc_context(style_rc(session, style.font_size)):
        begin(fig)
        ax = fig.add_subplot(111)
        if not session.ready:
            ax.text(0.5, 0.5, "load + georeference a model", transform=ax.transAxes,
                    ha="center", va="center", color="0.4")
            ax.set_axis_off()
            return ax
        lines = _section_lines(session, sections_list, style)
        extent = _index_extent(session, style, lines)
        x0, x1, y0, y1 = extent
        bg = style.map_background
        bg_layer = find_layer(layers, bg[6:]) if bg.startswith("layer:") else None
        if bg_layer is not None:
            from . import layers as L

            loaded_bg = session.layer(bg_layer)
            im = draw_raster_image(ax, session, bg_layer, loaded_bg, extent, zorder=0)
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label=L.label(bg_layer, loaded_bg))
        elif bg == "plan" and plan is not None:
            sl = slicing.plan_slice(session.model, session.maptr, plan.depth_m, plan.reference,
                                    plan.crinkle)
            pcolor = color_of(session, plan.color)
            pc = add_polys(ax, sl, make_cmap(pcolor.cmap, pcolor.n_colors), rho_norm(pcolor), zorder=0)
            add_colorbar(fig, ax, pc, "right", extend=cbar_extend(pcolor))
            ax.set_title(plan_title(plan, plan.depth_m, session), fontsize="small")
        elif bg == "surface":
            nx = 300
            ny = max(int(nx * (y1 - y0) / max(x1 - x0, 1.0)), 2)
            E, N = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
            x, y = session.maptr.inverse(E, N)
            im = ax.pcolormesh(E, N, session.model.surface()(x, y), cmap="terrain",
                               shading="auto", zorder=0, rasterized=True)
            fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label="Elevation (m)")
        _map_axes(ax, extent)
        draw_layers_on_map(ax, session, layers, extent,
                           skip=bg_layer.name if bg_layer is not None else "")
        _map_axes(ax, extent)
        draw_stations(ax, session.stations_display(), marker="o", size=6, color="white")
        halo = [pe.withStroke(linewidth=3, foreground="white")]
        for i, (sec, v) in enumerate(zip(sections_list, lines)):
            if v is None:
                continue
            hot = i == current
            ax.plot(v[:, 0], v[:, 1], color="#d62728" if hot else "k", lw=2.0 if hot else 1.2,
                    ls="-" if hot else "--", zorder=8, path_effects=halo)
            for (x, y), lab in ((v[0], sec.label), (v[-1], sec.label + "′")):
                ax.text(x, y, lab, fontweight="bold", ha="center", va="bottom", zorder=9,
                        path_effects=halo, color="#d62728" if hot else "k")
    return ax


def _section_lines(session, sections_list, style) -> list:
    """Display-CRS vertices per section (None where it can't be built)."""
    from . import sections as SEC

    lines = []
    for sec in sections_list:
        try:
            lines.append(SEC.section_vertices(sec, session, style))
        except ValueError:
            lines.append(None)
    return lines


def _index_extent(session, style, lines) -> tuple[float, float, float, float]:
    """Stations + map padding, grown to hold every section line."""
    from ..project import ExtentInfo

    x0, x1, y0, y1 = session.extent(ExtentInfo(mode="stations", padding_m=style.map_padding_m))
    for v in lines:
        if v is not None:
            x0, x1 = min(x0, v[:, 0].min()), max(x1, v[:, 0].max())
            y0, y1 = min(y0, v[:, 1].min()), max(y1, v[:, 1].max())
    return x0, x1, y0, y1


INSET_CORNERS = ("upper right", "upper left", "lower right", "lower left")


def draw_inset_map(ax, session, sec, style, others=()):
    """A small index map in a corner of the section axes *ax*: stations, the
    *others* as thin grey lines, *sec* in red with its A end labelled.
    Returns the inset axes (None if the section has no line)."""
    import matplotlib as mpl

    rest = [s for s in others if s.label != sec.label]
    lines = _section_lines(session, [sec] + rest, style)
    if lines[0] is None:
        return None
    x0, x1, y0, y1 = _index_extent(session, style, lines)
    dx, dy = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    size = max(float(getattr(style, "inset_size_in", 1.4)), 0.3)
    w_in, h_in = (size, size * dy / dx) if dx >= dy else (size * dx / dy, size)
    corner = style.inset_corner if style.inset_corner in INSET_CORNERS else "upper right"
    fig = ax.get_figure(root=True)
    pad_in = 0.06

    def locate(_iax, renderer):
        # the section box is only final at draw time (aspect, layout): place the
        # inset then, shrunk to fit inside it on a very flat section
        from matplotlib.transforms import Bbox

        ax.apply_aspect()
        pb = ax.get_position()
        W, H = fig.get_size_inches()
        k = min(1.0, (pb.width * W - 2 * pad_in) / w_in, (pb.height * H - 2 * pad_in) / h_in)
        k = max(k, 0.05)
        w, h = w_in * k / W, h_in * k / H
        px, py = pad_in / W, pad_in / H
        xa = pb.x1 - px - w if corner.endswith("right") else pb.x0 + px
        ya = pb.y1 - py - h if corner.startswith("upper") else pb.y0 + py
        return Bbox.from_bounds(xa, ya, w, h)

    iax = fig.add_axes([0.0, 0.0, 0.1, 0.1], label="inset_map")
    iax.set_axes_locator(locate)
    iax.set_zorder(20)
    iax.patch.set_facecolor("white")
    iax.patch.set_alpha(0.9)
    st = session.stations_display()
    if len(st):
        iax.scatter(st[:, 0], st[:, 1], s=3.0, c="0.3", linewidths=0, zorder=2)
    for v in lines[1:]:
        if v is not None:
            iax.plot(v[:, 0], v[:, 1], color="0.55", lw=0.6, zorder=3)
    v = lines[0]
    iax.plot(v[:, 0], v[:, 1], color="#d62728", lw=1.6, zorder=4)
    iax.text(v[0, 0], v[0, 1], sec.label, color="#d62728", fontweight="bold",
             fontsize=0.6 * float(mpl.rcParams["font.size"]), ha="center", va="bottom", zorder=5)
    iax.set_xlim(x0, x1)
    iax.set_ylim(y0, y1)
    iax.set_aspect("equal", adjustable="datalim")
    iax.set_xticks([])
    iax.set_yticks([])
    for sp in iax.spines.values():
        sp.set_linewidth(0.6)
    return iax


def export_index_map(path: str, session, sections_list, style, plan=None, layers=None) -> str:
    w = width_of(session, 7.0)
    fig = new_figure(w, 6.0 * w / 7.0)
    render_index_map(fig, session, sections_list, style, plan, layers=layers)
    return save_figure(fig, path, session, style.dpi)


def render_layer_preview(fig, session, info, loaded):
    """One layer on its own over the stations -- a check that it lines up.
    Returns a one-line summary."""
    from ..project import ExtentInfo
    from . import layers as L

    fig.clear()
    fig.set_layout_engine("constrained")
    ax = fig.add_subplot(111)
    extent = session.extent(ExtentInfo(mode="stations", padding_m=50000.0))
    if isinstance(loaded, L.LoadedRaster):
        im = draw_raster_image(ax, session, info, loaded, extent, zorder=0)
        fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02, label=L.label(info, loaded))
        finite = np.isfinite(loaded.values)
        vmin, vmax = L.value_range(info, loaded)
        epsg = loaded.crs.to_epsg()
        summary = (f"{loaded.values.shape[1]} × {loaded.values.shape[0]} px, "
                   f"{'EPSG:%s' % epsg if epsg else loaded.crs.name}; "
                   f"{finite.mean() * 100:.0f}% valid; range {vmin:.4g} – {vmax:.4g} "
                   f"{info.units or loaded.length_unit}")
    else:
        import copy

        shown = copy.copy(info)          # same read settings -> same cached data
        shown.map_mode, shown.enabled = "show", True
        draw_layers_on_map(ax, session, [shown], extent)
        if isinstance(loaded, L.LoadedVector):
            types = ", ".join(sorted(set(loaded.gdf.geom_type)))
            summary = f"{loaded.n} features ({types})"
        else:
            summary = f"{loaded.n} points; columns: {', '.join(loaded.columns)}"
    _map_axes(ax, extent)
    draw_stations(ax, session.stations_display(), marker="v", size=10, color="white", zorder=12)
    ax.set_title(L.label(info, loaded), fontsize="small")
    note = L.unit_note(info, loaded)
    return f"{summary}<br><b>{note}</b>" if note else summary
