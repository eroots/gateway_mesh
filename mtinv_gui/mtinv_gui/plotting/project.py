"""Plotting project state -> ``*.mtplot.json``.

Same pattern as :mod:`mtinv_gui.project`: a dataclass tree, one JSON file,
heavy artefacts referenced by path.  Each figure's settings (``PlanInfo``,
later sections and layouts) are complete enough to redraw it, which is what
makes the saved file a set of *figure recipes*.
"""
from __future__ import annotations

import dataclasses
import json
import os
import typing
from dataclasses import asdict, dataclass, field


PLOT_PROJECT_SUFFIX = ".mtplot.json"


@dataclass
class ModelInfo:
    code: str = "GoFEM"              # key of backends.model_source.MODEL_SOURCES
    model_path: str = ""
    receivers_path: str = ""         # "" -> looked for next to the model


@dataclass
class GeorefInfo:
    """How the model's local frame maps to a real CRS (see backends/georef.py).

    ``method`` picks the inputs used; ``crs`` / ``x0`` / ``y0`` are the result
    (whatever the method) and are what the plots use.
    """
    method: str = "stations"         # "mtinv" | "stations" | "manual"
    mtinv_project: str = ""          # method "mtinv": the setup GUI's *.mtinv.json
    station_source: str = ""         # method "stations": EDI folder or .lst
    fit_crs: str = "auto"            # method "stations": "auto" or e.g. "EPSG:32608"
    manual_crs: str = ""             # method "manual"
    manual_x0: float = 0.0
    manual_y0: float = 0.0
    # result
    crs: str = ""
    x0: float = 0.0
    y0: float = 0.0
    fit_rms_m: float = -1.0          # < 0: not a fit
    fit_max_m: float = -1.0
    fit_matched: int = 0
    # what figures / GeoTIFFs are drawn in ("" -> the model CRS)
    display_crs: str = ""


@dataclass
class ExtentInfo:
    """Map extent in the display CRS."""
    mode: str = "stations"           # "stations" | "model" | "custom"
    padding_m: float = 20000.0       # around the stations
    custom: list[float] = field(default_factory=list)   # [x_min, x_max, y_min, y_max]


@dataclass
class TightExtentInfo(ExtentInfo):
    """GeoTIFFs default to a tighter box than figures."""
    padding_m: float = 10000.0


@dataclass
class ColorInfo:
    cmap: str = "turbo_r"
    n_colors: int = 32               # 0 -> continuous
    rho_min: float = 1.0
    rho_max: float = 10000.0
    extend_arrows: bool = False      # pointed colour-bar ends (values beyond the range)


@dataclass
class LayerInfo:
    """Another dataset drawn with the model (see backends/layers.py).

    ``kind``:
      * ``"raster"``  -- a grid of values (magnetics, gravity, topography);
      * ``"surface"`` -- a grid of depths or elevations (Curie depth, LAB,
        basement); ``surface_ref`` says which, after ``scale`` (e.g. km -> m);
      * ``"vector"``  -- points / lines / polygons (shp, gpkg, kml, geojson);
      * ``"points"``  -- a CSV / text table of coordinates (+ value / label /
        depth columns).
    """
    name: str = ""
    kind: str = "raster"
    path: str = ""
    enabled: bool = True
    # ---- reading
    crs: str = ""                    # override the file's CRS ("" = from the file;
                                     # CSV: lon/lat if the values fit, else display CRS)
    band: int = 1
    sublayer: str = ""               # vector: layer inside a multi-layer file
    valid_min: float = -1.0e30       # raw values outside [valid_min, valid_max] = nodata
    valid_max: float = 1.0e30
    scale: float = 1.0               # applied after the validity test
    units: str = ""
    surface_ref: str = "depth_bsl"   # "depth_bsl" | "depth_bsurf" | "elevation"
    depth_unit: str = "auto"         # surface values / CSV depths (after scale) are in:
                                     # "auto" | "m" | "km" | "ft"; converted to m only
                                     # where they become depths on a section
    x_col: str = ""                  # CSV columns ("" = guess)
    y_col: str = ""
    value_col: str = ""
    label_col: str = ""
    depth_col: str = ""              # CSV: depth/elevation per point (uses surface_ref)
    # ---- maps (plan views, index map)
    map_mode: str = "contours"       # raster/surface: "none" | "image" | "contours";
                                     # vector/points: "none" | "show"
    cmap: str = "turbo"
    auto_range: bool = True          # colour / profile range = 2nd-98th percentile
    vmin: float = 0.0
    vmax: float = 1.0
    alpha: float = 1.0
    contour_interval: float = 0.0    # 0 -> ~10 levels over the range
    contour_labels: bool = False
    color: str = "black"
    linewidth: float = 0.8
    marker: str = "o"
    size: float = 12.0
    label_field: str = ""            # vector: attribute to write next to features
    label_size: float = 6.0
    # ---- sections
    section_mode: str = "profile"    # raster: "none" | "profile"; surface: + "line";
                                     # vector/points: "none" | "markers"
    profile_axis: str = "auto"       # profile panel: "auto" | "left" | "right" -- every
                                     # profile gets its own y-axis; auto fills left, then right
    section_offset_m: float = 5000.0  # vector/points: max distance from the section line


@dataclass
class PlanInfo:
    """A plan-view (depth slice) figure."""
    depths_m: list[float] = field(default_factory=lambda: [
        0.0, 1000.0, 2500.0, 5000.0, 10000.0, 20000.0, 50000.0])
    depth_m: float = 5000.0          # the one previewed / exported singly
    main_layer: str = ""             # "" = model resistivity; else a raster layer's name
    reference: str = "sea_level"     # "sea_level" | "surface"
    crinkle: bool = False            # whole cells the depth passes through (ParaView's
                                     # crinkle slice), drawn flat; False = exact cut
    extent: ExtentInfo = field(default_factory=ExtentInfo)
    color: ColorInfo = field(default_factory=ColorInfo)
    show_stations: bool = True
    station_labels: bool = False
    station_marker: str = "v"
    station_size: float = 12.0
    station_color: str = "white"
    graticule: bool = False
    graticule_step_deg: float = 0.0  # 0 -> automatic
    colorbar: str = "right"          # "right" | "bottom" | "none"
    title: str = "{depth_label}"     # {depth_label} {depth_km} {depth_m} {model}
    width_in: float = 7.0
    height_in: float = 6.0
    font_size: float = 10.0
    dpi: int = 300
    out_dir: str = ""
    file_template: str = "plan_{depth_tag}m.png"


@dataclass
class GeotiffInfo:
    depths_m: list[float] = field(default_factory=lambda: [
        0.0, 1000.0, 2500.0, 5000.0, 10000.0, 20000.0])
    reference: str = "sea_level"
    pixel_m: float = 100.0
    extent: TightExtentInfo = field(default_factory=TightExtentInfo)
    quantity: str = "log10_rho"      # key of backends.geotiff.QUANTITIES
    layout: str = "per_depth"        # "per_depth" | "stack"
    out_dir: str = ""
    prefix: str = "depth_slice"


@dataclass
class SectionInfo:
    """Geometry of one vertical section (its look is the shared SectionStyle).

    ``kind``:
      * ``"ns"`` -- straight north-south line at easting ``coord_m``, from
        northing ``start_m`` to ``end_m``;
      * ``"ew"`` -- east-west line at northing ``coord_m``, easting range;
      * ``"path"`` -- polyline ``vertices`` ([[E, N], ...]);
      * ``"stations"`` -- through the named ``stations`` in order, as a
        ``station_path`` polyline / spline / best-fit straight line,
        extended by ``extend_m`` at both ends.
    All coordinates are display-CRS metres.  ``start_m == end_m`` means
    "span the stations plus padding".
    """
    label: str = "A"
    kind: str = "ns"
    coord_m: float = 0.0
    start_m: float = 0.0
    end_m: float = 0.0
    vertices: list[list[float]] = field(default_factory=list)
    stations: list[str] = field(default_factory=list)
    station_path: str = "polyline"   # "polyline" | "spline" | "line"
    extend_m: float = 10000.0
    reverse: bool = False            # swap which end is A / A'
    x_axis: str = "distance"         # "distance" | "easting" | "northing"
    ve: float = 1.0                  # vertical exaggeration


@dataclass
class SectionStyle:
    """Shared look of every section figure (so a set of sections matches)."""
    auto_top: bool = True            # top of the plot = highest surface + 5 %
    z_top_m: float = 3000.0          # elevation (m) when not auto
    depth_bottom_m: float = 100000.0  # bottom of the plot, depth below sea level (m)
    vaxis: str = "depth"             # "depth" | "elevation"
    show_interp: bool = False        # False: true cells; True: interp_method
    interp_method: str = "linear"    # see backends.sections.INTERP_METHODS
    interp_nx: int = 600             # grid nodes along the section
    interp_nz: int = 300             # grid nodes vertically
    rbf_kernel: str = "thin_plate_spline"
    rbf_smoothing: float = 0.0
    rbf_epsilon: float = 1.0
    rbf_neighbors: int = 64
    gauss_sigma_px: float = 2.0
    color: ColorInfo = field(default_factory=ColorInfo)
    colorbar: str = "right"
    show_surface: bool = True        # model surface line
    show_stations: bool = True
    station_labels: bool = False
    station_marker: str = "v"
    station_size: float = 30.0
    station_color: str = "white"
    station_max_offset_m: float = 5000.0   # N-S / E-W / path: stations this close are shown
    end_labels: bool = True          # A ... A'
    title: str = "{description}{ve_text}"
    width_in: float = 10.0
    height_in: float = 0.0           # 0 -> from the vertical exaggeration
    font_size: float = 10.0
    dpi: int = 300
    out_dir: str = ""
    file_template: str = "section_{label}.png"
    map_background: str = "surface"  # index map: "surface" | "plan" | "none" | "layer:<name>"
    profile_height_in: float = 1.2   # panel above the section for raster profiles
    topo_profile: bool = False       # model-surface topography as a profile in that panel
    topo_color: str = "tab:brown"
    legend: bool = True              # legend for depth-surface lines / markers
    map_padding_m: float = 20000.0   # index map + default N-S / E-W span
    inset_map: bool = False          # small index map on each section figure
    inset_corner: str = "upper right"  # "upper right" | "upper left" | "lower right" | "lower left"
    inset_size_in: float = 1.4       # its longer side (in)


@dataclass
class HouseStyle:
    """One look for every figure (when ``enabled``): font, size, line widths,
    resolution, and a single colour scale shared by plan views, sections,
    index maps and layouts."""
    enabled: bool = False
    font_family: str = "DejaVu Sans"
    font_size: float = 8.0           # pt; titles / labels / ticks follow from it
    line_width: float = 0.6          # axes frame + ticks (pt)
    width_mm: float = 174.0          # every figure's width
    dpi: int = 600                   # raster output (PNG / TIFF)
    editable_text: bool = True       # PDF / SVG keep text as text (not outlines)
    link_colors: bool = True         # every figure uses ``color`` below
    color: ColorInfo = field(default_factory=ColorInfo)


@dataclass
class LayoutPanel:
    """One cell of a multi-panel layout.

    ``kind``: ``"plan"`` (the layout's plan settings at ``depth_m``, with an
    optional ``main_layer``), ``"section"`` (the project section labelled
    ``section``), ``"index"`` (index map of the layout's sections) or
    ``"empty"``.
    """
    kind: str = "plan"
    depth_m: float = 5000.0
    main_layer: str = ""
    section: str = ""
    title: str = ""                  # "" -> the panel's own title


@dataclass
class LayoutInfo:
    rows: int = 2
    cols: int = 2
    panels: list[LayoutPanel] = field(default_factory=list)   # row-major
    plan: PlanInfo = field(default_factory=PlanInfo)          # look of plan panels
    section_style: SectionStyle = field(default_factory=SectionStyle)
    shared_colorbar: str = "bottom"  # "bottom" | "right" | "none" (then per panel)
    panel_labels: str = "(a)"        # "(a)" | "a)" | "a" | "A" | "none"
    hide_shared_axes: bool = True    # axis labels only on the left column / bottom row
                                     # (between panels showing the same range)
    width_in: float = 7.0            # house style width overrides
    height_in: float = 0.0           # 0 -> from the panels' shapes
    wspace: float = 0.02
    hspace: float = 0.02


@dataclass
class RecipeInfo:
    """A saved figure: settings snapshotted when it was added, so it redraws
    the same later -- or on a newer model iteration (tab 1), which is the point.
    The model, georeferencing, sections' geometry and layers stay live."""
    name: str = ""
    kind: str = "plan"               # "plan" | "section" | "index" | "layout"
    section_group: str = "all"       # index map: which sections (backends.sections.SECTION_GROUPS)
    out_path: str = ""               # file written by Export (extension = format)
    plan: PlanInfo = field(default_factory=PlanInfo)
    section: SectionInfo = field(default_factory=SectionInfo)
    section_style: SectionStyle = field(default_factory=SectionStyle)
    layout: LayoutInfo = field(default_factory=LayoutInfo)


@dataclass
class PlotProject:
    name: str = "untitled"
    workdir: str = ""
    model: ModelInfo = field(default_factory=ModelInfo)
    georef: GeorefInfo = field(default_factory=GeorefInfo)
    plan: PlanInfo = field(default_factory=PlanInfo)
    geotiff: GeotiffInfo = field(default_factory=GeotiffInfo)
    layers: list[LayerInfo] = field(default_factory=list)
    sections: list[SectionInfo] = field(default_factory=list)
    section_style: SectionStyle = field(default_factory=SectionStyle)
    current_section: int = 0
    section_group: str = "all"       # Sections tab sub-tab: "ns" | "ew" | "other" | "all"
    house: HouseStyle = field(default_factory=HouseStyle)
    recipes: list[RecipeInfo] = field(default_factory=list)

    @property
    def path(self) -> str:
        return os.path.join(self.workdir, self.name + PLOT_PROJECT_SUFFIX)

    def save(self, path: str | None = None) -> str:
        path = path or self.path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "PlotProject":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        proj = rebuild(cls, raw)
        proj.workdir = os.path.dirname(os.path.abspath(path))
        return proj



def rebuild(dc_type, raw):
    """Plain dict (from JSON) -> dataclass tree, including ``list[Dataclass]``
    fields at any depth.  Missing keys keep their defaults, unknown keys are
    ignored -- old project files open, new ones don't break old code."""
    if not isinstance(raw, dict):
        return raw
    hints = typing.get_type_hints(dc_type)
    kwargs = {}
    for f in dataclasses.fields(dc_type):
        if f.name not in raw:
            continue
        val, hint = raw[f.name], hints.get(f.name)
        if dataclasses.is_dataclass(hint) and isinstance(val, dict):
            kwargs[f.name] = rebuild(hint, val)
        elif typing.get_origin(hint) is list and isinstance(val, list):
            (item,) = typing.get_args(hint) or (None,)
            kwargs[f.name] = [rebuild(item, v) if dataclasses.is_dataclass(item) else v for v in val]
        else:
            kwargs[f.name] = val
    return dc_type(**kwargs)


def copy_into(dst, src) -> None:
    """Overwrite every field of dataclass *dst* with a deep copy of *src*'s."""
    import copy

    for f in dataclasses.fields(dst):
        setattr(dst, f.name, copy.deepcopy(getattr(src, f.name)))
