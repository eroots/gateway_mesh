"""Project state: the single object every panel reads from and writes to.

A project is just a JSON file plus a working directory.  Panels mutate the
:class:`Project` in memory and call :meth:`Project.save`; heavy artefacts
(DEM arrays, meshes, solver inputs) are written next to the JSON as normal
files and referenced here by path.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any

from . import config


@dataclass
class InterpInfo:
    """Optional pre-processing: resample a folder of EDIs onto one shared,
    log-spaced period grid before the Survey / EDIs step (task 0)."""
    source_path: str = ""
    period_min_s: float = 0.001
    period_max_s: float = 1000.0
    periods_per_decade: float = 6.0
    interp_type: str = "slinear"
    out_dir: str = ""
    last_output_dir: str = ""   # set after a successful run; feeds "Use on Survey tab"


@dataclass
class SurveyInfo:
    """Summary derived from the loaded EDI files (task 0)."""
    edi_dir: str = ""
    edi_files: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)   # station ids to skip
    n_stations: int = 0
    # bounding box in geographic degrees
    min_lon: float = 0.0
    max_lon: float = 0.0
    min_lat: float = 0.0
    max_lat: float = 0.0
    # period coverage
    period_min_s: float = 0.0
    period_max_s: float = 0.0
    n_periods: int = 0


@dataclass
class ProjectionInfo:
    """Local coordinate frame (task 2).

    ``proj_kind`` (new projects default to ``"utm"``; saved projects keep their
    choice) picks how the model frame is built (see
    ``backends/projection.py``) -- every option is a map projection to metres:
      * ``"local"`` -- a projection built around the survey, origin
        exactly (0, 0) at the survey centre; ``local_proj`` picks which one:
        ``"tmerc"`` Transverse Mercator, ``"stere"`` Oblique Stereographic,
        ``"aeqd"`` Azimuthal Equidistant, ``"lcc"`` Lambert Conformal Conic
        (standard parallels ``local_lat1/2``).
      * ``"utm"`` -- UTM zone auto-picked from the survey centre
        (``utm_zone`` / ``northern_hemisphere`` / ``epsg`` filled in).
      * ``"epsg"`` -- a directly-entered code (e.g. EPSG:3979).
    For ``utm`` / ``epsg`` the model stays in that national grid, minus the
    origin's projected coordinate.

    ``origin_locked`` -- the user pinned ``origin_lon/lat`` by hand; the Survey
    tab then stops re-centring it on the survey when a new dataset loads.
    """
    proj_kind: str = "utm"           # "utm" | "local" | "epsg"
    local_proj: str = "tmerc"        # proj_kind == "local": tmerc | stere | aeqd | lcc
    local_lat1: float = 0.0          # lcc standard parallels, from the survey lat span
    local_lat2: float = 0.0
    utm_zone: int = 0
    northern_hemisphere: bool = True
    epsg: int = 0
    # model origin in geographic degrees (defaults to survey bbox centre)
    origin_lon: float = 0.0
    origin_lat: float = 0.0
    origin_locked: bool = False
    dem_sample_rate_m: float = config.DEFAULTS["dem_sample_rate_m"]
    grid_method: str = config.DEFAULTS["grid_method"]
    projected_dem_path: str = ""   # .npz written by the projection step
    regional_sample_rate_m: float = 2000.0   # posting of the regional grid in that .npz


@dataclass
class TopoInfo:
    """Downloaded / supplied elevation data (task 1)."""
    source: str = "GMRT"             # "GMRT" | "OpenTopography" | "local"
    local_path: str = ""             # user-supplied GeoTIFF / NetCDF
    out_dir: str = ""                # where downloads + topo_big.npz are written ("" -> workdir)
    downloaded_path: str = ""        # file fetched by the download step
    pad_arcsec: float = config.DEFAULTS["topo_pad_arcsec"]
    sample_arcsec: float = config.DEFAULTS["topo_sample_arcsec"]
    resampled_path: str = ""         # .npz of (lon_grid, lat_grid, z_grid)
    use_bathymetry: bool = True
    # A coarse regional DEM for the mesh padding (the detailed one above covers the
    # survey): "none" | "GMRT" | "OpenTopography" | "local"
    regional_source: str = "none"
    regional_local_path: str = ""
    regional_pad_km: float = 600.0   # how far past the survey bbox it reaches
    regional_sample_arcsec: float = 60.0
    regional_downloaded_path: str = ""
    regional_resampled_path: str = ""   # topo_regional.npz


@dataclass
class MeshInfo:
    """deal.II mesh design parameters (task 3).

    ``dx_blocks`` / ``dz_*`` / ``refinements`` are *outputs* of the data-driven
    sizing in ``backends/mesh_design.py`` -- empty until "Recompute from data"
    is run (or the user edits them by hand).  ``design_rules`` holds any
    overrides to :class:`mesh_design.DesignRules`; ``scales`` caches the last
    values pulled from the survey so the panel can redisplay them.
    """
    dx_blocks: list[float] = field(default_factory=list)   # half-widths, m (backend mirrors)
    dz_air: list[float] = field(default_factory=list)      # small -> large (backend reverses)
    dz_earth: list[float] = field(default_factory=list)
    background_resistivity_ohm_m: float = config.DEFAULTS["background_resistivity_ohm_m"]
    use_topography: bool = True
    use_bathymetry: bool = True            # legacy; superseded by ocean_mode
    # Ocean (page 2 of the Mesh tab): "none" -> bathymetry ignored (flat sea level),
    # "thin_sheet" -> top ocean_thickness_m carries the real water-column conductance,
    # "real" -> the seafloor is part of the fitted surface.
    ocean_mode: str = "none"
    ocean_resistivity_ohm_m: float = config.DEFAULTS["ocean_resistivity_ohm_m"]
    ocean_thickness_m: float = 1500.0      # thin-sheet thickness X
    # Which below-sea-level DEM points are sea (mesh_dem.SEA_RULES): "connected" to the
    # open sea, or "all"; the polygon files force areas to land / to sea.
    sea_rule: str = "connected"
    sea_gap_km: float = 10.0               # "connected": below-sea-level areas this close join up
    land_polygons_path: str = ""
    sea_polygons_path: str = ""
    # form-based refinement stages (biggest -> smallest); see mesh_design.RefinementStage
    refinement_stages: list[dict[str, Any]] = field(default_factory=list)
    suggest_locked: bool = False   # user hand-set the survey-refinement levels; stop auto-syncing
    refinements: list[dict[str, Any]] = field(default_factory=list)  # legacy raw ops (unused)
    design_rules: dict[str, Any] = field(default_factory=dict)       # DesignRules overrides
    scales: dict[str, Any] = field(default_factory=dict)            # cached survey scales
    output_dir: str = ""           # where "Build Full Mesh" writes model.tria/.vtu/... ;
                                   # blank -> the project working directory. The GoFEM Files
                                   # step reads the mesh from here too.
    mesh_tria_path: str = ""
    mesh_vtu_path: str = ""
    # which code the build writes for (backends.mesh.OUTPUT_FORMATS); the design is shared
    output_format: str = "gofem"
    simpeg_vertical_ratio: float = 1.0     # SimPEG octree: finest cell width / height


@dataclass
class GofemInfo:
    """Solver input file settings (final step)."""
    output_dir: str = "gofem_run"
    data_type: str = "Z"            # "Z" | "PT" | "Tipper" | "Z_Tipper"
    error_floor: float = config.DEFAULTS["error_floor_z"]
    period_min_s: float = config.DEFAULTS["period_min_s"]
    period_max_s: float = config.DEFAULTS["period_max_s"]
    period_percentage: float = config.DEFAULTS["period_percentage"]
    n_parallel_frequencies: int = 16
    element_order: int = 1
    boundary_condition: str = "Dirichlet2D"
    refine_around_receivers: int = 2       # modeling prm; inversion prm uses this - 1
    target_rms: float = 1.0
    max_iterations: int = 50
    inner_iterations: int = 50
    regularization_update: int = 5
    scaling_factor_s0: float = 0.1         # initial regularization scaling
    active_domain_box: list[float] = field(default_factory=list)  # [] -> derive from scales
    data_file_path: str = ""              # last written .gdat
    # FEMTIC run (written when the mesh was built "For" FEMTIC): control.dat
    femtic_iterations: int = 20           # 0 = a forward run only
    femtic_threads: int = 4               # OpenMP threads per MPI process
    femtic_trade_off: float = 10.0


@dataclass
class RefineInfo:
    """Refine Model tab: its own inputs, independent of the other tabs, since the model
    being refined often predates this project.  See ``backends/refine.py``."""
    tria: str = ""
    materials: str = ""
    receivers: str = ""
    station_mode: str = "none"      # "none" | "add" | "replace"
    stations_edi: str = ""          # folder or .lst of the new stations
    frame: str = "fit"              # "fit" | "project" | "manual"
    frame_project: str = ""         # .mtinv.json the model was built with
    frame_fit_edis: str = ""        # folder or .lst of the *existing* receivers' EDIs
    frame_crs: str = ""
    frame_x0: float = 0.0
    frame_y0: float = 0.0
    rules: list[dict[str, Any]] = field(default_factory=list)   # refine.RefineRule dicts
    smooth: bool = False            # smooth the refined cells' conductivities (refine.smooth_refined)
    smooth_mode: str = "relative"   # "relative" (x the original cell's size) | "km"
    smooth_horizontal: float = 0.25
    smooth_vertical: float = 0.25
    refit: bool = False
    refit_dem: str = ""             # projected DEM .npz, in the model's frame
    blend_km: float = 0.0           # fade at the DEM's edge; 0 -> automatic
    out_dir: str = ""
    stem: str = "model_refined"
    max_cells: int = 5_000_000
    write_mask: bool = False
    fixed_ids: list[int] = field(default_factory=list)
    write_gofem: bool = True
    existing_data: str = ""         # .gdat
    existing_frequencies: str = ""
    active_domain_box: str = ""     # the inversion's: n0, e0, z0, n1, e1, z1 in metres
    frame_file: str = ""            # model_frame.json, for frame == "file"
    # workflow "map": put this model's conductivities on another mesh (backends/remap.py)
    workflow: str = "refine"        # "refine" | "map"
    target_tria: str = ""
    target_materials: str = ""
    target_receivers: str = ""
    target_frame: str = "project"   # "project" | "file" | "fit" | "manual", as frame
    target_frame_project: str = ""
    target_frame_file: str = ""
    target_frame_fit_edis: str = ""
    target_frame_crs: str = ""
    target_frame_x0: float = 0.0
    target_frame_y0: float = 0.0
    fill_ids: list[int] = field(default_factory=lambda: [2])   # target ids that get the model's values
    map_smooth: float = 0.25        # x the old cell's size; 0 = the old cells as they are
    map_depth: str = "surface"      # "surface": depth below the surface; "elevation"


@dataclass
class Project:
    name: str = "untitled"
    code: str = "GoFEM"
    workdir: str = ""
    interp: InterpInfo = field(default_factory=InterpInfo)
    survey: SurveyInfo = field(default_factory=SurveyInfo)
    topo: TopoInfo = field(default_factory=TopoInfo)
    projection: ProjectionInfo = field(default_factory=ProjectionInfo)
    mesh: MeshInfo = field(default_factory=MeshInfo)
    gofem: GofemInfo = field(default_factory=GofemInfo)
    refine: RefineInfo = field(default_factory=RefineInfo)

    # ------------------------------------------------------------------ io ---
    @property
    def path(self) -> str:
        return os.path.join(self.workdir, self.name + config.PROJECT_FILE_SUFFIX)

    def save(self, path: str | None = None) -> str:
        path = path or self.path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        proj = _from_dict(cls, raw)
        # old files stored the sea *conductivity* (3.2 S/m) in this resistivity field
        if abs(proj.mesh.ocean_resistivity_ohm_m - 3.2) < 1e-9:
            proj.mesh.ocean_resistivity_ohm_m = config.DEFAULTS["ocean_resistivity_ohm_m"]
        if proj.mesh.ocean_mode not in ("none", "thin_sheet", "real"):
            proj.mesh.ocean_mode = "none"
        if proj.mesh.output_format not in ("gofem", "femtic", "simpeg"):
            proj.mesh.output_format = "gofem"
        proj.workdir = os.path.dirname(os.path.abspath(path))
        return proj


def _from_dict(dc_type, raw: dict):
    """Rebuild a nested dataclass tree from a plain dict, ignoring extras.

    Nested dataclasses are detected via ``field.default_factory`` (every nested
    dataclass field here is ``field(default_factory=<DataclassType>)``).  This
    avoids evaluating the string annotations, which breaks on Python 3.8 for
    PEP 585 generics like ``list[dict[str, Any]]``.

    Missing keys fall back to the dataclass default (forward compatible);
    unknown keys in *raw* are ignored (backward compatible).
    """
    kwargs = {}
    for f in dataclasses.fields(dc_type):
        if f.name not in raw:
            continue
        val = raw[f.name]
        factory = getattr(f, "default_factory", dataclasses.MISSING)
        if (factory is not dataclasses.MISSING
                and dataclasses.is_dataclass(factory)
                and isinstance(val, dict)):
            kwargs[f.name] = _from_dict(factory, val)
        else:
            kwargs[f.name] = val
    return dc_type(**kwargs)
