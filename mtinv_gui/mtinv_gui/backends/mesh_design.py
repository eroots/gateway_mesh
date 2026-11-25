"""Derive a starting mesh from the survey data using standard 3-D MT rules.

Length scale everywhere is the skin depth  ``delta = 503 * sqrt(rho * T)``  m.

  delta_min = delta(T_min, rho_shallow)   -> near-surface cell size
  delta_max = delta(T_max, rho_deep)      -> model depth + edge padding

Choices locked in with the user:
  * rho for depth/padding = rho_ssq at the LONGEST period (rho_deep)
  * air column mirrors the total earth thickness (GoFEM Dirichlet2D convention)
  * horizontal core cell = min_station_spacing / 3

Everything is returned as editable numbers + a list of pass/warn diagnostics;
nothing here needs deal.II.
"""
from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass, field
from typing import Any


def skin_depth(period_s: float, rho_ohm_m: float) -> float:
    return 503.0 * math.sqrt(max(rho_ohm_m, 1e-6) * max(period_s, 1e-12))


def round_nice(v: float) -> float:
    """Snap a value to a round number one order of magnitude below it.

    < 100 -> nearest 10, < 1000 -> nearest 100, < 1e4 -> nearest 1000, and so
    on. Never returns 0 for a positive input (floors at the step).
    """
    if v <= 0:
        return v
    if v < 1e2:
        step = 1e1
    elif v < 1e3:
        step = 1e2
    elif v < 1e4:
        step = 1e3
    elif v < 1e5:
        step = 1e4
    else:
        step = 1e5
    return max(round(v / step) * step, step)


# --------------------------------------------------------------------------- #
@dataclass
class DesignRules:
    """Constants behind the auto-sizing. All are exposed as editable fields.

    The mesh is a deal.II octree: a *coarse* base grid that is adaptively
    refined near the survey. So the base cell sizes are large; the small cells
    the inversion sees come from ``surface_refine_levels`` isotropic splits.
    """
    # -- base grid (NOT sized to fit between stations -- that is what the
    #    survey-area refinement is for) --------------------------------------
    base_cell_count: int = 20             # base half-blocks spanning the padded half-width
    base_cell_override_m: float = 0.0     # 0 -> auto (base_cell_count rule); else use this
    override_cell_lists: bool = False     # True -> dx_blocks/dz_earth are hand-typed, not derived
    vertical_growth: float = 1.25          # geometric factor for base earth layers
    dz_floor_m: float = 10.0              # absolute minimum cell size
    max_dz_fraction: float = 0.20         # cap any base dz at this * model depth
    depth_skin_multiple: float = 3.0      # model bottom = this * delta_max  (>= 3)
    pad_skin_multiple: float = 3.0        # edge padding   = this * delta_max  (>= 3)
    horizontal_pad_growth: float = 1.5    # geometric factor for padding cells (graded dx only)
    core_buffer_cells: int = 2

    # -- survey-refinement suggestion (bottom of the Base Cells step) --------
    cells_between_sites: float = 3.0      # target # cells between adjacent stations
    suggestion_use_median: bool = False   # False -> use minimum spacing, True -> median

    # legacy / unused fields kept so old projects still load
    surface_refine_levels: int = 5
    first_layer_fraction: float = 8.0
    skin_cell_fraction: float = 2.0
    max_receiver_refine_levels: int = 0
    constant_dx: bool = True              # True -> every dx half-block == dx_base (uniform)
    round_grid: bool = True               # snap dx/dz to the nearest 1e3 / 1e4 / 1e5
    # optional resistive-mantle transition: a 20 ohm-m half-space below 410 km.
    # It limits the penetration depth, so the model bottom is capped at 1000 km.
    mantle_transition: bool = False
    mantle_depth_km: float = 410.0
    mantle_resistivity_ohm_m: float = 20.0
    mantle_max_model_depth_km: float = 1000.0

    def merged(self, overrides: dict | None) -> "DesignRules":
        if not overrides:
            return self
        d = dataclasses.asdict(self)
        d.update({k: v for k, v in overrides.items() if k in d})
        return DesignRules(**d)


@dataclass
class Diagnostic:
    name: str
    value: float
    target: str
    ok: bool
    note: str = ""


@dataclass
class MeshSuggestion:
    dx_blocks: list[float]                 # half-widths, centre -> edge (backend mirrors)
    dz_air: list[float]                    # small -> large (backend reverses)
    dz_earth: list[float]
    stages: list[dict[str, Any]]           # form-based refinement stages (see RefinementStage)
    meta: dict[str, float]
    diagnostics: list[Diagnostic] = field(default_factory=list)

    @property
    def n_base_cells(self) -> int:
        nx = 2 * len(self.dx_blocks)
        nz = len(self.dz_air) + len(self.dz_earth)
        return nx * nx * nz


# --------------------------------------------------------------------------- #
# Refinement stages -- meaningful, form-based, biggest -> smallest.
# --------------------------------------------------------------------------- #
STAGE_KINDS = {
    "survey_area": "Refine the whole survey area (auto polygon = station hull + buffer).",
    "stations":    "Refine a radius around every station.",
    "zone":        "Refine a rectangular / elliptical zone you place by hand.",
    "interface":   "Refine along the air/earth interface.",
    "coastline":   "Refine along the coastline (0 m contour of the DEM).",
}


@dataclass
class RefinementStage:
    kind: str = "survey_area"
    enabled: bool = True
    levels: int = 1
    taper: bool = True                 # shrink the refined depth each level
    shape: str = "ellipse"            # "ellipse" (refine_around_points, pnorm 2) |
                                      # "rectangle" (refine_around_points, pnorm 1) |
                                      # "convex_hull" (refine_within_polygon) |
                                      # "custom_polygon" (refine_within_polygon, custom vertices)
    polygon_pnorm: float = 1.0        # convex_hull / custom_polygon only: refine_within_polygon
                                      # pnorm, 1.0-2.0 -- 1 = exact boundary test, >1 = bounding-
                                      # circle approximation (gofem sizes it off the footprint diagonal)
    # survey_area -- shells are numbered outer (0, the ``buffer_km`` /
    # ``max_depth_km`` shell) -> inner. Each further shell shrinks the
    # previous one geometrically (``buffer_shrink`` / ``depth_shrink``,
    # 0.25-0.75), unless the matching ``*_override`` flag + explicit
    # ``*_km_list`` (outer -> inner) replaces the whole sequence.
    buffer_km: float = 20.0
    buffer_shrink: float = 0.5
    buffer_override: bool = False
    buffer_km_list: list = field(default_factory=list)
    max_depth_km: float = 100.0
    depth_shrink: float = 0.55
    depth_override: bool = False
    depth_km_list: list = field(default_factory=list)
    polygon_xy_km: list = field(default_factory=list)   # shape == "custom_polygon": custom footprint
    # stations
    radius_km: float = 5.0
    depth_km: float = 20.0
    # zone
    center_x_km: float = 0.0
    center_y_km: float = 0.0
    half_x_km: float = 50.0
    half_y_km: float = 50.0
    # coastline (reuses radius_km / depth_km / levels / taper / shape): the 0 m DEM
    # contour is first cleaned up -- wiggles smaller than ``simplify_km`` are
    # straightened and loops enclosing less than ``min_area_km2`` are dropped
    # (0 = off) -- then sampled every ``radius_km`` and refined around.
    simplify_km: float = 0.0
    min_area_km2: float = 0.0

    def asdict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def default_stages(rules: DesignRules, *, model_depth_m: float,
                   survey_levels: int | None = None,
                   c_response_max_km: float = 0.0) -> list[dict[str, Any]]:
    levels = max(int(rules.surface_refine_levels if survey_levels is None else survey_levels), 0)
    if c_response_max_km > 0:
        shell0_km = round(1.25 * c_response_max_km, 1)
    else:                              # no C-response yet -- fall back to the old estimate
        shell0_km = max(round(0.35 * model_depth_m / 1e3, -1), 50.0)
    survey = RefinementStage(
        kind="survey_area", levels=levels,
        buffer_km=shell0_km, max_depth_km=shell0_km,
        taper=True, shape="ellipse")
    stations = RefinementStage(kind="stations", enabled=False, levels=1,
                               radius_km=5.0, depth_km=10.0, taper=False)
    return [survey.asdict(), stations.asdict()]


def survey_area_shells(stage: dict, *, model_depth_m: float = 0.0) -> tuple[list[float], list[float]]:
    """Per-shell ``(buffer_km, depth_km)`` for a survey_area stage, outer -> inner.

    Shell 0 is ``buffer_km`` / ``max_depth_km`` beyond the footprint. Each
    further shell shrinks the previous one geometrically -- ``buffer_shrink``
    laterally, ``depth_shrink`` in depth (only while ``taper`` is set,
    otherwise every shell keeps shell 0's depth) -- unless the matching
    ``*_override`` flag with a non-empty ``*_km_list`` (outer -> inner)
    replaces the geometric sequence outright. Shared by the mesh-build op
    expansion and the GUI's "current dx / dz" preview so both agree.
    """
    levels = max(int(stage.get("levels", 1)), 1)
    buf_list = [float(v) for v in (stage.get("buffer_km_list") or [])]
    if stage.get("buffer_override") and buf_list:
        buffers = buf_list
    else:
        buf0 = float(stage.get("buffer_km", 0.0))
        shrink = float(stage.get("buffer_shrink", 0.5))
        buffers = [buf0 * (shrink ** i) for i in range(levels)]

    n = len(buffers)
    dep_list = [float(v) for v in (stage.get("depth_km_list") or [])]
    if stage.get("depth_override") and dep_list:
        depths = [dep_list[min(i, len(dep_list) - 1)] for i in range(n)]
    else:
        dep0 = float(stage.get("max_depth_km", 0.0)) or (0.4 * model_depth_m / 1e3)
        dshrink = float(stage.get("depth_shrink", 0.55))
        taper = bool(stage.get("taper", True))
        depths = [dep0 * (dshrink ** i) if taper else dep0 for i in range(n)]
    return buffers, depths


def stage_op_count(stage: dict) -> int:
    """How many deal.II ops a stage expands to -- no geometry needed."""
    if not stage.get("enabled", True):
        return 0
    if stage.get("kind") == "interface":
        return 1
    if stage.get("kind") == "survey_area":
        return len(survey_area_shells(stage)[0])
    return max(int(stage.get("levels", 1)), 1)


def stages_to_ops(stages: list[dict], station_xy, *, delta_min_m: float,
                  model_depth_m: float, coast_fn=None, coast_exclude=(1,),
                  ocean_surface: bool = False):
    """Expand form stages into backend op dicts.

    Returns ``(ops, per_stage_counts)``.  ``per_stage_counts[i]`` is how many ops
    stage ``i`` produced (0 if disabled) so the UI can map a stage to a step
    range in the viewer.  Polygons are real shapely geometries -- no tokens.

    ``coast_fn(stage) -> [[north, east, 0], ...]`` supplies the (cleaned,
    sampled) coastline for ``coastline`` stages; without it such a stage
    expands to nothing.  ``coast_exclude`` are the material ids those ops skip
    (the real-ocean build passes ``()`` so the water layer above the seafloor
    is refined too) and ``ocean_surface`` makes ``interface`` ops also refine
    the air/water cells above the interface over ocean.
    """
    hull = None
    if any(st.get("enabled", True) and st.get("kind") == "survey_area"
           and st.get("shape", "ellipse") in ("convex_hull", "custom_polygon")
           for st in stages):
        import shapely.geometry as sgeom

        hull = sgeom.MultiPoint([tuple(p) for p in station_xy]).convex_hull
    xs = [p[0] for p in station_xy]
    ys = [p[1] for p in station_xy]
    survey_cx, survey_cy = 0.5 * (max(xs) + min(xs)), 0.5 * (max(ys) + min(ys))
    survey_half_x, survey_half_y = 0.5 * (max(xs) - min(xs)), 0.5 * (max(ys) - min(ys))
    ops: list[dict] = []
    counts: list[int] = []
    zfloor = 3.0 * delta_min_m
    for st in stages:
        if not st.get("enabled", True):
            counts.append(0)
            continue
        kind = st.get("kind")
        levels = max(int(st.get("levels", 1)), 1)
        pnorm = 1 if st.get("shape") == "rectangle" else 2
        taper = bool(st.get("taper", True))
        n0 = len(ops)

        if kind == "survey_area":
            shape = st.get("shape", "ellipse")
            buffers_km, depths_km = survey_area_shells(st, model_depth_m=model_depth_m)
            if shape in ("ellipse", "rectangle"):
                # gofem has no separate "ellipse" op -- this is refine_around_points
                # (mesh.py's "points" kind) with a single point at the survey centre:
                # pnorm=2 there IS a real ellipsoid test (coupled per-axis distance),
                # pnorm=1 a true independent-axis box ("degenerates into a
                # parallelepiped", per gofem's own is_within_ellipsoid docstring).
                # refine_within_polygon's pnorm>1 branch only *approximates* an
                # ellipse -- a circle sized off the footprint's bounding diagonal,
                # not fitted to the survey's actual x/y aspect ratio -- so it's the
                # wrong tool for this shape even though it also accepts pnorm.
                sa_pnorm = 1 if shape == "rectangle" else 2
                for buf_km, dep_km in zip(buffers_km, depths_km):
                    rx = survey_half_x + buf_km * 1e3
                    ry = survey_half_y + buf_km * 1e3
                    ops.append({"op": "points", "points": [[survey_cx, survey_cy, 0.0]],
                               "center": [survey_cx, survey_cy, 0.0],
                               "radii": [rx, ry, max(dep_km * 1e3, zfloor)],
                               "repeat": 1, "pnorm": sa_pnorm})
            else:   # convex_hull / custom_polygon -- need the exact boundary test
                if shape == "custom_polygon" and st.get("polygon_xy_km"):
                    base_poly = sgeom.Polygon([(float(x) * 1e3, float(y) * 1e3)
                                               for x, y in st["polygon_xy_km"]])
                else:
                    base_poly = hull
                poly_pnorm = min(max(float(st.get("polygon_pnorm", 1.0)), 1.0), 2.0)
                for buf_km, dep_km in zip(buffers_km, depths_km):
                    poly = base_poly.buffer(buf_km * 1e3)
                    ops.append({"op": "polygon", "polygon": poly, "repeat": 1,
                                "z_range": [0.0, max(dep_km * 1e3, zfloor)],
                                "pnorm": poly_pnorm})

        elif kind == "stations":
            radius = float(st.get("radius_km", 5.0)) * 1e3
            depth = float(st.get("depth_km", 20.0)) * 1e3
            for i in range(levels):
                d = depth * (0.6 ** i) if taper else depth
                ops.append({"op": "receivers", "material_ids": "all_earth",
                            "repeat": 1, "center": [0, 0, 0],
                            "radii": [radius, radius, max(d, zfloor)],
                            "pnorm": pnorm, "exclude": [1]})

        elif kind == "zone":
            cx = float(st.get("center_x_km", 0.0)) * 1e3
            cy = float(st.get("center_y_km", 0.0)) * 1e3
            hx = float(st.get("half_x_km", 50.0)) * 1e3
            hy = float(st.get("half_y_km", 50.0)) * 1e3
            depth = float(st.get("depth_km", 50.0)) * 1e3
            for i in range(levels):
                d = depth * (0.6 ** i) if taper else depth
                ops.append({"op": "points", "material_ids": "all_earth",
                            "repeat": 1, "center": [cx, cy, 0],
                            "radii": [hx, hy, max(d, zfloor)],
                            "points": [[cx, cy, 0]], "pnorm": pnorm, "exclude": [1]})

        elif kind == "interface":
            ops.append({"op": "interface", "material_ids": "all_earth",
                        "repeat": levels, "center": [0, 0, 0],
                        "radii": [1e7, 1e7], "pnorm": 1,
                        "ocean_surface": bool(ocean_surface)})

        elif kind == "coastline":
            pts = coast_fn(st) if coast_fn is not None else []
            radius = float(st.get("radius_km", 10.0)) * 1e3
            depth = float(st.get("depth_km", 20.0)) * 1e3
            # always `levels` ops (a no-op when there is no coast) so the step
            # numbering matches mesh_design.stage_op_count
            plist = [list(map(float, p)) for p in pts]
            for i in range(levels):
                d = depth * (0.6 ** i) if taper else depth
                # "coast", not "points": thousands of points need mesh.py's
                # KD-tree test, gofem.refine_around_points is cells x points
                ops.append({"op": "coast", "points": plist,
                            "repeat": 1, "center": [0, 0, 0],
                            "radii": [radius, radius, max(d, zfloor)],
                            "pnorm": pnorm, "exclude": list(coast_exclude)})

        counts.append(len(ops) - n0)
    return ops, counts


def suggest_survey_levels(dx_base_m: float, spacing_m: float,
                          cells_between: float) -> tuple[int, float]:
    """(# survey-area refinements, resulting cell size) to fit *cells_between*
    cells between stations *spacing_m* apart, starting from *dx_base_m*."""
    target = max(spacing_m / max(cells_between, 1e-6), 1.0)
    if dx_base_m <= target:
        return 0, dx_base_m
    n = int(math.ceil(math.log2(dx_base_m / target)))
    return n, dx_base_m / (2 ** n)


# --------------------------------------------------------------------------- #
def suggest_mesh(*, footprint_x_m: float, footprint_y_m: float,
                 spacing_min_m: float, spacing_median_m: float,
                 period_min_s: float, period_max_s: float,
                 rho_shallow: float, rho_deep: float, relief_m: float = 0.0,
                 c_response_max_km: float = 0.0,
                 rules: DesignRules | None = None) -> MeshSuggestion:
    r = rules or DesignRules()

    delta_min = skin_depth(period_min_s, rho_shallow)
    delta_max = skin_depth(period_max_s, rho_deep)
    span = max(footprint_x_m, footprint_y_m)

    pad = r.pad_skin_multiple * delta_max
    x_half_target = span / 2.0 + pad

    # ---- base cell: whatever spans the padded half-width in base_cell_count --
    if r.base_cell_override_m and r.base_cell_override_m > 0:
        dx_base = float(r.base_cell_override_m)
    else:
        dx_base = x_half_target / max(int(r.base_cell_count), 1)
    if r.round_grid:
        dx_base = round_nice(dx_base)

    # ---- vertical: base layers ~ half the base cell, growing geometrically --
    # HARD RULE: the base first layer is never thinner than 0.5*dx_base (no
    # matter what the C-response target says).  Fine near-surface layers come
    # from a later refinement stage, so they only appear around the survey.
    half_dx = 0.5 * dx_base
    dz_min = round_nice(half_dx) if r.round_grid else half_dx
    dz_min = max(dz_min, half_dx, r.dz_floor_m)     # never below half the base cell
    dz_base_1 = dz_min
    z_bot = r.depth_skin_multiple * delta_max
    mantle_capped = False
    if r.mantle_transition and z_bot > r.mantle_max_model_depth_km * 1e3:
        z_bot, mantle_capped = r.mantle_max_model_depth_km * 1e3, True
    dz_cap = r.max_dz_fraction * z_bot

    dz_earth: list[float] = []
    z, d = 0.0, dz_base_1
    while z < z_bot and len(dz_earth) < 200:
        d = min(d, dz_cap)
        dz_earth.append(float(round(d, 2)))
        z += d
        d *= r.vertical_growth
    snapped, prev = [], 0.0
    for v in dz_earth:
        nv = round_nice(v) if r.round_grid else v
        nv = max(nv, dz_min, prev)                # floor + keep non-decreasing
        snapped.append(nv)
        prev = nv
    dz_earth = snapped or [dz_base_1]
    dz_air = list(dz_earth)                       # match total earth thickness

    # ---- horizontal blocks ------------------------------------------------- #
    if r.constant_dx:
        n = max(math.ceil(x_half_target / dx_base), 1)
        dx_blocks = [float(dx_base)] * n
    else:
        n_core_half = math.ceil((span / 2.0) / dx_base) + r.core_buffer_cells
        dx_blocks = [float(dx_base)] * n_core_half
        cur, d = sum(dx_blocks), dx_base * r.horizontal_pad_growth
        while cur < x_half_target and len(dx_blocks) < 200:
            dv = round_nice(d) if r.round_grid else round(d, 2)
            dx_blocks.append(float(dv))
            cur += dv
            d *= r.horizontal_pad_growth
    x_half = sum(dx_blocks)

    # ---- survey-refinement suggestion: levels for the horizontal target -- #
    spacing = spacing_median_m if r.suggestion_use_median else spacing_min_m
    sug_levels, sug_cell = suggest_survey_levels(dx_base, spacing, r.cells_between_sites)

    model_depth = sum(dz_earth)
    meta = {
        "delta_min_m": delta_min, "delta_max_m": delta_max,
        "rho_shallow_ohm_m": rho_shallow, "rho_deep_ohm_m": rho_deep,
        "dx_base_m": dx_base, "dz_base_1_m": dz_base_1, "dz_min_floor_m": dz_min,
        "model_depth_m": model_depth, "x_half_m": x_half,
        "edge_pad_m": x_half - span / 2.0, "footprint_span_m": span,
        "suggested_survey_levels": float(sug_levels),
        "suggested_cell_m": sug_cell,
        "suggestion_spacing_m": spacing,
        "suggestion_target_m": spacing / max(r.cells_between_sites, 1e-6),
        "mantle_transition": 1.0 if r.mantle_transition else 0.0,
        "mantle_depth_capped": 1.0 if mantle_capped else 0.0,
        "mantle_depth_m": r.mantle_depth_km * 1e3,
    }
    stages = default_stages(r, model_depth_m=model_depth, survey_levels=sug_levels,
                            c_response_max_km=c_response_max_km)
    sugg = MeshSuggestion(dx_blocks, dz_air, dz_earth, stages, meta)
    sugg.diagnostics = evaluate(sugg, spacing_min_m=spacing_min_m,
                                delta_min=delta_min, delta_max=delta_max)
    return sugg


def collect_survey_scales(project) -> dict:
    """Pull footprint / spacing / period range / SSQ resistivity / relief.

    Works from the raw EDIs + the project's local frame; the DEM steps are
    optional (relief defaults to 0 when no projected DEM exists yet).
    """
    import numpy as np

    from . import edi_io, edi_z
    from . import projection as P

    surv = project.survey
    excl = set(surv.excluded)
    stations = [st for st in edi_io.read_edi_headers(surv.edi_dir)
                if st.station_id not in excl]
    if len(stations) < 2:
        raise ValueError("need at least two stations -- load EDIs on the Survey tab")

    lons = np.array([st.lon for st in stations])
    lats = np.array([st.lat for st in stations])
    kind = project.projection.proj_kind or "local"
    epsg = project.projection.epsg or edi_io.suggest_utm(
        float(lons.mean()), float(lats.mean()))[2]
    olon = project.projection.origin_lon or float(0.5 * (lons.min() + lons.max()))
    olat = project.projection.origin_lat or float(0.5 * (lats.min() + lats.max()))
    frame = P.make_local_frame(kind, epsg, olon, olat,
                               project.projection.local_proj,
                               project.projection.local_lat1,
                               project.projection.local_lat2)
    # mesh frame [north, east]: station_xy_m overlays the deal.II mesh in the
    # 3-D viewer, and footprint_x/y feed refinement sizing applied to that mesh.
    receivers, _ = P.project_stations(stations, frame, swap_xy=True)
    xy = np.asarray(receivers)[:, :2]              # col 0 = north (mesh x), col 1 = east (mesh y)
    footprint_x = float(xy[:, 0].max() - xy[:, 0].min())   # north-south span
    footprint_y = float(xy[:, 1].max() - xy[:, 1].min())   # east-west span
    spacing = P.station_spacing_stats(receivers)

    curve = edi_z.ssq_resistivity_curve(edi_z.read_dir(surv.edi_dir, excl))

    relief_m = 0.0
    pdem_path = project.projection.projected_dem_path
    if pdem_path and os.path.isfile(pdem_path):
        pdem = P.load_projected_dem(pdem_path, epsg)
        tc = pdem.topo_cart                        # [easting, northing, elev]
        inside = ((np.abs(tc[:, :, 0]) <= footprint_y / 2.0)   # easting  vs E-W span
                  & (np.abs(tc[:, :, 1]) <= footprint_x / 2.0))  # northing vs N-S span
        zc = tc[:, :, 2][inside] if inside.any() else tc[:, :, 2]
        relief_m = float(np.nanmax(zc) - np.nanmin(zc))

    return {
        "footprint_x_m": footprint_x, "footprint_y_m": footprint_y,
        "spacing_min_m": spacing["min_m"], "spacing_median_m": spacing["median_m"],
        "period_min_s": curve.period_min_s, "period_max_s": curve.period_max_s,
        "rho_shallow": curve.rho_shallow, "rho_deep": curve.rho_deep,
        "rho_geomean": curve.rho_geomean, "relief_m": relief_m,
        "delta_min_m": skin_depth(curve.period_min_s, curve.rho_shallow),
        "delta_max_m": skin_depth(curve.period_max_s, curve.rho_deep),
        "n_stations": len(stations), "epsg": epsg, "curve": curve,
        "station_xy_m": xy.tolist(),
    }


def evaluate(s: MeshSuggestion, *, spacing_min_m: float,
             delta_min: float, delta_max: float) -> list[Diagnostic]:
    depth = s.meta["model_depth_m"]
    pad = s.meta["edge_pad_m"]
    sug_cell = s.meta["suggested_cell_m"]
    capped = bool(s.meta.get("mantle_depth_capped"))
    return [
        Diagnostic("model depth" + (" (capped by mantle transition)" if capped else ""),
                   depth,
                   "~ 1000 km cap" if capped else f">= 3 x delta_max = {3*delta_max:,.0f} m",
                   True if capped else depth >= 3.0 * delta_max,
                   "bottom boundary far below the deepest sensitivity"),
        Diagnostic("edge padding", pad, f">= 3 x delta_max = {3*delta_max:,.0f} m",
                   pad >= 3.0 * delta_max,
                   "side boundaries do not contaminate the data"),
        Diagnostic("suggested near-survey cell", sug_cell,
                   f"target {s.meta['suggestion_target_m']:,.0f} m", True,
                   "from the survey-area refinement suggestion below"),
        Diagnostic("air / earth thickness balance",
                   sum(s.dz_air) - sum(s.dz_earth), "~ 0 m",
                   abs(sum(s.dz_air) - sum(s.dz_earth)) < 1.0,
                   "GoFEM Dirichlet2D expects a matched air column"),
        Diagnostic("base cells (pre-refinement)", float(s.n_base_cells),
                   "informational", True,
                   "grows with the refinement recipe; aim for a 250-300k active region"),
    ]
