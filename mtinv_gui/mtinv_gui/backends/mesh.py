"""Task 3 -- design the inversion mesh.

This is the step with the hardest dependencies: it needs the deal.II Python
bindings (``PyDealII``) and ``pyGoFEM`` (``gofem.mesh_utils`` etc.).  The whole
GUI is expected to run in one Linux/WSL environment where these are importable
(their locations are configured in Project > Settings and injected via
``config.apply_sys_paths()``).

:func:`run` consumes the JSON blob produced by :func:`mesh_spec_from_project`
and emits ``model.tria``, ``model.vtu``, ``model.materials`` and
``model.mask``.

Mesh recipe, in order:
  * ``dx`` / ``dy`` (mirrored) padding blocks, ``dz_air`` (reversed)
    + ``dz_earth`` -> full cell-size lists.
  * ``Triangulation('3D').generate_subdivided_steps_hyper_rectangle``.
  * initial material ids: z<0 -> air, else -> earth.
  * build padding ellipsoids / polygons + ``refine_dict`` with
    ops ``interface`` | ``points`` | ``polygon`` | ``receivers``.
  * execute refinements (``gofem.mesh_utils.refine_*``).
  * ``Topography(callback).fit_to(triangulation, z_top, z_bottom)``.
  * ``project_points_on_interface`` -> ``receivers_projected``.
  * thin-sheet ocean: per-cell conductance from bathymetry.
  * ``model.mask`` (free-domain = earth ids only).
  * write ``model.tria`` / ``model.vtu`` / ``model.materials``.
"""
from __future__ import annotations

import os
import threading
from typing import Any

#: file names written by :func:`run` (also referenced by the .prm writers).
MESH_TRIA = "model.tria"
MESH_VTU = "model.vtu"
MESH_MATERIALS = "model.materials"
MESH_MASK = "model.mask"
RECEIVERS_CSV = "receivers_projected.csv"
RECEIVERS_PARAVIEW = "receivers_paraview.csv"

def mesh_output_dir(project) -> str:
    """Absolute directory the mesh files are written to / read from.

    ``project.mesh.output_dir`` if the user set one (``~`` and relative paths
    resolved against the project working dir), else the working dir itself.
    Single source of truth -- the mesh build (step 5) and the GoFEM .prm
    writers (step 6) both call this so they never disagree.
    """
    base = os.path.abspath(project.workdir) if project.workdir else os.getcwd()
    chosen = (getattr(project.mesh, "output_dir", "") or "").strip()
    if not chosen:
        return base
    chosen = os.path.expanduser(chosen)
    return chosen if os.path.isabs(chosen) else os.path.normpath(os.path.join(base, chosen))


def mesh_spec_from_project(project) -> dict:
    """Everything a runner needs.  Refinement is carried as form *stages*
    (see :class:`mtinv_gui.backends.mesh_design.RefinementStage`); ``run`` turns
    them into deal.II ops once it knows the projected station footprint."""
    m = project.mesh
    dr = m.design_rules or {}
    return {
        "dx_blocks": m.dx_blocks,
        "dz_air": m.dz_air,
        "dz_earth": m.dz_earth,
        "background_resistivity_ohm_m": m.background_resistivity_ohm_m,
        "mantle_transition": bool(dr.get("mantle_transition", False)),
        "mantle_depth_m": float(dr.get("mantle_depth_km", 410.0)) * 1000.0,
        "mantle_resistivity_ohm_m": float(dr.get("mantle_resistivity_ohm_m", 20.0)),
        "use_topography": m.use_topography,
        "ocean_mode": m.ocean_mode,
        "ocean_resistivity_ohm_m": m.ocean_resistivity_ohm_m,
        "ocean_thickness_m": m.ocean_thickness_m,
        "output_format": getattr(m, "output_format", "gofem") or "gofem",
        "simpeg_vertical_ratio": float(getattr(m, "simpeg_vertical_ratio", 1.0) or 1.0),
        "sea_rule": getattr(m, "sea_rule", "connected") or "connected",
        "sea_gap_km": float(getattr(m, "sea_gap_km", 10.0)),
        "land_polygons_path": getattr(m, "land_polygons_path", ""),
        "sea_polygons_path": getattr(m, "sea_polygons_path", ""),
        "refinement_stages": [dict(s) for s in m.refinement_stages],
        "scales": dict(m.scales),
        "projected_dem_path": project.projection.projected_dem_path,
        "edi_dir": project.survey.edi_dir,
        "excluded": project.survey.excluded,
        "proj_kind": project.projection.proj_kind or "local",
        "local_proj": project.projection.local_proj,
        "local_lat1": project.projection.local_lat1,
        "local_lat2": project.projection.local_lat2,
        "epsg": project.projection.epsg,
        "origin_lon": project.projection.origin_lon,
        "origin_lat": project.projection.origin_lat,
    }


def initial_grid_summary(dx_blocks, dz_air, dz_earth) -> dict:
    """Cheap preview without deal.II: extent + cell counts + air/earth balance."""
    dx = list(dx_blocks) + list(dx_blocks)[::-1]
    dz = list(dz_air)[::-1] + list(dz_earth)
    ncells = len(dx) * len(dx) * len(dz)
    return {
        "extent_x_m": float(sum(dx)),
        "extent_y_m": float(sum(dx)),
        "extent_z_m": float(sum(dz)),
        "n_cells_initial": int(ncells),
        "air_thickness_m": float(sum(dz_air)),
        "earth_thickness_m": float(sum(dz_earth)),
        "air_earth_mismatch_m": float(sum(dz_earth) - sum(dz_air)),
        "smallest_dz_m": float(min(dz)),
        "largest_dz_m": float(max(dz)),
    }


# --------------------------------------------------------------------------- #
def check_stack() -> dict[str, str]:
    """Report whether PyDealII / gofem import (after applying configured paths).

    ``gofem.mtpy_utils`` also needs ``mtpy`` installed (step 5 dependency), so a
    failure there while the other gofem modules import fine just means mtpy is
    not in the env yet.
    """
    from .. import config

    added = config.apply_sys_paths()
    out: dict[str, str] = {
        "sys.path added": ", ".join(added) if added else "(none -- using installed packages)"
    }
    for name in ("PyDealII.Release", "gofem", "gofem.parameter_handler",
                 "gofem.mesh_utils", "gofem.mtpy_utils"):
        try:
            __import__(name)
            out[name] = "ok"
        except Exception as exc:  # noqa: BLE001
            out[name] = f"{type(exc).__name__}: {exc}"
    return out


def _resolve_material_ids(value, earth_ids: list[int]) -> list[int]:
    if value in ("all_earth", "earth", None):
        return list(earth_ids)
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _resolve_polygon(poly):
    """Return a shapely polygon for a refinement op.

    Stages already emit real shapely geometries; a list of ``[x, y]`` pairs is
    also accepted.
    """
    import shapely.geometry as sgeom

    if hasattr(poly, "exterior") or hasattr(poly, "geoms"):
        return poly
    return sgeom.Polygon(poly)


def _ensure_points_in_polygon_no_osmnx(mesh_utils_mod) -> None:
    """``gofem.mesh_utils.points_in_polygon`` (used by ``refine_within_polygon``
    / ``refine_at_polygon_boundary`` for the exact, pnorm=1 containment test)
    does a lazy ``import osmnx`` purely as a speed-up (quadrat-cutting + an
    r-tree index before falling back to exact geometry checks). ``environment.yml``
    doesn't install ``osmnx``, a large dependency for a speed-up only.

    Substitute a dependency-free equivalent using shapely's own vectorized
    point-in-polygon test when osmnx is missing; leave gofem's own
    implementation alone (matches upstream exactly) when it's available.
    """
    try:
        import osmnx  # noqa: F401
        return
    except ImportError:
        pass

    def _points_in_polygon(points_xy, polygon, quadrat_width=None):
        import numpy as np
        import shapely.vectorized

        xc = np.asarray(points_xy[0], dtype="float64")
        yc = np.asarray(points_xy[1], dtype="float64")
        mask = shapely.vectorized.contains(polygon.buffer(1e-6), xc, yc)

        class _Matches:
            index = np.flatnonzero(mask)

        return _Matches()

    mesh_utils_mod.points_in_polygon = _points_in_polygon


def _op_signature(op: dict):
    """Hashable-ish stand-in for a refinement op dict, so two ops can be
    compared for "did this actually change" without relying on shapely
    geometry equality or object identity."""
    out = {}
    for k, v in op.items():
        if hasattr(v, "wkt"):              # shapely geometry
            out[k] = v.wkt
        elif isinstance(v, (list, tuple)):
            out[k] = tuple(v)
        else:
            out[k] = v
    return tuple(sorted(out.items()))


def _preview_base_key(spec: dict):
    """Everything that affects the base grid / materials / receivers -- i.e.
    everything built *before* any refinement op runs.  If this is unchanged
    between two preview calls, the cached triangulation's un-refined state is
    still valid to build on."""
    return (
        tuple(spec.get("dx_blocks") or ()),
        tuple(spec.get("dz_air") or ()),
        tuple(spec.get("dz_earth") or ()),
        float(spec.get("background_resistivity_ohm_m", 0.0)),
        bool(spec.get("mantle_transition")),
        float(spec.get("mantle_depth_m", 410_000.0)),
        float(spec.get("mantle_resistivity_ohm_m", 20.0)),
        float(spec.get("ocean_resistivity_ohm_m", 1.0 / 3.2)),
        spec.get("ocean_mode", "none"),
        spec.get("edi_dir"), tuple(sorted(spec.get("excluded") or ())),
        spec.get("epsg"), float(spec.get("origin_lon", 0.0)), float(spec.get("origin_lat", 0.0)),
        # the DEM and the sea mask: the coastline and ocean-surface refinements read them
        spec.get("projected_dem_path"), spec.get("sea_rule"), spec.get("sea_gap_km"),
        spec.get("land_polygons_path"), spec.get("sea_polygons_path"),
    )


class _PreviewCache:
    """One-slot cache for the "Preview to step N" path.

    deal.II refinement here only ever subdivides cells (nothing sets a
    coarsen flag), so a previously-built triangulation can only ever be
    *extended* with more ops, never rewound.  When the requested step count
    grows and nothing before it changed, this lets ``run()`` pick up the
    cached triangulation and apply just the new ops instead of rebuilding
    everything from op 0 -- which matters because the priciest ops (fine
    refinement near stations) tend to be the last ones applied.  Any change
    to the base grid/materials, or to an *earlier* op, invalidates the whole
    cache -- there's no cheaper way to "unrefine" than starting over.
    """
    def __init__(self):
        self.base_key = None
        self.op_sigs: list = []
        self.n_applied = 0
        self.tria = None

    def reusable_for(self, base_key, op_sigs, stop) -> bool:
        return (self.tria is not None and self.base_key == base_key
                and stop >= self.n_applied
                and op_sigs[:self.n_applied] == self.op_sigs[:self.n_applied])

    def store(self, base_key, op_sigs, n_applied, tria):
        self.base_key = base_key
        self.op_sigs = op_sigs
        self.n_applied = n_applied
        self.tria = tria


_preview_cache = _PreviewCache()


# --------------------------------------------------------------------------- #
# cell-count checkpoints: ask before a build grows past them
# --------------------------------------------------------------------------- #
#: a build asks (through ``run(confirm=...)``) before going past each of these
CELL_COUNT_CHECKPOINTS = (500_000, 1_000_000)


class MeshBuildCancelled(Exception):
    """The user declined to go on past a cell-count checkpoint."""


class CellCountGuard:
    """Asks ``confirm(info) -> bool`` once per checkpoint the cell count reaches;
    raises :class:`MeshBuildCancelled` on a no.

    ``info``: ``n_cells`` (the count, or a lower bound when ``predicted``), ``n_now``
    (cells before the step), ``checkpoint`` (the largest one reached), ``step`` (what is
    about to run, or just ran), ``predicted``."""

    def __init__(self, confirm, checkpoints=CELL_COUNT_CHECKPOINTS, log=print):
        self.confirm = confirm
        self.pending = sorted(int(c) for c in checkpoints)
        self.log = log
        self.step = ""

    def check(self, n_cells: int, *, n_now: int, predicted: bool) -> None:
        reached = [c for c in self.pending if n_cells >= c]
        if not reached or self.confirm is None:
            return
        self.pending = [c for c in self.pending if c not in reached]
        info = dict(n_cells=int(n_cells), n_now=int(n_now), checkpoint=reached[-1],
                    step=self.step, predicted=predicted)
        if not self.confirm(info):
            if n_now == 0:
                self.log("build stopped by the user: the base grid would have %d cells" % n_cells)
            elif predicted:
                self.log("build stopped by the user at %d cells (%s would make at least %d)"
                         % (n_now, self.step or "the next step", n_cells))
            else:
                self.log("build stopped by the user: %s made %d cells" % (self.step, n_cells))
            raise MeshBuildCancelled(
                "stopped before going past %s cells" % f"{reached[-1]:,}")
        self.log("continuing past %s cells (user confirmed)" % f"{reached[-1]:,}")


class _GuardedTria:
    """A triangulation stand-in for the refinement helpers (pyGoFEM's only call
    ``dim``, ``active_cells`` and ``execute_coarsening_and_refinement``): before a
    refinement runs, it predicts the new count from the flags (8 children each, so
    +7; deal.II's smoothing can add ~10 % more, so it's a lower bound) and checks it;
    after, it checks the real count, catching what the prediction missed before the
    next step."""

    def __init__(self, tria, guard: CellCountGuard):
        self._tria = tria
        self._guard = guard

    def __getattr__(self, name):
        return getattr(self._tria, name)

    def execute_coarsening_and_refinement(self):
        t = self._tria
        n0 = t.n_active_cells()
        n_flag = sum(1 for c in t.active_cells() if c.refine_flag != "no_refinement")
        self._guard.check(n0 + 7 * n_flag, n_now=n0, predicted=True)
        t.execute_coarsening_and_refinement()
        self._guard.check(t.n_active_cells(), n_now=n0, predicted=False)
#: What each inversion code needs from the one mesh design (``spec["output_format"]``).
#: The GoFEM files are always written: the viewers, the Refine tab and the plotting
#: app read them.  ``ocean_modes``: the ocean models the code can take.  ``fit_topography``: deform the
#: cells to the DEM.  SimPEG's octree can't be deformed, but the fitted GoFEM mesh is what
#: tells it where to be fine (along the real surface), so it keeps the fit too: see
#: :mod:`.simpeg_mesh`.  A code that needs the unfitted mesh would set it False.
OUTPUT_FORMATS: dict[str, dict] = {
    "gofem": {"label": "GoFEM", "fit_topography": True,
              "ocean_modes": ("none", "thin_sheet", "real")},
    "femtic": {"label": "GoFEM + FEMTIC (DHEXA)", "fit_topography": True,
               "ocean_modes": ("none", "real")},
    "simpeg": {"label": "GoFEM + SimPEG (OcTree)", "fit_topography": True,
               "ocean_modes": ("none", "real")},
}


def output_format(spec: dict) -> tuple[str, dict]:
    """``(key, OUTPUT_FORMATS entry)`` for a spec; raises on a combination the code can't take."""
    key = (spec.get("output_format") or "gofem").lower()
    if key not in OUTPUT_FORMATS:
        raise ValueError(f"unknown mesh output format {key!r} (have: {', '.join(OUTPUT_FORMATS)})")
    fmt = OUTPUT_FORMATS[key]
    ocean = spec.get("ocean_mode") or "none"
    if ocean not in fmt["ocean_modes"]:
        raise ValueError(f"{fmt['label']} can't take the '{ocean}' ocean model "
                         f"(it takes: {', '.join(fmt['ocean_modes'])})")
    return key, fmt


_preview_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# Ocean / coastline helpers.  They only touch cells through center(), level(),
# material_id, measure(), faces(), neighbor() and refine_flag -- the subset of
# the PyDealII accessor this build is known to have.
# --------------------------------------------------------------------------- #
def _cell_arrays(tria):
    """Centre x/y/z, refinement level and material id of every active cell
    (same order as ``tria.active_cells()``)."""
    import numpy as np

    cx, cy, cz, lvl, mat = [], [], [], [], []
    for c in tria.active_cells():
        p = c.center()
        cx.append(p.x)
        cy.append(p.y)
        cz.append(p.z)
        lvl.append(c.level())
        mat.append(c.material_id)
    return (np.array(cx, dtype="float64"), np.array(cy, dtype="float64"),
            np.array(cz, dtype="float64"), np.array(lvl, dtype=int),
            np.array(mat, dtype=int))


def refine_coast(tria, points, radii, pnorm, exclude, repeat: int = 1) -> None:
    """Refine every cell within the ``radii`` ellipsoid (pnorm 2) / box (pnorm 1)
    of any coastline point.  Equivalent to ``refine_around_points`` but uses a
    KD-tree, since a coastline is thousands of points rather than a few
    stations (cells x points would take forever in pure Python)."""
    import numpy as np
    from scipy.spatial import cKDTree

    pts = np.asarray(points, dtype="float64").reshape(-1, 3)[:, :2]
    if not len(pts):
        return
    tree = cKDTree(pts)
    r_h, r_v = float(radii[0]), float(radii[2])
    for _ in range(max(int(repeat), 1)):
        cx, cy, cz, _lvl, mat = _cell_arrays(tria)
        xy = np.c_[cx, cy]
        if pnorm == 1:
            d, _i = tree.query(xy, p=np.inf)
            hit = (d <= r_h) & (np.abs(cz) <= r_v)
        else:
            d, _i = tree.query(xy)
            hit = (d / r_h) ** 2 + (cz / r_v) ** 2 <= 1.0
        hit &= ~np.isin(mat, list(exclude))
        for flag, c in zip(hit, tria.active_cells()):
            if flag:
                c.refine_flag = "isotropic"
        tria.execute_coarsening_and_refinement()


def refine_air_over_ocean(tria, topo_interp, air_id: int, earth_ids) -> None:
    """Real-ocean mode: refine the air cells sitting on the air/earth interface
    over ocean DEM points -- after the topography fit these are the water
    cells above the seafloor, so they need the same resolution as the earth
    cells ``refine_at_interface`` just split below them."""
    for cell in tria.active_cells():
        if cell.material_id != air_id:
            continue
        touches = False
        for n, face in enumerate(cell.faces()):
            if not face.at_boundary() and cell.neighbor(n).material_id in earth_ids:
                touches = True
                break
        if not touches:
            continue
        p = cell.center()
        if float(topo_interp([[p.x, p.y]])[0]) < 0.0:
            cell.refine_flag = "isotropic"
    tria.execute_coarsening_and_refinement()


_refine_coast = refine_coast                    # older names
_refine_air_over_ocean = refine_air_over_ocean


def assign_ocean(tria, mode: str, topo_interp, *, air_id: int, earth_id: int,
                 ocean_id: int, dz_earth, sheet_m: float, sigma_sea: float,
                 sigma_bg: float, materials: dict, log=print,
                 expected_volume: float = 0.0, inland=None) -> dict:
    """Give ocean cells their material ids (after the topography fit).

    ``mode`` ``"real"``: air cells whose centre ended up below sea level become
    material *ocean_id* (``sigma_sea``).  ``"thin_sheet"``: cells overlapping the
    top ``sheet_m`` over ocean DEM points get one of ~100 conductivity classes
    that preserve the water-column conductance (see :mod:`.ocean`).
    ``materials`` is updated in place.  Returns a stats dict.

    ``topo_interp`` must read land below sea level as land (``MeshDem.ocean_elevation``);
    ``inland([[north, east], ...])`` says where that land is, so the air above it in a
    real-ocean build isn't turned into water.
    """
    import numpy as np

    from . import ocean as O

    cx, cy, cz, lvl, mat = _cell_arrays(tria)
    stats = {"mode": mode, "n_ocean_cells": 0}

    if mode == "real":
        sel = O.classify_real_ocean(cz, mat == air_id)
        if inland is not None and sel.any():
            sel[sel] &= ~np.asarray(inland(np.c_[cx[sel], cy[sel]]), dtype=bool)
        vol = 0.0
        for flag, c in zip(sel, tria.active_cells()):
            if flag:
                c.material_id = ocean_id
                vol += float(c.measure())
        stats["n_ocean_cells"] = int(sel.sum())
        if sel.any():
            materials[ocean_id] = ("sea", float(sigma_sea))
        stats["water_volume_m3"] = vol
        stats["expected_volume_m3"] = float(expected_volume)
        log("ocean (real): %d water cells" % stats["n_ocean_cells"])
        if expected_volume > 0:
            ratio = vol / expected_volume
            stats["volume_ratio"] = ratio
            log("  resolved water volume = %.0f%% of the DEM's" % (100.0 * ratio))
            if ratio < 0.7:
                log("  WARNING: the seafloor is only partly resolved -- water shallower than "
                    "the cells above the seabed is lost.  Add an 'Interface' (and 'Along "
                    "Coastline') refinement stage, or use the thin-sheet ocean.")
        return stats

    # thin sheet
    depth = -np.asarray(topo_interp(np.c_[cx, cy]), dtype="float64")
    sigma = O.thin_sheet_conductivity(cz, lvl, depth, dz_earth, sheet_m, sigma_sea, sigma_bg)
    sigma = np.where(mat == air_id, sigma_bg, sigma)
    cls, csig = O.bin_conductivities(sigma, sigma_bg)
    ids = O.SHEET_FIRST_ID
    for flag, k, c in zip(cls >= 0, cls, tria.active_cells()):
        if flag:
            c.material_id = ids + int(k)
    for k, s in enumerate(csig):
        materials[ids + k] = ("sheet%d" % (k + 1), float(s))
    stats["n_ocean_cells"] = int((cls >= 0).sum())
    stats["n_classes"] = int(len(csig))
    log("ocean (thin sheet, %.0f m): %d cells in %d conductivity classes"
        % (sheet_m, stats["n_ocean_cells"], len(csig)))
    if len(csig):
        log("  sheet conductivity %.3g .. %.3g S/m (background %.3g)"
            % (csig.min(), csig.max(), sigma_bg))
    elif (depth > 0).any():
        log("  WARNING: no cell picked up the sheet -- raise the sheet thickness "
            "or refine the near-surface cells")
    return stats


# --------------------------------------------------------------------------- #
# The build, one step per function.  ``run`` calls them in order for the GUI;
# the mesh notebook calls them one cell at a time, with the refinements in
# between written as plain pyGoFEM calls.
# --------------------------------------------------------------------------- #
#: material ids the setup writes: air, earth (inverted), real-ocean water, mantle
AIR_ID, EARTH_ID, OCEAN_ID, MANTLE_ID = 1, 2, 3, 4


def _pygofem(name: str):
    """Fetch a pyGoFEM helper, tolerating which submodule it lives in."""
    import importlib

    for modname in ("gofem.mesh_utils", "gofem", "gofem.utils", "gofem.grid"):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        if hasattr(mod, name):
            return getattr(mod, name)
    raise ImportError(
        f"pyGoFEM: could not locate '{name}' in gofem.mesh_utils; "
        f"adjust mtinv_gui/backends/mesh.py "
        f"if your pyGoFEM exposes it elsewhere.")


def prepare_pygofem():
    """Make deal.II and pyGoFEM importable (dev paths from the settings file, if
    set) and replace pyGoFEM's optional ``osmnx`` speed-up when it's missing.
    Call once before using ``gofem.mesh_utils``; returns that module."""
    from .. import config

    config.apply_sys_paths()
    import gofem.mesh_utils  # noqa: F401  -- early, clear failure if pyGoFEM is missing

    _ensure_points_in_polygon_no_osmnx(gofem.mesh_utils)
    return gofem.mesh_utils


def read_mesh_dem(dem_path: str, *, sea_rule: str = "connected", gap_km: float = 10.0,
                  land_polygons_path: str = "", sea_polygons_path: str = "", frame=None, log=print):
    """The projected DEM as the build uses it (:class:`mesh_dem.MeshDem`): the detailed
    grid, the regional one if the file has it, and where the sea is.  The polygon files
    (see ``projection.read_polygons``) need the model *frame*."""
    from . import mesh_dem
    from . import projection as P

    polys = {}
    for key, path in (("land_polygons", land_polygons_path), ("sea_polygons", sea_polygons_path)):
        if path:
            if frame is None:
                raise ValueError("the sea / land polygons need the model frame")
            polys[key] = P.read_polygons(path, frame)
    dem = mesh_dem.MeshDem.load(dem_path, sea_rule=sea_rule, gap_km=gap_km, **polys)
    if len(dem.grids) > 1:
        log("DEM: detailed grid %d x %d plus a regional grid %d x %d"
            % (dem.grids[0].shape[1], dem.grids[0].shape[0], dem.grids[1].shape[1], dem.grids[1].shape[0]))
    return dem


def warn_dem_shortfall(dem, bounds, ocean_mode: str = "none", log=print) -> dict[str, float]:
    """Log a warning when the DEM doesn't cover the mesh *bounds*; returns the shortfall.
    Past the DEM's edge the mesh repeats the edge values: a plateau on land, and with an
    ocean model the whole padding can turn into deep sea."""
    from . import mesh_dem

    short = dem.shortfall(bounds)
    if any(v > 0 for v in short.values()):
        log("WARNING: the projected DEM stops short of the mesh by %s. Past its edge the mesh "
            "gets the DEM's edge values%s. Use a bigger (regional) DEM on the Topography tab."
            % (mesh_dem.shortfall_text(short),
               ": with an ocean model, the sea floor at the edge fills the padding"
               if ocean_mode != "none" else ""))
    return short


def cell_size_lists(dx_blocks, dz_air, dz_earth):
    """``(dx, dy, dz)`` for the whole grid: ``dx_blocks`` runs from the centre out
    and is mirrored, ``dz_air`` runs from the surface up and is reversed."""
    dx = list(dx_blocks) + list(dx_blocks)[::-1]
    return dx, list(dx), list(dz_air)[::-1] + list(dz_earth)


def model_bounds(dx, dy) -> tuple[float, float, float, float]:
    """``(east_min, east_max, north_min, north_max)`` of a grid centred on (0, 0)."""
    return (-sum(dy) / 2.0, sum(dy) / 2.0, -sum(dx) / 2.0, sum(dx) / 2.0)


def base_grid(dx, dy, dz, log=print):
    """The coarse starting grid, centred on (0, 0, 0): x = north, y = east, z down."""
    import numpy as np
    import PyDealII.Release as dealii

    mesh_size = np.array([sum(dx), sum(dy), sum(dz)], dtype="float64")
    t = dealii.Triangulation("3D")
    t.generate_subdivided_steps_hyper_rectangle(
        [list(dx), list(dy), list(dz)],
        dealii.Point((-mesh_size / 2.0).tolist()),
        dealii.Point((mesh_size / 2.0).tolist()),
        False)
    log("base grid: %d cells (%.0f x %.0f x %.0f km)"
        % (t.n_active_cells(), mesh_size[0] / 1e3, mesh_size[1] / 1e3, mesh_size[2] / 1e3))
    return t


def initial_materials(tria, background_resistivity_ohm_m: float, *,
                      mantle_depth_m: float | None = None,
                      mantle_resistivity_ohm_m: float = 20.0, log=print) -> dict:
    """Air above z = 0, earth below (and a fixed mantle below ``mantle_depth_m`` if
    given).  Returns ``{id: (name, conductivity S/m)}``; the ocean adds to it later."""
    for cell in tria.active_cells():
        z = cell.center().z
        if z < 0:
            cell.material_id = AIR_ID
        elif mantle_depth_m is not None and z > mantle_depth_m:
            cell.material_id = MANTLE_ID
        else:
            cell.material_id = EARTH_ID
    mats: dict[int, tuple[str, float]] = {
        AIR_ID: ("air", 1e-8),
        EARTH_ID: ("ground", 1.0 / float(background_resistivity_ohm_m)),
    }
    if mantle_depth_m is not None:
        mats[MANTLE_ID] = ("mantle_transition", 1.0 / float(mantle_resistivity_ohm_m))
        log("mantle transition: %g ohm-m below %.0f km (fixed, not inverted)"
            % (mantle_resistivity_ohm_m, mantle_depth_m / 1e3))
    return mats


def fit_topography(tria, dem, *, land: bool = True, bathymetry: bool = False,
                   log=print) -> dict:
    """Move the air/earth interface to the DEM (a :class:`mesh_dem.MeshDem`): land
    elevations if ``land`` (land below sea level too, at its own elevation), the seafloor
    if ``bathymetry``; the sea surface stays at z = 0 otherwise.

    Only for a mesh that hasn't been fitted yet (see ``refine.refit_topography``).
    Returns ``{"relief_m", "band_m", "degenerate"}``.
    """
    import numpy as np

    Topography = _pygofem("Topography")

    # Callback convention (deal.II z is +down): negative raises the surface (land,
    # elev > 0), positive pushes the interface down to the seafloor (sea branch of
    # gofem's Topography).
    def _topo_cb(p):
        elev, inland = dem.point(p[0], p[1])
        if elev > 0:
            return -elev if land else 0.0
        if elev < 0:
            if inland:
                return -elev if land else 0.0       # land below sea level: down to it
            if bathymetry:
                return -elev
        return 0.0

    if len(dem.grids) == 1:
        elev_all = dem.grids[0][:, :, 2]
    else:
        elev_all = np.concatenate([g[:, :, 2].ravel() for g in dem.grids])
    relief = float(np.nanmax(np.abs(elev_all))) if bathymetry \
        else max(float(np.nanmax(elev_all)), 0.0)
    band = max(3.0 * relief, 2000.0)
    topo = Topography(_topo_cb, dim=3, center=[0, 0, 0],
                      radii=[1e7, 1e7, max(10.0 * relief, 5e4)])
    # gofem's sea branch interpolates between z_top_sea and z_0_sea (= 0);
    # its default z_top_sea = 0 would divide by zero.
    topo.z_top_sea = -band
    topo.fit_to(tria, z_top=-band, z_bottom=band)
    degenerate = sum(1 for c in tria.active_cells() if c.measure() <= 0)
    if degenerate:
        log("WARNING: %d degenerate cells after topography fit "
            "-- reduce relief influence or coarsen near-surface cells" % degenerate)
    else:
        log("topography fitted (relief %.0f m, band +/-%.0f m%s)"
            % (relief, band, ", with bathymetry" if bathymetry else ""))
    return {"relief_m": relief, "band_m": band, "degenerate": degenerate}


def add_ocean(tria, ocean_mode: str, dem, *, bounds, dz_earth,
              materials: dict, background_resistivity_ohm_m: float,
              ocean_resistivity_ohm_m: float = 1.0 / 3.2, sheet_m: float = 1500.0,
              log=print) -> dict:
    """Give the ocean cells their materials, after the topography fit.  ``ocean_mode``
    is ``"thin_sheet"`` or ``"real"`` (see :func:`assign_ocean`); *dem* a
    :class:`mesh_dem.MeshDem`; ``bounds`` is :func:`model_bounds`.  ``materials`` is
    updated in place.  Returns the stats (empty when the DEM has no ocean inside the model)."""
    ds = dem.ocean_stats(bounds)
    if ds.get("inland_km2"):
        log("DEM: %.0f km² of land below sea level inside the model (kept as land)" % ds["inland_km2"])
    if not ds["has_ocean"]:
        log("ocean mode '%s' selected but the DEM has no ocean inside the model "
            "-- nothing to do" % ocean_mode)
        return {}
    log("DEM ocean: %.0f%% of the footprint, up to %.0f m deep"
        % (100 * ds["ocean_frac"], ds["max_depth_m"]))
    return assign_ocean(
        tria, ocean_mode, dem.ocean_elevation, air_id=AIR_ID, earth_id=EARTH_ID,
        ocean_id=OCEAN_ID, dz_earth=dz_earth, sheet_m=float(sheet_m),
        sigma_sea=1.0 / float(ocean_resistivity_ohm_m),
        sigma_bg=1.0 / float(background_resistivity_ohm_m), materials=materials,
        log=log, expected_volume=dem.expected_water_volume(bounds),
        inland=dem.inland if dem.has_inland else None)


def project_receivers(tria, receivers, *, seafloor: bool = False, names=None, log=print):
    """Stations ``[[north, east, z], ...]`` moved onto the air/earth interface (and,
    with ``seafloor``, offshore ones down onto the water/earth interface).

    A station on a vertical cell face (or on a line GoFEM's receiver refinement would
    add) is first moved about 2 m into its cell: see ``refine.off_cell_faces``."""
    import numpy as np
    import PyDealII.Release as dealii

    from . import refine

    project_points_on_interface = _pygofem("project_points_on_interface")
    mapping = dealii.MappingQGeneric(dim=3, spacedim=3, degree=1)

    def project(points):
        out = project_points_on_interface(
            triangulation=tria, points=points,
            material_top=AIR_ID, material_bottom=EARTH_ID, mapping=mapping)
        if seafloor:
            out = project_points_on_interface(
                triangulation=tria, points=out,
                material_top=OCEAN_ID, material_bottom=EARTH_ID, mapping=mapping)
        return out

    out = project(receivers)
    # each station's cell, at the depth it is written at (0.1 m below the interface)
    footprints = np.full((len(out), 4), np.nan)
    for k, (x, y, z) in enumerate(np.asarray(out, dtype="float64")):
        try:
            cell = tria.find_active_cell_around_point(dealii.Point([x, y, z + 0.1]), mapping)
        except Exception:  # noqa: BLE001 - off the mesh: left to the receiver checks
            continue
        vs = [cell.get_vertex(i) for i in range(8)]
        footprints[k] = (min(v.x for v in vs), max(v.x for v in vs),
                         min(v.y for v in vs), max(v.y for v in vs))
    ok = np.isfinite(footprints).all(axis=1)
    xy, moved = refine.off_cell_faces(np.asarray(out, dtype="float64")[ok, :2], footprints[ok])
    if not moved:
        return out
    index = np.flatnonzero(ok)
    shifted = np.array(receivers, dtype="float64")
    shifted[index, :2] = xy
    for k, dn, de in moved:
        label = names[index[k]] if names is not None else f"station {index[k]}"
        log(f"{label} was on a cell face (GoFEM may not find it): moved {dn:+.1f} m north, "
            f"{de:+.1f} m east")
    return project(shifted.tolist() if isinstance(receivers, list) else shifted)


def write_mask(tria, path: str, free_ids=(EARTH_ID,)) -> str:
    """``model.mask``: ``level index 1`` for cells the inversion may change, ``0`` else."""
    free_ids = set(free_ids)
    with open(path, "w") as fh:
        for cell in tria.active_cells():
            fh.write("%i %i %i\n" % (cell.level(), cell.index(),
                                     1 if cell.material_id in free_ids else 0))
    return path


def write_materials(materials: dict, path: str, permittivity: float = 0.0) -> str:
    """``model.materials``: ``id name conductivity permittivity`` per material."""
    with open(path, "w") as fh:
        fh.write("# id name conductivity permitivitty\n")
        fh.write("iso\n")
        for mid, (name, sigma) in materials.items():
            fh.write("%d %s %0.6e %0.6e\n" % (mid, name, sigma, permittivity))
    return path


def write_receivers(names, receivers_projected, out_dir: str) -> tuple[str, str]:
    """``receivers_projected.csv`` (for GoFEM; 0.1 m below the surface) and
    ``receivers_paraview.csv`` (x, y, z only)."""
    import numpy as np

    rec_path = os.path.join(out_dir, RECEIVERS_CSV)
    with open(rec_path, "w") as fh:
        for name, rec in zip(names, receivers_projected):
            fh.write("Dipole %s 1 %0.6e %0.6e %0.6e\n"
                     % (name, rec[0], rec[1], rec[2] + 0.1))
    pv_path = os.path.join(out_dir, RECEIVERS_PARAVIEW)
    np.savetxt(pv_path, np.asarray(receivers_projected), delimiter=",", fmt="%s")
    return rec_path, pv_path


def write_mesh_files(tria, materials: dict, names, receivers_projected, out_dir: str,
                     free_ids=(EARTH_ID,), log=print) -> dict:
    """Everything GoFEM needs from the mesh step: ``model.tria`` / ``.vtu`` /
    ``.materials`` / ``.mask`` and the receiver files.  Returns their paths."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {"mask": write_mask(tria, os.path.join(out_dir, MESH_MASK), free_ids)}
    paths["vtu"] = os.path.join(out_dir, MESH_VTU)
    paths["tria"] = os.path.join(out_dir, MESH_TRIA)
    tria.write(paths["vtu"], "vtu")
    tria.save(paths["tria"])
    paths["materials"] = write_materials(materials, os.path.join(out_dir, MESH_MATERIALS))
    paths["receivers"], paths["receivers_paraview"] = write_receivers(
        names, receivers_projected, out_dir)
    log("wrote %s, %s, %s, %s, %s"
        % (MESH_TRIA, MESH_VTU, MESH_MATERIALS, MESH_MASK, RECEIVERS_CSV))
    return paths


def _apply_ops(tria, all_ops, start_idx, stop, *, guard, earth_ids, air_id, dem, receivers,
               log=print) -> None:
    """Refinement ops ``start_idx`` .. ``stop - 1`` of :func:`run`, each through the
    cell-count guard."""
    import numpy as np

    refine_at_interface = _pygofem("refine_at_interface")
    refine_around_points = _pygofem("refine_around_points")
    refine_within_polygon = _pygofem("refine_within_polygon")
    real_tria = tria
    tria = _GuardedTria(real_tria, guard)
    for i, op in enumerate(all_ops[start_idx:stop], start=start_idx):
        kind = op.get("op")
        rep = int(op.get("repeat", 1))
        pnorm = op.get("pnorm", 2)
        before = real_tria.n_active_cells()
        guard.step = "refinement #%d (%s)" % (i, kind)
        if kind == "interface":
            ids = _resolve_material_ids(op.get("material_ids"), earth_ids)
            if op.get("ocean_surface") and dem is not None:
                for _r in range(rep):       # earth below, then the water cells above
                    refine_at_interface(
                        tria, material_ids=ids, repeat=1, center=op.get("center", [0, 0, 0]),
                        radii=op["radii"], pnorm=pnorm)
                    refine_air_over_ocean(tria, dem.ocean_elevation, air_id, earth_ids)
            else:
                refine_at_interface(
                    tria, material_ids=ids, repeat=rep,
                    center=op.get("center", [0, 0, 0]), radii=op["radii"], pnorm=pnorm)
        elif kind == "coast":
            refine_coast(tria, op.get("points", []), op["radii"], pnorm,
                         op.get("exclude", [air_id]), rep)
        elif kind == "points":
            refine_around_points(
                tria, points=op.get("points", [[0, 0, 0]]),
                center=op.get("center", [0, 0, 0]), radii=op["radii"], repeat=rep,
                exclude_materials=op.get("exclude", [air_id]), pnorm=pnorm)
        elif kind == "receivers":
            idx = op.get("rec_idx")
            pts = (np.asarray(receivers)[idx, :].tolist() if idx else receivers)
            refine_around_points(
                tria, points=pts, center=op.get("center", [0, 0, 0]),
                radii=op["radii"], repeat=rep,
                exclude_materials=op.get("exclude", [air_id]), pnorm=pnorm)
        elif kind == "polygon":
            refine_within_polygon(
                tria, polygon=_resolve_polygon(op["polygon"]),
                repeat=rep, z_range=op["z_range"], pnorm=pnorm)
        else:
            raise ValueError(f"refinement #{i}: unknown op {kind!r}")
        log("  refine #%d (%s): +%d -> %d cells" % (
            i, kind, tria.n_active_cells() - before, tria.n_active_cells()))


def run(spec: dict, out_dir: str, log=print,
        stop_after_refinement: int | None = None, confirm=None) -> dict:
    """Build the inversion mesh in-process (needs PyDealII + gofem).

    Includes the ocean
    (``spec["ocean_mode"]``: none / thin_sheet / real) and coastline-refinement
    additions -- see :mod:`mtinv_gui.backends.ocean`.  Writes ``model.tria`` / ``.vtu`` / ``.materials``
    / ``model.mask`` and ``receivers_projected.csv`` into *out_dir*.

    ``stop_after_refinement``: when set, stop after applying that many
    refinement ops, write ``model_step<N>.vtu`` + return section clouds for the
    viewer, and skip topography / receiver projection / file writing.

    ``confirm``: called as ``confirm(info) -> bool`` when the mesh is about to reach
    (or has just reached) each of :data:`CELL_COUNT_CHECKPOINTS`; False stops the build
    with :class:`MeshBuildCancelled` before anything is written (see
    :class:`CellCountGuard`).  None = no checks.
    """
    import numpy as np

    fmt_key, fmt = output_format(spec)
    prepare_pygofem()

    from . import edi_io
    from . import projection as P

    os.makedirs(out_dir, exist_ok=True)
    air_id = AIR_ID
    bg_res = float(spec["background_resistivity_ohm_m"])
    earth_ids = [EARTH_ID]
    sea_res = float(spec.get("ocean_resistivity_ohm_m") or 1.0 / 3.2)

    # -- projected DEM (topography, bathymetry, coastline all read it) ---- #
    ocean_mode = spec.get("ocean_mode") or "none"
    dem_path = spec.get("projected_dem_path") or ""
    dem = None
    if dem_path and os.path.isfile(dem_path):
        frame = P.frame_from_projection(spec) \
            if (spec.get("land_polygons_path") or spec.get("sea_polygons_path")) else None
        dem = read_mesh_dem(dem_path, sea_rule=spec.get("sea_rule") or "connected",
                            gap_km=float(spec.get("sea_gap_km", 10.0)),
                            land_polygons_path=spec.get("land_polygons_path") or "",
                            sea_polygons_path=spec.get("sea_polygons_path") or "",
                            frame=frame, log=log)
    elif ocean_mode != "none":
        log("WARNING: ocean mode '%s' needs the projected DEM -- building without an ocean"
            % ocean_mode)
        ocean_mode = "none"
    use_topo = bool(spec.get("use_topography")) and dem is not None

    # -- 1. cell-size lists --------------------------------- #
    dx, dy, dz = cell_size_lists(spec["dx_blocks"], spec["dz_air"], spec["dz_earth"])
    if not (dx and dz):
        raise ValueError("empty dx/dz -- run 'Recompute from data' on the Mesh tab first")
    if dem is not None and (use_topo or ocean_mode != "none"):
        warn_dem_shortfall(dem, model_bounds(dx, dy), ocean_mode, log=log)

    mantle_on = bool(spec.get("mantle_transition"))
    mantle_z = float(spec.get("mantle_depth_m", 410_000.0))
    mantle_res = float(spec.get("mantle_resistivity_ohm_m", 20.0))

    guard = CellCountGuard(confirm, CELL_COUNT_CHECKPOINTS, log=log)

    def _build_base():
        # -- 2. base triangulation, 3. initial materials ---- #
        guard.step = "the base grid"
        n_base = len(dx) * len(dy) * len(dz)
        guard.check(n_base, n_now=0, predicted=False)
        t = base_grid(dx, dy, dz, log=log)
        mats = initial_materials(t, bg_res, mantle_depth_m=mantle_z if mantle_on else None,
                                 mantle_resistivity_ohm_m=mantle_res, log=log)
        return t, mats

    # -- 4. receivers in the local frame --------------------------- #
    if not (spec.get("origin_lon") or spec.get("origin_lat")):
        raise ValueError("no local frame -- set the projection / origin on the Survey tab")
    stations = [s for s in edi_io.read_edi_headers(spec["edi_dir"])
                if s.station_id not in set(spec.get("excluded", []))]
    frame = P.frame_from_projection(spec)
    log("model frame: " + P.frame_label(frame))
    # swap_xy=True: the deal.II grid is built [dx, dy, dz] with axis 0 = North
    # (GoFEM convention), so receivers, every refinement op derived from them,
    # the receiver projection and receivers_projected.csv are all in
    # [north, east, z].  MeshDem looks points up as (north, east) for
    # the same reason -- keep the two in step.
    receivers, receiver_names = P.project_stations(stations, frame, swap_xy=True)
    log("receivers: %d stations projected into the local (mesh) frame" % len(receivers))

    # -- 5. refinements: expand form stages -> deal.II ops -------- #
    from . import mesh_design as _md

    stages = spec.get("refinement_stages")
    if stages is None:                       # legacy projects
        all_ops = list(spec.get("refinements", []))
    else:
        sc = spec.get("scales") or {}
        rxy0 = np.asarray(receivers, dtype="float64")[:, :2]
        d_min = sc.get("delta_min_m") or 500.0
        depth = float(sum(spec["dz_earth"]))

        def _coast_fn(stage):
            """Cleaned + sampled coastline for one 'Along Coastline' stage, in
            the mesh frame.  The DEM contour is in (east, north); the box is the
            model footprint grown by the stage radius."""
            if dem is None:
                log("WARNING: coastline refinement needs the projected DEM -- skipped")
                return []
            m = float(stage.get("radius_km", 10.0)) * 1e3
            hx, hy = sum(dx) / 2.0 + m, sum(dy) / 2.0 + m          # north, east
            pts, info = dem.coast_points(stage, (-hy, hy, -hx, hx))
            log("coastline: %d lines (%d dropped), %d -> %d vertices after simplifying, "
                "%.0f km long -> %d refinement points every %.1f km%s"
                % (info["n_lines"], info["n_lines_raw"] - info["n_lines"],
                   info["n_points_raw"], info["n_points"], info["length_m"] / 1e3,
                   len(pts), info["spacing_m"] / 1e3,
                   " (spacing widened to cap the point count)"
                   if info["spacing_widened"] else ""))
            return pts.tolist()

        all_ops, _counts = _md.stages_to_ops(
            stages, rxy0.tolist(), delta_min_m=d_min, model_depth_m=depth,
            coast_fn=_coast_fn,
            # real ocean: the water layer above the seafloor is still "air" at this
            # point, so the coastline must not skip it
            coast_exclude=() if ocean_mode == "real" else (air_id,),
            ocean_surface=(ocean_mode == "real"))
    n_total = len(all_ops)
    stop = n_total if stop_after_refinement is None \
        else max(0, min(int(stop_after_refinement), n_total))

    preview = stop_after_refinement is not None
    start_idx = 0
    materials = None
    if preview:
        base_key = _preview_base_key(spec)
        op_sigs = [_op_signature(op) for op in all_ops[:stop]]
        with _preview_lock:
            if _preview_cache.reusable_for(base_key, op_sigs, stop):
                tria = _preview_cache.tria
                start_idx = _preview_cache.n_applied
                log("reusing cached mesh through step %d (%d cells) -- applying steps %d-%d only"
                    % (start_idx, tria.n_active_cells(), start_idx + 1, stop)
                    if start_idx < stop else
                    "reusing cached mesh -- nothing new to apply")
            else:
                tria, materials = _build_base()
    else:
        tria, materials = _build_base()

    try:
        _apply_ops(tria, all_ops, start_idx, stop, guard=guard, earth_ids=earth_ids,
                   air_id=air_id, dem=dem, receivers=receivers, log=log)
    except MeshBuildCancelled:
        if preview:                  # the cached mesh may be part-refined: start over next time
            with _preview_lock:
                _preview_cache.__init__()
        raise
    log("after refinement: %d cells, smallest %.0f m, largest %.0f m" % (
        tria.n_active_cells(), tria.minimal_cell_diameter(), tria.maximal_cell_diameter()))


    if preview:
        with _preview_lock:
            _preview_cache.store(base_key, op_sigs, stop, tria)
        os.makedirs(out_dir, exist_ok=True)
        vtu = os.path.join(out_dir, f"model_step{stop}.vtu")
        tria.write(vtu, "vtu")
        rxy = np.asarray(receivers, dtype="float64")[:, :2]
        return {
            "step": stop, "n_total_steps": n_total,
            "n_cells": tria.n_active_cells(),
            "smallest_cell_m": tria.minimal_cell_diameter(),
            "largest_cell_m": tria.maximal_cell_diameter(),
            "vtu": vtu,
            "model_half_x": float(sum(dx) / 2.0),
            "model_half_y": float(sum(dy) / 2.0),
            "model_z_bottom": float(sum(dz) / 2.0),
            "station_x_min": float(rxy[:, 0].min()), "station_x_max": float(rxy[:, 0].max()),
            "station_y_min": float(rxy[:, 1].min()), "station_y_max": float(rxy[:, 1].max()),
        }

    # -- 6. fit to topography --------------------------- #
    # Only "real" ocean mode feeds bathymetry in; "none" and "thin_sheet" leave
    # the sea surface flat at z = 0.
    if not fmt["fit_topography"]:
        log("no topography fit for %s" % fmt["label"])
    elif dem is not None and (use_topo or ocean_mode == "real"):
        fit_topography(tria, dem, land=use_topo,
                       bathymetry=(ocean_mode == "real"), log=log)
    else:
        log("no topography fit -- flat air/earth interface at z = 0")

    # -- 6b. ocean materials ----------------------------------------- #
    ocean_stats = {}
    if ocean_mode in ("thin_sheet", "real"):
        ocean_stats = add_ocean(
            tria, ocean_mode, dem, bounds=model_bounds(dx, dy),
            dz_earth=spec["dz_earth"], materials=materials,
            background_resistivity_ohm_m=bg_res, ocean_resistivity_ohm_m=sea_res,
            sheet_m=float(spec.get("ocean_thickness_m") or 1500.0), log=log)

    # -- 7. project receivers onto the interface -------- #
    # offshore stations sit on the seafloor: the earth is topped by water, not air
    receivers_projected = project_receivers(
        tria, receivers,
        seafloor=bool(ocean_stats.get("mode") == "real" and ocean_stats.get("n_ocean_cells")),
        names=receiver_names, log=log)

    # -- 8, 9. mask, mesh, materials, receivers ----------- #
    paths = write_mesh_files(tria, materials, receiver_names, receivers_projected, out_dir,
                             free_ids=earth_ids, log=log)
    if fmt_key == "femtic":
        from . import femtic

        paths.update(femtic.write_mesh(tria, materials, out_dir, air_id=air_id,
                                       free_ids=earth_ids, sea_ids=(OCEAN_ID,), log=log))
    elif fmt_key == "simpeg":
        from . import simpeg_mesh

        paths.update(simpeg_mesh.write_mesh(
            tria, materials, out_dir, dem=dem,
            ids=dict(air=AIR_ID, earth=EARTH_ID, sea=OCEAN_ID, mantle=MANTLE_ID),
            free_ids=earth_ids, land=use_topo, bathymetry=(ocean_mode == "real"),
            mantle_depth_m=mantle_z if mantle_on else None,
            names=receiver_names, receivers=receivers,
            vertical_ratio=float(spec.get("simpeg_vertical_ratio") or 1.0), log=log))
    return {
        **{k: v for k, v in paths.items() if k.startswith(("femtic_", "simpeg_"))},
        "n_cells": tria.n_active_cells(),
        "smallest_cell_m": tria.minimal_cell_diameter(),
        "largest_cell_m": tria.maximal_cell_diameter(),
        "tria": paths["tria"], "vtu": paths["vtu"], "materials": paths["materials"],
        "mask": paths["mask"], "receivers": paths["receivers"],
    }
