"""Refine an existing (usually inverted) GoFEM model.

Input is a ``.tria`` plus its ``.materials``.  An inverted model differs from a
freshly built one in ways that matter here:

  * Every inverted cell has its own material id; fixed cells keep their low
    original ids.  Refined children inherit the parent's id (so they start from
    its conductivity) and are separate parameters to GoFEM, so ids never change.
  * The mesh is already fitted to topography.  ``Topography.fit_to`` shifts vertices
    relative to where they are, so calling it again applies the relief twice.
    :func:`refit_topography` moves the surface by the *difference* from where it
    is now, which needs neither the old DEM nor the band the mesh was built with.
  * ``gofem.mesh_utils.project_points_on_interface`` only knows one earth id, and
    with per-cell ids it returns the points unchanged without an error.  Stations
    are placed from the surface faces here instead (:func:`project_to_surface`),
    and every station is checked against the mesh before anything is written
    (:func:`verify_receivers`).

Coordinates are the mesh frame: x = north, y = east, z = depth (positive down), so
elevation = -z.  Refined meshes must be read by GoFEM, so keep to deal.II 9.3.
"""
from __future__ import annotations

import dataclasses
import glob
import math
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mesh_design import survey_area_shells

AIR_ID = 1
_FRAME_LIMIT_M = 100.0                   # a right frame fits its stations to centimetres
#: stations are written this far below the surface
RECEIVER_OFFSET_M = 0.1
#: ids at or above this much of a jump from the id below are taken as inverted
_ID_GAP_RATIO = 10.0

RULE_KINDS = {
    "box": "A rectangle, as a column between two depths.",
    "ellipse": "An ellipse, as a column between two depths.",
    "polygon": "A polygon, as a column between two depths, or a bowl below it (pnorm > 1).",
    "stations": "An ellipsoid around each station.",
}


@dataclass
class RefineRule:
    """One refinement rule.  Distances are km; depths are below the local surface.

    A rule makes one refinement per shell, outer to inner (:func:`rule_shells`).  Each
    splits the cells that overlap the region grown by the shell's buffer, down to the
    shell's depth, unless that would take them below ``min_size_km`` (horizontal) or
    ``min_dz_km`` (vertical).  With no buffer and no taper, every shell is the region itself.
    """
    kind: str = "box"
    enabled: bool = True
    levels: int = 1
    min_size_km: float = 0.0
    min_dz_km: float = 0.0
    depth_top_km: float = 0.0
    depth_bottom_km: float = 50.0
    # box / ellipse
    center_north_km: float = 0.0
    center_east_km: float = 0.0
    half_north_km: float = 50.0
    half_east_km: float = 50.0
    # polygon: [[north_km, east_km], ...]
    polygon_km: list = field(default_factory=list)
    # stations: "new" | "all"; depth_bottom_km is the ellipsoid's vertical radius
    stations: str = "new"
    radius_km: float = 5.0
    # polygon: 1 is a flat-bottomed column; above 1 the bottom is rounded, as pyGoFEM's
    # refine_within_polygon does it (2 is a bowl; higher is flatter, closer to 1 more pointed)
    pnorm: float = 1.0
    # shells: named as in mesh_design.RefinementStage; depth_bottom_km is shell 0's depth.
    # These defaults are what a rule saved before shells existed has to mean.
    buffer_km: float = 0.0
    buffer_shrink: float = 0.5
    buffer_override: bool = False
    buffer_km_list: list = field(default_factory=list)
    taper: bool = False
    depth_shrink: float = 0.55
    depth_override: bool = False
    depth_km_list: list = field(default_factory=list)
    # "refine" splits cells; "coarsen" merges split cells back into their parents (never past
    # the model's original grid), inside the region or, with ``outside``, everywhere the region
    # (grown by buffer_km, down to depth_bottom_km) doesn't reach: the padding.  ``levels`` is
    # how many times; cells aren't merged into ones wider than ``max_size_km`` (0: no limit).
    action: str = "refine"
    outside: bool = False
    max_size_km: float = 0.0

    def asdict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def rule_from_dict(raw: dict) -> RefineRule:
    names = {f.name for f in dataclasses.fields(RefineRule)}
    return RefineRule(**{k: v for k, v in raw.items() if k in names})


def rule_shells(rule: RefineRule) -> list[tuple[float, float]]:
    """``(buffer_km, depth_km)`` per shell, outer to inner: the Mesh tab's survey-area sequence."""
    buffers, depths = survey_area_shells({**rule.asdict(), "max_depth_km": rule.depth_bottom_km})
    if min(buffers) < 0 or min(depths) < 0:
        raise ValueError("a rule's shell buffers and depths can't be negative")
    return list(zip(buffers, depths))


def rule_polygon(rule: RefineRule, buffer_km: float = 0.0):
    """A polygon rule's region as a shapely geometry, (north, east) metres."""
    import shapely.geometry as sgeom

    poly = sgeom.Polygon([(float(a) * 1e3, float(b) * 1e3) for a, b in rule.polygon_km])
    return poly.buffer(buffer_km * 1e3) if buffer_km > 0 else poly


# --------------------------------------------------------------------------- #
# materials
# --------------------------------------------------------------------------- #
@dataclass
class Material:
    id: int
    name: str
    conductivity: float


def read_materials(path: str) -> dict[int, Material]:
    """``.materials``: an ``iso`` line, then ``id name conductivity permittivity``."""
    out: dict[int, Material] = {}
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 3 or line.lstrip().startswith("#"):
                continue
            try:
                out[int(parts[0])] = Material(int(parts[0]), parts[1], float(parts[2]))
            except ValueError:
                continue
    return out


def guess_fixed_ids(ids) -> list[int]:
    """Which ids are fixed cells?  GoFEM gives inverted cells new, much larger ids,
    so the fixed ones are those below the first big jump.  With no jump this is a
    model that hasn't been inverted, where only air is certain."""
    unique = sorted({int(i) for i in ids})
    for a, b in zip(unique, unique[1:]):
        if a > 0 and b / a >= _ID_GAP_RATIO and b >= 1000:
            return [i for i in unique if i <= a]
    return [i for i in unique if i == AIR_ID]


def water_ids(materials: dict[int, Material]) -> list[int]:
    """Real-ocean water cells: the setup GUI names that material ``sea``."""
    return [m.id for m in materials.values() if m.name.lower() == "sea"]


# --------------------------------------------------------------------------- #
# loading + cell arrays
# --------------------------------------------------------------------------- #
def load_tria(path: str):
    from .. import config

    config.apply_sys_paths()
    import PyDealII.Release as dealii

    tria = dealii.Triangulation("3D")
    tria.load(path)
    return tria


def _cell_arrays(tria):
    """Centre x/y/z and material id per active cell, in ``active_cells()`` order."""
    cells = tria.active_cells()
    n = len(cells)
    xyz = np.empty((n, 3), dtype="float64")
    mat = np.empty(n, dtype="int64")
    for i, c in enumerate(cells):
        p = c.center()
        xyz[i] = p.x, p.y, p.z
        mat[i] = c.material_id
    return cells, xyz, mat


@dataclass
class ModelInfo:
    n_cells: int
    n_air: int
    n_ids: int
    smallest_m: float
    largest_m: float
    extent_m: tuple                         # north min/max, east min/max, z min/max
    elevation_m: tuple                      # surface min/max
    fixed_ids: list
    water_ids: list
    missing_ids: list                       # in the .tria, absent from the .materials
    materials: dict


def inspect(tria, materials: dict[int, Material] | None = None) -> ModelInfo:
    _cells, xyz, mat = _cell_arrays(tria)
    materials = materials or {}
    ids = np.unique(mat)
    surf = surface_faces(tria, top_ids=[AIR_ID, *water_ids(materials)])
    ext = (xyz[:, 0].min(), xyz[:, 0].max(), xyz[:, 1].min(), xyz[:, 1].max(),
           xyz[:, 2].min(), xyz[:, 2].max())
    return ModelInfo(
        n_cells=len(mat), n_air=int((mat == AIR_ID).sum()), n_ids=len(ids),
        smallest_m=tria.minimal_cell_diameter(), largest_m=tria.maximal_cell_diameter(),
        extent_m=tuple(float(v) for v in ext),
        elevation_m=(float(-surf.z.max()), float(-surf.z.min())),
        fixed_ids=guess_fixed_ids(ids), water_ids=water_ids(materials),
        missing_ids=[int(i) for i in ids if materials and int(i) not in materials],
        materials=materials)


# --------------------------------------------------------------------------- #
# the surface
# --------------------------------------------------------------------------- #
class SurfaceFaces:
    """The top of the earth: every face between an earth cell and air or water.

    Faces are axis-aligned rectangles in plan (only z was ever moved), so a point
    is placed by finding its rectangle and interpolating bilinearly, which is what
    the face geometry is.
    """

    def __init__(self, faces: np.ndarray):
        # (n, 4, 3): corners ordered (x0,y0), (x1,y0), (x0,y1), (x1,y1)
        self.faces = faces
        self.x0 = faces[:, 0, 0]
        self.x1 = faces[:, 1, 0]
        self.y0 = faces[:, 0, 1]
        self.y1 = faces[:, 2, 1]
        self.z = faces[:, :, 2].mean(axis=1)

    def __len__(self) -> int:
        return len(self.faces)

    def locate(self, x: float, y: float) -> int:
        """Index of the face over (x, y), or -1 if it is off the mesh."""
        hit = np.flatnonzero((x >= self.x0) & (x < self.x1) & (y >= self.y0) & (y < self.y1))
        if hit.size:
            return int(hit[0])
        hit = np.flatnonzero((x >= self.x0) & (x <= self.x1) & (y >= self.y0) & (y <= self.y1))
        return int(hit[0]) if hit.size else -1

    def z_at(self, i: int, x: float, y: float) -> float:
        u = (x - self.x0[i]) / (self.x1[i] - self.x0[i])
        v = (y - self.y0[i]) / (self.y1[i] - self.y0[i])
        z = self.faces[i, :, 2]
        return float((1 - u) * (1 - v) * z[0] + u * (1 - v) * z[1]
                     + (1 - u) * v * z[2] + u * v * z[3])

    def centres_tree(self):
        from scipy.spatial import cKDTree

        return cKDTree(self.faces[:, :, :2].mean(axis=1))

    def vertices(self) -> np.ndarray:
        """Unique (x, y, z) surface vertices."""
        pts = self.faces.reshape(-1, 3)
        _u, i = np.unique(np.round(pts[:, :2], 3), axis=0, return_index=True)
        return pts[i]


def surface_faces(tria, top_ids=(AIR_ID,)) -> SurfaceFaces:
    """Collect the surface.  Face 4 of a hex is its lower-z face, which is the top
    here since z points down (the mesh is built axis-aligned and only z ever moves)."""
    top = {int(i) for i in top_ids}
    rows = []
    for c in tria.active_cells():
        if c.material_id in top:
            continue
        try:
            above = c.neighbor(4).material_id
        except Exception:  # noqa: BLE001 -- the model's top boundary
            continue
        if above not in top:
            continue
        f = c.faces()[4]
        rows.append([[v.x, v.y, v.z] for v in (f.get_vertex(i) for i in range(4))])
    if not rows:
        raise ValueError("no air / earth interface found in this model")
    corners = np.asarray(rows, dtype="float64")
    # robust to vertex order: sort each face's corners into (x0,y0) (x1,y0) (x0,y1) (x1,y1)
    x, y = corners[:, :, 0], corners[:, :, 1]
    rank = (x > x.mean(axis=1, keepdims=True)).astype(int) \
        + 2 * (y > y.mean(axis=1, keepdims=True)).astype(int)
    order = np.argsort(rank, axis=1)
    return SurfaceFaces(np.take_along_axis(corners, order[:, :, None], axis=1))


# --------------------------------------------------------------------------- #
# refinement rules
# --------------------------------------------------------------------------- #
def _half_sizes(cells, idx) -> np.ndarray:
    """(len(idx), 3) half-widths of those cells: north, east, and mean half-thickness
    (from the volume, so it follows topography)."""
    out = np.empty((len(idx), 3), dtype="float64")
    for k, i in enumerate(idx):
        c = cells[i]
        a, b = c.get_vertex(0), c.get_vertex(7)         # opposite corners
        dx, dy = abs(b.x - a.x), abs(b.y - a.y)
        out[k] = 0.5 * dx, 0.5 * dy, 0.5 * c.measure() / max(dx * dy, 1e-30)
    return out


def _in_region(rule: RefineRule, x, y, depth, station_xy=None, half=None, shell=None) -> np.ndarray:
    """Which cells overlap the rule's region?  *x*, *y*, *depth* are cell centres, with
    depth below the surface; *half* is :func:`_half_sizes` (omit to test points); *shell*
    is one of :func:`rule_shells` (omit for the region as drawn, to its bottom depth).

    It has to be overlap, not the centre being inside: a small polygon within a large
    cell, or a shallow depth range in a thick surface cell, could otherwise refine nothing.
    """
    x, y, depth = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64"), np.asarray(depth, dtype="float64")
    half = np.zeros((len(x), 3)) if half is None else half
    hx, hy, hz = half[:, 0], half[:, 1], half[:, 2]
    buffer_km, bottom_km = shell if shell is not None else (0.0, rule.depth_bottom_km)

    if rule.kind == "stations":
        pts = np.asarray(station_xy if station_xy is not None else [], dtype="float64").reshape(-1, 2)
        if not len(pts):
            return np.zeros(len(x), dtype=bool)
        from scipy.spatial import cKDTree

        _d, i = cKDTree(pts).query(np.c_[x, y])
        dn = np.maximum(np.abs(x - pts[i, 0]) - hx, 0.0)
        de = np.maximum(np.abs(y - pts[i, 1]) - hy, 0.0)
        dv = np.maximum(depth - hz, 0.0)
        rh = max((rule.radius_km + buffer_km) * 1e3, 1.0)
        rz = max(bottom_km * 1e3, 1.0)
        return (dn / rh) ** 2 + (de / rh) ** 2 + (dv / rz) ** 2 <= 1.0

    column = (depth + hz >= rule.depth_top_km * 1e3) & (depth - hz <= bottom_km * 1e3)
    cn, ce = rule.center_north_km * 1e3, rule.center_east_km * 1e3
    an = max((rule.half_north_km + buffer_km) * 1e3, 1.0)
    ae = max((rule.half_east_km + buffer_km) * 1e3, 1.0)
    if rule.kind == "box":
        return column & (np.abs(x - cn) < an + hx) & (np.abs(y - ce) < ae + hy)
    if rule.kind == "ellipse":
        dn = np.maximum(np.abs(x - cn) - hx, 0.0)
        de = np.maximum(np.abs(y - ce) - hy, 0.0)
        return column & ((dn / an) ** 2 + (de / ae) ** 2 <= 1.0)
    if rule.kind == "polygon":
        if len(rule.polygon_km) < 3:
            raise ValueError("a polygon rule needs at least three vertices")
        import shapely.geometry as sgeom
        import shapely.prepared
        import shapely.vectorized

        poly = rule_polygon(rule, buffer_km)
        inside = shapely.vectorized.contains(poly, x, y)
        n0, e0, n1, e1 = poly.bounds
        near = (~inside) & column & (x + hx > n0) & (x - hx < n1) & (y + hy > e0) & (y - hy < e1) \
            & ((hx > 0) | (hy > 0))
        prep = shapely.prepared.prep(poly)
        for i in np.flatnonzero(near):
            inside[i] = prep.intersects(sgeom.box(x[i] - hx[i], y[i] - hy[i], x[i] + hx[i], y[i] + hy[i]))
        if rule.pnorm > 1:
            inside &= _in_bowl(poly, rule, bottom_km, x, y, depth, hx, hy, hz)
        return column & inside
    raise ValueError(f"unknown refinement rule {rule.kind!r}")


def _in_bowl(poly, rule: RefineRule, bottom_km: float, x, y, depth, hx, hy, hz) -> np.ndarray:
    """Cells that overlap a rounded polygon's bowl, the shape pyGoFEM's
    ``refine_within_polygon(pnorm > 1)`` gives (the Mesh tab's polygon pnorm): an ellipsoid
    centred on the polygon's centroid at the top depth, as wide as the circle round the
    polygon (half its bounding box's diagonal, plus 5%), reaching down to the bottom depth.
    pyGoFEM measures that from the top depth too, so the two differ when the top isn't 0."""
    n0, e0, n1, e1 = poly.bounds
    radius = max(0.5 * math.hypot(n1 - n0, e1 - e0) * 1.05, 1.0)
    centre = poly.centroid
    top = rule.depth_top_km * 1e3
    rz = max((bottom_km - rule.depth_top_km) * 1e3, 1.0)
    dn = np.maximum(np.abs(x - centre.x) - hx, 0.0)
    de = np.maximum(np.abs(y - centre.y) - hy, 0.0)
    dz = np.maximum(depth - hz - top, 0.0)
    p = float(rule.pnorm)
    return (dn / radius) ** p + (de / radius) ** p + (dz / rz) ** p <= 1.0


def level_jumps(tria) -> np.ndarray:
    """Per active cell (``active_cells()`` order): how many levels finer the finest cell
    sharing one of its vertices is."""
    cells = tria.active_cells()
    lv = np.array([c.level() for c in cells])
    vid = np.array([[c.vertex_index(i) for i in range(8)] for c in cells])
    vmax = np.full(int(vid.max()) + 1, -1)
    np.maximum.at(vmax, vid.ravel(), np.repeat(lv, 8))
    return vmax[vid].max(axis=1) - lv


def balance_levels(tria, log=print) -> int:
    """Split coarse cells until no vertex joins cells more than one level apart; returns
    the cells added.

    deal.II only keeps *face* neighbours within one level, so a region's corner can leave
    cells two levels apart along an edge.  GoFEM's edge elements keep their unknowns on
    edges, and p4est meshes are always balanced this way.  (Added while chasing NaNs in
    GoFEM's forward runs on a refined model; balancing turned out not to be their cause.)
    Splitting never moves the surface.
    """
    added = passes = 0
    while True:
        jump = level_jumps(tria)
        flag = np.flatnonzero(jump >= 2)
        if not len(flag):
            break
        cells = tria.active_cells()
        before = tria.n_active_cells()
        for k in flag:
            cells[k].refine_flag = "isotropic"
        tria.execute_coarsening_and_refinement()
        added += tria.n_active_cells() - before
        passes += 1
    if added:
        log(f"balanced the levels: {added:,} cells added in {passes} passes, so no vertex joins "
            f"cells more than one level apart")
    return added


# --------------------------------------------------------------------------- #
# smoothing the refined cells
# --------------------------------------------------------------------------- #
SMOOTH_MODES = {
    "relative": "x the original cell's size",
    "km": "km",
}
_KEY = 1 << 40                              # (level, index) -> one integer
#: neighbour pairs per batch, and the most a smoothing may need in all
_PAIRS_PER_BATCH = 5_000_000
_PAIRS_LIMIT = 3_000_000_000


@dataclass
class CellSnapshot:
    """The active cells as they were before refining: what :func:`smooth_refined` reads."""
    keys: np.ndarray                        # level * _KEY + index
    xyz: np.ndarray
    mat: np.ndarray
    half: np.ndarray                        # half-widths north, east, thickness
    volume: np.ndarray


def snapshot_cells(tria) -> CellSnapshot:
    cells, xyz, mat = _cell_arrays(tria)
    keys = np.fromiter((c.level() * _KEY + c.index() for c in cells), dtype="int64", count=len(cells))
    half = _half_sizes(cells, range(len(cells)))
    return CellSnapshot(keys, xyz, mat, half, 8.0 * half.prod(axis=1))


def smooth_refined(tria, materials: dict[int, Material], before: CellSnapshot, *,
                   mode: str = "relative", horizontal: float = 0.25, vertical: float = 0.25,
                   fixed_ids=(), exclude_keys=(), log=print) -> dict:
    """Smooth log10(conductivity) over the cells refining created, so the old cells'
    blocks aren't imprinted on them.  Each such cell gets a new material id of its own
    (added to *materials*); every other cell keeps its id and value.

    The value is the model *before* refining, Gaussian-filtered (standard deviations sh
    across, sv down) and read at the new cell's centre: each original cell counts by the
    Gaussian's integral over it (:func:`_gaussian_of_boxes`), as a ``gaussian_filter`` on a
    fine grid would, without the grid.  Air, sea and *fixed_ids* neither give nor get
    values, and the average is over the rest, so near the surface it uses the earth only.  *mode* "relative": sh and sv are
    *horizontal* and *vertical* times the width and thickness of the original cell the
    new one came from, so the smoothing follows the blocks being smoothed, at every
    depth.  "km": fixed lengths.  Averaging log values can't leave the range the model
    already has, so bounds such as GoFEM's BOUNDED transformation still hold.

    With "relative", 0.25 is light: across an old cell's face the change is spread over
    about half its width; 0.5 blurs a pattern of single cells out almost entirely.

    Returns ``cells`` (how many changed), ``new_ids`` (id -> Material, already in
    *materials*), ``change`` (abs change in log10 per changed cell).
    """
    if mode not in SMOOTH_MODES:
        raise ValueError(f"unknown smoothing mode {mode!r}")
    if horizontal <= 0 or vertical <= 0:
        raise ValueError("smoothing lengths must be above zero")
    cells, xyz, mat = _cell_arrays(tria)
    keys = np.fromiter((c.level() * _KEY + c.index() for c in cells), dtype="int64", count=len(cells))
    fixed = np.array(sorted({int(i) for i in fixed_ids} | {AIR_ID} | set(water_ids(materials))))
    sigma = {i: m.conductivity for i, m in materials.items()}

    def free(ids):
        cond = np.array([sigma.get(int(i), np.nan) for i in ids], dtype="float64")
        ok = ~np.isin(ids, fixed) & np.isfinite(cond) & (cond > 0)
        return ok, np.log10(np.where(ok, cond, 1.0))

    ok_now, _ = free(mat)
    # A cell that was there before keeps its key and centre.  deal.II reuses the slots of
    # merged cells, so a key alone isn't enough: a refined cell can come back under one.
    order = np.argsort(before.keys)
    pos = np.clip(np.searchsorted(before.keys[order], keys), 0, len(order) - 1)
    match = order[pos]
    same = (before.keys[match] == keys) & (np.abs(before.xyz[match] - xyz).max(axis=1) < 1e-3)
    targets = np.flatnonzero(ok_now & ~same & ~np.isin(keys, np.asarray(list(exclude_keys), dtype="int64")))
    empty = {"cells": 0, "new_ids": {}, "change": np.zeros(0)}
    if not len(targets):
        log("smoothing: no refined cells to smooth")
        return empty
    ok_src, value_src = free(before.mat)
    src = np.flatnonzero(ok_src)
    if not len(src):
        log("smoothing: the original model has no free cells to take values from")
        return empty

    t_xyz = xyz[targets]
    t_half = _half_sizes(cells, targets)
    if mode == "relative":
        # the original cell a new one came from: its own id, in an inverted model
        ids, first, counts = np.unique(before.mat, return_index=True, return_counts=True)
        pos = np.searchsorted(ids, mat[targets])
        found = (pos < len(ids)) & (ids[np.minimum(pos, len(ids) - 1)] == mat[targets])
        single = found & (counts[np.minimum(pos, len(ids) - 1)] == 1)
        parent_half = np.where(single[:, None], before.half[first[np.minimum(pos, len(ids) - 1)]],
                               2.0 * t_half)
        if (~single).any():
            log(f"smoothing: {int((~single).sum()):,} refined cells share their id with other "
                "original cells (not an inverted model?), so twice their own size is used for them")
        sh = horizontal * 2.0 * parent_half[:, :2].max(axis=1)
        sv = vertical * 2.0 * parent_half[:, 2]
    else:
        sh = np.full(len(targets), horizontal * 1e3)
        sv = np.full(len(targets), vertical * 1e3)

    new_value = _gaussian_of_boxes(
        t_xyz, sh, sv, before.xyz[src], before.half[src], value_src[src], log=log)

    old_value = np.log10([sigma[int(i)] for i in mat[targets]])
    new_value = np.where(np.isfinite(new_value), new_value, old_value)
    next_id = max(max(materials), int(mat.max())) + 1
    new_ids = {}
    for k, i in enumerate(targets):
        mid = next_id + k
        new_ids[mid] = Material(mid, materials[int(mat[i])].name, float(10.0 ** new_value[k]))
        cells[i].material_id = mid
    materials.update(new_ids)
    change = np.abs(new_value - old_value)
    lengths = (f"{horizontal:g} x the original cells' width, {vertical:g} x their thickness"
               if mode == "relative" else f"{horizontal:g} km across, {vertical:g} km down")
    log(f"smoothed {len(targets):,} refined cells ({lengths}): log10 conductivity changed by up to "
        f"{change.max():.2f} (median {np.median(change):.3f}); new ids {next_id:,} to {next_id + len(targets) - 1:,}")
    return {"cells": len(targets), "new_ids": new_ids, "change": change}


def _gaussian_of_boxes(points, sh, sv, centres, half, values, log=print) -> np.ndarray:
    """At each point, the Gaussian-filtered value of a field that is constant on boxes
    (``centres`` +- ``half``): sum of value x the Gaussian's integral over each box,
    divided by the integral over all boxes, so space with no box (air) doesn't count.
    Per axis the integral is a difference of normal CDFs, so this is exact for boxes,
    whatever their size relative to the kernel.  NaN where no box is within reach."""
    from scipy.spatial import cKDTree
    from scipy.special import ndtr

    def integral(lo, hi, t, s):
        return ndtr((hi - t) / s) - ndtr((lo - t) / s)

    # Stretch z so most kernels are near round; then a ball reaches every box they touch.
    stretch = float(np.median(sh / sv))
    scale = np.array([1.0, 1.0, stretch])
    reach = 3.0 * np.maximum(sh, stretch * sv)          # the kernel's weight past 3 sigma: < 0.3 %
    diag = np.linalg.norm(half * scale, axis=1)
    # Boxes in size classes (factors of 2), each with its own tree, so a few huge padding
    # cells don't widen the search for every point.
    size_class = np.floor(np.log2(np.maximum(diag, 1e-9))).astype(int)
    num = np.zeros(len(points))
    den = np.zeros(len(points))
    q_all = points * scale
    for cls in np.unique(size_class):
        members = np.flatnonzero(size_class == cls)
        tree = cKDTree(centres[members] * scale)
        radius = reach + diag[members].max()
        counts = np.asarray(tree.query_ball_point(q_all, radius, return_length=True))
        if counts.sum() > _PAIRS_LIMIT:
            raise ValueError(f"these smoothing lengths reach {int(counts.sum()):,} cell pairs, too many: "
                             "make them shorter, or use lengths relative to the original cells")
        first = 0
        while first < len(points):
            last = first + 1
            total = counts[first]
            while last < len(points) and total + counts[last] <= _PAIRS_PER_BATCH:
                total += counts[last]
                last += 1
            if total:
                nbrs = tree.query_ball_point(q_all[first:last], radius[first:last])
                lens = np.fromiter((len(n) for n in nbrs), dtype="int64", count=last - first)
                row = np.repeat(np.arange(first, last), lens)
                col = members[np.fromiter((j for n in nbrs for j in n), dtype="int64", count=int(lens.sum()))]
                p, c, h = points[row], centres[col], half[col]
                w = (integral(c[:, 0] - h[:, 0], c[:, 0] + h[:, 0], p[:, 0], sh[row])
                     * integral(c[:, 1] - h[:, 1], c[:, 1] + h[:, 1], p[:, 1], sh[row])
                     * integral(c[:, 2] - h[:, 2], c[:, 2] + h[:, 2], p[:, 2], sv[row]))
                num += np.bincount(row, w * values[col], minlength=len(points))
                den += np.bincount(row, w, minlength=len(points))
            first = last
    out = np.full(len(points), np.nan)
    ok = den > 1e-12
    out[ok] = num[ok] / den[ok]
    if (~ok).any():
        log(f"smoothing: {int((~ok).sum()):,} cells have no original cell within reach and keep their value")
    return out


# --------------------------------------------------------------------------- #
# coarsening
# --------------------------------------------------------------------------- #
def _cell_keys(cells) -> np.ndarray:
    return np.fromiter((c.level() * _KEY + c.index() for c in cells), dtype="int64", count=len(cells))


def _cell_boxes(cells, idx) -> np.ndarray:
    """(len(idx), 6): north0, north1, east0, east1, z0, z1 of each cell's vertices."""
    out = np.empty((len(idx), 6))
    for k, i in enumerate(idx):
        vs = [cells[i].get_vertex(j) for j in range(8)]
        xs, ys, zs = [v.x for v in vs], [v.y for v in vs], [v.z for v in vs]
        out[k] = min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)
    return out


def _children_of(parent_box: np.ndarray, child_xyz: np.ndarray, child_half: np.ndarray) -> np.ndarray:
    """For each child centre, the index of the parent box holding it (-1 if none).

    A child is half its parent's size, so the parent's centre is about one child half-width
    away along each axis: each child looks within 1.5 x its own half-diagonal, a handful of
    candidates whatever the cell sizes (searching from the parents, a 100 km padding cell
    would sweep up thousands of small children).  The box decides: the footprint is exact
    (cells have vertical sides), and a centre within two boxes' depth ranges goes to the
    one whose middle is nearer."""
    from scipy.spatial import cKDTree

    centre = np.c_[parent_box[:, 0:2].mean(axis=1), parent_box[:, 2:4].mean(axis=1),
                   parent_box[:, 4:6].mean(axis=1)]
    reach = 1.5 * np.linalg.norm(child_half, axis=1)
    lists = cKDTree(centre).query_ball_point(child_xyz, reach)
    owner = np.full(len(child_xyz), -1, dtype="int64")
    for k, cands in enumerate(lists):
        x, y, z = child_xyz[k]
        best, gap = -1, np.inf
        for j in cands:
            b = parent_box[j]
            if b[0] <= x <= b[1] and b[2] <= y <= b[3] and b[4] <= z <= b[5] and abs(z - centre[j, 2]) < gap:
                best, gap = j, abs(z - centre[j, 2])
        owner[k] = best
    return owner


def coarsen_rule(tria, rule: RefineRule, surf: SurfaceFaces, *, station_xy=None, materials=None,
                 fixed_ids=(AIR_ID,), top_ids=(AIR_ID,), log=print) -> dict:
    """Merge split cells back into their parents, up to ``rule.levels`` times, inside the
    rule's region or (``rule.outside``) outside it.

    deal.II merges a parent only when all 8 of its children are flagged, so the children
    are flagged one kind at a time: the inverted (free) cells together, then each fixed id
    (air, sea, ...) on its own.  A parent is never made of air and earth, or of two fixed
    materials.  Air and sea (*top_ids*) go with the column they're over: they count as at the
    region's top depth.  A merged inverted parent gets a new material id of its own, with the
    volume-weighted mean of its children's log10 conductivity (added to *materials*); a fixed
    one keeps the id its children had.  Merging coarsens the surface back to the parent's
    corners, so stations are placed on the finished surface afterwards, as always.

    Returns ``passes``, ``shells`` (= levels), ``added`` (negative), ``n_cells``, ``merged``
    (how many cells were merged) and ``new_ids`` (id -> Material) and ``keys`` (the merged
    cells, for :func:`smooth_refined` to leave alone).
    """
    fixed = sorted({int(i) for i in fixed_ids})
    top = {int(i) for i in top_ids}
    start = tria.n_active_cells()
    tree = surf.centres_tree()
    shell = (rule.buffer_km, rule.depth_bottom_km)
    new_ids: dict[int, Material] = {}
    merged_keys: list[int] = []
    merged = passes = 0
    for level in range(max(int(rule.levels), 1)):
        changed = False
        _c, _x, mat = _cell_arrays(tria)
        for group in [None, *[i for i in fixed if (mat == i).any()]]:     # None: the inverted cells
            label = "inverted cells" if group is None else f"id {group}"
            cells, xyz, mat = _cell_arrays(tria)
            in_group = ~np.isin(mat, fixed) if group is None else mat == group
            lv = np.fromiter((c.level() for c in cells), dtype="int64", count=len(cells))
            idx = np.flatnonzero(in_group & (lv > 0))
            if not len(idx):
                continue
            half = _half_sizes(cells, idx)
            _d, nearest = tree.query(xyz[idx, :2])
            depth = xyz[idx, 2] - surf.z[nearest]
            if group is not None and group in top:
                depth = np.maximum(depth, rule.depth_top_km * 1e3)
            # A parent merges only when all 8 children are flagged, and a child doesn't know which
            # way its parent lies, so test where the parent could be: the child grown to 3 half-widths
            # each way.  Inside: the parent touches the region; outside: it stays clear of it.
            touches = _in_region(rule, xyz[idx, 0], xyz[idx, 1], depth, station_xy, 3.0 * half, shell)
            take = ~touches if rule.outside else touches
            if rule.max_size_km > 0:
                take &= 4.0 * np.maximum(half[:, 0], half[:, 1]) <= rule.max_size_km * 1e3 + 1e-6
            idx, half = idx[take], half[take]
            if not len(idx):
                continue
            free = group is None
            if free:
                if not materials:
                    raise ValueError("coarsening an inverted model needs its .materials, to average "
                                     "the merged cells' conductivities")
                missing = sorted({int(i) for i in mat[idx]} - set(materials))
                if missing:
                    raise ValueError(f"{len(missing)} material ids aren't in the .materials (e.g. {missing[:5]})")
                child_xyz = xyz[idx]
                child_vol = 8.0 * half.prod(axis=1)
                child_half = half.copy()
                child_val = np.log10([materials[int(i)].conductivity for i in mat[idx]])
                child_name = [materials[int(i)].name for i in mat[idx]]
            before = set(_cell_keys(cells).tolist())
            for i in idx:
                cells[i].coarsen_flag = True
            tria.execute_coarsening_and_refinement()
            cells, xyz, mat = _cell_arrays(tria)
            keys = _cell_keys(cells)
            parents = np.flatnonzero(~np.isin(keys, list(before)))
            if not len(parents):
                continue
            changed = True
            passes += 1
            merged += 8 * len(parents)
            merged_keys.extend(int(k) for k in keys[parents])
            if free:
                owner = _children_of(_cell_boxes(cells, parents), child_xyz, child_half)
                got = owner >= 0
                num = np.bincount(owner[got], child_vol[got] * child_val[got], minlength=len(parents))
                den = np.bincount(owner[got], child_vol[got], minlength=len(parents))
                count = np.bincount(owner[got], minlength=len(parents))
                if (count != 8).any():
                    raise RuntimeError(f"coarsening: {int((count != 8).sum())} merged cells didn't find "
                                       "their 8 children; nothing was written")
                first_child = {int(o): k for k, o in reversed(list(enumerate(owner))) if o >= 0}
                next_id = max(max(materials), int(mat.max())) + 1
                for j, i in enumerate(parents):
                    m = Material(next_id + j, child_name[first_child[j]], float(10.0 ** (num[j] / den[j])))
                    new_ids[m.id] = m
                    materials[m.id] = m
                    cells[i].material_id = m.id
            else:
                for i in parents:
                    cells[i].material_id = int(group)
            log(f"  pass {level + 1}, {label}: {8 * len(parents):,} cells merged into {len(parents):,} "
                f"-> {tria.n_active_cells():,} cells")
        if not changed:
            if level == 0:
                log("  nothing to merge: no split cells there (or above the size limit)")
            break
    return {"passes": passes, "shells": max(int(rule.levels), 1), "added": tria.n_active_cells() - start,
            "n_cells": tria.n_active_cells(), "merged": merged, "new_ids": new_ids, "keys": merged_keys}


def apply_rule(tria, rule: RefineRule, surf: SurfaceFaces, *, station_xy=None,
               exclude_ids=(AIR_ID,), max_cells: int | None = None, materials=None,
               fixed_ids=(), log=print) -> dict:
    """Run one rule; returns ``{"passes", "shells", "added", "n_cells"}`` (a coarsening
    rule also ``"merged"``, see :func:`coarsen_rule`, and its ``added`` is negative).

    Neighbouring cells also get split where deal.II needs them within one level of
    each other.  Those are always the coarser cell, so they stay above the minimum.
    """
    if rule.action == "coarsen":
        return coarsen_rule(tria, rule, surf, station_xy=station_xy, materials=materials,
                            fixed_ids=[*exclude_ids, *fixed_ids], top_ids=exclude_ids, log=log)
    if rule.action != "refine":
        raise ValueError(f"unknown rule action {rule.action!r}")
    tree = surf.centres_tree()
    start = tria.n_active_cells()
    passes = 0
    shells = rule_shells(rule)
    idle = None                     # the last shell that had nothing to split
    exclude = np.asarray(sorted({int(i) for i in exclude_ids}), dtype="int64")
    for k, shell in enumerate(shells):
        buffer_km, bottom_km = shell
        if idle is not None and buffer_km <= idle[0] and bottom_km <= idle[1]:
            continue                # inside that one: its cells are no bigger now, so nothing here either
        cells, xyz, mat = _cell_arrays(tria)
        _d, nearest = tree.query(xyz[:, :2])
        depth = xyz[:, 2] - surf.z[nearest]
        idx = np.flatnonzero(~np.isin(mat, exclude))
        half = _half_sizes(cells, idx)
        inside = _in_region(rule, xyz[idx, 0], xyz[idx, 1], depth[idx], station_xy, half, shell)
        # splitting halves a cell, so its full width (2 x half) / 2 must stay above the minimum
        big_enough = (np.maximum(half[:, 0], half[:, 1]) >= rule.min_size_km * 1e3 - 1e-6) \
            & (half[:, 2] >= rule.min_dz_km * 1e3 - 1e-6)
        idx = idx[inside & big_enough]
        where = f"shell {k} (+{buffer_km:g} km, to {bottom_km:g} km)"
        if not len(idx):
            idle = shell            # an inner shell can still have work if the lists aren't nested
            log(f"  {where}: nothing to split")
            continue
        projected = tria.n_active_cells() + 7 * len(idx)
        if max_cells and projected > max_cells:
            raise ValueError(f"this rule would take the model to at least {projected:,} cells "
                             f"(limit {max_cells:,}). Shrink the region or raise the minimum size.")
        for i in idx:
            cells[i].refine_flag = "isotropic"
        tria.execute_coarsening_and_refinement()
        passes += 1
        log(f"  {where}: {len(idx):,} cells split -> {tria.n_active_cells():,} cells")
    return {"passes": passes, "shells": len(shells), "added": tria.n_active_cells() - start,
            "n_cells": tria.n_active_cells()}


# --------------------------------------------------------------------------- #
# topography
# --------------------------------------------------------------------------- #
@dataclass
class DemCheck:
    """Does a DEM look like it's in the model's frame?  ``status``: "ok", "unknown" (no
    relief to compare), or a problem: "swapped", "offset", "mismatch", "no_overlap"."""
    status: str
    message: str
    n_points: int = 0
    r: float = float("nan")
    rms_m: float = float("nan")
    relief_m: float = float("nan")
    shift_m: tuple | None = None            # (north, east) to move the DEM by to align it

    @property
    def is_problem(self) -> bool:
        return self.status in ("swapped", "offset", "mismatch", "no_overlap")


def _direction(shift_m: tuple) -> str:
    """(north, east) metres -> "9 km south, 3 km east"."""
    parts = []
    for value, plus, minus in ((shift_m[0], "north", "south"), (shift_m[1], "east", "west")):
        if abs(value) >= 500.0:
            parts.append(f"{abs(value) / 1e3:.0f} km {plus if value > 0 else minus}")
    return ", ".join(parts) or "under a kilometre"


def is_lonlat_dem(path: str) -> bool:
    """True for the Topography tab's lon/lat ``.npz``, False for Project DEM's, in the model's frame."""
    not_a_dem = ValueError(f"{os.path.basename(path)} isn't a DEM from the Topography or Project DEM tab")
    try:
        npz = np.load(path)
    except (ValueError, EOFError) as exc:
        raise not_a_dem from exc
    if not hasattr(npz, "files"):                # a plain .npy
        raise not_a_dem
    with npz:
        files = npz.files
    if "arr_0" not in files:
        raise not_a_dem
    return "arr_2" in files


_dem_cache: tuple = (None, None)      # the last lon/lat DEM resampled: it's slow, and used more than once


def _lonlat_dem_in_frame(path: str, georef, max_points: int) -> np.ndarray:
    """A lon/lat DEM resampled onto a regular grid in the model's frame.

    A lon/lat box is tilted in most projections, so its bounding box has corners the DEM
    doesn't cover, and elevations there would be invented.  The grid stays inside it.
    Spacing is the DEM's own, or coarser if that would exceed *max_points*.
    """
    from pyproj import Transformer
    from scipy.interpolate import RegularGridInterpolator

    from .topography import load_dem_npz

    dem = load_dem_npz(path)
    if not np.isfinite(dem.z).all():
        raise ValueError(f"{os.path.basename(path)} has missing elevations")
    forward = Transformer.from_crs(4326, georef.crs, always_xy=True)

    def edge(lon, lat):
        east, north = forward.transform(*np.broadcast_arrays(lon, lat))
        return np.asarray(east) - georef.x0, np.asarray(north) - georef.y0

    n0 = float(edge(dem.lon, dem.lat[0])[1].max())
    n1 = float(edge(dem.lon, dem.lat[-1])[1].min())
    e0 = float(edge(dem.lon[0], dem.lat)[0].max())
    e1 = float(edge(dem.lon[-1], dem.lat)[0].min())
    if n1 <= n0 or e1 <= e0:
        raise ValueError(f"{os.path.basename(path)} doesn't make a usable area in the model's frame")
    step = max((n1 - n0) / (len(dem.lat) - 1), (e1 - e0) / (len(dem.lon) - 1),
               math.sqrt((n1 - n0) * (e1 - e0) / max_points))
    east, north = np.meshgrid(np.arange(e0, e1, step), np.arange(n0, n1, step))
    lon, lat = Transformer.from_crs(georef.crs, 4326, always_xy=True).transform(east + georef.x0,
                                                                                north + georef.y0)
    z = RegularGridInterpolator((dem.lat, dem.lon), dem.z)(np.c_[lat.ravel(), lon.ravel()])
    return np.dstack([east, north, z.reshape(east.shape)])


def read_dem(path: str, georef=None, max_points: int = 4_000_000) -> np.ndarray:
    """``(ny, nx, 3)`` [easting, northing, elevation] in the model's frame, from either kind of
    ``.npz``.  A lon/lat DEM needs the model's frame, *georef*."""
    global _dem_cache

    if not is_lonlat_dem(path):
        topo_cart = np.load(path)["arr_0"]
        if topo_cart.ndim != 3 or topo_cart.shape[2] != 3:
            raise ValueError(f"{os.path.basename(path)} isn't a DEM from the Topography or Project DEM tab")
        return topo_cart
    if georef is None:
        raise ValueError(f"{os.path.basename(path)} is in lon/lat, so it needs the model's frame: "
                         "set the Model Frame on the Stations page")
    key = (os.path.abspath(path), os.path.getmtime(path), georef.crs, georef.x0, georef.y0, max_points)
    if _dem_cache[0] != key:
        _dem_cache = (key, _lonlat_dem_in_frame(path, georef, max_points))
    return _dem_cache[1]


def check_dem_frame(surf: SurfaceFaces, dem_path: str, max_points: int = 5000, georef=None) -> DemCheck:
    """Compare a projected DEM with the model's surface, to catch a DEM in the wrong frame.

    Overlap proves little: models are centred on (0, 0), so a DEM from another region
    can cover one completely.  Correlation alone misses small shifts too, because terrain
    varies slowly: on real data a 10 km error still gives r = 0.98.  So this also slides
    the DEM around: in the right frame, no shift should fit better than none.
    """
    from scipy.spatial import cKDTree

    topo_cart = read_dem(dem_path, georef)                 # [easting, northing, elevation]
    north, east = topo_cart[:, :, 1].ravel(), topo_cart[:, :, 0].ravel()
    elev = np.maximum(topo_cart[:, :, 2].ravel(), 0.0)     # sea is flat, as in the mesh
    n0, n1, e0, e1 = north.min(), north.max(), east.min(), east.max()
    tree = cKDTree(np.c_[north, east])

    verts = surf.vertices()
    inside = (verts[:, 0] >= n0) & (verts[:, 0] <= n1) & (verts[:, 1] >= e0) & (verts[:, 1] <= e1)
    pts = verts[inside]
    if len(pts) < 20:
        return DemCheck("no_overlap", "This DEM barely overlaps the model. Is it in the model's frame?", len(pts))
    if len(pts) > max_points:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_points, replace=False)]
    x, y, h = pts[:, 0], pts[:, 1], -pts[:, 2]
    relief = float(h.std())

    def sample(a, b):
        return elev[tree.query(np.c_[a, b])[1]]

    def misfit(d):
        return float(np.sqrt(np.mean((d - h) ** 2)))

    at_zero = sample(x, y)
    rms = misfit(at_zero)
    if relief < 5.0:
        return DemCheck("unknown", "The model's surface is flat here, so there's nothing to compare the DEM "
                        "with. Check by hand that it's in the model's frame.", len(pts), rms_m=rms, relief_m=relief)
    r = float(np.corrcoef(at_zero, h)[0, 1]) if at_zero.std() > 0 else 0.0
    base = dict(n_points=len(pts), r=r, rms_m=rms, relief_m=relief)
    summary = f"(misfit {rms:.0f} m rms against {relief:.0f} m of relief, correlation {r:.2f})"

    swapped = misfit(sample(y, x))
    if swapped < 0.7 * rms:
        return DemCheck("swapped", f"This DEM fits much better with north and east swapped ({swapped:.0f} m "
                        f"rms against {rms:.0f} m). Check its axes.", **base)

    # steps from about two DEM cells, up to a quarter of the DEM's width
    cell = float(np.median(np.abs(np.diff(np.unique(np.round(north))))))
    top = 0.25 * min(n1 - n0, e1 - e0)
    steps = np.geomspace(max(2.0 * cell, 1e3), max(top, 2.0 * cell), 7)
    steps = np.r_[-steps[::-1], 0.0, steps]
    best, best_shift = rms, None
    for dn in steps:
        for de in steps:
            if dn == 0.0 and de == 0.0:
                continue
            value = misfit(sample(x + dn, y + de))    # looking dn further north in the DEM...
            if value < best:
                best, best_shift = value, (-float(dn), -float(de))    # ...means moving the DEM back by dn
    if best_shift is not None and best < rms / 1.25 and best < 0.5 * relief and rms > max(5.0, 0.05 * relief):
        return DemCheck("offset", f"This DEM fits better if moved about {_direction(best_shift)} "
                        f"({best:.0f} m rms against {rms:.0f} m). It may be in a different frame or "
                        "have a different origin.", shift_m=best_shift, **base)
    if r < 0.5 or rms > 0.7 * relief:
        return DemCheck("mismatch", f"This DEM doesn't match the model's surface {summary}. It may be in "
                        "the wrong frame, or from another area.", **base)
    return DemCheck("ok", f"This DEM matches the model's surface {summary}.", **base)


def refit_topography(tria, dem_path: str, *, top_ids=(AIR_ID,), blend_km: float | None = None,
                     log=print, georef=None) -> dict:
    """Move the surface onto a DEM (see :func:`read_dem`).

    Each vertex shifts by (DEM - current surface) at its own x, y, tapering linearly
    to nothing at +/- a band around the surface.  Vertices already on the DEM don't
    move, so running it twice, or with the DEM the model was built from, can't
    double the relief.  As when building a mesh, sea stays flat at z = 0.

    A DEM says nothing beyond its own edges, so the surface outside it stays where it
    is, and the shift fades out over ``blend_km`` inside the edge (default: 5% of the
    DEM's width).  So a detailed regional DEM can be used on a much larger model.

    Junctions between coarse and fine cells keep their shape: a vertex there has to stay
    on the coarse cell's face (deal.II enforces this), so it can't follow the DEM.
    Real-ocean models (water cells above a fitted seafloor) aren't supported.
    """
    from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

    if len({int(i) for i in top_ids} - {AIR_ID}):
        raise ValueError("refitting topography isn't supported for models with sea cells yet")
    topo_cart = read_dem(dem_path, georef)                 # [easting, northing, elevation]
    dem = NearestNDInterpolator((topo_cart[:, :, 1].ravel(), topo_cart[:, :, 0].ravel()),
                                topo_cart[:, :, 2].ravel())
    n0, n1 = float(topo_cart[:, :, 1].min()), float(topo_cart[:, :, 1].max())
    e0, e1 = float(topo_cart[:, :, 0].min()), float(topo_cart[:, :, 0].max())
    blend = 0.05 * min(n1 - n0, e1 - e0) if blend_km is None else float(blend_km) * 1e3

    def weight(x, y):
        """1 well inside the DEM, 0 at its edge and beyond."""
        inset = np.minimum.reduce([np.asarray(x) - n0, n1 - np.asarray(x),
                                   np.asarray(y) - e0, e1 - np.asarray(y)])
        return np.clip(inset / max(blend, 1e-6), 0.0, 1.0)

    def target_z(x, y):
        return -np.maximum(dem(x, y), 0.0)

    surf = surface_faces(tria, top_ids)
    verts = surf.vertices()
    w = weight(verts[:, 0], verts[:, 1])
    covered = w > 0
    if not covered.any():
        raise ValueError("this DEM doesn't overlap the model. Is it in the model's coordinate frame?")
    misfit = target_z(verts[:, 0], verts[:, 1]) - verts[:, 2]
    shift = misfit * w
    band = max(3.0 * float(np.abs(shift).max()), 2000.0)
    current = LinearNDInterpolator(verts[:, :2], verts[:, 2])
    columns: dict[tuple, tuple] = {}

    def move(p):
        x, y, z = p
        key = (round(x, 3), round(y, 3))
        col = columns.get(key)
        if col is None:
            f = float(weight(x, y))
            z0 = float(current(x, y)) if f > 0 else float("nan")
            col = (z0, f * (float(target_z(x, y)) - z0)) if z0 == z0 else (0.0, 0.0)
            columns[key] = col
        z0, delta = col
        if delta == 0.0 or abs(z - z0) >= band:
            return [x, y, z]
        return [x, y, z + delta * (1.0 - abs(z - z0) / band)]

    tria.transform(move)
    degenerate = sum(1 for c in tria.active_cells() if c.measure() <= 0)
    if degenerate:
        raise ValueError(f"{degenerate} cells collapsed fitting this DEM: its relief is too "
                         "rough for the cells near the surface. Refine first, or smooth the DEM.")
    after = surface_faces(tria, top_ids).vertices()
    inside = weight(after[:, 0], after[:, 1]) >= 1.0
    left = target_z(after[:, 0], after[:, 1]) - after[:, 2]
    full = w >= 1.0
    stats = {
        "rms_before_m": float(np.sqrt(np.mean(misfit[full] ** 2))) if full.any() else 0.0,
        "rms_after_m": float(np.sqrt(np.mean(left[inside] ** 2))) if inside.any() else 0.0,
        "max_shift_m": float(np.abs(shift).max()), "band_m": band,
        "n_covered": int(covered.sum()), "n_vertices": len(verts),
    }
    log("topography refit: %d of %d surface vertices are within the DEM; misfit there %.1f -> %.1f m "
        "rms, largest shift %.1f m" % (stats["n_covered"], stats["n_vertices"], stats["rms_before_m"],
                                       stats["rms_after_m"], stats["max_shift_m"]))
    return stats


# --------------------------------------------------------------------------- #
# receivers
# --------------------------------------------------------------------------- #
def read_receivers(path: str) -> tuple[list[str], np.ndarray]:
    """``Dipole <name> 1 <north> <east> <depth>`` -> names, (n, 3)."""
    names, xyz = [], []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                xyz.append([float(v) for v in parts[3:6]])
            except ValueError:
                continue
            names.append(parts[1])
    return names, np.asarray(xyz, dtype="float64").reshape(-1, 3)


def _receiver_line(name: str, x: float, y: float, z: float) -> str:
    return "Dipole %s 1 %0.6e %0.6e %0.6e\n" % (name, x, y, z)


def _as_written(v: float) -> float:
    """The value after a trip through the receiver file's 6-decimal format."""
    return float("%0.6e" % v)


#: a station must be at least this far (m) from its cell's vertical faces, and from the lines
#: that splitting the cell up to FACE_LEVELS times would add
FACE_GAP_M = 1.0
FACE_LEVELS = 3


def off_cell_faces(xy, footprints, *, gap_m: float = FACE_GAP_M, levels: int = FACE_LEVELS):
    """Move stations that sit on a vertical cell face into their cell.

    A point on a face belongs to two cells.  GoFEM splits the mesh between processes and
    each one looks for the receivers in the cells it owns, so when the cells on either
    side belong to different processes neither may take the station: "Not all cells for
    receiver points were found", depending on how the mesh happened to be split.  GoFEM's
    "Refine cells around receivers" splits a station's cell, so the lines at k / 2^levels
    of it count as faces too.  A station closer than *gap_m* to one moves to 2 x *gap_m*
    from it, into the cell, along that axis only (as written to the receiver file).

    *footprints*: (n, 4) north0, north1, east0, east1 of each station's cell.  Returns the
    new (n, 2) positions and, per moved station, (index, d_north, d_east).
    """
    out = np.array(xy, dtype="float64")[:, :2].copy()
    moved = []
    parts = 2 ** int(levels)
    for k, (n0, n1, e0, e1) in enumerate(np.asarray(footprints, dtype="float64")):
        shift = [0.0, 0.0]
        for axis, (lo, hi) in enumerate(((n0, n1), (e0, e1))):
            v = _as_written(out[k, axis])
            step = (hi - lo) / parts
            j = int(np.clip(np.round((v - lo) / step), 0, parts))
            line = lo + j * step
            if abs(v - line) >= gap_m:
                continue
            if j == 0:
                side = 1.0                              # the cell's own faces: inwards
            elif j == parts:
                side = -1.0
            else:
                side = np.sign(v - line) or 1.0         # an inner line: away from it
            new = _as_written(line + side * 2.0 * gap_m)
            shift[axis] = new - out[k, axis]
            out[k, axis] = new
        if shift != [0.0, 0.0]:
            moved.append((k, shift[0], shift[1]))
    return out, moved


def surface_footprints(surf: SurfaceFaces, xy) -> np.ndarray:
    """(n, 4) north0, north1, east0, east1 of the surface cell under each station."""
    out = np.full((len(xy), 4), np.nan)
    for k, (x, y) in enumerate(np.asarray(xy, dtype="float64")):
        i = surf.locate(_as_written(x), _as_written(y))
        if i >= 0:
            out[k] = surf.x0[i], surf.x1[i], surf.y0[i], surf.y1[i]
    return out


def merge_stations(old_names, old_xy, new_names, new_xy, mode: str = "add", tol_m: float = 1.0):
    """Combine existing receivers with new stations.

    ``mode`` "add" keeps both (a station in both lists stays once, as long as it's in
    the same place); "replace" keeps only the new ones.  Returns
    ``(names, xy, is_new, notes)``.
    """
    new_xy = np.asarray(new_xy, dtype="float64").reshape(-1, 2)
    old_xy = np.asarray(old_xy, dtype="float64").reshape(-1, 2)
    if mode == "replace" or not len(old_names):
        return list(new_names), new_xy, np.ones(len(new_names), dtype=bool), []
    if mode != "add":
        raise ValueError(f"unknown station mode {mode!r}")
    where = {n: i for i, n in enumerate(old_names)}
    names, xy, notes = list(old_names), [p for p in old_xy], []
    is_new = [False] * len(names)
    for n, p in zip(new_names, new_xy):
        i = where.get(n)
        if i is None:
            names.append(n)
            xy.append(p)
            is_new.append(True)
        elif np.hypot(*(old_xy[i] - p)) <= tol_m:
            notes.append(f"{n} is already a receiver in the same place: kept once")
        else:
            raise ValueError(f"station {n} is already a receiver, {np.hypot(*(old_xy[i] - p)):.0f} m "
                             "away. Rename one of them, or check the coordinate frame.")
    return names, np.asarray(xy, dtype="float64").reshape(-1, 2), np.asarray(is_new, dtype=bool), notes


def project_to_surface(surf: SurfaceFaces, names, xy) -> np.ndarray:
    """Surface depth at each station, on x and y as the receiver file will store them
    (rounding can move a station up to a metre on a slope).  Raises if any are off the mesh."""
    out = np.empty((len(names), 3), dtype="float64")
    outside = []
    for k, (n, (x, y)) in enumerate(zip(names, np.asarray(xy, dtype="float64"))):
        x, y = _as_written(x), _as_written(y)
        i = surf.locate(x, y)
        if i < 0:
            outside.append(n)
            continue
        out[k] = x, y, surf.z_at(i, x, y)
    if outside:
        raise ValueError(f"{len(outside)} stations are outside the model: {', '.join(outside[:10])}"
                         + (" ..." if len(outside) > 10 else ""))
    return out


def receiver_positions(surf: SurfaceFaces, names, xy) -> np.ndarray:
    """(north, east, depth) to write for each station: on the surface,
    ``RECEIVER_OFFSET_M`` below it, rounded as the receiver file stores it."""
    on_surface = project_to_surface(surf, names, xy)
    return np.c_[on_surface[:, :2], [_as_written(z + RECEIVER_OFFSET_M) for z in on_surface[:, 2]]]


def station_cell_sizes(surf: SurfaceFaces, xy) -> np.ndarray:
    """Horizontal size (m) of the surface cell under each station; NaN if off the mesh."""
    out = []
    for x, y in np.asarray(xy, dtype="float64").reshape(-1, 2):
        i = surf.locate(x, y)
        out.append(max(surf.x1[i] - surf.x0[i], surf.y1[i] - surf.y0[i]) if i >= 0 else np.nan)
    return np.asarray(out, dtype="float64")


def verify_receivers(tria, names, written_xyz, *, top_ids=(AIR_ID,), above_m: float = 0.2) -> list:
    """Check stations against the mesh itself, at the depths in the receiver file:
    earth at the station, and air (or water) ``above_m`` higher.  Returns the failures."""
    import PyDealII.Release as dealii

    top = {int(i) for i in top_ids}
    mapping = dealii.MappingQGeneric(dim=3, spacedim=3, degree=1)
    bad = []
    for n, (x, y, z) in zip(names, written_xyz):
        try:
            at = tria.find_active_cell_around_point(dealii.Point([x, y, z]), mapping).material_id
            up = tria.find_active_cell_around_point(dealii.Point([x, y, z - above_m]), mapping).material_id
        except Exception as exc:  # noqa: BLE001
            bad.append((n, f"not found in the mesh ({exc})"))
            continue
        if at in top or up not in top:
            bad.append((n, f"material {at} at the station, {up} {above_m} m above"))
    return bad


# --------------------------------------------------------------------------- #
# new stations -> mesh frame
# --------------------------------------------------------------------------- #
def stations_in_model_frame(stations, georef) -> tuple[list[str], np.ndarray]:
    """EDI stations (``.lon``, ``.lat``, ``.station_id``) -> names, (north, east) in the
    model.  *georef* is a ``plotting.backends.georef.Georef``."""
    from pyproj import Transformer

    tr = Transformer.from_crs(4326, georef.crs, always_xy=True)
    lon = np.array([s.lon for s in stations], dtype="float64")
    lat = np.array([s.lat for s in stations], dtype="float64")
    east, north = tr.transform(lon, lat)
    return [s.station_id for s in stations], np.c_[np.asarray(north) - georef.y0,
                                                   np.asarray(east) - georef.x0]


def _check_project_frame(path: str, georef, receivers_path: str) -> str:
    """Refuse a setup project whose frame doesn't fit the model's receivers.  Nothing else
    would catch it: the stations would land on the mesh, just in the wrong place, and a
    project's frame changes if it's reused for another survey.  Returns a note if it
    can't tell, because the receivers or the project's EDIs aren't available."""
    from ..plotting.backends import georef as G
    from ..plotting.backends.model_source import GoFEMVtuSource
    from ..project import Project
    from . import edi_io

    name = os.path.basename(path)
    instead = "Fit the frame to the receivers' EDIs, or enter it by hand."
    if not receivers_path or not os.path.isfile(receivers_path):
        return " (not checked: no receivers)"
    edi_dir = Project.load(path).survey.edi_dir
    stations = edi_io.read_edi_headers(edi_dir) if edi_dir and os.path.exists(edi_dir) else []
    if not stations:
        return " (not checked: can't read the project's EDIs)"
    rec = GoFEMVtuSource().read_receivers(receivers_path)
    try:
        fit = G.fit_from_stations(rec, stations, georef.crs)
    except ValueError as exc:
        raise ValueError(f"{name} doesn't look like the project this model was built with: none of "
                         f"its stations are in the model's receivers. {instead}") from exc
    offset = math.hypot(fit.georef.x0 - georef.x0, fit.georef.y0 - georef.y0)
    if offset > _FRAME_LIMIT_M or fit.rms_m > _FRAME_LIMIT_M:
        raise ValueError(f"{name}'s frame doesn't fit this model: its origin is {offset / 1e3:.1f} km "
                         f"from where the receivers put it ({fit.rms_m:.0f} m rms). {instead}")
    return f", checked against {fit.n_matched} receivers"


def resolve_georef(spec: dict):
    """The model's frame, from ``spec["frame"]`` (``project`` / ``fit`` / ``file`` / ``manual``).
    Returns ``(georef, description)``."""
    from ..plotting.backends import georef as G
    from ..plotting.backends.model_source import GoFEMVtuSource
    from . import edi_io

    kind = spec.get("frame", "project")
    if kind == "project":
        path = spec.get("frame_project", "")
        if not path or not os.path.isfile(path):
            raise ValueError("choose the setup project (.mtinv.json) this model was built with")
        gr, _rec = G.from_mtinv_project(path)
        note = _check_project_frame(path, gr, spec.get("receivers", ""))
        return gr, f"from {os.path.basename(path)}: {G.crs_label(gr.crs)}{note}"
    if kind == "fit":
        if not spec.get("receivers") or not spec.get("frame_fit_edis"):
            raise ValueError("fitting the model's frame needs its receivers (Model page) and "
                             "their EDIs (Stations page)")
        rec = GoFEMVtuSource().read_receivers(spec["receivers"])
        fit = G.auto_fit(rec, edi_io.read_edi_headers(spec["frame_fit_edis"]))
        if fit.rms_m > _FRAME_LIMIT_M:
            raise ValueError(f"the receivers don't fit their EDIs in any standard frame (best "
                             f"{fit.rms_m:.0f} m rms). Use the setup project the model was built "
                             f"with, or enter the frame by hand.")
        return fit.georef, (f"fitted to {fit.n_matched} receivers: {G.crs_label(fit.georef.crs)}, "
                            f"{fit.rms_m:.2f} m rms")
    if kind == "file":
        from . import projection as P

        path = spec.get("frame_file", "")
        if not path or not os.path.isfile(path):
            raise ValueError("choose the model frame file (model_frame.json) the mesh was built in")
        gr = G.from_frame(P.load_frame(path))
        return gr, f"from {os.path.basename(path)}: {G.crs_label(gr.crs)}"
    if kind == "manual":
        gr = G.manual(spec.get("frame_crs", ""), float(spec.get("frame_x0", 0.0)),
                      float(spec.get("frame_y0", 0.0)))
        if not gr.valid:
            raise ValueError("enter the model's CRS")
        return gr, f"entered by hand: {G.crs_label(gr.crs)}"
    raise ValueError(f"unknown frame source {kind!r}")


# --------------------------------------------------------------------------- #
# plan-view slices
# --------------------------------------------------------------------------- #
@dataclass
class ModelCells:
    """The earth cells, as plain arrays (the mesh frame; z positive down)."""
    north: np.ndarray
    east: np.ndarray
    half_north: np.ndarray
    half_east: np.ndarray
    z_top: np.ndarray
    z_bottom: np.ndarray
    log10_rho: np.ndarray                   # NaN where the .materials doesn't list the id


def model_cells(tria, materials: dict[int, Material] | None = None) -> ModelCells:
    cells, xyz, mat = _cell_arrays(tria)
    earth = np.flatnonzero(mat != AIR_ID)
    half = _half_sizes(cells, earth)
    rho = {i: -math.log10(m.conductivity) for i, m in (materials or {}).items() if m.conductivity > 0}
    return ModelCells(xyz[earth, 0], xyz[earth, 1], half[:, 0], half[:, 1],
                      xyz[earth, 2] - half[:, 2], xyz[earth, 2] + half[:, 2],
                      np.array([rho.get(int(i), np.nan) for i in mat[earth]], dtype="float64"))


def slice_depths(cells: ModelCells, min_step_m: float = 100.0, rel_step: float = 0.05) -> np.ndarray:
    """Depths worth slicing at, from the cell centres, spaced more widely with depth."""
    centres = np.unique(np.round(0.5 * (cells.z_top + cells.z_bottom)))
    out, i = [], 0
    while i < len(centres):
        out.append(centres[i])
        i = int(np.searchsorted(centres, centres[i] + max(min_step_m, rel_step * abs(centres[i])), side="left"))
    return np.array(out, dtype="float64")


def plan_slice(cells: ModelCells, depth_m: float) -> tuple[np.ndarray, np.ndarray]:
    """The cells at a depth below sea level: ``(n, 4)`` north0, north1, east0, east1 in
    metres, and their log10 resistivity."""
    at = (cells.z_top <= depth_m) & (depth_m < cells.z_bottom)
    return (np.c_[cells.north[at] - cells.half_north[at], cells.north[at] + cells.half_north[at],
                  cells.east[at] - cells.half_east[at], cells.east[at] + cells.half_east[at]],
            cells.log10_rho[at])


# --------------------------------------------------------------------------- #
# the whole job
# --------------------------------------------------------------------------- #
def output_paths(out_dir: str, stem: str = "model_refined") -> dict[str, str]:
    return {
        "tria": os.path.join(out_dir, f"{stem}.tria"),
        "vtu": os.path.join(out_dir, f"{stem}.vtu"),
        "materials": os.path.join(out_dir, f"{stem}.materials"),
        "mask": os.path.join(out_dir, f"{stem}.mask"),
        "receivers": os.path.join(out_dir, "receivers_refined-projected.csv"),
        "paraview": os.path.join(out_dir, "receivers_refined-paraview.csv"),
    }


def guess_companions(tria_path: str) -> dict[str, str]:
    """A model's ``.materials`` (same name) and receivers (next to it, or a level or
    two up, where inversion results sit below the mesh)."""
    out = {"materials": "", "receivers": ""}
    stem = os.path.splitext(tria_path)[0]
    if os.path.isfile(stem + ".materials"):
        out["materials"] = stem + ".materials"
    here = os.path.dirname(os.path.abspath(tria_path))
    for d in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        for name in ("receivers_refined-projected.csv", "receivers_projected.csv"):
            if os.path.isfile(os.path.join(d, name)):
                out["receivers"] = os.path.join(d, name)
                return out
    return out


def guess_domain_box(tria_path: str) -> tuple[str, str]:
    """``(box, file)`` from an ``input_inversion*.prm`` near the model (empty if none).  In
    each folder, plain ``input_inversion.prm`` comes first, then the newest variant."""
    here = os.path.dirname(os.path.abspath(tria_path))
    for d in (here, os.path.dirname(here), os.path.dirname(os.path.dirname(here))):
        found = glob.glob(os.path.join(glob.escape(d), "input_inversion*.prm"))
        found.sort(key=lambda p: (os.path.basename(p) != "input_inversion.prm", -os.path.getmtime(p)))
        for path in found:
            with open(path, "r", errors="replace") as fh:
                for line in fh:
                    if "Active domain box" in line and "=" in line:
                        value = line.split("=", 1)[1].split("#")[0].strip()
                        if value:
                            return value, path
    return "", ""


def existing_outputs(out_dir: str, stem: str = "model_refined") -> list[str]:
    """Output files that a build into *out_dir* would overwrite."""
    return [p for p in output_paths(out_dir, stem).values() if os.path.isfile(p)]


def spec_from_info(info) -> dict:
    """Job description for :func:`run`, from a ``project.RefineInfo``."""
    spec = dataclasses.asdict(info)
    spec["refit_dem"] = info.refit_dem if info.refit else ""
    spec["blend_km"] = info.blend_km or None
    spec["fixed_ids"] = list(info.fixed_ids) if info.write_mask else None
    spec["smooth_fixed_ids"] = list(info.fixed_ids) or None
    return spec


def run(spec: dict, out_dir: str = "", log=print, write: bool = True) -> dict:
    """Refine a model.  With ``write=False`` nothing is saved (an estimate that runs
    the real refinement, on the model as loaded from disk).

    *spec*: ``tria``, ``materials``, ``receivers`` (optional); ``station_mode`` ("none" /
    "add" / "replace") with ``stations_edi`` and the frame keys of :func:`resolve_georef`;
    ``rules`` (:class:`RefineRule` dicts); ``refit_dem`` and ``blend_km`` (optional);
    ``fixed_ids`` (optional; gives a mask); ``max_cells`` (optional); ``stem`` (optional);
    ``smooth`` (optional) with ``smooth_mode``, ``smooth_horizontal``, ``smooth_vertical``,
    see :func:`smooth_refined`; ``smooth_fixed_ids``: the ids that aren't inverted, for
    smoothing and coarsening (default: guessed from the ids).

    Stations are checked against the finished mesh before anything is written, and a
    failure stops the job, so a bad receiver file is never produced.
    """
    from . import edi_io

    log(f"loading {os.path.basename(spec['tria'])}")
    tria = load_tria(spec["tria"])
    start_cells = tria.n_active_cells()
    materials = read_materials(spec["materials"]) if spec.get("materials") else {}
    log(f"{start_cells:,} cells, {len(materials):,} materials")
    top_ids = [AIR_ID, *water_ids(materials)]
    if spec.get("refit_dem") and len(top_ids) > 1:
        raise ValueError("refitting topography isn't supported for models with sea cells yet")
    if write and os.path.abspath(output_paths(out_dir, spec.get("stem") or "model_refined")["tria"]) \
            == os.path.abspath(spec["tria"]):
        raise ValueError("this would overwrite the input model: choose another output folder or name")
    if materials:
        _c, _x, mat = _cell_arrays(tria)
        missing = sorted(set(int(i) for i in np.unique(mat)) - set(materials))
        if missing:
            raise ValueError(f"{len(missing)} material ids in the .tria aren't in the .materials "
                             f"(e.g. {missing[:5]}). Are these from the same model and iteration?")

    # stations --------------------------------------------------------------- #
    old_names, old_xyz = [], np.zeros((0, 3))
    if spec.get("receivers"):
        old_names, old_xyz = read_receivers(spec["receivers"])
    names, xy, is_new, frame_note = list(old_names), old_xyz[:, :2], np.zeros(len(old_names), dtype=bool), ""
    mode = spec.get("station_mode", "none")
    georef = None
    if mode != "none" or (spec.get("refit_dem") and is_lonlat_dem(spec["refit_dem"])):
        georef, frame_note = resolve_georef(spec)
        log("model frame " + frame_note)
    if mode != "none":
        stations = edi_io.read_edi_headers(spec["stations_edi"])
        new_names, new_xy = stations_in_model_frame(stations, georef)
        names, xy, is_new, notes = merge_stations(old_names, old_xyz[:, :2], new_names, new_xy, mode)
        for note in notes:
            log("  " + note)
        log(f"stations: {int(is_new.sum())} new, {len(names) - int(is_new.sum())} kept")

    surf = surface_faces(tria, top_ids)
    if len(names):
        project_to_surface(surf, names, xy)                # fail now if any are off the mesh
    size_before = station_cell_sizes(surf, xy) if len(names) else np.zeros(0)

    # refine ------------------------------------------------------------------ #
    smooth = bool(spec.get("smooth")) and any(rule_from_dict(r).enabled for r in spec.get("rules", []))
    if smooth and not materials:
        raise ValueError("smoothing needs the model's .materials file")
    before = snapshot_cells(tria) if smooth else None
    model_fixed = spec.get("smooth_fixed_ids") or (guess_fixed_ids(materials) if materials else [AIR_ID])
    results = []
    for k, raw in enumerate(spec.get("rules", [])):
        rule = rule_from_dict(raw)
        if not rule.enabled:
            continue
        log(f"rule {k + 1} ({rule.kind})")
        station_xy = None
        if rule.kind == "stations":
            station_xy = xy[is_new] if rule.stations == "new" else xy
            if not len(station_xy):
                log("  no stations to refine around")
        results.append({"rule": k, "action": rule.action, **apply_rule(
            tria, rule, surf, station_xy=station_xy, exclude_ids=top_ids,
            max_cells=spec.get("max_cells"), materials=materials, fixed_ids=model_fixed, log=log)})
    coarsened = [r for r in results if r["action"] == "coarsen"]
    new_materials = {m.id: m for r in coarsened for m in r["new_ids"].values()}
    # Only balance a mesh the rules changed: with no refinement, the model stays as it was run.
    balanced = balance_levels(tria, log=log) if any(r["added"] for r in results) else 0
    smoothed = None
    if smooth and any(r["added"] for r in results):
        smoothed = smooth_refined(
            tria, materials, before, mode=spec.get("smooth_mode", "relative"),
            horizontal=float(spec.get("smooth_horizontal", 0.25)),
            vertical=float(spec.get("smooth_vertical", 0.25)),
            fixed_ids=model_fixed, exclude_keys=[k for r in coarsened for k in r["keys"]], log=log)
        new_materials.update(smoothed["new_ids"])

    refit = None
    dem_check = None
    if spec.get("refit_dem"):
        dem_check = check_dem_frame(surf, spec["refit_dem"], georef=georef)    # the surface before refining
        log(("WARNING: " if dem_check.is_problem else "") + dem_check.message)
        refit = refit_topography(tria, spec["refit_dem"], top_ids=top_ids,
                                 blend_km=spec.get("blend_km"), log=log, georef=georef)

    # stations on the finished surface ---------------------------------------- #
    surf = surface_faces(tria, top_ids)
    written, moved, failures, nudged = np.zeros((0, 3)), None, [], []
    if len(names):
        xy, nudged = off_cell_faces(xy, surface_footprints(surf, xy))
        for k, dn, de in nudged:
            log(f"{names[k]} was on a cell face (GoFEM may not find it): moved {dn:+.1f} m north, "
                f"{de:+.1f} m east")
        written = receiver_positions(surf, names, xy)
        if len(old_names) and refit is None and not any(r["merged"] for r in coarsened):
            # refining alone can't move the surface, so the kept stations mustn't move
            where = {n: i for i, n in enumerate(old_names)}
            common = [(i, where[n]) for i, n in enumerate(names) if n in where]
            if common:
                moved = float(max(abs(written[i, 2] - old_xyz[j, 2]) for i, j in common))
                log(f"existing stations moved up to {moved:.3f} m vertically")
                if moved > 1.0:
                    log("WARNING: more than expected without a refit. Do the receivers belong "
                        "to this model?")
        log("checking every station against the mesh")
        failures = verify_receivers(tria, names, written, top_ids=top_ids)
        if failures:
            raise ValueError(f"{len(failures)} stations aren't just below the surface, so nothing "
                             f"was written: " + "; ".join(f"{n}: {w}" for n, w in failures[:5]))
        log(f"{len(names)} stations have earth at their depth and air or water above")

    result = {
        "n_cells_before": start_cells, "n_cells": tria.n_active_cells(),
        "smallest_m": tria.minimal_cell_diameter(), "largest_m": tria.maximal_cell_diameter(),
        "rules": results, "balanced_cells": balanced, "smoothed": smoothed, "refit": refit, "dem_check": dem_check, "names": names, "is_new": is_new,
        "station_size_before_m": size_before,
        "station_size_m": station_cell_sizes(surf, xy) if len(names) else np.zeros(0),
        "receivers_moved_m": moved, "frame": frame_note,
        "off_face": [(names[k], dn, de) for k, dn, de in nudged],
        "cells": model_cells(tria, materials),
    }
    if not write:
        return result

    result["paths"] = write_refined(tria, out_dir, spec.get("stem") or "model_refined",
                                    materials_path=spec.get("materials", ""),
                                    new_materials=new_materials or None,
                                    fixed_ids=spec.get("fixed_ids"), top_ids=top_ids,
                                    names=names, written=written, log=log)
    return result


def write_refined(tria, out_dir: str, stem: str = "model_refined", *, materials_path: str = "",
                  fixed_ids=None, top_ids=(AIR_ID,), names=(), written=None, new_materials=None,
                  log=print) -> dict:
    """Write a refined model: ``<stem>.tria`` / ``.vtu``, a copy of the ``.materials``
    (refining keeps every id) with *new_materials* (id -> :class:`Material`, from
    :func:`smooth_refined`) added at the end, a ``.mask`` when *fixed_ids* is given (those ids and
    *top_ids* fixed, the rest free), and the receivers when there are stations.
    *written* is the stations' (north, east, depth) as checked by
    :func:`verify_receivers`.  Returns the paths written."""
    os.makedirs(out_dir, exist_ok=True)
    paths = output_paths(out_dir, stem)
    tria.save(paths["tria"])
    tria.write(paths["vtu"], "vtu")
    if materials_path:
        same = os.path.abspath(materials_path) == os.path.abspath(paths["materials"])
        if new_materials and same:
            raise ValueError("this would change the input .materials: choose another output folder or name")
        if not same:
            shutil.copyfile(materials_path, paths["materials"])
        if new_materials:
            with open(paths["materials"], "rb") as fh:
                data = fh.read()
            ends_with_newline = not data or data.endswith(b"\n")
            with open(paths["materials"], "a") as fh:
                if not ends_with_newline:
                    fh.write("\n")
                for m in new_materials.values():
                    fh.write(f"{m.id}\t{m.name}\t{m.conductivity:.8g}\t0\n")
    else:
        paths.pop("materials")
    if fixed_ids is not None:
        fixed = {int(i) for i in fixed_ids} | set(top_ids)
        with open(paths["mask"], "w") as fh:
            for c in tria.active_cells():
                fh.write("%i %i %i\n" % (c.level(), c.index(), 0 if c.material_id in fixed else 1))
    else:
        paths.pop("mask")
    if len(names):
        with open(paths["receivers"], "w") as fh:
            for n, (x, y, z) in zip(names, written):
                fh.write(_receiver_line(n, x, y, z))
        np.savetxt(paths["paraview"], written - [0.0, 0.0, RECEIVER_OFFSET_M], delimiter=",", fmt="%s")
    else:
        paths.pop("receivers")
        paths.pop("paraview")
    log("wrote " + ", ".join(os.path.basename(p) for p in paths.values()))
    return paths
