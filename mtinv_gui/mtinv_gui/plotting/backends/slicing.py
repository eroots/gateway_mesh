"""Cut a :class:`~.model_source.ModelGrid` into 2-D pieces for plotting.

Everything here returns *true cell polygons* -- the cut of each model cell
with the slicing surface, coloured by that cell's value -- never an
interpolation, so a figure shows the model exactly as the inversion has it.

Depth references (``reference``):
  * ``"sea_level"`` -- a flat cut at elevation ``-depth`` (depth below sea
    level; negative depths are above sea level).
  * ``"surface"``  -- a cut at ``depth`` below the model's own surface
    (:meth:`ModelGrid.surface`), i.e. it follows the topography.

Crinkle slices (``crinkle=True``, as in ParaView): instead of cutting cells,
draw the whole map footprint of every cell the depth passes through.  Cells
warped by the topography fit, or split by hanging nodes, cut into polygons
that don't quite meet (slivers, gaps, overlaps); footprints tile the map.

Results are cached per (model, map transform, parameters) so flipping between
depths or redrawing a figure does not re-cut the mesh.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

REFERENCES = {"sea_level": "below sea level", "surface": "below surface"}


@dataclass
class PolySlice:
    """Cell polygons in display coordinates.

    ``polys[i]`` is a ``(k, 2)`` vertex array; ``values[i]`` its log10(rho).
    """
    polys: list
    values: np.ndarray

    @property
    def n(self) -> int:
        return len(self.polys)

    def bounds(self) -> tuple[float, float, float, float]:
        if not self.polys:
            return (0.0, 0.0, 0.0, 0.0)
        allv = np.concatenate(self.polys)
        return (float(allv[:, 0].min()), float(allv[:, 0].max()),
                float(allv[:, 1].min()), float(allv[:, 1].max()))


@dataclass
class PlanSlice(PolySlice):
    depth_m: float = 0.0
    reference: str = "sea_level"


def depth_label(depth_m: float, reference: str, unit: str = "km") -> str:
    """``"5 km below sea level"`` / ``"250 m above sea level"`` / ..."""
    ref = REFERENCES.get(reference, reference)
    if reference == "sea_level" and depth_m < 0:
        ref = "above sea level"
    d = abs(depth_m)
    if unit == "km":
        txt = f"{d / 1000.0:g} km"
    else:
        txt = f"{d:g} m"
    return f"{txt} {ref}"


# --------------------------------------------------------------------------- #
class _LRU:
    def __init__(self, size: int = 64):
        self.size = size
        self._d: OrderedDict = OrderedDict()

    def get(self, key):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key, value):
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.size:
            self._d.popitem(last=False)

    def clear(self):
        self._d.clear()


_cache = _LRU()


def clear_cache() -> None:
    _cache.clear()


def _polys_from_polydata(pd) -> tuple[list, np.ndarray]:
    """``(list of (k, 3) point-index arrays, cell ids)`` for every polygon."""
    from vtkmodules.util.numpy_support import vtk_to_numpy

    cells = pd.GetPolys()
    if cells is None or cells.GetNumberOfCells() == 0:
        return [], np.zeros(0, dtype=int)
    offsets = vtk_to_numpy(cells.GetOffsetsArray())
    conn = vtk_to_numpy(cells.GetConnectivityArray())
    polys = [conn[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
    # polys come after verts + lines in VTK's cell numbering
    n_before = pd.GetNumberOfVerts() + pd.GetNumberOfLines()
    return polys, np.arange(n_before, n_before + len(polys))


def _surface_offset(model) -> np.ndarray:
    """Per-point ``z - surface(x, y)`` on the earth grid (cached on *model*)."""
    off = getattr(model, "_surface_offset", None)
    if off is None:
        e = model.earth
        pts = np.asarray(e.points)
        off = pts[:, 2] - model.surface()(pts[:, 0], pts[:, 1])
        model._surface_offset = off
    return off


def plan_slice(model, maptr, depth_m: float, reference: str = "sea_level",
               crinkle: bool = False) -> PlanSlice:
    """Horizontal (``sea_level``) or terrain-following (``surface``) cut;
    *crinkle*: whole cells instead (see :func:`crinkle_plan_slice`)."""
    if crinkle:
        return crinkle_plan_slice(model, maptr, depth_m, reference)
    key = ("plan", id(model), maptr.key, float(depth_m), reference)
    hit = _cache.get(key)
    if hit is not None:
        return hit

    earth = model.earth
    if reference == "surface":
        work = earth.copy(deep=False)
        work.point_data["_dsurf"] = _surface_offset(model)
        cut = work.contour(isosurfaces=[-float(depth_m)], scalars="_dsurf")
    else:
        cut = earth.slice(normal="z", origin=(0.0, 0.0, -float(depth_m)))

    idx, cell_ids = _polys_from_polydata(cut)
    if idx:
        pts = np.asarray(cut.points)
        E, N = maptr.forward(pts[:, 0], pts[:, 1])
        exy = np.c_[E, N]
        polys = [exy[i] for i in idx]
        values = np.asarray(cut.cell_data["log10_rho"])[cell_ids]
    else:
        polys, values = [], np.zeros(0)
    out = PlanSlice(polys=polys, values=np.asarray(values, dtype="float64"),
                    depth_m=float(depth_m), reference=reference)
    _cache.put(key, out)
    return out


def _cell_columns(model) -> dict:
    """Per earth cell: map footprint (4 corners, local frame), and per corner
    column the point ids of its bottom / top (cached on *model*).

    Corners are matched by position, not by VTK ordering (hexahedra and
    voxels order them differently): sorted by angle around the cell centre,
    the two corners of each vertical edge come out next to each other."""
    cached = getattr(model, "_cell_columns", None)
    if cached is not None:
        return cached
    from .model_source import _hex_connectivity

    e = model.earth
    pts = np.asarray(e.points)
    conn = _hex_connectivity(e)
    xy = pts[conn][:, :, :2]
    d = xy - xy.mean(1, keepdims=True)
    order = np.argsort(np.arctan2(d[:, :, 1], d[:, :, 0]), axis=1, kind="stable")
    ids = np.take_along_axis(conn, order, axis=1).reshape(-1, 4, 2)   # (n, column, pair)
    z = pts[ids][..., 2]
    lo = np.where(z[..., 0] <= z[..., 1], ids[..., 0], ids[..., 1])
    hi = np.where(z[..., 0] <= z[..., 1], ids[..., 1], ids[..., 0])
    cached = dict(foot=pts[lo][..., :2], lo=lo, hi=hi,
                  vals=np.asarray(e.cell_data["log10_rho"], dtype="float64"))
    model._cell_columns = cached
    return cached


def crinkle_plan_slice(model, maptr, depth_m: float, reference: str = "sea_level") -> PlanSlice:
    """Every earth cell the depth passes through, drawn flat as its map
    footprint (a crinkle slice).

    Footprints of neighbouring cells meet exactly, so there are no slivers or
    gaps.  Where a column has two cells reaching the depth (a tilted layer
    boundary, or a depth exactly on a face), the one that contains the depth
    at its footprint's centre is drawn on top (on a face: the cell below,
    as :func:`sample_points` does)."""
    key = ("plan_crinkle", id(model), maptr.key, float(depth_m), reference)
    hit = _cache.get(key)
    if hit is not None:
        return hit
    cc = _cell_columns(model)
    if reference == "surface":
        h = _surface_offset(model)          # z - surface, per point
    else:
        h = np.asarray(model.earth.points)[:, 2]
    zc = -float(depth_m)
    hlo, hhi = h[cc["lo"]], h[cc["hi"]]      # (n, 4) bottom / top of each corner column
    # a cell whose bottom only touches the depth is left out: the one below has it
    sel = np.where((hlo.min(1) < zc) & (hhi.max(1) >= zc))[0]
    bot, top = hlo[sel].mean(1), hhi[sel].mean(1)
    centre = (bot < zc) & (zc <= top)
    # draw order: cells containing the depth at their centre last (on top);
    # the rest by how far their centre range is from the depth
    miss = np.where(centre, 0.0, np.minimum(np.abs(bot - zc), np.abs(top - zc)))
    order = np.lexsort((-miss, centre))
    sel = sel[order]
    if len(sel):
        foot = cc["foot"][sel]                                   # (m, 4, 2) local
        E, N = maptr.forward(foot[..., 0].ravel(), foot[..., 1].ravel())
        exy = np.c_[E, N].reshape(-1, 4, 2)
        polys = list(exy)
        values = cc["vals"][sel]
    else:
        polys, values = [], np.zeros(0)
    out = PlanSlice(polys=polys, values=values, depth_m=float(depth_m), reference=reference)
    _cache.put(key, out)
    return out


# --------------------------------------------------------------------------- #
def sample_points(model, x, y, z) -> np.ndarray:
    """log10(rho) of the earth cell containing each local point (NaN where
    there is none -- air or outside the mesh).

    Points that land exactly on a face (e.g. a flat surface at z = 0) are
    retried 0.5 m lower, so a depth-0 slice is not all NaN.
    """
    earth = model.earth
    vals = np.asarray(earth.cell_data["log10_rho"])
    pts = np.c_[np.ravel(x), np.ravel(y), np.ravel(z)].astype("float64")
    cid = np.asarray(earth.find_containing_cell(pts))
    miss = cid < 0
    if miss.any():
        retry = pts[miss] - np.array([0.0, 0.0, 0.5])
        cid[miss] = np.asarray(earth.find_containing_cell(retry))
    out = np.full(len(pts), np.nan)
    ok = cid >= 0
    out[ok] = vals[cid[ok]]
    return out.reshape(np.shape(x))


def sample_depth_grid(model, maptr, depth_m: float, reference: str,
                      E: np.ndarray, N: np.ndarray) -> np.ndarray:
    """log10(rho) at display-CRS points ``(E, N)`` and the given depth."""
    x, y = maptr.inverse(E, N)
    if reference == "surface":
        z = model.surface()(x, y) - float(depth_m)
    else:
        z = np.full(np.shape(x), -float(depth_m))
    return sample_points(model, x, y, z)
