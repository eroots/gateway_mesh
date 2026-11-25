"""The projected DEM as the mesh build uses it: one or two grids, and where the sea is.

**Two grids.** ``DEM_projected_gridded.npz`` holds the detailed grid around the survey
(``arr_0``) and, optionally, a coarse ``regional`` grid that reaches across the mesh
padding: one grid fine enough for the survey and wide enough for the padding would be
far too big.  A point inside the detailed grid's extent reads the detailed grid, any
other point the regional one.

**Where the sea is.** Below sea level isn't enough: land below sea level (Death Valley,
the shores of the Dead Sea, polders) must stay land, at its own elevation, with air above
it.  ``sea_rule`` decides:

* ``"connected"`` (default): below sea level *and* connected to the open sea.  On the
  coarsest grid that means reaching the edge of the DEM, which should extend out to sea;
  a below-sea-level area of the detailed grid is sea where it overlaps sea in the
  regional grid.  Areas closer than ``gap_km`` (default 10 km) count as connected: a DEM
  doesn't resolve a narrow strait or a fjord's mouth (at 3 km, San Francisco Bay looks
  enclosed), and true land below sea level lies much further from the sea (Death Valley,
  the Dead Sea, the Salton Sea).
* ``"all"``: everything below sea level is sea (what the setup GUI did up to 0.0.55).

Polygons override the rule: ``land_polygons`` are never sea; ``sea_polygons`` are sea
wherever they're below sea level (an enclosed sea, or a polder the rule takes for sea when
a channel joins it to the ocean).  They're shapely geometries in the DEM's
``(easting, northing)`` frame; ``projection.read_polygons`` reads them from a file.

The ocean code (:mod:`.ocean`) treats elevation < 0 as water, so it is given *ocean
views* of the grids, in which land below sea level reads ``LAND_VIEW_M``.  The real
elevations are only changed in those views; the topography fit uses the real ones.

Every lookup takes mesh-frame points ``[[north, east], ...]`` (the mesh has x = north),
as ``mesh.py`` always has.  For a single grid with no land below sea level everything here
gives exactly what the build did before this module existed.
"""
from __future__ import annotations

import numpy as np

from . import ocean as O

SEA_RULES = {"connected": "Below Sea Level and Connected to the Open Sea",
             "all": "Everything Below Sea Level"}
#: what land below sea level reads in the ocean views (above 0, so no coastline around it)
LAND_VIEW_M = 1.0
_EIGHT = np.ones((3, 3), dtype=bool)


def grid_extent(grid) -> tuple[float, float, float, float]:
    """``(east_min, east_max, north_min, north_max)`` of a projected DEM grid."""
    return (float(np.nanmin(grid[:, :, 0])), float(np.nanmax(grid[:, :, 0])),
            float(np.nanmin(grid[:, :, 1])), float(np.nanmax(grid[:, :, 1])))


def file_extent(path: str) -> tuple[float, float, float, float]:
    """The extent of everything in a projected DEM file (both grids)."""
    with np.load(path) as npz:
        boxes = [grid_extent(npz[k]) for k in ("arr_0", "regional") if k in npz.files]
    return (min(b[0] for b in boxes), max(b[1] for b in boxes),
            min(b[2] for b in boxes), max(b[3] for b in boxes))


def shortfall(extent, bounds) -> dict[str, float]:
    """How far (m) *extent* stops short of *bounds* on each side; all zero when it covers
    them.  Both are ``(east_min, east_max, north_min, north_max)``."""
    e0, e1, n0, n1 = bounds
    return {"west": max(0.0, extent[0] - e0), "east": max(0.0, e1 - extent[1]),
            "south": max(0.0, extent[2] - n0), "north": max(0.0, n1 - extent[3])}


def shortfall_text(short: dict[str, float]) -> str:
    """"40 km to the west, 12 km to the north" (empty when nothing is short)."""
    return ", ".join(f"{v / 1e3:.0f} km to the {side}" for side, v in short.items() if v > 0)


def _inside(extent, east, north):
    return (east >= extent[0]) & (east <= extent[1]) & (north >= extent[2]) & (north <= extent[3])


def _nearest_index(grid, east, north):
    """Nearest (row, col) of a regular grid for each point (clipped to the grid)."""
    x, y = grid[0, :, 0], grid[:, 0, 1]

    def idx(axis, v):
        if len(axis) < 2 or axis[-1] == axis[0]:
            return np.zeros(np.shape(v), dtype=int)
        k = np.rint((np.asarray(v) - axis[0]) / (axis[-1] - axis[0]) * (len(axis) - 1))
        return np.clip(k, 0, len(axis) - 1).astype(int)

    return idx(y, north), idx(x, east)


def _spacing(grid) -> float:
    """The smaller grid step (m) of a regular grid."""
    dx = abs(float(grid[0, 1, 0] - grid[0, 0, 0])) if grid.shape[1] > 1 else np.inf
    dy = abs(float(grid[1, 0, 1] - grid[0, 0, 1])) if grid.shape[0] > 1 else np.inf
    return max(min(dx, dy), 1e-9) if np.isfinite(min(dx, dy)) else 1.0


def _contains(geom, grid, where):
    """``where`` & inside *geom*, tested only where it matters (and inside its bounds)."""
    out = np.zeros(where.shape, dtype=bool)
    if geom is None or geom.is_empty or not where.any():
        return out
    import shapely.vectorized

    x0, y0, x1, y1 = geom.bounds
    e, n = grid[:, :, 0], grid[:, :, 1]
    cand = where & (e >= x0) & (e <= x1) & (n >= y0) & (n <= y1)
    if cand.any():
        out[cand] = shapely.vectorized.contains(geom, e[cand], n[cand])
    return out


def _outside_runs(pts, extent):
    """The parts of a polyline outside *extent*, as separate open lines."""
    out_mask = ~_inside(extent, pts[:, 0], pts[:, 1])
    runs, start = [], None
    for k, flag in enumerate(out_mask):
        if flag and start is None:
            start = k
        elif not flag and start is not None:
            runs.append(pts[start:k])
            start = None
    if start is not None:
        runs.append(pts[start:])
    return [r for r in runs if len(r) > 1]


class MeshDem:
    """See the module docstring.  *grids*: projected DEM arrays ``(ny, nx, 3)`` of
    (easting, northing, elevation), detailed first."""

    def __init__(self, grids, *, sea_rule: str = "connected", gap_km: float = 10.0,
                 land_polygons=None, sea_polygons=None):
        self.grids = [np.asarray(g, dtype="float64") for g in grids if g is not None]
        if not self.grids:
            raise ValueError("no DEM grid")
        if sea_rule not in SEA_RULES:
            raise ValueError(f"unknown sea rule {sea_rule!r} (one of {', '.join(SEA_RULES)})")
        self.sea_rule = sea_rule
        self.gap_km = max(float(gap_km), 0.0)
        self.extents = [grid_extent(g) for g in self.grids]
        self.below = [np.isfinite(g[:, :, 2]) & (g[:, :, 2] < 0.0) for g in self.grids]
        self.sea = self._sea_masks(land_polygons, sea_polygons)
        self.inland_masks = [b & ~s for b, s in zip(self.below, self.sea)]
        self.ocean_grids = []
        for g, inland in zip(self.grids, self.inland_masks):
            if inland.any():
                g = g.copy()
                g[:, :, 2][inland] = LAND_VIEW_M
            self.ocean_grids.append(g)
        self._interps: list = [None] * len(self.grids)
        self._raw = None
        self._flat = None

    @classmethod
    def load(cls, path: str, **kwargs) -> "MeshDem":
        """From a projected DEM file (``arr_0``, and ``regional`` if it has one)."""
        with np.load(path) as npz:
            grids = [npz["arr_0"]] + ([npz["regional"]] if "regional" in npz.files else [])
        return cls(grids, **kwargs)

    # ---- the sea ------------------------------------------------------------ #
    def _sea_masks(self, land_geom, sea_geom):
        from scipy import ndimage

        n = len(self.grids)
        masks: list = [None] * n
        for i in reversed(range(n)):                    # coarsest first: it seeds the finer ones
            g, below = self.grids[i], self.below[i]
            if self.sea_rule == "all":
                sea = below.copy()
            else:
                joined = below
                steps = int(np.ceil(self.gap_km * 1e3 / (2.0 * _spacing(g)))) if self.gap_km > 0 else 0
                if steps:                       # each side grows half the gap
                    joined = ndimage.binary_dilation(below, structure=_EIGHT, iterations=steps)
                labels, _count = ndimage.label(joined, structure=_EIGHT)
                if i == n - 1:
                    seeds = np.r_[labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]]
                else:
                    rows, cols = np.nonzero(below)
                    e, nn = g[rows, cols, 0], g[rows, cols, 1]
                    coarse = self.grids[i + 1]
                    inside = _inside(self.extents[i + 1], e, nn)
                    ci, cj = _nearest_index(coarse, e[inside], nn[inside])
                    hit = masks[i + 1][ci, cj]
                    seeds = labels[rows[inside][hit], cols[inside][hit]]
                seeds = np.unique(seeds)
                sea = below & np.isin(labels, seeds[seeds > 0])
            sea &= ~_contains(land_geom, g, sea)
            sea |= _contains(sea_geom, g, below & ~sea)
            masks[i] = sea
        return masks

    @property
    def topo_cart(self):
        """The detailed grid (what a single-grid file holds)."""
        return self.grids[0]

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(east_min, east_max, north_min, north_max)`` of all the grids together."""
        return (min(b[0] for b in self.extents), max(b[1] for b in self.extents),
                min(b[2] for b in self.extents), max(b[3] for b in self.extents))

    def shortfall(self, bounds) -> dict[str, float]:
        """How far the DEM stops short of the mesh *bounds* (see :func:`shortfall`)."""
        return shortfall(self.extent, bounds)

    # ---- point lookups -------------------------------------------------------- #
    def _interp(self, i):
        if self._interps[i] is None:
            from scipy.interpolate import NearestNDInterpolator

            g = self.grids[i]
            self._interps[i] = NearestNDInterpolator(
                (g[:, :, 1].ravel(), g[:, :, 0].ravel()), g[:, :, 2].ravel())
        return self._interps[i]

    def _split(self, pts):
        """Which grid each mesh-frame point reads: the finest whose extent holds it."""
        which = np.full(len(pts), len(self.grids) - 1, dtype=int)
        for i in reversed(range(len(self.grids) - 1)):
            which[_inside(self.extents[i], pts[:, 1], pts[:, 0])] = i
        return which

    def _lookup(self, points, arrays):
        pts = np.asarray(points, dtype="float64").reshape(-1, 2)
        out = np.empty(len(pts), dtype=arrays[0].dtype)
        which = self._split(pts) if len(self.grids) > 1 else np.zeros(len(pts), dtype=int)
        for i in np.unique(which):
            sel = which == i
            _d, idx = self._interp(i).tree.query(pts[sel])
            out[sel] = arrays[i].ravel()[idx]
        return out

    def point(self, north: float, east: float) -> tuple[float, bool]:
        """(elevation, land below sea level?) at one mesh-frame point: the fast path for
        the topography fit, which asks once per mesh vertex."""
        i = 0
        if len(self.grids) > 1:
            i = len(self.grids) - 1
            for j in range(len(self.grids) - 1):
                e0, e1, n0, n1 = self.extents[j]
                if e0 <= east <= e1 and n0 <= north <= n1:
                    i = j
                    break
        _d, k = self._interp(i).tree.query((north, east))
        if self._flat is None:
            self._flat = [(g[:, :, 2].ravel(), m.ravel()) for g, m in zip(self.grids, self.inland_masks)]
        z, inland = self._flat[i]
        return float(z[k]), bool(inland[k])

    def elevation(self, points) -> np.ndarray:
        """Nearest DEM elevation at mesh-frame points ``[[north, east], ...]``."""
        if len(self.grids) == 1:
            return self._interp(0)(np.asarray(points, dtype="float64").reshape(-1, 2))
        return self._lookup(points, [g[:, :, 2] for g in self.grids])

    def ocean_elevation(self, points) -> np.ndarray:
        """Like :meth:`elevation`, but land below sea level reads ``LAND_VIEW_M``: what the
        ocean code needs (elevation < 0 means water)."""
        if not self.has_inland:
            return self.elevation(points)
        return self._lookup(points, [g[:, :, 2] for g in self.ocean_grids])

    def inland(self, points) -> np.ndarray:
        """True at points of land below sea level."""
        if not self.has_inland:
            return np.zeros(np.asarray(points).reshape(-1, 2).shape[0], dtype=bool)
        return self._lookup(points, self.inland_masks)

    @property
    def has_inland(self) -> bool:
        return any(m.any() for m in self.inland_masks)

    # ---- the ocean code's inputs ------------------------------------------------ #
    def _cells(self, bounds):
        """Per grid, the points inside *bounds* not covered by a finer grid, and the area
        of one of its cells."""
        out = []
        for i, g in enumerate(self.grids):
            sel = O._bounds_mask(g, bounds)
            for j in range(i):
                sel &= ~_inside(self.extents[j], g[:, :, 0], g[:, :, 1])
            x, y = g[0, :, 0], g[:, 0, 1]
            area = abs(float(x[1] - x[0])) * abs(float(y[1] - y[0])) if len(x) > 1 and len(y) > 1 else 0.0
            out.append((sel, area))
        return out

    def ocean_stats(self, bounds=None) -> dict:
        """:func:`ocean.dem_ocean_stats` over all the grids, plus ``inland_km2``: the area of
        land below sea level."""
        cells = self._cells(bounds)
        inland_km2 = sum(float((m & sel).sum()) * area for m, (sel, area) in
                         zip(self.inland_masks, cells)) / 1e6
        if len(self.grids) == 1:
            return {**O.dem_ocean_stats(self.ocean_grids[0], bounds), "inland_km2": inland_km2}
        z = np.concatenate([g[:, :, 2][sel] for g, (sel, _a) in zip(self.ocean_grids, cells)])
        w = np.concatenate([np.full(int(sel.sum()), area) for sel, area in cells])
        ok = np.isfinite(z)
        z, w = z[ok], w[ok]
        if z.size == 0:
            return {"has_ocean": False, "ocean_frac": 0.0, "max_depth_m": 0.0,
                    "mean_depth_m": 0.0, "max_elev_m": 0.0, "inland_km2": inland_km2}
        sea = z < 0.0
        return {
            "has_ocean": bool(sea.any()),
            "ocean_frac": float(w[sea].sum() / w.sum()),
            "max_depth_m": float(-z.min()) if sea.any() else 0.0,
            "mean_depth_m": float(-(z[sea] * w[sea]).sum() / w[sea].sum()) if sea.any() else 0.0,
            "max_elev_m": float(z.max()),
            "inland_km2": inland_km2,
        }

    def expected_water_volume(self, bounds=None) -> float:
        """:func:`ocean.expected_water_volume` over all the grids."""
        if len(self.grids) == 1:
            return O.expected_water_volume(self.ocean_grids[0], bounds)
        total = 0.0
        for g, (sel, area) in zip(self.ocean_grids, self._cells(bounds)):
            depth = np.nan_to_num(np.where(g[:, :, 2] < 0.0, -g[:, :, 2], 0.0))
            total += float(depth[sel].sum() * area)
        return total

    def raw_coastlines(self):
        """:func:`ocean.extract_coastlines` of the ocean views: the detailed grid's, and the
        regional grid's outside the detailed one's extent."""
        if self._raw is None:
            raw = list(O.extract_coastlines(self.ocean_grids[0]))
            for i in range(1, len(self.grids)):
                for pts, closed in O.extract_coastlines(self.ocean_grids[i]):
                    runs = [pts]
                    for j in range(i):
                        runs = [r for run in runs for r in _outside_runs(run, self.extents[j])]
                    raw += [(r, closed and len(runs) == 1 and len(r) == len(pts)) for r in runs]
            self._raw = raw
        return self._raw

    def coast_points(self, stage: dict, bounds=None):
        """:func:`ocean.stage_coast_points` for an 'Along Coastline' stage."""
        return O.stage_coast_points(None, stage, bounds, raw=self.raw_coastlines())
