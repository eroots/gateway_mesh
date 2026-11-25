"""Ocean support for the mesh build -- pure numpy, no deal.II / shapely.

Two ways to put the sea into the model (``MeshInfo.ocean_mode``):

``"thin_sheet"``
    The mesh surface stays flat at z = 0 offshore (bathymetry is ignored by the
    topography fit).  Cells in the top *X* metres over ocean DEM points get a
    conductivity that preserves the *conductance* of the real water column:
    ``S = depth * sigma_sea`` is spread over the sheet thickness *X*, i.e.
    ``sigma_sheet = max(S / X, sigma_bg)``.  A cell that only partly overlaps
    the sheet gets the thickness-weighted mix with the background, so the
    column conductance is exact whatever the cell sizes are.  Cells are binned
    into a modest number of log-spaced conductivity classes so the materials
    file stays small.

``"real"``
    The seafloor is part of the fitted surface (see ``mesh.run``): former-air
    cells that end up below sea level (z > 0, z positive down) are water.

Everything here takes elevation < 0 as water.  ``mesh_dem.MeshDem`` tells land below
sea level from the sea, and hands these functions *ocean views* of the DEM in which that
land reads above 0.

Coastline helpers (0 m contour of the projected DEM, polyline simplification,
small-island removal, densification) feed the "Along Coastline" refinement stage
and its preview.  Coordinates here are the DEM's ``(easting, northing)``;
``mesh.run`` swaps to the deal.II ``(north, east)`` order.
"""
from __future__ import annotations

import numpy as np

#: log-spaced conductivity classes used by the thin-sheet binning
SHEET_CLASSES = 100
#: first material id handed to a thin-sheet class (1 air, 2 earth, 3 sea, 4 mantle)
SHEET_FIRST_ID = 5
#: cells whose sheet conductivity exceeds the background by less than this
#: fraction are left as ordinary (inverted) earth
SHEET_REL_TOL = 0.02


# --------------------------------------------------------------------------- #
# DEM statistics
# --------------------------------------------------------------------------- #
def _bounds_mask(topo_cart, bounds):
    """Boolean (ny, nx) mask of DEM points inside ``bounds`` = (xmin, xmax, ymin, ymax)
    in the DEM's (easting, northing) frame; ``None`` -> everything."""
    if bounds is None:
        return np.ones(topo_cart.shape[:2], dtype=bool)
    x0, x1, y0, y1 = bounds
    return ((topo_cart[:, :, 0] >= x0) & (topo_cart[:, :, 0] <= x1)
            & (topo_cart[:, :, 1] >= y0) & (topo_cart[:, :, 1] <= y1))


def dem_ocean_stats(topo_cart, bounds=None) -> dict:
    """Ocean coverage of the projected DEM (elevation < 0 counts as ocean: give it
    ``mesh_dem.MeshDem``'s ocean view to leave out land below sea level)."""
    inside = _bounds_mask(topo_cart, bounds)
    z = np.asarray(topo_cart[:, :, 2], dtype="float64")
    z = z[inside & np.isfinite(topo_cart[:, :, 2])]
    if z.size == 0:
        return {"has_ocean": False, "ocean_frac": 0.0, "max_depth_m": 0.0,
                "mean_depth_m": 0.0, "max_elev_m": 0.0}
    sea = z < 0.0
    return {
        "has_ocean": bool(sea.any()),
        "ocean_frac": float(sea.mean()),
        "max_depth_m": float(-z.min()) if sea.any() else 0.0,
        "mean_depth_m": float(-z[sea].mean()) if sea.any() else 0.0,
        "max_elev_m": float(z.max()),
    }


def expected_water_volume(topo_cart, bounds=None) -> float:
    """Volume (m^3) of water implied by the DEM inside ``bounds``."""
    inside = _bounds_mask(topo_cart, bounds)
    x = topo_cart[0, :, 0]
    y = topo_cart[:, 0, 1]
    if len(x) < 2 or len(y) < 2:
        return 0.0
    cell_area = abs(float(x[1] - x[0])) * abs(float(y[1] - y[0]))
    depth = np.where(topo_cart[:, :, 2] < 0.0, -topo_cart[:, :, 2], 0.0)
    depth = np.nan_to_num(depth)
    return float(depth[inside].sum() * cell_area)


# --------------------------------------------------------------------------- #
# Coastline (0 m contour)
# --------------------------------------------------------------------------- #
def _contour_lines(x, y, z, level):
    """List of (N, 2) arrays tracing ``z == level``."""
    zm = np.ma.masked_invalid(z)
    try:
        import contourpy

        gen = contourpy.contour_generator(x, y, zm, line_type=contourpy.LineType.Separate)
        return [np.asarray(l, dtype="float64") for l in gen.lines(level)]
    except (ImportError, AttributeError, TypeError, ValueError):
        # contourpy missing / too old -> matplotlib's own contouring (any version)
        from matplotlib.figure import Figure

        cs = Figure().subplots().contour(x, y, zm, levels=[level])
        return [np.asarray(s, dtype="float64") for s in cs.allsegs[0]]


def extract_coastlines(topo_cart, level: float = 0.0):
    """Raw ``(points, closed)`` polylines of the *level* contour of the DEM."""
    x = np.asarray(topo_cart[0, :, 0], dtype="float64")
    y = np.asarray(topo_cart[:, 0, 1], dtype="float64")
    z = np.asarray(topo_cart[:, :, 2], dtype="float64")
    out = []
    for line in _contour_lines(x, y, z, float(level)):
        if len(line) < 2:
            continue
        closed = bool(len(line) > 3 and np.allclose(line[0], line[-1]))
        out.append((line, closed))
    return out


def polyline_length(pts) -> float:
    return float(np.hypot(*np.diff(pts, axis=0).T).sum()) if len(pts) > 1 else 0.0


def polygon_area(pts) -> float:
    """Absolute shoelace area of a (closed or open) ring."""
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _douglas_peucker(p, tol):
    n = len(p)
    if n <= 2:
        return p
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = p[i], p[j]
        seg = p[i + 1:j]
        ab = b - a
        length = float(np.hypot(ab[0], ab[1]))
        if length == 0.0:
            d = np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1])
        else:
            d = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / length
        k = int(np.argmax(d))
        if d[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack.append((i, m))
            stack.append((m, j))
    return p[keep]


def simplify_polyline(pts, tol: float, closed: bool = False):
    """Douglas-Peucker; deviations from the original smaller than *tol* (m) are
    removed.  A closed ring is split at the point farthest from its start so the
    start point is not forced to survive."""
    pts = np.asarray(pts, dtype="float64")
    if tol <= 0.0 or len(pts) <= 3:
        return pts
    if not closed:
        return _douglas_peucker(pts, tol)
    ring = pts[:-1] if np.allclose(pts[0], pts[-1]) else pts
    if len(ring) <= 3:
        return np.vstack([ring, ring[:1]])
    k = int(np.argmax(np.hypot(ring[:, 0] - ring[0, 0], ring[:, 1] - ring[0, 1])))
    a = _douglas_peucker(ring[:k + 1], tol)
    b = _douglas_peucker(np.vstack([ring[k:], ring[:1]]), tol)
    out = np.vstack([a[:-1], b[:-1]])
    return np.vstack([out, out[:1]])


def densify(pts, step: float):
    """Insert points so no gap along the polyline exceeds *step* (m)."""
    pts = np.asarray(pts, dtype="float64")
    if step <= 0.0 or len(pts) < 2:
        return pts
    seg = np.diff(pts, axis=0)
    length = np.hypot(seg[:, 0], seg[:, 1])
    n = np.maximum(np.ceil(length / step).astype(int), 1)
    idx = np.repeat(np.arange(len(seg)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    frac = (np.arange(int(n.sum())) - start) / np.repeat(n, n)
    out = pts[idx] + seg[idx] * frac[:, None]
    return np.vstack([out, pts[-1:]])


def resample(pts, step: float):
    """Points on the polyline evenly spaced (by arc length) at about *step* (m),
    always including both ends.  Unlike :func:`densify` this also *thins* a
    polyline whose vertices are much closer together than *step*."""
    pts = np.asarray(pts, dtype="float64")
    if len(pts) < 2:
        return pts
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    if total <= 0.0:
        return pts[:1]
    n = max(int(np.ceil(total / max(step, 1e-9))), 1)
    t = np.linspace(0.0, total, n + 1)
    return np.c_[np.interp(t, s, pts[:, 0]), np.interp(t, s, pts[:, 1])]


def process_coastlines(topo_cart, simplify_m: float = 0.0, min_area_m2: float = 0.0,
                       level: float = 0.0, raw=None):
    """Extract, drop small features, simplify.

    Closed loops enclosing less than *min_area_m2* are dropped (small islands /
    lakes); open lines (cut by the DEM edge) shorter than the perimeter of a
    square of that area are dropped too.  Returns ``(polylines, info)`` with
    *polylines* a list of ``(points, closed)``.  Pass a previous
    :func:`extract_coastlines` result as *raw* to skip re-contouring (the
    preview dialog does while the user drags the simplify slider).
    """
    if raw is None:
        raw = extract_coastlines(topo_cart, level)
    min_open = 4.0 * float(np.sqrt(max(min_area_m2, 0.0)))
    kept = []
    n_raw_pts = sum(len(p) for p, _ in raw)
    for pts, closed in raw:
        if min_area_m2 > 0.0:
            if closed and polygon_area(pts) < min_area_m2:
                continue
            if not closed and polyline_length(pts) < min_open:
                continue
        kept.append((simplify_polyline(pts, simplify_m, closed), closed))
    info = {
        "n_lines_raw": len(raw), "n_lines": len(kept),
        "n_points_raw": int(n_raw_pts),
        "n_points": int(sum(len(p) for p, _ in kept)),
        "length_m": float(sum(polyline_length(p) for p, _ in kept)),
    }
    return kept, info


def coastline_points(polylines, spacing_m: float, bounds=None,
                     max_points: int = 6000):
    """Points along the coastline, one every *spacing_m* of arc length, inside ``bounds``.

    If that would exceed *max_points* the spacing is widened and the returned
    info says so.  Returns ``(points (N, 2) array [easting, northing], info)``.
    """
    spacing = max(float(spacing_m), 1.0)
    total = sum(polyline_length(p) for p, _ in polylines)
    widened = False
    if total / spacing > max_points:
        spacing = total / max_points
        widened = True
    chunks = [resample(p, spacing) for p, _ in polylines if len(p) > 1]
    pts = np.vstack(chunks) if chunks else np.zeros((0, 2))
    if bounds is not None and len(pts):
        x0, x1, y0, y1 = bounds
        pts = pts[(pts[:, 0] >= x0) & (pts[:, 0] <= x1)
                  & (pts[:, 1] >= y0) & (pts[:, 1] <= y1)]
    return pts, {"spacing_m": spacing, "spacing_widened": widened, "n_points": len(pts)}


def stage_coast_points(topo_cart, stage: dict, bounds=None, raw=None):
    """Mesh-frame points ``[north, east, 0]`` for a coastline refinement stage.  *raw*:
    coastlines already extracted (see :func:`process_coastlines`)."""
    simplify = float(stage.get("simplify_km", 0.0)) * 1e3
    min_area = float(stage.get("min_area_km2", 0.0)) * 1e6
    lines, info = process_coastlines(topo_cart, simplify, min_area, raw=raw)
    pts, pinfo = coastline_points(lines, float(stage.get("radius_km", 10.0)) * 1e3, bounds)
    info.update(pinfo)
    mesh_pts = np.zeros((len(pts), 3))
    mesh_pts[:, 0] = pts[:, 1]          # north
    mesh_pts[:, 1] = pts[:, 0]          # east
    return mesh_pts, info


# --------------------------------------------------------------------------- #
# Thin-sheet ocean
# --------------------------------------------------------------------------- #
def layer_extent(zc, level, dz_earth):
    """Top / bottom z of cells that sit in the (undeformed) base layer stack.

    ``dz_earth`` are the base earth layer thicknesses (top -> bottom); a cell
    refined *level* times is ``dz_k / 2**level`` thick, with k found from where
    its centre lies.  Exact wherever the topography fit did not move the nodes
    (everything offshore in thin-sheet mode).
    """
    dz = np.asarray(dz_earth, dtype="float64")
    edges = np.concatenate([[0.0], np.cumsum(dz)])
    k = np.clip(np.searchsorted(edges, np.asarray(zc, dtype="float64"), side="right") - 1,
                0, len(dz) - 1)
    t = dz[k] / (2.0 ** np.asarray(level, dtype="float64"))
    return zc - 0.5 * t, zc + 0.5 * t, t


def thin_sheet_conductivity(zc, level, depth_m, dz_earth, sheet_m: float,
                            sigma_sea: float, sigma_bg: float):
    """Per-cell conductivity (S/m) of the thin-sheet ocean.

    ``depth_m`` is the water depth below each cell's (x, y) (<= 0 on land).
    Cells outside the sheet, or over land, come back as exactly ``sigma_bg``.
    """
    zc = np.asarray(zc, dtype="float64")
    depth = np.maximum(np.asarray(depth_m, dtype="float64"), 0.0)
    top, bot, t = layer_extent(zc, level, dz_earth)
    overlap = np.clip(np.minimum(bot, sheet_m) - np.maximum(top, 0.0), 0.0, None)
    sigma_sheet = np.maximum(depth * float(sigma_sea) / float(sheet_m), sigma_bg)
    frac = np.where(t > 0, overlap / t, 0.0)
    sigma = sigma_bg + frac * (sigma_sheet - sigma_bg)
    sigma[depth <= 0.0] = sigma_bg
    return sigma


def bin_conductivities(sigma, sigma_bg: float, n_classes: int = SHEET_CLASSES,
                       rel_tol: float = SHEET_REL_TOL):
    """Group the cells that really are conductive into log-spaced classes.

    Returns ``(class_index, class_sigma)``: ``class_index[i]`` is -1 for cells
    that stay ordinary earth, else an index into ``class_sigma`` (the geometric
    mean of the members, so the class carries their typical conductivity).
    """
    sigma = np.asarray(sigma, dtype="float64")
    cls = np.full(sigma.shape, -1, dtype=int)
    sel = sigma > sigma_bg * (1.0 + rel_tol)
    if not sel.any():
        return cls, np.zeros(0)
    ls = np.log(sigma[sel])
    lo, hi = float(ls.min()), float(ls.max())
    n = max(int(n_classes), 1)
    if hi - lo < 1e-9:
        idx = np.zeros(ls.shape, dtype=int)
    else:
        idx = np.minimum(((ls - lo) / (hi - lo) * n).astype(int), n - 1)
    used = np.unique(idx)
    remap = np.full(n, -1, dtype=int)
    remap[used] = np.arange(len(used))
    idx = remap[idx]
    sums = np.bincount(idx, weights=ls, minlength=len(used))
    counts = np.bincount(idx, minlength=len(used))
    cls[sel] = idx
    return cls, np.exp(sums / counts)


def classify_real_ocean(zc, is_air):
    """Real-ocean mode: air-material cells whose centre ended up below sea level
    (z > 0, z positive down) after the topography fit are water."""
    return np.asarray(is_air, dtype=bool) & (np.asarray(zc, dtype="float64") > 0.0)
