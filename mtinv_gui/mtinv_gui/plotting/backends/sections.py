"""Vertical sections: straight N-S / E-W lines, arbitrary paths, station lines.

**Exact cut.**  A section path is a polyline in the display CRS.  Each segment
is cut with its own vertical plane, keeping only the strip between that
segment's two end vertices, so the cut is exact at bends (adjacent strips meet
on the vertical line through the shared vertex -- no gaps, no overlap).
Every cut vertex gets its distance along the path directly from the segment
geometry: ``s = s_start + (p - a) . u``.  (VTK's ``vtkPolyPlane`` /
``slice_along_line`` is not used: it extends the path past its ends and its
signed distance flips behind bends, producing phantom surfaces.)

**Interpolated.**  Optional smoothed renderings computed from the cut cells'
centroids on a regular (distance, elevation) grid, masked to where the model
has earth.  Each result is cached per (section, method, parameters), so
switching between the true cells and an interpolation is instant after the
first time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import slicing
from .model_source import _hex_connectivity

#: method key -> label
INTERP_METHODS = {
    "linear": "linear (Delaunay)",
    "cubic": "cubic (Clough-Tocher)",
    "nearest": "nearest centroid (Voronoi)",
    "rbf": "radial basis function",
    "natural": "natural neighbour",
    "gaussian": "Gaussian-smoothed cells",
}

STATION_PATHS = {"polyline": "straight segments through the stations",
                 "spline": "smooth spline through the stations",
                 "line": "best-fit straight line"}

KIND_LABELS = {"ns": "N-S", "ew": "E-W", "path": "path", "stations": "stations"}

#: Sections tab sub-tabs: group key -> label ("all" = every section)
SECTION_GROUPS = {"ns": "N-S", "ew": "E-W", "other": "Paths + stations", "all": "All"}


def section_group(sec) -> str:
    """``"ns"`` / ``"ew"`` / ``"other"`` (drawn paths and station lines)."""
    return sec.kind if sec.kind in ("ns", "ew") else "other"


def in_group(sec, group: str) -> bool:
    return group == "all" or section_group(sec) == group


def group_members(sections_list, sec) -> list:
    """The sections in *sec*'s own group (what a section's inset map shows)."""
    g = section_group(sec)
    return [s for s in sections_list if section_group(s) == g]


# --------------------------------------------------------------------------- #
@dataclass
class Profile:
    """A section path.  ``disp`` / ``local`` are ``(M, 2)`` vertices in the
    display CRS / model local frame; ``s`` is the cumulative distance along
    the path in display-CRS metres."""
    disp: np.ndarray
    local: np.ndarray
    s: np.ndarray

    @classmethod
    def from_display(cls, verts, maptr, max_seg_m: float = 5000.0) -> "Profile":
        """When the display and model CRS differ, a straight display line is
        slightly curved in the model frame, so long segments are split into
        pieces of at most *max_seg_m* before mapping."""
        v = np.asarray(verts, dtype="float64").reshape(-1, 2)
        keep = np.r_[True, np.hypot(*np.diff(v, axis=0).T) > 1e-6]
        v = v[keep]
        if len(v) < 2:
            raise ValueError("a section path needs at least two distinct points")
        if not maptr.same:
            pieces = [v[:1]]
            for a, b in zip(v[:-1], v[1:]):
                n = max(int(np.ceil(np.hypot(*(b - a)) / max_seg_m)), 1)
                t = np.linspace(0.0, 1.0, n + 1)[1:, None]
                pieces.append(a + t * (b - a))
            v = np.concatenate(pieces)
        x, y = maptr.inverse(v[:, 0], v[:, 1])
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(v, axis=0).T))]
        return cls(disp=v, local=np.c_[x, y], s=s)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def key(self) -> tuple:
        return tuple(np.round(self.disp, 3).ravel())

    def at(self, s):
        """``(x, y, E, N)`` at distances *s* (clipped to the path)."""
        s = np.clip(np.asarray(s, dtype="float64"), 0.0, self.length)
        out = [np.interp(s, self.s, arr) for arr in
               (self.local[:, 0], self.local[:, 1], self.disp[:, 0], self.disp[:, 1])]
        return tuple(out)

    def project(self, E, N) -> tuple[np.ndarray, np.ndarray]:
        """Distance along the path and perpendicular offset (display CRS m) of
        each point -- nearest point on the polyline."""
        P = np.c_[np.ravel(E), np.ravel(N)].astype("float64")
        A, B = self.disp[:-1], self.disp[1:]
        D = B - A
        L2 = np.maximum((D ** 2).sum(1), 1e-12)
        t = np.clip(((P[:, None, :] - A[None]) * D[None]).sum(2) / L2[None], 0.0, 1.0)
        foot = A[None] + t[..., None] * D[None]
        dist = np.hypot(*(P[:, None, :] - foot).transpose(2, 0, 1))
        j = dist.argmin(1)
        i = np.arange(len(P))
        s = self.s[j] + t[i, j] * np.sqrt(L2[j])
        return s, dist[i, j]


def station_lookup(session) -> dict:
    """``{normalised name: (E, N)}`` for the loaded receivers (display CRS)."""
    from .georef import _norm_name

    xy = session.stations_display()
    return {_norm_name(n): tuple(p) for n, p in zip(session.receivers.names, xy)}


def _station_vertices(names, lookup, mode: str, extend_m: float):
    from .georef import _norm_name

    pts = [lookup[_norm_name(n)] for n in names if _norm_name(n) in lookup]
    pts = np.asarray(pts, dtype="float64").reshape(-1, 2)
    if len(pts) < 2:
        raise ValueError("a station section needs at least two stations with known positions")
    if mode == "line":
        c = pts.mean(0)
        _, _, vt = np.linalg.svd(pts - c)
        u = vt[0]
        t = (pts - c) @ u
        if t[-1] < t[0]:           # keep the stations' order: first station -> A
            u, t = -u, -t
        verts = np.array([c + t.min() * u, c + t.max() * u])
    elif mode == "spline" and len(pts) >= 3:
        from scipy.interpolate import splev, splprep

        tck, _u = splprep(pts.T, s=0.0, k=min(3, len(pts) - 1))
        dense = np.array(splev(np.linspace(0.0, 1.0, max(40 * len(pts), 200)), tck)).T
        verts = _simplify(dense, 25.0)
    else:
        verts = pts
    if extend_m > 0:
        u0 = verts[0] - verts[1]
        u1 = verts[-1] - verts[-2]
        verts = np.r_[[verts[0] + extend_m * u0 / np.hypot(*u0)], verts,
                      [verts[-1] + extend_m * u1 / np.hypot(*u1)]]
    return verts


def _simplify(xy: np.ndarray, tol_m: float) -> np.ndarray:
    """Douglas-Peucker (via shapely) -- fewer segments to cut, same path to *tol_m*."""
    try:
        from shapely.geometry import LineString

        return np.asarray(LineString(xy).simplify(tol_m).coords)
    except Exception:  # noqa: BLE001 -- shapely missing: keep every point
        return xy


def default_span(session, style, along: str) -> tuple[float, float]:
    """Stations' extent (+ padding) in easting or northing."""
    from ..project import ExtentInfo

    x0, x1, y0, y1 = session.extent(ExtentInfo(mode="stations", padding_m=style.map_padding_m))
    return (y0, y1) if along == "north" else (x0, x1)


def section_vertices(sec, session, style) -> np.ndarray:
    """Display-CRS vertices of a :class:`~..project.SectionInfo`, A -> A'."""
    if sec.kind in ("ns", "ew"):
        a, b = float(sec.start_m), float(sec.end_m)
        if a == b:
            a, b = default_span(session, style, "north" if sec.kind == "ns" else "east")
        c = float(sec.coord_m)
        verts = np.array([[c, a], [c, b]]) if sec.kind == "ns" else np.array([[a, c], [b, c]])
    elif sec.kind == "stations":
        verts = _station_vertices(sec.stations, station_lookup(session), sec.station_path,
                                  float(sec.extend_m))
    else:
        verts = np.asarray(sec.vertices, dtype="float64").reshape(-1, 2)
    return verts[::-1] if sec.reverse else verts


def build_profile(sec, session, style) -> Profile:
    return Profile.from_display(section_vertices(sec, session, style), session.maptr)


# --------------------------------------------------------------------------- #
@dataclass
class SectionCut(slicing.PolySlice):
    """Cell polygons in (distance, elevation); ``en[i]`` the display (E, N)
    of each vertex of ``polys[i]`` (for Easting / Northing x-axes)."""
    en: list = field(default_factory=list)

    def centroids(self) -> np.ndarray:
        if not self.polys:
            return np.zeros((0, 2))
        return np.array([p.mean(0) for p in self.polys])


def _cell_xy_bounds(model):
    """Per earth cell: corner xy ``(n, 8, 2)`` and its bbox (cached on the model)."""
    cached = getattr(model, "_cell_xy", None)
    if cached is None:
        e = model.earth
        pts = np.asarray(e.points)
        conn = _hex_connectivity(e)
        corners = pts[conn]
        xy = corners[:, :, :2]
        cached = dict(corners=corners, xy=xy, vals=np.asarray(e.cell_data["log10_rho"]),
                      xmin=xy[:, :, 0].min(1), xmax=xy[:, :, 0].max(1),
                      ymin=xy[:, :, 1].min(1), ymax=xy[:, :, 1].max(1),
                      celltypes=np.asarray(e.celltypes))
        model._cell_xy = cached
    return cached


def _sub_grid(cx, sel):
    """UnstructuredGrid of just the cells *sel* (each owns its 8 points)."""
    import pyvista as pv

    n = len(sel)
    pts = cx["corners"][sel].reshape(-1, 3)
    cells = np.c_[np.full(n, 8), np.arange(8 * n).reshape(n, 8)].ravel()
    g = pv.UnstructuredGrid(cells, cx["celltypes"][sel], pts)
    g.cell_data["log10_rho"] = cx["vals"][sel]
    return g


def cut_section(model, maptr, prof: Profile) -> SectionCut:
    """Exact cell polygons along *prof* (cached)."""
    key = ("section", id(model), maptr.key, prof.key)
    hit = slicing._cache.get(key)
    if hit is not None:
        return hit
    cx = _cell_xy_bounds(model)
    polys, ens, vals = [], [], []
    for j in range(len(prof.local) - 1):
        a, b = prof.local[j], prof.local[j + 1]
        d = b - a
        L = float(np.hypot(*d))
        if L <= 0:
            continue
        u = d / L
        nrm = np.array([-u[1], u[0]])
        cand = np.where((cx["xmax"] >= min(a[0], b[0])) & (cx["xmin"] <= max(a[0], b[0]))
                        & (cx["ymax"] >= min(a[1], b[1])) & (cx["ymin"] <= max(a[1], b[1])))[0]
        if not len(cand):
            continue
        C = cx["xy"][cand] - a
        dn, dt = C @ nrm, C @ u
        sel = cand[(dn.min(1) <= 0) & (dn.max(1) >= 0) & (dt.max(1) >= 0) & (dt.min(1) <= L)]
        if not len(sel):
            continue
        cut = _sub_grid(cx, sel).slice(normal=(nrm[0], nrm[1], 0.0), origin=(a[0], a[1], 0.0))
        if cut.n_cells:
            cut = cut.clip(normal=(-u[0], -u[1], 0.0), origin=(a[0], a[1], 0.0))
        if cut.n_cells:
            cut = cut.clip(normal=(u[0], u[1], 0.0), origin=(b[0], b[1], 0.0))
        if not cut.n_cells:
            continue
        idx, cell_ids = slicing._polys_from_polydata(cut)
        if not idx:
            continue
        pts = np.asarray(cut.points)
        scale = (prof.s[j + 1] - prof.s[j]) / L            # local m -> display m
        sv = prof.s[j] + ((pts[:, :2] - a) @ u) * scale
        E, N = maptr.forward(pts[:, 0], pts[:, 1])
        sz = np.c_[sv, pts[:, 2]]
        en = np.c_[E, N]
        polys += [sz[i] for i in idx]
        ens += [en[i] for i in idx]
        vals.append(np.asarray(cut.cell_data["log10_rho"])[cell_ids])
    out = SectionCut(polys=polys, values=np.concatenate(vals) if vals else np.zeros(0), en=ens)
    slicing._cache.put(key, out)
    return out


# --------------------------------------------------------------------------- #
@dataclass
class SectionGrid:
    """Regular (distance, elevation) grid -- node centres."""
    s: np.ndarray
    z: np.ndarray
    values: np.ndarray            # (nz, nx) log10 rho, NaN outside the earth


def _grid_axes(prof, z_lo, z_hi, nx, nz):
    return (np.linspace(0.0, prof.length, max(int(nx), 2)),
            np.linspace(z_lo, z_hi, max(int(nz), 2)))


def exact_grid(model, maptr, prof, z_lo, z_hi, nx, nz) -> SectionGrid:
    """The true cells sampled on the grid (also the earth mask for the
    interpolations).  Cached."""
    key = ("exact_grid", id(model), maptr.key, prof.key, float(z_lo), float(z_hi), int(nx), int(nz))
    hit = slicing._cache.get(key)
    if hit is not None:
        return hit
    S, Z = _grid_axes(prof, z_lo, z_hi, nx, nz)
    x, y, _E, _N = prof.at(S)
    X = np.broadcast_to(x, (len(Z), len(S)))
    Y = np.broadcast_to(y, (len(Z), len(S)))
    ZZ = np.broadcast_to(Z[:, None], (len(Z), len(S)))
    out = SectionGrid(S, Z, slicing.sample_points(model, X, Y, ZZ))
    slicing._cache.put(key, out)
    return out


def interp_params(style) -> tuple:
    m = style.interp_method
    if m == "rbf":
        return (m, style.rbf_kernel, float(style.rbf_smoothing), float(style.rbf_epsilon),
                int(style.rbf_neighbors))
    if m == "gaussian":
        return (m, float(style.gauss_sigma_px))
    return (m,)


def interpolated_grid(model, maptr, prof, cut: SectionCut, style, z_lo, z_hi) -> SectionGrid:
    """Smoothed rendering per ``style.interp_method`` (cached per parameters)."""
    params = interp_params(style)
    nx, nz = int(style.interp_nx), int(style.interp_nz)
    key = ("interp", id(model), maptr.key, prof.key, float(z_lo), float(z_hi), nx, nz, params)
    hit = slicing._cache.get(key)
    if hit is not None:
        return hit
    ex = exact_grid(model, maptr, prof, z_lo, z_hi, nx, nz)
    mask = np.isnan(ex.values)
    method = params[0]
    SS, ZZ = np.meshgrid(ex.s, ex.z)
    if method == "gaussian":
        vals = _gaussian_nan(ex.values, float(params[1]))
    else:
        pts = cut.centroids()
        v = np.asarray(cut.values)
        keep = (pts[:, 1] >= z_lo - 0.25 * (z_hi - z_lo))      # skip far-below cells
        pts, v = pts[keep], v[keep]
        # pin both section ends to the true cells: the outermost centroids sit
        # half a cell inside, and hull-edge triangles there smear values over
        # the full depth
        for col in (0, -1):
            ok = ~np.isnan(ex.values[:, col])
            pts = np.r_[pts, np.c_[np.full(ok.sum(), ex.s[col]), ex.z[ok]]]
            v = np.r_[v, ex.values[ok, col]]
        if len(pts) < 4:
            raise ValueError("too few cells on this section to interpolate")
        # scale so distance and elevation weigh the same per grid node
        sx = max(prof.length / max(nx - 1, 1), 1e-9)
        sz = max((z_hi - z_lo) / max(nz - 1, 1), 1e-9)
        P = pts / [sx, sz]
        Q = np.c_[SS.ravel() / sx, ZZ.ravel() / sz]
        vals = _interp_points(method, params, P, v, Q).reshape(SS.shape)
        # outside the centroids' hull (top row of cells, ends): fall back to cells
        holes = np.isnan(vals) & ~mask
        vals[holes] = ex.values[holes]
    vals = np.where(mask, np.nan, vals)
    out = SectionGrid(ex.s, ex.z, vals)
    slicing._cache.put(key, out)
    return out


def _interp_points(method, params, P, v, Q):
    from scipy import interpolate as I

    if method in ("linear", "cubic", "nearest"):
        return I.griddata(P, v, Q, method=method)
    if method == "rbf":
        _m, kernel, smoothing, epsilon, neighbors = params
        rbf = I.RBFInterpolator(P, v, kernel=kernel, smoothing=smoothing, epsilon=epsilon,
                                neighbors=min(int(neighbors), len(P)) or None)
        return rbf(Q)
    if method == "natural":
        try:
            import naturalneighbor
        except ImportError:
            raise ValueError("natural neighbour needs the optional 'naturalneighbor' package: "
                             "see Install in README.md") from None
        # the package is 3-D only: stack two copies at z = 0 / 1, read the layer at 0
        P3 = np.r_[np.c_[P, np.zeros(len(P))], np.c_[P, np.ones(len(P))]]
        v3 = np.r_[v, v]
        xs, ys = np.unique(Q[:, 0]), np.unique(Q[:, 1])
        rng = [[xs[0], xs[-1], len(xs) * 1j], [ys[0], ys[-1], len(ys) * 1j], [0, 1, 2j]]
        g = naturalneighbor.griddata(P3, v3, rng)[:, :, 0]    # (nx, nz)
        return g.T.ravel()
    raise ValueError(f"unknown interpolation {method!r}")


def _gaussian_nan(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur that ignores NaNs (normalised convolution)."""
    from scipy.ndimage import gaussian_filter

    ok = ~np.isnan(a)
    num = gaussian_filter(np.where(ok, a, 0.0), sigma)
    den = gaussian_filter(ok.astype(float), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 1e-6, num / den, np.nan)


# --------------------------------------------------------------------------- #
@dataclass
class StationsOnSection:
    names: list
    s: np.ndarray
    z: np.ndarray            # elevation of the receiver
    offset: np.ndarray       # perpendicular distance from the path (m)


def stations_on_section(sec, session, style, prof: Profile) -> StationsOnSection:
    """Station section: its own stations.  Others: every station within
    ``style.station_max_offset_m`` of the path (projected onto it)."""
    from .georef import _norm_name

    names = session.receivers.names
    if not names:
        return StationsOnSection([], np.zeros(0), np.zeros(0), np.zeros(0))
    xy = session.stations_display()
    s, off = prof.project(xy[:, 0], xy[:, 1])
    if sec.kind == "stations":
        want = {_norm_name(n) for n in sec.stations}
        keep = np.array([_norm_name(n) in want for n in names])
    else:
        keep = (off <= float(style.station_max_offset_m)) & (s > 0.0) & (s < prof.length)
    idx = np.where(keep)[0]
    return StationsOnSection([names[i] for i in idx], s[idx],
                             session.receivers.xyz[idx, 2], off[idx])


def surface_profile(model, prof: Profile, n: int = 1000):
    """``(s, z)`` of the model surface along the path."""
    s = np.linspace(0.0, prof.length, n)
    x, y, _E, _N = prof.at(s)
    return s, model.surface()(x, y)


def z_limits(model, style, prof: Profile | None = None) -> tuple[float, float]:
    """``(bottom, top)`` elevation (m) of the plot."""
    bottom = -float(style.depth_bottom_m)
    if style.auto_top:
        if prof is not None:
            top = float(np.nanmax(surface_profile(model, prof, 400)[1]))
        else:
            top = model.surface().z_range[1]
        top = max(top, 0.0) + max(0.015 * (top - bottom), 200.0)   # room for station markers
    else:
        top = float(style.z_top_m)
    if top <= bottom:
        top = bottom + 1.0
    return bottom, top


# --------------------------------------------------------------------------- #
@dataclass
class SectionData:
    """Everything a section figure needs; built off the GUI thread."""
    profile: Profile
    cut: SectionCut
    stations: StationsOnSection
    z_lo: float
    z_hi: float
    grid: SectionGrid | None = None
    surface: tuple | None = None
    layer_items: list = field(default_factory=list)    # layers.SectionItems
    layer_errors: list = field(default_factory=list)   # [(LayerInfo, exception)]


def prepare(session, sec, style, layers=None) -> SectionData:
    """Cut (+ interpolate when ``style.show_interp``); every step cached.
    *layers* (``project.layers``) add profiles, depth lines and markers."""
    from . import layers as L

    prof = build_profile(sec, session, style)
    cut = cut_section(session.model, session.maptr, prof)
    z_lo, z_hi = z_limits(session.model, style, prof)
    grid = None
    if style.show_interp:
        grid = interpolated_grid(session.model, session.maptr, prof, cut, style, z_lo, z_hi)
    items, errors = [], []
    if style.topo_profile:
        from ..project import LayerInfo

        s_t, z_t = surface_profile(session.model, prof, 800)
        topo = LayerInfo(name="Topography", kind="raster", units="m", color=style.topo_color,
                         profile_axis="auto", linewidth=1.0)
        items.append(L.SectionItems(topo, "profile", s_t, z_t,
                                    vrange=session.model.surface().z_range))
    ok, bad = session.active_layers(layers or [])
    errors += bad
    for info, loaded in ok:
        try:
            it = L.section_items(info, loaded, prof, session.model, session.maptr)
        except Exception as exc:  # noqa: BLE001
            errors.append((info, exc))
            continue
        if it is not None:
            items.append(it)
    return SectionData(profile=prof, cut=cut, stations=stations_on_section(sec, session, style, prof),
                       z_lo=z_lo, z_hi=z_hi, grid=grid,
                       surface=surface_profile(session.model, prof) if style.show_surface else None,
                       layer_items=items, layer_errors=errors)


def next_label(existing) -> str:
    """A, B, ..., Z, AA, AB, ... not already used."""
    used = set(existing)
    i = 0
    while True:
        n, lab = i, ""
        while True:
            lab = chr(ord("A") + n % 26) + lab
            n = n // 26 - 1
            if n < 0:
                break
        if lab not in used:
            return lab
        i += 1


def read_path_file(path: str, maptr) -> np.ndarray:
    """Two numeric columns (E, N in the display CRS, or lon, lat -- detected
    from the value ranges); comma / whitespace separated, header lines skipped."""
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            parts = line.replace(",", " ").replace(";", " ").split()
            try:
                rows.append([float(parts[0]), float(parts[1])])
            except (ValueError, IndexError):
                continue
    a = np.asarray(rows, dtype="float64").reshape(-1, 2)
    if len(a) < 2:
        raise ValueError(f"no coordinate pairs found in {path}")
    if np.all(np.abs(a[:, 0]) <= 180.0) and np.all(np.abs(a[:, 1]) <= 90.0):
        E, N = maptr.from_lonlat(a[:, 0], a[:, 1])
        a = np.c_[E, N]
    return a


def read_station_list(path: str) -> list[str]:
    """``.lst``: one station per line (``.edi`` optional, ``#`` comments)."""
    out = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            name = line.strip()
            if name and not name.startswith("#"):
                out.append(name[:-4] if name.lower().endswith(".edi") else name)
    return out


# --------------------------------------------------------------------------- #
# section sets (.json): save / load section definitions
# --------------------------------------------------------------------------- #
SECTION_SET_FORMAT = "mtplot-sections"


def write_section_set(path: str, sections_list, maptr=None) -> str:
    """Write the definitions of *sections_list* to a JSON file.  Coordinates
    are display-CRS metres; the CRS is stored with them, and path vertices
    also as lon/lat so they can be read back under another display CRS."""
    import json
    import os
    from dataclasses import asdict

    out = {"format": SECTION_SET_FORMAT, "version": 1, "crs": "", "crs_name": "", "sections": []}
    if maptr is not None:
        out["crs"] = maptr.display_crs.to_wkt()
        out["crs_name"] = maptr.display_label
    for sec in sections_list:
        d = asdict(sec)
        if maptr is not None and sec.kind == "path" and sec.vertices:
            v = np.asarray(sec.vertices, dtype="float64").reshape(-1, 2)
            lon, lat = maptr.to_lonlat(v[:, 0], v[:, 1])
            d["vertices_lonlat"] = np.c_[lon, lat].tolist()
        out["sections"].append(d)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    return path


def is_section_set(path: str) -> bool:
    import json

    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh).get("format") == SECTION_SET_FORMAT
    except (OSError, ValueError, AttributeError):
        return False


def read_section_set(path: str, maptr=None, existing_labels=()) -> tuple[list, list[str]]:
    """Section definitions from :func:`write_section_set` -> ``(sections, notes)``.

    Labels already in *existing_labels* are replaced by free ones.  When the
    file's CRS differs from *maptr*'s display CRS, path sections are rebuilt
    from their lon/lat copy; N-S / E-W / station sections keep their numbers
    (a note says so)."""
    import json

    from ..project import SectionInfo, rebuild

    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict) or raw.get("format") != SECTION_SET_FORMAT:
        raise ValueError(f"{path} is not a saved section set")
    notes = []
    same_crs = True
    if maptr is not None and raw.get("crs"):
        from pyproj import CRS

        try:
            same_crs = CRS.from_user_input(raw["crs"]) == maptr.display_crs
        except Exception:  # noqa: BLE001
            same_crs = False
        if not same_crs:
            notes.append(f"saved in {raw.get('crs_name') or 'another CRS'}, shown in "
                         f"{maptr.display_label}")
    used = set(existing_labels)
    out = []
    for d in raw.get("sections", []):
        sec = rebuild(SectionInfo, d)
        if not same_crs:
            ll = d.get("vertices_lonlat")
            if sec.kind == "path" and ll:
                a = np.asarray(ll, dtype="float64").reshape(-1, 2)
                E, N = maptr.from_lonlat(a[:, 0], a[:, 1])
                sec.vertices = np.c_[E, N].tolist()
            elif sec.kind in ("ns", "ew"):
                notes.append(f"{sec.label}: {KIND_LABELS[sec.kind]} position kept as numbers "
                             "(not converted between CRSs)")
        if sec.label in used:
            new = next_label(used)
            notes.append(f"{sec.label} renamed {new} (label already used)")
            sec.label = new
        used.add(sec.label)
        out.append(sec)
    if not out:
        raise ValueError(f"no sections in {path}")
    return out, notes
