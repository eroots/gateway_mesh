"""Task 2 -- project + interpolate the DEM into the model's local frame.

The model's local frame -- what stations, the DEM and the deal.II mesh all
live in -- is always a *map projection* to metres; there is no projection-free
local frame.  ``LocalFrame.kind`` picks the family:

  * ``"local"``: a projection built around the survey itself, origin exactly
    (0, 0) at the survey centre.  ``LocalFrame.local_proj`` picks which:
      - ``"tmerc"``  Transverse Mercator, central meridian through the origin
                     (conformal; scale error grows east-west, ~ (dx/2R)^2).
      - ``"stere"``  Oblique Stereographic at the origin (conformal; scale
                     error grows radially, ~ (r/2R)^2) -- good for roughly
                     round footprints.
      - ``"aeqd"``   Azimuthal Equidistant at the origin (not conformal; true
                     distance from the origin along every direction, tangential
                     scale grows with radius).
      - ``"lcc"``    Lambert Conformal Conic, standard parallels at 1/6 and 5/6
                     of the survey's latitude span (conformal; minimises scale
                     error for east-west elongated surveys).
  * ``"utm"`` / ``"epsg"``: project to that EPSG and subtract the EPSG
    coordinate of the origin.  Keeps the model in a standard national grid
    (handy to overlay other GIS layers) at the cost of that grid's own
    convergence + scale distortion.

For a large survey the point of choosing among these is to minimise spatial
distortion; :func:`distortion` measures it (scale error and grid-vs-true-north
rotation over the stations) so the choice can be made on numbers.

Port notes:
  * ``cartopy.crs.UTM`` -> ``pyproj.Transformer(EPSG:4326 -> target CRS)``.
    ``always_xy=True`` so ``transform(lon, lat)`` returns ``(easting, northing)``.
  * Elevation is carried through the horizontal transform unchanged.
  * Local origin ``(x0, y0)`` = target-CRS coords of the survey-bbox centre,
    subtracted from every coordinate.  For ``kind="local"`` that is ~(0, 0)
    already, so the subtraction is a no-op (bar float residue).
  * ``topo_cart`` shape ``(res_grid, res_grid, 3)``: [...,0]=local easting,
    [...,1]=local northing, [...,2]=elevation (m) -- conventional (E, N).
  * Station axis swap: the GoFEM mesh has X = North (deal.II axis 0), so its
    receivers are ``[northing-shift, easting-shift, 0]``.  That
    swap is a property of the *mesh*, not of the projection -- so
    :func:`project_stations` now returns conventional ``(easting, northing)``
    by default (matches ``topo_cart`` and every map view), and the one place
    that feeds the deal.II mesh (``backends/mesh.py``) asks for ``swap_xy=True``
    explicitly, alongside the topography KD-tree it builds transposed for the
    same reason.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


LOCAL_PROJECTIONS = {
    "tmerc": "Transverse Mercator",
    "stere": "Oblique Stereographic",
    "aeqd": "Azimuthal Equidistant",
    "lcc": "Lambert Conformal Conic",
}


@dataclass
class LocalFrame:
    kind: str = "local"          # "local" | "utm" | "epsg"
    epsg: int = 0                # target EPSG for kind in ("utm", "epsg")
    origin_lon: float = 0.0
    origin_lat: float = 0.0
    local_proj: str = "tmerc"    # kind == "local": key of LOCAL_PROJECTIONS
    lat1: float = 0.0            # lcc standard parallels (0 -> derived from origin)
    lat2: float = 0.0
    x0: float = 0.0              # target-CRS easting of the model origin (0 for "local")
    y0: float = 0.0              # target-CRS northing of the model origin (0 for "local")


@dataclass
class ProjectedDem:
    """``topo_cart`` -- shape (n, n, 3): [...,0]=x, [...,1]=y, [...,2]=z (m, local).

    ``regional``: an optional coarser grid of the same layout covering a wider area
    (the mesh padding); the mesh uses ``topo_cart`` wherever it reaches.
    """
    topo_cart: np.ndarray
    frame: LocalFrame
    regional: np.ndarray | None = None

    @property
    def extent_local(self) -> dict:
        tc = self.topo_cart
        return {
            "x_min": float(tc[:, :, 0].min()), "x_max": float(tc[:, :, 0].max()),
            "y_min": float(tc[:, :, 1].min()), "y_max": float(tc[:, :, 1].max()),
            "z_min": float(tc[:, :, 2].min()), "z_max": float(tc[:, :, 2].max()),
        }


def standard_parallels(lat_min: float, lat_max: float) -> tuple[float, float]:
    """Lambert standard parallels at 1/6 and 5/6 of the latitude span (the usual
    "one-sixth rule" for keeping scale error small and balanced)."""
    span = float(lat_max) - float(lat_min)
    return float(lat_min) + span / 6.0, float(lat_max) - span / 6.0


def _local_proj4(local_proj: str, origin_lon: float, origin_lat: float,
                 lat1: float = 0.0, lat2: float = 0.0) -> str:
    base = (f"+lat_0={origin_lat} +lon_0={origin_lon} +x_0=0 +y_0=0 "
            "+ellps=WGS84 +units=m +no_defs +type=crs")
    if local_proj == "stere":
        return f"+proj=stere +k=1 {base}"
    if local_proj == "aeqd":
        return f"+proj=aeqd {base}"
    if local_proj == "lcc":
        l1 = lat1 or origin_lat
        l2 = lat2 or origin_lat
        return f"+proj=lcc +lat_1={l1} +lat_2={l2} {base}"
    return f"+proj=tmerc +k=1 {base}"          # "tmerc" and anything unrecognised


def _target_crs(frame: LocalFrame):
    from pyproj import CRS

    if frame.kind == "local":
        return CRS.from_proj4(_local_proj4(frame.local_proj, frame.origin_lon,
                                           frame.origin_lat, frame.lat1, frame.lat2))
    return CRS.from_epsg(int(frame.epsg))


def _transformer(frame: LocalFrame):
    from pyproj import Transformer

    return Transformer.from_crs(4326, _target_crs(frame), always_xy=True)


def make_local_frame(kind: str, epsg: int, origin_lon: float, origin_lat: float,
                     local_proj: str = "tmerc", lat1: float = 0.0,
                     lat2: float = 0.0) -> LocalFrame:
    """Build the model's local frame.

    *kind* ``"local"`` -> a survey-centred projection (*local_proj*, origin at
    0,0); ``"utm"`` / ``"epsg"`` -> that EPSG, with the origin's projected
    coordinate stored as ``(x0, y0)`` and subtracted downstream.
    """
    frame = LocalFrame(kind=str(kind or "local"), epsg=int(epsg or 0),
                       origin_lon=float(origin_lon), origin_lat=float(origin_lat),
                       local_proj=str(local_proj or "tmerc"),
                       lat1=float(lat1 or 0.0), lat2=float(lat2 or 0.0))
    x0, y0 = _transformer(frame).transform(origin_lon, origin_lat)
    frame.x0, frame.y0 = float(x0), float(y0)
    return frame


def frame_from_projection(p) -> LocalFrame:
    """:class:`LocalFrame` from a ``ProjectionInfo`` *or* a plain dict with the
    same keys (the mesh spec) -- the one place the frame is assembled, so every
    step (DEM, stations, mesh, scales) builds an identical one."""
    def g(key, default=None):
        return p.get(key, default) if isinstance(p, dict) else getattr(p, key, default)

    return make_local_frame(g("proj_kind") or "local", g("epsg") or 0,
                            g("origin_lon") or 0.0, g("origin_lat") or 0.0,
                            g("local_proj") or "tmerc",
                            g("local_lat1") or 0.0, g("local_lat2") or 0.0)


def frame_label(frame: LocalFrame) -> str:
    """Explicit, human-readable name of the projection in use."""
    if frame.kind == "local":
        name = LOCAL_PROJECTIONS.get(frame.local_proj, frame.local_proj)
        extra = (f", standard parallels {frame.lat1:.2f}° / {frame.lat2:.2f}°"
                 if frame.local_proj == "lcc" else "")
        return (f"Local {name} centred on ({frame.origin_lon:.4f}, "
                f"{frame.origin_lat:.4f}){extra}")
    return f"{epsg_name(frame.epsg)} (EPSG:{frame.epsg})"


frame_summary = frame_label          # older name


def distortion(frame: LocalFrame, lons, lats) -> dict | None:
    """Spatial distortion of *frame* over the points ``(lons, lats)`` (degrees).

    Returns scale factors (1.0 = true length; both the meridional and parallel
    factors are included, so non-conformal projections are not flattered) and
    the meridian convergence (angle between grid north and true north, deg):

        ``scale_min`` / ``scale_max``, ``max_err_pct`` (= max |k-1| in %),
        ``m_per_km`` (= worst-case length error per km), ``angular_deg``
        (shape distortion; 0 for a conformal projection), ``conv_min`` /
        ``conv_max``.

    ``None`` if pyproj can't evaluate it (bad EPSG, no points).
    """
    try:
        from pyproj import Proj

        lons = np.asarray(lons, dtype="float64")
        lats = np.asarray(lats, dtype="float64")
        if lons.size == 0:
            return None
        f = Proj(_target_crs(frame)).get_factors(lons, lats)
        k = np.concatenate([np.atleast_1d(f.meridional_scale),
                            np.atleast_1d(f.parallel_scale)])
        conv = np.atleast_1d(f.meridian_convergence)
        err = float(np.max(np.abs(k - 1.0)) * 100.0)
        ang = float(np.max(np.abs(np.atleast_1d(f.angular_distortion))))
        return {"scale_min": float(k.min()), "scale_max": float(k.max()),
                "max_err_pct": err, "m_per_km": err * 10.0,
                "angular_deg": ang,
                "conv_min": float(conv.min()), "conv_max": float(conv.max())}
    except Exception:  # noqa: BLE001
        return None


def project_dem(dem, frame: LocalFrame, sample_rate_m: float,
                method: str = "nearest") -> ProjectedDem:
    """Resample a DEM onto a regular grid in the model frame."""
    from scipy.interpolate import griddata

    lon_grid, lat_grid = dem.meshgrid()
    east, north = _transformer(frame).transform(lon_grid, lat_grid)
    pts = np.c_[east.ravel(), north.ravel()]
    zvals = np.asarray(dem.z, dtype="float64").ravel()

    res_grid = max(int((east.max() - east.min()) / float(sample_rate_m)), 2)
    xs = np.linspace(east.min(), east.max(), res_grid)
    ys = np.linspace(north.min(), north.max(), res_grid)
    X, Y = np.meshgrid(xs, ys)

    grid = griddata(pts, zvals, (X, Y), method=method)
    if np.isnan(grid).any() and method != "nearest":
        gnear = griddata(pts, zvals, (X, Y), method="nearest")
        grid[np.isnan(grid)] = gnear[np.isnan(grid)]

    topo_cart = np.zeros((res_grid, res_grid, 3))
    topo_cart[:, :, 0] = X - frame.x0
    topo_cart[:, :, 1] = Y - frame.y0
    topo_cart[:, :, 2] = grid
    return ProjectedDem(topo_cart=topo_cart, frame=frame)


def project_lonlat(lons, lats, frame: LocalFrame, swap_xy: bool = False):
    """``(lons, lats)`` in degrees -> ``(xs, ys)`` in metres, local to the model
    origin.  Standalone (no :class:`Station` objects, no DEM) so the Survey tab
    can preview the projected station layout without running the DEM steps.

    ``swap_xy`` mirrors :func:`project_stations` (GoFEM convention X = northing);
    the map view leaves it ``False`` for a conventional east-right plot.
    """
    east, north = _transformer(frame).transform(np.asarray(lons, dtype="float64"),
                                                np.asarray(lats, dtype="float64"))
    de = np.asarray(east) - frame.x0
    dn = np.asarray(north) - frame.y0
    return (dn, de) if swap_xy else (de, dn)


def epsg_name(epsg: int) -> str:
    """Human-readable CRS name for *epsg* (best effort; falls back to the code)."""
    try:
        from pyproj import CRS

        return CRS.from_epsg(int(epsg)).name
    except Exception:  # noqa: BLE001 -- pyproj missing, or not a real EPSG
        return f"EPSG:{epsg}"


def project_stations(stations, frame: LocalFrame, swap_xy: bool = False
                     ) -> tuple[list[list[float]], list[str]]:
    """Return ``(receivers, receiver_names)`` in local metres about the origin.

    *stations* is a list of objects with ``.lon``, ``.lat`` and ``.station_id``
    (:class:`mtinv_gui.backends.edi_io.Station`).

    ``swap_xy=False`` (default) -> ``[easting, northing, 0]``, conventional and
    aligned with ``topo_cart`` / every map.  ``swap_xy=True`` ->
    ``[northing, easting, 0]``, the GoFEM mesh frame (X = North); only
    ``backends/mesh.py`` needs that, and asks for it explicitly.
    """
    tr = _transformer(frame)
    receivers: list[list[float]] = []
    names: list[str] = []
    for st in stations:
        east, north = tr.transform(st.lon, st.lat)
        de, dn = east - frame.x0, north - frame.y0
        receivers.append([dn, de, 0.0] if swap_xy else [de, dn, 0.0])
        names.append(st.station_id)
    return receivers, names


def save_projected_dem(pdem: ProjectedDem, out_path: str) -> str:
    """``np.savez_compressed(out_path, topo_cart)``, plus ``regional=`` when there is one
    (``arr_0`` stays the detailed grid, so readers that only know it still work)."""
    extra = {"regional": pdem.regional} if pdem.regional is not None else {}
    np.savez_compressed(out_path, pdem.topo_cart, **extra)
    return out_path if out_path.endswith(".npz") else out_path + ".npz"


def load_projected_dem(path: str, epsg: int = 0) -> ProjectedDem:
    npz = np.load(path)
    tc = npz["arr_0"]
    regional = npz["regional"] if "regional" in npz.files else None
    return ProjectedDem(topo_cart=tc, frame=LocalFrame(epsg=int(epsg)), regional=regional)


def read_polygons(path: str, frame: LocalFrame):
    """The polygons in a vector file (shapefile, GeoPackage, GeoJSON, KML, ...) as one
    shapely geometry in the model frame, ``(easting, northing)`` metres from the origin.
    A file without a CRS is taken as longitude / latitude."""
    import geopandas as gpd
    from shapely.affinity import translate
    from shapely.ops import unary_union

    gdf = gpd.read_file(path)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    geoms = [g for g in gdf.to_crs(_target_crs(frame)).geometry
             if g is not None and g.geom_type in ("Polygon", "MultiPolygon")]
    if not geoms:
        raise ValueError(f"{os.path.basename(path)} has no polygons")
    return translate(unary_union(geoms), -frame.x0, -frame.y0)


def save_frame(frame: LocalFrame, path: str) -> str:
    """The model frame as JSON (``model_frame.json``), so later steps rebuild the
    same one with :func:`load_frame`.  The CRS is written too, for other tools:
    model ``(x, y)`` = CRS ``(easting - x0, northing - y0)``, with x = east here
    (the mesh itself has x = north)."""
    import dataclasses
    import json
    import warnings

    crs = _target_crs(frame)
    epsg = crs.to_epsg()
    with warnings.catch_warnings():          # "may lose information": not for these simple CRSs
        warnings.simplefilter("ignore", UserWarning)
        crs_text = f"EPSG:{epsg}" if epsg else crs.to_proj4()
    out = {**dataclasses.asdict(frame), "label": frame_label(frame), "crs": crs_text}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    return path


def load_frame(path: str) -> LocalFrame:
    """The frame :func:`save_frame` wrote."""
    import json

    with open(path) as fh:
        d = json.load(fh)
    return make_local_frame(d.get("kind", "local"), d.get("epsg", 0), d["origin_lon"],
                            d["origin_lat"], d.get("local_proj", "tmerc"),
                            d.get("lat1", 0.0), d.get("lat2", 0.0))


def station_spacing_stats(receivers) -> dict:
    """Mean / median / min / max nearest-neighbour distance."""
    from scipy.spatial.distance import cdist

    xy = np.asarray(receivers, dtype="float64")[:, :2]
    d = cdist(xy, xy)
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return {
        "mean_m": float(nn.mean()), "median_m": float(np.median(nn)),
        "min_m": float(nn.min()), "max_m": float(nn.max()),
    }
