"""Other datasets drawn with the model: rasters, depth surfaces, vector
layers, point tables.

Everything is brought into the **display CRS** on load (vectors, points) or on
sampling (rasters -- sampled at display-CRS points through pyproj, so a raster
in any CRS lines up with the model without a separate warp step).  Loaded
layers are cached on the session keyed by file, read settings and display CRS.

Profiles are bilinear samples at the path's own points, which also works for
paths that are not straight N-S / E-W.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import numpy as np

KINDS = {"raster": "raster (values)", "surface": "depth surface",
         "vector": "vector (shp / gpkg / kml)", "points": "point table (CSV)"}
SURFACE_REFS = {"depth_bsl": "depth below sea level", "depth_bsurf": "depth below surface",
                "elevation": "elevation"}
MAP_MODES = {"raster": ["none", "image", "contours"], "surface": ["none", "image", "contours"],
             "vector": ["none", "show"], "points": ["none", "show"]}
SECTION_MODES = {"raster": ["none", "profile"], "surface": ["none", "line", "profile"],
                 "vector": ["none", "markers"], "points": ["none", "markers"]}

#: length unit -> metres
LENGTH_UNITS = {"m": 1.0, "km": 1000.0, "ft": 0.3048}
DEPTH_UNITS = {"auto": "auto (from the values)", "m": "metres", "km": "kilometres", "ft": "feet"}

RASTER_EXT = (".tif", ".tiff", ".nc", ".grd", ".cdf", ".npz", ".asc", ".img", ".ers")
VECTOR_EXT = (".shp", ".gpkg", ".kml", ".kmz", ".geojson", ".json", ".gdb", ".gml", ".tab")
TABLE_EXT = (".csv", ".txt", ".dat", ".xyz", ".tsv")


def guess_kind(path: str) -> str:
    ext = os.path.splitext(path.rstrip("/\\"))[1].lower()
    if ext in VECTOR_EXT or os.path.isdir(path):
        return "vector"
    if ext in TABLE_EXT:
        return "points"
    return "raster"


# --------------------------------------------------------------------------- #
@dataclass
class LoadedRaster:
    """Pixel-centre axes ``xs`` / ``ys`` (ascending) in ``crs``; ``values`` is
    ``(ny, nx)``, NaN = nodata, already scaled."""
    xs: np.ndarray
    ys: np.ndarray
    values: np.ndarray
    crs: object                       # pyproj.CRS
    to_m: float = 1.0                 # depth surfaces: values x to_m = metres
    length_unit: str = ""             # depth surfaces: the unit of ``values``
    unit_guessed: bool = False        # ... worked out from the values ("auto")
    _interp: object = None
    _tr: dict = field(default_factory=dict)

    def percentile_range(self, lo: float = 2.0, hi: float = 98.0) -> tuple[float, float]:
        v = self.values[np.isfinite(self.values)]
        if not v.size:
            return 0.0, 1.0
        a, b = np.percentile(v, [lo, hi])
        return (float(a), float(b)) if b > a else (float(a), float(a) + 1.0)

    def sample(self, E, N, display_crs) -> np.ndarray:
        """Bilinear values at display-CRS points (NaN off the grid / at nodata)."""
        from pyproj import Transformer
        from scipy.interpolate import RegularGridInterpolator

        E = np.asarray(E, dtype="float64")
        N = np.asarray(N, dtype="float64")
        if self.crs != display_crs:
            key = display_crs.to_wkt()
            tr = self._tr.get(key)
            if tr is None:
                tr = self._tr[key] = Transformer.from_crs(display_crs, self.crs, always_xy=True)
            X, Y = tr.transform(E, N)
            X, Y = np.asarray(X), np.asarray(Y)
        else:
            X, Y = E, N
        if self._interp is None:
            self._interp = RegularGridInterpolator((self.ys, self.xs), self.values,
                                                   method="linear", bounds_error=False,
                                                   fill_value=np.nan)
        return self._interp(np.c_[np.ravel(Y), np.ravel(X)]).reshape(np.shape(E))


@dataclass
class LoadedVector:
    gdf: object                       # GeoDataFrame in the display CRS
    labels: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.gdf)


@dataclass
class LoadedPoints:
    E: np.ndarray
    N: np.ndarray
    values: np.ndarray | None = None
    labels: list | None = None
    depth: np.ndarray | None = None   # metres, meaning per surface_ref
    columns: list = field(default_factory=list)
    length_unit: str = ""             # unit of the depth column (before conversion)
    unit_guessed: bool = False

    @property
    def n(self) -> int:
        return len(self.E)


# --------------------------------------------------------------------------- #
def _crs(text):
    from pyproj import CRS

    return CRS.from_user_input(text)


def _apply_validity(a: np.ndarray, info) -> np.ndarray:
    a = np.asarray(a, dtype="float64")
    a = np.where(np.isfinite(a) & (a >= info.valid_min) & (a <= info.valid_max), a, np.nan)
    return a * float(info.scale)


def guess_length_unit(values) -> str:
    """``"km"`` or ``"m"`` for a set of depths / elevations.

    The depths MT sections show -- basement, Curie point, Moho, LAB -- are
    hundreds to hundreds of thousands of metres, i.e. ~0.1-700 in km: if 95 %
    of the values are below 300 they are taken as km.  A shallow surface
    genuinely in metres (all < 300 m) is misread -- the GUI shows what was
    assumed, and ``depth_unit`` overrides it."""
    v = np.abs(np.asarray(values, dtype="float64"))
    v = v[np.isfinite(v)]
    if not v.size:
        return "m"
    return "km" if np.percentile(v, 95) < 300.0 else "m"


def depth_factor(info, values) -> tuple[float, str, bool]:
    """``(metres per value, unit, guessed)`` for a depth surface / CSV depths."""
    unit = (info.depth_unit or "auto").lower()
    guessed = unit not in LENGTH_UNITS
    if guessed:
        unit = guess_length_unit(values)
    return LENGTH_UNITS[unit], unit, guessed


def load_raster(info, maptr) -> LoadedRaster:
    """GeoTIFF / GMT / anything rasterio opens; NetCDF via the setup GUI's DEM
    reader (geographic grids); ``.npz`` as written by the setup GUI --
    ``topo_cart`` (model local frame) or a lon / lat / z triple.

    Values stay in the file's unit (times ``scale``) for maps and profiles;
    for a depth surface, ``to_m`` converts them to metres for sections."""
    r = _load_raster_any(info, maptr)
    if info.kind == "surface":
        r.to_m, r.length_unit, r.unit_guessed = depth_factor(info, r.values)
    return r


def _load_raster_any(info, maptr) -> LoadedRaster:
    path = info.path
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return _load_npz(info, maptr)
    try:
        return _load_rasterio(info)
    except Exception as exc:  # noqa: BLE001 -- fall back for NetCDF without GDAL support
        if ext not in (".nc", ".grd", ".cdf"):
            raise
        from ...backends.topography import read_dem

        try:
            dem = read_dem(path)
        except Exception:  # noqa: BLE001
            raise exc from None
        crs = _crs(info.crs or "EPSG:4326")
        return LoadedRaster(np.asarray(dem.lon, float), np.asarray(dem.lat, float),
                            _apply_validity(dem.z, info), crs)


def _load_rasterio(info) -> LoadedRaster:
    import rasterio

    with rasterio.open(info.path) as src:
        band = min(max(int(info.band), 1), src.count)
        a = src.read(band).astype("float64")
        if src.nodata is not None and np.isfinite(src.nodata):
            a[a == src.nodata] = np.nan
        elif src.nodata is not None:
            a[~np.isfinite(a)] = np.nan
        a[a <= -1.0e30] = np.nan            # float32-min "nodata" written without a tag
        T = src.transform
        if T.b != 0 or T.d != 0:
            raise ValueError(f"{os.path.basename(info.path)}: rotated rasters are not supported")
        crs_text = info.crs or (src.crs.to_wkt() if src.crs else "")
        if not crs_text:
            raise ValueError(f"{os.path.basename(info.path)} has no CRS -- set one in the layer")
        xs = T.c + (np.arange(src.width) + 0.5) * T.a
        ys = T.f + (np.arange(src.height) + 0.5) * T.e
    if xs[0] > xs[-1]:
        xs, a = xs[::-1], a[:, ::-1]
    if ys[0] > ys[-1]:
        ys, a = ys[::-1], a[::-1, :]
    return LoadedRaster(xs, ys, _apply_validity(a, info), _crs(crs_text))


def _load_npz(info, maptr) -> LoadedRaster:
    npz = np.load(info.path)
    keys = list(npz.keys())
    first = npz[keys[0]]
    if first.ndim == 3 and first.shape[2] == 3:
        # setup GUI's projected DEM: topo_cart [..., 0/1/2] = local x / y / z
        tc = first
        xs, ys = tc[0, :, 0], tc[:, 0, 1]
        z = tc[:, :, 2]
        if np.ptp(xs) == 0:               # stored transposed
            xs, ys, z = tc[:, 0, 0], tc[0, :, 1], tc[:, :, 2].T
        g = maptr.georef
        crs = _crs(info.crs or g.crs)
        xs = xs + (0.0 if info.crs else g.x0)
        ys = ys + (0.0 if info.crs else g.y0)
    elif len(keys) >= 3:
        lon, lat, z = (np.asarray(npz[k], float) for k in keys[:3])
        if lon.ndim == 2:
            xs = np.unique(lon.round(10))
            ys = np.unique(lat.round(10))
            if len(xs) * len(ys) != z.size:
                raise ValueError("lon/lat/z npz is not a regular grid")
            order = np.lexsort((lon.ravel(), lat.ravel()))
            z = z.ravel()[order].reshape(len(ys), len(xs))
        else:
            xs, ys = lon, lat
        crs = _crs(info.crs or "EPSG:4326")
    else:
        raise ValueError("npz must hold topo_cart or lon / lat / z arrays")
    xs, ys, z = np.asarray(xs, float), np.asarray(ys, float), np.asarray(z, float)
    if xs[0] > xs[-1]:
        xs, z = xs[::-1], z[:, ::-1]
    if ys[0] > ys[-1]:
        ys, z = ys[::-1], z[::-1, :]
    return LoadedRaster(xs, ys, _apply_validity(z, info), crs)


def list_sublayers(path: str) -> list[str]:
    try:
        import pyogrio

        return [str(r[0]) for r in pyogrio.list_layers(path)]
    except Exception:  # noqa: BLE001
        return []


def load_vector(info, maptr) -> LoadedVector:
    import geopandas as gpd

    kw = {"layer": info.sublayer} if info.sublayer else {}
    try:
        import pyogrio  # noqa: F401
        kw["engine"] = "pyogrio"     # older geopandas defaults to fiona, which won't read KML
    except ImportError:
        pass
    gdf = gpd.read_file(info.path, **kw)
    if info.crs:
        gdf = gdf.set_crs(info.crs, allow_override=True)
    elif gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs(maptr.display_crs)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    labels = []
    if info.label_field and info.label_field in gdf.columns:
        labels = [("" if v is None else str(v)) for v in gdf[info.label_field]]
    return LoadedVector(gdf.reset_index(drop=True), labels)


def vector_fields(path: str, sublayer: str = "") -> list[str]:
    try:
        import pyogrio

        kw = {"layer": sublayer} if sublayer else {}
        return [c for c in pyogrio.read_info(path, **kw)["fields"]]
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------- #
_NUM_BLANKS = re.compile(r"^[+-]?[\d ]*\.?[\d ]*([eE][+-]?\d+)?$")


def _to_number(text) -> float:
    """Float, also for Fortran fixed-width output that writes zeros as blanks
    (``"-130. 22103"`` -> -130.022103)."""
    t = str(text).strip()
    if not t:
        return np.nan
    try:
        return float(t)
    except ValueError:
        pass
    if " " in t and _NUM_BLANKS.match(t):
        try:
            return float(t.replace(" ", "0"))
        except ValueError:
            pass
    return np.nan


def read_table(path: str):
    """``pandas.DataFrame`` of a CSV / whitespace table, all cells as text,
    column names stripped."""
    import pandas as pd

    with open(path, "r", errors="replace") as fh:
        head = fh.readline()
    sep = "," if "," in head else (";" if ";" in head else (r"\t" if "\t" in head else r"\s+"))
    df = pd.read_csv(path, sep=sep, dtype=str, engine="python", skipinitialspace=False,
                     keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def table_columns(path: str) -> list[str]:
    try:
        return list(read_table(path).columns)
    except Exception:  # noqa: BLE001
        return []


_X_NAMES = ("lon", "long", "longitude", "x", "easting", "east", "e")
_Y_NAMES = ("lat", "latitude", "y", "northing", "north", "n")


def _pick(cols, wanted, explicit):
    if explicit:
        return explicit if explicit in cols else None
    low = {c.lower(): c for c in cols}
    for w in wanted:
        if w in low:
            return low[w]
    return None


def load_points(info, maptr) -> LoadedPoints:
    df = read_table(info.path)
    cols = list(df.columns)
    xc = _pick(cols, _X_NAMES, info.x_col)
    yc = _pick(cols, _Y_NAMES, info.y_col)
    if xc is None or yc is None:
        raise ValueError(f"could not find coordinate columns in {os.path.basename(info.path)} "
                         f"(columns: {', '.join(cols)}) -- pick them in the layer")
    x = np.array([_to_number(v) for v in df[xc]])
    y = np.array([_to_number(v) for v in df[yc]])
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]

    def num(col):
        return np.array([_to_number(v) for v in df[col]])[ok]

    if info.crs:
        from pyproj import Transformer

        tr = Transformer.from_crs(_crs(info.crs), maptr.display_crs, always_xy=True)
        E, N = (np.asarray(v) for v in tr.transform(x, y))
    elif np.all(np.abs(x) <= 180.0) and np.all(np.abs(y) <= 90.0):
        E, N = maptr.from_lonlat(x, y)
    else:
        E, N = x, y                      # assume the display CRS
    values = _apply_validity(num(info.value_col), info) if info.value_col in cols else None
    labels = [str(v).strip() for v in df[info.label_col][ok]] if info.label_col in cols else None
    depth, unit, guessed = None, "", False
    if info.depth_col in cols:
        depth = num(info.depth_col) * float(info.scale)
        f, unit, guessed = depth_factor(info, depth)
        depth = depth * f
    return LoadedPoints(np.asarray(E, float), np.asarray(N, float), values, labels, depth, cols,
                        unit, guessed)


# --------------------------------------------------------------------------- #
def read_key(info, maptr) -> tuple:
    """Cache identity: everything that changes what is *read*."""
    try:
        mtime = os.path.getmtime(info.path)
    except OSError:
        mtime = 0.0
    return (info.kind, info.path, mtime, info.crs, int(info.band), info.sublayer,
            float(info.valid_min), float(info.valid_max), float(info.scale), info.depth_unit,
            info.x_col, info.y_col, info.value_col, info.label_col, info.depth_col,
            info.label_field, maptr.key)


def load(info, maptr):
    if info.kind in ("raster", "surface"):
        return load_raster(info, maptr)
    if info.kind == "vector":
        return load_vector(info, maptr)
    return load_points(info, maptr)


def value_range(info, loaded) -> tuple[float, float]:
    if not info.auto_range:
        return float(info.vmin), float(info.vmax)
    if isinstance(loaded, LoadedRaster):
        return loaded.percentile_range()
    if isinstance(loaded, LoadedPoints) and loaded.values is not None:
        v = loaded.values[np.isfinite(loaded.values)]
        if v.size:
            a, b = np.percentile(v, [2, 98])
            return float(a), float(max(b, a + 1e-9))
    return float(info.vmin), float(info.vmax)


def surface_elevation(info, values, model_surface_z) -> np.ndarray:
    """Depth-surface values in **metres** -> elevation, per ``surface_ref``."""
    v = np.asarray(values, dtype="float64")
    if info.surface_ref == "elevation":
        return v
    if info.surface_ref == "depth_bsurf":
        return np.asarray(model_surface_z, dtype="float64") - v
    return -v


def label(info, loaded=None) -> str:
    """``"name (units)"``; a depth surface without units gets its length unit."""
    units = info.units or getattr(loaded, "length_unit", "")
    return f"{info.name} ({units})" if units else info.name


def unit_note(info, loaded) -> str:
    """One line on how depths were read, for the GUI ("" if not a depth layer)."""
    unit = getattr(loaded, "length_unit", "")
    if not unit:
        return ""
    how = "worked out from the values -- set 'depth units' if wrong" if loaded.unit_guessed \
        else "as set"
    return f"depths in {unit} ({how}); sections convert to metres"


# --------------------------------------------------------------------------- #
@dataclass
class MapGrid:
    E: np.ndarray
    N: np.ndarray
    values: np.ndarray


def grid_on_extent(raster: LoadedRaster, maptr, extent, n: int = 500) -> MapGrid:
    """The raster resampled on a display-CRS grid covering *extent* -- no finer
    than the raster's own pixels."""
    x0, x1, y0, y1 = extent
    # approximate pixel size in display metres
    px = max(float(np.median(np.diff(raster.xs))), float(np.median(np.diff(raster.ys))))
    if raster.crs.is_geographic:
        px *= 111_000.0
    nx = int(min(n, max((x1 - x0) / max(px, 1e-9), 2)))
    ny = int(min(n * (y1 - y0) / max(x1 - x0, 1.0), max((y1 - y0) / max(px, 1e-9), 2)))
    E, N = np.meshgrid(np.linspace(x0, x1, max(nx, 2)), np.linspace(y0, y1, max(ny, 2)))
    return MapGrid(E, N, raster.sample(E, N, maptr.display_crs))


def contour_levels(info, vrange) -> np.ndarray | int:
    if info.contour_interval > 0:
        a, b = vrange
        d = float(info.contour_interval)
        return np.arange(np.floor(a / d) * d, b + 0.5 * d, d)
    return 10


# --------------------------------------------------------------------------- #
@dataclass
class SectionItems:
    """What a layer contributes to one section."""
    info: object
    kind: str                         # "profile" | "line" | "markers"
    s: np.ndarray
    values: np.ndarray                # profile: values; line / markers: elevation (m)
    labels: list = field(default_factory=list)
    vrange: tuple = (0.0, 1.0)        # profile axis range -- the layer's, so every
                                      # section of a set shares it


def section_items(info, loaded, prof, model, maptr) -> SectionItems | None:
    mode = info.section_mode
    if mode == "none":
        return None
    if isinstance(loaded, LoadedRaster):
        s = np.linspace(0.0, prof.length, 800)
        x, y, E, N = prof.at(s)
        v = loaded.sample(E, N, maptr.display_crs)
        if mode == "line" and info.kind == "surface":
            return SectionItems(info, "line", s,
                                surface_elevation(info, v * loaded.to_m, model.surface()(x, y)))
        return SectionItems(info, "profile", s, v, vrange=value_range(info, loaded))
    if mode != "markers":
        return None
    if isinstance(loaded, LoadedPoints):
        s, off = prof.project(loaded.E, loaded.N)
        keep = (off <= float(info.section_offset_m)) & (s > 0) & (s < prof.length)
        s = s[keep]
        x, y, _E, _N = prof.at(s)
        surf = model.surface()(x, y)
        z = surface_elevation(info, loaded.depth[keep], surf) if loaded.depth is not None else surf
        labs = [loaded.labels[i] for i in np.where(keep)[0]] if loaded.labels else []
        return SectionItems(info, "markers", s, z, labs)
    if isinstance(loaded, LoadedVector):
        return _vector_on_section(info, loaded, prof, model)
    return None


def _vector_on_section(info, vec: LoadedVector, prof, model) -> SectionItems:
    """Points within the offset (projected), and where lines / polygon edges
    cross the section line."""
    from shapely.geometry import LineString, MultiPoint, Point

    line = LineString(prof.disp)
    buf = line.buffer(float(info.section_offset_m))
    s_out, labs = [], []
    for i, geom in enumerate(vec.gdf.geometry):
        if geom is None or not geom.intersects(buf):
            continue
        lab = vec.labels[i] if vec.labels else ""
        gt = geom.geom_type
        if gt in ("Point", "MultiPoint"):
            pts = [geom] if gt == "Point" else list(geom.geoms)
            for p in pts:
                if p.distance(line) <= info.section_offset_m:
                    s_out.append(line.project(p))
                    labs.append(lab)
            continue
        edge = geom.boundary if gt in ("Polygon", "MultiPolygon") else geom
        hit = edge.intersection(line)
        if hit.is_empty:
            continue
        if isinstance(hit, Point):
            pts = [hit]
        elif isinstance(hit, MultiPoint):
            pts = list(hit.geoms)
        else:                              # collinear overlap -> use its midpoint(s)
            pts = [g.interpolate(0.5, normalized=True) for g in getattr(hit, "geoms", [hit])]
        for p in pts:
            s_out.append(line.project(p))
            labs.append(lab)
    s = np.asarray(s_out, dtype="float64")
    x, y, _E, _N = prof.at(s)
    z = model.surface()(x, y) if len(s) else np.zeros(0)
    return SectionItems(info, "markers", s, z, labs)
