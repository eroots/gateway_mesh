"""Task 1 -- obtain and prepare topography / bathymetry for the survey region.

Downloads a DEM (or takes one you already have), clips to ``survey bbox +/-
padding``, resamples onto a regular ``d_ll`` grid with ``RegularGridInterpolator``,
fills NaNs and writes ``np.savez(topo_big, lon_grid, lat_grid, z_grid)``.

    provider        key?   coverage                     notes
    --------        ----   --------                     -----
    GMRT            no     global topo + bathymetry     ~variable res, GeoTIFF (EPSG:4326)
    OpenTopography  yes    SRTM/COP/NASADEM/GEBCO       demtype selectable, GeoTIFF
    local           -      user-supplied GeoTIFF/NetCDF

Output of :func:`resample_dem` / :func:`save_dem_npz` has the
``topo_big.npz`` layout: ``arr_0=lon_grid, arr_1=lat_grid, arr_2=z_grid``
(all 2-D meshgrids), which :mod:`mtinv_gui.backends.projection` consumes.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

GMRT_GRIDSERVER = "https://www.gmrt.org/services/GridServer"
OPENTOPO_GLOBALDEM = "https://portal.opentopography.org/API/globaldem"


@dataclass
class DemGrid:
    """Regular geographic DEM.

    ``lon`` / ``lat`` are 1-D ascending cell-centre vectors; ``z`` is
    ``(nlat, nlon)`` metres (positive up, NaN where missing).
    """
    lon: np.ndarray
    lat: np.ndarray
    z: np.ndarray
    crs_epsg: int = 4326

    def meshgrid(self):
        return np.meshgrid(self.lon, self.lat)

    @property
    def shape(self):
        return self.z.shape


_M_PER_DEG_LAT = 111_320.0    # ~ mean metres per degree of latitude


def arcsec_to_km(arcsec: float, lat_deg: float) -> tuple[float, float]:
    """Approx (north-south, east-west) kilometres for *arcsec* at *lat_deg*."""
    import math

    m_ns = arcsec / 3600.0 * _M_PER_DEG_LAT
    m_ew = m_ns * math.cos(math.radians(lat_deg))
    return m_ns / 1000.0, m_ew / 1000.0


def dem_resolution(dem: "DemGrid") -> dict:
    """Grid posting of a DEM in degrees / arc-seconds / metres (at mid-latitude)."""
    import math

    dlon = float(np.abs(np.diff(dem.lon)).mean()) if dem.lon.size > 1 else float("nan")
    dlat = float(np.abs(np.diff(dem.lat)).mean()) if dem.lat.size > 1 else float("nan")
    mid_lat = float(np.mean(dem.lat))
    return {
        "nx": int(dem.lon.size), "ny": int(dem.lat.size),
        "dlon_deg": dlon, "dlat_deg": dlat,
        "dlon_arcsec": dlon * 3600.0, "dlat_arcsec": dlat * 3600.0,
        "dx_m": dlon * _M_PER_DEG_LAT * math.cos(math.radians(mid_lat)),
        "dy_m": dlat * _M_PER_DEG_LAT,
    }


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def _padded_bbox(bbox: dict, pad_deg: float) -> tuple[float, float, float, float]:
    return (bbox["MinLon"] - pad_deg, bbox["MaxLon"] + pad_deg,
            bbox["MinLat"] - pad_deg, bbox["MaxLat"] + pad_deg)


# --------------------------------------------------------------------------- #
# size estimates (before downloading / resampling)
# --------------------------------------------------------------------------- #
#: OpenTopography global DEMs: native posting (arc-seconds) and the API's area
#: limit per request (km^2), from opentopography.org/developers
OPENTOPO_SPECS = {
    "SRTMGL1": (1.0, 450_000.0),
    "NASADEM": (1.0, 450_000.0),
    "COP30": (1.0, 450_000.0),
    "SRTMGL3": (3.0, 4_050_000.0),
    "COP90": (3.0, 4_050_000.0),
    "GEBCOIceTopo": (15.0, 125_000_000.0),
}
#: GMRT's "default" resolution: the finest of its levels (longitude spacing
#: 9/512 deg x 2^k; latitude spacing that x cos(latitude), square in metres)
#: with fewer than 1000 nodes in both directions; "med" / "high" halve /
#: quarter that.  Measured: 1-2 MB NetCDF, 4 bytes per node.
GMRT_BASE_DLON = 9.0 / 512.0
GMRT_MAX_NODES = 1000
GMRT_STEPS = {"default": 1, "med": 2, "high": 4}
GMRT_FINEST_K = -5                # 9/512/32 deg, ~50 m: the finest "default" seen


def box_area_km2(w: float, e: float, s: float, n: float) -> float:
    """Area of a lon/lat box on a sphere."""
    import math

    R = 6371.0
    return (R * R * math.radians(max(e - w, 0.0))
            * abs(math.sin(math.radians(n)) - math.sin(math.radians(s))))


def _grid_estimate(w, e, s, n, dlon, dlat, bytes_per_node) -> dict:
    import math

    nx = int(math.ceil((e - w) / dlon + 0.5))     # = len(np.arange(w, e + d / 2, d))
    ny = int(math.ceil((n - s) / dlat + 0.5))
    clat = 0.5 * (s + n)
    return {"nx": nx, "ny": ny, "nodes": nx * ny,
            "dx_m": dlon * _M_PER_DEG_LAT * math.cos(math.radians(clat)),
            "dy_m": dlat * _M_PER_DEG_LAT,
            "mb": nx * ny * bytes_per_node / 1e6,
            "area_km2": box_area_km2(w, e, s, n)}


def estimate_download(provider: str, bbox: dict, pad_deg: float, demtype: str = "SRTMGL1",
                      resolution: str = "default") -> dict | None:
    """What :func:`download` would fetch: ``nx``, ``ny``, ``nodes``, posting
    ``dx_m`` / ``dy_m``, file size ``mb``, ``area_km2``, plus ``limit_km2``
    (the provider's cap, 0 = none) and ``exact`` (False for an estimate the
    provider may not follow).  None for an unknown provider / dataset."""
    import math

    w, e, s, n = _padded_bbox(bbox, pad_deg)
    s, n = max(s, -90.0), min(n, 90.0)
    if not (e > w and n > s):
        return None
    if provider == "OpenTopography":
        spec = OPENTOPO_SPECS.get(demtype)
        if spec is None:
            return None
        d = spec[0] / 3600.0
        # GeoTIFFs: SRTM / NASADEM are 16-bit, Copernicus and GEBCO 32-bit floats
        bpn = 2.0 if demtype.startswith(("SRTM", "NASADEM")) else 4.0
        out = _grid_estimate(w, e, s, n, d, d, bpn)
        out.update(limit_km2=spec[1], exact=True)
        return out
    if provider == "GMRT":
        cosl = max(math.cos(math.radians(0.5 * (s + n))), 1e-3)
        k = 0
        while True:                               # coarsen until both sides fit
            dlon = GMRT_BASE_DLON * 2.0 ** k
            if (e - w) / dlon < GMRT_MAX_NODES and (n - s) / (dlon * cosl) < GMRT_MAX_NODES:
                break
            k += 1
        while k > GMRT_FINEST_K:                  # refine while both still fit
            dlon = GMRT_BASE_DLON * 2.0 ** (k - 1)
            if (e - w) / dlon >= GMRT_MAX_NODES or (n - s) / (dlon * cosl) >= GMRT_MAX_NODES:
                break
            k -= 1
        dlon = GMRT_BASE_DLON * 2.0 ** k / GMRT_STEPS.get(resolution, 1)
        out = _grid_estimate(w, e, s, n, dlon, dlon * cosl, 4.0)
        out.update(limit_km2=0.0, exact=False)
        return out
    return None


def estimate_resample(bbox: dict, pad_arcsec: float, sample_arcsec: float) -> dict:
    """Size of :func:`resample_dem`'s grid and its ``.npz`` (lon, lat and z
    grids, 8 bytes each per node); ``peak_mb`` is the memory resampling needs
    (about 7 grids of that size at once)."""
    w, e, s, n = _padded_bbox(bbox, pad_arcsec / 3600.0)
    d = max(sample_arcsec, 1e-6) / 3600.0
    out = _grid_estimate(w, e, s, n, d, d, 24.0)
    out["peak_mb"] = out["nodes"] * 8.0 * 7 / 1e6
    return out


def describe_estimate(est: dict | None) -> str:
    """One line, e.g. ``"1,234 x 987 nodes (~95 m), ~2.4 MB"``."""
    if est is None:
        return "size unknown"
    mb = est["mb"]
    size = f"{mb / 1e3:.1f} GB" if mb >= 1e3 else (f"{mb:.0f} MB" if mb >= 10 else f"{mb:.1f} MB")
    return (f"{est['nx']:,} x {est['ny']:,} nodes ({est['nodes'] / 1e6:.3g} M, "
            f"~{0.5 * (est['dx_m'] + est['dy_m']):,.0f} m), {'~' if not est.get('exact', True) else ''}"
            f"{size}")


def download_gmrt(bbox: dict, out_path: str, pad_deg: float = 1.0,
                  resolution: str = "default", layer: str = "topo",
                  fmt: str = "netcdf", timeout: int = 120) -> str:
    """Download a GMRT grid for ``bbox +/- pad_deg``.

    ``fmt`` defaults to ``netcdf`` (COARDS lon/lat/z) so the read path needs no
    GDAL/libtiff -- just scipy.  ``resolution`` in {low, med, high, max,
    default}; ``layer`` in {topo, topo-mask}.  No API key required.  The written
    file gets the extension matching *fmt*; the actual path is returned.
    """
    import requests

    ext = {"netcdf": ".nc", "geotiff": ".tif", "esriascii": ".asc"}.get(fmt, ".nc")
    if not out_path.endswith(ext):
        out_path = os.path.splitext(out_path)[0] + ext
    w, e, s, n = _padded_bbox(bbox, pad_deg)
    params = {
        "minlongitude": w, "maxlongitude": e,
        "minlatitude": s, "maxlatitude": n,
        "format": fmt, "resolution": resolution, "layer": layer,
    }
    resp = requests.get(GMRT_GRIDSERVER, params=params, timeout=timeout)
    resp.raise_for_status()
    if len(resp.content) < 1024:
        raise RuntimeError(f"GMRT returned {len(resp.content)} bytes: {resp.text[:200]}")
    with open(out_path, "wb") as fh:
        fh.write(resp.content)
    return out_path


def download_opentopography(bbox: dict, out_path: str, api_key: str,
                            demtype: str = "SRTMGL1", pad_deg: float = 1.0,
                            timeout: int = 180) -> str:
    """Download an OpenTopography global DEM GeoTIFF for the padded bbox.

    ``demtype`` -- see :data:`mtinv_gui.config.OPENTOPO_DEMS`.  Requires a free
    key from https://portal.opentopography.org/.
    """
    import requests

    if not api_key:
        raise RuntimeError("OpenTopography needs an API key (Project > Settings).")
    w, e, s, n = _padded_bbox(bbox, pad_deg)
    params = {
        "demtype": demtype, "west": w, "east": e, "south": s, "north": n,
        "outputFormat": "GTiff", "API_Key": api_key,
    }
    resp = requests.get(OPENTOPO_GLOBALDEM, params=params, timeout=timeout)
    if resp.status_code == 401:
        raise RuntimeError("OpenTopography rejected the API key (401).")
    resp.raise_for_status()
    with open(out_path, "wb") as fh:
        fh.write(resp.content)
    return out_path


def download(provider: str, bbox: dict, out_path: str, *, pad_deg: float = 1.0,
             api_key: str = "", demtype: str = "SRTMGL1",
             resolution: str = "default", gmrt_format: str = "netcdf") -> str:
    """Dispatch to the right provider; returns the actual written file path."""
    if provider == "GMRT":
        return download_gmrt(bbox, out_path, pad_deg=pad_deg,
                             resolution=resolution, fmt=gmrt_format)
    if provider == "OpenTopography":
        tif = os.path.splitext(out_path)[0] + ".tif"
        return download_opentopography(bbox, tif, api_key, demtype=demtype,
                                       pad_deg=pad_deg)
    raise ValueError(f"unknown provider {provider!r}")


# --------------------------------------------------------------------------- #
# read
# --------------------------------------------------------------------------- #
def read_dem(path: str) -> DemGrid:
    """Load a NetCDF/GMT grid (scipy, no GDAL) or a GeoTIFF (rasterio)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".nc", ".grd", ".cdf"):
        return _read_netcdf(path)
    return _read_geotiff(path)


def _read_geotiff(path: str) -> DemGrid:
    try:
        import rasterio
    except ImportError as exc:  # e.g. libtiff.so.5 soname mismatch in the conda env
        raise RuntimeError(
            f"cannot import rasterio ({exc}). GeoTIFF reading needs a working "
            "GDAL. Use the GMRT provider (downloads NetCDF, read without GDAL) "
            "or fix the env: conda install -c conda-forge 'libgdal<3.8' "
            "'libtiff<4.6' 'rasterio=1.3.9'") from exc

    with rasterio.open(path) as ds:
        band = ds.read(1, masked=True).astype("float64")
        z = np.ma.filled(band, np.nan)
        t = ds.transform
        nx, ny = ds.width, ds.height
        lon = t.c + t.a * (np.arange(nx) + 0.5)
        lat = t.f + t.e * (np.arange(ny) + 0.5)
        epsg = ds.crs.to_epsg() if ds.crs else 4326
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        z = z[::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        z = z[:, ::-1]
    if epsg not in (4326, None):
        raise ValueError(
            f"DEM CRS is EPSG:{epsg}; expected geographic EPSG:4326. "
            "Re-download in lon/lat or reproject first.")
    return DemGrid(lon=lon, lat=lat, z=z, crs_epsg=4326)


_LON_NAMES = ("lon", "longitude", "x")
_LAT_NAMES = ("lat", "latitude", "y")
_Z_NAMES = ("elevation", "z", "altitude", "Band1", "topo")


def _read_netcdf(path: str) -> DemGrid:
    """Read a COARDS / GMT NetCDF grid.

    Tries ``scipy.io.netcdf_file`` (handles NetCDF-3 classic, which is what GMRT
    returns) with no GDAL involved, then falls back to xarray for NetCDF-4.
    Also handles the legacy GMT layout (``x_range`` / ``y_range`` / ``dimension``
    / flat ``z``).
    """
    try:
        return _read_netcdf_scipy(path)
    except Exception as exc_scipy:  # noqa: BLE001
        try:
            return _read_netcdf_xarray(path)
        except Exception as exc_xr:  # noqa: BLE001
            raise RuntimeError(
                f"could not read NetCDF grid {path}: scipy path -> {exc_scipy}; "
                f"xarray path -> {exc_xr}") from exc_xr


def _read_netcdf_scipy(path: str) -> DemGrid:
    from scipy.io import netcdf_file

    with netcdf_file(path, "r", mmap=False) as f:
        varnames = list(f.variables)
        if "x_range" in varnames and "z" in varnames:          # legacy GMT grid
            xr_ = f.variables["x_range"][:]
            yr_ = f.variables["y_range"][:]
            dim = f.variables["dimension"][:]
            nx, ny = int(dim[0]), int(dim[1])
            lon = np.linspace(float(xr_[0]), float(xr_[1]), nx)
            lat = np.linspace(float(yr_[0]), float(yr_[1]), ny)
            z = np.asarray(f.variables["z"][:], dtype="float64").reshape(ny, nx)
            z = np.flipud(z)                                    # GMT stores top row first
            return _finalize(lon, lat, z)

        lon_name = next((n for n in _LON_NAMES if n in varnames), None)
        lat_name = next((n for n in _LAT_NAMES if n in varnames), None)
        z_name = next((n for n in _Z_NAMES if n in varnames and n not in (lon_name, lat_name)), None)
        if not (lon_name and lat_name and z_name):
            raise ValueError(f"unrecognised NetCDF variables: {varnames}")
        lon = np.asarray(f.variables[lon_name][:], dtype="float64")
        lat = np.asarray(f.variables[lat_name][:], dtype="float64")
        z = np.asarray(f.variables[z_name][:], dtype="float64")
        fv = getattr(f.variables[z_name], "_FillValue", None)
    if fv is not None:
        z = np.where(z == fv, np.nan, z)
    if z.ndim == 1:
        z = z.reshape(lat.size, lon.size)
    if z.shape == (lon.size, lat.size) and z.shape[0] != z.shape[1]:
        z = z.T
    return _finalize(lon, lat, z)


def _read_netcdf_xarray(path: str) -> DemGrid:
    import xarray as xr

    ds = xr.open_dataset(path)
    lon_name = next((n for n in _LON_NAMES if n in ds.coords or n in ds), None)
    lat_name = next((n for n in _LAT_NAMES if n in ds.coords or n in ds), None)
    z_name = next((n for n in _Z_NAMES if n in ds and n not in (lon_name, lat_name)), None)
    if not (lon_name and lat_name and z_name):
        raise ValueError(f"could not identify lon/lat/elevation vars: {list(ds.variables)}")
    lon = np.asarray(ds[lon_name].values, dtype="float64")
    lat = np.asarray(ds[lat_name].values, dtype="float64")
    z = np.asarray(ds[z_name].values, dtype="float64")
    if z.shape == (lon.size, lat.size) and z.shape[0] != z.shape[1]:
        z = z.T
    return _finalize(lon, lat, z)


def _finalize(lon: np.ndarray, lat: np.ndarray, z: np.ndarray) -> DemGrid:
    if lon[0] > lon[-1]:
        lon, z = lon[::-1], z[:, ::-1]
    if lat[0] > lat[-1]:
        lat, z = lat[::-1], z[::-1, :]
    return DemGrid(lon=lon, lat=lat, z=z, crs_epsg=4326)


# --------------------------------------------------------------------------- #
# clip + resample
# --------------------------------------------------------------------------- #
def fill_nans(z: np.ndarray) -> np.ndarray:
    """Nearest-neighbour fill of NaN holes."""
    mask = np.isnan(z)
    if not mask.any():
        return z
    if mask.all():
        raise ValueError("DEM tile is entirely NaN")
    from scipy.interpolate import NearestNDInterpolator

    yy, xx = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    good = ~mask
    interp = NearestNDInterpolator(np.c_[yy[good], xx[good]], z[good])
    out = z.copy()
    out[mask] = interp(np.c_[yy[mask], xx[mask]])
    return out


def resample_dem(dem: DemGrid, bbox: dict, pad_arcsec: float,
                 sample_arcsec: float) -> DemGrid:
    """Clip to ``bbox +/- pad`` and resample to ``sample_arcsec`` posting."""
    from scipy.interpolate import RegularGridInterpolator

    pad = pad_arcsec / 3600.0
    d = sample_arcsec / 3600.0
    w, e, s, n = _padded_bbox(bbox, pad)
    # keep the request inside the available data
    w, e = max(w, dem.lon.min()), min(e, dem.lon.max())
    s, n = max(s, dem.lat.min()), min(n, dem.lat.max())
    if not (e > w and n > s):
        raise ValueError("padded survey box does not overlap the DEM extent")

    new_lon = np.arange(w, e + d / 2, d)
    new_lat = np.arange(s, n + d / 2, d)
    rgi = RegularGridInterpolator((dem.lat, dem.lon), dem.z,
                                  bounds_error=False, fill_value=np.nan)
    LON, LAT = np.meshgrid(new_lon, new_lat)
    z = rgi((LAT, LON))
    z[z > 2e4] = np.nan          # sanity clip
    z[z < -2e4] = np.nan
    z = fill_nans(z)
    return DemGrid(lon=new_lon, lat=new_lat, z=z, crs_epsg=4326)


# --------------------------------------------------------------------------- #
# persist
# --------------------------------------------------------------------------- #
def save_dem_npz(dem: DemGrid, out_path: str) -> str:
    lon_grid, lat_grid = dem.meshgrid()
    np.savez(out_path, lon_grid, lat_grid, dem.z)
    return out_path if out_path.endswith(".npz") else out_path + ".npz"


def load_dem_npz(path: str) -> DemGrid:
    npz = np.load(path)
    lon_grid, lat_grid, z_grid = npz["arr_0"], npz["arr_1"], npz["arr_2"]
    lon = lon_grid[0, :] if lon_grid.ndim == 2 else np.unique(lon_grid)
    lat = lat_grid[:, 0] if lat_grid.ndim == 2 else np.unique(lat_grid)
    return DemGrid(lon=np.asarray(lon), lat=np.asarray(lat),
                   z=np.asarray(z_grid), crs_epsg=4326)
