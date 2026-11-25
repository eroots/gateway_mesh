"""Global configuration constants, machine-level settings, capability probing."""
from __future__ import annotations

import importlib
import json
import os
import shutil

APP_NAME = "MT Inversion Setup"
ORG_NAME = "mtinv_gui"
PROJECT_FILE_SUFFIX = ".mtinv.json"

# Inversion codes the GUI is being built towards.  Only GOFEM is wired up now.
SUPPORTED_CODES = ["GoFEM"]          # later: "ModEM", "FEMTIC"

# Topography / bathymetry providers.
TOPO_PROVIDERS = ["GMRT", "OpenTopography", "local"]
# OpenTopography global DEM ids (demtype=) offered in the GUI.
OPENTOPO_DEMS = ["SRTMGL1", "SRTMGL3", "COP30", "COP90", "GEBCOIceTopo", "NASADEM"]

# --------------------------------------------------------------------------- #
# Machine-level settings (NOT per project): API keys, recent projects and optional
# paths to development copies of deal.II / pyGoFEM / mtpy.  environment.yml installs
# all three, so those paths normally stay blank.  Stored at
# ~/.mtinv_gui/settings.json so they survive across projects.
# --------------------------------------------------------------------------- #
SETTINGS_DIR = os.path.join(os.path.expanduser("~"), ".mtinv_gui")
SETTINGS_PATH = os.path.join(SETTINGS_DIR, "settings.json")

_SETTINGS_DEFAULTS = {
    "dealii_path": "",          # optional: dir containing PyDealII/ (else import as installed)
    "gofem_path": "",           # optional: checkout of the pyGoFEM repo (else import as installed)
    "mtpy_path": "",            # optional: explicit mtpy source dir (else import as installed)
    "opentopo_api_key": "",     # https://portal.opentopography.org/ -> free key
    "last_project": "",         # path of the most recently saved/opened project
    "last_plot_project": "",    # same, for the model-plotting app (run_plot.py)
    "restore_last_project": True,   # reopen it automatically on startup
}


def load_settings() -> dict:
    data = dict(_SETTINGS_DEFAULTS)
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as fh:
            data.update({k: v for k, v in json.load(fh).items() if k in data})
    except (OSError, ValueError):
        pass
    return data


def save_settings(data: dict) -> str:
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    merged = load_settings()
    merged.update({k: v for k, v in data.items() if k in _SETTINGS_DEFAULTS})
    with open(SETTINGS_PATH, "w", encoding="utf-8") as fh:
        json.dump(merged, fh, indent=2)
    return SETTINGS_PATH


#: which importable package each *_path setting is expected to make available.
_PATH_PACKAGE = {"gofem_path": "gofem", "mtpy_path": "mtpy"}


def apply_sys_paths(settings: dict | None = None) -> list[str]:
    """Prepend configured dealii/gofem/mtpy dirs to ``sys.path``.

    Call once before importing PyDealII / gofem / mtpy.  Returns the paths added.

    The convention (``sys.path.append(gofem_path)`` then
    ``from gofem.parameter_handler import *``) is that these point at the
    *directory containing* the package, not the package folder itself.  If a
    setting points straight at the ``gofem/`` (or ``mtpy/``) package dir, its
    parent is used instead so the import still resolves.
    """
    import sys

    settings = settings or load_settings()
    added = []
    for key in ("dealii_path", "gofem_path", "mtpy_path"):
        raw = (settings.get(key) or "").strip()
        if not raw:
            continue
        p = os.path.expanduser(raw).rstrip("/\\")
        if not os.path.isdir(p):
            continue
        pkg = _PATH_PACKAGE.get(key)
        if (pkg and os.path.basename(p) == pkg
                and os.path.isfile(os.path.join(p, "__init__.py"))):
            p = os.path.dirname(p)
        if p and p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    return added


def ensure_mtpy_numpy_compat() -> None:
    """``mtpy==1.1.5`` reads EDIs through numpy aliases (``np.float``,
    ``np.int``, ...) that numpy removed in 1.24, so any real EDI read through
    mtpy raises ``AttributeError`` without this. Call once before importing
    anything under ``mtpy.*``.

    Restoring the aliases is numpy's own documented fix for exactly this
    situation -- they were never anything but plain ``float``/``int``/etc.,
    so this can't change numpy's behaviour anywhere else, only un-remove
    names mtpy still expects to find.
    """
    import numpy as np

    # numpy 2 brought ``np.bool`` back as a real numpy type; overwriting it with
    # Python's bool breaks numpy itself (masked arrays fail).  mtpy 1.1.5 can't
    # run on numpy 2 anyway, so there is nothing to restore there.
    if int(np.__version__.split(".")[0]) >= 2:
        return
    # unconditional, not "if not hasattr(...)" -- numpy's own __getattr__ hook
    # for these deprecated-but-not-yet-removed names (bool/object/str) warns
    # on *access*, so even checking hasattr() first prints the noise this is
    # trying to avoid. Setting a real attribute always short-circuits that.
    for name, alias in (("float", float), ("int", int), ("bool", bool),
                        ("object", object), ("str", str), ("complex", complex)):
        setattr(np, name, alias)


_MTPY_OK: bool | None = None


def mtpy_available() -> bool:
    """True when ``mtpy`` 1.x can actually be imported here (after the numpy-compat
    shim and any configured ``mtpy_path``).  Memoised -- the import is the slow
    part and the answer is fixed for the run.

    The EDI readers (:mod:`backends.edi_io`, :mod:`backends.edi_z`) are
    mtpy-first: they use mtpy when this is True and fall back to their built-in
    text parser otherwise (or per-file, when mtpy chokes on a quirky EDI).
    """
    global _MTPY_OK
    if _MTPY_OK is None:
        try:
            apply_sys_paths()
            ensure_mtpy_numpy_compat()
            importlib.import_module("mtpy.core.mt")
            # The readers target the mtpy 1.x API (MT().Z.z / .freq, .lat ...).
            # mtpy 2 (what the plotting env ships) imports fine but has a
            # different API, so treat it as "not available" and use the text
            # parser directly instead of failing on every file.
            mtpy = importlib.import_module("mtpy")
            _MTPY_OK = str(getattr(mtpy, "__version__", "1")).split(".")[0] == "1"
        except Exception:
            _MTPY_OK = False
    return _MTPY_OK


def edi_reader_mode() -> str:
    """``"auto"`` (default), ``"mtpy"``, or ``"text"`` -- override with the
    ``MTINV_EDI_READER`` environment variable.

    ``auto`` : mtpy when importable, text parser otherwise, and text per-file
    when an mtpy read raises.  ``mtpy`` / ``text`` pin one path and never fall
    back -- used by the reader-parity tests.
    """
    mode = os.environ.get("MTINV_EDI_READER", "auto").strip().lower()
    return mode if mode in ("auto", "mtpy", "text") else "auto"


# --- Default numeric parameters ----------------------------------------------
DEFAULTS = {
    # topography download / resample
    "topo_pad_deg": 10.0 / 3600.0 * 3600,   # ~1 deg; see topo_panel
    "topo_pad_arcsec": 3600,                 # pad around survey bbox, arc-seconds
    "topo_sample_arcsec": 3.0,               # ~90 m target posting for the resampled DEM
    # projection / local grid
    "dem_sample_rate_m": 200.0,             # horizontal posting of the projected DEM grid
    "grid_method": "nearest",              # scipy.interpolate.griddata method
    # mesh (deal.II subdivided hyper-rectangle)
    "background_resistivity_ohm_m": 150.0,
    "ocean_resistivity_ohm_m": 1.0 / 3.2,
    "air_resistivity_ohm_m": 1.0e8,
    "mesh_air_id": 1,
    "mesh_earth_id": 2,
    "mesh_ocean_id": 3,
    # period / frequency selection
    "period_min_s": 0.01,
    "period_max_s": 10000.0,
    "period_percentage": 10.0,             # keep periods present in >= this % of sites
    # error floors for the data file
    "error_floor_z": 0.05,
    "error_floor_tipper_abs": 0.03,
    "error_floor_pt": 0.05,
}

# Optional third-party packages.  The GUI degrades gracefully when these are
# missing so that the skeleton still launches for layout review.
_OPTIONAL_MODULES = [
    "numpy", "scipy", "matplotlib", "pyproj", "pandas",
    "mtpy",            # EDI collection, period selection, GoFEM data writer
    "rasterio",        # GeoTIFF DEM I/O (Windows-friendly replacement for gdal)
    "xarray",          # NetCDF / GEBCO handling
    "shapely",         # coastline / refinement polygons
    "geopandas",       # station GeoDataFrame + CRS transforms
    "pyvista",         # 3-D mesh preview
    "pyvistaqt",       # embed the 3-D preview in Qt
    "PyDealII",        # deal.II python bindings -- required for mesh generation
    "gofem",           # pyGoFEM: parameter_handler, mesh_utils, mtpy_utils
    "requests",        # topography download
]


def probe_environment() -> dict[str, str | None]:
    """Return ``{module_name: version_or_None}`` for the optional dependencies."""
    apply_sys_paths()           # so a gofem / deal.II path set in Settings counts as found
    found: dict[str, str | None] = {}
    for name in _OPTIONAL_MODULES:
        try:
            mod = importlib.import_module(name)
            found[name] = getattr(mod, "__version__", "installed")
        except Exception:
            found[name] = None
    return found


def has_wsl() -> bool:
    """True when a ``wsl`` executable is on PATH (Windows -> Linux bridge)."""
    return shutil.which("wsl") is not None
