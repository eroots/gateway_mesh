"""Place a model's local frame in a real coordinate reference system.

A model's local coordinates are ``CRS coordinates - (x0, y0)`` in some *model
CRS* -- that is how the setup GUI builds the mesh (``backends/projection.py``:
project the stations, subtract the origin's projected coordinate).  So a
:class:`Georef` is just ``(crs, x0, y0)``.  Three ways to get one:

  * :func:`from_mtinv_project` -- the model was built with the setup GUI: read
    the frame straight from its ``*.mtinv.json``.
  * :func:`fit_from_stations` / :func:`auto_fit` -- older models: match the
    receiver names in the model's receiver file to station coordinates
    (EDIs), and solve for the origin by least squares.  The residual says
    whether the CRS is right (centimetres) or not (kilometres), which is what
    :func:`auto_fit` uses to pick the CRS: a few likely ones, then registered
    grids that cover the survey.
  * :func:`manual` -- type the CRS and origin.

Figures and GeoTIFFs are drawn in a *display CRS* which may differ from the
model CRS (e.g. a model built in UTM 8N shown with UTM 9N rasters).
:class:`MapTransform` converts between the two, so nothing downstream needs
to know.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...backends import projection as P

# A model's own CRS fits its stations to centimetres, so a worse fit is the wrong CRS.
GOOD_FIT_M = 1.0


@dataclass
class Georef:
    """Model local frame -> ``crs``: ``E = x + x0``, ``N = y + y0``.

    ``crs`` is anything ``pyproj.CRS.from_user_input`` accepts
    (``"EPSG:32608"`` or a PROJ string for the setup GUI's local frames).
    """
    crs: str = ""
    x0: float = 0.0
    y0: float = 0.0

    @property
    def valid(self) -> bool:
        return bool(self.crs)


@dataclass
class FitResult:
    georef: Georef
    n_matched: int
    n_receivers: int
    rms_m: float
    max_m: float
    unmatched: list[str] = field(default_factory=list)
    residuals_m: dict[str, float] = field(default_factory=dict)


def crs_label(crs: str) -> str:
    """Short human-readable name of *crs*."""
    if not crs:
        return "(none)"
    try:
        from pyproj import CRS

        c = CRS.from_user_input(crs)
        epsg = c.to_epsg()
        return f"{c.name} (EPSG:{epsg})" if epsg else c.name
    except Exception:  # noqa: BLE001
        return crs


def _norm_name(name: str) -> str:
    name = name.strip()
    if name.lower().endswith(".edi"):
        name = name[:-4]
    return name.lower()


# --------------------------------------------------------------------------- #
def manual(crs: str, x0: float, y0: float) -> Georef:
    return Georef(crs=str(crs), x0=float(x0), y0=float(y0))


def from_mtinv_project(path: str) -> tuple[Georef, str]:
    """``(georef, receivers_path_guess)`` from a setup-GUI project file."""
    import os

    from ...backends.mesh import RECEIVERS_CSV, mesh_output_dir
    from ...project import Project

    proj = Project.load(path)
    georef = from_frame(P.frame_from_projection(proj.projection))
    rec = os.path.join(mesh_output_dir(proj), RECEIVERS_CSV)
    return georef, rec if os.path.isfile(rec) else ""


def from_frame(frame) -> Georef:
    """The :class:`Georef` of a setup frame (``projection.LocalFrame``), e.g. one
    read back with ``projection.load_frame``."""
    import warnings

    crs = P._target_crs(frame)
    epsg = crs.to_epsg()
    with warnings.catch_warnings():          # "may lose information": not for these simple CRSs
        warnings.simplefilter("ignore", UserWarning)
        crs_text = f"EPSG:{epsg}" if epsg else crs.to_proj4()
    return Georef(crs=crs_text, x0=frame.x0, y0=frame.y0)


def fit_from_stations(receivers, stations, crs: str) -> FitResult:
    """Least-squares origin for *crs* from receivers matched to stations.

    *receivers* -- :class:`~.model_source.Receivers` (local E, N).
    *stations*  -- objects with ``station_id``, ``lon``, ``lat`` (EDI headers)
    or a ``{name: (lon, lat)}`` dict.  Names match case-insensitively,
    ignoring a ``.edi`` suffix.

    With a translation-only model, the origin is the mean of
    ``CRS coords - local coords`` and the residual spread is the misfit:
    ~0 when *crs* is the CRS the model was built in.
    """
    from pyproj import Transformer

    if isinstance(stations, dict):
        lonlat = {_norm_name(k): v for k, v in stations.items()}
    else:
        lonlat = {_norm_name(s.station_id): (s.lon, s.lat) for s in stations}
    idx, ll, unmatched = [], [], []
    for i, name in enumerate(receivers.names):
        hit = lonlat.get(_norm_name(name))
        if hit is None:
            unmatched.append(name)
        else:
            idx.append(i)
            ll.append(hit)
    if len(idx) < 1:
        raise ValueError("no receiver names matched the station list")
    ll = np.asarray(ll, dtype="float64")
    tr = Transformer.from_crs(4326, crs, always_xy=True)
    E, N = tr.transform(ll[:, 0], ll[:, 1])
    loc = receivers.xyz[idx, :2]
    dx = np.asarray(E) - loc[:, 0]
    dy = np.asarray(N) - loc[:, 1]
    x0, y0 = float(dx.mean()), float(dy.mean())
    res = np.hypot(dx - x0, dy - y0)
    return FitResult(
        georef=Georef(crs=crs, x0=x0, y0=y0),
        n_matched=len(idx), n_receivers=len(receivers),
        rms_m=float(np.sqrt(np.mean(res ** 2))), max_m=float(res.max()),
        unmatched=unmatched,
        residuals_m={receivers.names[i]: float(r) for i, r in zip(idx, res)})


def _station_lonlats(stations) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(stations, dict):
        lons, lats = np.asarray(list(stations.values()), dtype="float64").T
        return lons, lats
    return (np.array([s.lon for s in stations], dtype="float64"),
            np.array([s.lat for s in stations], dtype="float64"))


def candidate_crs(stations) -> list[str]:
    """CRSs worth trying for a survey: the UTM zone of its centre and both
    neighbours, plus the setup GUI's survey-centred local projections."""
    lons, lats = _station_lonlats(stations)
    lon_c = 0.5 * (lons.min() + lons.max())
    lat_c = 0.5 * (lats.min() + lats.max())
    zone = int((lon_c + 180.0) // 6.0) + 1
    base = 32600 if lat_c >= 0 else 32700
    out = [f"EPSG:{base + z}" for z in (zone, zone - 1, zone + 1) if 1 <= z <= 60]
    lat1, lat2 = P.standard_parallels(lats.min(), lats.max())
    for kind in P.LOCAL_PROJECTIONS:
        out.append(P._local_proj4(kind, lon_c, lat_c, lat1, lat2)
                   .replace(" +type=crs", ""))
    return out


def registered_crs(stations) -> list[str]:
    """Projected EPSG and ESRI CRSs whose area of use covers the whole survey:
    national and regional grids, such as a country's atlas Lambert."""
    from pyproj.aoi import AreaOfInterest
    from pyproj.database import query_crs_info
    from pyproj.enums import PJType

    lons, lats = _station_lonlats(stations)
    area = AreaOfInterest(float(lons.min()), float(lats.min()),
                          float(lons.max()), float(lats.max()))
    return [f"{info.auth_name}:{info.code}"
            for auth in ("EPSG", "ESRI")
            for info in query_crs_info(auth_name=auth, pj_types=PJType.PROJECTED_CRS,
                                       area_of_interest=area, contains=True)]


def _best_fit(receivers, stations, crs_list, best: FitResult | None = None) -> FitResult | None:
    for crs in crs_list:
        try:
            with np.errstate(invalid="ignore"):      # stations outside a CRS's domain: inf
                res = fit_from_stations(receivers, stations, crs)
        except Exception:  # noqa: BLE001 -- bad candidate, skip
            continue
        if np.isfinite(res.rms_m) and (best is None or res.rms_m < best.rms_m):
            best = res
    return best


def auto_fit(receivers, stations, extra: list[str] | None = None) -> FitResult:
    """Fit every :func:`candidate_crs` (plus *extra*) and keep the lowest RMS.  If
    none is right (see :data:`GOOD_FIT_M`), try :func:`registered_crs` as well."""
    best = _best_fit(receivers, stations, list(extra or []) + candidate_crs(stations))
    if best is None or best.rms_m > GOOD_FIT_M:
        best = _best_fit(receivers, stations, registered_crs(stations), best)
    if best is None:
        raise ValueError("could not fit any candidate CRS")
    return best


# --------------------------------------------------------------------------- #
class MapTransform:
    """Model local frame <-> display CRS (and lon/lat).

    When the display CRS is the model CRS this is a pure translation;
    otherwise points go local -> model CRS -> display CRS through pyproj.
    """

    def __init__(self, georef: Georef, display_crs: str = ""):
        from pyproj import CRS, Transformer

        if not georef.valid:
            raise ValueError("model is not georeferenced")
        self.georef = georef
        self.model_crs = CRS.from_user_input(georef.crs)
        self.display_crs = CRS.from_user_input(display_crs) if display_crs else self.model_crs
        self.same = self.model_crs == self.display_crs
        self._fwd = None if self.same else Transformer.from_crs(
            self.model_crs, self.display_crs, always_xy=True)
        self._inv = None if self.same else Transformer.from_crs(
            self.display_crs, self.model_crs, always_xy=True)
        self._to_ll = Transformer.from_crs(self.display_crs, 4326, always_xy=True)
        self._from_ll = Transformer.from_crs(4326, self.display_crs, always_xy=True)

    @property
    def key(self) -> tuple:
        """Hashable identity -- for caches."""
        g = self.georef
        return (g.crs, round(g.x0, 3), round(g.y0, 3), self.display_crs.to_wkt())

    @property
    def display_label(self) -> str:
        epsg = self.display_crs.to_epsg()
        return f"{self.display_crs.name}" + (f" (EPSG:{epsg})" if epsg else "")

    def forward(self, x, y):
        """Local model (x, y) -> display (E, N)."""
        E = np.asarray(x, dtype="float64") + self.georef.x0
        N = np.asarray(y, dtype="float64") + self.georef.y0
        if self.same:
            return E, N
        e, n = self._fwd.transform(E, N)
        return np.asarray(e), np.asarray(n)

    def inverse(self, e, n):
        """Display (E, N) -> local model (x, y)."""
        e = np.asarray(e, dtype="float64")
        n = np.asarray(n, dtype="float64")
        if not self.same:
            e, n = self._inv.transform(e, n)
            e, n = np.asarray(e), np.asarray(n)
        return e - self.georef.x0, n - self.georef.y0

    def to_lonlat(self, e, n):
        lon, lat = self._to_ll.transform(np.asarray(e, dtype="float64"),
                                         np.asarray(n, dtype="float64"))
        return np.asarray(lon), np.asarray(lat)

    def from_lonlat(self, lon, lat):
        e, n = self._from_ll.transform(np.asarray(lon, dtype="float64"),
                                       np.asarray(lat, dtype="float64"))
        return np.asarray(e), np.asarray(n)
