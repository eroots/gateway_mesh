"""Task 0 -- read a folder of EDI files.

:func:`read_edi_headers` is mtpy-first: it reads each file with
``mtpy.core.mt.MT`` when mtpy is importable, and falls back to a
dependency-free text parse otherwise -- or per-file, when mtpy chokes on a
quirky EDI (e.g. a tipper block with no matching variance block, which
mtpy 1.1.5 raises on).  So the station map still works before the scientific
env is set up, and mtpy is the source of truth once it is.  The parity of the
two paths is checked in ``tests/test_edi_readers.py``.

:func:`load_collection` returns a real ``EdiCollection`` for the downstream
steps that need the full impedance tensors (period selection, data writing).
It has no text fallback: a file mtpy rejects is read from a copy with just
those quirks fixed (:func:`mtpy_readable`), so every value still comes from mtpy.
"""
from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass

from .. import config


@dataclass
class Station:
    station_id: str
    path: str
    lon: float
    lat: float
    elev: float
    n_freq: int
    freq_min: float   # Hz
    freq_max: float   # Hz

    @property
    def period_min_s(self) -> float:
        return 1.0 / self.freq_max if self.freq_max else float("nan")

    @property
    def period_max_s(self) -> float:
        return 1.0 / self.freq_min if self.freq_min else float("nan")


_DMS_RE = re.compile(r"^\s*(-?)\s*(\d+):(\d+):(\d+(?:\.\d+)?)\s*$")


def _parse_coord(text: str) -> float:
    """Parse an EDI coordinate: either decimal degrees or ``DD:MM:SS.ss``."""
    text = text.strip().strip('"')
    m = _DMS_RE.match(text)
    if m:
        sign = -1.0 if m.group(1) == "-" else 1.0
        deg, minute, sec = float(m.group(2)), float(m.group(3)), float(m.group(4))
        return sign * (deg + minute / 60.0 + sec / 3600.0)
    return float(text)


def _find(pattern: str, blob: str) -> str | None:
    m = re.search(pattern, blob, re.IGNORECASE)
    return m.group(1) if m else None


def resolve_edi_files(path: str, pattern: str = "*.edi") -> list[str]:
    """Turn *path* into a concrete list of ``.edi`` file paths.

    Accepts:
      * a directory  -> every ``*.edi`` inside it;
      * a ``.lst`` file -> one site name per line (``.edi`` optional, ``#``
        comments allowed), resolved relative to the list file's directory;
      * a single ``.edi`` file.
    """
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, pattern)))

    if os.path.isfile(path) and path.lower().endswith(".lst"):
        base = os.path.dirname(os.path.abspath(path))
        out: list[str] = []
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                name = line.strip()
                if not name or name.startswith("#"):
                    continue
                if not name.lower().endswith(".edi"):
                    name += ".edi"
                cand = name if os.path.isabs(name) else os.path.join(base, name)
                if os.path.isfile(cand):
                    out.append(cand)
        return out

    if os.path.isfile(path) and path.lower().endswith(".edi"):
        return [path]
    return []


_headers_cache: dict[tuple[str, str], list[Station]] = {}


def read_edi_headers(edi_path: str, pattern: str = "*.edi", *, force: bool = False) -> list[Station]:
    """Return a :class:`Station` for every EDI resolved from *edi_path*.

    *edi_path* may be a directory, a ``.lst`` site list, or a single ``.edi``
    (see :func:`resolve_edi_files`).  mtpy-first with a text-parse fallback
    (see the module docstring), so the map view still works before the
    scientific env is set up.

    Cached in-process by ``(edi_path, pattern)``: this is called from several
    independent panels (Survey, Topography, Project DEM) on every
    ``notify_changed()`` fired anywhere in the app -- e.g. clicking a spin box
    on the Mesh tab -- and re-opening + re-parsing potentially hundreds of
    files each time was the dominant source of the resulting GUI lag,
    especially over a network/cloud-synced project folder (order of half a
    second per call, observed). EDI files aren't expected to change on disk
    mid-session; pass ``force=True`` for an explicit user-initiated reload
    (e.g. the Survey tab's Browse button) that should see any real changes.
    """
    key = (edi_path, pattern)
    if not force and key in _headers_cache:
        return _headers_cache[key]
    stations = _read_edi_headers_uncached(edi_path, pattern)
    _headers_cache[key] = stations
    return stations


def clear_edi_headers_cache(edi_path: str | None = None) -> None:
    """Drop cached :func:`read_edi_headers` results (all of them, or just
    those for *edi_path*) -- call after files there may have changed."""
    if edi_path is None:
        _headers_cache.clear()
    else:
        for key in [k for k in _headers_cache if k[0] == edi_path]:
            del _headers_cache[key]


def _station_id_from_file(path: str, blob: str | None = None) -> str:
    """``DATAID`` if the header has one, else the filename stem.  Used by both
    reader paths (and by :mod:`edi_z`) so a station's identity -- and therefore
    the Survey tab's exclude set -- doesn't shift when a file falls back from
    mtpy to the text parser."""
    if blob is None:
        with open(path, "r", errors="replace") as fh:
            blob = fh.read(8192)          # DATAID lives in the >HEAD block, at the top
    dataid = _find(r'DATAID\s*=\s*"?([^"\r\n]+)"?', blob)
    return (dataid or os.path.splitext(os.path.basename(path))[0]).strip()


def _station_via_text(path: str) -> Station | None:
    with open(path, "r", errors="replace") as fh:
        blob = fh.read()

    lat = _find(r"\bLAT\s*=\s*([^\s]+)", blob)
    lon = _find(r"\bLONG?\s*=\s*([^\s]+)", blob)
    if lat is None or lon is None:                    # fall back to REFLAT/REFLONG
        lat = lat or _find(r"REFLAT\s*=\s*([^\s]+)", blob)
        lon = lon or _find(r"REFLONG?\s*=\s*([^\s]+)", blob)
    if lat is None or lon is None:
        return None
    elev = _find(r"\bELEV\s*=\s*([-\d.eE+]+)", blob) or "0"

    freqs = _read_freq_block(blob)
    return Station(
        station_id=_station_id_from_file(path, blob),
        path=path,
        lon=_parse_coord(lon),
        lat=_parse_coord(lat),
        elev=float(elev),
        n_freq=len(freqs),
        freq_min=min(freqs) if freqs else float("nan"),
        freq_max=max(freqs) if freqs else float("nan"),
    )


def _station_via_mtpy(mt_mod, path: str) -> Station | None:
    import numpy as np

    obj = mt_mod.MT(path)
    z = getattr(obj, "Z", None)
    raw = getattr(z, "freq", None) if z is not None else None
    freqs = np.asarray(raw, dtype="float64").ravel() if raw is not None \
        else np.array([], dtype="float64")
    freqs = freqs[np.isfinite(freqs) & (freqs > 0)]
    lat, lon = obj.lat, obj.lon
    if lat is None or lon is None:
        return None
    return Station(
        station_id=_station_id_from_file(path),
        path=path,
        lon=float(lon),
        lat=float(lat),
        elev=float(obj.elev or 0.0),
        n_freq=int(freqs.size),
        freq_min=float(freqs.min()) if freqs.size else float("nan"),
        freq_max=float(freqs.max()) if freqs.size else float("nan"),
    )


def _mtpy_module_or_none(mode: str):
    """The ``mtpy.core.mt`` module when it should be used, else None.
    Raises (never returns None) when *mode* is the pinned ``"mtpy"``."""
    if mode == "text":
        return None
    if mode == "auto" and not config.mtpy_available():
        return None
    config.apply_sys_paths()
    config.ensure_mtpy_numpy_compat()
    import mtpy.core.mt as mt_mod       # let ImportError propagate under mode="mtpy"
    return mt_mod


# mtpy-> text fallbacks recorded since the last drain, so a panel can flag
# "mtpy couldn't read N files, used the built-in parser" after a load.  A
# plain list -- reads are serialised per panel through run_async.
_reader_fallbacks: list[tuple[str, str]] = []


def _note_fallback(path: str, exc: BaseException) -> None:
    _reader_fallbacks.append((path, f"{type(exc).__name__}: {exc}"))


def drain_reader_fallbacks() -> list[tuple[str, str]]:
    """Return + clear the ``(path, reason)`` list of files that fell back from
    mtpy to the text parser since the last call.  Panels drain this right
    after a load and surface a one-line summary."""
    global _reader_fallbacks
    out, _reader_fallbacks = _reader_fallbacks, []
    return out


def summarize_fallbacks(events: list[tuple[str, str]], total: int | None = None) -> str:
    """One-line, de-duplicated summary of :func:`drain_reader_fallbacks` output."""
    if not events:
        return ""
    from collections import Counter
    reasons = Counter(reason for _p, reason in events)
    top_reason, _ = reasons.most_common(1)[0]
    n = len(events)
    head = f"mtpy could not read {n}" + (f"/{total}" if total else "") + " EDI file(s)"
    return f"{head}; used the built-in text parser for those (e.g. {top_reason})."


def _read_edi_headers_uncached(edi_path: str, pattern: str = "*.edi") -> list[Station]:
    files = resolve_edi_files(edi_path, pattern)
    mode = config.edi_reader_mode()
    mt_mod = _mtpy_module_or_none(mode)

    stations: list[Station] = []
    for path in files:
        st: Station | None = None
        if mt_mod is not None:
            try:
                st = _station_via_mtpy(mt_mod, path)
            except Exception as exc:
                if mode == "mtpy":
                    raise
                _note_fallback(path, exc)   # quirky EDI -> text parse just this one
                st = None
        if st is None and mode != "mtpy":
            st = _station_via_text(path)
        if st is not None:
            stations.append(st)
    return stations


_EDI_EMPTY = 1.0e32   # EDI fill value for "no data" (also written 1.0E+32)


def _read_freq_block(blob: str) -> list[float]:
    # Stop at the next section header.  EDI headers may be indented
    # (`  >ZROT ...`), so terminate on "newline, optional whitespace, `>`" --
    # `\n>` alone let this run past FREQ into ZROT / the Z blocks, pulling in
    # rotation angles, `// <n>` counts and 1e32 fill values as bogus
    # "frequencies" (seen as period ranges like 1e-32 .. 1e6 s).
    m = re.search(r">FREQ\b[^\n]*\n(.*?)(?:\n\s*>|\Z)", blob, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    vals = []
    for tok in m.group(1).split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return [v for v in vals if v > 0 and math.isfinite(v) and abs(v) < _EDI_EMPTY]


def read_edi_periods(edi_path: str, pattern: str = "*.edi") -> dict[str, dict]:
    """Return ``{station_id: {"path": ..., "periods": [sorted period list]}}``
    for every EDI resolved from *edi_path* -- the full per-station period
    set, not just the count/min/max :func:`read_edi_headers` keeps, for e.g.
    a period-coverage histogram (the path travels with it so callers -- e.g.
    the before/after QC plot -- don't need a second pass to resolve it).

    mtpy-first with a text fallback, same as :func:`read_edi_headers`; not
    cached (called far less often -- once per explicit "reload stats" action,
    not from every panel's ``refresh()``).
    """
    mode = config.edi_reader_mode()
    mt_mod = _mtpy_module_or_none(mode)

    out: dict[str, dict] = {}
    for path in resolve_edi_files(edi_path, pattern):
        periods: list[float] | None = None
        if mt_mod is not None:
            try:
                import numpy as np

                fr = np.asarray(mt_mod.MT(path).Z.freq, dtype="float64").ravel()
                fr = fr[np.isfinite(fr) & (fr > 0)]
                periods = sorted((1.0 / fr).tolist()) if fr.size else []
            except Exception as exc:
                if mode == "mtpy":
                    raise
                _note_fallback(path, exc)
                periods = None
        if periods is None and mode != "mtpy":
            with open(path, "r", errors="replace") as fh:
                blob = fh.read()
            freqs = _read_freq_block(blob)
            periods = sorted(1.0 / f for f in freqs) if freqs else []
        if not periods:
            continue
        out[_station_id_from_file(path)] = {"path": path, "periods": periods}
    return out


# --------------------------------------------------------------------------- #
def bounding_box(stations: list[Station], exclude: set[str] | None = None) -> dict:
    exclude = exclude or set()
    lons = [s.lon for s in stations if s.station_id not in exclude]
    lats = [s.lat for s in stations if s.station_id not in exclude]
    if not lons:
        return {"MinLon": 0, "MaxLon": 0, "MinLat": 0, "MaxLat": 0}
    return {"MinLon": min(lons), "MaxLon": max(lons),
            "MinLat": min(lats), "MaxLat": max(lats)}


def suggest_utm(lon: float, lat: float) -> tuple[int, bool, int]:
    """Return ``(zone, northern_hemisphere, epsg)`` for a point."""
    zone = int((lon + 180.0) / 6.0) + 1
    zone = min(max(zone, 1), 60)
    north = lat >= 0.0
    epsg = (32600 if north else 32700) + zone
    return zone, north, epsg


def period_coverage(stations: list[Station], exclude: set[str] | None = None) -> dict:
    exclude = exclude or set()
    kept = [s for s in stations if s.station_id not in exclude and s.n_freq]
    if not kept:
        return {"period_min_s": 0.0, "period_max_s": 0.0, "n_periods": 0}
    pmin = min(s.period_min_s for s in kept)
    pmax = max(s.period_max_s for s in kept)
    nper = max(s.n_freq for s in kept)
    return {"period_min_s": pmin, "period_max_s": pmax, "n_periods": nper}


# --------------------------------------------------------------------------- #
_NO_TIPPER_VAR = "missing tipper variance"
_SIXTY = "a coordinate with 60 seconds or minutes"
_DMS_VALUE_RE = re.compile(r"^([ \t]*(?:REF)?(?:LAT|LONG|LON)[ \t]*=[ \t]*)(\"?)(-?[ \t]*\d+:\d+:\d+(?:\.\d+)?)\2",
                           re.IGNORECASE | re.MULTILINE)


def mtpy_readable(blob: str) -> tuple[str, list[str]]:
    """An EDI's text with the quirks mtpy 1.1.5 rejects fixed, and what was fixed.  Nothing
    else changes, so mtpy reads every value as written.

    * A tipper with no ``TXVAR.EXP`` / ``TYVAR.EXP`` block gets one of zeros: mtpy needs it,
      and zero is what it reads as "no error given", which GoFEM's writer then raises to
      the error floor.
    * A ``D:M:S`` coordinate with 60 seconds (or minutes) is written in decimal degrees.
    """
    fixes = []
    n_freq = len(_read_freq_block(blob))
    for comp in ("TX", "TY"):
        has_tipper = re.search(rf"^[ \t]*>[ \t]*{comp}R\.EXP\b", blob, re.IGNORECASE | re.MULTILINE)
        has_var = re.search(rf"^[ \t]*>[ \t]*{comp}VAR\.EXP\b", blob, re.IGNORECASE | re.MULTILINE)
        if has_tipper and not has_var and n_freq:
            zeros = "\n".join(" ".join(["0.000000E+00"] * len(row))
                              for row in (range(i, min(i + 5, n_freq)) for i in range(0, n_freq, 5)))
            block = f">{comp}VAR.EXP // {n_freq}\n{zeros}\n\n"
            end = list(re.finditer(r"^[ \t]*>[ \t]*END\b", blob, re.IGNORECASE | re.MULTILINE))
            at = end[-1].start() if end else len(blob)
            blob = blob[:at] + block + blob[at:]
            if _NO_TIPPER_VAR not in fixes:
                fixes.append(_NO_TIPPER_VAR)

    def decimal(m):
        value = m.group(3).replace(" ", "")
        _deg, minute, sec = (float(v) for v in value.lstrip("-").split(":"))
        if minute < 60 and sec < 60:
            return m.group(0)
        if _SIXTY not in fixes:
            fixes.append(_SIXTY)
        return f"{m.group(1)}{_parse_coord(value):.9f}"

    blob = _DMS_VALUE_RE.sub(decimal, blob)
    return blob, fixes


def _read_mt(path: str, tmp_dir: str) -> tuple[object, list[str]]:
    """``mtpy.core.mt.MT`` for *path*; if mtpy rejects the file, read a fixed copy instead
    (:func:`mtpy_readable`), still known by the original's path."""
    import mtpy.core.mt as mt  # noqa: WPS433 (optional dep)

    try:
        return mt.MT(path), []
    except Exception as first:  # noqa: BLE001 -- mtpy raises anything on a bad file
        with open(path, "r", errors="replace") as fh:
            fixed, fixes = mtpy_readable(fh.read())
        name = os.path.basename(path)
        if not fixes:
            raise ValueError(f"mtpy can't read {name}: {type(first).__name__}: {first}") from first
        copy = os.path.join(tmp_dir, name)
        with open(copy, "w") as fh:
            fh.write(fixed)
        try:
            obj = mt.MT(copy)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"mtpy can't read {name}, even with {', '.join(fixes)} fixed: "
                             f"{type(exc).__name__}: {exc}") from exc
        obj._fn = os.path.normpath(os.path.abspath(path))
        obj.save_dir = os.path.dirname(obj._fn)
        return obj, fixes


def load_collection(edi_path: str, exclude: set[str] | None = None, log=None):
    """Return a real ``mtpy`` ``EdiCollection`` (raises if mtpy is missing).

    Used by the period-selection and data-file steps that need the full
    transfer functions, not just the headers.  *edi_path* may be a directory or
    a ``.lst`` (see :func:`resolve_edi_files`).  Files mtpy rejects are read from a
    fixed copy (:func:`mtpy_readable`); *log* gets a line saying how many.
    """
    import tempfile
    from collections import Counter

    from .. import config
    config.ensure_mtpy_numpy_compat()
    from mtpy.core.edi_collection import EdiCollection  # noqa: WPS433 (optional dep)

    files = resolve_edi_files(edi_path)
    if exclude:
        files = [f for f in files
                 if os.path.splitext(os.path.basename(f))[0] not in exclude]
    objs, fixed = [], Counter()
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            obj, fixes = _read_mt(f, tmp)
            objs.append(obj)
            fixed.update(fixes)
            if fixes:
                fixed["files"] += 1
    if fixed and log is not None:
        n = fixed.pop("files")
        notes = {_NO_TIPPER_VAR: " (their tipper errors are the error floor)"}
        log(f"{n} of {len(files)} EDI files read from a fixed copy, as mtpy can't read them as written: "
            + "; ".join(f"{reason} in {count}{notes.get(reason, '')}" for reason, count in fixed.items()))
    collection = EdiCollection(mt_objs=objs)
    # Data files name stations from here, and they must match the receiver file, which
    # uses the same id as read_edi_headers.  mtpy alone only knows DATAID, and writes
    # "None" for an EDI without one.
    for obj in collection.mt_obj_list:
        obj.station = _station_id_from_file(obj.fn)
    return collection


# --------------------------------------------------------------------------- #
def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres (spherical earth, R = 6371 km)."""
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2)
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def station_spacing_geographic(stations: list[Station],
                               exclude: set[str] | None = None) -> dict:
    """Nearest-neighbour spacing stats (metres) from lon/lat -- no projection."""
    exclude = exclude or set()
    pts = [(s.lon, s.lat) for s in stations if s.station_id not in exclude]
    n = len(pts)
    if n < 2:
        return {"min_m": 0.0, "median_m": 0.0, "max_m": 0.0, "mean_m": 0.0, "n": n}
    nn = []
    for i in range(n):
        best = float("inf")
        for j in range(n):
            if i == j:
                continue
            d = haversine_m(pts[i][0], pts[i][1], pts[j][0], pts[j][1])
            if d < best:
                best = d
        nn.append(best)
    nn.sort()
    return {
        "min_m": nn[0], "max_m": nn[-1],
        "median_m": nn[n // 2] if n % 2 else 0.5 * (nn[n // 2 - 1] + nn[n // 2]),
        "mean_m": sum(nn) / n, "n": n,
    }
