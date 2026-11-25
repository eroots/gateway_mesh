"""Parity checks for the two EDI reader paths.

:mod:`mtinv_gui.backends.edi_io` and :mod:`mtinv_gui.backends.edi_z` are
mtpy-first with a built-in text-parser fallback (used when mtpy is missing, or
per-file when mtpy raises on a quirky EDI).  These tests pin each path with
``MTINV_EDI_READER`` and check that the fallback stays faithful to mtpy:

  * clean EDIs -> mtpy and text agree on coords, frequencies and the derived
    SSQ apparent resistivity / C-response, to floating-point tolerance;
  * quirky EDIs (indented headers, a tipper block with no variance block ...)
    -> ``auto`` still returns every station, because it drops to the text
    parser for the ones mtpy 1.1.5 chokes on.

Runs standalone (``python tests/test_edi_readers.py``) or under pytest if it
is installed.  Skips gracefully when mtpy or the sample data isn't present.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtinv_gui import config                       # noqa: E402
from mtinv_gui.backends import edi_io, edi_z       # noqa: E402

# --- sample data ---------------------------------------------------------- #
_GUI_DEV = os.path.normpath(os.path.join(ROOT, os.pardir))
CLEAN_DIR = os.path.join(_GUI_DEV, "edis")


def _make_quirky_edis(out_dir, n=6):
    """Copy *n* sample EDIs, breaking every other one the way some field EDIs
    are: indented section headers and a tipper block with no variance block."""
    names = sorted(f for f in os.listdir(CLEAN_DIR) if f.endswith(".edi"))[:n]
    for i, name in enumerate(names):
        with open(os.path.join(CLEAN_DIR, name)) as fh:
            lines = fh.read().splitlines()
        if i % 2 == 0:
            kept, skipping = [], False
            for line in lines:
                if line.startswith(">"):
                    skipping = line.startswith((">TXVAR.EXP", ">TYVAR.EXP"))
                    line = "  " + line
                if not skipping:
                    kept.append(line)
            lines = kept
        with open(os.path.join(out_dir, name), "w") as fh:
            fh.write("\n".join(lines) + "\n")


# --- helpers ------------------------------------------------------------- #
class _Skip(Exception):
    pass


def _read_headers(path, mode):
    os.environ["MTINV_EDI_READER"] = mode
    edi_io.clear_edi_headers_cache()
    try:
        return {s.station_id: s for s in edi_io.read_edi_headers(path, force=True)}
    finally:
        os.environ.pop("MTINV_EDI_READER", None)
        edi_io.clear_edi_headers_cache()


def _read_z(path, mode):
    os.environ["MTINV_EDI_READER"] = mode
    try:
        return {s.station_id: s for s in edi_z.read_dir(path)}
    finally:
        os.environ.pop("MTINV_EDI_READER", None)


def _close(a, b, *, rtol=1e-6, atol=0.0):
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)):
        return a is b or (isinstance(a, float) and isinstance(b, float)
                          and math.isnan(a) and math.isnan(b))
    return abs(a - b) <= atol + rtol * abs(b)


def _require(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --- checks ------------------------------------------------------------- #
def check_headers_parity_clean():
    if not config.mtpy_available():
        raise _Skip("mtpy not importable")
    if not os.path.isdir(CLEAN_DIR):
        raise _Skip(f"no sample EDIs at {CLEAN_DIR}")

    mt = _read_headers(CLEAN_DIR, "mtpy")
    tx = _read_headers(CLEAN_DIR, "text")
    _require(mt and tx, "one reader returned nothing")
    _require(set(mt) == set(tx),
             f"station sets differ: mtpy-only {set(mt) - set(tx)}, "
             f"text-only {set(tx) - set(mt)}")

    for sid in mt:
        a, b = mt[sid], tx[sid]
        _require(_close(a.lon, b.lon, atol=1e-6), f"{sid}: lon {a.lon} vs {b.lon}")
        _require(_close(a.lat, b.lat, atol=1e-6), f"{sid}: lat {a.lat} vs {b.lat}")
        _require(_close(a.elev, b.elev, atol=0.5), f"{sid}: elev {a.elev} vs {b.elev}")
        _require(a.n_freq == b.n_freq, f"{sid}: n_freq {a.n_freq} vs {b.n_freq}")
        _require(_close(a.freq_min, b.freq_min) and _close(a.freq_max, b.freq_max),
                 f"{sid}: freq range ({a.freq_min},{a.freq_max}) vs "
                 f"({b.freq_min},{b.freq_max})")
    return f"{len(mt)} stations identical (coords, freq range) mtpy vs text"


def check_impedance_parity_clean():
    import numpy as np

    if not config.mtpy_available():
        raise _Skip("mtpy not importable")
    if not os.path.isdir(CLEAN_DIR):
        raise _Skip(f"no sample EDIs at {CLEAN_DIR}")

    mt = _read_z(CLEAN_DIR, "mtpy")
    tx = _read_z(CLEAN_DIR, "text")
    _require(set(mt) == set(tx), "impedance station sets differ")

    worst = 0.0
    for sid in mt:
        a, b = mt[sid], tx[sid]
        oa, ob = np.argsort(a.freq), np.argsort(b.freq)
        _require(np.allclose(a.freq[oa], b.freq[ob], rtol=1e-6),
                 f"{sid}: frequency grids differ")
        for c in ("zxx", "zxy", "zyx", "zyy"):
            za, zb = a.z[c][oa], b.z[c][ob]
            m = np.abs(zb) > 0
            if m.any():
                rel = np.abs(za[m] - zb[m]) / np.abs(zb[m])
                worst = max(worst, float(np.nanmax(rel)))
        # the actual downstream quantities
        ra, rb = a.ssq_app_res()[oa], b.ssq_app_res()[ob]
        ca, cb = a.c_response_km()[oa].real, b.c_response_km()[ob].real
        good = np.isfinite(ra) & np.isfinite(rb) & (rb > 0)
        _require(np.allclose(ra[good], rb[good], rtol=1e-6),
                 f"{sid}: SSQ apparent resistivity differs")
        good = np.isfinite(ca) & np.isfinite(cb) & (np.abs(cb) > 0)
        _require(np.allclose(ca[good], cb[good], rtol=1e-6),
                 f"{sid}: C-response depth differs")

    curve_mt = edi_z.ssq_resistivity_curve(list(mt.values()))
    curve_tx = edi_z.ssq_resistivity_curve(list(tx.values()))
    _require(np.allclose(curve_mt.period_s, curve_tx.period_s, rtol=1e-6),
             "SSQ curve period grids differ")
    _require(np.allclose(curve_mt.rho_logmean, curve_tx.rho_logmean, rtol=1e-6),
             "SSQ curve rho_logmean differs")
    return (f"{len(mt)} stations: Z max rel diff {worst:.2e}, "
            f"SSQ curve identical")


def check_quirky_fallback(path=None):
    """*path*: a folder or ``.lst`` of quirky EDIs (default: made from the samples)."""
    if path is None:
        if not os.path.isdir(CLEAN_DIR):
            raise _Skip(f"no sample EDIs at {CLEAN_DIR}")
        with tempfile.TemporaryDirectory() as d:
            _make_quirky_edis(d)
            return check_quirky_fallback(d)

    import numpy as np

    if not config.mtpy_available():
        raise _Skip("mtpy not importable")

    auto = _read_z(path, "auto")
    text = _read_z(path, "text")
    try:
        mtpy = _read_z(path, "mtpy")             # pinned mtpy: expected to raise here
        mtpy_note = f"{len(mtpy)} stations (no raise)"
    except Exception as exc:
        mtpy, mtpy_note = {}, f"raised {type(exc).__name__}: {exc}"

    _require(len(text) > 0, "text parser read nothing from the quirky set")
    _require(set(text).issubset(set(auto)),
             f"auto lost stations the text parser handles: {set(text) - set(auto)}")
    _require(len(auto) >= len(text),
             "auto read fewer stations than the text parser -- fallback not engaging")
    _require(len(text) > len(mtpy),
             "quirky set didn't actually break pinned-mtpy -- pick a harder sample")

    # where auto used mtpy it must equal mtpy; where it fell back it must equal text
    for sid, a in auto.items():
        ref = text.get(sid)
        _require(ref is not None, f"{sid}: in auto but not in text (unexpected)")
        oa, ob = np.argsort(a.freq), np.argsort(ref.freq)
        if a.freq.shape == ref.freq.shape and np.allclose(a.freq[oa], ref.freq[ob], rtol=1e-6):
            for c in ("zxx", "zxy", "zyx", "zyy"):
                za, zb = a.z[c][oa], ref.z[c][ob]
                m = np.abs(zb) > 0
                _require(np.allclose(za[m], zb[m], rtol=1e-4) if m.any() else True,
                         f"{sid}: auto vs text impedance mismatch on {c}")
    return (f"text {len(text)} stations, auto {len(auto)}, pinned-mtpy {mtpy_note} "
            f"-- auto falls back per-file and keeps every station")


def check_data_file_names_match_the_receivers():
    """Sample EDIs have no DATAID: the data must still name stations as the receivers do."""
    if not config.mtpy_available():
        raise _Skip("mtpy not importable")
    if not os.path.isdir(CLEAN_DIR):
        raise _Skip(f"no sample EDIs at {CLEAN_DIR}")
    from mtinv_gui.backends import gofem_files

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.gdat")
        gofem_files.write_data_file(edi_io.load_collection(CLEAN_DIR), path, "Z", 0.05, (1e-4, 1e4))
        with open(path) as fh:
            names = {line.split()[3] for line in fh if not line.startswith("#")}
    want = {s.station_id for s in edi_io.read_edi_headers(CLEAN_DIR)}
    _require(names == want, f"data names differ from receiver names, e.g. {sorted(names ^ want)[:4]}")
    return f"{len(names)} stations named the same in data and receivers"


def check_quirky_data_writing():
    """Data from quirky copies of the samples must equal the clean originals', except that
    a tipper with no variance gets the error floor."""
    if not config.mtpy_available():
        raise _Skip("mtpy not importable")
    if not os.path.isdir(CLEAN_DIR):
        raise _Skip(f"no sample EDIs at {CLEAN_DIR}")
    import re
    import shutil

    from mtinv_gui.backends import gofem_files

    def rows(path):
        with open(path) as fh:
            return {tuple(line.split()[:4]): line.split()[4:] for line in fh if not line.startswith("#")}

    with tempfile.TemporaryDirectory() as d:
        clean, quirky, out = (os.path.join(d, k) for k in ("clean", "quirky", "out"))
        for p in (clean, quirky, out):
            os.makedirs(p)
        _make_quirky_edis(quirky)                         # every other file: no tipper variance
        names = sorted(os.listdir(quirky))
        for name in names:
            shutil.copy(os.path.join(CLEAN_DIR, name), clean)
        # the same latitude, written with more than 60 seconds, which mtpy refuses
        odd = os.path.join(quirky, names[1])
        blob = open(odd).read()
        m = re.search(r"^LAT=(-?)(\d+):(\d+):([\d.]+)", blob, re.MULTILINE)
        _require(m is not None, f"no D:M:S latitude in {names[1]}")
        sec = float(m.group(4)) + 60.0
        blob = blob.replace(m.group(0), f"LAT={m.group(1)}{m.group(2)}:{int(m.group(3)) - 1}:{sec:.2f}")
        with open(odd, "w") as fh:
            fh.write(blob)
        unchanged, fixes = edi_io.mtpy_readable(open(os.path.join(clean, names[0])).read())
        _require(not fixes, f"a clean EDI was changed: {fixes}")

        logged = []
        got = edi_io.load_collection(quirky, log=logged.append)
        want = edi_io.load_collection(clean)
        _require(len(logged) == 1 and "missing tipper variance in 3" in logged[0] and "60 seconds or minutes in 1" in logged[0],
                 f"log: {logged}")
        for a, b in zip(got.mt_obj_list, want.mt_obj_list):
            _require(a.station == b.station and os.path.dirname(a.fn) == quirky, f"{a.station}: {a.fn}")
            _require(abs(a.lat - b.lat) < 1e-8 and abs(a.lon - b.lon) < 1e-8, f"{a.station} moved")

        for kind in ("Z", "Tipper"):
            mine, ref = (os.path.join(out, f"{tag}_{kind}.gdat") for tag in ("quirky", "clean"))
            gofem_files.write_data_file(got, mine, kind, 0.05, (1e-4, 1e5))
            gofem_files.write_data_file(want, ref, kind, 0.05, (1e-4, 1e5))
            a, b = rows(mine), rows(ref)
            _require(a.keys() == b.keys(), f"{kind}: different rows")
            _require(all(a[k][0] == b[k][0] for k in a), f"{kind}: different values")
            if kind == "Z":
                _require(a == b, "Z errors differ")
            no_var = {os.path.splitext(n)[0] for i, n in enumerate(names) if i % 2 == 0}
            if kind == "Tipper":
                _require({k[3] for k in a} > no_var, "stations missing from the tipper data")
                _require(all(float(a[k][1]) == 0.05 for k in a if k[3] in no_var), "tipper errors aren't the floor")
                _require(all(a[k] == b[k] for k in a if k[3] not in no_var), "tipper with variance changed")

        broken = os.path.join(d, "broken")
        os.makedirs(broken)
        with open(os.path.join(broken, "bad.edi"), "w") as fh:
            fh.write(">HEAD\n>END\n")
        try:
            edi_io.load_collection(broken)
        except ValueError as exc:
            _require("bad.edi" in str(exc), f"error doesn't name the file: {exc}")
        else:
            raise AssertionError("an unreadable EDI was accepted")
    return f"{len(names)} stations, {len(a):,} tipper rows: {logged[0]}"


# --- pytest shims ------------------------------------------------------- #
def _wrap(fn):
    def _t():
        try:
            print(fn())
        except _Skip as exc:  # pragma: no cover
            try:
                import pytest
                pytest.skip(str(exc))
            except ImportError:
                print(f"SKIP: {exc}")
    _t.__name__ = fn.__name__
    return _t


test_headers_parity_clean = _wrap(check_headers_parity_clean)
test_impedance_parity_clean = _wrap(check_impedance_parity_clean)
test_quirky_fallback = _wrap(check_quirky_fallback)
test_data_file_names_match_the_receivers = _wrap(check_data_file_names_match_the_receivers)
test_quirky_data_writing = _wrap(check_quirky_data_writing)


if __name__ == "__main__":
    rc = 0
    for fn in (check_headers_parity_clean, check_impedance_parity_clean,
               check_quirky_fallback, check_data_file_names_match_the_receivers,
               check_quirky_data_writing):
        try:
            print(f"[ok]   {fn.__name__}: {fn()}")
        except _Skip as exc:
            print(f"[skip] {fn.__name__}: {exc}")
        except AssertionError as exc:
            print(f"[FAIL] {fn.__name__}: {exc}")
            rc = 1
    sys.exit(rc)
