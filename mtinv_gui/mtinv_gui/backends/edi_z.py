"""Read impedance tensors from EDI files -> SSQ resistivity, phase, C-response.

mtpy-first: :func:`read_station_z` takes ``Z`` from ``mtpy.core.mt.MT`` when
mtpy is importable, and falls back to parsing the ``>Z***R`` / ``>Z***I``
blocks directly (the same way :mod:`mtinv_gui.backends.edi_io` reads
``>FREQ``) otherwise, or per-file when an mtpy read raises.  Both paths return
``Z`` in the EDI's own practical/field units -- the parity of the two is
checked in ``tests/test_edi_readers.py``.

The rotational invariant used is the SSQ (sum-of-squares) impedance, computed
with *complex* squares so it carries a phase:

    Z_ssq = sqrt( (Zxx^2 + Zxy^2 + Zyx^2 + Zyy^2) / 2 )        # complex, SI ohm
    rho_ssq(T) = |Z_ssq|^2 / (omega * mu0)
    phase_ssq(T) = arg(Z_ssq)
    C(T)         = Z_ssq / (i * omega * mu0)                    # response length

EDI impedances are in practical/field units [mV/km]/[nT]; the SI conversion
factor is 4*pi/1e4.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

_EMPTY = 1.0e32
_COMPONENTS = ("zxx", "zxy", "zyx", "zyy")
_MU0 = 4.0 * math.pi * 1e-7
_FIELD_TO_OHM = 4.0 * math.pi / 1.0e4      # [mV/km]/[nT] -> ohm (SI)


@dataclass
class StationZ:
    station_id: str
    path: str
    freq: np.ndarray          # Hz, descending as written
    z: dict[str, np.ndarray]  # complex arrays keyed "zxx".. "zyy" (field units)

    @property
    def periods(self) -> np.ndarray:
        return 1.0 / self.freq

    @property
    def omega(self) -> np.ndarray:
        return 2.0 * math.pi * self.freq

    def z_ssq(self) -> np.ndarray:
        """Complex SSQ impedance in SI ohm (NaN where a component is missing)."""
        acc = np.zeros(self.freq.shape, dtype="complex128")
        for c in _COMPONENTS:
            acc = acc + self.z[c] ** 2
        return _FIELD_TO_OHM * np.sqrt(acc / 2.0)

    def ssq_app_res(self) -> np.ndarray:
        """rho_ssq(T) in ohm.m."""
        z = self.z_ssq()
        return np.abs(z) ** 2 / (self.omega * _MU0)

    def ssq_phase_deg(self) -> np.ndarray:
        z = self.z_ssq()
        return np.degrees(np.arctan2(z.imag, z.real))

    def c_response_km(self) -> np.ndarray:
        """Complex C-response in km (Re = real depth scale, Im = induction)."""
        z = self.z_ssq()
        return z / (1j * self.omega * _MU0) / 1.0e3


def _block(name: str, blob: str) -> np.ndarray | None:
    # Terminate at the next section header.  EDI allows a header to be indented
    # (`  >ZXXI ...`), so the terminator must be "newline, optional whitespace,
    # `>`" -- matching only `\n>` let one block swallow every following block
    # (and every `// <n>` count token), inflating the length and producing
    # arrays that no longer broadcast against FREQ.
    m = re.search(rf">{name}\b[^\n]*\n(.*?)(?:\n\s*>|\Z)", blob, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    vals = []
    for tok in m.group(1).split():
        try:
            v = float(tok)
        except ValueError:
            continue
        vals.append(np.nan if abs(v) >= _EMPTY else v)
    return np.asarray(vals, dtype="float64")


def read_station_z(path: str) -> StationZ | None:
    from .. import config

    mode = config.edi_reader_mode()
    if mode == "mtpy" or (mode == "auto" and config.mtpy_available()):
        try:
            st = _read_station_z_mtpy(path)
            if st is not None or mode == "mtpy":
                return st
            # mtpy read it but found no usable Z -> let the text parser try
        except Exception as exc:
            if mode == "mtpy":
                raise                    # pinned mtpy: surface the failure
            from . import edi_io
            edi_io._note_fallback(path, exc)   # quirky EDI -> text parse just this one
    if mode == "mtpy":
        return None
    return _read_station_z_text(path)


def _read_station_z_mtpy(path: str) -> StationZ | None:
    from .. import config

    config.apply_sys_paths()
    config.ensure_mtpy_numpy_compat()
    import mtpy.core.mt as mt

    from . import edi_io

    obj = mt.MT(path)
    z = getattr(obj, "Z", None)
    if z is None or z.z is None or z.freq is None:
        return None
    freq = np.asarray(z.freq, dtype="float64").ravel()
    zz = np.asarray(z.z, dtype="complex128")                 # (n, 2, 2), field units
    ok = np.isfinite(freq) & (freq > 0)
    if ok.sum() < 2:
        return None
    freq, zz = freq[ok], zz[ok]
    zdict = {"zxx": zz[:, 0, 0], "zxy": zz[:, 0, 1],
             "zyx": zz[:, 1, 0], "zyy": zz[:, 1, 1]}
    return StationZ(station_id=edi_io._station_id_from_file(path),
                    path=path, freq=freq, z=zdict)


def _read_station_z_text(path: str) -> StationZ | None:
    from . import edi_io

    with open(path, "r", errors="replace") as fh:
        blob = fh.read()

    freq = _block("FREQ", blob)
    if freq is None or freq.size == 0:
        return None

    parts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c in _COMPONENTS:
        re_part = _block(c.upper() + "R", blob)
        im_part = _block(c.upper() + "I", blob)
        if re_part is None or im_part is None:
            return None
        parts[c] = (re_part, im_part)

    # one common length across FREQ and every component -- a malformed file
    # with ragged blocks is truncated to what they all share, not left to
    # broadcast-crash downstream
    n = min([freq.size] + [min(r.size, i.size) for r, i in parts.values()])
    if n < 2:
        return None
    z = {c: r[:n] + 1j * i[:n] for c, (r, i) in parts.items()}
    freq = freq[:n]
    return StationZ(station_id=edi_io._station_id_from_file(path, blob),
                    path=path, freq=freq, z=z)


def read_dir(edi_path: str, exclude: set[str] | None = None) -> list[StationZ]:
    """*edi_path* may be a directory or a ``.lst`` (see ``edi_io.resolve_edi_files``)."""
    from . import edi_io

    exclude = exclude or set()
    out = []
    for p in edi_io.resolve_edi_files(edi_path):
        st = read_station_z(p)
        if st and st.station_id not in exclude:
            out.append(st)
    return out


# --------------------------------------------------------------------------- #
@dataclass
class ResistivityCurve:
    """Station-averaged SSQ response on a common log-period grid.

    The average is the geometric mean of the complex ``Z_ssq`` across stations
    (``exp(mean(log Z))`` -- averages log-magnitude and phase), from which the
    log-mean rho, phase and C-response are derived.
    """
    period_s: np.ndarray
    rho_logmean: np.ndarray
    rho_median: np.ndarray
    rho_p16: np.ndarray
    rho_p84: np.ndarray
    phase_logmean_deg: np.ndarray
    c_real_km: np.ndarray
    c_imag_km: np.ndarray
    n_stations: int

    def c_response_depth_km(self, period_s: float) -> float:
        """Re{C} (the penetration/investigation depth) interpolated to *period_s*.

        Clamped to the curve's endpoints outside the data's period range.
        """
        return float(np.interp(period_s, self.period_s, self.c_real_km,
                               left=self.c_real_km[0], right=self.c_real_km[-1]))

    @property
    def rho_shallow(self) -> float:
        return float(self.rho_logmean[0])

    @property
    def rho_deep(self) -> float:
        return float(self.rho_logmean[-1])

    @property
    def rho_geomean(self) -> float:
        good = self.rho_logmean[np.isfinite(self.rho_logmean) & (self.rho_logmean > 0)]
        return float(np.exp(np.mean(np.log(good)))) if good.size else float("nan")

    @property
    def period_min_s(self) -> float:
        return float(self.period_s[0])

    @property
    def period_max_s(self) -> float:
        return float(self.period_s[-1])


def ssq_resistivity_curve(stations: list[StationZ],
                          n_decades_bins: int = 8) -> ResistivityCurve:
    if not stations:
        raise ValueError("no stations with impedance data")

    all_t = np.concatenate([s.periods for s in stations])
    all_t = all_t[np.isfinite(all_t) & (all_t > 0)]
    t_lo, t_hi = all_t.min(), all_t.max()
    n_bins = max(int(round(n_decades_bins * math.log10(t_hi / t_lo))), 6)
    grid = np.logspace(math.log10(t_lo), math.log10(t_hi), n_bins)
    log_grid = np.log10(grid)

    zc = np.full((len(stations), grid.size), np.nan + 0j, dtype="complex128")
    for i, s in enumerate(stations):
        t = s.periods
        z = s.z_ssq()
        order = np.argsort(t)
        t, z = t[order], z[order]
        ok = np.isfinite(t) & np.isfinite(z) & (np.abs(z) > 0)
        if ok.sum() < 2:
            continue
        lt = np.log10(t[ok])
        re_i = np.interp(log_grid, lt, z[ok].real, left=np.nan, right=np.nan)
        im_i = np.interp(log_grid, lt, z[ok].imag, left=np.nan, right=np.nan)
        zc[i] = re_i + 1j * im_i

    # geometric mean of the complex Z across stations, per period
    logz = np.log(zc)
    good = np.isfinite(logz)
    n = good.sum(axis=0)
    with np.errstate(all="ignore"):
        z_gm = np.exp(np.where(good, logz, 0).sum(axis=0) / np.where(n > 0, n, np.nan))

    rho = np.abs(zc) ** 2 / (2.0 * math.pi / grid * _MU0)   # per-station rho on grid
    with np.errstate(all="ignore"):
        logmean = np.abs(z_gm) ** 2 / (2.0 * math.pi / grid * _MU0)
        med = np.nanmedian(rho, axis=0)
        p16 = np.nanpercentile(rho, 16, axis=0)
        p84 = np.nanpercentile(rho, 84, axis=0)
    phase = np.degrees(np.arctan2(z_gm.imag, z_gm.real))
    c = z_gm / (1j * (2.0 * math.pi / grid) * _MU0) / 1.0e3

    keep = np.isfinite(logmean)
    return ResistivityCurve(
        period_s=grid[keep], rho_logmean=logmean[keep], rho_median=med[keep],
        rho_p16=p16[keep], rho_p84=p84[keep], phase_logmean_deg=phase[keep],
        c_real_km=c[keep].real, c_imag_km=c[keep].imag,
        n_stations=int(np.isfinite(zc).any(axis=1).sum()))
