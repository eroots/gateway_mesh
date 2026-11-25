"""FEMTIC run inputs besides the mesh: ``observe.dat`` (the data) and ``control.dat``.

The writers take plain per-station arrays (:class:`SiteData`); :func:`sites_from_collection`
is the adapter for the setup GUI's mtpy ``EdiCollection``.  gateway_mt has its own writer
for its ``Data`` (``gateway_mt.IO.write_femtic``, Data Plot's Write > FEMTIC): the same file,
checked against this one on the same EDIs.  It picks the same frequencies,
and :func:`write_observe` applies the same error floors, as the GoFEM data file
(``gofem.mtpy_utils.write_edi_collection_to_gofem``), so a FEMTIC and a GoFEM run
of one project see the same data.

``observe.dat`` (FEMTIC ``ObservedData::inputObservedData`` and the station
classes; the manual's "File format of observe.dat"):

  ``MT n`` / ``PT n`` / ``VTF n`` blocks, then ``END``.  Per station:
  ``id  id_of_H_station  x_km  y_km``, ``n_freq``, then one line per frequency:
  MT:  ``f  ReZxx ImZxx ReZxy ImZxy ReZyx ImZyx ReZyy ImZyy`` + 8 standard errors
  PT:  ``f  P11 P12 P21 P22`` + 4 standard errors
  VTF: ``f  ReTzx ImTzx ReTzy ImTzy`` + 4 standard errors
  A negative error leaves that number out of the inversion.

Conventions, all different from an EDI file:
  * SI units: Z in V/A = ohm.  EDI's mV/km/nT times 4 pi 1e-4 (as GoFEM's writer);
  * time dependence e^(-i w t): the EDI's (e^(+i w t)) complex conjugate.  FEMTIC's
    own examples show it: Zxy in the 4th quadrant, a 1-D phase tensor of about -I;
  * x = north, y = east in km, the mesh frame.  Only x and y: FEMTIC puts each
    station on the earth surface of ``mesh.dat`` itself (land, or the seafloor);
  * station ids are unique across all blocks, so tipper stations get their own
    (``VTF_ID_OFFSET``: 101, 102, ... for under 100 stations, as in FEMTIC's
    examples).  ``observe_sites.csv`` maps ids back to names.

Like GoFEM's writer, Z isn't rotated: the EDI's x / y are taken as the mesh's north
/ east.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

OBSERVE_DAT = "observe.dat"
CONTROL_DAT = "control.dat"
SITES_CSV = "observe_sites.csv"

#: mV/km/nT -> V/A (ohm): mu0 * 1e3
Z_TO_SI = 4.0 * math.pi * 1e-4

DATA_TYPES = ("Z", "Z_offdiag", "PT", "Tipper", "Z_Tipper")

#: GoFEM's writer drops values this small (missing components written as 0 in EDIs)
_TINY = 1e-25


@dataclass
class SiteData:
    """One station's transfer functions on the run's frequencies, as in an EDI file:
    Z in mV/km/nT with e^(+i w t), tipper (Tzx, Tzy), standard errors; NaN = missing."""
    name: str
    north_m: float
    east_m: float
    freq: np.ndarray                    # (nf,) Hz
    z: np.ndarray | None = None         # (nf, 2, 2) complex
    z_err: np.ndarray | None = None     # (nf, 2, 2)
    tipper: np.ndarray | None = None    # (nf, 2) complex
    tipper_err: np.ndarray | None = None  # (nf, 2)


# --------------------------------------------------------------------------- #
# from the setup GUI's EdiCollection
# --------------------------------------------------------------------------- #
def sites_from_collection(collection, names, xyz, frequencies, *, ptol: float = 0.05,
                          log=print) -> list[SiteData]:
    """A :class:`SiteData` per station that has an EDI *and* a receiver.

    *names*, *xyz*: the mesh's receivers (``receivers_projected.csv``: north, east,
    depth in m).  Each EDI frequency within *ptol* of one of *frequencies* is written
    at that frequency (GoFEM's writer matches the same way), so every station shares
    the run's frequency list and FEMTIC solves only those.
    """
    freqs = np.asarray(sorted(frequencies, reverse=True), dtype="float64")
    pos = {n: p for n, p in zip(names, np.asarray(xyz, dtype="float64"))}
    sites, no_receiver = [], []
    for obj in collection.mt_obj_list:
        if obj.station not in pos:
            no_receiver.append(obj.station)
            continue
        own = np.asarray(obj.Z.freq, dtype="float64")
        idx = np.full(len(freqs), -1)
        for i, f in enumerate(freqs):
            hit = np.flatnonzero((own > f * (1 - ptol)) & (own < f * (1 + ptol)))
            if hit.size:
                idx[i] = hit[0]
        ok = idx >= 0
        nf = len(freqs)
        z = np.full((nf, 2, 2), np.nan, complex)
        ze = np.full((nf, 2, 2), np.nan)
        z[ok], ze[ok] = obj.Z.z[idx[ok]], obj.Z.z_err[idx[ok]]
        t = np.full((nf, 2), np.nan, complex)
        te = np.full((nf, 2), np.nan)
        tip = getattr(obj, "Tipper", None)
        if tip is not None and getattr(tip, "tipper", None) is not None:
            t[ok] = np.asarray(tip.tipper)[idx[ok]].reshape(-1, 2)
            if getattr(tip, "tipper_err", None) is not None:
                te[ok] = np.asarray(tip.tipper_err)[idx[ok]].reshape(-1, 2)
        p = pos[obj.station]
        sites.append(SiteData(obj.station, float(p[0]), float(p[1]), freqs, z, ze, t, te))
    if no_receiver:
        log(f"FEMTIC data: {len(no_receiver)} EDI(s) have no receiver in the mesh, left out: "
            + ", ".join(no_receiver[:10]) + (" ..." if len(no_receiver) > 10 else ""))
    have = {s.name for s in sites}
    missing = [n for n in names if n not in have]
    if missing:
        log(f"FEMTIC data: {len(missing)} receiver(s) have no EDI: "
            + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
    order = {n: i for i, n in enumerate(names)}
    return sorted(sites, key=lambda s: order[s.name])


# --------------------------------------------------------------------------- #
# response functions in FEMTIC's units and convention
# --------------------------------------------------------------------------- #
def _usable(v):
    return np.isfinite(v) & (np.abs(v) > _TINY)


def impedance_rows(site: SiteData, error_floor: float, *, offdiag_only: bool = False):
    """(freq, values (8,), errors (8,)) per frequency with any usable component.

    Errors follow GoFEM's 'rowwise' floor: the x row (Zxx, Zxy) gets
    max(err(Zxy), floor |Zxy|), the y row max(err(Zyx), floor |Zyx|), real and
    imaginary parts alike."""
    rows = []
    for f, z, ze in zip(site.freq, site.z, site.z_err):
        d_row = [np.fmax(ze[0, 1], error_floor * abs(z[0, 1])), np.fmax(ze[1, 0], error_floor * abs(z[1, 0]))]
        zf = np.conj(z) * Z_TO_SI                  # e^(-iwt), V/A
        vals, errs = [], []
        for i, j in ((0, 0), (0, 1), (1, 0), (1, 1)):
            e = d_row[i] * Z_TO_SI
            use_ij = not (offdiag_only and i == j)
            for part in (zf[i, j].real, zf[i, j].imag):
                ok = use_ij and _usable(part) and np.isfinite(e) and e > 0
                vals.append(part if ok else 0.0)
                errs.append(e if ok else -1.0)
        if max(errs) > 0:
            rows.append((float(f), np.array(vals), np.array(errs)))
    return rows


def phase_tensor(z, dz=None):
    """Phase tensor X^-1 Y of Z = X + iY (Caldwell et al. 2004), and with *dz* (2, 2)
    its linearly propagated error (as gofem.data_utils.calculate_phase_tensor)."""
    def pt(zz):
        x, y = zz.real, zz.imag
        return np.linalg.solve(x, y)

    p = pt(z)
    if dz is None:
        return p
    var = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            for unit in (1.0, 1j):
                h = 1e-7 * max(abs(z).max(), 1e-30)
                zp, zm = z.copy(), z.copy()
                zp[i, j] += unit * h
                zm[i, j] -= unit * h
                var += ((pt(zp) - pt(zm)) / (2 * h) * dz[i, j]) ** 2
    return p, np.sqrt(var)


def phase_tensor_rows(site: SiteData, error_floor: float):
    """(freq, PT (4,), errors (4,)) from the conjugated impedance and the floored
    errors of :func:`impedance_rows` (a 1-D earth gives about -I in this convention)."""
    rows = []
    for f, z, ze in zip(site.freq, site.z, site.z_err):
        d_row = [np.fmax(ze[0, 1], error_floor * abs(z[0, 1])), np.fmax(ze[1, 0], error_floor * abs(z[1, 0]))]
        zf = np.conj(z) * Z_TO_SI
        dz = np.array([[d_row[0]] * 2, [d_row[1]] * 2]) * Z_TO_SI
        if not (np.isfinite(zf).all() and np.isfinite(dz).all()):
            continue
        if abs(np.linalg.det(zf.real)) < 1e-300:
            continue
        p, dp = phase_tensor(zf, dz)
        if np.isfinite(p).all() and np.isfinite(dp).all():
            rows.append((float(f), p.ravel(), dp.ravel()))
    return rows


def tipper_rows(site: SiteData, error_floor: float):
    """(freq, values (4,), errors (4,)): conjugated (Tzx, Tzy), errors
    max(err, floor) as an absolute floor (GoFEM's writer does the same)."""
    rows = []
    if site.tipper is None:
        return rows
    te_all = site.tipper_err if site.tipper_err is not None else np.full(site.tipper.shape, np.nan)
    for f, t, te in zip(site.freq, site.tipper, te_all):
        vals, errs = [], []
        for k in range(2):
            e = max(te[k], error_floor) if np.isfinite(te[k]) else error_floor
            tc = np.conj(t[k])
            for part in (tc.real, tc.imag):
                ok = _usable(part) and np.isfinite(e) and e > 0
                vals.append(part if ok else 0.0)
                errs.append(e if ok else -1.0)
        if max(errs) > 0:
            rows.append((float(f), np.array(vals), np.array(errs)))
    return rows


# --------------------------------------------------------------------------- #
# writers
# --------------------------------------------------------------------------- #
def vtf_id_offset(n_sites: int) -> int:
    """100 for up to 99 stations, 1000 up to 999, ... (FEMTIC's examples: 1-36, 101-136)."""
    return 10 ** max(2, len(str(max(n_sites, 1))))


def _block(kind, entries):
    lines = [f"{kind} {len(entries)}"]
    for sid, site, rows in entries:
        lines.append(f"{sid} {sid} {site.north_m / 1e3:.6f} {site.east_m / 1e3:.6f}")
        lines.append(str(len(rows)))
        for f, vals, errs in rows:
            lines.append(f"{f:.6e} " + " ".join(f"{v:.6e}" for v in vals)
                         + " " + " ".join(f"{e:.6e}" for e in errs))
    return lines


def write_observe(sites: list[SiteData], path: str, data_type: str = "Z",
                  error_floor: float = 0.05, log=print) -> dict:
    """Write ``observe.dat`` and ``observe_sites.csv`` (id, block, name, north_m,
    east_m) next to it.  *data_type*: Z, Z_offdiag, PT, Tipper or Z_Tipper.
    Stations with nothing usable for a block are left out of it."""
    if data_type not in DATA_TYPES:
        raise ValueError(f"FEMTIC data type {data_type!r}: expected one of {', '.join(DATA_TYPES)}")
    kinds = {"Z": ["MT"], "Z_offdiag": ["MT"], "PT": ["PT"], "Tipper": ["VTF"],
             "Z_Tipper": ["MT", "VTF"]}[data_type]
    offset = vtf_id_offset(len(sites))
    lines, table, counts = [], [], {}
    for kind in kinds:
        entries = []
        for i, s in enumerate(sites):
            if kind == "MT":
                rows = impedance_rows(s, error_floor, offdiag_only=(data_type == "Z_offdiag"))
            elif kind == "PT":
                rows = phase_tensor_rows(s, error_floor)
            else:
                rows = tipper_rows(s, error_floor)
            if rows:
                sid = i + 1 + (offset if kind == "VTF" else 0)
                entries.append((sid, s, rows))
                table.append((sid, kind, s.name, s.north_m, s.east_m))
        dropped = len(sites) - len(entries)
        if dropped:
            log(f"FEMTIC data: {dropped} station(s) have no usable {kind} data, left out of that block")
        if entries:
            lines += _block(kind, entries)
        counts[kind] = len(entries)
    if not any(counts.values()):
        raise ValueError("no usable data for observe.dat")
    lines.append("END")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    csv = os.path.join(os.path.dirname(os.path.abspath(path)), SITES_CSV)
    with open(csv, "w") as fh:
        fh.write("id,block,name,north_m,east_m\n")
        for sid, kind, name, n, e in table:
            fh.write(f"{sid},{kind},{name},{n:.3f},{e:.3f}\n")
    n_freq = len({f for s in sites for f in s.freq})
    log(f"FEMTIC: wrote {os.path.basename(path)} ("
        + ", ".join(f"{k} {v} stations" for k, v in counts.items()) + f", {n_freq} frequencies)")
    return {"observe": path, "sites_csv": csv, "counts": counts}


def control_text(*, iterations: int = 20, n_threads: int = 4, trade_off: float = 10.0,
                 extra: dict | None = None) -> str:
    """``control.dat``: the manual's "recommended setting for initial trial" with
    MESH_TYPE 2 (DHEXA).  *iterations* 0 = a forward run only (FEMTIC then needs
    no trade-off parameter, but takes one).  *extra*: more ``{KEYWORD: "lines"}``."""
    items = {
        "NUM_THREADS": f"{int(n_threads)}",
        "MESH_TYPE": "2",
        "OUTPUT_PARAM": "1\n0",                 # resistivity only
        "DIV_NUM_RHS_FWD": "10",
        "INV_METHOD": "1",                      # data space
        "SMALL_VALUE": "1.0e-4",
        "DISTORTION": "0",
        "TRADE_OFF_PARAM": f"{float(trade_off):g}",
        "ITERATION": f"0 {int(iterations)}",
        "STEP_LENGTH": "0.5 0.1 0.8\n2\n0.6 1.2",
    }
    items.update(extra or {})
    return "".join(f"{k}\n{v}\n" for k, v in items.items()) + "END\n"


def write_control(path: str, **kw) -> str:
    with open(path, "w") as fh:
        fh.write(control_text(**kw))
    return path


# --------------------------------------------------------------------------- #
# reading back (tests, and later the plotter)
# --------------------------------------------------------------------------- #
_N_VALUES = {"MT": 16, "PT": 8, "VTF": 8}


def read_observe(path: str) -> dict:
    """{block: [(id, x_km, y_km, array (n_freq, 1 + values))]} for MT / PT / VTF
    blocks written with one owner / field type per station (the default)."""
    tok = open(path).read().split()
    out, i = {}, 0
    while i < len(tok) and tok[i] != "END":
        kind, n = tok[i], int(tok[i + 1])
        if kind not in _N_VALUES:
            raise ValueError(f"{path}: block {kind} not supported here")
        i += 2
        stations = []
        for _ in range(n):
            sid, x, y, nf = int(tok[i]), float(tok[i + 2]), float(tok[i + 3]), int(tok[i + 4])
            i += 5
            w = 1 + _N_VALUES[kind]
            arr = np.array(tok[i:i + nf * w], dtype="float64").reshape(nf, w)
            i += nf * w
            stations.append((sid, x, y, arr))
        out[kind] = stations
    return out
