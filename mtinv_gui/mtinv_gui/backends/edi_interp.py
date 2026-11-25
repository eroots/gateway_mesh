"""Task 0 -- optional pre-processing: interpolate a folder of EDIs onto one
shared, logarithmically-spaced period grid before the Survey / EDIs step.

Unlike :mod:`edi_io` (a dependency-free header reader so the map view works
before the scientific env is set up), this needs a real ``mtpy`` install --
it uses ``mtpy.core.mt.MT.interpolate`` to resample each station's actual
impedance tensor (and tipper, if present), then writes new EDI files.
"""
from __future__ import annotations

import math
import os

from . import edi_io


def periods_from_decade(period_min_s: float, period_max_s: float,
                        periods_per_decade: float) -> list[float]:
    """Log-spaced periods from *period_min_s* to *period_max_s* (both
    endpoints included), at *periods_per_decade* points per decade."""
    if period_min_s <= 0 or period_max_s <= period_min_s:
        raise ValueError("need 0 < period_min_s < period_max_s")
    import numpy as np

    n_decades = math.log10(period_max_s / period_min_s)
    n = max(int(round(n_decades * periods_per_decade)) + 1, 2)
    return np.logspace(math.log10(period_min_s), math.log10(period_max_s), n).tolist()


def interpolate_collection(edi_path: str, out_dir: str, *, period_min_s: float,
                           period_max_s: float, periods_per_decade: float,
                           interp_type: str = "slinear", log=print) -> dict:
    """Interpolate every EDI resolved from *edi_path* onto a shared log-period
    grid; write the results (same basenames) into *out_dir*.

    Stations don't all cover the same period band, and ``MT.interpolate``
    raises if asked to extrapolate past a station's own native range, so each
    station is first clipped to the intersection of the requested range and
    its own coverage -- a station with narrower coverage gets fewer periods
    (logged), not a batch-aborting crash. A station mtpy can't read/write at
    all is logged and skipped, same reasoning.

    Returns ``{"written": [...], "skipped": [...], "n_periods": int,
    "periods": [...], "out_dir": out_dir}``.
    """
    from .. import config
    config.apply_sys_paths()
    config.ensure_mtpy_numpy_compat()
    import numpy as np
    import mtpy.core.mt as mt

    periods = periods_from_decade(period_min_s, period_max_s, periods_per_decade)
    freqs_target = np.sort(1.0 / np.asarray(periods))

    files = edi_io.resolve_edi_files(edi_path)
    if not files:
        raise ValueError(f"no EDI files resolved from {edi_path!r}")
    os.makedirs(out_dir, exist_ok=True)

    written: list[str] = []
    skipped: list[str] = []
    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        try:
            mt_obj = mt.MT(path)
            if not mt_obj.station:                # mtpy doesn't always fill this on
                mt_obj.station = name              # read; write_mt_file needs it set
            fmin, fmax = float(mt_obj.Z.freq.min()), float(mt_obj.Z.freq.max())
            freqs = freqs_target[(freqs_target >= fmin) & (freqs_target <= fmax)]
            if len(freqs) < 2:
                skipped.append(name)
                log(f"  {name}: skipped -- native range {1/fmax:.4g}-{1/fmin:.4g} s "
                   f"doesn't overlap the requested {period_min_s:.4g}-{period_max_s:.4g} s")
                continue
            new_z, new_tipper = mt_obj.interpolate(
                freqs, interp_type=interp_type, bounds_error=False)
            out_fn = mt_obj.write_mt_file(
                save_dir=out_dir, fn_basename=f"{name}.edi", file_type="edi",
                new_Z_obj=new_z, new_Tipper_obj=new_tipper)
            written.append(out_fn)
            if len(freqs) < len(freqs_target):
                log(f"  {name}: {len(freqs)}/{len(freqs_target)} periods -- native "
                   f"range {1/fmax:.4g}-{1/fmin:.4g} s narrower than requested")
        except Exception as exc:  # noqa: BLE001 -- one bad station shouldn't abort the batch
            skipped.append(name)
            log(f"  {name}: ERROR -- {exc}")

    log(f"interpolated {len(written)}/{len(files)} stations to {len(freqs_target)} "
       f"periods ({period_min_s:.4g}-{period_max_s:.4g} s) -> {out_dir}")
    if skipped:
        log(f"  skipped: {', '.join(skipped)}")
    return {"written": written, "skipped": skipped, "n_periods": len(freqs_target),
           "periods": periods, "out_dir": out_dir}


def read_station_response(edi_fn: str) -> dict:
    """One station's period-indexed apparent resistivity / phase / tipper,
    for the before/after QC plot -- via mtpy, same as the interpolation
    itself (real Z, not just the header summary :mod:`edi_io` reads).

    ``phase_yx`` is shifted +180 deg into the same quadrant as ``phase_xy``
    for display -- mtpy's own plotting code (``plotresponse.py``) uses the
    same convention ("add 180 to place it in same quadrant as phase_xy").
    ``tipper`` is ``None`` when the EDI has no (or all-zero) tipper block.
    """
    from .. import config
    config.ensure_mtpy_numpy_compat()
    import numpy as np
    import mtpy.core.mt as mt

    mt_obj = mt.MT(edi_fn)
    z = mt_obj.Z
    out = {
        "periods": (1.0 / z.freq).tolist(),
        "res_xy": z.res_xy.tolist(), "res_yx": z.res_yx.tolist(),
        "res_xy_err": z.res_err_xy.tolist(), "res_yx_err": z.res_err_yx.tolist(),
        "phase_xy": z.phase_xy.tolist(), "phase_yx": (z.phase_yx + 180.0).tolist(),
        "phase_xy_err": z.phase_err_xy.tolist(), "phase_yx_err": z.phase_err_yx.tolist(),
        "tipper": None,
    }
    tp = mt_obj.Tipper
    if tp.tipper is not None and not np.all(tp.tipper == 0) and not np.all(np.isnan(tp.tipper)):
        out["tipper"] = {
            "periods": (1.0 / tp.freq).tolist(),
            "tzx_re": tp.tipper[:, 0, 0].real.tolist(), "tzx_im": tp.tipper[:, 0, 0].imag.tolist(),
            "tzy_re": tp.tipper[:, 0, 1].real.tolist(), "tzy_im": tp.tipper[:, 0, 1].imag.tolist(),
        }
    return out
