"""Final step -- write the GoFEM solver input files.

  frequencies                  one frequency (Hz) per line, from the periods
                               ``EdiCollection.select_periods`` keeps, filtered
                               to [period_min, period_max].
  <name>_<type>_<pct>.gdat     inversion data, via
                               ``gofem.mtpy_utils.write_edi_collection_to_gofem``
                               with an error floor.
  input_modeling.prm           ``ParameterHandler(inversion=False)``
  input_inversion.prm          ``ParameterHandler(inversion=True)``

``receivers_projected.csv`` is written by :func:`mtinv_gui.backends.mesh.run`.

The inversion
"Active domain box" is derived from the cached survey scales (footprint +/- one
skin depth laterally, ~1.5 skin depths deep) unless the user pins one.
"""
from __future__ import annotations

import math
import os

FREQUENCIES_FILE = "frequencies"
MODELING_PRM = "input_modeling.prm"
INVERSION_PRM = "input_inversion.prm"


# --------------------------------------------------------------------------- #
def build_edi_collection(project, log=None):
    from . import edi_io

    return edi_io.load_collection(project.survey.edi_dir,
                                  set(project.survey.excluded), log=log)


def select_periods(collection, percentage: float,
                   period_min_s: float, period_max_s: float) -> list[float]:
    """``EdiCollection.select_periods(percentage=...)`` filtered to the range."""
    import numpy as np

    periods = np.sort(np.asarray(
        collection.select_periods(show=False, period_list=None,
                                  percentage=percentage), dtype="float64"))
    return [float(p) for p in periods if period_min_s <= p <= period_max_s]


def write_frequencies(periods_s: list[float], out_path: str) -> str:
    import numpy as np

    if not periods_s:
        raise ValueError("no periods selected -- widen the range or lower the percentage")
    freqs = np.array([1.0 / p for p in sorted(periods_s)], dtype="float64")
    np.savetxt(out_path, freqs, header="frequency (Hz)")
    return out_path


def write_data_file(collection, out_path: str, data_type: str,
                    error_floor: float, period_range: tuple[float, float]) -> list[str]:
    """Wrap ``gofem.mtpy_utils.write_edi_collection_to_gofem``.

    ``data_type`` in {Z, PT, Tipper, Z_Tipper}.  ``Z_Tipper`` writes two files
    (``*_Z`` and ``*_T``); merge them by hand if you want a single input.
    """
    from .. import config

    config.apply_sys_paths()
    from gofem.mtpy_utils import write_edi_collection_to_gofem

    types = ["Z", "Tipper"] if data_type == "Z_Tipper" else [data_type]
    written = []
    base, ext = os.path.splitext(out_path)
    for t in types:
        target = out_path if len(types) == 1 else f"{base}_{t}{ext or '.gdat'}"
        write_edi_collection_to_gofem(
            edi_collection=collection, outfile=target,
            error_floor=float(error_floor), data_type=t,
            period_range=list(period_range))
        written.append(target)
    return written


# --------------------------------------------------------------------------- #
def _rel(from_dir: str, target: str) -> str:
    """POSIX relative path from *from_dir* to *target* (GoFEM reads forward slashes)."""
    return os.path.relpath(target, from_dir).replace(os.sep, "/")


def _mesh_files(project, files: dict | None = None) -> dict:
    """Paths the .prm files point at: the Mesh tab's outputs unless *files* overrides them."""
    from . import mesh as _mesh

    d = _mesh.mesh_output_dir(project)
    out = {"tria": os.path.join(d, _mesh.MESH_TRIA),
           "materials": os.path.join(d, _mesh.MESH_MATERIALS),
           "mask": os.path.join(d, _mesh.MESH_MASK),
           "frequencies": os.path.join(d, FREQUENCIES_FILE),
           "receivers": os.path.join(d, _mesh.RECEIVERS_CSV)}
    out.update(files or {})
    return out


def _active_domain_box(project) -> list[float]:
    if project.gofem.active_domain_box:
        return list(project.gofem.active_domain_box)
    sc = project.mesh.scales
    if sc:
        d_max = 503.0 * math.sqrt(max(sc["rho_deep"], 1e-6)
                                  * max(sc["period_max_s"], 1e-12))
        hx = sc["footprint_x_m"] / 2.0 + d_max
        hy = sc["footprint_y_m"] / 2.0 + d_max
        return [-hx, -hy, -5000.0, hx, hy, 1.5 * d_max]
    return [-500000.0, -500000.0, -4000.0, 500000.0, 500000.0, 400000.0]


def _cooling_schedule(s0: float, n_blocks: int = 6, per_block: int = 5) -> list[float]:
    return [s0 / (3.0 ** i) for i in range(n_blocks) for _ in range(per_block)]


def write_modeling_prm(project, out_path: str, files: dict | None = None) -> str:
    from .. import config

    config.apply_sys_paths()
    from gofem.parameter_handler import ParameterHandler

    g = project.gofem
    prm_dir = os.path.dirname(os.path.abspath(out_path))
    f = _mesh_files(project, files)

    prm = ParameterHandler(inversion=False, dim=3)
    prm.setitem(section="Modeling", parameter="Order", value=int(g.element_order))
    prm.setitem(section="Modeling", parameter="Refine cells around receivers",
                value=int(g.refine_around_receivers))
    prm.setitem(section="Modeling", parameter="BC", value=g.boundary_condition)
    prm.setitem(section="Modeling", parameter="Number of parallel frequencies",
                value=int(g.n_parallel_frequencies))
    prm.setitem(section="Solver", parameter="Preconditioner", value="Direct")
    prm.setitem(section="Model", parameter="Model definition file",
                value=_rel(prm_dir, f["tria"]))
    prm.setitem(section="Model", parameter="Materials definition file",
                value=_rel(prm_dir, f["materials"]))
    prm.setitem(section="Survey", parameter="Frequencies file",
                value=_rel(prm_dir, f["frequencies"]))
    prm.setitem(section="Survey", parameter="Stations file",
                value=_rel(prm_dir, f["receivers"]))
    prm.setitem(section="Output", parameter="Type", value="point")
    prm.setitem(section="Output", parameter="Data file", value="model")
    prm.setitem(section="Output", parameter="Parallel output", value="false")
    prm.write(out_path)
    return out_path


def write_inversion_prm(project, out_path: str, data_filename: str, files: dict | None = None,
                        box: list[float] | None = None) -> str:
    """*files* overrides the mesh files (its ``mask`` may be "" to leave the mask out);
    *box* overrides the active domain box."""
    from .. import config

    config.apply_sys_paths()
    from gofem.parameter_handler import ParameterHandler

    g = project.gofem
    prm_dir = os.path.dirname(os.path.abspath(out_path))
    f = _mesh_files(project, files)

    prm = ParameterHandler(inversion=True, dim=3)
    prm.setitem(section="Modeling", parameter="Order", value=int(g.element_order))
    prm.setitem(section="Modeling", parameter="Refine cells around receivers",
                value=max(int(g.refine_around_receivers) - 1, 0))
    prm.setitem(section="Modeling", parameter="BC", value=g.boundary_condition)
    prm.setitem(section="Modeling", parameter="Number of parallel frequencies",
                value=int(g.n_parallel_frequencies))
    prm.setitem(section="Solver", parameter="Preconditioner", value="Direct")
    prm.setitem(section="Model", parameter="Inversion model definition file",
                value=_rel(prm_dir, f["tria"]))
    prm.setitem(section="Model", parameter="Inversion materials definition file",
                value=_rel(prm_dir, f["materials"]))
    prm.setitem(section="Model", parameter="Active domain box",
                value=box if box else _active_domain_box(project))
    if f["mask"]:
        prm.setitem(section="Model", parameter="Active domain mask",
                    value=_rel(prm_dir, f["mask"]))
    prm.setitem(section="Survey", parameter="Frequencies file",
                value=_rel(prm_dir, f["frequencies"]))
    prm.setitem(section="Survey", parameter="Stations file",
                value=_rel(prm_dir, f["receivers"]))
    prm.setitem(section="Output", parameter="Type", value="")
    prm.setitem(section="Inversion", parameter="Inversion input data",
                value=_rel(prm_dir, os.path.join(prm_dir, os.path.basename(data_filename))))
    prm.setitem(section="Inversion", parameter="Scaling factor",
                value=_cooling_schedule(float(g.scaling_factor_s0)))
    prm.setitem(section="Inversion", parameter="Target RMS", value=float(g.target_rms))
    prm.setitem(section="Inversion", parameter="Number of iterations",
                value=int(g.max_iterations))
    prm.setitem(section="Inversion", parameter="Number of inner iterations",
                value=int(g.inner_iterations))
    prm.setitem(section="Inversion", parameter="Regularization update",
                value=int(g.regularization_update))
    prm.write(out_path)
    return out_path


# --------------------------------------------------------------------------- #
def write_all(project, out_dir: str, log=print) -> dict:
    """frequencies + data file + both .prm files in one go."""
    from . import mesh as _mesh

    os.makedirs(out_dir, exist_ok=True)
    g = project.gofem
    mesh_dir = _mesh.mesh_output_dir(project)

    log("loading EDI collection (mtpy)...")
    coll = build_edi_collection(project, log=log)

    periods = select_periods(coll, g.period_percentage, g.period_min_s, g.period_max_s)
    freq_path = write_frequencies(periods, os.path.join(mesh_dir, FREQUENCIES_FILE))
    log("wrote %s (%d frequencies)" % (freq_path, len(periods)))

    stem = f"{project.name}_{g.data_type}_{int(g.error_floor * 100)}pct.gdat"
    data_paths = write_data_file(coll, os.path.join(out_dir, stem), g.data_type,
                                 g.error_floor, (g.period_min_s, g.period_max_s))
    log("wrote data file(s): %s" % ", ".join(os.path.basename(p) for p in data_paths))

    mod = write_modeling_prm(project, os.path.join(out_dir, MODELING_PRM))
    inv = write_inversion_prm(project, os.path.join(out_dir, INVERSION_PRM), data_paths[0])
    log("wrote %s and %s" % (os.path.basename(mod), os.path.basename(inv)))
    out = {"frequencies": freq_path, "data": data_paths,
           "modeling_prm": mod, "inversion_prm": inv, "periods": periods}
    if project.mesh.output_format == "femtic":
        out.update(write_femtic_run(project, out_dir, coll, periods, log=log))
    return out


FEMTIC_DIR = "femtic"


def write_femtic_run(project, out_dir: str, collection, periods_s, log=print) -> dict:
    """A self-contained FEMTIC run folder, ``<out_dir>/femtic``: the mesh's
    ``mesh.dat`` + ``resistivity_block_iter0.dat`` copied from the mesh folder, and
    ``observe.dat`` + ``control.dat`` from the same EDIs, frequencies and error floor
    as the GoFEM data file."""
    import shutil

    from . import femtic, femtic_data
    from . import mesh as _mesh
    from .refine import read_receivers

    g = project.gofem
    mesh_dir = _mesh.mesh_output_dir(project)
    src = [os.path.join(mesh_dir, f) for f in (femtic.MESH_DAT, femtic.BLOCKS_DAT)]
    rec = os.path.join(mesh_dir, _mesh.RECEIVERS_CSV)
    missing = [os.path.basename(p) for p in src + [rec] if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(f"FEMTIC: {', '.join(missing)} not in {mesh_dir} -- build the mesh "
                                "with 'Mesh For' = GoFEM + FEMTIC first")
    run_dir = os.path.join(out_dir, FEMTIC_DIR)
    os.makedirs(run_dir, exist_ok=True)
    for p in src:
        shutil.copyfile(p, os.path.join(run_dir, os.path.basename(p)))
    names, xyz = read_receivers(rec)
    sites = femtic_data.sites_from_collection(collection, names, xyz,
                                              [1.0 / p for p in periods_s], log=log)
    obs = femtic_data.write_observe(sites, os.path.join(run_dir, femtic_data.OBSERVE_DAT),
                                    g.data_type, g.error_floor, log=log)
    ctl = femtic_data.write_control(os.path.join(run_dir, femtic_data.CONTROL_DAT),
                                    iterations=g.femtic_iterations, n_threads=g.femtic_threads,
                                    trade_off=g.femtic_trade_off)
    log(f"FEMTIC run folder: {run_dir} ({'forward run' if g.femtic_iterations == 0 else f'{g.femtic_iterations} iterations'})")
    return {"femtic_dir": run_dir, "femtic_observe": obs["observe"], "femtic_control": ctl}


# --------------------------------------------------------------------------- #
# inputs for a refined model (Refine Model tab)
# --------------------------------------------------------------------------- #
def _read_frequencies(path: str) -> list[float]:
    import numpy as np

    return [float(v) for v in np.atleast_1d(np.loadtxt(path, comments="#"))]


def merge_frequencies(*lists, rel_tol: float = 1e-4) -> list[float]:
    """All frequencies, highest first, with values within *rel_tol* kept once."""
    out: list[float] = []
    for f in sorted((v for values in lists for v in values), reverse=True):
        if not out or abs(out[-1] - f) > rel_tol * f:
            out.append(f)
    return out


def parse_domain_box(text: str) -> list[float]:
    """``n0, e0, z0, n1, e1, z1`` in metres -> six floats ([] if blank)."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        box = [float(v) for v in text.replace(",", " ").split()]
    except ValueError:
        box = []
    if len(box) != 6:
        raise ValueError(f"the active domain box needs six numbers (n0, e0, z0, n1, e1, z1): {text!r}")
    return box


def _data_rows(path: str) -> tuple[list[str], list[str]]:
    """``.gdat``: comment lines, then data lines whose 4th column is the station."""
    header, rows = [], []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            (header if line.lstrip().startswith("#") or not line.strip() else rows).append(line)
    return header, rows


def check_refined_inputs(project, spec: dict) -> list[float]:
    """Everything :func:`write_refined_inputs` needs from the user, checked up front so a
    long refinement doesn't end in an error.  Returns the active domain box."""
    # never a built-in default: a box that's too small silently freezes part of the model
    box = parse_domain_box(spec.get("active_domain_box", "")) or list(project.gofem.active_domain_box)
    if not box:
        raise ValueError("enter the Active Domain Box on the Output page: cells outside it aren't inverted")
    mode = spec.get("station_mode", "none")
    if mode == "none" and not spec.get("receivers"):
        raise ValueError("GoFEM files need stations: choose the receivers, or new stations")
    if mode != "none" and not spec.get("stations_edi"):
        raise ValueError("choose the new stations' EDIs")
    for label, key in (("data file", "existing_data"), ("frequencies file", "existing_frequencies")):
        path = (spec.get(key) or "").strip()
        if mode == "none" and not path:
            raise ValueError(f"choose the existing {label}: this model keeps its data")
        if mode != "replace" and path and not os.path.isfile(path):
            raise ValueError(f"can't find the existing {label}: {path}")
    return box


def write_refined_inputs(project, spec: dict, result: dict, log=print) -> dict:
    """Frequencies, data and both ``.prm`` files for a model written by ``refine.run``.

    The data depends on ``spec["station_mode"]``:
      * ``none``: the existing ``.gdat`` and frequencies, copied;
      * ``replace``: the new stations' EDIs only;
      * ``add``: the existing data plus the new stations', with frequencies combined.
    Inversion and modelling settings come from the GoFEM Files tab.
    """
    import numpy as np

    from . import edi_io

    paths = result["paths"]
    out_dir = os.path.dirname(os.path.abspath(paths["tria"]))
    g = project.gofem
    box = check_refined_inputs(project, spec)
    mode = spec.get("station_mode", "none")
    old_data = (spec.get("existing_data") or "").strip()
    old_freq = (spec.get("existing_frequencies") or "").strip()

    frequencies, rows, header = [], [], ["# DataType Frequency Source Receiver Value Error\n"]
    if mode != "replace" and old_data:
        header, rows = _data_rows(old_data)
        log(f"existing data: {len(rows):,} lines")
    if mode != "replace" and old_freq:
        frequencies = _read_frequencies(old_freq)

    if mode != "none":
        coll = edi_io.load_collection(spec["stations_edi"], log=log)
        periods = select_periods(coll, g.period_percentage, g.period_min_s, g.period_max_s)
        if not periods:
            raise ValueError("no periods selected: widen the range or lower the percentage "
                             "on the GoFEM Files tab")
        frequencies = merge_frequencies(frequencies, [1.0 / p for p in periods])
        new_names = {n for n, is_new in zip(result["names"], result["is_new"]) if is_new}
        tmp = os.path.join(out_dir, "_new_data.gdat")
        written = write_data_file(coll, tmp, g.data_type, g.error_floor, (g.period_min_s, g.period_max_s))
        added = 0
        for path in written:
            _h, new_rows = _data_rows(path)
            keep = [r for r in new_rows if r.split()[3] in new_names]
            rows.extend(keep)
            added += len(keep)
            os.remove(path)
        log(f"new data: {added:,} lines from {len(new_names)} stations")

    if not rows:
        raise ValueError("no data to write: there are no new stations and no existing data")
    known = set(result["names"])
    orphans = sorted({r.split()[3] for r in rows} - known)
    if orphans:
        log(f"WARNING: {len(orphans)} stations in the data have no receiver (e.g. "
            f"{', '.join(orphans[:5])}); GoFEM may reject this")

    stem = spec.get("stem") or "model_refined"
    freq_path = os.path.join(out_dir, FREQUENCIES_FILE)
    np.savetxt(freq_path, np.asarray(sorted(frequencies, reverse=True), dtype="float64"),
               header="frequency (Hz)")
    data_path = os.path.join(out_dir, f"{stem}_{g.data_type}.gdat")
    with open(data_path, "w") as fh:
        fh.writelines(header + rows)
    files = {"tria": paths["tria"], "materials": paths.get("materials", ""),
             "mask": paths.get("mask", ""), "frequencies": freq_path, "receivers": paths["receivers"]}
    mod = write_modeling_prm(project, os.path.join(out_dir, MODELING_PRM), files)
    inv = write_inversion_prm(project, os.path.join(out_dir, INVERSION_PRM), data_path, files, box=box)
    log(f"wrote {os.path.basename(freq_path)} ({len(frequencies)} frequencies), "
        f"{os.path.basename(data_path)}, {MODELING_PRM}, {INVERSION_PRM}")
    log("active domain box: " + ", ".join(f"{v:g}" for v in box))
    return {"frequencies": freq_path, "data": data_path, "modeling_prm": mod, "inversion_prm": inv,
            "n_frequencies": len(frequencies), "n_data": len(rows), "box": box}
