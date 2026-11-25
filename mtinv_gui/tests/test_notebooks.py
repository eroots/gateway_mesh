"""The notebooks in ``notebooks/``, and the helpers they use.

The helper tests are quick.  ``test_notebooks_run`` executes every notebook in order on
the sample data (in a temporary copy, so ``notebooks/work`` is left alone), then the mesh
notebook again with the real ocean and with none.  It downloads the DEM from GMRT, so it
is skipped without a connection, and takes a few minutes.

    python tests/test_notebooks.py            # everything
    python tests/test_notebooks.py quick      # the helpers and the notebook checks only
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtinv_gui.backends import mesh as M  # noqa: E402
from mtinv_gui.backends import projection as P  # noqa: E402

NB_DIR = os.path.join(ROOT, "notebooks")
ORDER = ["00_start_here", "01_interpolate_edis", "02_survey_and_topography", "03_mesh_and_receivers",
         "04_refine_inverted_model", "05_plot_model", "06_model_to_geotiff"]


# ---- helpers ------------------------------------------------------------------- #
def test_frame_file_round_trip_and_georef():
    from mtinv_gui.plotting.backends import georef as G

    for frame in (P.make_local_frame("local", 0, -121.45, 36.72, "lcc", 36.0, 37.5),
                  P.make_local_frame("utm", 32610, -121.45, 36.72)):
        with tempfile.TemporaryDirectory() as d:
            path = P.save_frame(frame, os.path.join(d, "model_frame.json"))
            back = P.load_frame(path)
            saved = json.load(open(path))
        assert back == frame
        gr = G.from_frame(back)
        assert gr.crs == saved["crs"]
        # a point 10 km east and 20 km north of the origin, both ways
        e, n = P.project_lonlat([-121.3], [36.9], frame)
        from pyproj import Transformer
        tx, ty = Transformer.from_crs(4326, gr.crs, always_xy=True).transform(-121.3, 36.9)
        assert abs(tx - gr.x0 - e[0]) < 1e-6 and abs(ty - gr.y0 - n[0]) < 1e-6


def test_mesh_plots_draw_a_small_mesh():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mtinv_gui.backends import mesh_plots

    M.prepare_pygofem()
    dx, dy, dz = M.cell_size_lists([20e3] * 3, [10e3, 20e3], [10e3, 20e3])
    tria = M.base_grid(dx, dy, dz, log=lambda *_: None)
    materials = M.initial_materials(tria, 100.0, log=lambda *_: None)
    cells = mesh_plots.mesh_cells(tria, materials)
    assert len(cells.north) == 6 * 6 * 2                       # the earth half
    assert np.allclose(cells.log10_rho, 2.0)
    ax = mesh_plots.plot_plan(cells, 5e3, stations=[[0.0, 0.0, 0.0]])
    assert len(ax.collections[0].get_paths()) == 36
    ax = mesh_plots.plot_section(cells, north_m=1.0, color_by="size", stations=[[50e3, 0, 0]])
    assert len(ax.collections[0].get_paths()) == 12
    assert len(ax.lines[0].get_xdata()) == 0                   # that station is 50 km off the section
    plt.close("all")


def test_committed_notebooks_are_clean():
    """No outputs or execution counts in the repository's notebooks (they'd hold paths and
    megabytes of images), and the order above matches the files."""
    import nbformat

    names = sorted(f[:-6] for f in os.listdir(NB_DIR) if f.endswith(".ipynb"))
    assert names == ORDER, names
    for name in names:
        nb = nbformat.read(os.path.join(NB_DIR, name + ".ipynb"), as_version=4)
        nbformat.validate(nb)
        for cell in nb.cells:
            if cell.cell_type == "code":
                assert not cell.outputs and cell.execution_count is None, f"{name} has outputs"


# ---- running them ---------------------------------------------------------------- #
def _gmrt_reachable() -> bool:
    import requests

    try:
        return requests.head("https://www.gmrt.org", timeout=15).status_code < 500
    except Exception:  # noqa: BLE001
        return False


def _run(nb_dir, name, settings=None, log=print):
    """Execute one notebook in *nb_dir*; *settings* replace lines of its Settings cell
    (``{"OCEAN = ": 'OCEAN = "real"'}``).  Returns the executed notebook."""
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(os.path.join(nb_dir, name + ".ipynb"), as_version=4)
    for old, new in (settings or {}).items():
        cell = next(c for c in nb.cells if c.cell_type == "code" and old in c.source)
        cell.source = "\n".join(new if line.startswith(old) else line for line in cell.source.splitlines())
    t = time.time()
    NotebookClient(nb, timeout=3600, kernel_name="python3",
                   resources={"metadata": {"path": nb_dir}}).execute()
    log(f"  {name}{' ' + str(list((settings or {}).values())) if settings else ''}: {time.time() - t:.0f} s")
    return nb


def _text(nb) -> str:
    return "\n".join(o.get("text", "") for c in nb.cells for o in c.get("outputs", []))


def dem_at(mesh_dir, xyz):
    """The notebooks' projected DEM at SimPEG points (x east, y north)."""
    work = os.path.dirname(mesh_dir)
    dem = M.read_mesh_dem(os.path.join(work, "02_survey", "DEM_projected_gridded.npz"), log=lambda *_: None)
    return dem.elevation(np.c_[xyz[:, 1], xyz[:, 0]])


def test_notebooks_run(log=print):
    if not _gmrt_reachable():
        log("  skipped: GMRT (the DEM download) isn't reachable")
        return
    with tempfile.TemporaryDirectory() as tmp:
        nb_dir = os.path.join(tmp, "notebooks")
        shutil.copytree(NB_DIR, nb_dir, ignore=shutil.ignore_patterns("work", ".ipynb_checkpoints"))
        work = os.path.join(nb_dir, "work")
        out = {name: _run(nb_dir, name, log=log) for name in ORDER}

        # 01-02
        assert len(os.listdir(os.path.join(work, "01_interpolated_edis"))) == 31
        frame = P.load_frame(os.path.join(work, "02_survey", "model_frame.json"))
        assert frame.kind == "local" and frame.local_proj == "tmerc"
        import re
        largest = re.search(r"EDI minus DEM elevation: median -?\d+ m, largest (\d+) m", _text(out["02_survey_and_topography"]))
        assert largest and int(largest.group(1)) < 100        # the detailed DEM is in place

        # 03: the mesh covers the DEM, has the thin-sheet ocean, and every station is on its surface
        text = _text(out["03_mesh_and_receivers"])
        assert "the DEM covers the whole mesh" in text and "100 conductivity classes" in text
        mesh_dir = os.path.join(work, "03_mesh")
        for f in ("model.tria", "model.vtu", "model.materials", "model.mask", "receivers_projected.csv"):
            assert os.path.getsize(os.path.join(mesh_dir, f)) > 0, f
        from mtinv_gui.backends import refine as R
        names, xyz = R.read_receivers(os.path.join(mesh_dir, "receivers_projected.csv"))
        tria = R.load_tria(os.path.join(mesh_dir, "model.tria"))
        assert len(names) == 31 and not R.verify_receivers(tria, names, xyz)
        assert not any(f.endswith(".gdat") for f in os.listdir(mesh_dir))      # no data written

        # 04: new stations kept, files written
        refined = os.path.join(work, "04_refine")
        names, _xyz = R.read_receivers(os.path.join(refined, "receivers_refined-projected.csv"))
        assert len(names) == 40 and "40 stations checked" in _text(out["04_refine_inverted_model"])

        # 05-06
        figs = os.listdir(os.path.join(work, "05_figures"))
        assert {"plan_1000m.png", "section_A.png", "section_C.png", "section_map.png"} <= set(figs)
        assert len([f for f in os.listdir(os.path.join(work, "06_geotiff")) if f.endswith(".tif")]) == 7

        # the mesh notebook with the other ocean modes
        # (real ocean, and FEMTIC and SimPEG output of the same mesh)
        nb = _run(nb_dir, "03_mesh_and_receivers",
                  {"OCEAN = ": 'OCEAN = "real"', "FEMTIC = ": "FEMTIC = True", "SIMPEG = ": "SIMPEG = True",
                   "SIMPEG_VERTICAL_RATIO = ": "SIMPEG_VERTICAL_RATIO = 4.0"}, log=log)
        assert "ocean (real):" in _text(nb) and "FEMTIC: wrote mesh.dat" in _text(nb)
        from mtinv_gui.backends import femtic as F
        fm = F.read_mesh_dat(os.path.join(mesh_dir, "mesh.dat"))
        assert fm.n_elements == R.load_tria(os.path.join(mesh_dir, "model.tria")).n_active_cells()
        assert F.check_mesh(fm) == [] and len(fm.surface)
        from mtinv_gui.backends import simpeg_mesh as S
        assert "SimPEG: wrote octree_mesh.txt" in _text(nb)
        tree, arrays = S.read_model(mesh_dir)
        assert tree.n_cells == len(arrays["material"]) and len(arrays["receiver_names"]) == 31
        assert (arrays["material"] == M.OCEAN_ID).any() and arrays["active"].any()
        # each station on the octree's earth surface, within a cell of the DEM
        loc = arrays["receiver_locations"]
        below = tree.get_containing_cells(loc - [0, 0, 1e-3])
        assert (arrays["material"][below] == M.EARTH_ID).all()
        log(f"  SimPEG: {tree.n_cells} cells; station minus DEM elevation up to "
            f"{np.abs(loc[:, 2] - dem_at(mesh_dir, loc)).max():.0f} m")
        nb = _run(nb_dir, "03_mesh_and_receivers", {"OCEAN = ": 'OCEAN = "none"'}, log=log)
        assert "DEM ocean" not in _text(nb) and "\n2 materials" in "\n" + _text(nb)


if __name__ == "__main__":
    quick = "quick" in sys.argv[1:]
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            if quick and name == "test_notebooks_run":
                continue
            try:
                fn()
                print("ok   ", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL ", name, "->", repr(exc))
    sys.exit(1 if failed else 0)
