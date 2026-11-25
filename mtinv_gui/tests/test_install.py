"""Is this environment set up the way the GUIs and GoFEM need?

    python tests/test_install.py

Run it after creating the environment.  The deal.II check matters most: GoFEM
installations built against deal.II 9.3 can't read meshes from newer versions.
"""
from __future__ import annotations

import glob
import inspect
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtinv_gui import config  # noqa: E402

config.apply_sys_paths()


def test_dealii_is_93():
    found = glob.glob(os.path.join(sys.prefix, "conda-meta", "dealii-*.json"))
    assert found, f"no conda deal.II package in {sys.prefix}"
    version = os.path.basename(found[0]).split("-")[1]
    assert version.startswith("9.3."), f"deal.II {version}: meshes may not be readable by GoFEM"


def test_dealii_writes_a_mesh():
    import PyDealII.Release as dealii

    tria = dealii.Triangulation("3D")
    tria.generate_hyper_cube()
    tria.refine_global(1)
    assert tria.n_active_cells() == 8
    d = tempfile.mkdtemp()
    try:
        path = os.path.join(d, "cube.tria")
        tria.save(path)
        assert os.path.getsize(path) > 0
    finally:
        shutil.rmtree(d)


def test_pygofem_is_the_fork():
    import gofem.mesh_utils
    import gofem.parameter_handler  # noqa: F401

    params = inspect.signature(gofem.mesh_utils.refine_within_polygon).parameters
    assert "pnorm" in params, "upstream pyGoFEM: install the fork named in environment.yml"


def test_mtpy_1_for_gofem_files():
    assert config.mtpy_available(), "mtpy 1.x is needed to interpolate EDIs and write GoFEM data"
    from gofem.mtpy_utils import write_edi_collection_to_gofem  # noqa: F401
    from mtpy.core.edi_collection import EdiCollection  # noqa: F401


def test_plotting_stack():
    import geopandas  # noqa: F401
    import pyogrio  # noqa: F401
    import pyvista  # noqa: F401
    import pyvistaqt  # noqa: F401
    import rasterio  # noqa: F401
    from PyQt5 import QtWidgets  # noqa: F401


def test_discretize_for_simpeg_output():
    """The SimPEG output builds a discretize TreeMesh (pip, pinned in environment.yml)."""
    import discretize
    assert discretize.__version__.startswith("0.11"), discretize.__version__


def test_notebooks_can_run():
    from jupyter_client.kernelspec import KernelSpecManager
    import jupyterlab  # noqa: F401
    import nbclient  # noqa: F401

    spec = KernelSpecManager().get_kernel_spec("python3")
    assert os.path.realpath(spec.argv[0]).startswith(os.path.realpath(sys.prefix)) \
        or spec.argv[0] in ("python", "python3"), f"the notebooks' kernel isn't this environment: {spec.argv}"


def test_map_projections_have_their_data():
    import pyproj
    import rasterio.crs

    east, north = pyproj.Transformer.from_crs(4326, 3979, always_xy=True).transform(-95.0, 49.0)
    assert abs(east) < 1e7 and abs(north) < 1e7
    assert rasterio.crs.CRS.from_epsg(3979).is_projected


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name} -> {exc!r}")
    sys.exit(1 if failed else 0)
