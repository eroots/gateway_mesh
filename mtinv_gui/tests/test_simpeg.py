"""SimPEG output (``backends/simpeg_mesh.py``): the fitted GoFEM mesh design as a
discretize TreeMesh, with earth / air / sea cells from the DEM.

    python tests/test_simpeg.py

If SimPEG itself is importable (it isn't part of the mtinv environment), a forward run
on a half-space octree checks the files work in SimPEG: about 100 ohm-m from a 1000 ohm-m
primary field.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from mtinv_gui.backends import femtic as F  # noqa: E402
from mtinv_gui.backends import mesh as M  # noqa: E402
from mtinv_gui.backends import mesh_dem as MD  # noqa: E402
from mtinv_gui.backends import simpeg_mesh as S  # noqa: E402
from test_femtic import _build, shore  # noqa: E402
from test_mesh_dem import QUIET, grid  # noqa: E402

IDS = dict(air=M.AIR_ID, earth=M.EARTH_ID, sea=M.OCEAN_ID, mantle=M.MANTLE_ID)
STATIONS = {"SEA": [0.0, -50e3], "SHORE": [0.0, -20e3], "LAND": [0.0, 5e3], "BASIN": [0.0, 20e3]}


def _shore_dem():
    return MD.MeshDem([grid(80e3, 1e3, shore)])


def _write(d, *, ocean="real", ratio=32.0, mantle=None):
    tria, materials = _build(ocean)
    dem = _shore_dem()
    rec = [[n, e, 0.0] for n, e in STATIONS.values()]
    res = S.write_mesh(tria, materials, d, dem=dem, ids=IDS, free_ids=[M.EARTH_ID],
                       land=True, bathymetry=(ocean == "real"), mantle_depth_m=mantle,
                       names=list(STATIONS), receivers=rec, vertical_ratio=ratio, log=QUIET)
    return tria, materials, dem, res


def test_axes_cover_the_mesh_centred_on_the_survey_and_sea_level():
    bounds = (-60e3, 60e3, -50e3, 50e3, -65e3, 80e3)                  # 65 km of air, 80 km deep
    hs, ns, origin = S.tree_axes(bounds, 1250.0, 1000.0, vertical_ratio=4)
    hs, ns, origin = np.array(hs), np.array(ns), np.array(origin)
    assert all(n & (n - 1) == 0 for n in ns)                           # powers of 2
    assert np.allclose(np.round(hs * 1000), hs * 1000)                 # whole mm (the UBC file)
    assert hs[0] <= 1000.001 and hs[1] <= 1250.001 and np.isclose(hs[2], min(hs[:2]) / 4, atol=1e-3)
    # the centre (0, 0, sea level) is the middle of the octree: a boundary at every level
    assert np.allclose(origin + ns * hs / 2, 0.0, atol=1e-6)
    lo, hi = origin, origin + ns * hs
    assert lo[0] <= -50e3 and hi[0] >= 50e3 and lo[1] <= -60e3 and hi[1] >= 60e3
    assert lo[2] <= -80e3 and hi[2] >= 65e3


def test_sea_level_is_a_cell_boundary():
    tria, materials = _build("none", topo=lambda e, n: 0 * e)
    with tempfile.TemporaryDirectory() as d:
        S.write_mesh(tria, materials, d, dem=None, ids=IDS, free_ids=[M.EARTH_ID], log=QUIET,
                     vertical_ratio=3)
        mesh, _a = S.read_model(d)
    top = mesh.cell_centers[:, 2] + mesh.h_gridded[:, 2] / 2
    bottom = top - mesh.h_gridded[:, 2]
    assert not ((bottom < -1e-6) & (top > 1e-6)).any()


def test_every_gofem_cell_gets_octree_cells_at_least_as_fine():
    tria, _materials = _build("real")
    M.fit_topography  # (the build above already fitted it)
    xyz, vid, _lv, _mat = F.tria_arrays(tria)
    centre, size = S.cell_boxes(xyz, vid)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    mesh = S.build_tree(centre, size, (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]), vertical_ratio=2)
    cells = mesh.get_containing_cells(np.c_[centre[:, 1], centre[:, 0], -centre[:, 2]])
    h = mesh.h_gridded[cells]
    assert (h[:, 0] <= size[:, 1] + 1e-6).all() and (h[:, 1] <= size[:, 0] + 1e-6).all()
    assert np.allclose(h[:, 0] / h[:, 2], h[0, 0] / h[0, 2])           # one shape everywhere
    assert mesh.n_cells < 5 * tria.n_active_cells()


def test_cells_are_earth_air_and_sea_by_the_dem():
    with tempfile.TemporaryDirectory() as d:
        _tria, materials, dem, _res = _write(d)
        mesh, a = S.read_model(d)
    cc, mat = mesh.cell_centers, a["material"]
    surface, sea = S.surface_elevation(dem, cc[:, 1], cc[:, 0], land=True, bathymetry=True)
    assert ((mat == M.EARTH_ID) == (cc[:, 2] < surface)).all()
    water = mat == M.OCEAN_ID
    assert water.any() and sea[water].all() and (cc[water, 2] < 0).all()
    # the dry basin (below sea level, inland) gets no water
    inland = dem.inland(np.c_[cc[:, 1], cc[:, 0]])
    assert inland.any() and not (water & inland).any()
    # conductivities from the materials, earth cells the active (inverted) ones
    sig = {k: v[1] for k, v in materials.items()}
    assert np.allclose(a["conductivity"][mat == M.EARTH_ID], sig[M.EARTH_ID])
    assert np.allclose(a["conductivity"][water], sig[M.OCEAN_ID])
    assert (a["active"] == (mat == M.EARTH_ID)).all()


def test_files_read_back_in_the_same_cell_order():
    with tempfile.TemporaryDirectory() as d:
        _write(d)
        mesh, a = S.read_model(d)
        for key, name in S.MODEL_FILES.items():
            ubc = mesh.read_model_UBC(os.path.join(d, name))
            assert np.allclose(ubc, np.asarray(a[key], dtype=float)), key
        csv = open(os.path.join(d, S.RECEIVERS_CSV)).read().splitlines()
    assert csv[0] == "name,x_east_m,y_north_m,z_m" and len(csv) == 1 + len(STATIONS)


def test_receivers_sit_on_the_discretised_surface():
    with tempfile.TemporaryDirectory() as d:
        _write(d)
        mesh, a = S.read_model(d)
    loc = a["receiver_locations"]
    assert list(a["receiver_names"]) == list(STATIONS)
    assert np.allclose(loc[:, :2], [[e, n] for n, e in STATIONS.values()])
    solid = mesh.get_containing_cells(loc - [0, 0, 1e-3])
    air = mesh.get_containing_cells(loc + [0, 0, 1e-3])
    assert (a["material"][solid] == M.EARTH_ID).all()                   # earth just below
    assert (a["material"][air] != M.EARTH_ID).all()                     # air or sea just above
    # the sea station is on the seafloor, under water
    sea = list(STATIONS).index("SEA")
    assert a["material"][air[sea]] == M.OCEAN_ID and loc[sea, 2] < 0


def test_a_fixed_mantle_is_not_inverted():
    with tempfile.TemporaryDirectory() as d:
        tria, _m = _build("none")
        materials = {M.AIR_ID: ("air", 1e-8), M.EARTH_ID: ("ground", 0.01), M.MANTLE_ID: ("mantle", 0.05)}
        S.write_mesh(tria, materials, d, dem=_shore_dem(), ids=IDS, free_ids=[M.EARTH_ID],
                     mantle_depth_m=30e3, names=["LAND"], receivers=[[0, 5e3, 0]], log=QUIET)
        mesh, a = S.read_model(d)
    deep = -mesh.cell_centers[:, 2] > 30e3
    assert (a["material"][deep] == M.MANTLE_ID).all() and not a["active"][deep].any()
    assert not (a["material"] == M.OCEAN_ID).any()                     # no ocean model: no water


def test_the_gui_build_writes_the_octree():
    """mesh.run with 'Mesh For' = SimPEG: the repository's synthetic stations, flat, a small
    grid and one survey-area stage."""
    from dataclasses import asdict

    from mtinv_gui.backends import edi_io
    from mtinv_gui.backends.mesh_design import RefinementStage
    from mtinv_gui.project import Project

    p = Project()
    p.survey.edi_dir = os.path.join(os.path.dirname(ROOT), "edis")
    stations = edi_io.read_edi_headers(p.survey.edi_dir)
    p.projection.proj_kind, p.projection.local_proj = "local", "tmerc"
    p.projection.origin_lon = float(np.mean([st.lon for st in stations]))
    p.projection.origin_lat = float(np.mean([st.lat for st in stations]))
    p.mesh.dx_blocks = [20e3] * 4 + [60e3, 120e3]
    p.mesh.dz_air = [500.0, 2e3, 10e3, 50e3, 150e3]
    p.mesh.dz_earth = [500.0, 2e3, 10e3, 50e3, 150e3]
    p.mesh.refinement_stages = [asdict(RefinementStage(kind="survey_area", levels=2, buffer_km=10.0,
                                                       max_depth_km=20.0))]
    p.mesh.output_format = "simpeg"
    p.mesh.simpeg_vertical_ratio = 2.0
    spec = M.mesh_spec_from_project(p)
    with tempfile.TemporaryDirectory() as d:
        res = M.run(spec, d, log=QUIET)
        mesh, a = S.read_model(d)
        assert os.path.isfile(res["tria"]) and res["simpeg_mesh"].endswith(S.MESH_TXT)
    assert mesh.n_cells == res["simpeg_cells"] == len(a["material"])
    assert len(a["receiver_names"]) == len(stations)
    assert np.allclose(a["receiver_locations"][:, 2], 0.0)                 # flat: at sea level
    h = mesh.h_gridded
    assert np.isclose(h[:, 0].min() / h[:, 2].min(), 2.0, rtol=1e-4)
    assert h[:, 0].min() <= 20e3                                            # the base cells' width


def test_thin_sheet_is_refused():
    try:
        M.output_format({"output_format": "simpeg", "ocean_mode": "thin_sheet"})
    except ValueError:
        pass
    else:
        raise AssertionError("SimPEG took a thin-sheet ocean")


def test_simpeg_runs_on_the_octree():
    try:
        from simpeg import maps
        from simpeg.electromagnetics import natural_source as nsem
    except ImportError:
        print("     (skipped: SimPEG isn't installed here)")
        return
    with tempfile.TemporaryDirectory() as d:
        tria, materials = _build("none", topo=lambda e, n: 0 * e)
        S.write_mesh(tria, materials, d, dem=None, ids=IDS, free_ids=[M.EARTH_ID],
                     names=["C"], receivers=[[5e3, 5e3, 0]], vertical_ratio=4, log=QUIET)
        mesh, a = S.read_model(d)
    sigma, active = a["conductivity"], a["active"].astype(bool)
    rxs = [nsem.receivers.Impedance(a["receiver_locations"], orientation="yx", component=c)
           for c in ("apparent_resistivity", "phase")]
    survey = nsem.Survey([nsem.sources.PlanewaveXYPrimary(rxs, frequency=f) for f in (1.0, 0.1)])
    sim = nsem.simulation.Simulation3DPrimarySecondary(
        mesh, survey=survey, sigmaPrimary=np.where(sigma > 1e-6, 1e-3, sigma),
        sigmaMap=maps.InjectActiveCells(mesh, active, sigma[~active]) * maps.ExpMap(nP=int(active.sum())))
    rho, phase = sim.dpred(np.log(sigma[active])).reshape(2, 2).T
    assert np.allclose(rho, 100, rtol=0.05) and np.allclose(phase, 45, atol=3), (rho, phase)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print("FAIL ", name, "->", repr(exc))
    sys.exit(1 if failed else 0)
