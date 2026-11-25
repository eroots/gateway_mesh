"""FEMTIC output (``backends/femtic.py``): the built deal.II mesh as a DHEXA
``mesh.dat`` + ``resistivity_block_iter0.dat``.

    python tests/test_femtic.py

``check_mesh`` repeats the checks FEMTIC makes when it reads a mesh, and the ones its
solver relies on without checking (neighbour order, no chained hanging edges).  Set
``FEMTIC_EXAMPLE_MESH`` to a makeDHexaMesh ``mesh.dat`` (e.g. FEMTIC's
``examples/INV_MT_01/DHexa/mesh.dat``) to also check the reader and the checks on it.
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
from mtinv_gui.backends import femtic_data as FD  # noqa: E402
from mtinv_gui.backends import mesh as M  # noqa: E402
from mtinv_gui.backends import mesh_dem as MD  # noqa: E402
from test_mesh_dem import QUIET, grid  # noqa: E402


def shore(east, north):
    """Sea 150 m deep in the west, a shore at -30 km, land rising to 400 m to the east and
    a dry basin 80 m below sea level at 20 km east, all with gentle (~10 km) slopes."""
    land = 50.0 + 350.0 * np.clip((east + 30e3) / 90e3, 0, 1)
    z = -150.0 + (land + 150.0) / (1 + np.exp(-(east + 30e3) / 6e3))
    basin = 1 / (1 + np.exp((np.hypot(east - 20e3, north) - 10e3) / 6e3))
    return z + (-80.0 - z) * basin


def cliff(east, north):
    """The basin of test_mesh_dem: a 380 m step down at its edge."""
    return np.where(np.hypot(east - 20e3, north) < 15e3, -200.0, 300.0) + 0 * north


def _build(ocean_mode="real", balance=False, topo=shore):
    """A small mesh over the shore DEM through the same steps as mesh.run."""
    M.prepare_pygofem()
    from gofem.mesh_utils import refine_around_points, refine_at_interface

    from mtinv_gui.backends import refine as R

    dem = MD.MeshDem([grid(80e3, 1e3, topo)])
    dx, dy, dz = M.cell_size_lists([10e3] * 6, [100.0, 200.0, 1e3, 4e3, 20e3, 40e3],
                                   [100.0, 200.0, 1e3, 4e3, 20e3, 40e3])
    tria = M.base_grid(dx, dy, dz, log=QUIET)
    materials = M.initial_materials(tria, 100.0, log=QUIET)
    refine_at_interface(tria, material_ids=[M.EARTH_ID], repeat=1, center=[0, 0, 0],
                        radii=[1e7, 1e7], pnorm=1)
    if ocean_mode == "real":
        M.refine_air_over_ocean(tria, dem.ocean_elevation, M.AIR_ID, [M.EARTH_ID])
    # two levels more around one point: level jumps along edges and corners
    refine_around_points(tria, points=[[5e3, 5e3, 0.0]], center=[0, 0, 0],
                         radii=[8e3, 8e3, 3e3], repeat=2, exclude_materials=[M.AIR_ID])
    if balance:
        R.balance_levels(tria, log=QUIET)
    M.fit_topography(tria, dem, land=True, bathymetry=(ocean_mode == "real"), log=QUIET)
    M.add_ocean(tria, ocean_mode, dem, bounds=M.model_bounds(dx, dy), dz_earth=dz[len(dz) // 2:],
                materials=materials, background_resistivity_ohm_m=100.0, log=QUIET)
    return tria, materials


def _convert(tria):
    xyz, vid, lv, mat = F.tria_arrays(tria)
    m = F.build_mesh(xyz, vid, lv, mat, surface_below_ids={M.AIR_ID, M.OCEAN_ID},
                     earth_top_ids={M.EARTH_ID})
    return m, mat


def test_a_built_mesh_passes_femtics_checks_and_round_trips():
    tria, materials = _build("real")
    with tempfile.TemporaryDirectory() as d:
        paths = F.write_mesh(tria, materials, d, air_id=M.AIR_ID, free_ids=[M.EARTH_ID],
                             sea_ids=(M.OCEAN_ID,), log=QUIET)
        m = F.read_mesh_dat(paths["femtic_mesh"])
        block, rho, fixed = F.read_blocks_dat(paths["femtic_blocks"])
    assert F.check_mesh(m) == []
    ref, mat = _convert(tria)
    assert m.n_elements == tria.n_active_cells()
    assert np.allclose(m.nodes, ref.nodes, atol=1e-5)
    assert (m.elements == ref.elements).all() and m.neighbours == ref.neighbours
    assert sum(len(nb) == 4 for row in m.neighbours for nb in row) > 0    # hanging faces exist
    # blocks: 0 = air, then the sea, then one per earth cell
    assert fixed[0] and abs(rho[0] - 1e8) < 1 and (block[mat == M.AIR_ID] == 0).all()
    sea = np.unique(block[mat == M.OCEAN_ID])
    assert len(sea) == 1 and fixed[sea[0]] and abs(rho[sea[0]] - materials[M.OCEAN_ID][1] ** -1) < 1e-6
    earth = block[mat == M.EARTH_ID]
    assert len(np.unique(earth)) == len(earth) and not fixed[earth].any()
    assert np.allclose(rho[earth], 100.0)


def test_the_earth_surface_is_land_under_air_and_the_seafloor():
    tria, _materials = _build("real")
    m, mat = _convert(tria)
    top = m.surface[:, 0]
    assert (m.surface[:, 1] == 4).all() and (mat[top] == M.EARTH_ID).all()
    above = [mat[m.neighbours[e][4]] for e in top]
    assert any((a == M.AIR_ID).any() for a in above)
    assert any((a == M.OCEAN_ID).any() for a in above)
    # every land cell under air / sea is in the list, and nothing else
    expect = {e for e in np.flatnonzero(mat == M.EARTH_ID)
              if np.isin(mat[m.neighbours[e][4]], (M.AIR_ID, M.OCEAN_ID)).any()}
    assert set(top.tolist()) == expect


def test_side_faces_stay_vertical_after_the_topography_fit():
    tria, _materials = _build("none")
    m, _mat = _convert(tria)
    assert F.check_mesh(m) == [] and len(F.folded_elements(m)) == 0
    assert np.ptp(m.nodes[m.elements[:, :4], 2], axis=1).max() > 1.0     # the cells are deformed


def test_cells_folded_by_a_dem_step_are_a_warning():
    tria, materials = _build("none", topo=cliff)
    m, _mat = _convert(tria)
    assert F.check_mesh(m) == [] and len(F.folded_elements(m)) > 0
    lines = []
    with tempfile.TemporaryDirectory() as d:
        paths = F.write_mesh(tria, materials, d, air_id=M.AIR_ID, free_ids=[M.EARTH_ID],
                             log=lines.append)
        assert os.path.isfile(paths["femtic_mesh"])
    assert paths["femtic_folded"] == len(F.folded_elements(m))
    assert any(line.startswith("WARNING") and "folded" in line for line in lines)


def test_corner_level_jumps_need_no_balancing():
    """deal.II keeps cells sharing an edge within one level, so its jumps are at corners
    only and FEMTIC can take the mesh as it is."""
    from mtinv_gui.backends import refine as R

    tria, _materials = _build("none", balance=False)
    assert (R.level_jumps(tria) >= 2).any()
    assert F.check_mesh(_convert(tria)[0]) == []


def _boxes_mesh(boxes, links):
    """A FemticMesh from boxes (x0, x1, y0, y1, z0, z1) and {(element, face): [neighbours]}."""
    nodes, elements = {}, []
    for x0, x1, y0, y1, z0, z1 in boxes:
        corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                   (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        elements.append([nodes.setdefault(c, len(nodes)) for c in corners])
    neighbours = [[list(links.get((e, f), [])) for f in range(6)] for e in range(len(boxes))]
    return F.FemticMesh(list(nodes), elements, neighbours, [[(0, 0)]] * 6, [(0, 4)])


def test_a_chained_hanging_edge_is_caught():
    """A (2 m cube) has four 1 m cells on its x+ face; the lower one's y- face, beyond A,
    has four 0.5 m cells, so its edge on A's face hangs twice."""
    boxes = [(0, 2, 0, 2, 0, 2)]                                         # A
    boxes += [(2, 3, y, y + 1, z, z + 1) for z in (0, 1) for y in (0, 1)]  # B, order (y, z)
    boxes += [(x, x + .5, -.5, 0, z, z + .5) for z in (0, .5) for x in (2, 2.5)]  # C, (x, z)
    links = {(0, 1): [1, 2, 3, 4], (1, 2): [5, 6, 7, 8]}
    links.update({(b, 0): [0] for b in (1, 2, 3, 4)})
    links.update({(c, 3): [1] for c in (5, 6, 7, 8)})
    problems = F.check_mesh(_boxes_mesh(boxes, links))
    assert problems and all("coarser face" in p for p in problems), problems
    # without the C cells it's fine
    fine = {k: v for k, v in links.items() if k != (1, 2) and k[0] < 5}
    assert F.check_mesh(_boxes_mesh(boxes[:5], fine)) == []
    # neighbours listed out of order are caught too
    swapped = {**fine, (0, 1): [2, 1, 3, 4]}
    assert any("wrong order" in p for p in F.check_mesh(_boxes_mesh(boxes[:5], swapped)))


def test_ocean_models_per_format():
    M.output_format({"output_format": "femtic", "ocean_mode": "real"})
    M.output_format({"ocean_mode": "thin_sheet"})                       # GoFEM is the default
    try:
        M.output_format({"output_format": "femtic", "ocean_mode": "thin_sheet"})
    except ValueError as exc:
        assert "thin_sheet" in str(exc)
    else:
        raise AssertionError("FEMTIC took a thin-sheet ocean")


# ---- observe.dat / control.dat ---------------------------------------------------- #
MU0 = 4e-7 * np.pi


def _half_space_site(name="A", north=1e3, east=-2e3, rho=100.0, freqs=(1.0, 0.1, 0.01)):
    """A 1-D site as an EDI holds it: mV/km/nT, e^(+iwt), 2 % errors, a small tipper."""
    f = np.asarray(freqs, dtype="float64")
    z_si = np.sqrt(2 * np.pi * f * MU0 * rho) * np.exp(1j * np.pi / 4)    # e^(+iwt): 45 deg
    z = np.zeros((len(f), 2, 2), complex)
    z[:, 0, 1], z[:, 1, 0] = z_si, -z_si
    z /= FD.Z_TO_SI
    ze = 0.02 * np.abs(z).max(axis=(1, 2))[:, None, None] * np.ones((1, 2, 2))
    t = np.tile([0.1 + 0.05j, -0.2 + 0.01j], (len(f), 1))
    return FD.SiteData(name, north, east, f, z, ze, t, np.full((len(f), 2), 0.01))


def test_observe_is_in_si_units_with_femtics_time_convention():
    site = _half_space_site()
    with tempfile.TemporaryDirectory() as d:
        FD.write_observe([site], os.path.join(d, "observe.dat"), "Z_Tipper", 0.05, log=QUIET)
        obs = FD.read_observe(os.path.join(d, "observe.dat"))
        csv = open(os.path.join(d, FD.SITES_CSV)).read().splitlines()
    (sid, x, y, mt), = obs["MT"]
    assert sid == 1 and (x, y) == (1.0, -2.0)                               # km, x = north
    zxy = mt[:, 3] + 1j * mt[:, 4]
    zyx = mt[:, 5] + 1j * mt[:, 6]
    # FEMTIC's examples: Zxy in the 4th quadrant, Zyx in the 2nd
    assert (zxy.real > 0).all() and (zxy.imag < 0).all() and (zyx.real < 0).all() and (zyx.imag > 0).all()
    rho_a = np.abs(zxy) ** 2 / (2 * np.pi * mt[:, 0] * MU0)
    assert np.allclose(rho_a, 100.0, rtol=1e-5)
    # the floor: 5 % of |Zxy| (> the 2 % error) on the x row, re and im alike
    assert np.allclose(mt[:, 11:13], 0.05 * np.abs(zxy)[:, None], rtol=1e-5)
    # diagonals are 0 in the file: GoFEM drops them, here their errors go negative
    assert (mt[:, [1, 2, 7, 8]] == 0).all() and (mt[:, [9, 10, 15, 16]] == -1).all()
    (vid, _x, _y, vtf), = obs["VTF"]
    assert vid == 101                                                       # ids unique across blocks
    assert np.allclose(vtf[0, 1:5], [0.1, -0.05, -0.2, -0.01]) and np.allclose(vtf[0, 5:], 0.05)
    assert csv[1:] == ["1,MT,A,1000.000,-2000.000", "101,VTF,A,1000.000,-2000.000"]


def test_phase_tensor_matches_gofem_and_is_minus_identity_in_1d():
    M.prepare_pygofem()
    from gofem.data_utils import calculate_phase_tensor

    site = _half_space_site()
    rows = FD.phase_tensor_rows(site, 0.05)
    assert np.allclose([r[1] for r in rows], [-1, 0, 0, -1], atol=1e-9)    # FEMTIC's INV_PT_01: ~ -I
    rng = np.random.default_rng(1)
    for _ in range(5):
        z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        dz = np.abs(rng.normal(size=(2, 2))) * 0.1
        p, dp = FD.phase_tensor(z, dz)
        p2, dp2 = calculate_phase_tensor(z, dz)
        assert np.allclose(p, p2) and np.allclose(dp, dp2, rtol=1e-4)


def test_missing_values_and_stations_without_tipper():
    a = _half_space_site("A")
    a.z[1] = np.nan                                       # a frequency missing at A
    b = _half_space_site("B", north=0, east=0)
    b.tipper[:] = 0                                       # B has no tipper
    lines = []
    with tempfile.TemporaryDirectory() as d:
        res = FD.write_observe([a, b], os.path.join(d, "observe.dat"), "Z_Tipper", 0.05, log=lines.append)
        obs = FD.read_observe(res["observe"])
    assert res["counts"] == {"MT": 2, "VTF": 1}
    assert len(obs["MT"][0][3]) == 2 and len(obs["MT"][1][3]) == 3          # A's missing line dropped
    assert [s[0] for s in obs["VTF"]] == [101]
    assert any("no usable VTF" in line for line in lines)
    try:
        FD.write_observe([b], os.path.join(tempfile.gettempdir(), "never.dat"), "Tipper", log=QUIET)
    except ValueError:
        pass
    else:
        raise AssertionError("wrote an observe.dat with no data")


def test_sites_from_a_collection_share_the_run_frequencies():
    from types import SimpleNamespace as NS

    def obj(name, freqs):
        f = np.asarray(freqs, dtype="float64")
        z = np.ones((len(f), 2, 2), complex) * f[:, None, None]
        return NS(station=name, Z=NS(freq=f, z=z, z_err=0.1 * np.abs(z)),
                  Tipper=NS(tipper=np.zeros((len(f), 1, 2), complex), tipper_err=np.zeros((len(f), 1, 2))))

    coll = NS(mt_obj_list=[obj("S2", [10.2, 1.0, 0.1]), obj("S1", [9.8, 0.5]), obj("X", [1.0])])
    lines = []
    sites = FD.sites_from_collection(coll, ["S1", "S2", "S3"], [[1, 2, 0], [3, 4, 0], [5, 6, 0]],
                                     [0.1, 1.0, 10.0], log=lines.append)
    assert [s.name for s in sites] == ["S1", "S2"]                          # receiver order
    assert list(sites[0].freq) == [10.0, 1.0, 0.1]                          # the run's, not the EDI's
    assert sites[0].z[0, 0, 0] == 9.8 and np.isnan(sites[0].z[1:]).all()    # 0.5 Hz isn't a run frequency
    assert sites[1].z[0, 0, 0] == 10.2 and sites[1].z[2, 0, 0] == 0.1
    assert (sites[1].north_m, sites[1].east_m) == (3.0, 4.0)
    assert any("X" in line for line in lines) and any("S3" in line for line in lines)


def test_control_dat():
    text = FD.control_text(iterations=0, n_threads=8, trade_off=3.0, extra={"MEM_LIMIT": "8000"})
    lines = text.split()
    assert lines[-1] == "END"
    def after(key):
        return lines[lines.index(key) + 1]
    assert after("MESH_TYPE") == "2" and after("NUM_THREADS") == "8" and after("ITERATION") == "0"
    assert lines[lines.index("ITERATION") + 2] == "0" and after("DISTORTION") == "0"
    assert after("TRADE_OFF_PARAM") == "3" and after("MEM_LIMIT") == "8000"
    # every keyword once (FEMTIC stops on a repeat)
    keys = [w for w in lines if w.isupper() and w != "END"]
    assert len(keys) == len(set(keys))


def test_observe_from_the_sample_edis():
    """The adapter on real mtpy objects: the notebooks' sample EDIs."""
    from mtinv_gui.backends import edi_io, gofem_files

    edis = os.path.join(ROOT, "notebooks", "sample_data", "edis")
    if not os.path.isdir(edis):
        print("     (skipped: no sample EDIs)")
        return
    coll = edi_io.load_collection(edis)
    names = [o.station for o in coll.mt_obj_list]
    xyz = [[1e3 * i, -1e3 * i, 0.0] for i in range(len(names))]
    periods = gofem_files.select_periods(coll, 50, 1e-3, 1e4)
    sites = FD.sites_from_collection(coll, names, xyz, [1 / p for p in periods], log=QUIET)
    with tempfile.TemporaryDirectory() as d:
        res = FD.write_observe(sites, os.path.join(d, "observe.dat"), "Z_Tipper", 0.05, log=QUIET)
        obs = FD.read_observe(res["observe"])
    assert res["counts"] == {"MT": len(names), "VTF": len(names)}
    freqs = {round(r[0], 12) for _sid, _x, _y, arr in obs["MT"] for r in arr}
    assert len(freqs) <= len(periods)                    # FEMTIC solves only the run's frequencies
    # the same numbers as an EDI-side calculation, conjugated, in V/A
    o = coll.mt_obj_list[0]
    sid, _x, _y, arr = obs["MT"][0]
    k = np.argmin(abs(o.Z.freq - arr[0, 0]))
    assert np.allclose(arr[0, 3:5], [o.Z.z[k, 0, 1].real * FD.Z_TO_SI, -o.Z.z[k, 0, 1].imag * FD.Z_TO_SI],
                       rtol=1e-5)


def test_a_makedhexamesh_mesh_reads_and_passes():
    path = os.environ.get("FEMTIC_EXAMPLE_MESH", "")
    if not os.path.isfile(path):
        print("     (skipped: set FEMTIC_EXAMPLE_MESH to check a makeDHexaMesh mesh.dat)")
        return
    m = F.read_mesh_dat(path)
    assert F.check_mesh(m) == []
    with tempfile.TemporaryDirectory() as d:
        again = F.read_mesh_dat(F.write_mesh_dat(m, os.path.join(d, "mesh.dat")))
    assert np.allclose(again.nodes, m.nodes) and again.neighbours == m.neighbours
    assert all((a == b).all() for a, b in zip(again.planes, m.planes))
    assert (again.surface == m.surface).all()


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
