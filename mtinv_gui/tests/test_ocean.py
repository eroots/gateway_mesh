"""Ocean / coastline logic (``backends/ocean.py`` + the cell helpers in ``mesh.py``).

No deal.II needed: the mesh helpers are exercised with stand-in cell objects that
have the handful of accessors the real PyDealII cells provide.

Runs standalone (``python tests/test_ocean.py``) or under pytest.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtinv_gui.backends import mesh as M  # noqa: E402
from mtinv_gui.backends import mesh_design as MD  # noqa: E402
from mtinv_gui.backends import ocean as O  # noqa: E402

SIGMA_BG = 1.0 / 150.0
DZ = [1000.0, 1250.0, 1500.0, 2000.0]


def _dem(n=201, half=100e3):
    """Land west of a wobbly coast, sea east of it, one tiny and one big island."""
    xs = np.linspace(-half, half, n)
    X, Y = np.meshgrid(xs, xs)
    coast = 20e3 * np.sin(Y / 15e3) + 2e3 * np.sin(Y / 1.5e3) + 5e3
    Z = np.where(X < coast, 300.0, -50.0 - (X - coast) * 0.05)
    Z[(abs(X - 50e3) < 2e3) & (abs(Y - 60e3) < 2e3)] = 100.0       # 16 km^2 island
    Z[(abs(X - 50e3) < 10e3) & (abs(Y + 50e3) < 10e3)] = 100.0     # 400 km^2 island
    tc = np.zeros((n, n, 3))
    tc[:, :, 0], tc[:, :, 1], tc[:, :, 2] = X, Y, Z
    return tc


# -- coastline ---------------------------------------------------------------
def test_simplify_drops_small_wiggles():
    pts = np.array([[0, 0], [1, 0.1], [2, -0.1], [3, 0.0], [4, 0]], dtype=float)
    out = O.simplify_polyline(pts, 0.5)
    assert out.tolist() == [[0, 0], [4, 0]]
    assert len(O.simplify_polyline(pts, 0.0)) == len(pts)


def test_closed_ring_stays_closed():
    t = np.linspace(0, 2 * np.pi, 200)
    ring = np.c_[np.cos(t), np.sin(t)] * 1000.0
    ring[-1] = ring[0]
    out = O.simplify_polyline(ring, 20.0, closed=True)
    assert np.allclose(out[0], out[-1]) and 4 <= len(out) < 100


def test_small_islands_are_dropped():
    tc = _dem()
    _, raw = O.process_coastlines(tc, 0.0, 0.0)
    _, kept = O.process_coastlines(tc, 1500.0, 100e6)         # 100 km^2
    assert raw["n_lines"] == 3 and kept["n_lines"] == 2
    assert kept["n_points"] < raw["n_points"] / 5


def test_resample_spacing_and_cap():
    lines, _ = O.process_coastlines(_dem(), 1500.0, 100e6)
    pts, info = O.coastline_points(lines, 10e3)
    assert not info["spacing_widened"]
    assert 30 <= len(pts) <= 45                               # ~370 km of coast / 10 km
    pts2, info2 = O.coastline_points(lines, 10e3, max_points=10)
    assert info2["spacing_widened"] and len(pts2) <= 14


def test_stage_points_are_in_mesh_frame():
    tc = _dem()
    pts, _ = O.stage_coast_points(tc, {"radius_km": 10, "simplify_km": 1.5, "min_area_km2": 100})
    # east of the model origin is +y in the mesh frame; the main coast sits near easting 5 km
    assert pts.shape[1] == 3 and np.allclose(pts[:, 2], 0)
    assert abs(np.median(pts[:, 1])) < 30e3                   # mesh y = easting


# -- ocean cells -------------------------------------------------------------
def test_thin_sheet_preserves_column_conductance():
    zc = np.array([500.0, 1625.0, 3000.0])
    t = np.array([1000.0, 1250.0, 1500.0])
    s = O.thin_sheet_conductivity(zc, np.zeros(3), np.full(3, 2000.0), DZ, 2000.0, 3.2, SIGMA_BG)
    assert abs((s * t).sum() - (SIGMA_BG * t.sum() + (3.2 - SIGMA_BG) * 2000.0)) < 1e-6
    assert s[2] == SIGMA_BG                                   # below the sheet: untouched


def test_thin_sheet_refined_cell_and_land():
    # level-1 cell in layer 1 (625 m thick, 1000-1625 m) fully inside X = 2 km
    s = O.thin_sheet_conductivity([1312.5], [1], [500.0], DZ, 2000.0, 3.2, SIGMA_BG)
    assert abs(s[0] - 500.0 * 3.2 / 2000.0) < 1e-9
    s = O.thin_sheet_conductivity([500.0], [0], [-300.0], DZ, 2000.0, 3.2, SIGMA_BG)
    assert s[0] == SIGMA_BG                                   # land (negative depth)


def test_binning_keeps_background_cells_out():
    cls, vals = O.bin_conductivities(np.array([SIGMA_BG, 0.5, 0.505, 3.0, SIGMA_BG]), SIGMA_BG)
    assert cls[0] == cls[4] == -1 and cls[1] == cls[2] != cls[3]
    assert len(vals) == 2 and abs(vals[cls[3]] - 3.0) < 1e-9


def test_real_ocean_classification():
    sel = O.classify_real_ocean([-1.0, 2.0, 5.0], [True, True, False])
    assert sel.tolist() == [False, True, False]


# -- mesh helpers with stand-in cells ----------------------------------------
class _P:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


class _Cell:
    def __init__(self, x, y, z, level, mat, vol=1.0):
        self._p, self._l, self._v = _P(x, y, z), level, vol
        self.material_id, self.refine_flag = mat, None

    def center(self):
        return self._p

    def level(self):
        return self._l

    def measure(self):
        return self._v


class _Tria:
    def __init__(self, cells):
        self.cells, self.n_refines = cells, 0

    def active_cells(self):
        return iter(self.cells)

    def execute_coarsening_and_refinement(self):
        self.n_refines += 1


def _interp(tc):
    from scipy.interpolate import NearestNDInterpolator

    return NearestNDInterpolator((tc[:, :, 1].ravel(), tc[:, :, 0].ravel()), tc[:, :, 2].ravel())


def _sea_east_dem():
    n = 21
    E, N = np.meshgrid(np.linspace(-1e5, 1e5, n), np.linspace(-1e5, 1e5, n))
    tc = np.zeros((n, n, 3))
    tc[:, :, 0], tc[:, :, 1], tc[:, :, 2] = E, N, np.where(E > 0, -1000.0, 200.0)
    return tc


def test_assign_ocean_thin_sheet():
    cells = [_Cell(0, 50e3, 500, 0, 2), _Cell(0, 50e3, 1625, 0, 2), _Cell(0, 50e3, 3000, 0, 2),
             _Cell(0, -50e3, 500, 0, 2), _Cell(0, 50e3, -500, 0, 1)]
    mats = {1: ("air", 1e-8), 2: ("ground", SIGMA_BG)}
    st = M.assign_ocean(_Tria(cells), "thin_sheet", _interp(_sea_east_dem()), air_id=1,
                        earth_id=2, ocean_id=3, dz_earth=DZ, sheet_m=2000.0, sigma_sea=3.2,
                        sigma_bg=SIGMA_BG, materials=mats, log=lambda *_: None)
    ids = [c.material_id for c in cells]
    assert st["n_ocean_cells"] == 2 and ids[2:] == [2, 2, 1]  # deep, land, air untouched
    assert ids[0] != ids[1] and min(ids[:2]) >= O.SHEET_FIRST_ID
    assert abs(mats[ids[0]][1] - 1.6) < 1e-9                  # 1000 m * 3.2 / 2000 m


def test_assign_ocean_real():
    cells = [_Cell(0, 50e3, 500, 0, 1, 5.0), _Cell(0, 50e3, -500, 0, 1, 5.0),
             _Cell(0, 50e3, 2000, 0, 2)]
    mats = {1: ("air", 1e-8), 2: ("ground", SIGMA_BG)}
    logs: list[str] = []
    st = M.assign_ocean(_Tria(cells), "real", _interp(_sea_east_dem()), air_id=1, earth_id=2,
                        ocean_id=3, dz_earth=DZ, sheet_m=2000.0, sigma_sea=3.2,
                        sigma_bg=SIGMA_BG, materials=mats, log=logs.append,
                        expected_volume=20.0)
    assert [c.material_id for c in cells] == [3, 1, 2]
    assert mats[3] == ("sea", 3.2) and abs(st["volume_ratio"] - 0.25) < 1e-9
    assert any("WARNING" in m for m in logs)                  # < 70 % resolved


def test_refine_coast_flags_nearby_earth_cells_only():
    cells = [_Cell(0, 0, 500, 0, 2), _Cell(0, 5e3, 500, 0, 2), _Cell(0, 30e3, 500, 0, 2),
             _Cell(0, 5e3, 50e3, 0, 2), _Cell(0, 5e3, -500, 0, 1)]
    tria = _Tria(cells)
    M._refine_coast(tria, [[0, 0, 0], [0, 10e3, 0]], [10e3, 10e3, 20e3], 2, [1])
    assert [c.refine_flag for c in cells] == ["isotropic", "isotropic", None, None, None]
    assert tria.n_refines == 1


# -- stage expansion ---------------------------------------------------------
def test_coastline_and_interface_stage_ops():
    stages = [{"kind": "coastline", "enabled": True, "levels": 2, "radius_km": 10,
               "depth_km": 20, "taper": True, "shape": "ellipse"},
              {"kind": "interface", "enabled": True, "levels": 2},
              {"kind": "coastline", "enabled": False, "levels": 3}]
    ops, counts = MD.stages_to_ops(stages, [[0, 0], [1e3, 2e3]], delta_min_m=100.0,
                                   model_depth_m=1e5, coast_fn=lambda s: [[0, 0, 0], [1, 1, 0]],
                                   ocean_surface=True)
    assert counts == [2, 1, 0] and counts == [MD.stage_op_count(s) for s in stages]
    assert [o["op"] for o in ops] == ["coast", "coast", "interface"]
    assert ops[0]["radii"][2] == 20e3 and ops[1]["radii"][2] == 12e3     # taper 0.6/level
    assert ops[2]["ocean_surface"] is True and ops[0]["exclude"] == [1]
    # without a coast function the stage still occupies its steps (empty point list)
    ops, counts = MD.stages_to_ops(stages[:1], [[0, 0], [1e3, 2e3]], delta_min_m=100.0,
                                   model_depth_m=1e5)
    assert counts == [2] and ops[0]["points"] == []


def test_project_migration_and_defaults():
    import json
    import tempfile

    from mtinv_gui.project import Project

    p = Project()
    assert p.mesh.ocean_mode == "none" and abs(p.mesh.ocean_resistivity_ohm_m - 1 / 3.2) < 1e-12
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "a.mtinv.json")
        p.save(f)
        raw = json.load(open(f))
        raw["mesh"]["ocean_resistivity_ohm_m"] = 3.2          # old files stored sigma here
        del raw["mesh"]["ocean_mode"]
        json.dump(raw, open(f, "w"))
        q = Project.load(f)
    assert abs(q.mesh.ocean_resistivity_ohm_m - 1 / 3.2) < 1e-12 and q.mesh.ocean_mode == "none"


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok   ", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL ", name, "->", repr(exc))
    sys.exit(1 if failed else 0)
