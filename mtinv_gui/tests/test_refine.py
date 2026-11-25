"""Refining an inverted model (``backends/refine.py``), on small synthetic models.

Each test builds a mesh the way the setup GUI does, fits it to a known surface and
gives every earth cell its own material id, like an inversion's output.

    python tests/test_refine.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mtinv_gui import config  # noqa: E402
from mtinv_gui.backends import refine as R  # noqa: E402

config.apply_sys_paths()
import PyDealII.Release as dealii  # noqa: E402
from gofem.mesh_utils import Topography, project_points_on_interface  # noqa: E402

AIR, EARTH = 1, 2
BAND = 3000.0
QUIET = lambda *_a: None  # noqa: E731


def elevation(x, y):
    """Land, 200-1000 m."""
    return 600.0 + 400.0 * np.sin(np.asarray(x) / 90e3) * np.cos(np.asarray(y) / 70e3)


def write_dem(path, fn=elevation):
    xs = np.linspace(-500e3, 500e3, 1001)                       # past the mesh, so no blend applies
    east, north = np.meshgrid(xs, xs)
    np.savez_compressed(path, np.dstack([east, north, fn(north, east)]))
    return path


def build(fn=elevation, fit=True):
    dx = [40e3] * 12
    dz = [40e3, 20e3, 10e3, 5e3, 5e3, 5e3, 5e3, 10e3, 20e3, 40e3]
    size = np.array([sum(dx), sum(dx), sum(dz)])
    t = dealii.Triangulation("3D")
    t.generate_subdivided_steps_hyper_rectangle([dx, dx, dz], dealii.Point((-size / 2).tolist()),
                                                dealii.Point((size / 2).tolist()), False)
    for c in t.active_cells():
        c.material_id = AIR if c.center().z < 0 else EARTH
    if fit:
        Topography(lambda p: -float(fn(p[0], p[1])), dim=3, center=[0, 0, 0],
                   radii=[1e7, 1e7, 5e4]).fit_to(t, z_top=-BAND, z_bottom=BAND)
    return t


def inverted(fn=elevation):
    """Built, fitted, per-cell ids, saved + reloaded."""
    t = build(fn)
    next_id = 5000
    for c in t.active_cells():
        if c.material_id != AIR:
            c.material_id = next_id
            next_id += 1
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.tria")
        t.save(p)
        return R.load_tria(p)


def ids_of(t):
    return {c.material_id for c in t.active_cells()}


def sizes(t, where):
    out = []
    for c in t.active_cells():
        p = c.center()
        if c.material_id != AIR and where(p.x, p.y, p.z):
            a, b = c.get_vertex(0), c.get_vertex(7)
            out.append(max(abs(b.x - a.x), abs(b.y - a.y)))
    return np.array(out)


# ---- rules ------------------------------------------------------------------ #
def test_material_ids_survive_refinement_and_reload():
    t = inverted()
    before = ids_of(t)
    rule = R.RefineRule(kind="box", levels=2, half_north_km=60, half_east_km=60, depth_bottom_km=20)
    R.apply_rule(t, rule, R.surface_faces(t), log=QUIET)
    assert ids_of(t) == before
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "r.tria")
        t.save(p)
        assert ids_of(R.load_tria(p)) == before


def test_minimum_size_stops_refinement():
    t = inverted()
    rule = R.RefineRule(kind="box", levels=10, min_size_km=8.0, half_north_km=40, half_east_km=40,
                        depth_bottom_km=10)
    res = R.apply_rule(t, rule, R.surface_faces(t), log=QUIET)
    assert res["passes"] == 2                     # 40 -> 20 -> 10 km; a third would give 5 km
    assert sizes(t, lambda *_a: True).min() >= 8e3
    assert sizes(t, lambda x, y, _z: abs(x) < 30e3 and abs(y) < 30e3).min() == 10e3


def test_minimum_vertical_size_stops_refinement():
    t = inverted()
    rule = R.RefineRule(kind="box", levels=10, min_dz_km=2.0, half_north_km=40, half_east_km=40,
                        depth_top_km=0.5, depth_bottom_km=3)
    assert R.apply_rule(t, rule, R.surface_faces(t), log=QUIET)["passes"] == 1    # 5 -> 2.5 km


def test_levels_cap_passes_and_air_stays_coarse():
    t = inverted()
    air_before = sum(1 for c in t.active_cells() if c.material_id == AIR)
    rule = R.RefineRule(kind="ellipse", levels=1, half_north_km=60, half_east_km=60, depth_bottom_km=5)
    res = R.apply_rule(t, rule, R.surface_faces(t), log=QUIET)
    assert res["passes"] == 1 and res["added"] > 0
    # only deal.II's one-level rule touches air, not the rule itself: most air is unsplit
    assert sum(1 for c in t.active_cells() if c.material_id == AIR) < air_before * 8


def test_depth_is_below_the_local_surface():
    """A column from the surface down must catch cells under high ground, which are above
    sea level and would be missed by a rule in absolute depth."""
    t = inverted(lambda x, y: np.full_like(np.asarray(x, dtype=float), 900.0))
    surf = R.surface_faces(t)
    rule = R.RefineRule(kind="box", levels=1, half_north_km=500, half_east_km=500,
                        depth_top_km=0, depth_bottom_km=2)
    cells, xyz, mat = R._cell_arrays(t)
    _d, near = surf.centres_tree().query(xyz[:, :2])
    depth = xyz[:, 2] - surf.z[near]
    earth = np.flatnonzero(mat != AIR)
    half = R._half_sizes(cells, earth)
    hit = R._in_region(rule, xyz[earth, 0], xyz[earth, 1], depth[earth], None, half)
    assert hit.any()
    assert (depth[earth][hit] - half[hit, 2] <= 2000.0).all()
    assert not hit[(depth[earth] - half[:, 2]) > 2000.0].any()


def test_regions_catch_cells_they_only_overlap():
    """A region smaller than a cell, or missing its centre, must still refine it."""
    half = np.array([[20e3, 20e3, 5e3]])
    x, y, d = [0.0], [0.0], [5e3]
    small = dict(depth_top_km=0, depth_bottom_km=1)
    corner = dict(center_north_km=15, center_east_km=15, half_north_km=2, half_east_km=2)
    assert R._in_region(R.RefineRule(kind="box", **small, **corner), x, y, d, None, half)[0]
    assert R._in_region(R.RefineRule(kind="ellipse", **small, **corner), x, y, d, None, half)[0]
    poly = R.RefineRule(kind="polygon", polygon_km=[[14, 14], [16, 14], [15, 16]], **small)
    assert R._in_region(poly, x, y, d, None, half)[0]
    near = R.RefineRule(kind="stations", radius_km=1, depth_bottom_km=1)
    assert R._in_region(near, x, y, d, [[15e3, 15e3]], half)[0]
    # ... but not one that misses it
    far = dict(center_north_km=30, center_east_km=30, half_north_km=2, half_east_km=2)
    assert not R._in_region(R.RefineRule(kind="box", **small, **far), x, y, d, None, half)[0]
    away = R.RefineRule(kind="polygon", polygon_km=[[30, 30], [35, 30], [30, 35]], **small)
    assert not R._in_region(away, x, y, d, None, half)[0]
    assert not R._in_region(near, x, y, d, [[40e3, 40e3]], half)[0]
    assert not R._in_region(R.RefineRule(kind="box", depth_top_km=20, depth_bottom_km=30),
                            x, y, d, None, half)[0]


def test_small_region_refines_down_to_its_own_size():
    t = inverted()
    rule = R.RefineRule(kind="box", levels=4, min_size_km=4, center_north_km=15, center_east_km=15,
                        half_north_km=1, half_east_km=1, depth_top_km=0, depth_bottom_km=1)
    res = R.apply_rule(t, rule, R.surface_faces(t), log=QUIET)
    assert res["passes"] == 3                                  # 40 -> 20 -> 10 -> 5 km
    assert sizes(t, lambda *_a: True).min() == 5e3
    assert sizes(t, lambda a, b, _z: abs(a) > 120e3 or abs(b) > 120e3).min() == 40e3     # stays local


def test_polygon_and_station_regions():
    x, y = np.meshgrid(np.linspace(-50e3, 50e3, 21), np.linspace(-50e3, 50e3, 21))
    x, y, d = x.ravel(), y.ravel(), np.full(x.size, 1000.0)
    tri = R.RefineRule(kind="polygon", polygon_km=[[0, 0], [40, 0], [0, 40]], depth_bottom_km=5)
    hit = R._in_region(tri, x, y, d)
    assert hit.any() and (x[hit] + y[hit] <= 40e3).all() and (x[hit] >= 0).all()
    near = R.RefineRule(kind="stations", radius_km=10, depth_bottom_km=5)
    hit = R._in_region(near, x, y, d, station_xy=[[0.0, 0.0]])
    assert hit.any() and (np.hypot(x[hit], y[hit]) <= 10e3).all()
    assert not R._in_region(near, x, y, d, station_xy=[]).any()


def test_cell_limit_stops_before_splitting():
    t = inverted()
    n = t.n_active_cells()
    rule = R.RefineRule(kind="box", levels=5, half_north_km=200, half_east_km=200, depth_bottom_km=50)
    try:
        R.apply_rule(t, rule, R.surface_faces(t), max_cells=n + 10, log=QUIET)
    except ValueError as exc:
        assert "limit" in str(exc)
    else:
        raise AssertionError("no limit error")
    assert t.n_active_cells() == n


# ---- topography ------------------------------------------------------------- #
def test_refining_does_not_move_the_surface():
    t = inverted()
    xy = np.random.default_rng(1).uniform(-200e3, 200e3, (40, 2))
    names = [f"s{i}" for i in range(len(xy))]
    before = R.project_to_surface(R.surface_faces(t), names, xy)
    R.apply_rule(t, R.RefineRule(kind="box", levels=2, half_north_km=150, half_east_km=150,
                                 depth_bottom_km=20), R.surface_faces(t), log=QUIET)
    after = R.project_to_surface(R.surface_faces(t), names, xy)
    assert np.abs(after[:, 2] - before[:, 2]).max() < 1e-6


def test_balancing_leaves_no_two_level_jumps():
    """A region refined twice leaves cells two levels apart along its corner edges (the
    refined model that gave NaNs in GoFEM had 34 of them); balancing removes them without
    moving the surface or changing any material id."""
    t = inverted()
    xy = np.random.default_rng(2).uniform(-200e3, 200e3, (30, 2))
    names = [f"s{i}" for i in range(len(xy))]
    R.apply_rule(t, R.RefineRule(kind="box", levels=3, half_north_km=20, half_east_km=20,
                                 depth_bottom_km=10), R.surface_faces(t), log=QUIET)
    assert R.level_jumps(t).max() >= 2
    before = R.project_to_surface(R.surface_faces(t), names, xy)
    ids = ids_of(t)
    added = R.balance_levels(t, log=QUIET)
    assert added > 0 and R.level_jumps(t).max() <= 1
    assert ids_of(t) == ids
    after = R.project_to_surface(R.surface_faces(t), names, xy)
    assert np.abs(after[:, 2] - before[:, 2]).max() < 1e-6
    assert R.balance_levels(t, log=QUIET) == 0                 # nothing left to do


def test_run_balances_the_refined_model():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.tria")
        inverted().save(p)
        rule = R.RefineRule(kind="box", levels=3, half_north_km=20, half_east_km=20, depth_bottom_km=10)
        result = R.run({"tria": p, "rules": [rule.asdict()]}, d, log=QUIET, write=True)
        assert result["balanced_cells"] > 0
        assert R.level_jumps(R.load_tria(result["paths"]["tria"])).max() <= 1


def test_run_with_no_rules_leaves_the_mesh_alone():
    """Stations dropped on the old model: no balancing either, even if the model has jumps."""
    with tempfile.TemporaryDirectory() as d:
        t = inverted()
        rule = R.RefineRule(kind="box", levels=3, half_north_km=20, half_east_km=20, depth_bottom_km=10)
        R.apply_rule(t, rule, R.surface_faces(t), log=QUIET)
        assert R.level_jumps(t).max() >= 2
        p = os.path.join(d, "m.tria")
        t.save(p)
        result = R.run({"tria": p, "rules": []}, d, log=QUIET, write=True)
        assert result["balanced_cells"] == 0
        assert R.load_tria(result["paths"]["tria"]).n_active_cells() == t.n_active_cells()


# ---- coarsening ------------------------------------------------------------------- #
def _refined_checkerboard():
    t = inverted()
    mats = _checkerboard(t)
    original = {(round(c.center().x), round(c.center().y), round(c.center().z)): mats[c.material_id].conductivity
                for c in t.active_cells()}
    n0 = t.n_active_cells()
    rule = R.RefineRule(kind="box", levels=2, half_north_km=100, half_east_km=100, depth_bottom_km=30)
    R.apply_rule(t, rule, R.surface_faces(t), log=QUIET)
    return t, mats, original, n0


def test_coarsening_inside_undoes_refining_with_the_same_values():
    t, mats, original, n0 = _refined_checkerboard()
    refined = t.n_active_cells()
    rule = R.RefineRule(kind="box", action="coarsen", levels=3, half_north_km=100, half_east_km=100,
                        depth_bottom_km=30)
    res = R.apply_rule(t, rule, R.surface_faces(t), materials=mats, log=QUIET)
    assert t.n_active_cells() == n0 and res["added"] == n0 - refined
    assert res["merged"] > 0 and len(res["new_ids"]) > 0
    for c in t.active_cells():
        p = c.center()
        key = (round(p.x), round(p.y), round(p.z))
        assert key in original                                       # the original grid again
        assert c.material_id in mats
        assert abs(np.log10(mats[c.material_id].conductivity) - np.log10(original[key])) < 1e-9


def test_coarsening_outside_keeps_the_region_and_averages_by_volume():
    t, mats, _original, _n0 = _refined_checkerboard()
    # a mixed parent: give one parent's children different values, as an inversion would
    keep = R.RefineRule(kind="box", half_north_km=20, half_east_km=20, depth_bottom_km=30)
    cells, xyz, mat = R._cell_arrays(t)
    inside_before = {(round(x), round(y), round(z)) for x, y, z in xyz
                     if abs(x) < 20e3 and abs(y) < 20e3 and z < 25e3 and z > 0}
    rng = np.random.default_rng(0)
    for c in cells:
        if c.material_id != AIR:
            mid = max(mats) + 1
            mats[mid] = R.Material(mid, "ground", float(10 ** rng.uniform(-3, -1)))
            c.material_id = mid
    snap = {(round(c.center().x), round(c.center().y), round(c.center().z)): (c.measure(), mats[c.material_id].conductivity)
            for c in t.active_cells() if c.material_id != AIR}
    rule = R.RefineRule(**{**keep.asdict(), "action": "coarsen", "outside": True, "levels": 2})
    before_n = t.n_active_cells()
    res = R.apply_rule(t, rule, R.surface_faces(t), materials=mats, log=QUIET)
    assert t.n_active_cells() < before_n
    after = {(round(c.center().x), round(c.center().y), round(c.center().z)) for c in t.active_cells()}
    assert {k for k in inside_before if k in snap} <= after          # the kept region is untouched
    assert all(c.material_id in mats for c in t.active_cells())
    assert all(c.material_id == AIR for c in t.active_cells() if c.center().z < -BAND)   # air stays air
    # one merged cell: its value is the volume-weighted mean of log10 conductivity of its children
    new = [c for c in t.active_cells() if c.material_id in res["new_ids"]]
    c = new[len(new) // 2]
    a, b = c.get_vertex(0), c.get_vertex(7)
    lo = np.minimum([a.x, a.y, a.z], [b.x, b.y, b.z])
    hi = np.maximum([a.x, a.y, a.z], [b.x, b.y, b.z])
    kids = [(v, s) for (x, y, z), (v, s) in snap.items()
            if lo[0] < x < hi[0] and lo[1] < y < hi[1] and lo[2] - 1 < z < hi[2] + 1]
    # (children may be merged twice: average over all original cells inside it)
    want = sum(v * np.log10(s) for v, s in kids) / sum(v for v, _s in kids)
    assert len(kids) in (8, 64)
    assert abs(np.log10(mats[c.material_id].conductivity) - want) < 1e-6


def test_coarsening_respects_the_size_limit_and_never_mixes_materials():
    t, mats, _o, _n0 = _refined_checkerboard()                  # 40 km cells split to 10 km
    rule = R.RefineRule(kind="box", action="coarsen", outside=True, levels=3, half_north_km=1,
                        half_east_km=1, depth_bottom_km=1, max_size_km=20)
    R.apply_rule(t, rule, R.surface_faces(t), materials=mats, log=QUIET)
    widths = sizes(t, lambda x, y, z: True)
    assert widths.max() <= 40e3 + 1 and np.isclose(widths, 20e3).any()
    assert sorted({c.material_id for c in t.active_cells() if c.center().z < -BAND}) == [AIR]


def test_run_coarsens_then_refines_and_writes_every_id():
    with tempfile.TemporaryDirectory() as d:
        tria, mats, rec, _names = _files(d)
        rules = [R.RefineRule(kind="box", levels=2, half_north_km=100, half_east_km=100, depth_bottom_km=20),
                 R.RefineRule(kind="box", action="coarsen", outside=True, levels=2, half_north_km=30,
                              half_east_km=30, depth_bottom_km=20),
                 R.RefineRule(kind="stations", stations="all", radius_km=5, depth_bottom_km=5, levels=1)]
        spec = {"tria": tria, "materials": mats, "receivers": rec, "rules": [r.asdict() for r in rules],
                "smooth": True}
        result = R.run(spec, os.path.join(d, "out"), log=QUIET, write=True)
        assert [r["action"] for r in result["rules"]] == ["refine", "coarsen", "refine"]
        assert result["rules"][1]["added"] < 0
        written = R.read_materials(result["paths"]["materials"])
        t = R.load_tria(result["paths"]["tria"])
        assert ids_of(t) <= set(written)
        assert not R.verify_receivers(t, *R.read_receivers(result["paths"]["receivers"]))


# ---- smoothing the refined cells ------------------------------------------- #
def _checkerboard(t, size_m=40e3):
    """log10 conductivity -2 / 0 in a checkerboard of the 40 km base cells."""
    mats = {AIR: R.Material(AIR, "air", 1e-8)}
    for c in t.active_cells():
        if c.material_id != AIR:
            p = c.center()
            odd = (int(np.floor(p.x / size_m)) + int(np.floor(p.y / size_m))) % 2
            mats[c.material_id] = R.Material(c.material_id, "ground", 10.0 ** (-2 + 2 * odd))
    return mats


def _smoothed(mode="relative", h=0.25, v=0.25):
    t = inverted()
    mats = _checkerboard(t)
    before = R.snapshot_cells(t)
    old = {(c.level(), c.index()): c.material_id for c in t.active_cells()}
    rule = R.RefineRule(kind="box", levels=2, half_north_km=60, half_east_km=60, depth_bottom_km=20)
    R.apply_rule(t, rule, R.surface_faces(t), log=QUIET)
    result = R.smooth_refined(t, mats, before, mode=mode, horizontal=h, vertical=v, log=QUIET)
    return t, mats, old, result


def test_smoothing_gives_refined_cells_own_ids_and_leaves_the_rest():
    t, mats, old, result = _smoothed()
    new = [c for c in t.active_cells() if (c.level(), c.index()) not in old]
    kept = [c for c in t.active_cells() if (c.level(), c.index()) in old]
    earth_new = [c for c in new if c.material_id != AIR]
    assert result["cells"] == len(earth_new) > 1000
    assert len({c.material_id for c in earth_new}) == len(earth_new)          # one id each
    assert all(c.material_id in mats for c in earth_new)
    assert all(c.material_id == old[(c.level(), c.index())] for c in kept)    # untouched
    assert all(c.material_id == AIR for c in new if c.center().z < -BAND)       # air stays air
    logs = np.log10([mats[c.material_id].conductivity for c in earth_new])
    assert logs.min() >= -2.0 - 1e-9 and logs.max() <= 0.0 + 1e-9                # never past the old range


def test_smoothing_is_a_gaussian_filter_of_the_old_blocks():
    """Across a face between -2 and 0 blocks the value follows the error function."""
    from scipy.special import erf
    t, mats, _old, result = _smoothed(h=0.25)
    first_new = min(result["new_ids"])
    sigma = 0.25 * 40e3
    checked = 0
    for c in t.active_cells():
        p = c.center()
        # refined cells near the x = 0 face (the pattern doesn't change with depth)
        if c.material_id >= first_new and abs(p.x) < 16e3 and 0 < p.y < 20e3:
            got = np.log10(mats[c.material_id].conductivity)
            # y in [0, 40 km] is odd (0) for x < 0, even (-2) for x > 0; y faces at 0 and 40 km
            fx = 0.5 * (1 - erf(p.x / (sigma * np.sqrt(2))))            # share of the x < 0 block
            fy = 0.5 * (erf(p.y / (sigma * np.sqrt(2))) - erf((p.y - 40e3) / (sigma * np.sqrt(2))))
            want_odd_share = fx * fy + (1 - fx) * (1 - fy)
            assert abs(got - (-2 + 2 * want_odd_share)) < 0.03, (p.x, p.y, got, want_odd_share)
            checked += 1
    assert checked >= 8


def test_smoothing_relative_and_km_agree_when_the_lengths_do():
    _t1, m1, _o1, r1 = _smoothed("relative", 0.25, 0.25)
    _t2, m2, _o2, r2 = _smoothed("km", 10.0, 0.25 * 5.0)     # the refined base cells are 40 km x 5 km
    a = np.array([m.conductivity for m in r1["new_ids"].values()])
    b = np.array([m.conductivity for m in r2["new_ids"].values()])
    assert np.allclose(np.log10(a), np.log10(b), atol=0.05)
    assert abs(np.median(r1["change"]) - np.median(r2["change"])) < 0.05


def test_smoothing_more_blurs_more():
    light = _smoothed(h=0.15, v=0.15)[3]["change"]
    strong = _smoothed(h=0.5, v=0.5)[3]["change"]
    assert np.median(strong) > np.median(light) + 0.2


def test_run_smooths_and_writes_the_new_materials():
    with tempfile.TemporaryDirectory() as d:
        tria, mats, rec, _names = _files(d)
        rule = R.RefineRule(kind="box", levels=1, half_north_km=50, half_east_km=50, depth_bottom_km=20)
        out = os.path.join(d, "out")
        spec = {"tria": tria, "materials": mats, "receivers": rec, "rules": [rule.asdict()],
                "smooth": True, "smooth_mode": "relative", "smooth_horizontal": 0.25,
                "smooth_vertical": 0.25}
        result = R.run(spec, out, log=QUIET, write=True)
        assert result["smoothed"]["cells"] > 0
        written = R.read_materials(result["paths"]["materials"])
        assert len(written) == len(R.read_materials(mats)) + result["smoothed"]["cells"]
        t = R.load_tria(result["paths"]["tria"])
        assert ids_of(t) <= set(written)                      # every id has a conductivity
        with open(mats) as fh:
            original = fh.read()
        with open(result["paths"]["materials"]) as fh:
            assert fh.read().startswith(original)             # the old lines are untouched
        # with no rules, nothing is smoothed and the materials are a plain copy
        result = R.run({**spec, "rules": []}, os.path.join(d, "out2"), log=QUIET, write=True)
        assert result["smoothed"] is None
        with open(result["paths"]["materials"]) as fh:
            assert fh.read() == original


def test_fit_to_on_a_fitted_mesh_is_why_refit_exists():
    """Documents the trap: applying fit_to again nearly doubles the relief."""
    t = inverted()
    Topography(lambda p: -float(elevation(p[0], p[1])), dim=3, center=[0, 0, 0],
               radii=[1e7, 1e7, 5e4]).fit_to(t, z_top=-BAND, z_bottom=BAND)
    assert (-R.surface_faces(t).vertices()[:, 2]).max() > 1500.0


def test_refit_with_the_same_dem_changes_nothing_at_old_vertices():
    with tempfile.TemporaryDirectory() as d:
        dem = write_dem(os.path.join(d, "dem.npz"))
        t = inverted()
        old = R.surface_faces(t).vertices()
        R.refit_topography(t, dem, log=QUIET)
        new = {(round(a, 3), round(b, 3)): c for a, b, c in R.surface_faces(t).vertices()}
        # the DEM here is a 1 km grid sampled by nearest neighbour, as when building a mesh
        assert max(abs(new[(round(a, 3), round(b, 3))] - c) for a, b, c in old) < 1e-6
        assert (-np.array(list(new.values()))).max() < 1001.0        # not doubled


def test_refit_after_refining_matches_a_fresh_build_and_can_repeat():
    with tempfile.TemporaryDirectory() as d:
        dem = write_dem(os.path.join(d, "dem.npz"))
        rule = R.RefineRule(kind="box", levels=2, half_north_km=120, half_east_km=120, depth_bottom_km=20)

        fresh = build(fit=False)
        R.apply_rule(fresh, rule, R.surface_faces(fresh), log=QUIET)
        data = np.load(dem)["arr_0"]
        from scipy.interpolate import NearestNDInterpolator

        near = NearestNDInterpolator((data[:, :, 1].ravel(), data[:, :, 0].ravel()), data[:, :, 2].ravel())
        Topography(lambda p: -float(near([p])[0]), dim=3, center=[0, 0, 0],
                   radii=[1e7, 1e7, 5e4]).fit_to(fresh, z_top=-BAND, z_bottom=BAND)
        want = {(round(a, 3), round(b, 3)): c for a, b, c in R.surface_faces(fresh).vertices()}

        t = inverted()
        R.apply_rule(t, rule, R.surface_faces(t), log=QUIET)
        vol = sum(c.measure() for c in t.active_cells())
        R.refit_topography(t, dem, log=QUIET)
        got = R.surface_faces(t).vertices()
        assert max(abs(want[(round(a, 3), round(b, 3))] - c) for a, b, c in got) < 1e-6
        assert abs(sum(c.measure() for c in t.active_cells()) - vol) / vol < 1e-9

        R.refit_topography(t, dem, log=QUIET)                # again: nothing to do
        again = R.surface_faces(t).vertices()
        assert np.abs(again[:, 2] - got[:, 2]).max() < 1e-6


def test_refit_follows_a_more_detailed_dem():
    def detailed(x, y):
        return elevation(x, y) + 60.0 * np.sin(np.asarray(x) / 9e3) * np.sin(np.asarray(y) / 11e3)

    with tempfile.TemporaryDirectory() as d:
        dem = write_dem(os.path.join(d, "dem.npz"), detailed)
        t = inverted()
        # the detail is ~50 km across, so this needs cells of 5 km; what remains is at coarse /
        # fine junctions, where a vertex must stay on its coarse neighbour's face
        R.apply_rule(t, R.RefineRule(kind="box", levels=3, half_north_km=120, half_east_km=120,
                                     depth_bottom_km=20), R.surface_faces(t), log=QUIET)
        stats = R.refit_topography(t, dem, log=QUIET)
        assert stats["rms_after_m"] < stats["rms_before_m"] / 3


def test_refit_with_a_regional_dem_only_changes_the_area_it_covers():
    """A DEM is silent beyond its edges: the rest of the model must not move, and there
    must be no cliff where it ends."""
    def raised(x, y):
        return elevation(x, y) + 200.0

    with tempfile.TemporaryDirectory() as d:
        xs = np.linspace(-100e3, 100e3, 201)
        east, north = np.meshgrid(xs, xs)
        dem = os.path.join(d, "regional.npz")
        np.savez_compressed(dem, np.dstack([east, north, raised(north, east)]))
        t = inverted()
        R.apply_rule(t, R.RefineRule(kind="box", levels=2, half_north_km=140, half_east_km=140,
                                     depth_bottom_km=20), R.surface_faces(t), log=QUIET)
        before = R.surface_faces(t).vertices()
        R.refit_topography(t, dem, blend_km=40.0, log=QUIET)
        after = R.surface_faces(t).vertices()
        moved = -(after[:, 2] - before[:, 2])
        reach = np.maximum(np.abs(before[:, 0]), np.abs(before[:, 1]))
        assert np.abs(moved[reach >= 100e3]).max() < 1e-6      # outside: untouched
        centre = reach <= 50e3
        assert centre.any() and np.abs(moved[centre] - 200.0).max() < 25.0
        ramp = (reach > 60e3) & (reach < 100e3)                # across the blend: partial, no cliff
        assert ramp.any() and moved[ramp].min() >= -1.0 and moved[ramp].max() <= 225.0
        assert (moved[ramp] > 1.0).any() and (moved[ramp] < 199.0).any()
        far = os.path.join(d, "far.npz")
        np.savez_compressed(far, np.dstack([east + 1e8, north, raised(north, east)]))
        try:
            R.refit_topography(t, far, log=QUIET)
        except ValueError as exc:
            assert "overlap" in str(exc)
        else:
            raise AssertionError("no error for a DEM off the model")


def _hills(x, y):
    """Terrain with detail at several scales, so that a shifted copy really differs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    return (600.0 + 300.0 * np.sin(x / 40e3 + 1.0) * np.cos(y / 30e3)
            + 150.0 * np.sin(x / 13e3) * np.sin(y / 17e3 + 0.5))


def _hilly_surface():
    t = inverted(_hills)
    R.apply_rule(t, R.RefineRule(kind="box", levels=2, half_north_km=200, half_east_km=200,
                                 depth_bottom_km=20), R.surface_faces(t), log=QUIET)
    refit = os.path.join(tempfile.mkdtemp(), "fine.npz")
    write_dem(refit, _hills)
    R.refit_topography(t, refit, log=QUIET)
    return t, R.surface_faces(t)


def build_copy(t):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.tria")
        t.save(p)
        return R.load_tria(p)


class _Frame:
    """A model's frame in UTM 15N, like plotting.backends.georef.Georef."""
    crs, x0, y0 = "EPSG:32615", 500000.0, 5400000.0


def _write_lonlat_dem(path, fn=elevation, span_deg=(8.0, 6.0), step_deg=0.02):
    """As the Topography tab writes one: lon, lat and elevation grids."""
    from pyproj import Transformer

    to_lonlat = Transformer.from_crs(_Frame.crs, 4326, always_xy=True)
    lon0, lat0 = to_lonlat.transform(_Frame.x0, _Frame.y0)
    lon, lat = np.meshgrid(np.arange(lon0 - span_deg[0], lon0 + span_deg[0], step_deg),
                           np.arange(lat0 - span_deg[1], lat0 + span_deg[1], step_deg))
    east, north = Transformer.from_crs(4326, _Frame.crs, always_xy=True).transform(lon, lat)
    np.savez(path, lon, lat, fn(north - _Frame.y0, east - _Frame.x0))
    return path


def test_lonlat_dem_is_resampled_into_the_model_frame():
    with tempfile.TemporaryDirectory() as d:
        path = _write_lonlat_dem(os.path.join(d, "topo.npz"))
        assert R.is_lonlat_dem(path)
        grid = R.read_dem(path, _Frame)
        assert grid.ndim == 3 and grid.shape[2] == 3
        east, north, z = grid[:, :, 0], grid[:, :, 1], grid[:, :, 2]
        assert np.sqrt(np.mean((z - elevation(north, east)) ** 2)) < 2.0     # terrain 90 km long, grid 2 km
        assert R.read_dem(path, _Frame) is grid                              # not done twice


def test_lonlat_dem_grid_stays_inside_the_tilted_footprint():
    """UTM tilts the box, so its bounding box has corners with no DEM."""
    from pyproj import Transformer

    with tempfile.TemporaryDirectory() as d:
        path = _write_lonlat_dem(os.path.join(d, "topo.npz"))
        lon, lat = np.load(path)["arr_0"], np.load(path)["arr_1"]
        grid = R.read_dem(path, _Frame)
        back = Transformer.from_crs(_Frame.crs, 4326, always_xy=True)
        x, y = back.transform(grid[:, :, 0] + _Frame.x0, grid[:, :, 1] + _Frame.y0)
        assert x.min() >= lon.min() and x.max() <= lon.max()
        assert y.min() >= lat.min() and y.max() <= lat.max()


def test_lonlat_dem_needs_the_frame_and_other_files_are_refused():
    t, surf = _hilly_surface()
    with tempfile.TemporaryDirectory() as d:
        path = _write_lonlat_dem(os.path.join(d, "topo.npz"), _hills)
        try:
            R.check_dem_frame(surf, path)
            raise AssertionError("a lon/lat DEM was used with no frame")
        except ValueError as exc:
            assert "lon/lat" in str(exc)
        other = os.path.join(d, "other.npz")
        np.savez(other, values=np.zeros(3))
        flat = os.path.join(d, "flat.npz")
        np.savez(flat, np.zeros((3, 3)))
        plain = os.path.join(d, "plain.npy")
        np.save(plain, np.zeros((3, 3, 3)))
        text = os.path.join(d, "text.npz")
        with open(text, "w") as fh:
            fh.write("not a DEM")
        empty = os.path.join(d, "empty.npz")
        open(empty, "w").close()
        for bad in (other, flat, plain, text, empty):
            try:
                R.read_dem(bad)
                raise AssertionError(f"{bad} was accepted")
            except ValueError as exc:
                assert "isn't a DEM" in str(exc)


def test_lonlat_dem_check_and_refit_match_a_projected_one():
    t, surf = _hilly_surface()
    with tempfile.TemporaryDirectory() as d:
        path = _write_lonlat_dem(os.path.join(d, "topo.npz"), _hills)
        check = R.check_dem_frame(surf, path, georef=_Frame)
        assert check.status == "ok", check.message
        projected = write_dem(os.path.join(d, "projected.npz"), _hills)
        want = R.refit_topography(build_copy(t), projected, log=QUIET)
        stats = R.refit_topography(t, path, log=QUIET, georef=_Frame)
        # the surface is already on these hills, apart from coarse/fine junctions
        assert abs(stats["rms_after_m"] - want["rms_after_m"]) < 2.0, (stats, want)
        assert abs(stats["max_shift_m"] - want["max_shift_m"]) < 0.1 * want["max_shift_m"], (stats, want)


def test_lonlat_dem_in_the_wrong_frame_is_caught():
    class Elsewhere(_Frame):
        y0 = _Frame.y0 + 60e3

    t, surf = _hilly_surface()
    with tempfile.TemporaryDirectory() as d:
        path = _write_lonlat_dem(os.path.join(d, "topo.npz"), _hills)
        check = R.check_dem_frame(surf, path, georef=Elsewhere)
        assert check.is_problem, check.message


def test_run_uses_the_frame_for_a_lonlat_dem():
    t, surf = _hilly_surface()
    with tempfile.TemporaryDirectory() as d:
        path = _write_lonlat_dem(os.path.join(d, "topo.npz"), _hills)
        tria = os.path.join(d, "m.tria")
        t.save(tria)
        spec = {"tria": tria, "refit_dem": path, "frame": "manual", "frame_crs": _Frame.crs,
                "frame_x0": _Frame.x0, "frame_y0": _Frame.y0}
        res = R.run(spec, log=QUIET, write=False)
        assert res["dem_check"].status == "ok" and res["refit"]["rms_after_m"] < 30.0, res["refit"]
        assert "UTM" in res["frame"] or "32615" in res["frame"]
        try:
            R.run({"tria": tria, "refit_dem": path, "frame": "fit"}, log=QUIET, write=False)
            raise AssertionError("refit ran with no frame")
        except ValueError as exc:
            assert "frame" in str(exc)


# ---- plan-view slices -------------------------------------------------------- #
def test_slice_shows_one_layer_of_cells_with_their_resistivity():
    t = build(fit=False)
    next_id = 5000
    materials = {AIR: R.Material(AIR, "air", 1e-8)}
    for c in t.active_cells():
        if c.material_id != AIR:
            c.material_id = next_id
            materials[next_id] = R.Material(next_id, "m", 10.0 ** (-1 - c.center().x / 1e6))
            next_id += 1
    cells = R.model_cells(t, materials)
    rects, values = R.plan_slice(cells, 2500.0)                  # inside the 5 km layer at 0-5 km
    assert len(rects) == 12 * 12
    assert np.allclose(rects[:, 1] - rects[:, 0], 40e3) and np.allclose(rects[:, 3] - rects[:, 2], 40e3)
    north = 0.5 * (rects[:, 0] + rects[:, 1])
    assert np.allclose(values, 1 + north / 1e6)
    assert len(R.plan_slice(cells, -2500.0)[0]) == 0             # air isn't drawn
    assert len(R.plan_slice(cells, 1e7)[0]) == 0                 # below the model


def test_slice_shows_refined_cells_and_marks_unknown_ids():
    t = inverted()
    R.apply_rule(t, R.RefineRule(kind="box", levels=1, half_north_km=30, half_east_km=30, depth_bottom_km=50),
                 R.surface_faces(t), log=QUIET)
    cells = R.model_cells(t)                                     # no .materials
    rects, values = R.plan_slice(cells, 20e3)
    widths = rects[:, 1] - rects[:, 0]
    assert widths.min() == 20e3 and widths.max() == 40e3
    assert np.isnan(values).all()
    area = np.sum((rects[:, 1] - rects[:, 0]) * (rects[:, 3] - rects[:, 2]))
    assert np.isclose(area, (12 * 40e3) ** 2)                    # no gaps or overlaps


def test_slice_depths_cover_the_model_and_every_one_has_cells():
    cells = R.model_cells(inverted())
    depths = R.slice_depths(cells)
    assert np.all(np.diff(depths) > 0) and 5 < len(depths) < 200
    assert all(len(R.plan_slice(cells, d)[0]) > 0 for d in depths)
    assert depths.max() >= 60e3                                  # the model goes to 70 km


# ---- shells ------------------------------------------------------------------ #
def _active_cells(t):
    """Every active cell as (rounded centre, level), to compare two meshes exactly."""
    return sorted((round(c.center().x, 3), round(c.center().y, 3), round(c.center().z, 3), c.level())
                  for c in t.active_cells())


def test_a_rule_saved_before_shells_keeps_its_meaning():
    rule = R.rule_from_dict({"kind": "box", "levels": 3, "depth_bottom_km": 20.0, "half_north_km": 60.0})
    assert R.rule_shells(rule) == [(0.0, 20.0)] * 3


def test_shells_follow_the_mesh_tab_sequence():
    from mtinv_gui.backends import mesh_design as MD

    fields = dict(levels=4, buffer_km=80.0, buffer_shrink=0.5, taper=True, depth_shrink=0.5)
    stage = MD.RefinementStage(kind="survey_area", max_depth_km=40.0, **fields).asdict()
    want = list(zip(*MD.survey_area_shells(stage)))
    assert R.rule_shells(R.RefineRule(depth_bottom_km=40.0, **fields)) == want
    assert want == [(80.0, 40.0), (40.0, 20.0), (20.0, 10.0), (10.0, 5.0)]
    # lists replace the sequence; a short depth list repeats its last value, as in the Mesh tab
    listed = R.RefineRule(levels=9, buffer_override=True, buffer_km_list=[30, 9, 2],
                          depth_override=True, depth_km_list=[25, 6])
    assert R.rule_shells(listed) == [(30.0, 25.0), (9.0, 6.0), (2.0, 6.0)]
    try:
        R.rule_shells(R.RefineRule(buffer_override=True, buffer_km_list=[-1.0]))
    except ValueError as exc:
        assert "negative" in str(exc)
    else:
        raise AssertionError("a negative buffer was accepted")


def test_one_shell_rule_equals_the_rules_it_replaces():
    """What the tab is for: N hand-made rules with changing buffers and depths, as one rule."""
    square = [[-30.0, -30.0], [30.0, -30.0], [30.0, 30.0], [-30.0, 30.0]]
    place = dict(center_north_km=20.0, center_east_km=-20.0, half_north_km=40.0, half_east_km=30.0)
    fields = dict(levels=3, buffer_km=80.0, buffer_shrink=0.5, taper=True, depth_shrink=0.5, depth_bottom_km=40.0)
    for kind, region in (("box", place), ("ellipse", place), ("polygon", {"polygon_km": square})):
        one = inverted()
        res = R.apply_rule(one, R.RefineRule(kind=kind, **region, **fields), R.surface_faces(one), log=QUIET)
        assert res["passes"] == res["shells"] == 3 and res["added"] > 0, (kind, res)

        by_hand = inverted()
        surf = R.surface_faces(by_hand)
        for buffer_km, depth_km in ((80.0, 40.0), (40.0, 20.0), (20.0, 10.0)):
            if kind == "polygon":
                grown = R.rule_polygon(R.RefineRule(polygon_km=square), buffer_km)
                single = dict(polygon_km=[[n / 1e3, e / 1e3] for n, e in grown.exterior.coords])
            else:
                single = {**place, "half_north_km": place["half_north_km"] + buffer_km,
                          "half_east_km": place["half_east_km"] + buffer_km}
            R.apply_rule(by_hand, R.RefineRule(kind=kind, levels=1, depth_bottom_km=depth_km, **single),
                         surf, log=QUIET)
        assert _active_cells(one) == _active_cells(by_hand), kind


def test_stations_rule_grows_its_radius_with_the_buffer():
    one, by_hand = inverted(), inverted()
    pts = np.array([[10e3, 10e3]])
    res = R.apply_rule(one, R.RefineRule(kind="stations", radius_km=10, levels=2, buffer_km=40, taper=True,
                                         depth_shrink=0.5, depth_bottom_km=20),
                       R.surface_faces(one), station_xy=pts, log=QUIET)
    assert res["passes"] == 2
    surf = R.surface_faces(by_hand)
    for radius, depth in ((50, 20), (30, 10)):
        R.apply_rule(by_hand, R.RefineRule(kind="stations", radius_km=radius, depth_bottom_km=depth),
                     surf, station_xy=pts, log=QUIET)
    assert _active_cells(one) == _active_cells(by_hand)


def test_shells_nest_with_the_finest_cells_in_the_middle():
    t = inverted()
    base = {c.level() for c in t.active_cells()}
    assert len(base) == 1
    start = base.pop()
    rule = R.RefineRule(kind="box", levels=3, half_north_km=20, half_east_km=20, buffer_km=100, buffer_shrink=0.5,
                        taper=True, depth_shrink=0.5, depth_bottom_km=40)
    surf = R.surface_faces(t)
    R.apply_rule(t, rule, surf, log=QUIET)
    tree = surf.centres_tree()

    def level_at(north, east, depth):
        best = None
        for c in t.active_cells():
            p = c.center()
            _d, i = tree.query([[p.x, p.y]])
            below = p.z - surf.z[i[0]]
            half = R._half_sizes([c], np.array([0]))[0]
            if abs(p.x - north) <= half[0] and abs(p.y - east) <= half[1] and abs(below - depth) <= half[2] \
                    and c.material_id != AIR:
                best = c.level()
        return best

    assert level_at(0, 0, 1000) == start + 3                     # inside every shell
    assert level_at(0, 0, 60e3) == start                         # below the outer shell's depth
    assert level_at(210e3, 210e3, 1000) == start                 # outside the outer shell


def test_an_empty_outer_shell_does_not_stop_the_inner_ones():
    t = inverted()
    # the outer shell's depth is above the top depth, so it has no cells; the next one has
    rule = R.RefineRule(kind="box", half_north_km=30, half_east_km=30, depth_top_km=20, depth_override=True,
                        depth_km_list=[1, 40], buffer_override=True, buffer_km_list=[0, 0])
    res = R.apply_rule(t, rule, R.surface_faces(t), log=QUIET)
    assert (res["shells"], res["passes"]) == (2, 1) and res["added"] > 0


def test_minimum_size_still_holds_across_shells():
    t = inverted()
    rule = R.RefineRule(kind="box", levels=6, min_size_km=15.0, half_north_km=100, half_east_km=100,
                        buffer_km=60, depth_bottom_km=30)
    res = R.apply_rule(t, rule, R.surface_faces(t), log=QUIET)
    assert res["passes"] < res["shells"] == 6
    earth = np.array([c.material_id != AIR for c in t.active_cells()])
    cells, _xyz, _mat = R._cell_arrays(t)
    half = R._half_sizes(cells, np.flatnonzero(earth))
    assert 2 * half[:, :2].max(axis=1).min() >= 15e3 - 1e-3


def test_polygon_buffer_covers_the_ring_around_it():
    tri = R.RefineRule(kind="polygon", polygon_km=[[0, 0], [40, 0], [0, 40]], depth_bottom_km=5)
    x, y, d = np.array([-5e3, -15e3, 20e3]), np.array([10e3, 10e3, 10e3]), np.zeros(3)
    assert R._in_region(tri, x, y, d).tolist() == [False, False, True]
    assert R._in_region(tri, x, y, d, shell=(10.0, 5.0)).tolist() == [True, False, True]


def test_rounded_polygon_is_pygofems_shape():
    """pnorm > 1 must select what pyGoFEM's refine_within_polygon does, as the Mesh tab gets it."""
    import shapely.geometry as sgeom
    from gofem.mesh_utils import is_within_ellipsoid

    verts = [[-20, -30], [40, -10], [30, 35], [-25, 20]]
    poly = sgeom.Polygon([(a * 1e3, b * 1e3) for a, b in verts])
    rng = np.random.default_rng(1)
    x, y = rng.uniform(-60e3, 60e3, 4000), rng.uniform(-60e3, 60e3, 4000)
    depth = rng.uniform(0.0, 50e3, 4000)
    n0, e0, n1, e1 = poly.bounds
    radius = 0.5 * np.hypot(n1 - n0, e1 - e0) * 1.05
    for p in (1.5, 2.0, 4.0):
        rule = R.RefineRule(kind="polygon", polygon_km=verts, depth_bottom_km=40.0, pnorm=p)
        mine = R._in_region(rule, x, y, depth)
        theirs = np.array([poly.contains(sgeom.Point(a, b)) and
                           is_within_ellipsoid([a, b, d], [poly.centroid.x, poly.centroid.y, 0.0],
                                               [radius, radius, 40e3], pnorm=p)
                           for a, b, d in zip(x, y, depth)])
        assert np.array_equal(mine, theirs), (p, int((mine != theirs).sum()))
        assert 0 < mine.sum() < R._in_region(R.RefineRule(kind="polygon", polygon_km=verts, depth_bottom_km=40.0),
                                             x, y, depth).sum()


def test_rounded_polygon_refines_a_bowl():
    square = [[-150.0, -150.0], [150.0, -150.0], [150.0, 150.0], [-150.0, 150.0]]
    flat, bowl = (R.RefineRule(kind="polygon", polygon_km=square, depth_bottom_km=40.0, pnorm=p) for p in (1.0, 2.0))
    # full depth in the middle; near a corner only the top
    x, y, d = np.array([0.0, 140e3, 140e3]), np.array([0.0, 140e3, 140e3]), np.array([35e3, 35e3, 5e3])
    assert R._in_region(flat, x, y, d).tolist() == [True, True, True]
    assert R._in_region(bowl, x, y, d).tolist() == [True, False, True]

    t = inverted()
    surf = R.surface_faces(t)
    R.apply_rule(t, flat, surf, log=QUIET)                      # 20 km cells: 40 km ones all touch the bowl
    cells, xyz, mat = R._cell_arrays(t)
    _d, near = surf.centres_tree().query(xyz[:, :2])
    idx = np.flatnonzero(mat != AIR)
    half = R._half_sizes(cells, idx)
    below = xyz[idx, 2] - surf.z[near[idx]]
    in_flat, in_bowl = (R._in_region(r, xyz[idx, 0], xyz[idx, 1], below, None, half) for r in (flat, bowl))
    assert in_bowl.sum() < in_flat.sum() and not (in_bowl & ~in_flat).any()   # a strict subset
    added = []
    for rule in (flat, bowl):
        t = inverted()
        rule.levels = 2
        added.append(R.apply_rule(t, rule, R.surface_faces(t), log=QUIET)["added"])
    assert 0 < added[1] < added[0]


def test_rule_form_edits_shells_and_saved_rules_load_unchanged():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    from mtinv_gui.gui import refine_panel as RP
    from mtinv_gui.project import Project

    app = QApplication.instance() or QApplication([])  # noqa: F841
    project = Project()
    old = {"kind": "box", "levels": 2, "half_north_km": 60.0, "half_east_km": 60.0, "depth_bottom_km": 20.0}
    project.refine.rules.append(dict(old))
    panel = RP.RefinePanel(project)
    panel.refresh()
    panel.rule_list.setCurrentRow(0)
    assert not panel.r_taper.isChecked() and panel.r_buffer.value() == 0.0
    assert R.rule_shells(R.rule_from_dict(project.refine.rules[0])) == [(0.0, 20.0)] * 2

    panel._add_rule("box")                                       # a new rule tapers, as in the Mesh tab
    assert panel.rule_list.currentRow() == 1 and panel.r_taper.isChecked()
    panel.r_levels.setValue(3)
    panel.r_buffer.setValue(40.0)
    rule = project.refine.rules[1]
    assert [b for b, _d in R.rule_shells(R.rule_from_dict(rule))] == [40.0, 20.0, 10.0]
    assert "3 levels, buffer 40 to 10 km" in panel.rule_list.item(1).text()
    assert len(RP.shell_outlines(rule)) == 3
    assert panel.r_shells_lbl.text().count("Shell") == 3
    # every outline is outside the region's own
    assert RP.rule_outline(rule, 10.0)[0][0] == rule["center_north_km"] * 1e3 - (rule["half_north_km"] + 10.0) * 1e3

    panel.r_buffer_override.setChecked(True)
    panel.r_buffer_list.setText("9, 4")
    panel.r_buffer_list.editingFinished.emit()
    assert rule["buffer_km_list"] == [9.0, 4.0] and not panel.r_levels.isEnabled()
    panel.r_buffer_list.setText("9, oops")                       # keeps the last good list
    panel.r_buffer_list.editingFinished.emit()
    assert rule["buffer_km_list"] == [9.0, 4.0] and panel.r_buffer_list.text() == "9, 4"

    assert panel.r_pnorm.isHidden()                              # a box has a flat bottom
    panel._add_rule("polygon")
    assert not panel.r_pnorm.isHidden() and panel.r_pnorm.value() == 1.0
    panel.r_pnorm.setValue(2.0)
    assert project.refine.rules[2]["pnorm"] == 2.0 and "rounded bottom (pnorm 2)" in panel.rule_list.item(2).text()

    panel.rule_list.setCurrentRow(0)                             # back to the old rule, still unchanged
    assert not panel.r_buffer_override.isChecked() and panel.r_levels.isEnabled()
    assert all(project.refine.rules[0][k] == v for k, v in old.items())


def test_map_widens_to_show_shells_and_keeps_the_view_otherwise():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    from mtinv_gui.gui.mesh_viewer import MeshViewer
    from mtinv_gui.gui.refine_panel import rule_outline

    app = QApplication.instance() or QApplication([])  # noqa: F841
    viewer = MeshViewer()
    viewer.set_stations([[0.0, 0.0], [10e3, 10e3]])
    viewer._render_2d()
    ax = viewer._plan_ax
    limits = ax.get_xlim(), ax.get_ylim()
    box = {"kind": "box", "center_north_km": 5.0, "center_east_km": 5.0, "half_north_km": 1.0, "half_east_km": 1.0}

    viewer.set_preview_polygon(rule_outline(box), [rule_outline(box, 2.0)])      # fits: the zoom stays
    assert (ax.get_xlim(), ax.get_ylim()) == limits
    viewer.set_preview_polygon(rule_outline(box), [rule_outline(box, 500.0), rule_outline(box, 2.0)])
    (e0, e1), (n0, n1) = ax.get_xlim(), ax.get_ylim()
    assert e0 < -496 and e1 > 506 and n0 < -496 and n1 > 506
    labels = [line.get_label() for line in ax.get_lines()]
    assert labels.count("shells") == 1                           # one legend entry for all of them
    assert len(ax.get_lines()) == 3                              # the region and its two shells
    viewer.set_preview_polygon(rule_outline(box))                # no shells: gone, the view stays
    assert len(ax.get_lines()) == 1 and ax.get_xlim() == (e0, e1)


def test_3d_view_keeps_the_cells_clip_box_keeps():
    """``cells_in_box`` replaces ``clip_box(crinkle=True)``, which took 84 s on a real refined model."""
    import pyvista as pv

    from mtinv_gui.gui.mesh_viewer import cells_in_box

    grid = pv.ImageData(dimensions=(9, 7, 6), spacing=(1000.0, 2000.0, 500.0),
                        origin=(-4000.0, -6000.0, -1000.0)).cast_to_unstructured_grid()
    grid = grid.separate_cells()                                 # every cell its own points, like deal.II's .vtu
    grid.cell_data["material"] = np.arange(grid.n_cells)
    boxes = [(-1500.0, 1500.0, -2500.0, 3100.0, -200.0, 900.0),    # faces inside cells
             (-2000.0, 1000.0, -2000.0, 2000.0, 0.0, 500.0),       # faces on cell faces
             (1e6, 2e6, 1e6, 2e6, 0.0, 1.0)]                       # nothing in it
    for box in boxes:
        want = grid.clip_box(box, invert=False, crinkle=True)
        got = cells_in_box(grid, box)
        assert sorted(got.cell_data["material"]) == sorted(want.cell_data["material"]) if want.n_cells \
            else got.n_cells == 0, box


def test_slice_on_the_map_can_change_depth_and_turn_off():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication

    from mtinv_gui.gui.mesh_viewer import MeshViewer

    app = QApplication.instance() or QApplication([])  # noqa: F841
    viewer = MeshViewer()
    fig = viewer.canvas.figure

    def layer(n):
        edges = np.arange(n) * 1000.0
        return np.c_[edges, edges + 1000.0, edges, edges + 1000.0], np.linspace(0.0, 4.0, n)

    for _ in range(2):                                           # off, then on and off again
        viewer.set_model_slice(*layer(4), "first depth")
        viewer.set_model_slice(*layer(9), "second depth")        # a different number of cells
        fig.canvas.draw()
        artist = viewer._slice_artist
        assert len(artist.get_paths()) == 9 and len(artist.get_facecolor()) == 9
        assert len(fig.axes) == 2                                # the map and one colorbar
        assert fig.axes[0].get_title() == "second depth"
        viewer.set_model_slice(None)
        fig.canvas.draw()
        assert len(fig.axes) == 1 and not fig.axes[0].collections


def test_run_returns_the_refined_cells():
    t = inverted()
    with tempfile.TemporaryDirectory() as d:
        tria = os.path.join(d, "m.tria")
        t.save(tria)
        rule = R.RefineRule(kind="box", levels=1, half_north_km=30, half_east_km=30, depth_bottom_km=50).asdict()
        res = R.run({"tria": tria, "rules": [rule]}, log=QUIET, write=False)
        assert len(R.plan_slice(res["cells"], 20e3)[0]) > 12 * 12


def _prm(path, box, age_s=0):
    with open(path, "w") as fh:
        fh.write(f"subsection Model\n\tset Active domain box = {box}\nend\n")
    when = os.path.getmtime(path) - age_s
    os.utime(path, (when, when))


def test_domain_box_comes_from_a_named_variant_and_prefers_the_plain_file():
    with tempfile.TemporaryDirectory() as d:
        below = os.path.join(d, "ZK")
        os.mkdir(below)
        tria = os.path.join(below, "m.tria")
        _prm(os.path.join(d, "input_inversion-old.prm"), "1, 1, 1, 2, 2, 2", age_s=100)
        _prm(os.path.join(d, "input_inversion-Z.prm"), "3, 3, 3, 4, 4, 4")
        box, source = R.guess_domain_box(tria)                   # a level up; the newest variant
        assert box.startswith("3,") and os.path.basename(source) == "input_inversion-Z.prm"
        _prm(os.path.join(d, "input_inversion.prm"), "5, 5, 5, 6, 6, 6", age_s=1000)
        assert R.guess_domain_box(tria)[0].startswith("5,")      # the plain name, though older
        _prm(os.path.join(below, "input_inversion-near.prm"), "7, 7, 7, 8, 8, 8", age_s=2000)
        assert R.guess_domain_box(tria)[0].startswith("7,")      # the nearest folder first
        with open(os.path.join(below, "input_inversion-nobox.prm"), "w") as fh:
            fh.write("nothing here\n")
        assert R.guess_domain_box(tria)[0].startswith("7,")      # one with no box is skipped


def _setup_project(d, lon, lat, name="p.mtinv.json"):
    """A saved setup project on the sample EDIs, in a survey-centred frame at (lon, lat)."""
    from mtinv_gui.project import Project

    p = Project()
    p.survey.edi_dir = os.path.join(os.path.dirname(ROOT), "edis")
    p.projection.proj_kind, p.projection.local_proj = "local", "tmerc"
    p.projection.origin_lon, p.projection.origin_lat = lon, lat
    return p.save(os.path.join(d, name))


def test_setup_project_frame_is_checked_against_the_receivers():
    from mtinv_gui.backends import edi_io
    from mtinv_gui.plotting.backends import georef as G
    from mtinv_gui.project import Project

    with tempfile.TemporaryDirectory() as d:
        stations = edi_io.read_edi_headers(os.path.join(os.path.dirname(ROOT), "edis"))
        lon = float(np.mean([s.lon for s in stations]))
        lat = float(np.mean([s.lat for s in stations]))
        built = _setup_project(d, lon, lat, "built.mtinv.json")
        gr = G.from_mtinv_project(built)[0]
        xy = np.c_[R.stations_in_model_frame(stations, gr)[1], np.zeros(len(stations))]
        rec = os.path.join(d, "receivers_projected.csv")
        with open(rec, "w") as fh:
            for s, (n, e, z) in zip(stations, xy):
                fh.write(f"Dipole {s.station_id} 1 {n:e} {e:e} {z:e}\n")

        spec = {"frame": "project", "frame_project": built, "receivers": rec}
        got, note = R.resolve_georef(spec)
        assert got.crs == gr.crs and f"checked against {len(stations)} receivers" in note, note

        # the same stations, but the project was re-centred after the model was built
        moved = _setup_project(d, lon + 1.0, lat, "moved.mtinv.json")
        try:
            R.resolve_georef({**spec, "frame_project": moved})
            raise AssertionError("a project in another frame was accepted")
        except ValueError as exc:
            assert "doesn't fit this model" in str(exc), exc

        # reused for other stations: nothing in common with these receivers
        with open(rec) as fh:
            renamed = fh.read().replace("Dipole ", "Dipole other_")
        other = os.path.join(d, "other.csv")
        with open(other, "w") as fh:
            fh.write(renamed)
        try:
            R.resolve_georef({**spec, "receivers": other})
            raise AssertionError("a project with none of the receivers was accepted")
        except ValueError as exc:
            assert "none of its stations" in str(exc), exc

        # it can't tell: use the frame, and say so
        assert "not checked: no receivers" in R.resolve_georef({**spec, "receivers": ""})[1]
        gone = Project.load(built)
        gone.survey.edi_dir = os.path.join(d, "no_such_folder")
        path = gone.save(os.path.join(d, "gone.mtinv.json"))
        assert "not checked: can't read" in R.resolve_georef({**spec, "frame_project": path})[1]


def test_dem_check_passes_the_right_dem():
    t, surf = _hilly_surface()
    with tempfile.TemporaryDirectory() as d:
        check = R.check_dem_frame(surf, write_dem(os.path.join(d, "a.npz"), _hills))
        assert check.status == "ok" and not check.is_problem, check.message
        assert check.r > 0.9 and check.shift_m is None


def test_dem_check_catches_a_shifted_dem():
    t, surf = _hilly_surface()
    with tempfile.TemporaryDirectory() as d:
        # the same terrain, but placed 12 km north and 8 km west of where the model has it
        check = R.check_dem_frame(surf, write_dem(os.path.join(d, "a.npz"),
                                                  lambda n, e: _hills(np.asarray(n) - 12e3, np.asarray(e) + 8e3)))
        assert check.status == "offset" and check.is_problem, check.message
        # to line up, the DEM has to move back: south and east
        assert abs(check.shift_m[0] + 12e3) < 6e3 and abs(check.shift_m[1] - 8e3) < 6e3, check.shift_m
        assert "south" in check.message and "east" in check.message, check.message
        # and doing what it says really does fit better
        xs = np.linspace(-500e3, 500e3, 1001)
        east, north = np.meshgrid(xs, xs)
        moved = np.dstack([east + check.shift_m[1], north + check.shift_m[0],
                           _hills(north - 12e3, east + 8e3)])
        path = os.path.join(d, "moved.npz")
        np.savez_compressed(path, moved)
        assert R.check_dem_frame(surf, path).rms_m < check.rms_m / 1.25


def test_dem_check_catches_swapped_axes():
    t, surf = _hilly_surface()
    with tempfile.TemporaryDirectory() as d:
        check = R.check_dem_frame(surf, write_dem(os.path.join(d, "a.npz"), lambda n, e: _hills(e, n)))
        assert check.status == "swapped", check.message


def test_dem_check_catches_another_area():
    t, surf = _hilly_surface()

    def elsewhere(n, e):
        n, e = np.asarray(n, dtype=float), np.asarray(e, dtype=float)
        return 200.0 + 180.0 * np.sin(n / 5e3 + 2.0) * np.sin(e / 6e3)

    with tempfile.TemporaryDirectory() as d:
        check = R.check_dem_frame(surf, write_dem(os.path.join(d, "a.npz"), elsewhere))
        assert check.is_problem and check.status in ("mismatch", "offset"), check.message


def test_dem_check_with_no_overlap_or_no_relief():
    t, surf = _hilly_surface()
    with tempfile.TemporaryDirectory() as d:
        xs = np.linspace(-100e3, 100e3, 101)
        east, north = np.meshgrid(xs, xs)
        far = os.path.join(d, "far.npz")
        np.savez_compressed(far, np.dstack([east + 1e8, north, _hills(north, east)]))
        assert R.check_dem_frame(surf, far).status == "no_overlap"
        flat_mesh = R.surface_faces(inverted(lambda x, y: np.full_like(np.asarray(x, dtype=float), 300.0)))
        check = R.check_dem_frame(flat_mesh, write_dem(os.path.join(d, "a.npz"), _hills))
        assert check.status == "unknown" and not check.is_problem


def test_run_reports_a_bad_dem_and_a_good_one_stays_quiet():
    with tempfile.TemporaryDirectory() as d:
        tria, mats, rec, _n = _files(d)
        good = write_dem(os.path.join(d, "good.npz"))
        res = R.run({"tria": tria, "materials": mats, "receivers": rec, "refit_dem": good}, log=QUIET, write=False)
        assert not res["dem_check"].is_problem, res["dem_check"].message
        res = R.run({"tria": tria, "materials": mats, "receivers": rec}, log=QUIET, write=False)
        assert res["dem_check"] is None
        said = []
        bad = write_dem(os.path.join(d, "bad.npz"), lambda x, y: elevation(y, x) * 0.2 + 700.0)
        res = R.run({"tria": tria, "materials": mats, "receivers": rec, "refit_dem": bad},
                    log=said.append, write=False)
        assert res["dem_check"].is_problem and any(s.startswith("WARNING") for s in said)


def test_refit_keeps_sea_flat_and_refuses_real_ocean():
    def coast(x, y):
        return np.where(np.asarray(y) < -100e3, -2000.0, elevation(x, y))

    with tempfile.TemporaryDirectory() as d:
        dem = write_dem(os.path.join(d, "dem.npz"), coast)
        t = inverted()
        R.refit_topography(t, dem, log=QUIET)
        v = R.surface_faces(t).vertices()
        offshore = v[v[:, 1] < -150e3]
        assert len(offshore) and np.abs(offshore[:, 2]).max() < 1e-6
        try:
            R.refit_topography(t, dem, top_ids=[AIR, 3], log=QUIET)
        except ValueError as exc:
            assert "sea cells" in str(exc)
        else:
            raise AssertionError("real ocean should be refused")


# ---- receivers -------------------------------------------------------------- #
def _stations(n=30):
    xy = np.random.default_rng(3).uniform(-200e3, 200e3, (n, 2))
    xy = np.r_[xy, [[0.0, 0.0], [40e3, 40e3], [40e3, 7e3]]]    # a vertex, a cell edge
    return [f"s{i}" for i in range(len(xy))], xy


def _lines_gap(v, lo, hi, parts=2 ** R.FACE_LEVELS):
    """Distance from v to the nearest of the k / parts lines of [lo, hi]."""
    lines = lo + (hi - lo) * np.arange(parts + 1) / parts
    return np.abs(lines - v).min()


def test_off_cell_faces_moves_only_stations_on_faces_or_split_lines():
    fp = np.array([[0.0, 8000.0, -4000.0, 4000.0]] * 6)        # one 8 km cell, lines every 1 km
    xy = np.array([[0.0, 500.0],          # on the north0 face
                   [8000.0, 500.0],       # on the north1 face
                   [3000.4, 500.0],       # 0.4 m off a split line (3 km)
                   [2500.0, -4000.0],     # on the east0 face
                   [2500.0, 0.0],         # on the middle line, east
                   [2500.0, 1500.0]])     # clear of everything
    out, moved = R.off_cell_faces(xy, fp)
    assert [k for k, _dn, _de in moved] == [0, 1, 2, 3, 4]
    assert out[0, 0] == 2.0 and out[1, 0] == 7998.0                  # into the cell
    assert out[3, 1] == -3998.0
    assert abs(out[2, 0] - 3002.0) < 1e-9                             # away from the line it was near
    assert np.allclose(out[5], xy[5]) and np.allclose(out[:, 1][[0, 1, 2]], 500.0)  # one axis only
    for (x, y), (n0, n1, e0, e1) in zip(out, fp):
        assert _lines_gap(x, n0, n1) >= R.FACE_GAP_M and _lines_gap(y, e0, e1) >= R.FACE_GAP_M
        assert n0 < x < n1 and e0 < y < e1


def test_run_moves_a_station_off_a_cell_face():
    """A station exactly on a vertical face could be lost by GoFEM ("Not all cells for receiver
    points were found"): the build moves it into a cell and says so."""
    with tempfile.TemporaryDirectory() as d:
        tria, mats, rec, names = _files(d)          # includes stations on a vertex and on an edge
        result = R.run({"tria": tria, "materials": mats, "receivers": rec, "rules": []},
                       os.path.join(d, "out"), log=QUIET, write=True)
        moved = {n for n, _dn, _de in result["off_face"]}
        assert {names[-3], names[-2], names[-1]} <= moved               # (0, 0), (40, 40) km, (40, 7) km
        t = R.load_tria(result["paths"]["tria"])
        surf = R.surface_faces(t)
        got_names, xyz = R.read_receivers(result["paths"]["receivers"])
        fp = R.surface_footprints(surf, xyz[:, :2])
        for (x, y), (n0, n1, e0, e1) in zip(xyz[:, :2], fp):
            assert _lines_gap(x, n0, n1) >= R.FACE_GAP_M - 1e-6
            assert _lines_gap(y, e0, e1) >= R.FACE_GAP_M - 1e-6
        assert not R.verify_receivers(t, got_names, xyz)


def test_fresh_build_projection_moves_a_station_off_a_cell_face():
    from mtinv_gui.backends import mesh as M

    t = build()                                              # 40 km base cells, faces at 40 km steps
    stations = [[40e3, 7e3, 0.0], [12e3, 13e3, 0.0]]         # on a north face; clear
    logs = []
    out = np.asarray(M.project_receivers(t, stations, names=["on_face", "clear"], log=logs.append))
    assert abs(out[0, 0] - 40e3) >= R.FACE_GAP_M and abs(out[0, 0] - 40e3) < 3.0
    assert out[1, 0] == 12e3 and out[1, 1] == 13e3
    assert len(logs) == 1 and "on_face" in logs[0]


def test_projection_matches_pygofem_and_handles_per_cell_ids():
    t = inverted()
    R.apply_rule(t, R.RefineRule(kind="box", levels=2, half_north_km=90, half_east_km=90,
                                 depth_bottom_km=20), R.surface_faces(t), log=QUIET)
    names, xy = _stations()
    mine = R.project_to_surface(R.surface_faces(t), names, xy)

    points = [[float(a), float(b), 0.0] for a, b in mine[:, :2]]
    mapping = dealii.MappingQGeneric(dim=3, spacedim=3, degree=1)
    raw = project_points_on_interface(t, points, AIR, EARTH, mapping)
    assert all(p[2] == 0.0 for p in raw)                       # upstream silently does nothing

    saved = [c.material_id for c in t.active_cells()]
    for c in t.active_cells():
        if c.material_id != AIR:
            c.material_id = EARTH
    theirs = project_points_on_interface(t, points, AIR, EARTH, mapping)
    for c, i in zip(t.active_cells(), saved):
        c.material_id = i
    assert np.abs(np.asarray(theirs)[:, 2] - mine[:, 2]).max() < 1e-3


def test_verify_receivers_passes_correct_and_catches_wrong():
    t = inverted()
    names, xy = _stations()
    surf = R.surface_faces(t)
    good = R.project_to_surface(surf, names, xy)
    good[:, 2] += R.RECEIVER_OFFSET_M
    assert R.verify_receivers(t, names, good) == []
    high = good.copy()
    high[:, 2] -= 50.0                                         # in the air
    assert len(R.verify_receivers(t, names, high)) == len(names)
    deep = good.copy()
    deep[:, 2] += 50.0                                         # buried
    assert len(R.verify_receivers(t, names, deep)) == len(names)


def test_stations_off_the_mesh_are_an_error():
    t = inverted()
    try:
        R.project_to_surface(R.surface_faces(t), ["far"], [[1e8, 0.0]])
    except ValueError as exc:
        assert "far" in str(exc)
    else:
        raise AssertionError("no error for a station off the mesh")


def test_merge_stations():
    old = np.array([[0.0, 0.0], [1000.0, 0.0]])
    new = np.array([[1000.2, 0.0], [5000.0, 0.0]])
    names, xy, is_new, notes = R.merge_stations(["a", "b"], old, ["b", "c"], new, "add")
    assert names == ["a", "b", "c"] and list(is_new) == [False, False, True] and len(notes) == 1
    assert xy[1, 0] == 1000.0                                  # the existing one is kept
    names, _xy, is_new, _n = R.merge_stations(["a", "b"], old, ["b", "c"], new, "replace")
    assert names == ["b", "c"] and is_new.all()
    try:
        R.merge_stations(["a"], old[:1], ["a"], [[9000.0, 0.0]], "add")
    except ValueError as exc:
        assert "a is already" in str(exc).replace("station ", "")
    else:
        raise AssertionError("no error for a name in two places")


def test_stations_in_model_frame_axes():
    from types import SimpleNamespace as NS

    from mtinv_gui.plotting.backends.georef import Georef

    st = [NS(station_id="a", lon=-3.0, lat=0.5)]
    names, xy = R.stations_in_model_frame(st, Georef("EPSG:32631", 500000.0, 50000.0))
    assert names == ["a"]
    from pyproj import Transformer

    e, n = Transformer.from_crs(4326, "EPSG:32631", always_xy=True).transform(-3.0, 0.5)
    assert abs(xy[0, 0] - (n - 50000.0)) < 1e-6 and abs(xy[0, 1] - (e - 500000.0)) < 1e-6


# ---- materials + the whole job ---------------------------------------------- #
def test_guess_fixed_ids():
    assert R.guess_fixed_ids([1, 7, 4491921, 4491922, 4500000]) == [1, 7]
    assert R.guess_fixed_ids([1, 2, 3, 4]) == [1]
    assert R.guess_fixed_ids([1, 5, 6, 104, 200000, 200001]) == [1, 5, 6, 104]


def _files(d):
    t = inverted()
    tria = os.path.join(d, "inv.tria")
    t.save(tria)
    mats = os.path.join(d, "inv.materials")
    with open(mats, "w") as fh:
        fh.write("iso\n")
        for i in sorted(ids_of(t)):
            fh.write(f"{i}\tlayer1\t{1e-3 + i * 1e-9:.6e}\t0\n")
    rec = os.path.join(d, "receivers_projected.csv")
    names, xy = _stations(10)
    p = R.project_to_surface(R.surface_faces(t), names, xy)
    with open(rec, "w") as fh:
        for n, (x, y, z) in zip(names, p):
            fh.write("Dipole %s 1 %0.6e %0.6e %0.6e\n" % (n, x, y, z + R.RECEIVER_OFFSET_M))
    return tria, mats, rec, names


def test_run_area_refinement_writes_a_model_gofem_can_use():
    with tempfile.TemporaryDirectory() as d:
        tria, mats, rec, names = _files(d)
        out = os.path.join(d, "out")
        rule = R.RefineRule(kind="box", levels=3, min_size_km=8, half_north_km=100, half_east_km=100,
                            depth_bottom_km=20)
        res = R.run({"tria": tria, "materials": mats, "receivers": rec, "rules": [rule.asdict()],
                     "fixed_ids": [AIR]}, out, log=QUIET)
        p = res["paths"]
        assert res["n_cells"] > res["n_cells_before"] and res["receivers_moved_m"] < 1e-2
        reloaded = R.load_tria(p["tria"])
        assert reloaded.n_active_cells() == res["n_cells"]
        assert ids_of(reloaded) == ids_of(R.load_tria(tria))
        assert open(p["materials"]).read() == open(mats).read()
        got, xyz = R.read_receivers(p["receivers"])
        assert got == names and R.verify_receivers(reloaded, got, xyz) == []
        # the mask: one line per cell, keyed to the refined mesh, air fixed
        rows = np.loadtxt(p["mask"], dtype=int)
        cells = reloaded.active_cells()
        assert len(rows) == len(cells)
        assert all((r[0], r[1]) == (c.level(), c.index()) and r[2] == (0 if c.material_id == AIR else 1)
                   for r, c in zip(rows[::500], cells[::500]))
        assert open(p["tria"], "rb").read(40).find(b"serialization") >= 0


def test_run_estimate_writes_nothing():
    with tempfile.TemporaryDirectory() as d:
        tria, mats, rec, _n = _files(d)
        out = os.path.join(d, "out")
        rule = R.RefineRule(kind="box", levels=1, half_north_km=50, half_east_km=50, depth_bottom_km=10)
        spec = {"tria": tria, "materials": mats, "receivers": rec, "rules": [rule.asdict()]}
        est = R.run(spec, out, log=QUIET, write=False)
        assert not os.path.exists(out)
        assert R.run(spec, out, log=QUIET)["n_cells"] == est["n_cells"]


def test_run_refuses_materials_from_another_model():
    with tempfile.TemporaryDirectory() as d:
        tria, mats, _rec, _n = _files(d)
        with open(mats, "w") as fh:
            fh.write("iso\n1\tair\t1e-8\t0\n")
        try:
            R.run({"tria": tria, "materials": mats}, d, log=QUIET, write=False)
        except ValueError as exc:
            assert "aren't in the .materials" in str(exc)
        else:
            raise AssertionError("no error for mismatched materials")


def test_run_stops_on_receivers_from_another_model_after_refit():
    """A refit moves the surface; every station is re-placed and verified afterwards."""
    with tempfile.TemporaryDirectory() as d:
        tria, mats, rec, names = _files(d)
        dem = write_dem(os.path.join(d, "dem.npz"), lambda x, y: elevation(x, y) + 300.0)
        res = R.run({"tria": tria, "materials": mats, "receivers": rec, "refit_dem": dem},
                    os.path.join(d, "o"), log=QUIET)
        old = R.read_receivers(rec)[1]
        new = R.read_receivers(res["paths"]["receivers"])[1]
        assert np.abs((old[:, 2] - new[:, 2]) - 300.0).max() < 25.0     # raised by ~300 m
        assert R.verify_receivers(R.load_tria(res["paths"]["tria"]), names, new) == []


def test_run_add_and_replace_stations():
    from mtinv_gui.backends import edi_io
    from mtinv_gui.plotting.backends.georef import Georef

    sample = os.path.join(os.path.dirname(ROOT), "edis")
    if not os.path.isdir(sample):
        print("     (sample EDIs not present -- skipped)")
        return
    stations = edi_io.read_edi_headers(sample)
    lon = np.mean([s.lon for s in stations])
    lat = np.mean([s.lat for s in stations])
    from pyproj import Transformer

    zone = int((lon + 180) // 6) + 1
    crs = f"EPSG:{32600 + zone}"
    x0, y0 = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    with tempfile.TemporaryDirectory() as d:
        tria, mats, rec, old_names = _files(d)
        rule = R.RefineRule(kind="stations", stations="new", levels=2, radius_km=30, depth_bottom_km=20)
        base = {"tria": tria, "materials": mats, "receivers": rec, "rules": [rule.asdict()],
                "stations_edi": sample, "frame": "manual", "frame_crs": crs, "frame_x0": x0, "frame_y0": y0}
        add = R.run({**base, "station_mode": "add"}, os.path.join(d, "a"), log=QUIET)
        names, xyz = R.read_receivers(add["paths"]["receivers"])
        assert names[:len(old_names)] == old_names and len(names) == len(old_names) + len(stations)
        assert int(add["is_new"].sum()) == len(stations)
        new = add["is_new"]
        assert (add["station_size_m"][new] < add["station_size_before_m"][new]).all()
        assert R.verify_receivers(R.load_tria(add["paths"]["tria"]), names, xyz) == []

        rep = R.run({**base, "station_mode": "replace"}, os.path.join(d, "r"), log=QUIET)
        names, _xyz = R.read_receivers(rep["paths"]["receivers"])
        assert len(names) == len(stations) and not set(names) & set(old_names)
        assert rep["receivers_moved_m"] is None


# ---- GoFEM files + project state --------------------------------------------- #
def _project(d):
    from mtinv_gui.project import Project

    p = Project(workdir=d)
    p.gofem.active_domain_box = [-1e5, -1e5, -4e3, 1e5, 1e5, 1e5]
    return p


def _prm_value(path, name):
    for line in open(path):
        parts = line.split("=", 1)
        if len(parts) == 2 and parts[0].split()[-1:] and " ".join(parts[0].split()[1:]) == name:
            return parts[1].strip()
    return None


def test_gofem_files_keeping_the_existing_data():
    from mtinv_gui.backends import gofem_files as F

    with tempfile.TemporaryDirectory() as d:
        tria, mats, rec, names = _files(d)
        gdat = os.path.join(d, "old.gdat")
        with open(gdat, "w") as fh:
            fh.write("# DataType Frequency Source Receiver Value Error\n")
            fh.writelines(f"RealZxy 1.000000e-01 Plane_wave {n} 1.0e-03 1.0e-04\n" for n in names)
        freq = os.path.join(d, "frequencies")
        np.savetxt(freq, [1.0, 0.1], header="frequency (Hz)")
        spec = {"tria": tria, "materials": mats, "receivers": rec, "existing_data": gdat,
                "existing_frequencies": freq, "stem": "m2"}
        out = os.path.join(d, "o")
        res = R.run(spec, out, log=QUIET)
        got = F.write_refined_inputs(_project(d), spec, res, log=QUIET)
        assert open(got["data"]).read() == open(gdat).read() and got["n_frequencies"] == 2
        for prm, tria_key, mat_key in ((got["inversion_prm"], "Inversion model definition file",
                                        "Inversion materials definition file"),
                                       (got["modeling_prm"], "Model definition file",
                                        "Materials definition file")):
            assert _prm_value(prm, tria_key) == "m2.tria"
            assert _prm_value(prm, mat_key) == "m2.materials"
            assert _prm_value(prm, "Stations file") == "receivers_refined-projected.csv"
            assert _prm_value(prm, "Frequencies file") == "frequencies"
        assert _prm_value(got["inversion_prm"], "Inversion input data") == os.path.basename(got["data"])
        assert not (_prm_value(got["inversion_prm"], "Active domain mask") or "").endswith(".mask")
        assert got["box"] == [-1e5, -1e5, -4e3, 1e5, 1e5, 1e5]
        mine = F.write_refined_inputs(_project(d), {**spec, "active_domain_box": "-2e6, -2e6, -4000, 2e6, 2e6, 410000"},
                                      res, log=QUIET)
        assert mine["box"] == [-2e6, -2e6, -4000.0, 2e6, 2e6, 410000.0]
        assert _prm_value(mine["inversion_prm"], "Active domain box").replace(" ", "").startswith("-2000000")
        no_box = _project(d)
        no_box.gofem.active_domain_box = []
        try:
            F.write_refined_inputs(no_box, spec, res, log=QUIET)
        except ValueError as exc:
            assert "Active Domain Box" in str(exc)
        else:
            raise AssertionError("must not fall back to a default box")
        try:
            F.write_refined_inputs(_project(d), {**spec, "existing_data": ""}, res, log=QUIET)
        except ValueError as exc:
            assert "existing data file" in str(exc)
        else:
            raise AssertionError("no error without the existing data")


def test_gofem_files_new_data_alongside_and_replacing():
    from pyproj import Transformer

    from mtinv_gui.backends import edi_io
    from mtinv_gui.backends import gofem_files as F

    sample = os.path.join(os.path.dirname(ROOT), "edis")
    if not (os.path.isdir(sample) and config.mtpy_available()):
        print("     (sample EDIs or mtpy not present -- skipped)")
        return
    stations = edi_io.read_edi_headers(sample)
    lon, lat = np.mean([s.lon for s in stations]), np.mean([s.lat for s in stations])
    crs = f"EPSG:{32600 + int((lon + 180) // 6) + 1}"
    x0, y0 = Transformer.from_crs(4326, crs, always_xy=True).transform(lon, lat)
    with tempfile.TemporaryDirectory() as d:
        tria, mats, rec, old_names = _files(d)
        gdat = os.path.join(d, "old.gdat")
        with open(gdat, "w") as fh:
            fh.write("# DataType Frequency Source Receiver Value Error\n")
            fh.writelines(f"RealZxy 1.234500e+03 Plane_wave {n} 1.0e-03 1.0e-04\n" for n in old_names)
        freq = os.path.join(d, "frequencies")
        np.savetxt(freq, [1234.5], header="frequency (Hz)")
        base = {"tria": tria, "materials": mats, "receivers": rec, "stations_edi": sample,
                "frame": "manual", "frame_crs": crs, "frame_x0": x0, "frame_y0": y0,
                "existing_data": gdat, "existing_frequencies": freq}
        project = _project(d)

        spec = {**base, "station_mode": "add"}
        res = R.run(spec, os.path.join(d, "a"), log=QUIET)
        got = F.write_refined_inputs(project, spec, res, log=QUIET)
        rows = [line.split() for line in open(got["data"]) if not line.startswith("#")]
        assert {r[3] for r in rows} == set(res["names"])
        assert any(r[1] == "1.234500e+03" for r in rows)
        freqs = np.atleast_1d(np.loadtxt(got["frequencies"]))
        assert np.isclose(freqs, 1234.5).sum() == 1 and len(freqs) > 1 and (np.diff(freqs) < 0).all()

        spec = {**base, "station_mode": "replace"}
        res = R.run(spec, os.path.join(d, "r"), log=QUIET)
        got = F.write_refined_inputs(project, spec, res, log=QUIET)
        rows = [line.split() for line in open(got["data"]) if not line.startswith("#")]
        assert {r[3] for r in rows} == {s.station_id for s in stations}
        assert not np.isclose(np.atleast_1d(np.loadtxt(got["frequencies"])), 1234.5).any()


def test_missing_gofem_inputs_are_caught_before_any_work():
    from mtinv_gui.backends import gofem_files as F

    with tempfile.TemporaryDirectory() as d:
        ok = {"receivers": "r.csv", "existing_data": __file__, "existing_frequencies": __file__}
        assert F.check_refined_inputs(_project(d), ok)
        cases = [({"existing_data": ""}, "existing data"), ({"existing_frequencies": "nope"}, "can't find"),
                 ({"receivers": ""}, "stations"), ({"station_mode": "add"}, "EDIs"),
                 ({"active_domain_box": "1, 2, 3"}, "six numbers")]
        for change, words in cases:
            try:
                F.check_refined_inputs(_project(d), {**ok, **change})
            except ValueError as exc:
                assert words in str(exc), (change, exc)
            else:
                raise AssertionError(f"no error for {change}")
        replace = {"station_mode": "replace", "stations_edi": "x"}
        assert F.check_refined_inputs(_project(d), replace)     # needs no existing data


def test_merge_frequencies():
    from mtinv_gui.backends.gofem_files import merge_frequencies

    assert merge_frequencies([1.0, 0.1], [0.1000001, 10.0]) == [10.0, 1.0, 0.1000001]


def test_prm_defaults_still_point_at_the_mesh_tab_files():
    from mtinv_gui.backends import gofem_files as F

    with tempfile.TemporaryDirectory() as d:
        p = _project(d)
        F.write_inversion_prm(p, os.path.join(d, "i.prm"), "x.gdat")
        F.write_modeling_prm(p, os.path.join(d, "m.prm"))
        assert _prm_value(os.path.join(d, "i.prm"), "Inversion model definition file") == "model.tria"
        assert _prm_value(os.path.join(d, "i.prm"), "Active domain mask") == "model.mask"
        assert _prm_value(os.path.join(d, "m.prm"), "Materials definition file") == "model.materials"


def test_project_keeps_refine_state_and_old_projects_load():
    import json

    from mtinv_gui.project import Project

    with tempfile.TemporaryDirectory() as d:
        p = Project(name="p", workdir=d)
        p.refine.tria = "a.tria"
        p.refine.rules.append(R.RefineRule(kind="stations").asdict())
        f = p.save()
        q = Project.load(f)
        assert q.refine.tria == "a.tria" and q.refine.rules[0]["kind"] == "stations"
        raw = json.load(open(f))
        del raw["refine"]
        json.dump(raw, open(f, "w"))
        assert Project.load(f).refine.station_mode == "none"     # saved before this tab existed
        s = R.spec_from_info(q.refine)
        assert s["refit_dem"] == "" and s["fixed_ids"] is None and s["blend_km"] is None


def test_overwrite_guard_and_companions():
    with tempfile.TemporaryDirectory() as d:
        tria, mats, rec, _n = _files(d)
        os.rename(tria, os.path.join(d, "model_refined.tria"))
        os.rename(mats, os.path.join(d, "model_refined.materials"))
        tria = os.path.join(d, "model_refined.tria")
        mats = os.path.join(d, "model_refined.materials")
        guess = R.guess_companions(tria)
        assert guess == {"materials": mats, "receivers": rec}
        assert R.guess_domain_box(tria) == ("", "")
        with open(os.path.join(d, "input_inversion.prm"), "w") as fh:
            fh.write("subsection Model\n\tset Active domain box = -2000000, -2000000, -4000, 2000000, 2000000, 410000\nend\n")
        assert R.guess_domain_box(tria)[0].startswith("-2000000")
        assert tria in R.existing_outputs(d)
        try:
            R.run({"tria": tria, "materials": mats}, d, log=QUIET)
        except ValueError as exc:
            assert "overwrite the input" in str(exc)
        else:
            raise AssertionError("input model would have been overwritten")


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
