"""Mapping a model onto another mesh (backends/remap.py), on the synthetic meshes of
test_refine: 12 x 12 cells of 40 km, fitted to a smooth topography."""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_refine as T  # noqa: E402

from mtinv_gui.backends import remap as M  # noqa: E402
from mtinv_gui.plotting.backends import georef as G  # noqa: E402

R = T.R
QUIET = T.QUIET
CRS = "EPSG:32610"


def _source(field):
    """An inverted-looking model: own id per earth cell, log10 conductivity = field(E, N) in CRS."""
    t = T.inverted()
    gr = G.manual(CRS, 500000.0, 4000000.0)
    mats = {T.AIR: R.Material(T.AIR, "air", 1e-8)}
    for c in t.active_cells():
        if c.material_id != T.AIR:
            p = c.center()
            mats[c.material_id] = R.Material(c.material_id, "ground", 10.0 ** field(p.y + gr.x0, p.x + gr.y0))
    return t, mats, gr


def _target():
    t = T.build()
    return t, {T.AIR: R.Material(T.AIR, "air", 1e-8), 2: R.Material(2, "ground", 1e-2)}


def _filled(t, mats, gr):
    """(E, N, log10) of the target's filled cells."""
    out = []
    for c in t.active_cells():
        if c.material_id >= 1000:
            p = c.center()
            out.append((p.y + gr.x0, p.x + gr.y0, np.log10(mats[c.material_id].conductivity)))
    return np.array(out)


def test_axis_weight_is_overlap_without_smoothing_and_conserves_with_it():
    lo, hi = np.array([0.0, 1.0, 0.5]), np.array([1.0, 2.0, 1.5])
    w = M._axis_weight(lo, hi, np.zeros(3), np.ones(3), np.zeros(3))
    assert np.allclose(w, [1.0, 0.0, 0.5])
    # blurred, the source intervals tiling the line still cover the target exactly once
    edges = np.arange(-10.0, 11.0)
    s = np.full(len(edges) - 1, 0.7)
    w = M._axis_weight(edges[:-1], edges[1:], np.full(len(s), 0.2), np.full(len(s), 1.3), s)
    assert abs(w.sum() - 1.0) < 1e-12


def test_same_mesh_same_frame_copies_every_cell_exactly():
    src, smats, gr = _source(lambda e, n: -3.0 + 2.0 * np.sin(e / 70e3) * np.cos(n / 50e3))
    tgt, tmats = _target()
    res = M.remap(src, smats, gr, tgt, tmats, gr, smooth=0.0, log=QUIET)
    assert res["kept"] == 0
    want = {(round(c.center().x), round(c.center().y), round(c.center().z)): smats[c.material_id].conductivity
            for c in src.active_cells() if c.material_id != T.AIR}
    n = 0
    for c in tgt.active_cells():
        if c.material_id == T.AIR:
            continue
        key = (round(c.center().x), round(c.center().y), round(c.center().z))
        assert abs(np.log10(tmats[c.material_id].conductivity) - np.log10(want[key])) < 1e-9
        n += 1
    assert n == res["cells"] > 0


def test_frames_are_followed():
    """Steps on the source's cell faces; the target's frame is shifted by one cell east and
    one south, so each target cell sits wholly on one side of both steps only if the frame
    change is applied (ignoring it would put every target cell across them)."""
    def field(e, n):
        return np.where(e < 520000.0, -3.0, -1.0) + np.where(n < 3980000.0, 0.5, 0.0)

    src, smats, sgr = _source(field)
    tgt, tmats = _target()
    tgr = G.manual(CRS, 540000.0, 3960000.0)
    res = M.remap(src, smats, sgr, tgt, tmats, tgr, smooth=0.0, log=QUIET)
    assert res["kept"] > 0                                     # the shifted target reaches past the source
    e, n, v = _filled(tgt, tmats, tgr).T
    assert len(v) > 100
    assert np.abs(v - field(e, n)).max() < 1e-6


def test_frames_with_different_projections():
    """Target in another CRS: a target cell reads the source where its lon/lat is."""
    def field(e, n):
        return -2.0 + (e - 500000.0) / 400e3 + (n - 4000000.0) / 600e3       # smooth: sampling error only

    src, smats, sgr = _source(field)
    tgt, tmats = _target()
    from pyproj import Transformer

    tm = "+proj=tmerc +lat_0=36.1 +lon_0=-123.1 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    tgr = G.manual(tm, 0.0, 0.0)
    res = M.remap(src, smats, sgr, tgt, tmats, tgr, smooth=0.0, log=QUIET)
    assert res["cells"] > 100
    e_t, n_t, v = _filled(tgt, tmats, tgr).T
    e, n = Transformer.from_crs(tm, CRS, always_xy=True).transform(e_t, n_t)
    err = v - field(np.asarray(e), np.asarray(n))
    # cells are 40 km: a target cell straddles source cells, a gradient of 0.1 decade per cell
    assert np.abs(err).max() < 0.12 and abs(err.mean()) < 0.02


def test_smoothing_blurs_steps_and_stays_in_range():
    def field(e, n):
        return np.where(e < 520000.0, -3.0, -1.0) + 0.0 * n

    src, smats, gr = _source(field)
    sharp = _target()
    blur = _target()
    M.remap(src, smats, gr, *sharp, gr, smooth=0.0, log=QUIET)
    res = M.remap(src, smats, gr, *blur, gr, smooth=0.5, log=QUIET)
    v_sharp = _filled(sharp[0], sharp[1], gr)[:, 2]
    v_blur = _filled(blur[0], blur[1], gr)[:, 2]
    assert set(np.round(v_sharp, 6)) == {-3.0, -1.0}
    assert ((v_blur > -2.99) & (v_blur < -1.01)).sum() > 10       # ramps across the step
    assert res["log10"].min() >= -3.0 - 1e-9 and res["log10"].max() <= -1.0 + 1e-9


def test_run_writes_a_model_gofem_can_use():
    with tempfile.TemporaryDirectory() as d:
        src, smats, sgr = _source(lambda e, n: -2.0 + 0.0 * e)
        src_tria = os.path.join(d, "inv.tria")
        src.save(src_tria)
        with open(os.path.join(d, "inv.materials"), "w") as fh:
            fh.write("iso\n" + "".join(f"{m.id}\t{m.name}\t{m.conductivity:.6e}\t0\n" for m in smats.values()))
        tgt, tmats = _target()
        tgt_tria = os.path.join(d, "model.tria")
        tgt.save(tgt_tria)
        with open(os.path.join(d, "model.materials"), "w") as fh:
            fh.write("# id name conductivity permitivitty\niso\n1 air 1e-08 0\n2 ground 1e-02 0\n")
        with open(os.path.join(d, "receivers_projected.csv"), "w") as fh:
            fh.write("Dipole s1 1 1.000000e+04 2.000000e+04 -5.000000e+02\n")
        spec = {"tria": src_tria, "materials": os.path.join(d, "inv.materials"), "frame": "manual",
                "frame_crs": CRS, "frame_x0": 500000.0, "frame_y0": 4000000.0,
                "target_tria": tgt_tria, "target_materials": os.path.join(d, "model.materials"),
                "target_receivers": os.path.join(d, "receivers_projected.csv"), "target_frame": "manual",
                "target_frame_crs": CRS, "target_frame_x0": 500000.0, "target_frame_y0": 4000000.0,
                "smooth": 0.0, "mask_fixed_ids": [1]}
        out = os.path.join(d, "out")
        res = M.run(spec, out, log=QUIET)
        mats = R.read_materials(res["paths"]["materials"])
        t = R.load_tria(res["paths"]["tria"])
        assert T.ids_of(t) <= set(mats)
        assert {c.material_id for c in t.active_cells()} - {1} == set(res["new_ids"])
        assert all(abs(np.log10(mats[i].conductivity) + 2.0) < 1e-6 for i in res["new_ids"])
        assert os.path.isfile(os.path.join(out, "receivers_projected.csv"))
        with open(res["paths"]["mask"]) as fh:
            assert sum(1 for _ in fh) == t.n_active_cells()
        assert t.n_active_cells() == tgt.n_active_cells()        # the mesh itself is untouched


def test_the_tabs_settings_make_a_job_run_accepts():
    from mtinv_gui.project import RefineInfo

    info = RefineInfo(workflow="map", tria="a.tria", materials="a.materials", frame="file", frame_file="f.json",
                      target_tria="b.tria", target_materials="b.materials", target_frame="manual",
                      target_frame_crs=CRS, target_frame_x0=1.0, target_frame_y0=2.0, fill_ids=[2, 5],
                      map_smooth=0.1, map_depth="elevation", write_mask=True, fixed_ids=[1, 4])
    spec = M.spec_from_info(info)
    assert spec["target_frame_crs"] == CRS and spec["target_frame_x0"] == 1.0 and spec["frame_file"] == "f.json"
    assert spec["fill_ids"] == [2, 5] and spec["smooth"] == 0.1 and spec["depth"] == "elevation"
    assert spec["mask_fixed_ids"] == [1, 4]
    tspec = {k[len("target_"):]: v for k, v in spec.items() if k.startswith("target_")}
    gr, _note = R.resolve_georef(tspec)                       # what run() does with them
    assert (gr.crs, gr.x0, gr.y0) == (CRS, 1.0, 2.0)


if __name__ == "__main__":
    import traceback

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception:  # noqa: BLE001
                failed += 1
                print(f"FAIL  {name}")
                traceback.print_exc()
    sys.exit(1 if failed else 0)
