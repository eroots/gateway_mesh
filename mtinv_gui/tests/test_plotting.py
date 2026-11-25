"""Model-plotting backends (``mtinv_gui/plotting/backends``).

A small synthetic GoFEM-style model (deal.II axes: X = North, Y = East, Z down;
every cell owns its 8 points) with a known layout pins down the axis
conventions, georeferencing, slicing and GeoTIFF output.

Needs the ``mtinv`` environment (pyvista, pyproj, rasterio).
Runs standalone (``python tests/test_plotting.py``) or under pytest.
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

from mtinv_gui.plotting.backends import geotiff as GT  # noqa: E402
from mtinv_gui.plotting.backends import georef as G  # noqa: E402
from mtinv_gui.plotting.backends import render, slicing  # noqa: E402
from mtinv_gui.plotting.backends.model_source import GoFEMVtuSource  # noqa: E402
from mtinv_gui.plotting.backends.session import PlotSession  # noqa: E402
from mtinv_gui.plotting.project import PlotProject  # noqa: E402

CRS, X0, Y0 = "EPSG:32609", 500000.0, 6000000.0
RECEIVERS = {"S1": (1000.0, 2000.0), "S2": (-3000.0, -7000.0), "S3": (6000.0, 11000.0),
             "S4": (-8000.0, 15000.0)}           # name -> (x = North, y = East)


def _write_synthetic(d: str) -> str:
    """North x East x depth = 20 x 40 x 12 km, 1 km cells, air above z = 0.

    Earth is 1 ohm-m east of the origin and above 5 km depth, 100 ohm-m elsewhere.
    """
    import pyvista as pv

    xs = np.arange(-10e3, 10e3 + 1, 1e3)          # north
    ys = np.arange(-20e3, 20e3 + 1, 1e3)          # east
    zs = np.arange(-2e3, 10e3 + 1, 1e3)           # depth (down)
    grid = pv.RectilinearGrid(xs, ys, zs).cast_to_unstructured_grid()
    c = grid.cell_centers().points
    sigma = np.where((c[:, 1] > 0) & (c[:, 2] < 5e3), 1.0, 0.01)
    sigma[c[:, 2] < 0] = 1e-8
    grid.cell_data["conductivity"] = sigma
    grid = grid.separate_cells().cell_data_to_point_data(pass_cell_data=False)
    path = os.path.join(d, "synthetic.vtu")
    grid.save(path)
    with open(os.path.join(d, "receivers_projected.csv"), "w") as fh:
        for name, (xn, ye) in RECEIVERS.items():
            fh.write(f"Dipole {name} 1 {xn:e} {ye:e} {-0.0:e}\n")
    return path


def _stations_lonlat() -> dict:
    """The receivers' true lon/lat if the model sits at (X0, Y0) in CRS."""
    from pyproj import Transformer

    tr = Transformer.from_crs(CRS, 4326, always_xy=True)
    out = {}
    for name, (xn, ye) in RECEIVERS.items():
        out[name] = tr.transform(ye + X0, xn + Y0)
    return out


def _session(d: str) -> tuple[PlotSession, PlotProject]:
    pp = PlotProject()
    pp.model.model_path = _write_synthetic(d)
    s = PlotSession()
    s.load_model(pp.model)
    pp.georef.method = "manual"
    pp.georef.manual_crs, pp.georef.manual_x0, pp.georef.manual_y0 = CRS, X0, Y0
    s.solve_georef(pp.georef)
    return s, pp


# --------------------------------------------------------------------------- #
def test_gofem_axes_and_receivers():
    with tempfile.TemporaryDirectory() as d:
        m = GoFEMVtuSource().load(_write_synthetic(d))
        b = m.earth_bounds()
        assert b[:4] == (-20e3, 20e3, -10e3, 10e3), b        # x = east, y = north
        assert b[4] == -10e3 and abs(b[5]) < 1e-9, b         # z up, surface at 0
        assert m.n_earth_cells == 20 * 40 * 10
        rec = GoFEMVtuSource().read_receivers(os.path.join(d, "receivers_projected.csv"))
        assert rec.names[0] == "S1" and np.allclose(rec.xyz[0], [2000.0, 1000.0, 0.0])
        lo, hi = m.surface().z_range
        assert abs(lo) < 1e-6 and abs(hi) < 1e-6


def test_sampling_values():
    with tempfile.TemporaryDirectory() as d:
        s, _ = _session(d)
        x = np.array([5000.0, -5000.0, 5000.0, 500.0])
        y = np.array([0.0, 0.0, 0.0, 500.0])
        z = np.array([-1000.0, -1000.0, -6000.0, 0.0])        # last: exactly on the surface
        v = slicing.sample_points(s.model, x, y, z)
        assert np.allclose(v, [0.0, 2.0, 2.0, 0.0]), v
        assert np.isnan(slicing.sample_points(s.model, [0.0], [0.0], [500.0]))[0]   # air


def test_manual_and_fitted_georef():
    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        fit = G.fit_from_stations(s.receivers, _stations_lonlat(), CRS)
        assert abs(fit.georef.x0 - X0) < 1e-3 and abs(fit.georef.y0 - Y0) < 1e-3
        assert fit.rms_m < 1e-3 and fit.n_matched == 4
        best = G.auto_fit(s.receivers, _stations_lonlat())
        assert best.georef.crs == CRS, best.georef.crs
        wrong = G.fit_from_stations(s.receivers, _stations_lonlat(), "EPSG:32608")
        assert wrong.rms_m > 1.0                               # wrong zone shows up


def test_auto_fit_finds_a_national_grid():
    """A model built in a registered grid that isn't a UTM zone or survey-centred."""
    from pyproj import Transformer

    grid, x0, y0 = "EPSG:3979", -1500000.0, 900000.0            # NAD83(CSRS) atlas Lambert
    with tempfile.TemporaryDirectory() as d:
        s, _ = _session(d)
        tr = Transformer.from_crs(grid, 4326, always_xy=True)
        lonlat = {n: tr.transform(ye + x0, xn + y0) for n, (xn, ye) in RECEIVERS.items()}
        assert grid not in G.candidate_crs(lonlat) and grid in G.registered_crs(lonlat)
        fit = G.auto_fit(s.receivers, lonlat)
        assert fit.rms_m < 0.1, (fit.georef.crs, fit.rms_m)
        assert fit.georef.crs in ("EPSG:3978", grid), fit.georef.crs    # the two datums tie
        assert abs(fit.georef.x0 - x0) < 2.0 and abs(fit.georef.y0 - y0) < 2.0


def test_display_crs_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        pp.georef.display_crs = "EPSG:32608"
        s.apply_georef(pp.georef)
        assert not s.maptr.same
        x, y = np.array([0.0, 15000.0]), np.array([0.0, -8000.0])
        e, n = s.maptr.forward(x, y)
        x2, y2 = s.maptr.inverse(e, n)
        assert np.allclose(x, x2, atol=1e-6) and np.allclose(y, y2, atol=1e-6)
        assert abs(e[0] - X0) > 1e5                             # really a different zone


def test_plan_slices():
    with tempfile.TemporaryDirectory() as d:
        s, _ = _session(d)
        sl = slicing.plan_slice(s.model, s.maptr, 2500.0, "sea_level")
        assert sl.n == 20 * 40 and set(np.round(sl.values, 6)) == {0.0, 2.0}
        assert np.allclose(sl.bounds(), (X0 - 20e3, X0 + 20e3, Y0 - 10e3, Y0 + 10e3))
        assert slicing.plan_slice(s.model, s.maptr, 2500.0, "sea_level") is sl   # cached
        su = slicing.plan_slice(s.model, s.maptr, 2500.0, "surface")             # flat surface
        assert su.n > 0 and set(np.round(su.values, 6)) == {0.0, 2.0}
        deep = slicing.plan_slice(s.model, s.maptr, 7500.0, "sea_level")
        assert set(np.round(deep.values, 6)) == {2.0}


def test_crinkle_plan_slice():
    """Crinkle slices: whole cell footprints, no gaps, the cell below on a face,
    and the cell containing the depth on top where a warped column has two."""
    from matplotlib.path import Path

    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        for depth in (2500.0, 5000.0, 7500.0):
            ex = slicing.plan_slice(s.model, s.maptr, depth, "sea_level")
            cr = slicing.plan_slice(s.model, s.maptr, depth, "sea_level", crinkle=True)
            assert cr is slicing.plan_slice(s.model, s.maptr, depth, "sea_level", crinkle=True)
            assert cr.n == 20 * 40, (depth, cr.n)                     # one cell per column
            area = sum(0.5 * abs(np.cross(p[:-1] - p[0], p[1:] - p[0]).sum()) for p in cr.polys)
            assert abs(area - 20e3 * 40e3) < 1.0, area                # footprints tile the model
            assert np.allclose(np.sort(cr.values), np.sort(ex.values)), depth
        # 5 km is the face between the 1 and 100 ohm-m cells: the one below wins
        cr = slicing.plan_slice(s.model, s.maptr, 5000.0, "sea_level", crinkle=True)
        assert np.allclose(cr.values, 2.0)
        su = slicing.plan_slice(s.model, s.maptr, 2500.0, "surface", crinkle=True)
        assert su.n == 20 * 40
        # one column of two cells with a tilted boundary (-4 km on the west, -2 km on
        # the east): at 3 km depth both reach the depth; the lower one holds it at the centre
        import pyvista as pv

        from mtinv_gui.plotting.backends.model_source import ModelGrid

        xy = [(0, 0), (1000, 0), (1000, 1000), (0, 1000)]
        mid = [-4000.0, -2000.0, -2000.0, -4000.0]
        lower = [(x, y, -6000.0) for x, y in xy] + [(x, y, m) for (x, y), m in zip(xy, mid)]
        upper = [(x, y, m) for (x, y), m in zip(xy, mid)] + [(x, y, 0.0) for x, y in xy]
        g = pv.UnstructuredGrid(np.array([8, *range(8), 8, *range(8, 16)]), np.array([12, 12]),
                                np.array(lower + upper, dtype=float))
        g.cell_data["log10_rho"] = np.array([0.0, 2.0])          # lower 1, upper 100 ohm-m
        m = ModelGrid(g, np.zeros(2, dtype=bool))
        cr = slicing.crinkle_plan_slice(m, s.maptr, 3000.0)
        assert cr.n == 2 and list(cr.values) == [2.0, 0.0], cr.values     # lower drawn last
        cr = slicing.crinkle_plan_slice(m, s.maptr, 1000.0)                # only the upper cell
        assert list(cr.values) == [2.0]
        assert abs(sum(0.5 * abs(np.cross(p[1:] - p[0], p[:-1] - p[0]).sum())
                       for p in cr.polys) - 1e6) < 1e-3


def test_geotiff_export():
    import rasterio

    with tempfile.TemporaryDirectory() as d:
        s, _ = _session(d)
        grid = GT.RasterGrid.from_bounds(X0 - 20e3, X0 + 20e3, Y0 - 10e3, Y0 + 10e3, 500.0)
        assert (grid.nx, grid.ny) == (80, 40)
        paths = GT.export(s.model, s.maptr, [1000.0, 6000.0], grid, d)
        assert [os.path.basename(p) for p in paths] == ["depth_slice_1000m.tif",
                                                        "depth_slice_6000m.tif"]
        with rasterio.open(paths[0]) as src:
            a = src.read(1)
            assert src.crs.to_epsg() == 32609 and src.res == (500.0, 500.0)
            r, c = src.index(X0 + 5000.0, Y0 + 250.0)          # east half -> 1 ohm-m
            assert abs(a[r, c] - 0.0) < 1e-6
            r, c = src.index(X0 - 5000.0, Y0 + 250.0)          # west half -> 100 ohm-m
            assert abs(a[r, c] - 2.0) < 1e-6
        stack = GT.export(s.model, s.maptr, [1000.0, 6000.0], grid, d, layout="stack",
                          quantity="rho", prefix="stack")
        with rasterio.open(stack[0]) as src:
            assert src.count == 2 and abs(np.nanmax(src.read(2)) - 100.0) < 1e-3


def test_depth_helpers():
    assert GT.parse_depths("0, 100 250;1e3") == [0.0, 100.0, 250.0, 1000.0]
    assert GT.generate_depths(100, 1000, 3, "log") == [100.0, 316.22776601683796, 1000.0]
    assert GT.generate_depths(0, 1000, 3, "linear", 10) == [0.0, 500.0, 1000.0]
    assert GT.depth_tag(-130) == "-130" and GT.depth_tag(12.5) == "12p5"
    assert slicing.depth_label(-250, "sea_level", "m") == "250 m above sea level"


def test_render_and_project_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        pp.plan.graticule = True
        out = render.export_plan(os.path.join(d, "plan.png"), s, pp.plan, 2500.0)
        assert os.path.getsize(out) > 10_000
        f = pp.save(os.path.join(d, "t.mtplot.json"))
        q = PlotProject.load(f)
        assert q.georef.crs == CRS and q.plan.graticule and q.geotiff.extent.padding_m == 10000.0
        assert type(q.plan.extent).__name__ == "ExtentInfo"


def _poly_area(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def test_ns_ew_sections():
    from mtinv_gui.plotting.backends import sections as SEC
    from mtinv_gui.plotting.project import SectionInfo

    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        st = pp.section_style
        ns = SectionInfo(kind="ns", coord_m=X0 + 5000.0, start_m=Y0 - 8000.0, end_m=Y0 + 8000.0)
        data = SEC.prepare(s, ns, st)
        assert abs(data.profile.length - 16000.0) < 1e-6
        cut = data.cut
        cz = cut.centroids()[:, 1]
        assert np.allclose(cut.values[cz > -5000.0], 0.0) and np.allclose(cut.values[cz < -5000.0], 2.0)
        ew = SectionInfo(kind="ew", coord_m=Y0 + 500.0, start_m=X0 - 20000.0, end_m=X0 + 20000.0)
        c2 = SEC.prepare(s, ew, st).cut.centroids()
        v2 = SEC.prepare(s, ew, st).cut.values
        west = c2[:, 0] < 20000.0                                  # distance from the west end
        assert np.allclose(v2[west], 2.0)
        assert np.allclose(v2[~west & (c2[:, 1] > -5000.0)], 0.0)
        # reversing swaps the ends
        ew.reverse = True
        c3 = SEC.prepare(s, ew, st)
        east_first = c3.cut.centroids()[:, 0] < 20000.0
        assert np.allclose(c3.cut.values[east_first & (c3.cut.centroids()[:, 1] > -5000.0)], 0.0)


def test_bent_path_is_exact():
    """A path with bends must cover length x depth exactly: no gaps or overlaps."""
    from mtinv_gui.plotting.backends import sections as SEC
    from mtinv_gui.plotting.project import SectionInfo

    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        verts = [[X0 - 10000.0, Y0 - 5000.0], [X0 + 5000.0, Y0], [X0 + 10000.0, Y0 + 8000.0]]
        sec = SectionInfo(kind="path", vertices=verts)
        data = SEC.prepare(s, sec, pp.section_style)
        L = data.profile.length
        expect = np.hypot(15000.0, 5000.0) + np.hypot(5000.0, 8000.0)
        assert abs(L - expect) < 1e-6
        area = sum(_poly_area(p) for p in data.cut.polys)
        assert abs(area - L * 10000.0) / (L * 10000.0) < 1e-6, (area, L * 10000.0)
        sd, off = data.profile.project([X0 + 5000.0], [Y0])
        assert abs(sd[0] - np.hypot(15000.0, 5000.0)) < 1e-6 and off[0] < 1e-6


def test_station_section_and_interpolation():
    from mtinv_gui.plotting.backends import sections as SEC
    from mtinv_gui.plotting.project import SectionInfo

    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        st = pp.section_style
        st.depth_bottom_m, st.interp_nx, st.interp_nz = 9000.0, 120, 60
        for mode in ("polyline", "spline", "line"):
            sec = SectionInfo(kind="stations", stations=["S2", "s1", "S3.edi"],
                              station_path=mode, extend_m=1000.0)
            data = SEC.prepare(s, sec, st)
            assert sorted(data.stations.names) == ["S1", "S2", "S3"], (mode, data.stations.names)
            order = np.argsort(data.stations.s)
            assert [data.stations.names[i] for i in order] == ["S2", "S1", "S3"], mode
        st.show_interp = True
        for method in ("linear", "nearest", "gaussian"):
            st.interp_method = method
            data = SEC.prepare(s, sec, st)
            g = data.grid
            assert g.values.shape == (60, 120)
            assert np.isnan(g.values[g.z > 1.0]).all()             # air stays masked
            ok = g.values[~np.isnan(g.values)]
            assert ok.min() >= -1e-6 and ok.max() <= 2.0 + 1e-6, method
            assert SEC.prepare(s, sec, st).grid is g                # cached
        out = render.export_section(os.path.join(d, "sec.png"), s, sec, st)
        assert os.path.getsize(out) > 10_000
        out = render.export_index_map(os.path.join(d, "idx.png"), s, [sec], st)
        assert os.path.getsize(out) > 10_000


def test_sections_roundtrip():
    from mtinv_gui.plotting.backends import sections as SEC
    from mtinv_gui.plotting.project import SectionInfo

    pp = PlotProject()
    pp.sections = [SectionInfo(label="A", kind="ns", coord_m=1.0),
                   SectionInfo(label="B", kind="stations", stations=["S1", "S2"])]
    pp.section_style.show_interp = True
    with tempfile.TemporaryDirectory() as d:
        q = PlotProject.load(pp.save(os.path.join(d, "t.mtplot.json")))
    assert [type(x).__name__ for x in q.sections] == ["SectionInfo", "SectionInfo"]
    assert q.sections[1].stations == ["S1", "S2"] and q.section_style.show_interp
    assert SEC.next_label(["A", "B"]) == "C" and SEC.next_label([chr(65 + i) for i in range(26)]) == "AA"


def test_section_groups_set_file_and_inset():
    """Sub-tab groups, saving / loading a section set, the inset index map."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from mtinv_gui.plotting.backends import figures as F
    from mtinv_gui.plotting.backends import sections as SEC
    from mtinv_gui.plotting.project import SectionInfo

    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        pp.sections = [SectionInfo(label="A", kind="ns", coord_m=X0 + 1000.0),
                       SectionInfo(label="B", kind="ew", coord_m=Y0),
                       SectionInfo(label="C", kind="ns", coord_m=X0 - 3000.0, ve=2.0),
                       SectionInfo(label="D", kind="path", vertices=[[X0 - 9e3, Y0 - 5e3],
                                                                    [X0 + 9e3, Y0 + 5e3]]),
                       SectionInfo(label="E", kind="stations", stations=["S1", "S2"])]
        groups = {g: [x.label for x in pp.sections if SEC.in_group(x, g)] for g in SEC.SECTION_GROUPS}
        assert groups == {"ns": ["A", "C"], "ew": ["B"], "other": ["D", "E"],
                          "all": ["A", "B", "C", "D", "E"]}, groups
        assert [x.label for x in SEC.group_members(pp.sections, pp.sections[2])] == ["A", "C"]

        # save + load: same definitions; clashing labels renamed
        f = SEC.write_section_set(os.path.join(d, "secs.json"), pp.sections, s.maptr)
        assert SEC.is_section_set(f) and not SEC.is_section_set(pp.model.model_path)
        back, notes = SEC.read_section_set(f, s.maptr)
        assert not notes and back == pp.sections
        back, notes = SEC.read_section_set(f, s.maptr, existing_labels=["A", "B", "C", "D", "E"])
        assert [x.label for x in back] == ["F", "G", "H", "I", "J"] and len(notes) == 5
        # another display CRS: the path comes back through lon/lat
        pp.georef.display_crs = "EPSG:3857"
        s.apply_georef(pp.georef)
        back, notes = SEC.read_section_set(f, s.maptr)
        ll = s.maptr.to_lonlat(*np.asarray(back[3].vertices).T)
        pp.georef.display_crs = ""
        s.apply_georef(pp.georef)
        ll0 = s.maptr.to_lonlat(*np.asarray(pp.sections[3].vertices).T)
        assert np.allclose(ll, ll0, atol=1e-7), (ll, ll0)
        assert any("N-S" in n for n in notes) and any("E-W" in n for n in notes)

        # inset map on the section figure (and none in a layout panel)
        st = pp.section_style
        st.depth_bottom_m, st.inset_map, st.inset_size_in = 9000.0, True, 0.6
        sec = pp.sections[0]
        for corner in render.INSET_CORNERS:
            st.inset_corner = corner
            fig = render.new_figure(8.0, 4.0)
            ax = render.render_section(fig, s, sec, st, others=SEC.group_members(pp.sections, sec))
            FigureCanvasAgg(fig).draw()
            insets = [a for a in fig.axes if a.get_label() == "inset_map"]
            assert len(insets) == 1, corner
            ib, pb = insets[0].get_position(), ax.get_position()
            assert pb.x0 - 1e-3 <= ib.x0 and ib.x1 <= pb.x1 + 1e-3, corner
            assert pb.y0 - 1e-3 <= ib.y0 and ib.y1 <= pb.y1 + 1e-3, corner
            right = ib.x0 + ib.x1 > pb.x0 + pb.x1
            upper = ib.y0 + ib.y1 > pb.y0 + pb.y1
            assert right == corner.endswith("right") and upper == corner.startswith("upper"), corner
            if corner == "upper right":     # a very flat section: the inset shrinks to fit
                st.depth_bottom_m = 1500.0
                fig = render.new_figure(10.0, 2.0)
                ax = render.render_section(fig, s, sec, st, others=pp.sections[:3])
                FigureCanvasAgg(fig).draw()
                (flat,) = [a for a in fig.axes if a.get_label() == "inset_map"]
                fb, pb2 = flat.get_position(), ax.get_position()
                assert pb2.y0 <= fb.y0 and fb.y1 <= pb2.y1, (fb, pb2)
                st.depth_bottom_m = 9000.0
            red = [l for l in insets[0].lines if l.get_color() == "#d62728"]
            assert len(red) == 1 and len(insets[0].lines) == 2      # A red, C grey
        out = render.export_section(os.path.join(d, "inset.png"), s, sec, st,
                                    others=SEC.group_members(pp.sections, sec))
        assert os.path.getsize(out) > 10_000
        outs = render.export_sections_all(os.path.join(d, "all"), s, pp.sections[:2], st,
                                          all_sections=pp.sections)
        assert len(outs) == 2
        # index-map recipes keep the sub-tab's group
        pp.section_group = "ns"
        r = F.recipe_index_map(pp)
        assert r.section_group == "ns" and "N-S" in r.name
        fig = render.new_figure(7.0, 6.0)
        ax = F.render_recipe(fig, s, pp, r)
        assert sum(t.get_text() in ("B", "D", "E") for t in ax.texts) == 0
        assert sum(t.get_text() in ("A", "C") for t in ax.texts) == 2


def _write_raster(path, crs, x0, y1, px, values, nodata=None):
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0],
                       count=1, dtype="float32", crs=crs, transform=from_origin(x0, y1, px, px),
                       nodata=nodata) as dst:
        dst.write(values.astype("float32"), 1)
    return path


def test_raster_layer_other_crs():
    """A raster in UTM 8N lines up with a model displayed in UTM 9N."""
    from pyproj import Transformer

    from mtinv_gui.plotting.backends import layers as L
    from mtinv_gui.plotting.project import LayerInfo

    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        tr = Transformer.from_crs(CRS, "EPSG:32608", always_xy=True)
        cx, cy = tr.transform(X0, Y0)
        n, px = 200, 500.0
        x_left, y_top = cx - n * px / 2, cy + n * px / 2
        xs = x_left + (np.arange(n) + 0.5) * px
        ys = y_top - (np.arange(n) + 0.5) * px
        vals = np.add.outer(np.zeros(n), xs / 1000.0)          # value = easting (km, UTM 8N)
        vals[:5, :5] = -9999.0
        path = _write_raster(os.path.join(d, "r.tif"), "EPSG:32608", x_left, y_top, px, vals, -9999.0)
        info = LayerInfo(name="r", kind="raster", path=path)
        r = s.layer(info)
        E, N = np.array([X0, X0 + 7000.0]), np.array([Y0, Y0 - 3000.0])
        got = r.sample(E, N, s.maptr.display_crs)
        want = np.asarray(tr.transform(E, N)[0]) / 1000.0
        assert np.allclose(got, want, atol=1e-6), (got, want)
        assert np.isnan(r.values).sum() == 25                   # nodata honoured
        assert s.layer(info) is r                               # cached
        info.scale = 1000.0
        assert s.layer(info) is not r                           # read settings change the key
        lo, hi = L.value_range(LayerInfo(kind="raster"), r)
        assert lo < hi


def test_surface_and_vector_on_section():
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    from mtinv_gui.plotting.backends import sections as SEC
    from mtinv_gui.plotting.project import LayerInfo, SectionInfo

    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        n = 100
        depth_km = np.full((n, n), 3.0)
        depth_km[0, 0] = -9999.0
        surf = _write_raster(os.path.join(d, "cpd.tif"), CRS, X0 - 25e3, Y0 + 25e3, 500.0, depth_km)
        gdf = gpd.GeoDataFrame({"NAME": ["F1", "P1", "far"]}, geometry=[
            LineString([(X0 - 10e3, Y0 + 2000.0), (X0 + 10e3, Y0 + 2000.0)]),   # crosses at N = +2 km
            Point(X0 + 5300.0, Y0 - 4000.0),                                    # 300 m off the line
            Point(X0 + 20e3, Y0)], crs=CRS)
        vpath = os.path.join(d, "f.gpkg")
        gdf.to_file(vpath, driver="GPKG")
        layers = [LayerInfo(name="CPD", kind="surface", path=surf, scale=1000.0, valid_min=-9000.0,
                            section_mode="line"),
                  LayerInfo(name="faults", kind="vector", path=vpath, label_field="NAME",
                            section_mode="markers", section_offset_m=1000.0)]
        sec = SectionInfo(kind="ns", coord_m=X0 + 5000.0, start_m=Y0 - 8000.0, end_m=Y0 + 8000.0)
        data = SEC.prepare(s, sec, pp.section_style, layers)
        assert not data.layer_errors, data.layer_errors
        line = next(it for it in data.layer_items if it.kind == "line")
        assert np.allclose(line.values, -3000.0)
        marks = next(it for it in data.layer_items if it.kind == "markers")
        got = sorted(zip(np.round(marks.s), marks.labels))
        assert got == [(4000.0, "P1"), (10000.0, "F1")], got
        out = render.export_section(os.path.join(d, "s.png"), s, sec, pp.section_style, layers=layers)
        assert os.path.getsize(out) > 10_000


def test_depth_units():
    """A depth grid in km must land at km depth on sections, not metres."""
    from mtinv_gui.plotting.backends import layers as L
    from mtinv_gui.plotting.backends import sections as SEC
    from mtinv_gui.plotting.project import LayerInfo, SectionInfo

    assert L.guess_length_unit([13.5, 20.0, 37.4]) == "km"
    assert L.guess_length_unit([1200.0, 5400.0, 37000.0]) == "m"
    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        km = _write_raster(os.path.join(d, "cpd_km.tif"), CRS, X0 - 25e3, Y0 + 25e3, 500.0,
                           np.full((100, 100), 3.0))
        sec = SectionInfo(kind="ns", coord_m=X0 + 5000.0, start_m=Y0 - 8000.0, end_m=Y0 + 8000.0)

        def line_z(**kw):
            info = LayerInfo(name="CPD", kind="surface", path=km, section_mode="line", **kw)
            data = SEC.prepare(s, sec, pp.section_style, [info])
            loaded = s.layer(info)
            return next(it for it in data.layer_items if it.kind == "line").values, loaded

        z, loaded = line_z()                                     # auto, no scale: km detected
        assert np.allclose(z, -3000.0) and loaded.length_unit == "km" and loaded.unit_guessed
        assert np.allclose(loaded.values[np.isfinite(loaded.values)], 3.0)   # maps stay in km
        assert L.label(LayerInfo(name="CPD", kind="surface"), loaded) == "CPD (km)"
        z, loaded = line_z(depth_unit="m")                       # explicit unit wins
        assert np.allclose(z, -3.0) and not loaded.unit_guessed
        z, _ = line_z(scale=1000.0)                              # older projects: scale=1000 still right
        assert np.allclose(z, -3000.0)
        z, _ = line_z(depth_unit="ft")
        assert np.allclose(z, -3.0 * 0.3048)
        csv = os.path.join(d, "p.csv")
        lon, lat = s.maptr.to_lonlat(np.array([X0]), np.array([Y0]))
        with open(csv, "w") as fh:
            fh.write(f"lon,lat,depth\n{lon[0]:.8f},{lat[0]:.8f},12.5\n")
        pts = s.layer(LayerInfo(name="p", kind="points", path=csv, depth_col="depth"))
        assert np.allclose(pts.depth, [12500.0]) and pts.length_unit == "km"


def test_profile_axes_each_get_their_own():
    """Several profiles over a section: one colour-coded y-axis each."""
    from matplotlib.colors import to_hex

    from mtinv_gui.plotting.backends import layers as L
    from mtinv_gui.plotting.backends import sections as SEC
    from mtinv_gui.plotting.project import LayerInfo, SectionInfo

    def item(name, axis="auto", color="black"):
        return L.SectionItems(LayerInfo(name=name, profile_axis=axis, color=color), "profile",
                              np.array([0.0, 1.0]), np.array([0.0, 1.0]), vrange=(0.0, 1.0))

    plan = render.profile_axes_plan([item("mag"), item("grav")])          # both default black
    assert [(p[0], p[1]) for p in plan] == [("left", 0), ("right", 0)]
    assert plan[0][2] == "black" and to_hex(plan[1][2]) != to_hex("black")
    plan = render.profile_axes_plan([item("a"), item("b"), item("c"), item("d", "left")])
    assert [(p[0], p[1]) for p in plan] == [("left", 0), ("right", 0), ("right", 1), ("left", 1)]
    assert len({to_hex(p[2]) for p in plan}) == 4
    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        st = pp.section_style
        st.topo_profile, st.topo_color = True, "tab:brown"
        sec = SectionInfo(kind="ew", coord_m=Y0, start_m=X0 - 5000.0, end_m=X0 + 5000.0)
        data = SEC.prepare(s, sec, st)
        topo = [it for it in data.layer_items if it.kind == "profile"]
        assert [it.info.name for it in topo] == ["Topography"] and np.allclose(topo[0].values, 0.0)
        fig = render.new_figure(*render.section_size(sec, st, data, s))
        render.render_section(fig, s, sec, st, data)
        pax = next(a for a in fig.axes if a.get_ylabel().startswith("Topography"))
        assert to_hex(pax.yaxis.label.get_color()) == to_hex("tab:brown")
        assert to_hex(pax.spines["left"].get_edgecolor()) == to_hex("tab:brown")


def test_points_table_and_profiles():
    from mtinv_gui.plotting.backends import layers as L
    from mtinv_gui.plotting.backends import sections as SEC
    from mtinv_gui.plotting.project import LayerInfo, SectionInfo

    assert abs(L._to_number("-130. 22103") - (-130.022103)) < 1e-12     # Fortran blanks
    assert L._to_number(" 10001 ") == 10001.0 and np.isnan(L._to_number("abc"))
    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        lon, lat = s.maptr.to_lonlat(np.array([X0, X0 + 1000.0]), np.array([Y0, Y0]))
        csv = os.path.join(d, "p.csv")
        with open(csv, "w") as fh:
            fh.write(" name , long , lat , depth\n")
            for i, (a, b) in enumerate(zip(lon, lat)):
                fh.write(f" p{i} , {a:.8f} , {b:.8f} , {2.5 + i}\n")
        info = LayerInfo(name="p", kind="points", path=csv, label_col="name", depth_col="depth",
                         scale=1000.0, section_mode="markers", section_offset_m=100.0)
        pts = s.layer(info)
        assert pts.n == 2 and pts.labels == ["p0", "p1"]
        assert np.allclose(pts.E, [X0, X0 + 1000.0], atol=0.01) and np.allclose(pts.depth, [2500, 3500])
        # lon/lat/z npz raster as a profile
        lo = np.linspace(lon[0] - 1, lon[0] + 1, 81)
        la = np.linspace(lat[0] - 1, lat[0] + 1, 61)
        LO, LA = np.meshgrid(lo, la)
        np.savez(os.path.join(d, "g.npz"), LO, LA, LA * 10.0)
        prof = LayerInfo(name="g", kind="raster", path=os.path.join(d, "g.npz"), units="mGal",
                         section_mode="profile", profile_axis="right")
        sec = SectionInfo(kind="ew", coord_m=Y0, start_m=X0 - 5000.0, end_m=X0 + 5000.0)
        data = SEC.prepare(s, sec, pp.section_style, [info, prof])
        kinds = sorted(it.kind for it in data.layer_items)
        assert kinds == ["markers", "profile"], kinds
        mk = next(it for it in data.layer_items if it.kind == "markers")
        assert np.allclose(sorted(mk.values), [-3500.0, -2500.0])           # depth below sea level
        pr = next(it for it in data.layer_items if it.kind == "profile")
        assert np.nanmax(np.abs(pr.values - lat[0] * 10.0)) < 1.0
        w, h = render.section_size(sec, pp.section_style, data)
        out = render.export_section(os.path.join(d, "s.png"), s, sec, pp.section_style, data)
        assert os.path.getsize(out) > 10_000 and h > 1.0
        pp.plan.main_layer = "g"
        out = render.export_plan(os.path.join(d, "plan.png"), s, pp.plan, 1000.0, [info, prof])
        assert os.path.getsize(out) > 10_000


def test_house_style():
    import matplotlib as mpl

    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        h = pp.house
        h.enabled, h.width_mm, h.font_size = True, 85.0, 7.0
        h.color.rho_min, h.color.rho_max = 3.0, 300.0
        s.house = h
        w, hgt = render.plan_size(pp.plan, s)
        assert abs(w - 85.0 / 25.4) < 1e-9 and abs(hgt / w - pp.plan.height_in / pp.plan.width_in) < 1e-9
        fig = render.new_figure(w, hgt)
        ax = render.render_plan(fig, s, pp.plan, 2500.0)
        norm = ax._mt_mappable.norm
        assert (norm.vmin, norm.vmax) == (3.0, 300.0)               # shared colour scale wins
        assert abs(ax.xaxis.label.get_fontsize() - 7.0) < 1e-9
        out = render.export_plan(os.path.join(d, "p.pdf"), s, pp.plan, 2500.0)
        pdf = open(out, "rb").read()
        assert b"/Type3" not in pdf and b"FontFile2" in pdf           # editable TrueType text
        h.link_colors = False
        assert render.color_of(s, pp.plan.color) is pp.plan.color
        h.enabled = False
        assert render.plan_size(pp.plan, s) == (pp.plan.width_in, pp.plan.height_in)
        assert mpl.rcParams["pdf.fonttype"] in (3, 42)              # rc changes don't leak


def test_layout_and_recipes():
    from mtinv_gui.plotting.backends import figures as F
    from mtinv_gui.plotting.project import LayoutPanel, SectionInfo

    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        pp.workdir = d
        pp.sections = [SectionInfo(label="A", kind="ew", coord_m=Y0, start_m=X0 - 15e3, end_m=X0 + 15e3)]
        pp.section_style.depth_bottom_m = 9000.0
        r = F.new_layout_recipe(pp, 2, 2, "mix")
        r.layout.panels = [LayoutPanel(kind="plan", depth_m=1000.0), LayoutPanel(kind="section", section="A"),
                           LayoutPanel(kind="index"), LayoutPanel(kind="empty")]
        fig = render.new_figure(*F.recipe_size(s, pp, r))
        axes = F.render_layout(fig, s, pp, r.layout)
        assert [a is None for a in axes] == [False, False, False, True]
        heads = [getattr(a, "_mt_head", a) for a in axes[:3]]
        titles = [h.get_title() or h.get_title(loc="left") for h in heads]
        assert [t.split()[0] for t in titles] == ["(a)", "(b)", "(c)"], titles
        assert F.panel_label(27, "A") == "B" and F.panel_label(1, "none") == ""
        bad = F.new_layout_recipe(pp, 1, 1, "broken")
        bad.layout.panels = [LayoutPanel(kind="section", section="nope")]   # drawn as a message
        pp.recipes = [F.recipe_from_plan(pp), F.recipe_from_section(pp, pp.sections[0]),
                      F.recipe_index_map(pp), r, bad]
        pp.recipes[0].out_path = "figs/plan.png"
        res = F.export_recipes(s, pp)
        assert all(not isinstance(p, Exception) for _n, p in res), res
        assert os.path.isfile(os.path.join(d, "figs", "plan.png"))
        res = F.export_recipes(s, pp, pp.recipes[:2], out_dir=os.path.join(d, "o"))
        assert sorted(os.listdir(os.path.join(d, "o"))) == ["plan.png", "section_A.pdf"]
        # the section recipe follows the live section geometry
        pp.sections[0].coord_m = Y0 + 3000.0
        assert F._live_section(pp, pp.recipes[1]).coord_m == Y0 + 3000.0


def test_colorbar_ends_and_shared_axes():
    from mtinv_gui.plotting.backends import figures as F
    from mtinv_gui.plotting.project import LayoutPanel, SectionInfo

    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        fig = render.new_figure(*render.plan_size(pp.plan, s))
        render.render_plan(fig, s, pp.plan, 2500.0)
        assert fig.axes[-1]._colorbar.extend == "neither"           # square ends by default
        pp.plan.color.extend_arrows = True
        render.render_plan(fig, s, pp.plan, 2500.0)
        assert fig.axes[-1]._colorbar.extend == "both"
        pp.plan.color.extend_arrows = False
        # 2 x 2 plan views: y labels on the left column, x labels on the bottom row
        r = F.new_layout_recipe(pp, 2, 2)
        r.layout.panels = [LayoutPanel(kind="plan", depth_m=z) for z in (1000.0, 2500.0, 6000.0, 8000.0)]
        fig = render.new_figure(*F.recipe_size(s, pp, r))
        axes = F.render_layout(fig, s, pp, r.layout)
        shown = [(bool(a.get_xlabel()), bool(a.get_ylabel())) for a in axes]
        assert shown == [(False, True), (False, False), (True, True), (True, False)], shown
        # sections of different lengths keep their own x-axes; equal ones share
        pp.sections = [SectionInfo(label="A", kind="ew", coord_m=Y0, start_m=X0 - 15e3, end_m=X0 + 15e3),
                       SectionInfo(label="B", kind="ew", coord_m=Y0 + 3e3, start_m=X0 - 15e3, end_m=X0 + 5e3),
                       SectionInfo(label="C", kind="ew", coord_m=Y0 - 3e3, start_m=X0 - 15e3, end_m=X0 + 15e3)]
        pp.section_style.depth_bottom_m = 9000.0
        for labels, expect in ((("A", "B"), [True, True]), (("A", "C"), [False, True])):
            r = F.new_layout_recipe(pp, 2, 1)
            r.layout.panels = [LayoutPanel(kind="section", section=l) for l in labels]
            fig = render.new_figure(*F.recipe_size(s, pp, r))
            axes = F.render_layout(fig, s, pp, r.layout)
            assert [bool(a.get_xlabel()) for a in axes] == expect, (labels, expect)
        r.layout.hide_shared_axes = False
        axes = F.render_layout(render.new_figure(*F.recipe_size(s, pp, r)), s, pp, r.layout)
        assert all(a.get_xlabel() for a in axes)


def test_batch_cli():
    from mtinv_gui.plotting import batch
    from mtinv_gui.plotting.backends import figures as F

    with tempfile.TemporaryDirectory() as d:
        s, pp = _session(d)
        pp.workdir = d
        pp.recipes = [F.recipe_from_plan(pp, "p1")]
        proj = pp.save(os.path.join(d, "t.mtplot.json"))
        assert batch.main([proj, "--list"]) == 0
        assert batch.main([proj, "--only", "nope"]) == 2
        other = os.path.join(d, "copy.vtu")
        import shutil

        shutil.copy(pp.model.model_path, other)
        assert batch.main([proj, "--model", other, "--out", os.path.join(d, "it2")]) == 0
        assert os.listdir(os.path.join(d, "it2")) == ["p1.pdf"]


def test_nested_project_roundtrip():
    from mtinv_gui.plotting.backends import figures as F
    from mtinv_gui.plotting.project import LayoutPanel, RecipeInfo

    pp = PlotProject()
    r = RecipeInfo(name="L", kind="layout")
    r.layout.panels = [LayoutPanel(kind="section", section="Q"), LayoutPanel(depth_m=7.0)]
    pp.recipes = [r, F.recipe_from_plan(pp)]
    pp.house.color.cmap = "viridis"
    with tempfile.TemporaryDirectory() as d:
        q = PlotProject.load(pp.save(os.path.join(d, "t.mtplot.json")))
    assert [type(p).__name__ for p in q.recipes[0].layout.panels] == ["LayoutPanel", "LayoutPanel"]
    assert q.recipes[0].layout.panels[1].depth_m == 7.0 and q.house.color.cmap == "viridis"
    assert type(q.recipes[1].plan.extent).__name__ == "ExtentInfo"


def test_vector_layers_leave_no_pyplot_figure():
    """geopandas' plot() ends with plt.draw(), which made an empty pyplot figure
    (shown as a blank output in notebooks, kept alive in the GUI)."""
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import shapely.geometry as sg

    plt.close("all")
    fig = render.new_figure(4, 4)
    ax = fig.add_subplot(111)
    gdf = gpd.GeoDataFrame(geometry=[sg.LineString([(0, 0), (1, 1)]), sg.Point(0.5, 0.2)])
    render._gpd_plot(gdf, ax=ax, color="k")
    assert plt.get_fignums() == []
    assert len(ax.lines) + len(ax.collections) > 0


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
