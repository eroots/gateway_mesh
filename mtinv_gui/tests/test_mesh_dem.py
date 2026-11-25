"""The projected DEM as the mesh build sees it (``backends/mesh_dem.py``): two grids,
land below sea level vs the sea, and a small build with both.

    python tests/test_mesh_dem.py
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

from mtinv_gui.backends import mesh as M  # noqa: E402
from mtinv_gui.backends import mesh_dem as MD  # noqa: E402
from mtinv_gui.backends import ocean as O  # noqa: E402
from mtinv_gui.backends import projection as P  # noqa: E402

QUIET = lambda *_a: None  # noqa: E731


def grid(half_m, step_m, fn):
    xs = np.arange(-half_m, half_m + step_m / 2, step_m)
    east, north = np.meshgrid(xs, xs)
    return np.dstack([east, north, fn(east, north)])


def coast(east, north):
    """Sea west of -30 km (1000 m deep), a basin 200 m below sea level 15 km across
    centred 20 km east, land at 300 m elsewhere."""
    z = np.where(east < -30e3, -1000.0, 300.0)
    return np.where(np.hypot(east - 20e3, north) < 15e3, -200.0, z)


def mesh_pts(east, north):
    return np.c_[np.atleast_1d(north), np.atleast_1d(east)]      # the mesh has x = north


# ---- one grid ---------------------------------------------------------------- #
def test_connected_sea_leaves_the_basin_as_land():
    g = grid(80e3, 1e3, coast)
    dem = MD.MeshDem([g])
    assert dem.has_inland
    sea, basin, land = mesh_pts(-50e3, 0.0), mesh_pts(20e3, 0.0), mesh_pts(0.0, 50e3)
    assert dem.inland(basin)[0] and not dem.inland(sea)[0] and not dem.inland(land)[0]
    assert dem.elevation(basin)[0] == -200.0                     # the real elevation stays
    assert dem.ocean_elevation(basin)[0] == MD.LAND_VIEW_M       # the ocean code sees land
    assert dem.ocean_elevation(sea)[0] == -1000.0
    stats = dem.ocean_stats()
    assert abs(stats["inland_km2"] - np.pi * 15 ** 2) < 30
    assert abs(stats["ocean_frac"] - 50 / 161) < 0.01
    # the coastline is the sea's only: every point near x = -30 km
    pts, _info = dem.coast_points({"radius_km": 5})
    assert len(pts) and np.all(np.abs(pts[:, 1] + 30e3) < 1.5e3)


def test_a_bay_behind_an_unresolved_strait_is_sea():
    """A bay cut off from the sea by 3 km of 'land' (a strait the DEM doesn't resolve)."""
    def bay(east, north):
        z = np.where(east < -30e3, -1000.0, 300.0)
        return np.where((east > -27e3) & (east < -10e3) & (np.abs(north) < 10e3), -50.0, z)

    g = grid(80e3, 1e3, bay)
    assert not MD.MeshDem([g]).inland(mesh_pts(-20e3, 0.0))[0]             # 10 km gap: joined
    assert MD.MeshDem([g], gap_km=0).inland(mesh_pts(-20e3, 0.0))[0]       # no gap: enclosed
    assert MD.MeshDem([g], gap_km=2).inland(mesh_pts(-20e3, 0.0))[0]       # 3 km > 2 km


def test_rule_all_is_the_old_behaviour():
    g = grid(80e3, 1e3, coast)
    dem = MD.MeshDem([g], sea_rule="all")
    assert not dem.has_inland
    assert dem.ocean_elevation(mesh_pts(20e3, 0.0))[0] == -200.0
    assert dem.ocean_stats() == {**O.dem_ocean_stats(g), "inland_km2": 0.0}
    pts, _ = dem.coast_points({"radius_km": 5})
    old, _ = O.stage_coast_points(g, {"radius_km": 5})
    assert np.array_equal(pts, old)


def test_a_dem_with_no_land_below_sea_level_is_read_as_before():
    from scipy.interpolate import NearestNDInterpolator

    g = grid(80e3, 1e3, lambda e, n: np.where(e < -30e3, -1000.0, 300.0 + 0.01 * n))
    dem = MD.MeshDem([g])
    old = NearestNDInterpolator((g[:, :, 1].ravel(), g[:, :, 0].ravel()), g[:, :, 2].ravel())
    q = np.random.default_rng(1).uniform(-90e3, 90e3, (500, 2))
    assert np.array_equal(dem.elevation(q), old(q))
    assert np.array_equal(dem.ocean_elevation(q), old(q))
    assert dem.ocean_stats((-60e3, 60e3, -50e3, 50e3)) == \
        {**O.dem_ocean_stats(g, (-60e3, 60e3, -50e3, 50e3)), "inland_km2": 0.0}
    assert dem.expected_water_volume() == O.expected_water_volume(g)


def test_polygons_override_the_rule():
    import shapely.geometry as sg

    g = grid(80e3, 1e3, coast)
    lake = sg.Point(20e3, 0.0).buffer(16e3)                      # call the basin sea
    bay = sg.box(-80e3, -10e3, -30e3, 10e3)                      # and part of the sea land
    dem = MD.MeshDem([g], sea_polygons=lake, land_polygons=bay)
    assert not dem.inland(mesh_pts(20e3, 0.0))[0]
    assert dem.ocean_elevation(mesh_pts(20e3, 0.0))[0] == -200.0
    assert dem.inland(mesh_pts(-50e3, 0.0))[0]                    # inside the 'land' box
    assert not dem.inland(mesh_pts(-50e3, 40e3))[0]               # outside it: still sea


def test_polygon_files_are_read_in_the_model_frame():
    import geopandas as gpd
    import shapely.geometry as sg

    frame = P.make_local_frame("utm", 32610, -121.45, 36.72)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "land.geojson")
        gpd.GeoDataFrame(geometry=[sg.box(-121.5, 36.7, -121.4, 36.75)], crs="EPSG:4326").to_file(path)
        geom = P.read_polygons(path, frame)
    e, n = P.project_lonlat([-121.45], [36.725], frame)
    assert geom.contains(sg.Point(e[0], n[0]))
    e, n = P.project_lonlat([-121.6], [36.725], frame)
    assert not geom.contains(sg.Point(e[0], n[0]))


def test_shortfall():
    g = grid(50e3, 1e3, coast)
    assert MD.MeshDem([g]).shortfall((-50e3, 50e3, -50e3, 50e3)) == \
        {"west": 0.0, "east": 0.0, "south": 0.0, "north": 0.0}
    short = MD.shortfall(MD.grid_extent(g), (-80e3, 50e3, -50e3, 70e3))
    assert short == {"west": 30e3, "east": 0.0, "south": 0.0, "north": 20e3}
    assert MD.shortfall_text(short) == "30 km to the west, 20 km to the north"


# ---- two grids ------------------------------------------------------------------ #
def two_grids():
    """A coarse regional grid with the sea along its west edge; a fine grid around the
    survey whose west part is that sea (its own edge doesn't reach the open sea, the
    regional grid shows the connection) and which holds the basin."""
    regional = grid(400e3, 5e3, coast)
    fine = grid(40e3, 0.5e3, coast)
    return fine, regional


def test_two_grids_read_the_fine_one_where_it_reaches():
    fine, regional = two_grids()
    fine[:, :, 2] += 7.0                                         # tell them apart
    dem = MD.MeshDem([fine, regional])
    assert dem.extent == MD.grid_extent(regional)
    assert dem.elevation(mesh_pts(10e3, 30e3))[0] == 307.0        # inside the fine grid
    assert dem.elevation(mesh_pts(10e3, 100e3))[0] == 300.0       # outside it
    assert dem.elevation(mesh_pts(-200e3, 0.0))[0] == -1000.0


def test_the_fine_grids_sea_comes_from_the_regional_one():
    fine, regional = two_grids()
    dem = MD.MeshDem([fine, regional])
    assert not dem.inland(mesh_pts(-35e3, 0.0))[0]                # sea, joined outside the fine grid
    assert dem.inland(mesh_pts(20e3, 0.0))[0]                     # the basin
    assert dem.inland_masks[0].any() and dem.inland_masks[1].any()   # both grids see the basin
    # on its own, the fine grid's sea still touches its edge: also sea
    assert not MD.MeshDem([fine]).inland(mesh_pts(-35e3, 0.0))[0]


def test_two_grid_coastline_and_stats():
    fine, regional = two_grids()
    dem = MD.MeshDem([fine, regional])
    raw = dem.raw_coastlines()
    e0, e1, n0, n1 = dem.extents[0]
    fine_lines = O.extract_coastlines(dem.ocean_grids[0])
    for pts, _closed in raw[len(fine_lines):]:                  # the regional grid's lines
        inside = (pts[:, 0] > e0) & (pts[:, 0] < e1) & (pts[:, 1] > n0) & (pts[:, 1] < n1)
        assert not inside.any()
    pts, _ = dem.coast_points({"radius_km": 5}, (-400e3, 400e3, -400e3, 400e3))
    assert np.all(np.abs(pts[:, 1] + 30e3) < 5e3)                # only the real coast
    stats = dem.ocean_stats((-400e3, 400e3, -400e3, 400e3))
    assert abs(stats["ocean_frac"] - 370 / 800) < 0.02           # area-weighted, not point-counted
    vol = dem.expected_water_volume((-400e3, 400e3, -400e3, 400e3))
    assert abs(vol / (1000.0 * 370e3 * 800e3) - 1) < 0.03


def test_two_grids_round_trip_through_the_file():
    fine, regional = two_grids()
    frame = P.make_local_frame("local", 0, -121.45, 36.72)
    with tempfile.TemporaryDirectory() as d:
        path = P.save_projected_dem(P.ProjectedDem(fine, frame, regional=regional),
                                    os.path.join(d, "dem.npz"))
        dem = M.read_mesh_dem(path, log=QUIET)
        assert len(dem.grids) == 2 and np.array_equal(dem.grids[1], regional)
        assert MD.file_extent(path) == MD.grid_extent(regional)
        back = P.load_projected_dem(path)
        assert np.array_equal(back.topo_cart, fine) and np.array_equal(back.regional, regional)
        single = P.save_projected_dem(P.ProjectedDem(fine, frame), os.path.join(d, "one.npz"))
        assert np.load(single).files == ["arr_0"]


# ---- a build ------------------------------------------------------------------------- #
def _build(ocean_mode):
    """A small mesh over the coast DEM, through the same steps as mesh.run."""
    M.prepare_pygofem()
    from gofem.mesh_utils import refine_at_interface

    from mtinv_gui.backends import refine as R

    dem = MD.MeshDem([grid(80e3, 1e3, coast)])
    dx, dy, dz = M.cell_size_lists([10e3] * 6, [200.0, 400.0, 1e3, 4e3, 20e3, 40e3],
                                   [200.0, 400.0, 1e3, 4e3, 20e3, 40e3])
    tria = M.base_grid(dx, dy, dz, log=QUIET)
    materials = M.initial_materials(tria, 100.0, log=QUIET)
    for _ in range(2):
        refine_at_interface(tria, material_ids=[M.EARTH_ID], repeat=1, center=[0, 0, 0],
                            radii=[1e7, 1e7], pnorm=1)
        if ocean_mode == "real":
            M.refine_air_over_ocean(tria, dem.ocean_elevation, M.AIR_ID, [M.EARTH_ID])
    M.fit_topography(tria, dem, land=True, bathymetry=(ocean_mode == "real"), log=QUIET)
    M.add_ocean(tria, ocean_mode, dem, bounds=M.model_bounds(dx, dy), dz_earth=dz[len(dz) // 2:],
                materials=materials, background_resistivity_ohm_m=100.0, log=QUIET)
    cells, xyz, mat = R._cell_arrays(tria)
    return tria, xyz, mat, R


def test_build_keeps_the_basin_dry_at_its_own_depth():
    tria, xyz, mat, R = _build("real")
    north, east = xyz[:, 0], xyz[:, 1]
    water = mat == M.OCEAN_ID
    in_basin = np.hypot(east - 20e3, north) < 12e3
    assert water[east < -35e3].any() and not water[in_basin].any()
    surf = R.surface_faces(tria, [M.AIR_ID, M.OCEAN_ID])
    z = R.project_to_surface(surf, ["basin", "land", "sea"],
                             [[0.0, 20e3], [0.0, 50e3], [0.0, -55e3]])[:, 2]
    assert abs(z[0] - 200.0) < 1.0        # the basin floor, 200 m below sea level, under air
    assert abs(z[1] + 300.0) < 1.0        # land
    assert abs(z[2] - 1000.0) < 1.0       # the sea floor, under water


def test_thin_sheet_leaves_the_basin_out():
    _tria, xyz, mat, _R = _build("thin_sheet")
    north, east = xyz[:, 0], xyz[:, 1]
    sheet = mat >= O.SHEET_FIRST_ID
    assert sheet[east < -35e3].any()
    assert not sheet[np.hypot(east - 20e3, north) < 12e3].any()



def test_download_size_estimates():
    """Before downloading: GMRT's grid (checked against real GMRT responses,
    2026-11) and OpenTopography's size and area limit; the resampled .npz."""
    from mtinv_gui.backends import topography as T

    def box(s, n, w, e):
        return {"MinLon": w, "MaxLon": e, "MinLat": s, "MaxLat": n}

    # (box, GMRT's actual nx, ny for resolution=default)
    for b, nx, ny in ((box(35.5, 37.5, -123, -120.5), 571, 570),
                      (box(30, 40, -125, -115), 570, 698),
                      (box(50, 55, -135, -125), 569, 469),
                      (box(50, 52, -127, -125), 456, 726),
                      (box(36.4, 36.6, -122, -121.8), 365, 455)):
        e = T.estimate_download("GMRT", b, 0.0)
        assert abs(e["nx"] - nx) <= 3 and abs(e["ny"] - ny) <= 3, (b, e, nx, ny)
        assert e["mb"] < 5.0 and not e["exact"]
    b = box(35, 38, -122, -118)
    e = T.estimate_download("OpenTopography", b, 1.0, "SRTMGL1")
    assert (e["nx"], e["ny"]) == (21601, 18001) and abs(e["mb"] - 2 * 21601 * 18001 / 1e6) < 1
    assert e["area_km2"] < e["limit_km2"]
    e = T.estimate_download("OpenTopography", b, 4.0, "COP30")
    assert e["area_km2"] > e["limit_km2"] == 450_000.0
    assert T.estimate_download("OpenTopography", b, 1.0, "nope") is None
    assert abs(T.box_area_km2(0, 1, 0, 1) - 12364.0) < 10.0          # 1 x 1 degree at the equator
    r = T.estimate_resample(b, 3600.0, 3.0)
    assert (r["nx"], r["ny"]) == (7201, 6001), r
    assert abs(r["mb"] - r["nodes"] * 24 / 1e6) < 1e-6
    # the estimate is the grid resample_dem actually makes
    import numpy as np

    lon = np.linspace(-124, -116, 200)
    lat = np.linspace(33, 40, 150)
    dem = T.DemGrid(lon=lon, lat=lat, z=np.zeros((150, 200)))
    rs = T.resample_dem(dem, b, 1800.0, 60.0)
    r = T.estimate_resample(b, 1800.0, 60.0)
    assert rs.z.shape == (r["ny"], r["nx"]), (rs.z.shape, r)


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
