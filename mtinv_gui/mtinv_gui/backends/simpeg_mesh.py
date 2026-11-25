"""SimPEG output: the built mesh design as a discretize ``TreeMesh`` (an octree).

An octree can't be deformed to the topography, so this isn't a one-to-one copy of the
GoFEM mesh (as the FEMTIC one is).  Instead the GoFEM mesh, already fitted to the
topography, says how fine the octree must be where: every deal.II cell asks for
octree cells no wider than itself at its own (fitted) centre, so the fine cells along
the surface, around the stations and the coast follow the real topography.  discretize
keeps the octree balanced (neighbours within one level).  The topography is then a
property of the cells, as SimPEG expects: each octree cell is earth, air, sea or mantle
by where its centre is relative to the DEM, with the same rules as the GoFEM build.

Only horizontal sizes are matched.  Octree cells are scaled copies of the finest one, so
GoFEM's thin near-surface layers (which run out across the whole padding) would force the
finest cells everywhere; instead the finest cell is ``vertical_ratio`` times flatter than
it is wide (1 = cubes) and every cell keeps that shape.

Frames: SimPEG's, x = east, y = north, z = up (elevation, sea level 0), metres, origin
at the survey centre (the same local frame as the GoFEM mesh, axes swapped).

Files (``write_mesh``):
  ``octree_mesh.txt``       UBC OcTree mesh (``discretize.TreeMesh.read_UBC``)
  ``octree_*.txt``          UBC models: conductivity (S/m), material id, active (1 = inverted)
  ``simpeg_model.npz``      the same in discretize's cell order, plus the receivers
  ``receivers_simpeg.csv``  name, x (east), y (north), z: on the discretised earth surface
                            (the seafloor offshore)
Needs ``discretize`` (SimPEG's mesh package).
"""
from __future__ import annotations

import os

import numpy as np

MESH_TXT = "octree_mesh.txt"
MODEL_FILES = {"conductivity": "octree_conductivity.txt", "material": "octree_material.txt",
               "active": "octree_active.txt"}
MODEL_NPZ = "simpeg_model.npz"
RECEIVERS_CSV = "receivers_simpeg.csv"


def _discretize():
    try:
        import discretize
    except ImportError as exc:                                # pragma: no cover
        raise ImportError("SimPEG output needs the 'discretize' package "
                          "(pip install discretize)") from exc
    return discretize


# --------------------------------------------------------------------------- #
# the octree
# --------------------------------------------------------------------------- #
def cell_boxes(xyz, vid):
    """Per deal.II cell: centre (north, east, z down) and horizontal extents
    (north, east), from vertices *xyz* and cell vertex ids *vid*
    (:func:`femtic.tria_arrays`)."""
    corners = xyz[vid]                                       # (n, 8, 3)
    centre = corners.mean(axis=1)
    size = corners.max(axis=1) - corners.min(axis=1)
    return centre, size[:, :2]


def tree_axes(bounds, h_min_north, h_min_east, vertical_ratio=1.0):
    """Finest cell sizes, counts (powers of 2) and origin per SimPEG axis (east, north, up)
    for an octree covering *bounds* ``(north0, north1, east0, east1, z0, z1)`` (deal.II
    frame, z down): the finest cells as wide as the requested ones (the narrowest GoFEM
    cells), and exactly *vertical_ratio* times wider than tall.

    The octree's box is a power of 2 of those cells, so it is usually bigger than *bounds*
    (up to twice as wide): fitting the box to *bounds* instead would shrink the finest cells
    by up to half, and multiply the refined cells by up to 8 (2.4 M instead of ~0.5 M on the
    notebooks' sample).  The extra margin gets the coarsest cells, so it costs little.

    The box is centred on the middle of *bounds* horizontally and on sea level vertically,
    so that the survey centre and sea level are boundaries of the cells at every level (an
    octree cell is aligned to multiples of its own size from the origin).  Sizes are whole
    millimetres, rounded down, as the UBC mesh file stores them (an octree read back from it
    must have the same cells)."""
    n0, n1, e0, e1, z0, z1 = bounds

    def mm(v):
        return np.floor(v * 1000.0 + 1e-6) / 1000.0

    def count(length, h):
        return 2 ** max(int(np.ceil(np.log2(length / h - 1e-9))), 1)

    h_e, h_n = mm(h_min_east), mm(h_min_north)
    n_e, n_n = count(e1 - e0, h_e), count(n1 - n0, h_n)
    h_z = mm(min(h_e, h_n) / float(vertical_ratio))
    n_z = count(2.0 * max(z1, -z0), h_z)
    origin = ((e0 + e1) / 2.0 - n_e / 2 * h_e, (n0 + n1) / 2.0 - n_n / 2 * h_n, -n_z / 2 * h_z)
    return [h_e, h_n, h_z], [n_e, n_n, n_z], origin


def build_tree(centre, size, bounds, *, vertical_ratio=1.0):
    """A ``discretize.TreeMesh`` over *bounds* in which each deal.II cell's centre
    (*centre*: north, east, z down) lies in an octree cell no wider than that cell
    (*size*: north, east extents)."""
    discretize = _discretize()
    (h_e, h_n, h_z), (n_e, n_n, n_z), origin = tree_axes(
        bounds, size[:, 0].min(), size[:, 1].min(), vertical_ratio)
    mesh = discretize.TreeMesh([np.full(n_e, h_e), np.full(n_n, h_n), np.full(n_z, h_z)],
                               origin=origin, diagonal_balance=False)
    # how many times wider than the finest cell each deal.II cell may be: 2^k
    ratio = np.minimum(size[:, 0] / h_n, size[:, 1] / h_e)
    k = np.floor(np.log2(np.maximum(ratio, 1.0)) + 1e-9).astype(int)
    levels = np.clip(mesh.max_level - k, 0, mesh.max_level)
    points = np.c_[centre[:, 1], centre[:, 0], -centre[:, 2]]   # east, north, up
    # inside the domain (a fitted cell centre can sit exactly on the boundary)
    lo = np.asarray(origin) + 1e-6
    hi = np.asarray(origin) + np.array([n_e * h_e, n_n * h_n, n_z * h_z]) - 1e-6
    mesh.insert_cells(np.clip(points, lo, hi), levels, finalize=True)
    return mesh


# --------------------------------------------------------------------------- #
# cell roles, the same rules as the GoFEM build
# --------------------------------------------------------------------------- #
def surface_elevation(dem, north, east, *, land: bool = True, bathymetry: bool = False):
    """The air (or water) / earth interface, as ``mesh.fit_topography`` puts it, and
    whether each point is sea: land elevations if *land* (land below sea level at its own
    elevation), the seafloor if *bathymetry*, sea level otherwise."""
    north = np.asarray(north, dtype="float64")
    if dem is None:
        return np.zeros_like(north), np.zeros(north.shape, bool)
    pts = np.c_[north, east]
    elev = np.asarray(dem.elevation(pts), dtype="float64")
    inland = np.asarray(dem.inland(pts), dtype=bool) if dem.has_inland else np.zeros(elev.shape, bool)
    sea = (elev < 0) & ~inland
    surface = np.zeros_like(elev)
    if land:
        surface = np.where(~sea, elev, surface)
    if bathymetry:
        surface = np.where(sea, elev, surface)
    return surface, sea


def cell_materials(mesh, dem, *, ids: dict, land: bool = True, bathymetry: bool = False,
                   mantle_depth_m: float | None = None) -> np.ndarray:
    """Material id per octree cell by its centre: earth below the surface (mantle below
    *mantle_depth_m*), sea between the seafloor and sea level where the DEM is sea (with
    *bathymetry*, i.e. a real ocean), air elsewhere.  *ids*: ``air``, ``earth``, ``sea``,
    ``mantle``."""
    cc = mesh.cell_centers
    surface, sea = surface_elevation(dem, cc[:, 1], cc[:, 0], land=land, bathymetry=bathymetry)
    z = cc[:, 2]
    mat = np.full(mesh.n_cells, ids["air"], dtype=int)
    earth = z < surface
    mat[earth] = ids["earth"]
    if mantle_depth_m is not None:
        mat[earth & (-z > mantle_depth_m)] = ids["mantle"]
    if bathymetry:
        mat[~earth & sea & (z < 0)] = ids["sea"]
    return mat


def earth_top(mesh, solid, east, north):
    """Elevation of the top of the highest *solid* cell in the column at each
    (east, north): the discretised earth surface (or seafloor) under a station."""
    cc, h = mesh.cell_centers, mesh.h_gridded
    top = cc[:, 2] + h[:, 2] / 2
    # cells whose upper neighbour isn't solid: probe just above their top
    probe = np.c_[cc[:, 0], cc[:, 1], np.minimum(top + 1e-3 * h[:, 2], mesh.nodes[:, 2].max() - 1e-6)]
    above = mesh.get_containing_cells(probe)
    surf = np.flatnonzero(solid & (~solid[above] | (top >= mesh.nodes[:, 2].max() - 1e-6)))
    x0, x1 = cc[surf, 0] - h[surf, 0] / 2, cc[surf, 0] + h[surf, 0] / 2
    y0, y1 = cc[surf, 1] - h[surf, 1] / 2, cc[surf, 1] + h[surf, 1] / 2
    out = np.full(len(east), np.nan)
    for i, (e, n) in enumerate(zip(east, north)):
        hit = (x0 <= e) & (e <= x1) & (y0 <= n) & (n <= y1)
        if hit.any():
            out[i] = top[surf[hit]].max()
    return out


# --------------------------------------------------------------------------- #
# files
# --------------------------------------------------------------------------- #
def write_mesh(tria, materials: dict, out_dir: str, *, dem, ids: dict, free_ids,
               land: bool = True, bathymetry: bool = False, mantle_depth_m: float | None = None,
               names=(), receivers=(), vertical_ratio: float = 1.0, log=print) -> dict:
    """Build the octree from the fitted triangulation and write the SimPEG files into
    *out_dir*.  *materials* ``{id: (name, sigma)}`` gives the conductivities, *free_ids*
    the inverted ones.  *receivers*: ``[[north, east, ...], ...]`` (mesh frame)."""
    from .femtic import tria_arrays

    xyz, vid, _level, _mat = tria_arrays(tria)
    centre, size = cell_boxes(xyz, vid)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])
    mesh = build_tree(centre, size, bounds, vertical_ratio=vertical_ratio)
    mat = cell_materials(mesh, dem, ids=ids, land=land, bathymetry=bathymetry,
                         mantle_depth_m=mantle_depth_m)
    sigma_of = {mid: s for mid, (_name, s) in materials.items()}
    sigma_of.setdefault(ids["sea"], 3.2)
    sigma = np.array([sigma_of[m] for m in mat]) if len(mat) else np.zeros(0)
    active = np.isin(mat, list(free_ids))

    rec = np.asarray(receivers, dtype="float64").reshape(-1, 3)[:, :2] if len(receivers) else np.zeros((0, 2))
    solid = np.isin(mat, [ids["earth"], ids["mantle"]] + list(free_ids))
    z_rec = earth_top(mesh, solid, rec[:, 1], rec[:, 0]) if len(rec) else np.zeros(0)
    if np.isnan(z_rec).any():
        raise ValueError("SimPEG: no earth under station(s) "
                         + ", ".join(n for n, z in zip(names, z_rec) if np.isnan(z)))

    os.makedirs(out_dir, exist_ok=True)
    models = {MODEL_FILES["conductivity"]: sigma, MODEL_FILES["material"]: mat.astype(float),
              MODEL_FILES["active"]: active.astype(float)}
    # discretize 0.11 writes the mesh file itself relative to the working directory
    mesh.write_UBC(os.path.abspath(os.path.join(out_dir, MESH_TXT)), models=models,
                   directory=os.path.abspath(out_dir))
    npz = os.path.join(out_dir, MODEL_NPZ)
    np.savez(npz, conductivity=sigma, material=mat, active=active,
             receiver_names=np.asarray(list(names), dtype=str),
             receiver_locations=np.c_[rec[:, 1], rec[:, 0], z_rec] if len(rec) else np.zeros((0, 3)),
             materials=np.array([(k, v[0], v[1]) for k, v in materials.items()], dtype=object),
             frame="x east, y north, z up (m); origin = survey centre")
    csv = os.path.join(out_dir, RECEIVERS_CSV)
    with open(csv, "w") as fh:
        fh.write("name,x_east_m,y_north_m,z_m\n")
        for n, (north, east), z in zip(names, rec, z_rec):
            fh.write(f"{n},{east:.3f},{north:.3f},{z:.3f}\n")
    if bathymetry and dem is not None:
        e0, n0, _z = mesh.origin
        e1, n1 = e0 + mesh.h[0].sum(), n0 + mesh.h[1].sum()
        expected = dem.expected_water_volume((e0, e1, n0, n1))
        if expected > 0:
            ratio = float(mesh.cell_volumes[mat == ids["sea"]].sum()) / expected
            log("SimPEG: resolved water volume = %.0f%% of the DEM's" % (100 * ratio))
            if ratio < 0.7:
                log("  WARNING: the octree loses water shallower than half its cells' height near "
                    "the coast (an octree can't follow the seafloor the way the GoFEM mesh does): "
                    "raise the vertical ratio or refine the coast")
    h = mesh.h_gridded
    counts = {k: int((mat == v).sum()) for k, v in ids.items() if (mat == v).any()}
    log("SimPEG: wrote %s (%d cells: %s; finest %.0f x %.0f x %.0f m, octree %d x %d x %d) "
        "and the conductivity / material / active models"
        % (MESH_TXT, mesh.n_cells, ", ".join(f"{k} {v}" for k, v in counts.items()),
           h[:, 0].min(), h[:, 1].min(), h[:, 2].min(), *mesh.shape_cells))
    return {"simpeg_mesh": os.path.join(out_dir, MESH_TXT), "simpeg_model": npz,
            "simpeg_receivers": csv, "simpeg_cells": int(mesh.n_cells)}


def read_model(out_dir: str):
    """``(TreeMesh, dict of arrays)`` as written by :func:`write_mesh`."""
    discretize = _discretize()
    mesh = discretize.TreeMesh.read_UBC(os.path.join(out_dir, MESH_TXT))
    with np.load(os.path.join(out_dir, MODEL_NPZ), allow_pickle=True) as f:
        arrays = {k: f[k] for k in f.files}
    return mesh, arrays
