"""A stand-in for an inversion result, so the refine and plotting notebooks have a
model to work on before you have a real one.

It takes the mesh written by ``03_mesh_and_receivers.ipynb`` and does what a GoFEM
inversion does to it: every free (earth) cell gets its own material id, with its own
resistivity.  The resistivities come from a simple made-up model, not from the data:

* 100 ohm-m background,
* a 10 ohm-m conductor 8-25 km deep, elongated north-south, under the middle of the survey,
* a 2000 ohm-m resistive block in the upper 15 km on the eastern side,
* 30 ohm-m below 150 km.

Fixed cells (air, ocean, the mantle transition) keep their ids, as in a real inversion.
"""
from __future__ import annotations

import os
import shutil

import numpy as np

#: first id given to the "inverted" cells; GoFEM also uses ids far above the fixed ones
FIRST_ID = 10000


def resistivity(north_m, east_m, depth_m):
    """The made-up model: ohm-m at mesh-frame points (depth below sea level, m)."""
    n, e, z = (np.asarray(v, dtype="float64") for v in (north_m, east_m, depth_m))
    rho = np.full(n.shape, 100.0)
    rho[z > 150e3] = 30.0
    resistor = (e > 15e3) & (e < 45e3) & (np.abs(n) < 40e3) & (z < 15e3)
    rho[resistor] = 2000.0
    conductor = ((n / 45e3) ** 2 + ((e + 5e3) / 15e3) ** 2 + ((z - 16e3) / 8.5e3) ** 2) <= 1.0
    rho[conductor] = 10.0
    return rho


def make_synthetic_inversion(mesh_dir: str, out_dir: str, stem: str = "model_synthetic") -> dict:
    """Write ``<stem>.tria``, ``<stem>.materials`` and ``<stem>.vtu`` (with the
    ``conductivity`` cell array GoFEM's output has) into *out_dir*, plus a copy of the
    mesh's ``receivers_projected.csv``.  Returns the paths."""
    import pyvista as pv

    from mtinv_gui.backends import mesh as M
    from mtinv_gui.backends import refine as R

    tria = R.load_tria(os.path.join(mesh_dir, M.MESH_TRIA))
    materials = R.read_materials(os.path.join(mesh_dir, M.MESH_MATERIALS))
    cells = tria.active_cells()
    xyz = np.array([[c.center().x, c.center().y, c.center().z] for c in cells])
    free = np.array([c.material_id == M.EARTH_ID for c in cells])
    rho = resistivity(xyz[:, 0], xyz[:, 1], xyz[:, 2])

    sigma = np.array([materials[c.material_id].conductivity for c in cells])
    lines = ["# id name conductivity permitivitty", "iso"]
    lines += ["%d %s %0.6e %0.6e" % (m.id, m.name, m.conductivity, 0.0)
              for m in materials.values() if m.id != M.EARTH_ID]
    next_id = FIRST_ID
    for k in np.flatnonzero(free):
        cells[k].material_id = next_id
        sigma[k] = 1.0 / rho[k]
        lines.append("%d c%d %0.6e %0.6e" % (next_id, next_id, sigma[k], 0.0))
        next_id += 1

    os.makedirs(out_dir, exist_ok=True)
    paths = {"tria": os.path.join(out_dir, stem + ".tria"),
             "materials": os.path.join(out_dir, stem + ".materials"),
             "vtu": os.path.join(out_dir, stem + ".vtu"),
             "receivers": os.path.join(out_dir, M.RECEIVERS_CSV)}
    tria.save(paths["tria"])
    with open(paths["materials"], "w") as fh:
        fh.write("\n".join(lines) + "\n")
    tria.write(paths["vtu"], "vtu")
    grid = pv.read(paths["vtu"])
    centres = np.asarray(grid.cell_centers().points)
    if len(centres) != len(cells) or np.abs(centres - xyz).max() > 1.0:
        raise RuntimeError("the .vtu cells aren't in the mesh's order")
    grid.cell_data["conductivity"] = sigma
    grid.save(paths["vtu"])
    shutil.copyfile(os.path.join(mesh_dir, M.RECEIVERS_CSV), paths["receivers"])
    return paths
