"""Map an existing (usually inverted) model onto a different mesh.

The other way to start a new inversion from an old model: instead of refining the old
mesh (``refine.py``), build a fresh mesh for the new data in the Mesh tab, then give its
earth cells the old model's conductivities.  Nothing about the new mesh changes but its
material ids, so it is exactly what the Mesh tab built.

How a value is found, for each target cell to fill:

* Its centre goes from the target's frame to the source's, through lon/lat (the two
  models usually have different projections and origins).
* Depth is matched below the *surface* by default: a cell 300 m under the new mesh's
  surface reads the old model 300 m under its surface, so different DEMs or cell sizes
  don't put new near-surface cells into the old model's air.  ``depth="elevation"``
  matches absolute elevation instead.
* The value is the old model's log10 conductivity averaged over the target cell: each
  old cell counts by the volume it shares with the target cell, so a small target cell
  gets the old cell it sits in (exactly, when the meshes match) and a large one the
  volume average of the old cells it covers.  ``smooth`` > 0 also blurs the old cells'
  blocks with a Gaussian of ``smooth`` x the width / thickness of the old cell found
  there, as the Refine tab's smoothing; both are done in closed form (:func:`box_average`).
* Every old earth cell counts, fixed ones too (padding, mantle): they are part of the
  model.  Old air and sea don't.  Target cells beyond the old model keep their value.

Each filled cell gets a material id of its own (from ``first_id``), as GoFEM's inverted
models have, so the result can be inverted cell by cell.
"""
from __future__ import annotations

import os
import shutil

import numpy as np

from . import refine as R

#: neighbour pairs per batch, and the most one mapping may need
_PAIRS_PER_BATCH = 5_000_000
_PAIRS_LIMIT = 3_000_000_000


def _ramp_integral(u, s):
    """Antiderivative of the normal CDF ramp of width *s* at *u* (s = 0: max(u, 0))."""
    from scipy.special import ndtr

    out = np.maximum(u, 0.0)
    m = s > 0
    if m.any():
        z = u[m] / s[m]
        out[m] = s[m] * (z * ndtr(z) + np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi))
    return out


def _axis_weight(s_lo, s_hi, t_lo, t_hi, sigma):
    """Per axis: the share of the target interval [t_lo, t_hi] that the source interval
    [s_lo, s_hi] covers, after blurring it with a Gaussian of *sigma* (0: plain overlap)."""
    g = _ramp_integral
    return (g(s_hi - t_lo, sigma) - g(s_hi - t_hi, sigma) - g(s_lo - t_lo, sigma)
            + g(s_lo - t_hi, sigma)) / (t_hi - t_lo)


def box_average(t_centres, t_half, sh, sv, s_centres, s_half, s_values) -> np.ndarray:
    """Each target box's average of a field constant on the source boxes, the boxes first
    blurred by Gaussians of *sh* across and *sv* down (per target; 0 = none), divided by
    how much of the target the source boxes cover (so air doesn't count).  NaN where no
    source box is within reach."""
    from scipy.spatial import cKDTree

    stretch = float(np.median(t_half[:, 0] / np.maximum(t_half[:, 2], 1e-9)))
    scale = np.array([1.0, 1.0, stretch])
    reach = np.linalg.norm(t_half * scale, axis=1) + 3.0 * np.maximum(sh, stretch * sv)
    diag = np.linalg.norm(s_half * scale, axis=1)
    size_class = np.floor(np.log2(np.maximum(diag, 1e-9))).astype(int)
    num = np.zeros(len(t_centres))
    den = np.zeros(len(t_centres))
    q_all = t_centres * scale
    for cls in np.unique(size_class):
        members = np.flatnonzero(size_class == cls)
        tree = cKDTree(s_centres[members] * scale)
        radius = reach + diag[members].max()
        counts = np.asarray(tree.query_ball_point(q_all, radius, return_length=True))
        if counts.sum() > _PAIRS_LIMIT:
            raise ValueError(f"mapping would compare {int(counts.sum()):,} cell pairs, too many: "
                             "use less smoothing")
        first = 0
        while first < len(t_centres):
            last, total = first + 1, counts[first]
            while last < len(t_centres) and total + counts[last] <= _PAIRS_PER_BATCH:
                total += counts[last]
                last += 1
            if total:
                nbrs = tree.query_ball_point(q_all[first:last], radius[first:last])
                lens = np.fromiter((len(n) for n in nbrs), dtype="int64", count=last - first)
                row = np.repeat(np.arange(first, last), lens)
                col = members[np.fromiter((j for n in nbrs for j in n), dtype="int64", count=int(lens.sum()))]
                tc, th, sc, sh_ = t_centres[row], t_half[row], s_centres[col], s_half[col]
                w = (_axis_weight(sc[:, 0] - sh_[:, 0], sc[:, 0] + sh_[:, 0], tc[:, 0] - th[:, 0], tc[:, 0] + th[:, 0], sh[row])
                     * _axis_weight(sc[:, 1] - sh_[:, 1], sc[:, 1] + sh_[:, 1], tc[:, 1] - th[:, 1], tc[:, 1] + th[:, 1], sh[row])
                     * _axis_weight(sc[:, 2] - sh_[:, 2], sc[:, 2] + sh_[:, 2], tc[:, 2] - th[:, 2], tc[:, 2] + th[:, 2], sv[row]))
                num += np.bincount(row, w * s_values[col], minlength=len(t_centres))
                den += np.bincount(row, w, minlength=len(t_centres))
            first = last
    out = np.full(len(t_centres), np.nan)
    ok = den > 1e-9
    out[ok] = num[ok] / den[ok]
    return out


def frame_transform(source_georef, target_georef):
    """A function (north, east) in the target mesh -> (north, east) in the source mesh."""
    from pyproj import Transformer

    tr = Transformer.from_crs(target_georef.crs, source_georef.crs, always_xy=True)

    def move(north, east):
        e, n = tr.transform(np.asarray(east, dtype="float64") + target_georef.x0,
                            np.asarray(north, dtype="float64") + target_georef.y0)
        return np.asarray(n) - source_georef.y0, np.asarray(e) - source_georef.x0

    return move


def _surface_z(surf: R.SurfaceFaces, north, east) -> np.ndarray:
    """Surface z (down) at each point, from the nearest surface face's middle."""
    _d, i = surf.centres_tree().query(np.c_[north, east])
    return surf.z[i]


def remap(source_tria, source_materials: dict, source_georef, target_tria, target_materials: dict,
          target_georef, *, fill_ids=(2,), smooth: float = 0.25, depth: str = "surface",
          first_id: int | None = None, log=print) -> dict:
    """Give the target mesh's *fill_ids* cells the source model's conductivities.

    Changes *target_tria*'s material ids in place and adds the new ids to
    *target_materials*.  Returns ``cells`` (filled), ``kept`` (beyond the source model:
    unchanged), ``new_ids`` (id -> Material) and ``log10`` (the filled cells' values).
    """
    if depth not in ("surface", "elevation"):
        raise ValueError(f"unknown depth matching {depth!r}")
    if smooth < 0:
        raise ValueError("smoothing can't be negative")
    # the source model, as boxes
    s_top = [R.AIR_ID, *R.water_ids(source_materials)]
    s_cells, s_xyz, s_mat = R._cell_arrays(source_tria)
    earth = np.flatnonzero(~np.isin(s_mat, s_top))
    missing = sorted({int(i) for i in s_mat[earth]} - set(source_materials))
    if missing:
        raise ValueError(f"{len(missing)} material ids in the source model aren't in its .materials "
                         f"(e.g. {missing[:5]}). Are they from the same model and iteration?")
    cond = np.array([source_materials[int(i)].conductivity for i in s_mat[earth]])
    if (cond <= 0).any():
        raise ValueError("the source model has conductivities of zero or less")
    s_half = R._half_sizes(s_cells, earth)
    s_xyz, s_val = s_xyz[earth], np.log10(cond)
    s_surf = R.surface_faces(source_tria, s_top)
    log(f"source: {len(earth):,} earth cells, log10 conductivity {s_val.min():.2f} to {s_val.max():.2f}")

    # the target cells to fill
    t_top = [R.AIR_ID, *R.water_ids(target_materials)]
    t_cells, t_xyz, t_mat = R._cell_arrays(target_tria)
    fill = np.flatnonzero(np.isin(t_mat, [int(i) for i in fill_ids]) & ~np.isin(t_mat, t_top))
    if not len(fill):
        raise ValueError(f"the target mesh has no cells with ids {list(fill_ids)} to fill")
    t_half = R._half_sizes(t_cells, fill)

    north, east = frame_transform(source_georef, target_georef)(t_xyz[fill, 0], t_xyz[fill, 1])
    z = t_xyz[fill, 2]
    if depth == "surface":
        t_surf = R.surface_faces(target_tria, t_top)
        z = z - _surface_z(t_surf, t_xyz[fill, 0], t_xyz[fill, 1]) + _surface_z(s_surf, north, east)
    points = np.c_[north, east, z]

    # smoothing lengths: a share of the size of the old cell at each target centre
    from scipy.spatial import cKDTree

    _d, near = cKDTree(s_xyz).query(points)
    sh = smooth * 2.0 * s_half[near, :2].max(axis=1)
    sv = smooth * 2.0 * s_half[near, 2]
    value = box_average(points, t_half, sh, sv, s_xyz, s_half, s_val)

    ok = np.isfinite(value)
    if first_id is None:
        top = max(max(target_materials, default=0), int(t_mat.max()))
        first_id = max(1000, 10 * top + 1)          # well above the fixed ids, as GoFEM numbers them
    new_ids = {}
    for k, i in enumerate(fill[ok]):
        m = R.Material(first_id + k, target_materials[int(t_mat[i])].name, float(10.0 ** value[ok][k]))
        new_ids[m.id] = m
        t_cells[i].material_id = m.id
    target_materials.update(new_ids)
    kept = int((~ok).sum())
    log(f"filled {int(ok.sum()):,} cells (ids {first_id:,} to {first_id + int(ok.sum()) - 1:,}), "
        f"log10 conductivity {value[ok].min():.2f} to {value[ok].max():.2f}"
        + (f"; {kept:,} beyond the source model keep their value" if kept else ""))
    return {"cells": int(ok.sum()), "kept": kept, "new_ids": new_ids, "log10": value[ok]}


def output_paths(out_dir: str, stem: str = "model_mapped") -> dict:
    return {"tria": os.path.join(out_dir, stem + ".tria"), "vtu": os.path.join(out_dir, stem + ".vtu"),
            "materials": os.path.join(out_dir, stem + ".materials"), "mask": os.path.join(out_dir, stem + ".mask")}


def write_mapped(tria, materials_path: str, new_ids: dict, out_dir: str, stem: str = "model_mapped",
                 *, mask_fixed_ids=None, receivers: str = "", log=print) -> dict:
    """``<stem>.tria`` / ``.vtu``, the target's ``.materials`` with *new_ids* added, a mask if
    *mask_fixed_ids* is given (those ids fixed, the rest free), and a copy of the target's
    receivers (the mesh is unchanged, so they still fit)."""
    paths = output_paths(out_dir, stem)
    if os.path.abspath(paths["materials"]) == os.path.abspath(materials_path):
        raise ValueError("this would overwrite the target mesh's .materials: choose another output folder or name")
    os.makedirs(out_dir, exist_ok=True)
    tria.save(paths["tria"])
    tria.write(paths["vtu"], "vtu")
    shutil.copyfile(materials_path, paths["materials"])
    with open(paths["materials"], "rb") as fh:
        data = fh.read()
    with open(paths["materials"], "a") as fh:
        if data and not data.endswith(b"\n"):
            fh.write("\n")
        for m in new_ids.values():
            fh.write(f"{m.id}\t{m.name}\t{m.conductivity:.8g}\t0\n")
    if mask_fixed_ids is not None:
        fixed = {int(i) for i in mask_fixed_ids}
        with open(paths["mask"], "w") as fh:
            for c in tria.active_cells():
                fh.write("%i %i %i\n" % (c.level(), c.index(), 0 if c.material_id in fixed else 1))
    else:
        paths.pop("mask")
    if receivers:
        dest = os.path.join(out_dir, os.path.basename(receivers))
        if os.path.abspath(dest) != os.path.abspath(receivers):
            shutil.copyfile(receivers, dest)
        paths["receivers"] = dest
    log("wrote " + ", ".join(os.path.basename(p) for p in paths.values()))
    return paths


def run(spec: dict, out_dir: str = "", log=print, write: bool = True) -> dict:
    """Map a model onto another mesh, from a job description:

    ``tria``, ``materials`` and the frame keys of :func:`refine.resolve_georef` (``frame``,
    ``frame_project`` / ``frame_fit_edis`` + ``receivers`` / ``frame_crs`` ...) for the
    source model; the same under ``target_`` for the target mesh (``target_tria``,
    ``target_materials``, ``target_receivers``, ``target_frame``, ...); ``fill_ids``,
    ``smooth``, ``depth``, ``stem``, ``mask_fixed_ids`` (optional).
    """
    log(f"source {os.path.basename(spec['tria'])}")
    source = R.load_tria(spec["tria"])
    source_materials = R.read_materials(spec["materials"])
    source_georef, note = R.resolve_georef(spec)
    log("  frame " + note)
    tspec = {k[len("target_"):]: v for k, v in spec.items() if k.startswith("target_")}
    log(f"target {os.path.basename(tspec['tria'])}")
    target = R.load_tria(tspec["tria"])
    target_materials = R.read_materials(tspec["materials"])
    target_georef, tnote = R.resolve_georef(tspec)
    log("  frame " + tnote)
    n_target = target.n_active_cells()
    result = remap(source, source_materials, source_georef, target, target_materials, target_georef,
                   fill_ids=spec.get("fill_ids") or (2,), smooth=float(spec.get("smooth", 0.25)),
                   depth=spec.get("depth", "surface"), log=log)
    result.update(n_cells=n_target, cells_model=R.model_cells(target, target_materials),
                  frames=(note, tnote), smallest_m=target.minimal_cell_diameter(),
                  largest_m=target.maximal_cell_diameter())
    if write:
        result["paths"] = write_mapped(target, tspec["materials"], result["new_ids"], out_dir,
                                       spec.get("stem") or "model_mapped",
                                       mask_fixed_ids=spec.get("mask_fixed_ids"),
                                       receivers=tspec.get("receivers", ""), log=log)
    return result


def spec_from_info(info) -> dict:
    """Job description for :func:`run`, from a ``project.RefineInfo`` (workflow "map")."""
    spec = {k: getattr(info, k) for k in ("tria", "materials", "receivers", "frame", "frame_project",
                                          "frame_fit_edis", "frame_file", "frame_crs", "frame_x0",
                                          "frame_y0", "stem")}
    for k in ("tria", "materials", "receivers", "frame", "frame_project", "frame_file", "frame_fit_edis",
              "frame_crs", "frame_x0", "frame_y0"):
        spec["target_" + k] = getattr(info, "target_" + k)
    spec.update(fill_ids=list(info.fill_ids) or [2], smooth=float(info.map_smooth), depth=info.map_depth,
                mask_fixed_ids=list(info.fixed_ids) if info.write_mask else None)
    return spec
