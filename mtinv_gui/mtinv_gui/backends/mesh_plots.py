"""Quick matplotlib views of a deal.II mesh, for notebooks and scripts (no Qt).

A plan slice at a depth and a vertical section through the earth cells, coloured by
resistivity (from the materials) or by cell size, with the cell outlines drawn so the
refinement is visible::

    cells = mesh_cells(tria, materials)
    plot_plan(cells, depth_m=1000, stations=receivers)
    plot_section(cells, north_m=0)

Positions are in km in the model frame, drawn east to the right and north up (the mesh
itself has x = north).  Depths are below sea level, positive down.
"""
from __future__ import annotations

import numpy as np

from . import refine as R


def mesh_cells(tria, materials=None) -> R.ModelCells:
    """The earth cells of *tria* as arrays (:class:`refine.ModelCells`).

    *materials*: ``{id: (name, conductivity)}`` as the mesh step builds it, or
    ``refine.read_materials`` output, or a ``.materials`` path; omit it to colour by
    cell size only.
    """
    if isinstance(materials, str):
        materials = R.read_materials(materials)
    elif materials:
        materials = {int(i): (m if isinstance(m, R.Material) else R.Material(int(i), m[0], float(m[1])))
                     for i, m in materials.items()}
    return R.model_cells(tria, materials)


def _values(cells, sel, color_by):
    if color_by == "size":
        return np.log10(2e-3 * np.maximum(cells.half_north[sel], cells.half_east[sel])), "log10 cell width (km)"
    return cells.log10_rho[sel], "log10 resistivity (ohm-m)"


def _draw(ax, rects, values, label, cmap, vmin, vmax, edges):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    verts = [[(x0, y0), (x1, y0), (x1, y1), (x0, y1)] for x0, x1, y0, y1 in rects]
    pc = PolyCollection(verts, cmap=cmap, edgecolors="0.2" if edges else "face",
                        linewidths=0.15 if edges else 0.0)
    pc.set_array(np.ma.masked_invalid(values))
    if vmin is not None or vmax is not None:
        pc.set_clim(vmin, vmax)
    ax.add_collection(pc)
    plt.colorbar(pc, ax=ax, label=label, shrink=0.8)
    return pc


def plot_plan(cells: R.ModelCells, depth_m: float, *, ax=None, stations=None,
              extent_km=None, color_by: str = "resistivity", cmap=None,
              vmin=None, vmax=None, edges: bool = True, title: str | None = None):
    """Earth cells at *depth_m* below sea level, seen from above.

    *stations*: ``[[north, east, ...], ...]`` in metres (the receivers).  *extent_km*:
    ``(east_min, east_max, north_min, north_max)``; default is the stations plus a
    margin, or the whole slice.  *color_by*: ``"resistivity"`` or ``"size"``.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _fig, ax = plt.subplots(figsize=(7, 6))
    rects_ne, _ = R.plan_slice(cells, depth_m)
    at = (cells.z_top <= depth_m) & (depth_m < cells.z_bottom)
    values, label = _values(cells, at, color_by)
    rects = rects_ne[:, [2, 3, 0, 1]] / 1e3          # east0, east1, north0, north1 in km
    cmap = cmap or ("turbo_r" if color_by == "resistivity" else "viridis")
    _draw(ax, rects, values, label, cmap, vmin, vmax, edges)
    if stations is not None and len(stations):
        st = np.asarray(stations, dtype="float64")
        ax.plot(st[:, 1] / 1e3, st[:, 0] / 1e3, "kv", ms=5, mfc="w", label="stations")
    if extent_km is None:
        if stations is not None and len(stations):
            pad = 0.5 * max(np.ptp(st[:, 1]), np.ptp(st[:, 0])) / 1e3 + 20.0
            cx, cy = 0.5 * (st[:, 1].min() + st[:, 1].max()) / 1e3, 0.5 * (st[:, 0].min() + st[:, 0].max()) / 1e3
            extent_km = (cx - pad, cx + pad, cy - pad, cy + pad)
        elif len(rects):
            extent_km = (rects[:, 0].min(), rects[:, 1].max(), rects[:, 2].min(), rects[:, 3].max())
    if extent_km is not None:
        ax.set_xlim(extent_km[0], extent_km[1])
        ax.set_ylim(extent_km[2], extent_km[3])
    ax.set_aspect("equal")
    ax.set_xlabel("east (km)")
    ax.set_ylabel("north (km)")
    ax.set_title(title or f"{at.sum():,} cells at {depth_m / 1e3:g} km below sea level")
    return ax


def plot_section(cells: R.ModelCells, *, north_m: float | None = None,
                 east_m: float | None = None, ax=None, stations=None, station_offset_m: float = 10e3,
                 range_km=None, depth_km=(-3.0, 50.0), color_by: str = "resistivity",
                 cmap=None, vmin=None, vmax=None, edges: bool = True, title: str | None = None):
    """A vertical slice through the earth cells: an east-west one at ``north_m``, or a
    north-south one at ``east_m``.  *range_km* limits the horizontal axis; *depth_km*
    is (top, bottom), negative above sea level.  Stations within *station_offset_m* of
    the section are drawn.  Cell tops and bottoms are their mean heights, so a fitted
    surface shows as steps."""
    import matplotlib.pyplot as plt

    if (north_m is None) == (east_m is None):
        raise ValueError("give exactly one of north_m (an east-west section) or east_m")
    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 4))
    if north_m is not None:
        at = np.abs(cells.north - north_m) <= cells.half_north
        along, half, axis_name, where = cells.east, cells.half_east, "east (km)", f"north = {north_m / 1e3:.1f} km"
    else:
        at = np.abs(cells.east - east_m) <= cells.half_east
        along, half, axis_name, where = cells.north, cells.half_north, "north (km)", f"east = {east_m / 1e3:.1f} km"
    rects = np.c_[along[at] - half[at], along[at] + half[at], cells.z_top[at], cells.z_bottom[at]] / 1e3
    values, label = _values(cells, at, color_by)
    cmap = cmap or ("turbo_r" if color_by == "resistivity" else "viridis")
    _draw(ax, rects, values, label, cmap, vmin, vmax, edges)
    if stations is not None and len(stations):
        st = np.asarray(stations, dtype="float64")
        col, off = (1, np.abs(st[:, 0] - north_m)) if north_m is not None else (0, np.abs(st[:, 1] - east_m))
        st = st[off <= station_offset_m]
        ax.plot(st[:, col] / 1e3, st[:, 2] / 1e3 if st.shape[1] > 2 else np.zeros(len(st)),
                "kv", ms=5, mfc="w")
    if range_km is None and len(rects):
        range_km = (rects[:, 0].min(), rects[:, 1].max())
    if range_km is not None:
        ax.set_xlim(*range_km)
    ax.set_ylim(depth_km[1], depth_km[0])
    ax.set_xlabel(axis_name)
    ax.set_ylabel("depth below sea level (km)")
    ax.set_title(title or f"section at {where}")
    return ax
