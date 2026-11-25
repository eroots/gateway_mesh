"""Read an inversion model into one code-independent representation.

Every inversion code stores its model differently (GoFEM: a deal.II ``.vtu``
with X = North and Z positive down; FEMTIC: a tetrahedral mesh + a resistivity
block file; ...).  A :class:`ModelSource` hides that and returns a
:class:`ModelGrid` in the *canonical local frame* the rest of the plotting code
uses:

  * ``x`` = easting, ``y`` = northing, ``z`` = elevation (positive **up**),
    all in metres relative to the model origin (the inversion's own local
    coordinates -- georeferencing to a real CRS is :mod:`.georef`'s job);
  * one value per cell, ``cell_data["log10_rho"]`` (log10 of resistivity in
    ohm-m);
  * air cells kept in the grid but flagged (``ModelGrid.is_air``) so the
    surface can be found and "air above" tests work.

Receiver files are code-specific too, so each source also knows how to read its
receivers into the same frame.

Add a new code by subclassing :class:`ModelSource` and registering it in
:data:`MODEL_SOURCES`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

#: conductivity at or below this is air (GoFEM writes 1e-8 S/m)
AIR_CONDUCTIVITY = 1e-7


@dataclass
class Receivers:
    """Station names + positions in the canonical local frame (E, N, Z-up)."""
    names: list[str] = field(default_factory=list)
    xyz: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def __len__(self) -> int:
        return len(self.names)


class ModelGrid:
    """An inversion model in the canonical local frame (see module docstring).

    ``grid`` is a ``pyvista.UnstructuredGrid`` holding *all* cells, air
    included; :attr:`earth` is the air-free subset (cached).  The surface
    (top of the earth cells) is built on first use by :meth:`surface`.
    """

    def __init__(self, grid, is_air: np.ndarray, source_path: str = "",
                 code: str = ""):
        self.grid = grid
        self.is_air = np.asarray(is_air, dtype=bool)
        self.source_path = source_path
        self.code = code
        self._earth = None
        self._surface = None

    # ------------------------------------------------------------------ #
    @property
    def earth(self):
        if self._earth is None:
            self._earth = self.grid.extract_cells(np.where(~self.is_air)[0])
        return self._earth

    @property
    def n_cells(self) -> int:
        return int(self.grid.n_cells)

    @property
    def n_earth_cells(self) -> int:
        return int((~self.is_air).sum())

    def log10_rho_range(self) -> tuple[float, float]:
        v = np.asarray(self.grid.cell_data["log10_rho"])[~self.is_air]
        return float(v.min()), float(v.max())

    def earth_bounds(self) -> tuple[float, ...]:
        return tuple(float(b) for b in self.earth.bounds)

    def surface(self) -> "Surface":
        if self._surface is None:
            self._surface = Surface.from_model(self)
        return self._surface


class Surface:
    """The top of the earth cells (topography / bathymetry as the mesh sees it).

    Built from the top faces of every earth cell whose centre has air (or
    nothing) directly above it.  Each deal.II cell owns its 8 points, so the
    usual "extract boundary faces" trick does not work (every face looks like
    a boundary) -- probing above each cell does, and is also correct across
    the hanging-node faces of an adaptively refined mesh.
    """

    def __init__(self, points: np.ndarray):
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        self.points = np.asarray(points, dtype="float64")
        self._lin = LinearNDInterpolator(self.points[:, :2], self.points[:, 2])
        self._near = NearestNDInterpolator(self.points[:, :2], self.points[:, 2])

    @classmethod
    def from_model(cls, model: ModelGrid) -> "Surface":
        g = model.grid
        conn = _hex_connectivity(g)
        corners = np.asarray(g.points)[conn]                     # (n, 8, 3)
        order = np.argsort(corners[:, :, 2], axis=1)
        top4 = np.take_along_axis(corners, order[:, 4:, None], axis=1)
        earth_idx = np.where(~model.is_air)[0]
        probe = top4[earth_idx].mean(axis=1) + np.array([0.0, 0.0, 1.0])
        cid = np.asarray(g.find_containing_cell(probe))
        above_air = (cid < 0) | model.is_air[np.maximum(cid, 0)]
        pts = top4[earth_idx[above_air]].reshape(-1, 3)
        pts = np.unique(np.round(pts, 3), axis=0)
        return cls(pts)

    def __call__(self, x, y) -> np.ndarray:
        """Surface elevation (m, local frame) at ``(x, y)``; nearest-neighbour
        outside the convex hull of the surface points."""
        x = np.asarray(x, dtype="float64")
        y = np.asarray(y, dtype="float64")
        z = self._lin(x, y)
        bad = np.isnan(z)
        if bad.any():
            z = np.where(bad, self._near(x, y), z)
        return z

    @property
    def z_range(self) -> tuple[float, float]:
        return float(self.points[:, 2].min()), float(self.points[:, 2].max())


def _hex_connectivity(grid) -> np.ndarray:
    """``(n_cells, 8)`` point ids -- every cell must be a VTK hexahedron (12)
    or voxel (11).  Only the corner *set* is used, never its ordering."""
    cells = np.asarray(grid.cells)
    if cells.size % 9 or not np.all(np.isin(np.asarray(grid.celltypes), (11, 12))):
        raise ValueError("expected an all-hexahedral mesh")
    return cells.reshape(-1, 9)[:, 1:]


# --------------------------------------------------------------------------- #
class ModelSource:
    """Base class: one per inversion code."""

    #: name shown in the GUI and stored in the project file
    code = ""
    #: file-dialog filter for the model file
    model_filter = "all files (*)"
    #: candidate receiver file names, looked for next to the model and one level up
    receiver_files: tuple[str, ...] = ()

    def load(self, path: str) -> ModelGrid:  # pragma: no cover - interface
        raise NotImplementedError

    def read_receivers(self, path: str) -> Receivers:  # pragma: no cover - interface
        raise NotImplementedError

    def find_receivers(self, model_path: str) -> str:
        """Best-guess receiver file for *model_path* ("" if none found).

        GoFEM inversion results usually live one directory below the mesh +
        receivers (``model1/ZK/model_ZK_31.vtu`` next to
        ``model1/receivers_projected.csv``), so look in both places.
        """
        here = os.path.dirname(os.path.abspath(model_path))
        for d in (here, os.path.dirname(here)):
            for name in self.receiver_files:
                cand = os.path.join(d, name)
                if os.path.isfile(cand):
                    return cand
        return ""


class GoFEMVtuSource(ModelSource):
    """GoFEM inversion output (deal.II ``DataOut`` ``.vtu``).

    deal.II's mesh frame is X = North, Y = East, Z = depth (positive down);
    ``conductivity`` is point data but constant per cell (each cell owns its
    8 points).  Mapping (x, y, z) -> (y, x, -z) is a swap plus a sign flip --
    two reflections, i.e. a proper rotation -- so cell orientation (and VTK's
    point-in-cell tests) are preserved.
    """

    code = "GoFEM"
    model_filter = "GoFEM model (*.vtu);;all files (*)"
    receiver_files = ("receivers_refined-projected.csv", "receivers_projected.csv",)

    def load(self, path: str) -> ModelGrid:
        import pyvista as pv

        raw = pv.read(path)
        conn = _hex_connectivity(raw)
        if "conductivity" in raw.cell_data:
            sigma = np.asarray(raw.cell_data["conductivity"], dtype="float64")
        elif "conductivity" in raw.point_data:
            sigma = np.asarray(raw.point_data["conductivity"], dtype="float64")[conn[:, 0]]
        else:
            raise ValueError(f"{os.path.basename(path)} has no 'conductivity' array "
                             f"(found: {', '.join(raw.array_names) or 'none'})")
        p = np.asarray(raw.points, dtype="float64")
        # voxels (11) assume axis-ordered corners, which the axis swap below
        # breaks -- renumber them as hexahedra (12) first
        celltypes = np.asarray(raw.celltypes).copy()
        conn = conn.copy()
        vox = celltypes == 11
        if vox.any():
            conn[vox] = conn[vox][:, [0, 1, 3, 2, 4, 5, 7, 6]]
            celltypes[vox] = 12
        cells = np.c_[np.full(len(conn), 8), conn].ravel()
        grid = pv.UnstructuredGrid(cells, celltypes, np.c_[p[:, 1], p[:, 0], -p[:, 2]])
        is_air = sigma <= AIR_CONDUCTIVITY
        with np.errstate(divide="ignore"):
            grid.cell_data["log10_rho"] = -np.log10(np.maximum(sigma, 1e-30))
        return ModelGrid(grid, is_air, source_path=path, code=self.code)

    def read_receivers(self, path: str) -> Receivers:
        """``receivers_projected.csv``: ``Dipole <name> 1 <x=N> <y=E> <z=depth>``
        per line (written by ``backends/mesh.py``)."""
        names, xyz = [], []
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 6:
                    continue
                try:
                    xn, ye, zd = (float(v) for v in parts[3:6])
                except ValueError:
                    continue
                names.append(parts[1])
                xyz.append([ye, xn, -zd])
        return Receivers(names, np.asarray(xyz, dtype="float64").reshape(-1, 3))


#: code name -> source class.  Add FEMTIC / ModEM here.
MODEL_SOURCES: dict[str, type[ModelSource]] = {
    GoFEMVtuSource.code: GoFEMVtuSource,
}


def get_source(code: str) -> ModelSource:
    try:
        return MODEL_SOURCES[code]()
    except KeyError:
        raise ValueError(f"no model reader for code {code!r} "
                         f"(available: {', '.join(MODEL_SOURCES)})") from None
