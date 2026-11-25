"""Runtime state that is too heavy for the project file: the loaded model,
its receivers and the map transform.  No Qt -- the GUI and headless/batch
rendering share it.
"""
from __future__ import annotations

import os

import numpy as np

from . import georef as G
from . import slicing
from .model_source import ModelGrid, Receivers, get_source


class PlotSession:
    def __init__(self):
        self.model: ModelGrid | None = None
        self.receivers: Receivers = Receivers()
        self.maptr: G.MapTransform | None = None
        self.last_fit: G.FitResult | None = None
        self._layers: dict = {}            # read_key -> loaded layer
        self.house = None                  # project.house (HouseStyle) -- set by the owner

    # ------------------------------------------------------------------ #
    @property
    def ready(self) -> bool:
        """Model loaded *and* georeferenced -- everything can plot."""
        return self.model is not None and self.maptr is not None

    def load_model(self, info) -> str:
        """Load ``info.model_path`` (+ receivers); returns the receivers path used."""
        src = get_source(info.code)
        model = src.load(info.model_path)
        model.surface()                        # build now (worker thread), not mid-plot
        rec_path = info.receivers_path or src.find_receivers(info.model_path)
        receivers = src.read_receivers(rec_path) if rec_path and os.path.isfile(rec_path) else Receivers()
        self.model, self.receivers = model, receivers
        slicing.clear_cache()
        return rec_path

    # ------------------------------------------------------------------ #
    def solve_georef(self, g) -> G.Georef:
        """Compute ``g.crs/x0/y0`` from ``g.method`` and its inputs (mutates *g*)."""
        self.last_fit = None
        g.fit_rms_m = g.fit_max_m = -1.0
        g.fit_matched = 0
        if g.method == "mtinv":
            ref, _ = G.from_mtinv_project(g.mtinv_project)
        elif g.method == "manual":
            ref = G.manual(g.manual_crs, g.manual_x0, g.manual_y0)
        else:
            from ...backends import edi_io

            if not len(self.receivers):
                raise ValueError("no receivers loaded -- set the receivers file first")
            stations = edi_io.read_edi_headers(g.station_source)
            if not stations:
                raise ValueError(f"no EDIs found in {g.station_source!r}")
            if g.fit_crs and g.fit_crs != "auto":
                fit = G.fit_from_stations(self.receivers, stations, g.fit_crs)
            else:
                fit = G.auto_fit(self.receivers, stations)
            self.last_fit = fit
            ref = fit.georef
            g.fit_rms_m, g.fit_max_m, g.fit_matched = fit.rms_m, fit.max_m, fit.n_matched
        g.crs, g.x0, g.y0 = ref.crs, ref.x0, ref.y0
        self.apply_georef(g)
        return ref

    def apply_georef(self, g) -> None:
        """(Re)build the map transform from the stored result in *g*."""
        if not g.crs:
            self.maptr = None
            return
        self.maptr = G.MapTransform(G.Georef(g.crs, g.x0, g.y0), g.display_crs)

    # ------------------------------------------------------------------ #
    def stations_display(self) -> np.ndarray:
        """``(n, 2)`` station positions in the display CRS."""
        if not len(self.receivers) or self.maptr is None:
            return np.zeros((0, 2))
        E, N = self.maptr.forward(self.receivers.xyz[:, 0], self.receivers.xyz[:, 1])
        return np.c_[E, N]

    def extent(self, ext) -> tuple[float, float, float, float]:
        """``(x_min, x_max, y_min, y_max)`` in the display CRS for an
        :class:`~..project.ExtentInfo`."""
        if ext.mode == "custom" and len(ext.custom) == 4:
            return tuple(float(v) for v in ext.custom)
        st = self.stations_display()
        if ext.mode == "stations" and len(st):
            p = float(ext.padding_m)
            return (st[:, 0].min() - p, st[:, 0].max() + p,
                    st[:, 1].min() - p, st[:, 1].max() + p)
        # whole model -- the earth bounds transformed (corners + edge midpoints)
        b = self.model.earth_bounds()
        xs = np.array([b[0], b[1], b[1], b[0], 0.5 * (b[0] + b[1]), 0.5 * (b[0] + b[1]), b[0], b[1]])
        ys = np.array([b[2], b[2], b[3], b[3], b[2], b[3], 0.5 * (b[2] + b[3]), 0.5 * (b[2] + b[3])])
        E, N = self.maptr.forward(xs, ys)
        return float(E.min()), float(E.max()), float(N.min()), float(N.max())

    # ------------------------------------------------------------------ #
    def layer(self, info):
        """The loaded data of a :class:`~..project.LayerInfo` (cached)."""
        from . import layers as L

        key = L.read_key(info, self.maptr)
        hit = self._layers.get(key)
        if hit is None:
            hit = self._layers[key] = L.load(info, self.maptr)
        return hit

    def active_layers(self, layer_infos):
        """``[(info, loaded)]`` for enabled layers that load; failures are
        returned separately as ``[(info, exception)]``."""
        ok, bad = [], []
        for info in layer_infos:
            if not info.enabled or not info.path:
                continue
            try:
                ok.append((info, self.layer(info)))
            except Exception as exc:  # noqa: BLE001 -- one bad file must not stop a figure
                bad.append((info, exc))
        return ok, bad
