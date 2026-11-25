"""Preview dialog for an 'Along Coastline' refinement stage.

Shows the DEM (the detailed grid over the regional one, if there is one), the raw
coastline (grey), the cleaned-up coastline that will actually be refined around (red),
the refinement points (black dots) and land below sea level (purple outline), with
live simplify / small-island controls so the user can see what gets smoothed
away before building anything.  Pure matplotlib + numpy -- no deal.II.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout,
)

from ..backends import ocean as O
from .mpl_canvas import MplCanvas


def _dspin(lo, hi, step, dec, val, suffix=""):
    from PyQt5.QtWidgets import QDoubleSpinBox

    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(dec)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


class CoastPreviewDialog(QDialog):
    """``dem``: the projected DEM as a ``mesh_dem.MeshDem`` (or one grid, (n, n, 3)
    [easting, northing, elev]); ``stage``: the coastline stage dict; ``stations_xy``:
    mesh-frame ``[north, east]`` rows; ``half_extent_m``: ``(north_half, east_half)`` of
    the model footprint."""

    def __init__(self, dem, stage: dict, stations_xy=None,
                 half_extent_m=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Coastline Preview")
        self.resize(900, 760)
        if not hasattr(dem, "raw_coastlines"):
            from ..backends.mesh_dem import MeshDem

            dem = MeshDem([dem])
        self._dem = dem
        self._stations = np.asarray(stations_xy, dtype="float64").reshape(-1, 2) \
            if stations_xy is not None and len(stations_xy) else np.zeros((0, 2))
        self._half = half_extent_m
        self._raw = dem.raw_coastlines()

        lay = QVBoxLayout(self)
        form = QFormLayout()
        self.simplify = _dspin(0, 200, 0.5, 2, float(stage.get("simplify_km", 0.0)), " km")
        self.simplify.setSpecialValueText("Off")
        self.min_area = _dspin(0, 1e6, 10, 0, float(stage.get("min_area_km2", 0.0)), " km²")
        self.min_area.setSpecialValueText("Off")
        self.radius = _dspin(0.1, 500, 1, 2, float(stage.get("radius_km", 10.0)), " km")
        form.addRow("Simplify Tolerance", self.simplify)
        form.addRow("Drop Loops Smaller Than", self.min_area)
        form.addRow("Refine Radius / Spacing", self.radius)
        lay.addLayout(form)

        self.info = QLabel("-")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: gray; font-size: 11px;")
        lay.addWidget(self.info)

        self.canvas = MplCanvas(self, figsize=(7, 6))
        lay.addWidget(self.canvas, 1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Apply To Stage")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._draw)
        for w in (self.simplify, self.min_area, self.radius):
            w.valueChanged.connect(lambda *_: self._timer.start())
        QTimer.singleShot(0, self._draw)

    # ------------------------------------------------------------------ #
    def values(self) -> dict:
        return {"simplify_km": self.simplify.value(), "min_area_km2": self.min_area.value(),
                "radius_km": self.radius.value()}

    def _draw(self):
        ax = getattr(self.canvas, "ax", None)
        if ax is None:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            km = 1e3
            lines, info = O.process_coastlines(
                None, self.simplify.value() * km, self.min_area.value() * 1e6,
                raw=self._raw)
            pts, pinfo = O.coastline_points(lines, self.radius.value() * km)
            self.canvas.clear()
            ax = self.canvas.ax

            grids = self._dem.grids[::-1]                 # the regional grid first, under
            lo = min(float(np.nanmin(g[:, :, 2])) for g in grids)
            hi = max(float(np.nanmax(g[:, :, 2])) for g in grids)
            norm = None
            if lo < 0.0 < hi:
                from matplotlib.colors import TwoSlopeNorm

                norm = TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)
            for k, g in enumerate(grids):
                step = max(int(max(g.shape[0], g.shape[1]) // 300), 1)
                X, Y, Z = (g[::step, ::step, i] for i in range(3))
                ax.pcolormesh(X / km, Y / km, Z, cmap="terrain", shading="auto", norm=norm,
                              vmin=None if norm else lo, vmax=None if norm else hi)
                inland = self._dem.inland_masks[len(grids) - 1 - k][::step, ::step]
                if inland.any():
                    ax.contour(X / km, Y / km, inland.astype(float), [0.5],
                               colors="purple", linewidths=1.0)
            if len(grids) > 1:
                e0, e1, n0, n1 = self._dem.extents[0]
                ax.plot(np.array([e0, e1, e1, e0, e0]) / km, np.array([n0, n0, n1, n1, n0]) / km,
                        ":", color="0.3", lw=0.8)
            for p, _closed in self._raw:
                ax.plot(p[:, 0] / km, p[:, 1] / km, color="0.45", lw=0.6)
            for p, _closed in lines:
                ax.plot(p[:, 0] / km, p[:, 1] / km, color="red", lw=1.4)
            if len(pts):
                ax.plot(pts[:, 0] / km, pts[:, 1] / km, "k.", ms=3)
            if len(self._stations):
                ax.plot(self._stations[:, 1] / km, self._stations[:, 0] / km,
                        "^", color="tab:blue", ms=4, mec="white", mew=0.4)
            if self._half:
                hn, he = self._half
                ax.plot(np.array([-he, he, he, -he, -he]) / km,
                        np.array([-hn, -hn, hn, hn, -hn]) / km, "k--", lw=0.8)
            ax.set_aspect("equal")
            ax.set_xlabel("East (km)")
            ax.set_ylabel("North (km)")
            ax.set_title("grey = raw coastline   red = refined coastline   dots = refinement points"
                         "   purple = land below sea level (not sea)", fontsize=8)
            self.canvas.redraw()

            self.info.setText(
                "%d of %d lines kept  ·  %d → %d vertices  ·  %.0f km of coast  ·  "
                "%d refinement points every %.1f km%s"
                % (info["n_lines"], info["n_lines_raw"], info["n_points_raw"],
                   info["n_points"], info["length_m"] / km, pinfo["n_points"],
                   pinfo["spacing_m"] / km,
                   "  (spacing widened to cap the point count)"
                   if pinfo["spacing_widened"] else ""))
        finally:
            QApplication.restoreOverrideCursor()
