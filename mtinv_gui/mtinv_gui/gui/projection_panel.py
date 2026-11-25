"""Step 3 -- project + interpolate the DEM into the local UTM frame."""
from __future__ import annotations

import os
import traceback

from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QVBoxLayout,
)

from ..backends import edi_io, projection as P, topography as topo
from .mpl_canvas import MplCanvas
from .panel_base import BasePanel
from .worker import run_async


class ProjectionPanel(BasePanel):
    title = "3 - Project DEM"

    def build(self) -> None:
        root = QHBoxLayout(self)
        left = QVBoxLayout()

        box = QGroupBox("Local grid")
        form = QFormLayout(box)
        self.frame_lbl = QLabel("-")
        self.sample = QDoubleSpinBox()
        self.sample.setRange(1, 100000)
        self.sample.setValue(self.project.projection.dem_sample_rate_m)
        self.sample.setSuffix(" m")
        self.method = QComboBox()
        self.method.addItems(["nearest", "linear", "cubic"])
        self.sample.valueChanged.connect(self._update_spacing)
        self.reg_sample = QDoubleSpinBox()
        self.reg_sample.setRange(10, 100000)
        self.reg_sample.setValue(self.project.projection.regional_sample_rate_m)
        self.reg_sample.setSuffix(" m")
        self.reg_sample.setToolTip(
            "Posting of the regional DEM's grid (Topography tab), which covers the mesh\n"
            "padding where the detailed grid doesn't reach. Coarse is fine there.")
        form.addRow("local frame", self.frame_lbl)
        form.addRow("grid posting", self.sample)
        form.addRow("regional grid posting", self.reg_sample)
        form.addRow("griddata method", self.method)
        left.addWidget(box)

        self.spacing_lbl = QLabel("-")
        self.spacing_lbl.setWordWrap(True)
        self.spacing_lbl.setStyleSheet("font-size: 11px;")
        left.addWidget(self.spacing_lbl)

        btns = QHBoxLayout()
        self.run_btn = QPushButton("Project + interpolate + save .npz")
        self.run_btn.clicked.connect(self._run)
        self.plot_btn = QPushButton("Plot DEM")
        self.plot_btn.clicked.connect(self._plot_clicked)
        btns.addWidget(self.run_btn)
        btns.addWidget(self.plot_btn)
        left.addLayout(btns)

        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.hide()
        left.addWidget(self.busy)

        self._pdem = None
        self._receivers = None
        self._loading_dem = False

        self.stats = QLabel("-")
        self.stats.setWordWrap(True)
        left.addWidget(self.stats)
        self.note = QLabel()
        self.note.setWordWrap(True)
        left.addWidget(self.note)
        left.addStretch(1)
        root.addLayout(left, 2)

        self.canvas = MplCanvas(figsize=(5, 6))
        root.addWidget(self.canvas, 3)

    # ------------------------------------------------------------------ #
    def _run(self):
        p = self.project.projection
        p.dem_sample_rate_m = self.sample.value()
        p.regional_sample_rate_m = self.reg_sample.value()
        p.grid_method = self.method.currentText()
        if not (p.origin_lon or p.origin_lat):
            self.log("set the local frame / origin on the Survey tab first")
            return
        src = self.project.topo.resampled_path
        if not src or not os.path.isfile(src):
            self.log("run the Topography step first (need topo_big.npz)")
            return
        stations = edi_io.read_edi_headers(self.project.survey.edi_dir)
        excl = set(self.project.survey.excluded)
        stations = [s for s in stations if s.station_id not in excl]
        out = os.path.join(self.project.workdir or ".", "DEM_projected_gridded.npz")
        frame = P.frame_from_projection(p)
        self.log("projecting into: " + P.frame_label(frame))
        t = self.project.topo
        reg_src = t.regional_resampled_path if t.regional_source != "none" else ""
        if reg_src and not os.path.isfile(reg_src):
            self.log("the regional DEM isn't resampled yet (Topography tab) -- projecting the "
                     "detailed one only")
            reg_src = ""

        def job():
            dem = topo.load_dem_npz(src)
            pdem = P.project_dem(dem, frame, p.dem_sample_rate_m, p.grid_method)
            if reg_src:
                pdem.regional = P.project_dem(topo.load_dem_npz(reg_src), frame,
                                              p.regional_sample_rate_m, p.grid_method).topo_cart
            receivers, names = P.project_stations(stations, frame)
            P.save_projected_dem(pdem, out)
            return pdem, receivers, names, out

        self.busy.show()
        self.run_btn.setEnabled(False)
        self.plot_btn.setEnabled(False)
        self.log("projecting DEM + stations into local frame")
        run_async(self, job, self._after, self._error)

    def _after(self, result):
        self.busy.hide()
        self.run_btn.setEnabled(True)
        self.plot_btn.setEnabled(True)
        pdem, receivers, names, out = result
        self.project.projection.projected_dem_path = out
        self._pdem, self._receivers = pdem, receivers
        sp = P.station_spacing_stats(receivers)
        ext = pdem.extent_local
        reg_txt = ""
        if pdem.regional is not None:
            r = pdem.regional
            reg_txt = (f"regional grid {r.shape[1]} x {r.shape[0]}: x [{r[..., 0].min()/1e3:.0f}, "
                       f"{r[..., 0].max()/1e3:.0f}] km, y [{r[..., 1].min()/1e3:.0f}, "
                       f"{r[..., 1].max()/1e3:.0f}] km\n")
        self.stats.setText(
            f"topo_cart {pdem.topo_cart.shape}  saved {os.path.basename(out)}\n" + reg_txt +
            f"local extent x [{ext['x_min']/1e3:.1f}, {ext['x_max']/1e3:.1f}] km, "
            f"y [{ext['y_min']/1e3:.1f}, {ext['y_max']/1e3:.1f}] km, "
            f"z [{ext['z_min']:.0f}, {ext['z_max']:.0f}] m\n"
            f"station spacing (m): mean {sp['mean_m']:.0f}, median {sp['median_m']:.0f}, "
            f"min {sp['min_m']:.0f}, max {sp['max_m']:.0f}")
        self._plot(pdem, receivers)
        self.notify_changed()

    def _error(self, exc: Exception):
        self._loading_dem = False
        self.busy.hide()
        self.run_btn.setEnabled(True)
        self.plot_btn.setEnabled(True)
        self.log(f"ERROR: {exc}")
        self.note.setText("<pre>" + "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]) + "</pre>")

    def _plot(self, pdem, receivers):
        if getattr(self.canvas, "ax", None) is None:
            return
        import numpy as np

        self.canvas.clear()
        ax = self.canvas.ax
        tc = pdem.topo_cart                       # [easting, northing, elev]
        grids = ([pdem.regional] if pdem.regional is not None else []) + [tc]
        vmin = min(float(np.nanmin(g[:, :, 2])) for g in grids)
        vmax = max(float(np.nanmax(g[:, :, 2])) for g in grids)
        for g in grids:
            step = max(int(max(g.shape[:2]) // 600), 1)
            pcm = ax.pcolormesh(g[::step, ::step, 0] / 1e3, g[::step, ::step, 1] / 1e3,
                                g[::step, ::step, 2], shading="auto", cmap="terrain",
                                vmin=vmin, vmax=vmax)
        if pdem.regional is not None:
            x0, x1 = tc[:, :, 0].min() / 1e3, tc[:, :, 0].max() / 1e3
            y0, y1 = tc[:, :, 1].min() / 1e3, tc[:, :, 1].max() / 1e3
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], "k--", lw=0.8)
        self.canvas.figure.colorbar(pcm, ax=ax, orientation="horizontal",
                                    shrink=0.8, pad=0.08, label="elevation (m)")
        # receivers are [easting, northing, 0] here (project_stations default) --
        # same axes as the DEM.  mesh.py re-projects with swap_xy=True for the
        # GoFEM mesh frame; that swap does not belong in this map.
        r = np.asarray(receivers).reshape(-1, 3) if len(receivers) else np.zeros((0, 3))
        if r.size:
            ax.scatter(r[:, 0] / 1e3, r[:, 1] / 1e3, s=8, c="r")
        ax.set_xlabel("easting - origin (km)")
        ax.set_ylabel("northing - origin (km)")
        ax.set_title("projected DEM + receivers" if r.size else "projected DEM (no receivers)")
        # clip the view to the extent of the plotted data (DEM covers the receivers)
        outer = pdem.regional if pdem.regional is not None else tc
        x_all = np.concatenate([outer[:, :, 0].ravel(), r[:, 0]]) / 1e3
        y_all = np.concatenate([outer[:, :, 1].ravel(), r[:, 1]]) / 1e3
        ax.set_xlim(float(x_all.min()), float(x_all.max()))
        ax.set_ylim(float(y_all.min()), float(y_all.max()))
        ax.set_aspect("equal", adjustable="box")
        self.canvas.redraw()

    # ------------------------------------------------------------------ #
    def _saved_projection_source(self):
        p = self.project.projection
        if ((p.origin_lon or p.origin_lat) and p.projected_dem_path
                and os.path.isfile(p.projected_dem_path)):
            return p.projected_dem_path
        return None

    def _plot_clicked(self):
        if self._pdem is not None:
            self._plot(self._pdem, self._receivers)
            return
        self._load_saved_projection(on_missing=lambda: self.log(
            "no projected DEM yet -- run 'Project + interpolate' first"))

    def _load_saved_projection(self, on_missing=None):
        """Load + plot the project's already-saved projected DEM, without recomputing."""
        if self._loading_dem:
            return
        p = self.project.projection
        src = self._saved_projection_source()
        if not src:
            if on_missing:
                on_missing()
            return
        edi_dir = self.project.survey.edi_dir
        stations = (edi_io.read_edi_headers(edi_dir)
                   if edi_dir and os.path.exists(edi_dir) else [])
        excl = set(self.project.survey.excluded)
        stations = [s for s in stations if s.station_id not in excl]

        frame = P.frame_from_projection(p)

        def job():
            pdem = P.load_projected_dem(src, p.epsg)
            receivers, names = P.project_stations(stations, frame)
            return pdem, receivers

        self._loading_dem = True
        self.busy.show()
        self.run_btn.setEnabled(False)
        self.plot_btn.setEnabled(False)
        self.log(f"loading saved projected DEM {src}")
        run_async(self, job, self._after_load_saved, self._error)

    def _after_load_saved(self, result):
        self._loading_dem = False
        self.busy.hide()
        self.run_btn.setEnabled(True)
        self.plot_btn.setEnabled(True)
        pdem, receivers = result
        self._pdem, self._receivers = pdem, receivers
        self._plot(pdem, receivers)

    def refresh(self) -> None:
        p = self.project.projection
        if p.origin_lon or p.origin_lat:
            self.frame_lbl.setText(P.frame_label(P.frame_from_projection(p)))
        else:
            self.frame_lbl.setText("set the local frame on the Survey tab first")
        self.sample.setValue(p.dem_sample_rate_m)
        self.reg_sample.setValue(p.regional_sample_rate_m)
        self.reg_sample.setEnabled(self.project.topo.regional_source != "none")
        self.method.setCurrentText(p.grid_method)
        self._update_spacing()
        self.note.setText(
            "Projects the DEM and stations into the model frame set on the "
            "Survey tab (a map projection to metres, origin 0,0 at the survey "
            "centre for the Local options). Map is easting/northing, aligned with "
            "the DEM; the X/Y swap for the GoFEM mesh (X = north) happens later, "
            "only when the mesh is built.")
        if self._pdem is None:
            self._load_saved_projection()   # silently -- nothing to plot yet is not an error here

    def _update_spacing(self):
        sp_path = self.project.survey.edi_dir
        if not (sp_path and os.path.exists(sp_path)):
            self.spacing_lbl.setText("")
            return
        sts = edi_io.read_edi_headers(sp_path)
        if len(sts) < 2:
            self.spacing_lbl.setText("")
            return
        sp = edi_io.station_spacing_geographic(sts, set(self.project.survey.excluded))
        cur = self.sample.value()
        self.spacing_lbl.setText(
            f"site spacing (nearest-neighbour): min {sp['min_m']/1e3:.2f} / "
            f"median {sp['median_m']/1e3:.2f} / max {sp['max_m']/1e3:.2f} km   "
            f"-- grid posting {cur:.0f} m is ~1/{max(sp['median_m']/cur, 0.01):.0f} "
            f"of the median spacing")
