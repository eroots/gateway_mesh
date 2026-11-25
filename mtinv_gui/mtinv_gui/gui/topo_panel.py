"""Step 2 -- download / load topography + bathymetry, clip and resample."""
from __future__ import annotations

import os
import traceback

from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QVBoxLayout,
)

from .. import config
from ..backends import edi_io
from ..backends import topography as topo
from .mpl_canvas import MplCanvas
from .panel_base import BasePanel
from .worker import run_async


class TopoPanel(BasePanel):
    title = "2 - Topography"

    def build(self) -> None:
        root = QHBoxLayout(self)
        left = QVBoxLayout()

        src_box = QGroupBox("Source")
        form = QFormLayout(src_box)
        self.provider = QComboBox()
        self.provider.addItems(config.TOPO_PROVIDERS)
        self.provider.currentTextChanged.connect(self._on_provider)
        self.demtype = QComboBox()
        self.demtype.addItems(config.OPENTOPO_DEMS)
        self.local_path = QLineEdit(self.project.topo.local_path)
        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.local_path, 1)
        row.addWidget(browse)
        form.addRow("provider", self.provider)
        form.addRow("OpenTopo dataset", self.demtype)
        form.addRow("local DEM", row)
        left.addWidget(src_box)

        out_row = QHBoxLayout()
        self.out_dir = QLineEdit(self.project.topo.out_dir)
        self.out_dir.setPlaceholderText("(blank -> project working directory)")
        out_browse = QPushButton("Browse...")
        out_browse.clicked.connect(self._browse_out)
        out_row.addWidget(self.out_dir, 1)
        out_row.addWidget(out_browse)
        of = QFormLayout()
        of.addRow("save DEM + .npz to", out_row)
        left.addLayout(of)

        clip_box = QGroupBox("Clip + resample")
        cf = QFormLayout(clip_box)
        self.pad = QDoubleSpinBox()
        self.pad.setRange(0, 100000)
        self.pad.setValue(self.project.topo.pad_arcsec)
        self.pad.setSuffix(" arcsec")
        self.sample = QDoubleSpinBox()
        self.sample.setRange(0.1, 3600)
        self.sample.setValue(self.project.topo.sample_arcsec)
        self.sample.setSuffix(" arcsec")
        self.pad.valueChanged.connect(self._update_info)
        self.sample.valueChanged.connect(self._update_info)
        cf.addRow("pad around survey bbox", self.pad)
        cf.addRow("target posting", self.sample)
        left.addWidget(clip_box)

        reg_box = QGroupBox("Regional DEM (covers the mesh padding)")
        rf = QFormLayout(reg_box)
        self.reg_provider = QComboBox()
        self.reg_provider.addItems(["none"] + list(config.TOPO_PROVIDERS))
        self.reg_provider.setToolTip(
            "A coarse DEM reaching across the whole mesh, padding included. The detailed DEM\n"
            "above is used wherever it reaches; this one everywhere else. Without it, the mesh\n"
            "repeats the detailed DEM's edge values out to its own edge (with an ocean, the\n"
            "sea floor at the edge fills the padding).")
        self.reg_provider.currentTextChanged.connect(self._on_reg_provider)
        self.reg_local = QLineEdit(self.project.topo.regional_local_path)
        reg_browse = QPushButton("Browse...")
        reg_browse.clicked.connect(self._browse_regional)
        reg_row = QHBoxLayout()
        reg_row.addWidget(self.reg_local, 1)
        reg_row.addWidget(reg_browse)
        self.reg_pad = QDoubleSpinBox()
        self.reg_pad.setRange(0, 20000)
        self.reg_pad.setDecimals(0)
        self.reg_pad.setValue(self.project.topo.regional_pad_km)
        self.reg_pad.setSuffix(" km")
        self.reg_sample = QDoubleSpinBox()
        self.reg_sample.setRange(1, 3600)
        self.reg_sample.setValue(self.project.topo.regional_sample_arcsec)
        self.reg_sample.setSuffix(" arcsec")
        self.reg_pad.valueChanged.connect(self._update_info)
        self.reg_sample.valueChanged.connect(self._update_info)
        rf.addRow("provider", self.reg_provider)
        rf.addRow("local DEM", reg_row)
        rf.addRow("pad around survey bbox", self.reg_pad)
        rf.addRow("target posting", self.reg_sample)
        self.reg_hint = QLabel("-")
        self.reg_hint.setWordWrap(True)
        self.reg_hint.setStyleSheet("color: gray; font-size: 11px;")
        rf.addRow(self.reg_hint)
        left.addWidget(reg_box)

        self.info = QLabel("-")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("font-size: 11px;")
        left.addWidget(self.info)
        self.size_info = QLabel("-")
        self.size_info.setWordWrap(True)
        self.size_info.setStyleSheet("font-size: 11px;")
        self.size_info.setToolTip(
            "Estimated before anything is fetched. The download covers the survey box plus the\n"
            "pad, at the provider's own posting (GMRT picks one from the box size, about 1000\n"
            "nodes a side; OpenTopography uses the dataset's). 'target posting' only sets the\n"
            "resampled .npz, so a fine posting makes that file large, not the download.")
        left.addWidget(self.size_info)

        btns = QHBoxLayout()
        self.dl_btn = QPushButton("Download DEM")
        self.dl_btn.clicked.connect(self._download)
        self.rs_btn = QPushButton("Clip + resample + save .npz")
        self.rs_btn.clicked.connect(self._resample)
        self.plot_btn = QPushButton("Plot DEM")
        self.plot_btn.clicked.connect(self._plot_clicked)
        btns.addWidget(self.dl_btn)
        btns.addWidget(self.rs_btn)
        btns.addWidget(self.plot_btn)
        left.addLayout(btns)

        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.hide()
        left.addWidget(self.busy)

        self.note = QLabel()
        self.note.setWordWrap(True)
        left.addWidget(self.note)
        left.addStretch(1)
        root.addLayout(left, 2)

        self.canvas = MplCanvas(figsize=(5, 6))
        root.addWidget(self.canvas, 3)

        self._dem: topo.DemGrid | None = None
        self._regional: topo.DemGrid | None = None
        self._dem_loading = False
        self._on_provider(self.provider.currentText())
        self._on_reg_provider(self.reg_provider.currentText())
        self.provider.currentTextChanged.connect(lambda _t: self._update_info())
        self.demtype.currentTextChanged.connect(lambda _t: self._update_info())
        self.reg_provider.currentTextChanged.connect(lambda _t: self._update_info())

    # ------------------------------------------------------------------ #
    def _on_provider(self, name: str):
        self.demtype.setEnabled(name == "OpenTopography")
        self.local_path.setEnabled(name == "local")
        self.dl_btn.setEnabled(name != "local")
        if hasattr(self, "reg_provider"):
            self._on_reg_provider(self.reg_provider.currentText())

    def _on_reg_provider(self, name: str):
        self.reg_local.setEnabled(name == "local")
        for w in (self.reg_pad, self.reg_sample):
            w.setEnabled(name != "none")
        self.dl_btn.setEnabled(self.provider.currentText() != "local"
                               or name in ("GMRT", "OpenTopography"))

    def _browse_regional(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select regional DEM", self.reg_local.text(),
            "raster (*.tif *.tiff *.nc *.grd);;all files (*)")
        if f:
            self.reg_local.setText(f)
            self.project.topo.regional_local_path = f

    def _regional_pad_deg(self) -> float:
        """The regional pad in degrees of longitude (the wider one) at the survey."""
        import math

        s = self.project.survey
        clat = 0.5 * (s.min_lat + s.max_lat) if s.max_lat else 0.0
        return self.reg_pad.value() / (111.2 * max(math.cos(math.radians(clat)), 0.05))

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(
            self, "Select DEM", self.local_path.text(),
            "raster (*.tif *.tiff *.nc *.grd);;all files (*)")
        if f:
            self.local_path.setText(f)
            self.project.topo.local_path = f

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(
            self, "Save topography files to", self._out_dir())
        if d:
            self.out_dir.setText(d)
            self.project.topo.out_dir = d

    def _out_dir(self) -> str:
        return (self.out_dir.text().strip()
                or self.project.workdir or os.getcwd())

    def _sync(self):
        t = self.project.topo
        t.source = self.provider.currentText()
        t.local_path = self.local_path.text().strip()
        t.out_dir = self.out_dir.text().strip()
        t.pad_arcsec = self.pad.value()
        t.sample_arcsec = self.sample.value()
        t.regional_source = self.reg_provider.currentText()
        t.regional_local_path = self.reg_local.text().strip()
        t.regional_pad_km = self.reg_pad.value()
        t.regional_sample_arcsec = self.reg_sample.value()

    def _stations(self):
        p = self.project.survey.edi_dir
        return edi_io.read_edi_headers(p) if p and os.path.exists(p) else []

    def _update_info(self):
        s = self.project.survey
        clat = 0.5 * (s.min_lat + s.max_lat) if s.max_lat else 0.0
        parts = []
        sts = self._stations()
        if len(sts) >= 2:
            sp = edi_io.station_spacing_geographic(
                sts, set(self.project.survey.excluded))
            parts.append(
                f"site spacing (nearest-neighbour): min {sp['min_m']/1e3:.2f} / "
                f"median {sp['median_m']/1e3:.2f} / max {sp['max_m']/1e3:.2f} km")
        pad_ns, pad_ew = topo.arcsec_to_km(self.pad.value(), clat)
        s_ns, s_ew = topo.arcsec_to_km(self.sample.value(), clat)
        parts.append(f"pad {self.pad.value():.0f}″ ≈ {pad_ns:.1f} km N-S / "
                     f"{pad_ew:.1f} km E-W   |   posting {self.sample.value():.1f}″ "
                     f"≈ {s_ns*1e3:.0f} m N-S / {s_ew*1e3:.0f} m E-W  (at lat {clat:.1f}°)")
        if self._dem is not None:
            r = topo.dem_resolution(self._dem)
            parts.append(
                f"loaded DEM: {r['nx']} x {r['ny']} nodes, posting "
                f"{r['dlat_arcsec']:.1f}″ ≈ {r['dy_m']:.0f} m N-S / "
                f"{r['dx_m']:.0f} m E-W")
        self.info.setText("\n".join(parts))
        self._update_reg_hint(clat)
        self._update_size_info()

    # ---- size estimates ------------------------------------------------ #
    #: above these, the estimate is shown in red and the action asks first
    BIG_DOWNLOAD_MB = 300.0
    BIG_NPZ_MB = 500.0

    def _estimates(self) -> list[dict]:
        """One entry per DEM this tab would fetch / write, with its estimates and
        warnings (``bad`` = the provider will refuse it)."""
        s = self.project.survey
        if not (s.max_lon > s.min_lon and s.max_lat > s.min_lat):
            return []
        bbox = self._bbox()
        out = []
        for which, provider, pad_deg, samp in (
                ("detailed", self.provider.currentText(), self.pad.value() / 3600.0,
                 self.sample.value()),
                ("regional", self.reg_provider.currentText(), self._regional_pad_deg(),
                 self.reg_sample.value())):
            if provider == "none":
                continue
            dl = (topo.estimate_download(provider, bbox, pad_deg, self.demtype.currentText())
                  if provider != "local" else None)
            rs = topo.estimate_resample(bbox, pad_deg * 3600.0, samp)
            warn, bad = [], False
            if dl is not None:
                if dl["limit_km2"] and dl["area_km2"] > dl["limit_km2"]:
                    bad = True
                    warn.append(f"OpenTopography refuses {self.demtype.currentText()} requests over "
                                f"{dl['limit_km2']:,.0f} km² (this box: {dl['area_km2']:,.0f} km²): "
                                "use a smaller pad or a coarser dataset")
                elif dl["mb"] > self.BIG_DOWNLOAD_MB:
                    warn.append("large download: a smaller pad or a coarser dataset "
                                "(e.g. SRTMGL3 / COP90) would do for most meshes")
                post = 0.5 * (dl["dx_m"] + dl["dy_m"])
                want = 0.5 * (rs["dx_m"] + rs["dy_m"])
                if post > 1.5 * want:
                    warn.append(f"the download (~{post:,.0f} m) is coarser than the target posting "
                                f"(~{want:,.0f} m): resampling won't add detail"
                                + (" (GMRT gets coarser as the box grows)" if provider == "GMRT"
                                   else ""))
            if rs["mb"] > self.BIG_NPZ_MB:
                warn.append(f"large .npz, and resampling needs ~{rs['peak_mb'] / 1e3:.1f} GB of "
                            "memory: a coarser target posting would help")
            out.append(dict(which=which, provider=provider, download=dl, resample=rs,
                            warn=warn, bad=bad))
        return out

    def _update_size_info(self):
        ests = self._estimates()
        if not ests:
            self.size_info.setText("size estimates: load EDIs first (no survey box)")
            return
        lines = []
        for e in ests:
            head = f"<b>{e['which']}</b> ({e['provider']})"
            if e["download"] is not None:
                lines.append(f"{head} download: {topo.describe_estimate(e['download'])}")
                head = "&nbsp;&nbsp;&nbsp;"
            lines.append(f"{head} resampled .npz: {topo.describe_estimate(e['resample'])}")
            lines += [f"<span style='color:#b91c1c'>&nbsp;&nbsp;&nbsp;⚠ {w}</span>" for w in e["warn"]]
        self.size_info.setText("<br>".join(lines))

    def _confirm_size(self, action: str) -> bool:
        """Before downloading / resampling: refuse what the provider will reject,
        and ask before anything large."""
        ests = self._estimates()
        bad = [e for e in ests if e["bad"]]
        if action == "download" and bad:
            QMessageBox.warning(self, "Download DEM", "\n\n".join(
                f"{e['which']} DEM: " + "; ".join(e["warn"]) for e in bad))
            return False
        big = []
        for e in ests:
            dl, rs = e["download"], e["resample"]
            if action == "download" and dl is not None and dl["mb"] > self.BIG_DOWNLOAD_MB:
                big.append(f"{e['which']} {e['provider']} download: {topo.describe_estimate(dl)}")
            if action == "resample" and rs["mb"] > self.BIG_NPZ_MB:
                big.append(f"{e['which']} .npz: {topo.describe_estimate(rs)}; resampling needs "
                           f"~{rs['peak_mb'] / 1e3:.1f} GB of memory")
        if not big:
            return True
        return QMessageBox.question(
            self, "Large DEM", "This is large:\n\n" + "\n".join(big)
            + "\n\nContinue anyway?", QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No) == QMessageBox.Yes

    def _update_reg_hint(self, clat: float):
        """How far the mesh reaches, so the regional pad can be set to cover it."""
        import math

        s = self.project.survey
        half_km = sum(self.project.mesh.dx_blocks or []) / 1e3
        r_ns, r_ew = topo.arcsec_to_km(self.reg_sample.value(), clat)
        txt = f"posting {self.reg_sample.value():.0f}″ ≈ {r_ns:.2f} km N-S / {r_ew:.2f} km E-W. "
        if half_km > 0 and s.max_lat > s.min_lat:
            span_ns = 0.5 * (s.max_lat - s.min_lat) * 111.2
            span_ew = 0.5 * (s.max_lon - s.min_lon) * 111.2 * math.cos(math.radians(clat))
            need = max(half_km - min(span_ns, span_ew), 0.0)
            txt += (f"The mesh (Mesh tab) reaches {half_km:.0f} km from the model centre, so the "
                    f"pad needs to be at least about {need:.0f} km"
                    + (" -- more than set here." if need > self.reg_pad.value() + 1
                       and self.reg_provider.currentText() != "none" else "."))
        else:
            txt += ("The mesh padding is usually ~3 skin depths at the longest period (hundreds "
                    "of km); 'Recompute From Data' on the Mesh tab shows how far it reaches.")
        self.reg_hint.setText(txt)

    def _bbox(self) -> dict:
        s = self.project.survey
        return {"MinLon": s.min_lon, "MaxLon": s.max_lon,
                "MinLat": s.min_lat, "MaxLat": s.max_lat}

    def _busy(self, on: bool):
        self.busy.setVisible(on)
        self.dl_btn.setEnabled(not on and (self.provider.currentText() != "local"
                                           or self.reg_provider.currentText() in ("GMRT", "OpenTopography")))
        self.rs_btn.setEnabled(not on)
        self.plot_btn.setEnabled(not on)

    # ------------------------------------------------------------------ #
    def _download(self):
        self._sync()
        s = self.project.survey
        if not (s.max_lon > s.min_lon):
            self.log("load EDIs first (no survey bbox)")
            return
        if not self._confirm_size("download"):
            self.log("download cancelled (size)")
            return
        settings = config.load_settings()
        provider = self.provider.currentText()
        os.makedirs(self._out_dir(), exist_ok=True)
        out_base = os.path.join(self._out_dir(), f"dem_{provider.lower()}")
        pad_deg = self.pad.value() / 3600.0
        reg = self.reg_provider.currentText()
        reg_base = os.path.join(self._out_dir(), f"dem_{reg.lower()}_regional")
        reg_pad = self._regional_pad_deg()
        kw = dict(api_key=settings.get("opentopo_api_key", ""), demtype=self.demtype.currentText())

        def job():
            path = (topo.download(provider, self._bbox(), out_base, pad_deg=pad_deg, **kw)
                    if provider != "local" else "")
            reg_path = (topo.download(reg, self._bbox(), reg_base, pad_deg=reg_pad, **kw)
                        if reg in ("GMRT", "OpenTopography") else "")
            return path, reg_path

        self._busy(True)
        self.log(f"downloading {provider} DEM "
                 f"({'NetCDF' if provider == 'GMRT' else 'GeoTIFF'})"
                 + (f" and a regional {reg} DEM ({self.reg_pad.value():.0f} km pad)"
                    if reg in ("GMRT", "OpenTopography") else ""))
        run_async(self, job, self._after_download, self._after_error)

    def _after_download(self, result):
        self._busy(False)
        path, reg_path = result
        if path:
            self.project.topo.downloaded_path = path
            self.log(f"downloaded {path} ({os.path.getsize(path)/1e3:.0f} kB)")
        if reg_path:
            self.project.topo.regional_downloaded_path = reg_path
            self.log(f"downloaded {reg_path} ({os.path.getsize(reg_path)/1e3:.0f} kB)")
        if path:
            self._load_and_plot(path)
        self._update_info()
        self.notify_changed()

    def _resample(self):
        self._sync()
        src = (self.local_path.text().strip() if self.provider.currentText() == "local"
               else self.project.topo.downloaded_path)
        if not src or not os.path.isfile(src):
            self.log("no DEM to resample -- download or pick a local file first")
            return
        reg = self.reg_provider.currentText()
        reg_src = ""
        if reg != "none":
            reg_src = (self.reg_local.text().strip() if reg == "local"
                       else self.project.topo.regional_downloaded_path)
            if not reg_src or not os.path.isfile(reg_src):
                self.log("no regional DEM to resample -- download it or pick a local file, "
                         "or set its provider to 'none'")
                return
        if not self._confirm_size("resample"):
            self.log("resample cancelled (size)")
            return
        bbox = self._bbox()
        pad, samp = self.pad.value(), self.sample.value()
        reg_pad, reg_samp = self._regional_pad_deg() * 3600.0, self.reg_sample.value()
        os.makedirs(self._out_dir(), exist_ok=True)
        out = os.path.join(self._out_dir(), "topo_big.npz")
        reg_out = os.path.join(self._out_dir(), "topo_regional.npz")

        def job():
            dem = topo.read_dem(src)
            rs = topo.resample_dem(dem, bbox, pad, samp)
            topo.save_dem_npz(rs, out)
            rs_reg = None
            if reg_src:
                rs_reg = topo.resample_dem(topo.read_dem(reg_src), bbox, reg_pad, reg_samp)
                topo.save_dem_npz(rs_reg, reg_out)
            return rs, out, rs_reg

        self._busy(True)
        self.log(f"clip + resample {src}" + (f" and {reg_src}" if reg_src else ""))
        run_async(self, job, self._after_resample, self._after_error)

    def _after_resample(self, result):
        self._busy(False)
        rs, out, rs_reg = result
        self._dem, self._regional = rs, rs_reg
        t = self.project.topo
        t.resampled_path = out
        t.regional_resampled_path = os.path.join(os.path.dirname(out), "topo_regional.npz") \
            if rs_reg is not None else ""
        self.log(f"resampled to {rs.shape}, saved {out}"
                 + (f"; regional {rs_reg.shape}, saved {t.regional_resampled_path}"
                    if rs_reg is not None else ""))
        self._plot(rs, title="resampled DEM (geographic)")
        self._update_info()
        self.notify_changed()

    def _after_error(self, exc: Exception):
        self._dem_loading = False
        self._busy(False)
        self.log(f"ERROR: {exc}")
        self.note.setText("<pre>" + "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]) + "</pre>")

    # ------------------------------------------------------------------ #
    def _saved_dem_source(self):
        """(path, loader) for a DEM already on disk from a previous session, if any."""
        t = self.project.topo
        if t.resampled_path and os.path.isfile(t.resampled_path):
            return t.resampled_path, topo.load_dem_npz
        if t.downloaded_path and os.path.isfile(t.downloaded_path):
            return t.downloaded_path, topo.read_dem
        if t.source == "local" and t.local_path and os.path.isfile(t.local_path):
            return t.local_path, topo.read_dem
        return None, None

    def _plot_clicked(self):
        if self._dem is not None:
            self._plot(self._dem, title="loaded DEM (geographic)")
            return
        self._load_saved_dem(on_missing=lambda: self.log(
            "no DEM loaded yet -- download, or clip + resample, a DEM first"))

    def _load_saved_dem(self, on_missing=None):
        """Load + plot whatever DEM the project already points at, without recomputing."""
        if self._dem_loading:
            return
        path, loader = self._saved_dem_source()
        if not path:
            if on_missing:
                on_missing()
            return
        self._dem_loading = True
        self._busy(True)
        self.log(f"loading saved DEM {path}")
        t = self.project.topo
        reg_path = t.regional_resampled_path if t.regional_source != "none" else ""

        def job():
            regional = topo.load_dem_npz(reg_path) if reg_path and os.path.isfile(reg_path) else None
            return loader(path), regional

        run_async(self, job, self._after_load_saved_dem, self._after_error)

    def _after_load_saved_dem(self, result):
        self._dem_loading = False
        self._busy(False)
        self._dem, self._regional = result
        self._plot(self._dem, title="loaded DEM (geographic)")
        self._update_info()

    def _load_and_plot(self, path):
        try:
            self._dem = topo.read_dem(path)
            self._plot(self._dem, title="downloaded DEM (geographic)")
        except Exception as exc:  # pragma: no cover - user IO
            self.log(f"could not read DEM for preview: {exc}")

    def _plot(self, dem, title=""):
        if getattr(self.canvas, "ax", None) is None:
            return
        import numpy as np

        self.canvas.clear()
        ax = self.canvas.ax
        reg = self._regional if self.reg_provider.currentText() != "none" else None
        grids = ([reg] if reg is not None else []) + [dem]
        vmin = min(float(np.nanmin(g.z)) for g in grids)
        vmax = max(float(np.nanmax(g.z)) for g in grids)
        for g in grids:
            lon, lat = g.meshgrid()
            step = max(int(max(g.z.shape) // 600), 1)          # keep big grids quick to draw
            pcm = ax.pcolormesh(lon[::step, ::step], lat[::step, ::step], g.z[::step, ::step],
                                shading="auto", cmap="terrain", vmin=vmin, vmax=vmax)
        if reg is not None:
            ax.plot([dem.lon.min(), dem.lon.max(), dem.lon.max(), dem.lon.min(), dem.lon.min()],
                    [dem.lat.min(), dem.lat.min(), dem.lat.max(), dem.lat.max(), dem.lat.min()],
                    "k--", lw=0.8, zorder=4)
            title = (title + " + regional") if title else "detailed + regional DEM"
        self.canvas.figure.colorbar(pcm, ax=ax, orientation="horizontal",
                                    shrink=0.8, pad=0.08, label="elevation (m)")
        s = self.project.survey
        ax.plot([s.min_lon, s.max_lon, s.max_lon, s.min_lon, s.min_lon],
                [s.min_lat, s.min_lat, s.max_lat, s.max_lat, s.min_lat],
                "r-", lw=1, zorder=4)
        excl = set(self.project.survey.excluded)
        sts = self._stations()
        used = [(st.lon, st.lat) for st in sts if st.station_id not in excl]
        drop = [(st.lon, st.lat) for st in sts if st.station_id in excl]
        if used:
            ax.scatter([x for x, _ in used], [y for _, y in used], s=12,
                       c="red", edgecolors="k", linewidths=0.3, zorder=5, label="stations")
        if drop:
            ax.scatter([x for x, _ in drop], [y for _, y in drop], s=12,
                       c="0.6", marker="x", zorder=5, label="excluded")
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        outer = reg if reg is not None else dem
        ax.set_xlim(float(outer.lon.min()), float(outer.lon.max()))
        ax.set_ylim(float(outer.lat.min()), float(outer.lat.max()))
        self.canvas.redraw()

    def refresh(self) -> None:
        t = self.project.topo
        if t.source in config.TOPO_PROVIDERS:
            self.provider.setCurrentText(t.source)
        self.local_path.setText(t.local_path)
        self.out_dir.setText(t.out_dir)
        self.pad.setValue(t.pad_arcsec)
        self.sample.setValue(t.sample_arcsec)
        self.reg_provider.setCurrentText(t.regional_source if t.regional_source in
                                         ["none"] + list(config.TOPO_PROVIDERS) else "none")
        self.reg_local.setText(t.regional_local_path)
        self.reg_pad.setValue(t.regional_pad_km)
        self.reg_sample.setValue(t.regional_sample_arcsec)
        self.note.setText(
            "GMRT = keyless global topo+bathymetry (NetCDF, no GDAL). "
            "OpenTopography needs a free API key (Project > Settings). "
            "'local' ingests your own NetCDF/GeoTIFF.")
        self._update_info()
        if self._dem is None:
            self._load_saved_dem()   # silently -- nothing to plot yet is not an error here
