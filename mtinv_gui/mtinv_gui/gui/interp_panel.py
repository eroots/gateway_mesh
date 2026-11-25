"""Step 0 -- optional pre-processing: interpolate a folder of EDIs onto one
shared, logarithmically-spaced period grid before the Survey / EDIs step.

Entirely skippable -- the Survey tab works fine loading raw EDIs directly.
This exists for surveys where stations weren't all recorded on the same
period grid and need resampling onto a common one before the rest of the
workflow, which otherwise assumes one.
"""
from __future__ import annotations

import math
import os
import traceback

from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..backends import edi_interp, edi_io
from .mpl_canvas import MplCanvas
from .panel_base import BasePanel
from .worker import run_async


def _dspin(lo, hi, step, dec, val, suffix=""):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setSingleStep(step)
    w.setDecimals(dec)
    w.setValue(val)
    if suffix:
        w.setSuffix(suffix)
    return w


class InterpPanel(BasePanel):
    title = "0 - Interpolate EDIs"

    def build(self) -> None:
        self._periods_by_station: dict[str, dict] = {}   # cached -- see _load_stats
        self._loading_stats = False
        self._qc_stations: list[str] = []   # sorted station ids, for the QC tab combo
        self._qc_index = -1
        self._qc_loading = False

        root = QHBoxLayout(self)
        left = QVBoxLayout()

        note = QLabel(
            "Optional. Resample every EDI in a folder onto one shared, "
            "logarithmically-spaced period grid, using mtpy. Skip this "
            "entirely and load your EDIs directly on the Survey tab if "
            "they're already on a common grid.")
        note.setWordWrap(True)
        note.setStyleSheet("color: gray;")
        left.addWidget(note)

        src_box = QGroupBox("Source EDIs")
        sf = QFormLayout(src_box)
        src_row = QHBoxLayout()
        self.src_edit = QLineEdit(self.project.interp.source_path)
        self.src_edit.setPlaceholderText("folder of *.edi  OR  a .lst site list")
        src_folder = QPushButton("Folder...")
        src_folder.clicked.connect(self._browse_src)
        src_list = QPushButton("List (.lst)...")
        src_list.clicked.connect(self._browse_src_list)
        src_row.addWidget(self.src_edit, 1)
        src_row.addWidget(src_folder)
        src_row.addWidget(src_list)
        sf.addRow("source", src_row)
        self.stats_btn = QPushButton("Load Period Stats")
        self.stats_btn.clicked.connect(self._load_stats)
        sf.addRow(self.stats_btn)
        left.addWidget(src_box)

        period_box = QGroupBox("Target period grid (logarithmic)")
        pf = QFormLayout(period_box)
        self.period_min = _dspin(1e-6, 1e6, 0.01, 6, self.project.interp.period_min_s, " s")
        self.period_max = _dspin(1e-6, 1e6, 10.0, 6, self.project.interp.period_max_s, " s")
        self.per_decade = _dspin(1.0, 50.0, 1.0, 1, self.project.interp.periods_per_decade)
        self.interp_type = QComboBox()
        self.interp_type.addItems(["slinear", "linear", "quadratic", "cubic"])
        self.interp_type.setCurrentText(self.project.interp.interp_type)
        self.interp_type.setToolTip(
            "scipy interp1d kind used on log-period Z/tipper (real+imag interpolated "
            "separately). 'slinear' is a safe default for typically noisy MT data; "
            "the higher-order splines can ring where nearby periods are missing.")
        pf.addRow("min period", self.period_min)
        pf.addRow("max period", self.period_max)
        pf.addRow("periods per decade", self.per_decade)
        pf.addRow("interpolation", self.interp_type)
        self.grid_lbl = QLabel("-")
        self.grid_lbl.setStyleSheet("color: gray; font-size: 11px;")
        pf.addRow("target grid", self.grid_lbl)
        for w in (self.period_min, self.period_max, self.per_decade):
            w.valueChanged.connect(self._update_grid_preview)
        left.addWidget(period_box)

        out_box = QGroupBox("Output")
        of = QFormLayout(out_box)
        out_row = QHBoxLayout()
        self.out_edit = QLineEdit(self.project.interp.out_dir)
        self.out_edit.setPlaceholderText("(blank -> <source>_interp next to the source)")
        out_browse = QPushButton("Browse...")
        out_browse.clicked.connect(self._browse_out)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(out_browse)
        of.addRow("save to", out_row)
        left.addWidget(out_box)

        btn_row = QHBoxLayout()
        self.run_btn = QPushButton("Interpolate")
        self.run_btn.clicked.connect(self._run)
        btn_row.addWidget(self.run_btn)
        self.use_btn = QPushButton("Use Output on Survey Tab →")
        self.use_btn.setEnabled(bool(self.project.interp.last_output_dir))
        self.use_btn.clicked.connect(self._use_on_survey)
        btn_row.addWidget(self.use_btn)
        btn_row.addStretch(1)
        left.addLayout(btn_row)

        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.hide()
        left.addWidget(self.busy)

        self.result_lbl = QLabel("")
        self.result_lbl.setWordWrap(True)
        left.addWidget(self.result_lbl)
        self.note2 = QLabel()
        self.note2.setWordWrap(True)
        left.addWidget(self.note2)
        left.addStretch(1)
        root.addLayout(left, 2)

        tabs = QTabWidget()
        root.addWidget(tabs, 3)

        cov_page = QWidget()
        right = QVBoxLayout(cov_page)
        self.stats_lbl = QLabel(
            "no period stats loaded yet -- pick a source and click "
            "“Load Period Stats”")
        self.stats_lbl.setWordWrap(True)
        self.stats_lbl.setStyleSheet("font-size: 11px;")
        right.addWidget(self.stats_lbl)
        self.canvas = MplCanvas(figsize=(5, 6))
        right.addWidget(self.canvas, 1)
        tabs.addTab(cov_page, "Period Coverage")

        qc_page = QWidget()
        qc_layout = QVBoxLayout(qc_page)
        qc_row = QHBoxLayout()
        qc_row.addWidget(QLabel("station:"))
        self.qc_combo = QComboBox()
        self.qc_combo.setMinimumWidth(140)
        self.qc_combo.currentIndexChanged.connect(self._qc_combo_changed)
        qc_row.addWidget(self.qc_combo)
        self.qc_prev_btn = QPushButton("< Prev")
        self.qc_prev_btn.clicked.connect(self._qc_prev)
        qc_row.addWidget(self.qc_prev_btn)
        self.qc_next_btn = QPushButton("Next >")
        self.qc_next_btn.clicked.connect(self._qc_next)
        qc_row.addWidget(self.qc_next_btn)
        qc_row.addStretch(1)
        qc_layout.addLayout(qc_row)
        self.qc_status_lbl = QLabel(
            "load period stats, then run an interpolation, to compare "
            "before/after station responses here")
        self.qc_status_lbl.setWordWrap(True)
        self.qc_status_lbl.setStyleSheet("color: gray; font-size: 11px;")
        qc_layout.addWidget(self.qc_status_lbl)
        self.qc_canvas = MplCanvas(figsize=(5, 7))
        # constrained layout handles stacked shared-x axes with height ratios
        # cleanly; the canvas's default tight_layout warns on this arrangement
        self.qc_canvas.figure.set_layout_engine("constrained")
        qc_layout.addWidget(self.qc_canvas, 1)
        tabs.addTab(qc_page, "Station QC")

        # re-draw the target-range overlay (cheap, in-memory) whenever the
        # grid changes -- the expensive part (reading every EDI) only runs
        # from _load_stats, explicitly
        for w in (self.period_min, self.period_max, self.per_decade):
            w.valueChanged.connect(self._render_stats)

        self._update_grid_preview()

    # ------------------------------------------------------------------ #
    def _browse_src(self):
        start = self.src_edit.text() or os.getcwd()
        if os.path.isfile(start):
            start = os.path.dirname(start)
        f = QFileDialog.getExistingDirectory(self, "Select folder of EDIs", start)
        if f:
            self.src_edit.setText(f)
            self._load_stats()

    def _browse_src_list(self):
        start = self.src_edit.text() or os.getcwd()
        if os.path.isfile(start):
            start = os.path.dirname(start)
        f, _ = QFileDialog.getOpenFileName(
            self, "Select site list", start, "site list (*.lst);;all files (*)")
        if f:
            self.src_edit.setText(f)
            self._load_stats()

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Save interpolated EDIs to", self._out_dir())
        if d:
            self.out_edit.setText(d)

    def _out_dir(self) -> str:
        out = self.out_edit.text().strip()
        if out:
            return out
        src = self.src_edit.text().strip().rstrip("/\\")
        if src.lower().endswith(".lst"):
            src = src[:-4]
        return (src + "_interp") if src else ""

    def _sync(self):
        p = self.project.interp
        p.source_path = self.src_edit.text().strip()
        p.period_min_s = self.period_min.value()
        p.period_max_s = self.period_max.value()
        p.periods_per_decade = self.per_decade.value()
        p.interp_type = self.interp_type.currentText()
        p.out_dir = self.out_edit.text().strip()

    def _update_grid_preview(self):
        pmin, pmax, per_dec = self.period_min.value(), self.period_max.value(), self.per_decade.value()
        if pmax <= pmin:
            self.grid_lbl.setText("max period must be greater than min period")
            return
        n = max(int(round(math.log10(pmax / pmin) * per_dec)) + 1, 2)
        self.grid_lbl.setText(f"{n} periods, {pmin:.4g}-{pmax:.4g} s")

    # ------------------------------------------------------------------ #
    def _load_stats(self):
        """Read every EDI's full period list (not just header min/max/count)
        for the stats line and the coverage histogram. This is the one
        expensive step (~0.5s/100 stations, disk I/O) -- cached in
        ``self._periods_by_station`` and reused for every subsequent
        min/max/per-decade tweak, which only re-bins/re-draws in memory."""
        if self._loading_stats:
            return
        src = self.src_edit.text().strip()
        if not src or not os.path.exists(src):
            self.log("pick a source folder or .lst file first")
            return
        self._loading_stats = True
        self.stats_btn.setEnabled(False)
        self.stats_lbl.setText("loading period stats…")
        run_async(self, lambda: edi_io.read_edi_periods(src), self._after_stats, self._error)

    def _after_stats(self, periods_by_station: dict):
        self._loading_stats = False
        self.stats_btn.setEnabled(True)
        self._periods_by_station = periods_by_station
        if not periods_by_station:
            self.stats_lbl.setText("no readable EDI files found at that source")
            self.canvas.clear()
            self.canvas.redraw()
            self._populate_qc_stations()
            return
        self._render_stats()
        self._populate_qc_stations()

    def _render_stats(self):
        """Recompute the stats line and redraw the coverage histogram from
        the cached ``self._periods_by_station`` -- no disk I/O, safe to call
        on every min/max/per-decade spin box tick."""
        periods_by_station = self._periods_by_station
        if not periods_by_station:
            return
        import numpy as np

        n_stations = len(periods_by_station)
        all_periods = np.concatenate(
            [np.asarray(sd["periods"]) for sd in periods_by_station.values()])
        n_samples = len(all_periods)
        lo, hi = float(all_periods.min()), float(all_periods.max())

        pmin, pmax = self.period_min.value(), self.period_max.value()
        target_ok = pmax > pmin
        full = partial = none = 0
        if target_ok:
            target_periods = edi_interp.periods_from_decade(pmin, pmax, self.per_decade.value())
            for sd in periods_by_station.values():
                plist = sd["periods"]
                smin, smax = plist[0], plist[-1]
                n_covered = sum(1 for p in target_periods if smin <= p <= smax)
                if n_covered >= len(target_periods):
                    full += 1
                elif n_covered >= 2:
                    partial += 1
                else:
                    none += 1

        stats = (f"{n_stations} stations, {n_samples} period samples total "
                f"(mean {n_samples/n_stations:.1f}/station)\n"
                f"overall period range: {lo:.4g} - {hi:.4g} s")
        if target_ok:
            stats += (f"\ntarget grid ({len(target_periods)} periods, {pmin:.4g}-{pmax:.4g} s): "
                     f"{full} fully covered, {partial} partial, {none} will be skipped")
        self.stats_lbl.setText(stats)

        # coverage histogram: for each log-spaced bin, how many DISTINCT
        # stations have at least one period in it (not total period samples --
        # a station with many periods packed in one bin shouldn't outweigh
        # ten stations with one period each).
        fig = self.canvas.figure
        fig.clear()
        ax = fig.add_subplot(1, 1, 1)
        n_decades = math.log10(hi / lo) if hi > lo else 1.0
        n_bins = max(int(round(n_decades * 10)), 1)   # fixed density, independent of the target grid
        edges = np.logspace(math.log10(lo), math.log10(hi), n_bins + 1)
        counts = np.zeros(n_bins, dtype=int)
        for sd in periods_by_station.values():
            plist = sd["periods"]
            idx = np.clip(np.searchsorted(edges, plist, side="right") - 1, 0, n_bins - 1)
            counts[np.unique(idx)] += 1
        ax.bar(edges[:-1], counts, width=np.diff(edges), align="edge",
              color="#4c72b0", edgecolor="black", linewidth=0.3)
        if target_ok:
            ax.axvspan(pmin, pmax, color="green", alpha=0.12, zorder=0,
                      label="target grid")
            ax.axvline(pmin, color="green", ls="--", lw=1)
            ax.axvline(pmax, color="green", ls="--", lw=1)
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("period (s)")
        ax.set_ylabel("stations with data")
        ax.set_title("period coverage across the dataset")
        self.canvas.redraw()

    # ------------------------------------------------------------------ #
    # Station QC: before/after apparent resistivity/phase/tipper, one
    # station at a time. "before" = the raw source EDI (path travels with
    # _periods_by_station, for free); "after" = the interpolated output
    # EDI, resolved by basename match against project.interp.last_output_dir
    # (absent until a successful _run()).
    def _populate_qc_stations(self):
        self._qc_stations = sorted(self._periods_by_station.keys())
        self.qc_combo.blockSignals(True)
        self.qc_combo.clear()
        self.qc_combo.addItems(self._qc_stations)
        self.qc_combo.blockSignals(False)
        if self._qc_stations:
            self._qc_index = 0
            self.qc_combo.setCurrentIndex(0)
            self._load_qc_station()
        else:
            self._qc_index = -1
            self.qc_canvas.clear()
            self.qc_canvas.redraw()

    def _qc_combo_changed(self, idx: int):
        if idx < 0 or idx == self._qc_index:
            return
        self._qc_index = idx
        self._load_qc_station()

    def _qc_prev(self):
        if self._qc_stations and self._qc_index > 0:
            self.qc_combo.setCurrentIndex(self._qc_index - 1)

    def _qc_next(self):
        if self._qc_stations and self._qc_index < len(self._qc_stations) - 1:
            self.qc_combo.setCurrentIndex(self._qc_index + 1)

    def _after_path_for(self, station_id: str) -> str | None:
        out_dir = self.project.interp.last_output_dir
        if not out_dir:
            return None
        cand = os.path.join(out_dir, f"{station_id}.edi")
        return cand if os.path.isfile(cand) else None

    def _load_qc_station(self):
        if self._qc_index < 0 or self._qc_loading:
            return
        station_id = self._qc_stations[self._qc_index]
        before_path = self._periods_by_station[station_id]["path"]
        after_path = self._after_path_for(station_id)
        self._qc_loading = True
        self.qc_prev_btn.setEnabled(False)
        self.qc_next_btn.setEnabled(False)
        self.qc_status_lbl.setText(f"loading {station_id}…")

        def job():
            before = edi_interp.read_station_response(before_path)
            after = edi_interp.read_station_response(after_path) if after_path else None
            return station_id, before, after

        run_async(self, job, self._after_qc_station, self._qc_error)

    def _after_qc_station(self, result):
        station_id, before, after = result
        self._qc_loading = False
        self.qc_prev_btn.setEnabled(self._qc_index > 0)
        self.qc_next_btn.setEnabled(self._qc_index < len(self._qc_stations) - 1)
        msg = f"{station_id} -- before: {len(before['periods'])} periods"
        if after is not None:
            msg += f", after: {len(after['periods'])} periods"
        else:
            msg += " (no interpolated output yet for this station)"
        self.qc_status_lbl.setText(msg)
        self._render_qc(station_id, before, after)

    def _qc_error(self, exc: Exception):
        self._qc_loading = False
        self.qc_prev_btn.setEnabled(self._qc_index > 0)
        self.qc_next_btn.setEnabled(self._qc_index < len(self._qc_stations) - 1)
        self.qc_status_lbl.setText(f"ERROR reading station: {exc}")

    def _render_qc(self, station_id: str, before: dict, after: dict | None):
        fig = self.qc_canvas.figure
        # a previous render leaves log-scaled, x-shared axes; fig.clear() resets
        # each to (0, 1) and the shared link propagates that onto the still-log
        # siblings -> "non-positive xlim on a log axis" warning. Drop the scales
        # back to linear first so the reset is harmless.
        for prev_ax in fig.axes:
            prev_ax.set_xscale("linear")
            prev_ax.set_yscale("linear")
        fig.clear()
        has_tipper = bool((before.get("tipper")) or (after and after.get("tipper")))
        n_rows = 3 if has_tipper else 2
        ratios = [3, 1.5, 1] if has_tipper else [3, 1.5]
        gs = fig.add_gridspec(n_rows, 1, height_ratios=ratios)
        ax_res = fig.add_subplot(gs[0])
        ax_ph = fig.add_subplot(gs[1], sharex=ax_res)
        ax_tp = fig.add_subplot(gs[2], sharex=ax_res) if has_tipper else None

        # before: light, open markers, dashed, low zorder. after: solid, filled, on top.
        def plot_pair(ax, data, key_xy, key_yx, style, label_suffix):
            if data is None:
                return
            p = data["periods"]
            if style == "before":
                kw_xy = dict(color="#4c72b0", marker="o", mfc="none", ls="--",
                            lw=1, ms=4, alpha=0.5, zorder=1)
                kw_yx = dict(color="#c44e52", marker="o", mfc="none", ls="--",
                            lw=1, ms=4, alpha=0.5, zorder=1)
            else:
                kw_xy = dict(color="#4c72b0", marker="o", ls="-",
                            lw=1.2, ms=4, alpha=0.9, zorder=2)
                kw_yx = dict(color="#c44e52", marker="o", ls="-",
                            lw=1.2, ms=4, alpha=0.9, zorder=2)
            ax.plot(p, data[key_xy], label=f"XY {label_suffix}", **kw_xy)
            ax.plot(p, data[key_yx], label=f"YX {label_suffix}", **kw_yx)

        plot_pair(ax_res, before, "res_xy", "res_yx", "before", "before")
        plot_pair(ax_res, after, "res_xy", "res_yx", "after", "after")
        ax_res.set_xscale("log")
        ax_res.set_yscale("log")
        ax_res.set_ylabel(r"app. resistivity ($\Omega\cdot$m)")
        ax_res.set_title(f"{station_id}: apparent resistivity, phase"
                         + (", tipper" if has_tipper else ""))
        ax_res.legend(loc="best", fontsize=7, ncol=2)
        ax_res.tick_params(labelbottom=False)
        ax_res.grid(True, which="both", alpha=0.25)

        plot_pair(ax_ph, before, "phase_xy", "phase_yx", "before", "before")
        plot_pair(ax_ph, after, "phase_xy", "phase_yx", "after", "after")
        ax_ph.set_ylabel("phase (deg)")
        ax_ph.grid(True, which="both", alpha=0.25)
        ax_ph.axhline(0, color="black", lw=0.5)

        if has_tipper:
            ax_ph.tick_params(labelbottom=False)

            def plot_tipper(data, style):
                if data is None or data.get("tipper") is None:
                    return
                tp = data["tipper"]
                p = tp["periods"]
                if style == "before":
                    kw_re = dict(color="#55a868", marker="o", mfc="none", ls="--",
                                lw=1, ms=4, alpha=0.5, zorder=1)
                    kw_im = dict(color="#8172b2", marker="o", mfc="none", ls="--",
                                lw=1, ms=4, alpha=0.5, zorder=1)
                else:
                    kw_re = dict(color="#55a868", marker="o", ls="-",
                                lw=1.2, ms=4, alpha=0.9, zorder=2)
                    kw_im = dict(color="#8172b2", marker="o", ls="-",
                                lw=1.2, ms=4, alpha=0.9, zorder=2)
                sfx = "before" if style == "before" else "after"
                ax_tp.plot(p, tp["tzy_re"], label=f"Re(Tzy) {sfx}", **kw_re)
                ax_tp.plot(p, tp["tzy_im"], label=f"Im(Tzy) {sfx}", **kw_im)

            plot_tipper(before, "before")
            plot_tipper(after, "after")
            ax_tp.set_ylabel("tipper")
            ax_tp.set_xlabel("period (s)")
            ax_tp.grid(True, which="both", alpha=0.25)
            ax_tp.axhline(0, color="black", lw=0.5)
            ax_tp.legend(loc="best", fontsize=7, ncol=2)
        else:
            ax_ph.set_xlabel("period (s)")

        self.qc_canvas.redraw()

    # ------------------------------------------------------------------ #
    def _run(self):
        self._sync()
        src = self.src_edit.text().strip()
        if not src or not os.path.exists(src):
            self.log("pick a source folder or .lst file first")
            return
        if self.period_max.value() <= self.period_min.value():
            self.log("max period must be greater than min period")
            return
        out_dir = self._out_dir()

        def job():
            return edi_interp.interpolate_collection(
                src, out_dir, period_min_s=self.period_min.value(),
                period_max_s=self.period_max.value(),
                periods_per_decade=self.per_decade.value(),
                interp_type=self.interp_type.currentText(), log=self.log)

        self.busy.show()
        self.run_btn.setEnabled(False)
        self.use_btn.setEnabled(False)
        self.result_lbl.setText("interpolating…")
        self.log(f"interpolating EDIs from {src} -> {out_dir}")
        run_async(self, job, self._after_run, self._error)

    def _after_run(self, result: dict):
        self.busy.hide()
        self.run_btn.setEnabled(True)
        self.project.interp.last_output_dir = result["out_dir"]
        self.use_btn.setEnabled(bool(result["written"]))
        msg = (f"wrote {len(result['written'])} / "
              f"{len(result['written']) + len(result['skipped'])} EDIs, "
              f"{result['n_periods']} periods, to {result['out_dir']}")
        if result["skipped"]:
            msg += f"  ({len(result['skipped'])} skipped -- see log)"
        self.result_lbl.setText(msg)
        if self._qc_index >= 0:
            self._load_qc_station()   # refresh so the QC plot picks up the new "after" data
        self.notify_changed()

    def _use_on_survey(self):
        out_dir = self.project.interp.last_output_dir
        if not out_dir:
            return
        self.project.survey.edi_dir = out_dir
        self.log(f"Survey tab source set to {out_dir} -- open the Survey tab to load it")
        self.notify_changed()

    def _error(self, exc: Exception):
        self._loading_stats = False
        self.stats_btn.setEnabled(True)
        self.busy.hide()
        self.run_btn.setEnabled(True)
        self.log(f"ERROR: {exc}")
        self.note2.setText("<pre>" + "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]) + "</pre>")

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        p = self.project.interp
        self.src_edit.setText(p.source_path)
        self.period_min.setValue(p.period_min_s)
        self.period_max.setValue(p.period_max_s)
        self.per_decade.setValue(p.periods_per_decade)
        if p.interp_type:
            self.interp_type.setCurrentText(p.interp_type)
        self.out_edit.setText(p.out_dir)
        self.use_btn.setEnabled(bool(p.last_output_dir))
        self._update_grid_preview()
        if not self._periods_by_station and p.source_path and os.path.exists(p.source_path):
            self._load_stats()   # once per session -- see _load_stats' docstring
