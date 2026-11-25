"""Step 4 -- data overview: SSQ apparent resistivity + C-response depth,
and the period window to use downstream."""
from __future__ import annotations

import traceback

from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import (
    QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QProgressBar,
    QPushButton, QVBoxLayout,
)

from ..backends import edi_io, edi_z
from .mpl_canvas import MplCanvas
from .panel_base import BasePanel
from .worker import run_async


class DataPanel(BasePanel):
    title = "4 - Data"

    def build(self) -> None:
        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.compute_btn = QPushButton("Compute SSQ resistivity + C-response depth")
        self.compute_btn.clicked.connect(self._compute)
        bar.addWidget(self.compute_btn)
        bar.addStretch(1)
        root.addLayout(bar)

        per_box = QGroupBox("Period window to use (feeds the data file; will feed mesh sizing)")
        pf = QFormLayout(per_box)
        val = QDoubleValidator(1e-6, 1e9, 8)
        val.setNotation(QDoubleValidator.ScientificNotation)
        g = self.project.gofem
        self.pmin = QLineEdit(f"{g.period_min_s:g}")
        self.pmax = QLineEdit(f"{g.period_max_s:g}")
        for e in (self.pmin, self.pmax):
            e.setValidator(val)
            e.setMaximumWidth(120)
            e.editingFinished.connect(self._on_period_committed)
        row = QHBoxLayout()
        row.addWidget(self.pmin)
        row.addWidget(QLabel("to"))
        row.addWidget(self.pmax)
        row.addWidget(QLabel("s"))
        row.addStretch(1)
        pf.addRow("min / max period", row)
        root.addWidget(per_box)

        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.hide()
        root.addWidget(self.busy)

        self.summary = QLabel("Z_ssq = sqrt((Zxx²+Zxy²+Zyx²+Zyy²)/2)  (complex);  "
                              "ρ = |Z|²/(ωμ₀);  C-response depth ≈ Re{Z/(iωμ₀)}")
        self.summary.setWordWrap(True)
        root.addWidget(self.summary)

        self.canvas = MplCanvas(figsize=(7, 8))
        root.addWidget(self.canvas, 1)

        self._per_station = None
        self._curve = None

    # ------------------------------------------------------------------ #
    def _periods(self) -> tuple[float, float]:
        g = self.project.gofem
        try:
            lo = float(self.pmin.text())
        except ValueError:
            lo = g.period_min_s
        try:
            hi = float(self.pmax.text())
        except ValueError:
            hi = g.period_max_s
        if hi < lo:
            lo, hi = hi, lo
        return lo, hi

    def _on_period_committed(self):
        lo, hi = self._periods()
        g = self.project.gofem
        if (lo, hi) == (g.period_min_s, g.period_max_s):
            return
        g.period_min_s, g.period_max_s = lo, hi
        self.pmin.setText(f"{lo:g}")
        self.pmax.setText(f"{hi:g}")
        if self._curve is not None:
            self._plot()
        self.notify_changed()

    def _compute(self):
        path = self.project.survey.edi_dir
        if not path:
            self.log("load EDIs on the Survey tab first")
            return
        excl = set(self.project.survey.excluded)
        self.busy.show()
        self.compute_btn.setEnabled(False)

        def job():
            edi_io.drain_reader_fallbacks()          # clear anything stale first
            sts = edi_z.read_dir(path, excl)
            fell_back = edi_io.drain_reader_fallbacks()
            curve = edi_z.ssq_resistivity_curve(sts)
            per = []
            for s in sts:
                t = s.periods
                order = t.argsort()
                per.append((t[order], s.ssq_app_res()[order],
                            s.c_response_km()[order]))
            return per, curve, fell_back

        run_async(self, job, self._after, self._error)

    def _after(self, result):
        self.busy.hide()
        self.compute_btn.setEnabled(True)
        self._per_station, self._curve, fell_back = result
        if fell_back:
            self.log("⚠ " + edi_io.summarize_fallbacks(fell_back))
        c = self._curve
        g = self.project.gofem
        # clamp the period window to the data range on first look
        if g.period_min_s <= c.period_min_s:
            g.period_min_s = c.period_min_s
        if g.period_max_s >= c.period_max_s:
            g.period_max_s = c.period_max_s
        self.pmin.setText(f"{g.period_min_s:g}")
        self.pmax.setText(f"{g.period_max_s:g}")
        self.summary.setText(
            f"{c.n_stations} stations   |   data periods "
            f"{c.period_min_s:.3g}–{c.period_max_s:.3g} s   |   "
            f"log-mean ρ_ssq: shallow {c.rho_shallow:.0f}   deep {c.rho_deep:.0f} Ω·m")
        self._plot()
        self.log(f"SSQ + C-response depth computed for {c.n_stations} stations")

    def _error(self, exc: Exception):
        self.busy.hide()
        self.compute_btn.setEnabled(True)
        self.log(f"ERROR: {exc}")
        self.summary.setText("<pre>" + "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]) + "</pre>")

    # ------------------------------------------------------------------ #
    def _plot(self):
        if getattr(self.canvas, "ax", None) is None or self._curve is None:
            return
        import numpy as np

        c = self._curve
        pmin, pmax = self._periods()
        t0, t1 = float(c.period_s.min()), float(c.period_s.max())
        fig = self.canvas.figure
        fig.clear()
        ax_r = fig.add_subplot(2, 1, 1)
        ax_d = fig.add_subplot(2, 1, 2)

        # -- apparent resistivity --
        rho_all = []
        for t, rho, _ in self._per_station:
            ok = np.isfinite(t) & np.isfinite(rho) & (rho > 0)
            if ok.sum() >= 2:
                ax_r.plot(t[ok], rho[ok], color="0.75", lw=0.6, alpha=0.5, zorder=1)
                rho_all.append(rho[ok])
        ax_r.fill_between(c.period_s, c.rho_p16, c.rho_p84, color="tab:blue",
                          alpha=0.18, zorder=2, label="16–84 %")
        ax_r.plot(c.period_s, c.rho_logmean, color="tab:blue", lw=2.5, zorder=3,
                  label="log-mean")
        self._marks(ax_r, c.period_s, c.rho_logmean, pmin, pmax)
        ax_r.set_xscale("log")
        ax_r.set_xlim(t0, t1)
        ax_r.set_yscale("log")
        self._robust_ylim(ax_r, rho_all)
        ax_r.set_ylabel(r"$\rho_{\mathrm{ssq}}$  ($\Omega\cdot$m)")
        ax_r.set_title("SSQ apparent resistivity")
        ax_r.grid(True, which="both", alpha=0.3)
        ax_r.legend(loc="best", fontsize=8)

        # -- C-response penetration depth = Re{C} --
        depth_lm = c.c_real_km
        d_all = []
        for t, _, cc in self._per_station:
            d = cc.real
            ok = np.isfinite(t) & np.isfinite(d) & (d > 0)
            if ok.sum() >= 2:
                ax_d.plot(t[ok], d[ok], color="0.75", lw=0.6, alpha=0.5, zorder=1)
                d_all.append(d[ok])
        ax_d.plot(c.period_s, depth_lm, color="tab:red", lw=2.5, zorder=3,
                  label="log-mean  Re{C}")
        ax_d.plot(c.period_s, np.hypot(c.c_real_km, c.c_imag_km), color="tab:red",
                  lw=1.2, ls="--", zorder=3, label="|C|")
        self._marks(ax_d, c.period_s, depth_lm, pmin, pmax)
        ax_d.set_xscale("log")
        ax_d.set_xlim(t0, t1)
        ax_d.set_yscale("log")
        self._robust_ylim(ax_d, d_all)
        ax_d.invert_yaxis()                       # deeper downward
        ax_d.set_xlabel("period (s)")
        ax_d.set_ylabel("penetration depth (km)")
        ax_d.set_title("C-response depth")
        ax_d.grid(True, which="both", alpha=0.3)
        ax_d.legend(loc="best", fontsize=8)

        self.canvas.redraw()

    @staticmethod
    def _robust_ylim(ax, arrays, lo_pct=1.0, hi_pct=99.0, pad=3.0):
        """Clamp a log y-axis to the 1–99th percentile of the station cloud
        (x pad each way) so a handful of bad-data stations with runaway
        values don't compress everything else into a flat line."""
        import numpy as np

        if not arrays:
            return
        v = np.concatenate([np.asarray(a) for a in arrays])
        v = v[np.isfinite(v) & (v > 0)]
        if v.size < 5:
            return
        lo = np.percentile(v, lo_pct) / pad
        hi = np.percentile(v, hi_pct) * pad
        if lo > 0 and hi > lo:
            ax.set_ylim(lo, hi)

    @staticmethod
    def _marks(ax, period, values, pmin, pmax):
        import numpy as np

        finite = np.isfinite(period) & np.isfinite(values)
        if finite.sum() < 2:
            return
        for p in (pmin, pmax):
            y = np.interp(p, period[finite], values[finite],
                          left=np.nan, right=np.nan)
            if np.isfinite(y):
                ax.plot([p], [y], "o", mfc="none", mec="k", mew=1.6, ms=11, zorder=6)

    def refresh(self) -> None:
        g = self.project.gofem
        self.pmin.setText(f"{g.period_min_s:g}")
        self.pmax.setText(f"{g.period_max_s:g}")
        if self._curve is not None:
            self._plot()
