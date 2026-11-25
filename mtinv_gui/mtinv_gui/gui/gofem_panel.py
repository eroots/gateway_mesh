"""Step 5 -- write GoFEM solver inputs (frequencies, data file, .prm files)."""
from __future__ import annotations

import os
import traceback

from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QProgressBar, QPushButton, QSpinBox, QVBoxLayout,
)

from ..backends import gofem_files, mesh as mesh_backend
from .panel_base import BasePanel
from .worker import run_async


class GofemPanel(BasePanel):
    title = "6 - GoFEM files"

    def build(self) -> None:
        root = QVBoxLayout(self)

        per_box = QGroupBox("Period selection (EdiCollection.select_periods)")
        pf = QFormLayout(per_box)
        self.pct = QDoubleSpinBox()
        self.pct.setRange(0, 100)
        self.pct.setValue(self.project.gofem.period_percentage)
        self.pct.setSuffix(" %")
        self.pmin = QDoubleSpinBox()
        self.pmin.setRange(1e-4, 1e5)
        self.pmin.setDecimals(4)
        self.pmin.setValue(self.project.gofem.period_min_s)
        self.pmax = QDoubleSpinBox()
        self.pmax.setRange(1e-4, 1e6)
        self.pmax.setValue(self.project.gofem.period_max_s)
        pf.addRow("keep periods in >= N% of sites", self.pct)
        pf.addRow("min period (s)", self.pmin)
        pf.addRow("max period (s)", self.pmax)
        root.addWidget(per_box)

        data_box = QGroupBox("Data file (write_edi_collection_to_gofem)")
        df = QFormLayout(data_box)
        self.dtype = QComboBox()
        self.dtype.addItems(["Z", "PT", "Tipper", "Z_Tipper"])
        self.efloor = QDoubleSpinBox()
        self.efloor.setRange(0, 1)
        self.efloor.setSingleStep(0.01)
        self.efloor.setValue(self.project.gofem.error_floor)
        df.addRow("data type", self.dtype)
        df.addRow("error floor", self.efloor)
        root.addWidget(data_box)

        prm_box = QGroupBox("Parameter files (ParameterHandler)")
        mf = QFormLayout(prm_box)
        self.order = QSpinBox()
        self.order.setRange(1, 3)
        self.order.setValue(self.project.gofem.element_order)
        self.bc = QComboBox()
        self.bc.addItems(["Dirichlet2D", "Dirichlet1D", "Neumann"])
        self.refrx = QSpinBox()
        self.refrx.setRange(0, 5)
        self.refrx.setValue(self.project.gofem.refine_around_receivers)
        self.nfreq = QSpinBox()
        self.nfreq.setRange(1, 512)
        self.nfreq.setValue(self.project.gofem.n_parallel_frequencies)
        self.rms = QDoubleSpinBox()
        self.rms.setRange(0.01, 100)
        self.rms.setValue(self.project.gofem.target_rms)
        self.iters = QSpinBox()
        self.iters.setRange(1, 1000)
        self.iters.setValue(self.project.gofem.max_iterations)
        self.inner = QSpinBox()
        self.inner.setRange(1, 1000)
        self.inner.setValue(self.project.gofem.inner_iterations)
        self.regupd = QSpinBox()
        self.regupd.setRange(1, 100)
        self.regupd.setValue(self.project.gofem.regularization_update)
        self.s0 = QDoubleSpinBox()
        self.s0.setRange(1e-4, 100)
        self.s0.setDecimals(4)
        self.s0.setValue(self.project.gofem.scaling_factor_s0)
        mf.addRow("element order", self.order)
        mf.addRow("boundary condition", self.bc)
        mf.addRow("refine cells around receivers", self.refrx)
        mf.addRow("parallel frequencies", self.nfreq)
        mf.addRow("target RMS", self.rms)
        mf.addRow("outer / inner iterations", self._pair(self.iters, self.inner))
        mf.addRow("regularization update", self.regupd)
        mf.addRow("initial scaling factor s0", self.s0)
        self.adbox = QLineEdit()
        self.adbox.setPlaceholderText("Active domain box: blank = derive from survey scales")
        mf.addRow("active domain box", self.adbox)
        root.addWidget(prm_box)

        self.femtic_box = QGroupBox("FEMTIC run (control.dat) -- the mesh is built for FEMTIC")
        ff = QFormLayout(self.femtic_box)
        self.f_iters = QSpinBox()
        self.f_iters.setRange(0, 1000)
        self.f_iters.setSpecialValueText("0 (forward run only)")
        self.f_threads = QSpinBox()
        self.f_threads.setRange(1, 256)
        self.f_tradeoff = QDoubleSpinBox()
        self.f_tradeoff.setRange(1e-4, 1e6)
        self.f_tradeoff.setDecimals(4)
        ff.addRow("iterations", self.f_iters)
        ff.addRow("threads per MPI process", self.f_threads)
        ff.addRow("trade-off parameter", self.f_tradeoff)
        hint = QLabel("Written to <i>&lt;output dir&gt;/femtic/</i> with copies of mesh.dat and "
                      "resistivity_block_iter0.dat: observe.dat has the same stations, frequencies and "
                      "error floor as the GoFEM data file (SI units, e<sup>-i&omega;t</sup>).")
        hint.setWordWrap(True)
        ff.addRow(hint)
        root.addWidget(self.femtic_box)

        out = QHBoxLayout()
        out.addWidget(QLabel("output dir (under project):"))
        self.outdir = QLineEdit(self.project.gofem.output_dir)
        out.addWidget(self.outdir, 1)
        root.addLayout(out)

        btns = QHBoxLayout()
        self.all_btn = QPushButton("Write frequencies + data + .prm files")
        self.all_btn.clicked.connect(self._all)
        btns.addWidget(self.all_btn)
        self.freq_btn = QPushButton("Frequencies only")
        self.freq_btn.clicked.connect(self._freqs_only)
        btns.addWidget(self.freq_btn)
        root.addLayout(btns)

        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.hide()
        root.addWidget(self.busy)

        self.result = QPlainTextEdit(readOnly=True)
        self.result.setMaximumHeight(140)
        root.addWidget(self.result)

        self.note = QLabel()
        self.note.setWordWrap(True)
        root.addWidget(self.note)
        self._refresh_note()
        root.addStretch(1)

    @staticmethod
    def _pair(a, b):
        from PyQt5.QtWidgets import QWidget

        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(a)
        lay.addWidget(b)
        return w

    # ------------------------------------------------------------------ #
    def _sync(self):
        g = self.project.gofem
        g.period_percentage = self.pct.value()
        g.period_min_s = self.pmin.value()
        g.period_max_s = self.pmax.value()
        g.data_type = self.dtype.currentText()
        g.error_floor = self.efloor.value()
        g.element_order = self.order.value()
        g.boundary_condition = self.bc.currentText()
        g.refine_around_receivers = self.refrx.value()
        g.n_parallel_frequencies = self.nfreq.value()
        g.target_rms = self.rms.value()
        g.max_iterations = self.iters.value()
        g.inner_iterations = self.inner.value()
        g.regularization_update = self.regupd.value()
        g.scaling_factor_s0 = self.s0.value()
        g.output_dir = self.outdir.text().strip() or "gofem_run"
        g.femtic_iterations = self.f_iters.value()
        g.femtic_threads = self.f_threads.value()
        g.femtic_trade_off = self.f_tradeoff.value()
        txt = self.adbox.text().strip()
        if txt:
            try:
                g.active_domain_box = [float(x) for x in txt.replace(",", " ").split()]
            except ValueError:
                self.log("active domain box: expected 6 numbers -- ignored")
                g.active_domain_box = []
        else:
            g.active_domain_box = []

    def _out_dir(self) -> str:
        return os.path.join(self.project.workdir or ".", self.project.gofem.output_dir)

    def _busy(self, on: bool):
        self.busy.setVisible(on)
        for b in (self.all_btn, self.freq_btn):
            b.setEnabled(not on)

    def _all(self):
        self._sync()
        out_dir = self._out_dir()
        self._busy(True)
        self.log("writing GoFEM inputs -> %s" % out_dir)

        def job():
            return gofem_files.write_all(self.project, out_dir, self.log)

        run_async(self, job, self._after_all, self._error)

    def _freqs_only(self):
        self._sync()
        g = self.project.gofem
        mesh_dir = mesh_backend.mesh_output_dir(self.project)
        self._busy(True)

        def job():
            coll = gofem_files.build_edi_collection(self.project, log=self.log)
            periods = gofem_files.select_periods(
                coll, g.period_percentage, g.period_min_s, g.period_max_s)
            path = gofem_files.write_frequencies(
                periods, os.path.join(mesh_dir, gofem_files.FREQUENCIES_FILE))
            return {"frequencies": path, "periods": periods}

        run_async(self, job, self._after_all, self._error)

    def _after_all(self, res):
        self._busy(False)
        g = self.project.gofem
        if res.get("data"):
            g.data_file_path = res["data"][0]
        lines = [f"periods kept: {len(res.get('periods', []))}"]
        for key in ("frequencies", "modeling_prm", "inversion_prm", "femtic_dir"):
            if res.get(key):
                lines.append(f"{key}: {res[key]}")
        for d in res.get("data", []):
            lines.append(f"data: {d}")
        self.result.setPlainText("\n".join(lines))
        self.log("GoFEM inputs written")
        self.notify_changed()

    def _error(self, exc: Exception):
        self._busy(False)
        self.log(f"ERROR: {exc}")
        self.result.setPlainText("".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)))

    def _refresh_note(self):
        mesh_dir = mesh_backend.mesh_output_dir(self.project)
        have = os.path.isfile(os.path.join(mesh_dir, mesh_backend.MESH_TRIA))
        state = "found" if have else "NOT found yet -- run the Mesh tab's 'Build Full Mesh'"
        self.note.setText(
            "Needs mtpy + pyGoFEM on the path (Project ▸ Settings).<br>"
            f"Mesh files are read from <b>{mesh_dir}</b> "
            f"(set on the Mesh tab) — model.tria {state}.")

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        g = self.project.gofem
        self._refresh_note()
        self.pct.setValue(g.period_percentage)
        self.pmin.setValue(g.period_min_s)
        self.pmax.setValue(g.period_max_s)
        self.dtype.setCurrentText(g.data_type)
        self.efloor.setValue(g.error_floor)
        self.order.setValue(g.element_order)
        self.bc.setCurrentText(g.boundary_condition)
        self.refrx.setValue(g.refine_around_receivers)
        self.nfreq.setValue(g.n_parallel_frequencies)
        self.rms.setValue(g.target_rms)
        self.iters.setValue(g.max_iterations)
        self.inner.setValue(g.inner_iterations)
        self.regupd.setValue(g.regularization_update)
        self.s0.setValue(g.scaling_factor_s0)
        self.outdir.setText(g.output_dir)
        self.adbox.setText(" ".join(f"{x:g}" for x in g.active_domain_box))
        self.f_iters.setValue(g.femtic_iterations)
        self.f_threads.setValue(g.femtic_threads)
        self.f_tradeoff.setValue(g.femtic_trade_off)
        self.femtic_box.setVisible(self.project.mesh.output_format == "femtic")
