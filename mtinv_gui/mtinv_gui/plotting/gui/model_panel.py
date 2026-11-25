"""Tab 1 -- load a model, place it on the map, choose the display CRS."""
from __future__ import annotations

import os

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from ...gui.worker import run_async
from ..backends import georef as G
from ..backends import render
from ..backends.model_source import MODEL_SOURCES, get_source
from .widgets import FigurePreview, PathRow, PlotPanel, float_edit, read_float, scroll_column

_METHODS = [
    ("stations", "Match receivers to EDI coordinates"),
    ("mtinv", "From setup-GUI project (.mtinv.json)"),
    ("manual", "Manual CRS + origin"),
]
_DISPLAY_PRESETS = ["", "EPSG:32607", "EPSG:32608", "EPSG:32609", "EPSG:32610",
                    "EPSG:32611", "EPSG:32612", "EPSG:32613", "EPSG:3979"]


class ModelPanel(PlotPanel):
    title = "1 - Model"

    def build(self) -> None:
        root = QHBoxLayout(self)
        col = QWidget()
        left = QVBoxLayout(col)

        # ---- model ------------------------------------------------------ #
        box = QGroupBox("Model")
        form = QFormLayout(box)
        self.code = QComboBox()
        self.code.addItems(list(MODEL_SOURCES))
        self.model_path = PathRow("inversion result, e.g. model_ZK_31.vtu",
                                  file_filter=get_source("GoFEM").model_filter)
        self.model_path.changed.connect(self._on_model_path)
        self.rec_path = PathRow("auto: receivers_projected.csv next to / above the model",
                                file_filter="receivers (*.csv *.txt);;all files (*)")
        self.load_btn = QPushButton("Load model")
        self.load_btn.clicked.connect(self.load_model)
        self.model_info = QLabel("no model loaded")
        self.model_info.setWordWrap(True)
        form.addRow("code", self.code)
        form.addRow("model", self.model_path)
        form.addRow("receivers", self.rec_path)
        form.addRow(self.load_btn)
        form.addRow(self.model_info)
        left.addWidget(box)

        # ---- georeferencing --------------------------------------------- #
        box = QGroupBox("Georeferencing (model frame -> real CRS)")
        v = QVBoxLayout(box)
        self.method = QComboBox()
        for key, label in _METHODS:
            self.method.addItem(label, key)
        v.addWidget(self.method)
        self.pages = QStackedWidget()

        pg = QWidget()
        f = QFormLayout(pg)
        self.station_src = PathRow("EDI folder or .lst", kinds=("folder", "file"),
                                   file_filter="site list (*.lst);;EDI (*.edi);;all files (*)")
        self.fit_crs = QComboBox()
        self.fit_crs.setEditable(True)
        self.fit_crs.addItems(["auto"] + _DISPLAY_PRESETS[1:])
        self.fit_crs.setToolTip("CRS the model was built in. 'auto' tries the survey's UTM zones\n"
                                "and the setup GUI's local projections and keeps the best fit.")
        f.addRow("stations", self.station_src)
        f.addRow("model CRS", self.fit_crs)
        self.pages.addWidget(pg)

        pg = QWidget()
        f = QFormLayout(pg)
        self.mtinv_path = PathRow("*.mtinv.json", file_filter="setup project (*.mtinv.json)")
        f.addRow("project", self.mtinv_path)
        self.pages.addWidget(pg)

        pg = QWidget()
        f = QFormLayout(pg)
        self.man_crs = QLineEdit()
        self.man_crs.setPlaceholderText("EPSG:32608")
        self.man_x0 = float_edit(0.0, 140)
        self.man_y0 = float_edit(0.0, 140)
        f.addRow("model CRS", self.man_crs)
        f.addRow("origin easting (m)", self.man_x0)
        f.addRow("origin northing (m)", self.man_y0)
        f.addRow(QLabel("Origin = CRS coordinates of the model's (0, 0)."))
        self.pages.addWidget(pg)

        self.method.currentIndexChanged.connect(self.pages.setCurrentIndex)
        v.addWidget(self.pages)
        self.solve_btn = QPushButton("Solve georeferencing")
        self.solve_btn.clicked.connect(self._solve)
        v.addWidget(self.solve_btn)
        self.geo_info = QLabel("not georeferenced")
        self.geo_info.setWordWrap(True)
        v.addWidget(self.geo_info)
        left.addWidget(box)

        # ---- display CRS ------------------------------------------------- #
        box = QGroupBox("Display CRS (figures + GeoTIFFs)")
        f = QFormLayout(box)
        self.display = QComboBox()
        self.display.setEditable(True)
        self.display.addItems(_DISPLAY_PRESETS)
        self.display.lineEdit().setPlaceholderText("same as model CRS")
        self.display.setToolTip("Draw everything in this CRS, e.g. to match your other "
                                "rasters. Blank = the model's own CRS.")
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_display)
        row = QHBoxLayout()
        row.addWidget(self.display, 1)
        row.addWidget(apply_btn)
        f.addRow("CRS", row)
        self.display_info = QLabel("")
        self.display_info.setWordWrap(True)
        self.display_info.setStyleSheet("color: gray;")
        f.addRow(self.display_info)
        left.addWidget(box)

        self.labels_chk = QCheckBox("label stations on the overview")
        self.labels_chk.toggled.connect(lambda _on: self._draw())
        left.addWidget(self.labels_chk)
        left.addStretch(1)
        root.addWidget(scroll_column(col))

        self.preview = FigurePreview()
        self.preview.set_size_inches(7.0, 6.0)
        root.addWidget(self.preview, 1)

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        m, g = self.project.model, self.project.georef
        self.code.setCurrentText(m.code)
        self.model_path.setText(m.model_path)
        self.rec_path.setText(m.receivers_path)
        self.method.setCurrentIndex(max(self.method.findData(g.method), 0))
        self.station_src.setText(g.station_source)
        self.fit_crs.setCurrentText(g.fit_crs or "auto")
        self.mtinv_path.setText(g.mtinv_project)
        self.man_crs.setText(g.manual_crs)
        self.man_x0.setText(f"{g.manual_x0:.3f}")
        self.man_y0.setText(f"{g.manual_y0:.3f}")
        self.display.setCurrentText(g.display_crs)
        self._show_georef()

    def _store(self) -> None:
        m, g = self.project.model, self.project.georef
        m.code = self.code.currentText()
        m.model_path = self.model_path.text()
        m.receivers_path = self.rec_path.text()
        g.method = self.method.currentData()
        g.station_source = self.station_src.text()
        g.fit_crs = self.fit_crs.currentText().strip() or "auto"
        g.mtinv_project = self.mtinv_path.text()
        g.manual_crs = self.man_crs.text().strip()
        g.manual_x0 = read_float(self.man_x0, 0.0)
        g.manual_y0 = read_float(self.man_y0, 0.0)
        g.display_crs = self.display.currentText().strip()

    def _on_model_path(self, path: str):
        if path and not self.rec_path.text():
            guess = get_source(self.code.currentText()).find_receivers(path)
            if guess:
                self.rec_path.edit.setPlaceholderText(f"auto: {guess}")

    # ------------------------------------------------------------------ #
    def load_model(self) -> None:
        self._store()
        path = self.project.model.model_path
        if not path or not os.path.isfile(path):
            self.log(f"model file not found: {path!r}")
            return
        self.load_btn.setEnabled(False)
        self.model_info.setText("loading...")
        self.log(f"loading {path}")
        run_async(self, lambda: self.session.load_model(self.project.model),
                  self._loaded, self._load_failed)

    def _loaded(self, rec_path: str):
        self.load_btn.setEnabled(True)
        s = self.session
        lo, hi = s.model.log10_rho_range()
        zlo, zhi = s.model.surface().z_range
        txt = (f"{s.model.n_earth_cells:,} earth cells ({s.model.n_cells:,} total); "
               f"ρ {10 ** lo:.3g} - {10 ** hi:.3g} Ω·m; surface {zlo:.0f} to {zhi:.0f} m.<br>")
        if len(s.receivers):
            txt += f"{len(s.receivers)} receivers from {os.path.basename(rec_path)}"
        else:
            txt += "<span style='color:#b45309'>no receiver file found -- set one to plot stations "\
                   "or fit the georeferencing</span>"
        self.model_info.setText(txt)
        self.log(f"model loaded: {s.model.n_earth_cells} earth cells, {len(s.receivers)} receivers")
        if self.project.georef.crs:
            self._apply_stored_georef()
        elif (self.project.georef.method == "stations" and self.project.georef.station_source
              and len(s.receivers)):
            self._solve()
        self._show_georef()
        self._draw()
        self.notify_changed()

    def _load_failed(self, exc):
        self.load_btn.setEnabled(True)
        self.model_info.setText(f"<span style='color:#b91c1c'>load failed: {exc}</span>")
        self.log(f"model load failed: {exc}")

    # ------------------------------------------------------------------ #
    def _solve(self) -> None:
        self._store()
        if self.project.georef.method == "stations" and self.session.model is None:
            self.log("load the model first (its receivers are needed for the fit)")
            return
        self.solve_btn.setEnabled(False)
        self.geo_info.setText("solving...")
        run_async(self, lambda: self.session.solve_georef(self.project.georef),
                  self._solved, self._solve_failed)

    def _solved(self, _ref):
        self.solve_btn.setEnabled(True)
        g = self.project.georef
        self.log(f"georeferenced: {G.crs_label(g.crs)} origin ({g.x0:.2f}, {g.y0:.2f})")
        self._show_georef()
        self._draw()
        self.notify_changed()

    def _solve_failed(self, exc):
        self.solve_btn.setEnabled(True)
        self.geo_info.setText(f"<span style='color:#b91c1c'>{exc}</span>")
        self.log(f"georeferencing failed: {exc}")

    def _apply_stored_georef(self) -> None:
        try:
            self.session.apply_georef(self.project.georef)
        except Exception as exc:  # noqa: BLE001
            self.log(f"stored georeferencing is invalid: {exc}")
            self.session.maptr = None

    def _apply_display(self) -> None:
        self._store()
        if not self.project.georef.crs:
            self.log("solve the georeferencing first")
            return
        try:
            self.session.apply_georef(self.project.georef)
        except Exception as exc:  # noqa: BLE001
            self.display_info.setText(f"<span style='color:#b91c1c'>{exc}</span>")
            return
        self.log(f"display CRS: {self.session.maptr.display_label}")
        self._show_georef()
        self._draw()
        self.notify_changed()

    def _show_georef(self) -> None:
        g = self.project.georef
        if not g.crs:
            self.geo_info.setText("not georeferenced")
            self.display_info.setText("")
            return
        txt = f"<b>{G.crs_label(g.crs)}</b><br>origin E {g.x0:,.2f} m, N {g.y0:,.2f} m"
        if g.fit_rms_m >= 0:
            colour = "#15803d" if g.fit_rms_m < 1.0 else ("#b45309" if g.fit_rms_m < 100 else "#b91c1c")
            txt += (f"<br><span style='color:{colour}'>fit: {g.fit_matched} stations, "
                    f"RMS {g.fit_rms_m:.3g} m, max {g.fit_max_m:.3g} m</span>")
            fit = self.session.last_fit
            if fit and fit.unmatched:
                txt += f"<br>unmatched receivers: {', '.join(fit.unmatched[:8])}"\
                       + (" ..." if len(fit.unmatched) > 8 else "")
            if g.fit_rms_m >= 100:
                txt += "<br>Large residual: wrong CRS? Try another, or the setup project."
        self.geo_info.setText(txt)
        if self.session.maptr is not None:
            self.display_info.setText(f"drawing in {self.session.maptr.display_label}")

    def _draw(self) -> None:
        render.render_station_map(self.preview.figure, self.session,
                                  labels=self.labels_chk.isChecked())
        self.preview.draw()
