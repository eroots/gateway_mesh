"""Tab 2 -- plan-view (depth slice) figures."""
from __future__ import annotations

import copy
import os
import time

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QProgressBar, QPushButton, QSpinBox,
    QVBoxLayout, QWidget,
)

from ..backends import render, slicing
from .widgets import (
    ColorBox, Debounce, DepthListEdit, ExtentBox, FigurePreview, PathRow,
    PlotPanel, run_async_progress, scroll_column,
)

_MARKERS = ["v", "^", "o", "s", "D", "."]
_CBAR = [("right", "right"), ("bottom", "bottom"), ("none", "none")]


class PlanPanel(PlotPanel):
    title = "3 - Plan view"

    def build(self) -> None:
        self._loading = False
        self._redraw = Debounce(self._draw, 250, self)
        root = QHBoxLayout(self)
        col = QWidget()
        left = QVBoxLayout(col)

        # ---- depth ------------------------------------------------------- #
        box = QGroupBox("Depth")
        form = QFormLayout(box)
        self.depths = DepthListEdit()
        self.depths.changed.connect(self._depths_changed)
        self.depth = QComboBox()
        self.depth.setEditable(True)
        self.depth.setToolTip("Depth shown in the preview (m). Pick from the list or type one.")
        self.depth.currentTextChanged.connect(self._changed)
        prev_b, next_b = QPushButton("◀"), QPushButton("▶")
        for b, step in ((prev_b, -1), (next_b, 1)):
            b.setMaximumWidth(32)
            b.clicked.connect(lambda _=False, s=step: self._step_depth(s))
        row = QHBoxLayout()
        row.addWidget(self.depth, 1)
        row.addWidget(prev_b)
        row.addWidget(next_b)
        self.reference = QComboBox()
        for key, label in slicing.REFERENCES.items():
            self.reference.addItem(label, key)
        self.reference.currentIndexChanged.connect(self._changed)
        form.addRow("depth list (m)", self.depths)
        form.addRow("preview depth", row)
        form.addRow("measured", self.reference)
        self.crinkle = QCheckBox("crinkle slice (whole cells, drawn flat)")
        self.crinkle.setToolTip(
            "Draw every cell the depth passes through as its whole map footprint, as\n"
            "ParaView's crinkle slice does, instead of cutting the cells exactly.\n"
            "Removes the slivers, gaps and overlaps the exact cut leaves where cells are\n"
            "warped (topography fit) or meet at hanging nodes. Cell edges then show\n"
            "as steps where a boundary crosses the depth at a slant (e.g. the ground\n"
            "surface on slices above sea level). GeoTIFFs are unaffected.")
        self.crinkle.toggled.connect(self._changed)
        form.addRow(self.crinkle)
        self.main_layer = QComboBox()
        self.main_layer.setToolTip("What fills the map: the model, or one of the raster layers\n"
                                   "(e.g. a magnetics map with the same stations, extent and overlays)")
        self.main_layer.currentIndexChanged.connect(self._changed)
        form.addRow("main image", self.main_layer)
        left.addWidget(box)

        # ---- map --------------------------------------------------------- #
        self.extent = ExtentBox("Map extent")
        self.extent.changed.connect(self._changed)
        self.extent.from_view.clicked.connect(self._extent_from_view)
        left.addWidget(self.extent)
        box = QGroupBox("Lat / lon grid")
        form = QFormLayout(box)
        self.graticule = QCheckBox("draw lon/lat lines (labels top + right)")
        self.graticule.toggled.connect(self._changed)
        self.grat_step = QDoubleSpinBox(decimals=2, minimum=0.0, maximum=30.0, singleStep=0.25,
                                        suffix=" °")
        self.grat_step.setSpecialValueText("auto")
        self.grat_step.valueChanged.connect(self._changed)
        form.addRow(self.graticule)
        form.addRow("spacing", self.grat_step)
        left.addWidget(box)

        # ---- colour ------------------------------------------------------ #
        self.color = ColorBox()
        self.color.changed.connect(self._changed)
        self.cbar = QComboBox()
        for key, label in _CBAR:
            self.cbar.addItem(label, key)
        self.cbar.currentIndexChanged.connect(self._changed)
        self.color.layout().addRow("colour bar", self.cbar)
        left.addWidget(self.color)

        # ---- stations ---------------------------------------------------- #
        box = QGroupBox("Stations")
        form = QFormLayout(box)
        self.st_show = QCheckBox("show")
        self.st_labels = QCheckBox("names")
        row = QHBoxLayout()
        row.addWidget(self.st_show)
        row.addWidget(self.st_labels)
        row.addStretch(1)
        self.st_marker = QComboBox()
        self.st_marker.addItems(_MARKERS)
        self.st_size = QDoubleSpinBox(decimals=1, minimum=0.5, maximum=200.0)
        self.st_color = QComboBox()
        self.st_color.setEditable(True)
        self.st_color.addItems(["white", "black", "magenta", "0.5"])
        for w in (self.st_show, self.st_labels):
            w.toggled.connect(self._changed)
        self.st_marker.currentIndexChanged.connect(self._changed)
        self.st_size.valueChanged.connect(self._changed)
        self.st_color.currentTextChanged.connect(self._changed)
        mrow = QHBoxLayout()
        mrow.addWidget(self.st_marker)
        mrow.addWidget(QLabel("size"))
        mrow.addWidget(self.st_size)
        mrow.addWidget(QLabel("colour"))
        mrow.addWidget(self.st_color, 1)
        form.addRow(row)
        form.addRow("marker", mrow)
        left.addWidget(box)

        # ---- figure + export --------------------------------------------- #
        box = QGroupBox("Figure")
        form = QFormLayout(box)
        self.title_edit = QLineEdit()
        self.title_edit.setToolTip("Placeholders: {depth_label} {depth_km} {depth_m} {model} "
                                   "{reference}. Blank = no title.")
        self.title_edit.editingFinished.connect(self._changed)
        self.w_in = QDoubleSpinBox(decimals=2, minimum=1.0, maximum=40.0, singleStep=0.25, suffix=" in")
        self.h_in = QDoubleSpinBox(decimals=2, minimum=1.0, maximum=40.0, singleStep=0.25, suffix=" in")
        for w in (self.w_in, self.h_in):
            w.editingFinished.connect(self._size_changed)
        srow = QHBoxLayout()
        srow.addWidget(self.w_in)
        srow.addWidget(QLabel("×"))
        srow.addWidget(self.h_in)
        self.font = QDoubleSpinBox(decimals=1, minimum=4.0, maximum=30.0, suffix=" pt")
        self.font.editingFinished.connect(self._changed)
        self.dpi = QSpinBox(minimum=50, maximum=1200, singleStep=50)
        form.addRow("title", self.title_edit)
        form.addRow("size", srow)
        form.addRow("font", self.font)
        form.addRow("raster dpi", self.dpi)
        left.addWidget(box)

        box = QGroupBox("Export")
        form = QFormLayout(box)
        self.out_dir = PathRow("folder for 'Export all depths'", kinds=("folder",))
        self.template = QLineEdit()
        self.template.setToolTip("File name per depth. Placeholders: {depth_tag} {depth_m} "
                                 "{reference} {model}. Extension picks the format "
                                 "(.png .pdf .svg .tif).")
        one = QPushButton("Export this depth...")
        one.clicked.connect(self._export_one)
        self.all_btn = QPushButton("Export all depths")
        self.all_btn.clicked.connect(self._export_all)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        brow = QHBoxLayout()
        brow.addWidget(one)
        brow.addWidget(self.all_btn)
        form.addRow("folder", self.out_dir)
        form.addRow("file name", self.template)
        form.addRow(brow)
        form.addRow(self.progress)
        left.addWidget(box)
        left.addStretch(1)
        root.addWidget(scroll_column(col))

        self.preview = FigurePreview()
        root.addWidget(self.preview, 1)

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        self._loading = True
        p = self.project.plan
        self.depths.set_depths(p.depths_m)
        self._fill_depth_combo(p.depths_m, p.depth_m)
        self.reference.setCurrentIndex(max(self.reference.findData(p.reference), 0))
        self.crinkle.setChecked(p.crinkle)
        self.main_layer.clear()
        self.main_layer.addItem("model resistivity", "")
        for lay in self.project.layers:
            if lay.kind in ("raster", "surface"):
                self.main_layer.addItem(f"layer: {lay.name}", lay.name)
        self.main_layer.setCurrentIndex(max(self.main_layer.findData(p.main_layer), 0))
        self.extent.load(p.extent)
        self.graticule.setChecked(p.graticule)
        self.grat_step.setValue(p.graticule_step_deg)
        self.color.load(p.color)
        linked = self.project.house.enabled and self.project.house.link_colors
        self.color.setEnabled(not linked)
        self.color.setTitle("Colour scale (house style, tab 6)" if linked else "Colour scale")
        self.cbar.setCurrentIndex(max(self.cbar.findData(p.colorbar), 0))
        self.st_show.setChecked(p.show_stations)
        self.st_labels.setChecked(p.station_labels)
        self.st_marker.setCurrentText(p.station_marker)
        self.st_size.setValue(p.station_size)
        self.st_color.setCurrentText(p.station_color)
        self.title_edit.setText(p.title)
        self.w_in.setValue(p.width_in)
        self.h_in.setValue(p.height_in)
        self.font.setValue(p.font_size)
        self.dpi.setValue(int(p.dpi))
        self.out_dir.setText(p.out_dir)
        self.template.setText(p.file_template)
        self.preview.set_size_inches(*render.plan_size(p, self.session))
        self._loading = False
        self._redraw.poke()

    def on_shown(self) -> None:
        self._redraw.poke()

    def _store(self) -> None:
        p = self.project.plan
        p.depths_m = self.depths.depths() or p.depths_m
        try:
            p.depth_m = float(self.depth.currentText())
        except ValueError:
            pass
        p.reference = self.reference.currentData()
        p.crinkle = self.crinkle.isChecked()
        p.main_layer = self.main_layer.currentData() or ""
        self.extent.store(p.extent)
        p.graticule = self.graticule.isChecked()
        p.graticule_step_deg = self.grat_step.value()
        self.color.store(p.color)
        p.colorbar = self.cbar.currentData()
        p.show_stations = self.st_show.isChecked()
        p.station_labels = self.st_labels.isChecked()
        p.station_marker = self.st_marker.currentText()
        p.station_size = self.st_size.value()
        p.station_color = self.st_color.currentText().strip() or "white"
        p.title = self.title_edit.text()
        p.width_in = self.w_in.value()
        p.height_in = self.h_in.value()
        p.font_size = self.font.value()
        p.dpi = self.dpi.value()
        p.out_dir = self.out_dir.text()
        p.file_template = self.template.text().strip() or "plan_{depth_tag}m.png"

    # ------------------------------------------------------------------ #
    def _fill_depth_combo(self, depths, current) -> None:
        self.depth.blockSignals(True)
        self.depth.clear()
        self.depth.addItems([f"{d:g}" for d in depths])
        self.depth.setCurrentText(f"{current:g}")
        self.depth.blockSignals(False)

    def _depths_changed(self):
        cur = self.depth.currentText()
        self._fill_depth_combo(self.depths.depths(), float(cur) if cur else 0.0)
        self._changed()

    def _step_depth(self, step: int):
        n = self.depth.count()
        if not n:
            return
        i = self.depth.findText(self.depth.currentText())
        i = 0 if i < 0 else min(max(i + step, 0), n - 1)
        self.depth.setCurrentIndex(i)

    def _changed(self, *_):
        if not self._loading:
            self._redraw.poke()

    def _size_changed(self):
        self._store()
        self.preview.set_size_inches(*render.plan_size(self.project.plan, self.session))
        self._changed()

    def _extent_from_view(self):
        lim = self.preview.current_limits()
        if lim:
            self.extent.set_custom(*lim)

    # ------------------------------------------------------------------ #
    def _draw(self) -> None:
        self._store()
        fig = self.preview.figure
        if not self.session.ready:
            fig.clear()
            fig.text(0.5, 0.5, "load + georeference a model on tab 1", ha="center",
                     va="center", color="0.4")
            self.preview.draw()
            return
        t = time.time()
        try:
            render.render_plan(fig, self.session, self.project.plan, layers=self.project.layers)
        except Exception as exc:  # noqa: BLE001 -- show, don't crash the GUI
            fig.clear()
            fig.text(0.5, 0.5, f"could not draw:\n{exc}", ha="center", va="center", color="#b91c1c")
            self.log(f"plan view failed: {exc}")
        self.preview.draw()
        dt = time.time() - t
        if dt > 2.0:
            self.log(f"plan view drawn in {dt:.1f} s")

    def _export_one(self) -> None:
        if not self.session.ready:
            return
        self._store()
        p = self.project.plan
        start = os.path.join(p.out_dir or self.project.workdir or os.getcwd(),
                             render.plan_file_name(p, p.depth_m, self.session))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", start,
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif);;all files (*)")
        if not path:
            return
        try:
            render.export_plan(path, self.session, p, layers=self.project.layers)
        except Exception as exc:  # noqa: BLE001
            self.log(f"export failed: {exc}")
            return
        self.log(f"wrote {path}")

    def _export_all(self) -> None:
        if not self.session.ready:
            return
        self._store()
        p = self.project.plan
        out = p.out_dir or QFileDialog.getExistingDirectory(self, "Export all depths to")
        if not out:
            return
        p.out_dir = out
        self.out_dir.setText(out)
        self.all_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(p.depths_m))
        self.progress.setValue(0)

        def done(paths):
            self.all_btn.setEnabled(True)
            self.progress.setVisible(False)
            self.log(f"wrote {len(paths)} plan views to {out}")

        def failed(exc):
            self.all_btn.setEnabled(True)
            self.progress.setVisible(False)
            self.log(f"batch export failed: {exc}")

        def prog(i, n, msg):
            self.progress.setValue(i)
            self.progress.setFormat(f"{i}/{n}  {msg}")

        snapshot = copy.deepcopy(p)       # the GUI may change p while the batch runs
        layers = copy.deepcopy(self.project.layers)
        run_async_progress(self, lambda cb: render.export_plan_all(out, self.session, snapshot, cb,
                                                                   layers=layers),
                           done, failed, prog)
