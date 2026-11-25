"""Tab 3 -- model -> GeoTIFF depth slices."""
from __future__ import annotations

import copy
import os

import numpy as np
from PyQt5.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel,
    QLineEdit, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from ..backends import geotiff as GT
from ..backends import render, slicing
from .widgets import (
    DepthListEdit, ExtentBox, FigurePreview, PathRow, PlotPanel,
    run_async_progress, scroll_column,
)


class GeotiffPanel(PlotPanel):
    title = "5 - GeoTIFF"

    def build(self) -> None:
        self._loading = False
        root = QHBoxLayout(self)
        col = QWidget()
        left = QVBoxLayout(col)

        box = QGroupBox("Depths")
        form = QFormLayout(box)
        self.depths = DepthListEdit()
        self.depths.changed.connect(self._update_info)
        self.reference = QComboBox()
        for key, label in slicing.REFERENCES.items():
            self.reference.addItem(label, key)
        form.addRow("depths (m)", self.depths)
        form.addRow("measured", self.reference)
        left.addWidget(box)

        box = QGroupBox("Raster")
        form = QFormLayout(box)
        self.pixel = QDoubleSpinBox(decimals=1, minimum=1.0, maximum=100000.0, suffix=" m")
        self.pixel.valueChanged.connect(self._update_info)
        self.quantity = QComboBox()
        for key, (label, _fn) in GT.QUANTITIES.items():
            self.quantity.addItem(label, key)
        self.layout_combo = QComboBox()
        for key, label in GT.LAYOUTS.items():
            self.layout_combo.addItem(label, key)
        self.layout_combo.currentIndexChanged.connect(self._update_info)
        form.addRow("pixel size", self.pixel)
        form.addRow("value", self.quantity)
        form.addRow("files", self.layout_combo)
        left.addWidget(box)

        self.extent = ExtentBox("Extent (display CRS)")
        self.extent.changed.connect(self._update_info)
        self.extent.from_view.clicked.connect(self._extent_from_view)
        left.addWidget(self.extent)

        box = QGroupBox("Output")
        form = QFormLayout(box)
        self.out_dir = PathRow("output folder", kinds=("folder",))
        self.prefix = QLineEdit()
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: gray;")
        self.prev_depth = QComboBox()
        self.prev_depth.setEditable(True)
        prev_btn = QPushButton("Preview")
        prev_btn.clicked.connect(self._preview)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("depth"))
        prow.addWidget(self.prev_depth, 1)
        prow.addWidget(prev_btn)
        self.export_btn = QPushButton("Write GeoTIFF(s)")
        self.export_btn.clicked.connect(self._export)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        form.addRow("folder", self.out_dir)
        form.addRow("file prefix", self.prefix)
        form.addRow(self.info)
        form.addRow(prow)
        form.addRow(self.export_btn)
        form.addRow(self.progress)
        left.addWidget(box)
        left.addStretch(1)
        root.addWidget(scroll_column(col))

        self.preview = FigurePreview()
        self.preview.set_size_inches(7.0, 6.0)
        root.addWidget(self.preview, 1)

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        self._loading = True
        g = self.project.geotiff
        self.depths.set_depths(g.depths_m)
        self.reference.setCurrentIndex(max(self.reference.findData(g.reference), 0))
        self.pixel.setValue(g.pixel_m)
        self.quantity.setCurrentIndex(max(self.quantity.findData(g.quantity), 0))
        self.layout_combo.setCurrentIndex(max(self.layout_combo.findData(g.layout), 0))
        self.extent.load(g.extent)
        self.out_dir.setText(g.out_dir)
        self.prefix.setText(g.prefix)
        self._loading = False
        self._update_info()

    def on_shown(self) -> None:
        self._update_info()

    def _store(self) -> None:
        g = self.project.geotiff
        g.depths_m = self.depths.depths() or g.depths_m
        g.reference = self.reference.currentData()
        g.pixel_m = self.pixel.value()
        g.quantity = self.quantity.currentData()
        g.layout = self.layout_combo.currentData()
        self.extent.store(g.extent)
        g.out_dir = self.out_dir.text()
        g.prefix = self.prefix.text().strip() or "depth_slice"

    def _grid(self) -> GT.RasterGrid:
        g = self.project.geotiff
        x0, x1, y0, y1 = self.session.extent(g.extent)
        return GT.RasterGrid.from_bounds(x0, x1, y0, y1, g.pixel_m)

    def _update_info(self, *_):
        if self._loading:
            return
        self._store()
        g = self.project.geotiff
        cur = self.prev_depth.currentText()
        self.prev_depth.clear()
        self.prev_depth.addItems([f"{d:g}" for d in g.depths_m])
        if cur:
            self.prev_depth.setCurrentText(cur)
        if not self.session.ready:
            self.info.setText("load + georeference a model on tab 1")
            return
        try:
            grid = self._grid()
        except ValueError as exc:
            self.info.setText(str(exc))
            return
        n_files = 1 if g.layout == "stack" else len(g.depths_m)
        mb = grid.nx * grid.ny * 4 * len(g.depths_m) / 1e6
        self.info.setText(f"{grid.nx} × {grid.ny} pixels, {len(g.depths_m)} depth(s) -> "
                          f"{n_files} file(s), ≤ {mb:.1f} MB uncompressed; "
                          f"{self.session.maptr.display_label}")

    def _extent_from_view(self):
        lim = self.preview.current_limits()
        if lim:
            self.extent.set_custom(*lim)

    # ------------------------------------------------------------------ #
    def _preview(self) -> None:
        if not self.session.ready:
            return
        self._store()
        try:
            depth = float(self.prev_depth.currentText())
            grid = self._grid()
        except ValueError as exc:
            self.log(f"preview: {exc}")
            return
        g = self.project.geotiff
        vals = GT.sample_slice(self.session.model, self.session.maptr, grid, depth, g.reference)
        color = render.color_of(self.session, self.project.plan.color)
        fig = self.preview.figure
        fig.clear()
        fig.set_layout_engine("constrained")
        ax = fig.add_subplot(111)
        im = ax.imshow(10.0 ** vals, extent=grid.extent, origin="upper", interpolation="nearest",
                       cmap=render.make_cmap(color.cmap, color.n_colors), norm=render.rho_norm(color))
        render._map_axes(ax, grid.extent)
        render.draw_stations(ax, self.session.stations_display(), size=8)
        ax.set_title(f"{slicing.depth_label(depth, g.reference)}  ({grid.nx} × {grid.ny} px, "
                     f"{np.isnan(vals).mean() * 100:.0f}% nodata)", fontsize="small")
        render.add_colorbar(fig, ax, im, "right", extend=render.cbar_extend(color))
        self.preview.draw()

    def _export(self) -> None:
        if not self.session.ready:
            return
        self._store()
        g = copy.deepcopy(self.project.geotiff)
        if not g.out_dir:
            self.log("choose an output folder first")
            return
        try:
            grid = self._grid()
        except ValueError as exc:
            self.log(str(exc))
            return
        self.export_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(g.depths_m))

        def done(paths):
            self.export_btn.setEnabled(True)
            self.progress.setVisible(False)
            self.log(f"wrote {len(paths)} GeoTIFF(s) to {g.out_dir}")

        def failed(exc):
            self.export_btn.setEnabled(True)
            self.progress.setVisible(False)
            self.log(f"GeoTIFF export failed: {exc}")

        def prog(i, n, msg):
            self.progress.setValue(i)
            self.progress.setFormat(f"{i}/{n}  {msg}")

        model, maptr = self.session.model, self.session.maptr
        run_async_progress(
            self, lambda cb: GT.export(model, maptr, g.depths_m, grid, g.out_dir,
                                       reference=g.reference, quantity=g.quantity,
                                       layout=g.layout, prefix=g.prefix, progress=cb),
            done, failed, prog)
        self.log(f"writing GeoTIFFs to {os.path.abspath(g.out_dir)} ...")
