"""Small reusable widgets for the plotting panels."""
from __future__ import annotations

from typing import Callable

from PyQt5.QtCore import QObject, QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFileDialog,
    QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QVBoxLayout,
    QWidget,
)

from ...gui.panel_base import BasePanel
from ..backends import geotiff as GT
from ..backends import render


class PlotPanel(BasePanel):
    """A :class:`BasePanel` that also sees the shared runtime session."""

    def __init__(self, project, session, parent=None):
        self.session = session
        super().__init__(project, parent)


def scroll_column(widget: QWidget, width: int = 440) -> QScrollArea:
    """Wrap a controls column in a vertically scrolling area of fixed width
    (no horizontal scrolling: the column's fields shrink to fit instead)."""
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setWidget(widget)
    sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    sc.setFixedWidth(width)
    sc.setFrameShape(QScrollArea.NoFrame)
    widget.setMaximumWidth(width - 18)
    return sc


def float_edit(value: float = 0.0, width: int = 90) -> QLineEdit:
    e = QLineEdit(f"{value:g}")
    e.setMaximumWidth(width)
    return e


def read_float(edit: QLineEdit, default: float) -> float:
    try:
        return float(edit.text().strip())
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
class PathRow(QWidget):
    """Line edit + browse button(s).  ``kinds``: any of "file", "folder", "save"."""

    changed = pyqtSignal(str)

    def __init__(self, placeholder: str = "", kinds=("file",), file_filter: str = "all files (*)",
                 parent=None):
        super().__init__(parent)
        self.file_filter = file_filter
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.editingFinished.connect(lambda: self.changed.emit(self.text()))
        lay.addWidget(self.edit, 1)
        labels = {"file": "File...", "folder": "Folder...", "save": "..."}
        for kind in kinds:
            b = QPushButton(labels[kind])
            b.setMaximumWidth(70)
            b.clicked.connect(lambda _=False, k=kind: self._browse(k))
            lay.addWidget(b)

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, text: str) -> None:
        self.edit.setText(text or "")

    def _browse(self, kind: str):
        start = self.text()
        if kind == "folder":
            p = QFileDialog.getExistingDirectory(self, "Choose folder", start)
        elif kind == "save":
            p, _ = QFileDialog.getSaveFileName(self, "Save as", start, self.file_filter)
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Choose file", start, self.file_filter)
        if p:
            self.setText(p)
            self.changed.emit(p)


# --------------------------------------------------------------------------- #
class ExtentBox(QGroupBox):
    """Edit an :class:`~..project.ExtentInfo` (values shown in km)."""

    changed = pyqtSignal()
    MODES = [("stations", "Around stations"), ("model", "Whole model"), ("custom", "Custom")]

    def __init__(self, title: str = "Extent", parent=None):
        super().__init__(title, parent)
        form = QFormLayout(self)
        self.mode = QComboBox()
        for key, label in self.MODES:
            self.mode.addItem(label, key)
        self.padding = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=5000.0, suffix=" km")
        self.custom = [QDoubleSpinBox(decimals=2, minimum=-1e5, maximum=1e5, suffix=" km")
                       for _ in range(4)]
        grid = QGridLayout()
        for i, (lab, w) in enumerate(zip(("E min", "E max", "N min", "N max"), self.custom)):
            grid.addWidget(QLabel(lab), i // 2, 2 * (i % 2))
            grid.addWidget(w, i // 2, 2 * (i % 2) + 1)
        self.from_view = QPushButton("Use current preview view")
        self.from_view.setToolTip("Copy the zoomed/panned preview limits into the custom extent")
        form.addRow("mode", self.mode)
        form.addRow("padding", self.padding)
        form.addRow(grid)
        form.addRow(self.from_view)
        self.mode.currentIndexChanged.connect(self._sync)
        self.mode.currentIndexChanged.connect(self.changed)
        self.padding.editingFinished.connect(self.changed)
        for w in self.custom:
            w.editingFinished.connect(self.changed)
        self._sync()

    def _sync(self):
        key = self.mode.currentData()
        self.padding.setEnabled(key == "stations")
        for w in self.custom:
            w.setEnabled(key == "custom")

    def load(self, ext) -> None:
        self.blockSignals(True)
        i = self.mode.findData(ext.mode)
        self.mode.setCurrentIndex(max(i, 0))
        self.padding.setValue(ext.padding_m / 1000.0)
        for w, v in zip(self.custom, ext.custom or [0, 0, 0, 0]):
            w.setValue(v / 1000.0)
        self._sync()
        self.blockSignals(False)

    def store(self, ext) -> None:
        ext.mode = self.mode.currentData()
        ext.padding_m = self.padding.value() * 1000.0
        ext.custom = [w.value() * 1000.0 for w in self.custom]

    def set_custom(self, x0, x1, y0, y1) -> None:
        self.blockSignals(True)
        self.mode.setCurrentIndex(self.mode.findData("custom"))
        for w, v in zip(self.custom, (x0, x1, y0, y1)):
            w.setValue(v / 1000.0)
        self._sync()
        self.blockSignals(False)
        self.changed.emit()


# --------------------------------------------------------------------------- #
class DepthGenerateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate depths")
        form = QFormLayout(self)
        self.start = float_edit(100.0)
        self.stop = float_edit(50000.0)
        self.n = QSpinBox(minimum=1, maximum=500, value=12)
        self.spacing = QComboBox()
        self.spacing.addItem("logarithmic", "log")
        self.spacing.addItem("linear", "linear")
        self.round_to = float_edit(10.0)
        self.mode = QComboBox()
        self.mode.addItems(["replace the list", "add to the list"])
        form.addRow("first depth (m)", self.start)
        form.addRow("last depth (m)", self.stop)
        form.addRow("number", self.n)
        form.addRow("spacing", self.spacing)
        form.addRow("round to (m, 0 = off)", self.round_to)
        form.addRow("", self.mode)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def depths(self) -> list[float]:
        return GT.generate_depths(read_float(self.start, 100.0), read_float(self.stop, 1e4),
                                  self.n.value(), self.spacing.currentData(),
                                  read_float(self.round_to, 0.0))


class DepthListEdit(QWidget):
    """Comma-separated depth list (m) + a generator dialog."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setToolTip("Depths in metres, comma separated. Negative = above sea level.")
        self.edit.editingFinished.connect(self.changed)
        gen = QPushButton("Generate...")
        gen.clicked.connect(self._generate)
        lay.addWidget(self.edit, 1)
        lay.addWidget(gen)

    def depths(self) -> list[float]:
        try:
            return GT.parse_depths(self.edit.text())
        except ValueError:
            return []

    def set_depths(self, depths) -> None:
        self.edit.setText(GT.format_depths(depths))

    def _generate(self):
        dlg = DepthGenerateDialog(self)
        if not dlg.exec_():
            return
        try:
            new = dlg.depths()
        except ValueError as exc:
            QMessageBox.warning(self, "Generate depths", str(exc))
            return
        if dlg.mode.currentIndex() == 1:
            new = sorted(set(self.depths()) | set(new))
        self.set_depths(new)
        self.changed.emit()


# --------------------------------------------------------------------------- #
class ColorBox(QGroupBox):
    """Edit a :class:`~..project.ColorInfo`."""

    changed = pyqtSignal()

    def __init__(self, title: str = "Colour scale", parent=None):
        super().__init__(title, parent)
        form = QFormLayout(self)
        self.cmap = QComboBox()
        self.cmap.setEditable(True)
        self.cmap.addItems(render.COLORMAPS)
        self.cmap.setToolTip("Any matplotlib colour map name, or cmc.<name> for Crameri maps.\n"
                             "Append _r to reverse.")
        self.n_colors = QSpinBox(minimum=0, maximum=256)
        self.n_colors.setToolTip("Number of discrete colours (0 = continuous)")
        self.rho_min = float_edit(1.0)
        self.rho_max = float_edit(1e4)
        row = QHBoxLayout()
        row.addWidget(self.rho_min)
        row.addWidget(QLabel("to"))
        row.addWidget(self.rho_max)
        row.addWidget(QLabel("Ω·m"))
        row.addStretch(1)
        self.arrows = QCheckBox("pointed colour-bar ends (values beyond the range)")
        self.arrows.toggled.connect(lambda _on: self.changed.emit())
        form.addRow("colour map", self.cmap)
        form.addRow("discrete colours", self.n_colors)
        form.addRow("resistivity", row)
        form.addRow(self.arrows)
        self.cmap.currentTextChanged.connect(lambda _t: self.changed.emit())
        self.n_colors.valueChanged.connect(lambda _v: self.changed.emit())
        self.rho_min.editingFinished.connect(self.changed)
        self.rho_max.editingFinished.connect(self.changed)

    def load(self, c) -> None:
        self.blockSignals(True)
        self.cmap.setCurrentText(c.cmap)
        self.n_colors.setValue(int(c.n_colors))
        self.rho_min.setText(f"{c.rho_min:g}")
        self.rho_max.setText(f"{c.rho_max:g}")
        self.arrows.setChecked(bool(getattr(c, "extend_arrows", False)))
        self.blockSignals(False)

    def store(self, c) -> None:
        c.cmap = self.cmap.currentText().strip() or "turbo_r"
        c.n_colors = self.n_colors.value()
        c.extend_arrows = self.arrows.isChecked()
        lo, hi = read_float(self.rho_min, c.rho_min), read_float(self.rho_max, c.rho_max)
        if 0 < lo < hi:
            c.rho_min, c.rho_max = lo, hi


# --------------------------------------------------------------------------- #
class FigurePreview(QWidget):
    """A matplotlib canvas that shows a figure at its *export* size.

    The figure is kept at ``width_in x height_in`` inches and the canvas is
    scaled to fit by changing the figure dpi, so fonts, line widths and
    layout look exactly like the saved file (while zoom / pan still work).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self.width_in, self.height_in = 7.0, 6.0
        self.figure = Figure(figsize=(self.width_in, self.height_in))
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self._holder = QWidget()
        self._holder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hl = QGridLayout(self._holder)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(self.canvas, 0, 0, Qt.AlignCenter)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self._holder, 1)
        self._holder.resizeEvent = self._on_holder_resize  # type: ignore[assignment]

    def set_size_inches(self, w: float, h: float) -> None:
        self.width_in, self.height_in = max(float(w), 0.5), max(float(h), 0.5)
        self._fit()

    def _on_holder_resize(self, event):
        QWidget.resizeEvent(self._holder, event)
        self._fit()

    def _fit(self):
        avail_w = max(self._holder.width() - 4, 50)
        avail_h = max(self._holder.height() - 4, 50)
        scale = min(avail_w / self.width_in, avail_h / self.height_in)   # logical px per inch
        dpr = getattr(self.canvas, "device_pixel_ratio", 1.0) or 1.0
        self.figure.set_dpi(scale * dpr)
        self.canvas.setFixedSize(int(self.width_in * scale), int(self.height_in * scale))
        self.figure.set_size_inches(self.width_in, self.height_in, forward=False)
        self.canvas.draw_idle()

    def draw(self) -> None:
        self.figure.set_size_inches(self.width_in, self.height_in, forward=False)
        self.canvas.draw_idle()

    def current_limits(self):
        axes = self.figure.get_axes()
        if not axes:
            return None
        ax = axes[0]
        return (*ax.get_xlim(), *ax.get_ylim())


# --------------------------------------------------------------------------- #
class Debounce(QObject):
    """Call *fn* once, *ms* after the last :meth:`poke`."""

    def __init__(self, fn: Callable, ms: int = 250, parent=None):
        super().__init__(parent)
        self._t = QTimer(self)
        self._t.setSingleShot(True)
        self._t.setInterval(ms)
        self._t.timeout.connect(fn)

    def poke(self, *_args):
        self._t.start()


class _ProgressWorker(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(object)
    progress = pyqtSignal(int, int, str)

    def __init__(self, fn: Callable):
        super().__init__()
        self._fn = fn

    def run(self):
        try:
            self.done.emit(self._fn(lambda i, n, msg: self.progress.emit(int(i), int(n), str(msg))))
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI
            self.failed.emit(exc)


def run_async_progress(owner: QObject, fn: Callable, on_done: Callable,
                       on_error: Callable, on_progress: Callable) -> QThread:
    """Like :func:`mtinv_gui.gui.worker.run_async`, but ``fn(progress)`` gets a
    ``progress(i, n, message)`` callback whose calls reach *on_progress* on the
    GUI thread."""
    thread = QThread(owner)
    worker = _ProgressWorker(fn)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)

    def _cleanup():
        thread.quit()
        thread.wait()
        worker.deleteLater()

    worker.progress.connect(on_progress)
    worker.done.connect(lambda r: (_cleanup(), on_done(r)))
    worker.failed.connect(lambda e: (_cleanup(), on_error(e)))
    if not hasattr(owner, "_bg_threads"):
        owner._bg_threads = []          # type: ignore[attr-defined]
    owner._bg_threads.append((thread, worker))
    thread.start()
    return thread
