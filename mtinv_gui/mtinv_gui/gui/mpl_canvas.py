"""Small Matplotlib-in-Qt canvas helper (optional dependency)."""
from __future__ import annotations

try:
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as _Canvas
    from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as _Toolbar
    from matplotlib.figure import Figure
    HAVE_MPL = True
except Exception:  # pragma: no cover - environment guard
    HAVE_MPL = False
    _Canvas = object  # type: ignore


if HAVE_MPL:
    from PyQt5.QtWidgets import QVBoxLayout, QWidget

    class MplCanvas(QWidget):
        """A figure + navigation toolbar in a single widget."""

        def __init__(self, parent=None, figsize=(5, 4)):
            super().__init__(parent)
            self.figure = Figure(figsize=figsize, tight_layout=True)
            self.canvas = _Canvas(self.figure)
            self.toolbar = _Toolbar(self.canvas, self)
            lay = QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.toolbar)
            lay.addWidget(self.canvas)
            self.ax = self.figure.add_subplot(111)

        def clear(self):
            self.figure.clear()
            self.ax = self.figure.add_subplot(111)

        def redraw(self):
            self.canvas.draw_idle()
else:
    from PyQt5.QtWidgets import QLabel

    class MplCanvas(QLabel):  # type: ignore[no-redef]
        """Placeholder shown when matplotlib is unavailable."""

        def __init__(self, parent=None, figsize=(5, 4)):
            super().__init__("matplotlib not installed -- plot preview disabled", parent)
            self.ax = None

        def clear(self):
            pass

        def redraw(self):
            pass
