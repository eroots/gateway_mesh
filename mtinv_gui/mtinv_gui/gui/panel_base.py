"""Shared base class for the workflow tabs."""
from __future__ import annotations

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget

from ..project import Project


class BasePanel(QWidget):
    """One workflow step.

    Subclasses build their UI in ``build()`` and pull fresh state from
    ``self.project`` in ``refresh()``.  Emit :attr:`project_changed` after
    mutating the project so sibling panels can update.
    """

    project_changed = pyqtSignal()
    status_message = pyqtSignal(str)

    #: tab label
    title = "Step"

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.build()
        self.refresh()

    # -- hooks ------------------------------------------------------------- #
    def build(self) -> None:  # pragma: no cover - UI
        """Create widgets/layout.  Override."""

    def refresh(self) -> None:  # pragma: no cover - UI
        """Re-read ``self.project`` into the widgets.  Override."""

    def on_shown(self) -> None:  # pragma: no cover - UI
        """Called when this panel's tab becomes the current one.  Optional --
        for cheap re-sync / transient-state recovery that shouldn't run on the
        full refresh() cascade.  Default: no-op."""

    # -- helpers --------------------------------------------------------- #
    def log(self, msg: str) -> None:
        self.status_message.emit(msg)

    def notify_changed(self) -> None:
        self.project_changed.emit()
