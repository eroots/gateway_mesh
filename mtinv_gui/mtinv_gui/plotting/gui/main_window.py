"""Top-level window of the model-plotting app: tabs, log dock, project menu."""
from __future__ import annotations

import datetime as _dt
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction, QDockWidget, QFileDialog, QMainWindow, QMessageBox, QPlainTextEdit,
    QTabWidget,
)

from ... import config
from ..backends.session import PlotSession
from ..project import PLOT_PROJECT_SUFFIX, PlotProject
from .figures_panel import FiguresPanel
from .geotiff_panel import GeotiffPanel
from .layers_panel import LayersPanel
from .model_panel import ModelPanel
from .plan_panel import PlanPanel
from .section_panel import SectionPanel

APP_NAME = "MT Model Plotting"


class PlotMainWindow(QMainWindow):
    def __init__(self, restore_last: bool = True):
        super().__init__()
        self.project = PlotProject(workdir=os.getcwd())
        self.session = PlotSession()
        self.session.house = self.project.house
        self._saved_path: str | None = None
        self.resize(1400, 900)

        self._build_log_dock()
        self._build_tabs()
        self._build_menu()
        self._update_title()
        self.log(f"{APP_NAME} - new project")
        if restore_last:
            self._maybe_restore()

    # ------------------------------------------------------------------ #
    def _update_title(self):
        name = os.path.basename(self._saved_path) if self._saved_path else "untitled"
        self.setWindowTitle(f"{APP_NAME} - {name}")

    def _remember(self, path: str):
        self._saved_path = path
        config.save_settings({"last_plot_project": path})
        self._update_title()

    def _maybe_restore(self):
        s = config.load_settings()
        path = s.get("last_plot_project") or ""
        if not (s.get("restore_last_project") and path and os.path.isfile(path)):
            return
        try:
            self._set_project(PlotProject.load(path))
        except Exception as exc:  # noqa: BLE001
            self.log(f"could not restore last session ({path}): {exc}")
            return
        self._saved_path = path
        self._update_title()
        self.log(f"restored last session from {path}")

    # ------------------------------------------------------------------ #
    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.panels = []
        for cls in (ModelPanel, LayersPanel, PlanPanel, SectionPanel, GeotiffPanel, FiguresPanel):
            panel = cls(self.project, self.session)
            panel.status_message.connect(self.log)
            panel.project_changed.connect(self._on_project_changed)
            self.tabs.addTab(panel, panel.title)
            self.panels.append(panel)
        self.model_panel = self.panels[0]
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

    def _on_tab_changed(self, index: int):
        if 0 <= index < len(self.panels):
            self.panels[index].on_shown()

    def _build_log_dock(self):
        self.log_view = QPlainTextEdit(readOnly=True)
        self.log_view.setMaximumBlockCount(5000)
        dock = QDockWidget("Log", self)
        dock.setWidget(self.log_view)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    def _build_menu(self):
        m = self.menuBar().addMenu("&Project")
        for text, slot, keys in (
            ("&New", self.new_project, "Ctrl+N"),
            ("&Open...", self.open_project, "Ctrl+O"),
            ("&Save", self.save_project, "Ctrl+S"),
            ("Save &As...", self.save_project_as, "Ctrl+Shift+S"),
        ):
            act = QAction(text, self)
            act.setShortcut(keys)
            act.triggered.connect(slot)
            m.addAction(act)
        m.addSeparator()
        restore = QAction("Reopen last project on startup", self, checkable=True)
        restore.setChecked(bool(config.load_settings().get("restore_last_project")))
        restore.toggled.connect(
            lambda on: config.save_settings({"restore_last_project": bool(on)}))
        m.addAction(restore)

    # ------------------------------------------------------------------ #
    def log(self, msg: str):
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{ts}] {msg}")

    def _on_project_changed(self):
        for p in self.panels:
            if p is not self.sender():
                p.refresh()

    def _set_project(self, project: PlotProject):
        """Swap in *project*, drop the old model, reload the new one if it has one."""
        self.project = project
        self.session = PlotSession()
        self.session.house = project.house
        for p in self.panels:
            p.project = project
            p.session = self.session
            p.refresh()
        if project.model.model_path and os.path.isfile(project.model.model_path):
            self.model_panel.load_model()

    # ---- project io -------------------------------------------------- #
    def new_project(self):
        self._saved_path = None
        self._set_project(PlotProject(workdir=os.getcwd()))
        self._update_title()
        self.log("new project")

    def open_project(self, f=None):
        if not f:  # (the menu action passes False)
            start = os.path.dirname(self._saved_path) if self._saved_path else os.getcwd()
            f, _ = QFileDialog.getOpenFileName(
                self, "Open plotting project", start,
                f"plotting project (*{PLOT_PROJECT_SUFFIX});;all files (*)")
        if not f:
            return
        try:
            proj = PlotProject.load(f)
        except Exception as exc:  # pragma: no cover - user IO
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self._set_project(proj)
        self._remember(f)
        self.log(f"opened {f}")

    def _sync_widgets(self):
        """Pull unsaved edits from every panel into the project."""
        for p in self.panels:
            store = getattr(p, "_store", None)
            if store:
                store()

    def save_project(self):
        if not self._saved_path:
            return self.save_project_as()
        self._sync_widgets()
        path = self.project.save(self._saved_path)
        self._remember(path)
        self.log(f"saved {path}")

    def save_project_as(self):
        start = self._saved_path or os.path.join(os.getcwd(), "plots" + PLOT_PROJECT_SUFFIX)
        f, _ = QFileDialog.getSaveFileName(
            self, "Save plotting project as", start,
            f"plotting project (*{PLOT_PROJECT_SUFFIX})")
        if not f:
            return
        if not f.endswith(PLOT_PROJECT_SUFFIX):
            f += PLOT_PROJECT_SUFFIX
        self._sync_widgets()
        self.project.workdir = os.path.dirname(os.path.abspath(f))
        self.project.name = os.path.basename(f).replace(PLOT_PROJECT_SUFFIX, "")
        path = self.project.save(f)
        self._remember(path)
        self.log(f"saved {path}")

    def closeEvent(self, event):
        if self._saved_path:
            try:
                self._sync_widgets()
                self.project.save(self._saved_path)
            except Exception as exc:  # noqa: BLE001
                self.log(f"could not save session on exit: {exc}")
        super().closeEvent(event)
