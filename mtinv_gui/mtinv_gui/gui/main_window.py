"""Top-level window: menu, workflow tabs, shared log dock, env status bar."""
from __future__ import annotations

import datetime as _dt
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction, QDockWidget, QFileDialog, QLabel, QMainWindow, QMessageBox,
    QPlainTextEdit, QTabWidget,
)

from .. import config
from ..project import Project
from .data_panel import DataPanel
from .gofem_panel import GofemPanel
from .interp_panel import InterpPanel
from .mesh_panel import MeshPanel
from .projection_panel import ProjectionPanel
from .refine_panel import RefinePanel
from .survey_panel import SurveyPanel
from .topo_panel import TopoPanel


class MainWindow(QMainWindow):
    def __init__(self, restore_last: bool = True):
        super().__init__()
        self.project = Project(workdir=os.getcwd())
        self._saved_path: str | None = None
        self.resize(1180, 820)

        self._build_log_dock()
        self._build_tabs()
        self._build_menu()
        self._build_statusbar()
        self._update_title()
        self.log(f"{config.APP_NAME} {__import__('mtinv_gui').__version__} - new project")
        if restore_last:
            self._maybe_restore()

    # ------------------------------------------------------------------ #
    def _update_title(self):
        name = os.path.basename(self._saved_path) if self._saved_path else "untitled"
        self.setWindowTitle(f"{config.APP_NAME} - {name}")

    def _remember(self, path: str):
        self._saved_path = path
        config.save_settings({"last_project": path})
        self._update_title()

    def _maybe_restore(self):
        s = config.load_settings()
        path = s.get("last_project") or ""
        if not (s.get("restore_last_project") and path and os.path.isfile(path)):
            return
        try:
            self.project = Project.load(path)
        except Exception as exc:  # noqa: BLE001
            self.log(f"could not restore last session ({path}): {exc}")
            return
        self._saved_path = path
        self._refresh_all()
        self._update_title()
        self.log(f"restored last session from {path}")

    # ------------------------------------------------------------------ #
    def _build_tabs(self):
        self.tabs = QTabWidget()
        self.panels = []
        for cls in (InterpPanel, SurveyPanel, TopoPanel, ProjectionPanel, DataPanel,
                    MeshPanel, GofemPanel, RefinePanel):
            panel = cls(self.project)
            panel.status_message.connect(self.log)
            panel.project_changed.connect(self._on_project_changed)
            self.tabs.addTab(panel, panel.title)
            self.panels.append(panel)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.setCentralWidget(self.tabs)

    def _on_tab_changed(self, index: int):
        """Let the panel being shown recover any transient UI state (e.g. the
        Mesh tab's Preview / Build buttons if a build died without reporting)."""
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
        self.restore_act = QAction("Reopen last project on startup", self, checkable=True)
        self.restore_act.setChecked(bool(config.load_settings().get("restore_last_project")))
        self.restore_act.toggled.connect(
            lambda on: config.save_settings({"restore_last_project": bool(on)}))
        m.addAction(self.restore_act)
        m.addSeparator()
        set_act = QAction("Settings...", self)
        set_act.triggered.connect(self._settings)
        m.addAction(set_act)
        env_act = QAction("Environment report...", self)
        env_act.triggered.connect(self._env_report)
        m.addAction(env_act)

        ver = QLabel(f"  v{__import__('mtinv_gui').__version__}  ")
        ver.setStyleSheet("color: gray;")
        self.menuBar().setCornerWidget(ver, Qt.TopRightCorner)

    def _build_statusbar(self):
        env = config.probe_environment()
        core = [k for k in ("numpy", "scipy", "matplotlib", "pyproj") if env.get(k)]
        heavy_missing = [k for k in ("mtpy", "PyDealII", "gofem") if not env.get(k)]
        msg = f"core: {', '.join(core) or 'none'}"
        if heavy_missing:
            msg += f"   |   missing for full run: {', '.join(heavy_missing)}"
        if config.has_wsl():
            msg += "   |   wsl: available"
        self.statusBar().addWidget(QLabel(msg))

    # ------------------------------------------------------------------ #
    def log(self, msg: str):
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        self.log_view.appendPlainText(f"[{ts}] {msg}")

    def _on_project_changed(self):
        for p in self.panels:
            if p is not self.sender():
                p.refresh()

    def _refresh_all(self):
        for p in self.panels:
            p.project = self.project
            p.refresh()

    # ---- project io -------------------------------------------------- #
    def new_project(self):
        self.project = Project(workdir=os.getcwd())
        self._saved_path = None
        self._refresh_all()
        self._update_title()
        self.log("new project")

    def open_project(self, f=None):
        if not f:  # (the menu action passes False)
            start = os.path.dirname(self._saved_path) if self._saved_path else os.getcwd()
            f, _ = QFileDialog.getOpenFileName(
                self, "Open project", start,
                f"project (*{config.PROJECT_FILE_SUFFIX});;all files (*)")
        if not f:
            return
        try:
            self.project = Project.load(f)
        except Exception as exc:  # pragma: no cover - user IO
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self._refresh_all()
        self._remember(f)
        self.log(f"opened {f}")

    def save_project(self):
        if not self._saved_path:
            return self.save_project_as()
        path = self.project.save(self._saved_path)
        self._remember(path)
        self.log(f"saved {path}")

    def save_project_as(self):
        start = self._saved_path or os.path.join(
            os.getcwd(), "project" + config.PROJECT_FILE_SUFFIX)
        f, _ = QFileDialog.getSaveFileName(
            self, "Save project as", start,
            f"project (*{config.PROJECT_FILE_SUFFIX})")
        if not f:
            return
        if not f.endswith(config.PROJECT_FILE_SUFFIX):
            f += config.PROJECT_FILE_SUFFIX
        self.project.workdir = os.path.dirname(os.path.abspath(f))
        self.project.name = os.path.basename(f).replace(config.PROJECT_FILE_SUFFIX, "")
        path = self.project.save(f)
        self._remember(path)
        self.log(f"saved {path}")

    # ---- keep the session file current on exit ---------------------- #
    def closeEvent(self, event):
        if self._saved_path:
            try:
                self.project.save(self._saved_path)
                self.log(f"session saved to {self._saved_path}")
            except Exception as exc:  # noqa: BLE001
                self.log(f"could not save session on exit: {exc}")
        super().closeEvent(event)

    def _settings(self):
        from .settings_dialog import SettingsDialog

        if SettingsDialog(self).exec_():
            self.log(f"settings saved to {config.SETTINGS_PATH}")

    def _env_report(self):
        env = config.probe_environment()
        lines = [f"{'OK  ' if v else 'MISS'}  {k:12s} {v or ''}" for k, v in env.items()]
        lines.append("")
        lines.append(f"wsl available: {config.has_wsl()}")
        QMessageBox.information(self, "Environment", "\n".join(lines))
