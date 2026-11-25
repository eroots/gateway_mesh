"""QApplication bootstrap."""
from __future__ import annotations

import sys

from .. import config


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    try:
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover - environment guard
        sys.stderr.write(
            "PyQt5 is required to run the GUI.\n"
            f"  import error: {exc}\n"
            "  conda activate mtinv\n")
        return 1

    from .main_window import MainWindow

    # The 3-D views need a native window, and by default Qt then makes all their siblings
    # native too: every tab page and many controls.  Those separate windows left stale
    # pieces of panels on screen when flipping tabs under WSLg.  Their parents are handled
    # in mesh_viewer._Interactor.
    QApplication.setAttribute(Qt.AA_DontCreateNativeWidgetSiblings)
    app = QApplication(argv)
    app.setApplicationName(config.APP_NAME)
    app.setOrganizationName(config.ORG_NAME)

    args = argv[1:]  # a project file, or --new for a blank project
    win = MainWindow(restore_last=not args)
    if args and args[0] != "--new":
        win.open_project(args[0])
    win.show()
    return app.exec_()
