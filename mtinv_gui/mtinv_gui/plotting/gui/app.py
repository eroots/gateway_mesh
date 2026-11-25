"""QApplication bootstrap for the model-plotting app."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    try:
        from PyQt5.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover - environment guard
        sys.stderr.write(
            "PyQt5 is required to run the plotting GUI.\n"
            f"  import error: {exc}\n"
            "  conda activate mtinv\n")
        return 1

    from .main_window import APP_NAME, PlotMainWindow

    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("mtinv_gui")

    args = argv[1:]  # a project file, or --new for a blank project
    win = PlotMainWindow(restore_last=not args)
    if args and args[0] != "--new":
        win.open_project(args[0])
    win.show()
    return app.exec_()
