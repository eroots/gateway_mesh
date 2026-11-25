"""Backend modules.

Each backend maps onto one step of the workflow and exposes plain
functions that take a :class:`mtinv_gui.project.Project` (plus explicit
arguments) and return results / write files.  The GUI layer never imports the
heavy scientific stack directly -- it goes through here, so the backends can be
swapped for subprocess / WSL implementations without touching the widgets.
"""
