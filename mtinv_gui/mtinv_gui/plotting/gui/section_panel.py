"""Tab 3 -- vertical sections: N-S / E-W lines, drawn paths, station lines.

Sub-tabs along the top split the sections into N-S, E-W, paths + stations, and
All (every section together); each shows only its own sections, in the list,
the index map, 'Export all' and 'Save to file'.  The look is shared.
Left: the section list, the selected section's geometry, and the look shared
by every section.  Right: an index map (click to draw a path or pick stations)
above the section preview.  Cutting and interpolating run off the GUI thread;
both are cached, so flipping between true cells and an interpolation is
instant after the first time.
"""
from __future__ import annotations

import copy
import os

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QShortcut,
    QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMessageBox, QProgressBar, QPushButton, QSpinBox, QSplitter,
    QStackedWidget, QTabBar, QVBoxLayout, QWidget,
)

from ...gui.worker import run_async
from ..backends import render
from ..backends import sections as SEC
from ..project import SectionInfo
from .widgets import (
    ColorBox, Debounce, FigurePreview, PathRow, PlotPanel, float_edit,
    read_float, run_async_progress, scroll_column,
)

_MARKERS = ["v", "^", "o", "s", "D", "."]
_RBF_KERNELS = ["thin_plate_spline", "linear", "cubic", "quintic", "multiquadric",
                "inverse_multiquadric", "gaussian"]
_MAP_BG = [("surface", "model surface"), ("plan", "plan view (tab 3 depth + colours)"),
           ("none", "none")]


class LineSeriesDialog(QDialog):
    """New N-S or E-W line(s): positions in km, one section per value."""

    def __init__(self, kind: str, default_km: float, parent=None):
        super().__init__(parent)
        coord = "easting" if kind == "ns" else "northing"
        along = "northing" if kind == "ns" else "easting"
        self.setWindowTitle(f"New {'N-S' if kind == 'ns' else 'E-W'} section(s)")
        form = QFormLayout(self)
        self.values = QLineEdit(f"{default_km:g}")
        self.values.setToolTip("One section per value, comma separated, e.g. 260, 280, 300.\n"
                               "Filled from start / step / count as you change them; or type a list.")
        self.start = float_edit(default_km)
        self.step = float_edit(20.0)
        self.count = QSpinBox(minimum=1, maximum=100, value=1)
        grow = QHBoxLayout()
        for lab, w in (("start", self.start), ("step", self.step), ("count", self.count)):
            grow.addWidget(QLabel(lab))
            grow.addWidget(w)
        self.start.textEdited.connect(self._fill)
        self.step.textEdited.connect(self._fill)
        self.count.valueChanged.connect(self._fill)
        self.n_lbl = QLabel("")
        self.n_lbl.setStyleSheet("color: gray;")
        self.values.textChanged.connect(self._count_label)
        self.span0 = QLineEdit()
        self.span1 = QLineEdit()
        for w in (self.span0, self.span1):
            w.setPlaceholderText("auto")
            w.setMaximumWidth(90)
        srow = QHBoxLayout()
        srow.addWidget(self.span0)
        srow.addWidget(QLabel("to"))
        srow.addWidget(self.span1)
        srow.addStretch(1)
        form.addRow("series", grow)
        form.addRow(f"{coord} (km)", self.values)
        form.addRow("", self.n_lbl)
        form.addRow(f"{along} range (km)", srow)
        form.addRow(QLabel("Blank range = span the stations plus the map padding."))
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)
        self._count_label()

    def _fill(self, *_):
        a, d = read_float(self.start, 0.0), read_float(self.step, 1.0)
        self.values.setText(", ".join(f"{a + i * d:g}" for i in range(self.count.value())))

    def _count_label(self, *_):
        try:
            n = len(self.coords_m())
            self.n_lbl.setText(f"→ {n} section{'s' if n != 1 else ''} (one per value)")
        except ValueError:
            self.n_lbl.setText("positions must be numbers (km)")

    def coords_m(self) -> list[float]:
        out = []
        for tok in self.values.text().replace(";", ",").split(","):
            tok = tok.strip()
            if tok:
                out.append(float(tok) * 1000.0)
        return out

    def span_m(self) -> tuple[float, float]:
        try:
            return float(self.span0.text()) * 1000.0, float(self.span1.text()) * 1000.0
        except ValueError:
            return 0.0, 0.0


class SectionPanel(PlotPanel):
    title = "4 - Sections"

    def build(self) -> None:
        self._loading = False
        self._current = -1
        self._mode = None                  # None | "draw" | "pick"
        self._target = None                # section index being redrawn (None = new one)
        self._draft: list = []             # vertices / station names being collected
        self._draft_artist = None
        self._map_ax = None
        self._gen = 0
        self._group = "all"                # sub-tab: key of SEC.SECTION_GROUPS
        self._rows: list[int] = []         # list row -> index in project.sections
        self._group_current: dict = {}     # sub-tab -> label of the section last selected there
        self._redraw = Debounce(self._draw_section, 250, self)
        self._remap = Debounce(self._draw_map, 250, self)

        outer = QVBoxLayout(self)
        self.group_bar = QTabBar()
        self.group_bar.setExpanding(False)
        self.group_bar.setDrawBase(True)
        for key, label in SEC.SECTION_GROUPS.items():
            self.group_bar.addTab(label)
            self.group_bar.setTabData(self.group_bar.count() - 1, key)
        self.group_bar.setToolTip("N-S, E-W and other sections each have their own list and index "
                                  "map; All shows every section together")
        self.group_bar.currentChanged.connect(self._group_changed)
        outer.addWidget(self.group_bar)
        root = QHBoxLayout()
        outer.addLayout(root, 1)
        col = QWidget()
        left = QVBoxLayout(col)

        # ---- section list ------------------------------------------------ #
        box = QGroupBox("Sections")
        v = QVBoxLayout(box)
        self.list = QListWidget()
        self.list.setMaximumHeight(140)
        self.list.currentRowChanged.connect(self._select)
        v.addWidget(self.list)
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        self.add_btns = {}
        for key, text, slot, row in (("ns", "N-S...", lambda: self._add_lines("ns"), row1),
                                     ("ew", "E-W...", lambda: self._add_lines("ew"), row1),
                                     ("other", "Draw path", self._add_path, row1),
                                     ("other", "Stations...", self._add_stations, row1),
                                     ("", "From file...", self._add_from_file, row2),
                                     ("", "Save to file...", self._save_to_file, row2),
                                     ("", "Duplicate", self._duplicate, row2),
                                     ("", "Delete", self._delete, row2)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
            if key:
                self.add_btns.setdefault(key, []).append(b)
            if text == "From file...":
                b.setToolTip("A saved section set (.json, from 'Save to file'), E/N or lon/lat "
                             "columns (a path), or a station list (.lst)")
            elif text == "Save to file...":
                b.setToolTip("Save the sections on this sub-tab (their full definitions) to a "
                             ".json file that 'From file...' reads back, here or in another project")
        v.addLayout(row1)
        v.addLayout(row2)
        left.addWidget(box)

        # ---- selected section ------------------------------------------- #
        self.sec_box = QGroupBox("Selected section")
        form = QFormLayout(self.sec_box)
        self.label = QLineEdit()
        self.label.setMaximumWidth(80)
        self.label.editingFinished.connect(self._sec_changed)
        self.sec_info = QLabel("")
        self.sec_info.setWordWrap(True)
        self.sec_info.setStyleSheet("color: gray;")
        form.addRow("label", self.label)
        form.addRow(self.sec_info)
        self.kind_pages = QStackedWidget()
        # ns / ew
        pg = QWidget()
        f = QFormLayout(pg)
        self.coord = QDoubleSpinBox(decimals=2, minimum=-1e5, maximum=1e5, suffix=" km")
        self.span0 = QDoubleSpinBox(decimals=2, minimum=-1e5, maximum=1e5, suffix=" km")
        self.span1 = QDoubleSpinBox(decimals=2, minimum=-1e5, maximum=1e5, suffix=" km")
        for w in (self.coord, self.span0, self.span1):
            w.editingFinished.connect(self._sec_changed)
        srow = QHBoxLayout()
        srow.addWidget(self.span0)
        srow.addWidget(QLabel("to"))
        srow.addWidget(self.span1)
        self.coord_lbl = QLabel("easting")
        self.span_lbl = QLabel("range")
        f.addRow(self.coord_lbl, self.coord)
        f.addRow(self.span_lbl, srow)
        f.addRow(QLabel("Equal range ends = span the stations + map padding."))
        self.kind_pages.addWidget(pg)
        # path
        pg = QWidget()
        f = QFormLayout(pg)
        self.path_info = QLabel("")
        redraw = QPushButton("Redraw on map")
        redraw.clicked.connect(self._redraw_path)
        f.addRow(self.path_info)
        f.addRow(redraw)
        self.kind_pages.addWidget(pg)
        # stations
        pg = QWidget()
        f = QFormLayout(pg)
        self.st_list = QLineEdit()
        self.st_list.setToolTip("Station names in order, comma separated (first = A end)")
        self.st_list.editingFinished.connect(self._sec_changed)
        self.st_path = QComboBox()
        for key, label in SEC.STATION_PATHS.items():
            self.st_path.addItem(label, key)
        self.st_path.currentIndexChanged.connect(self._sec_changed)
        self.extend = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=1000.0, suffix=" km")
        self.extend.editingFinished.connect(self._sec_changed)
        pick = QPushButton("Pick stations on map")
        pick.clicked.connect(self._repick_stations)
        f.addRow("stations", self.st_list)
        f.addRow("path", self.st_path)
        f.addRow("extend ends", self.extend)
        f.addRow(pick)
        self.kind_pages.addWidget(pg)
        form.addRow(self.kind_pages)
        self.reverse = QCheckBox("reverse direction (swap A / A′)")
        self.reverse.toggled.connect(self._sec_changed)
        self.x_axis = QComboBox()
        self.x_axis.currentIndexChanged.connect(self._sec_changed)
        self.ve = QDoubleSpinBox(decimals=2, minimum=0.05, maximum=100.0, singleStep=0.5)
        self.ve.editingFinished.connect(self._sec_changed)
        form.addRow(self.reverse)
        form.addRow("x axis", self.x_axis)
        form.addRow("vertical exag.", self.ve)
        left.addWidget(self.sec_box)

        # ---- vertical ---------------------------------------------------- #
        box = QGroupBox("Vertical extent (all sections)")
        form = QFormLayout(box)
        self.auto_top = QCheckBox("top = highest surface on the section")
        self.auto_top.toggled.connect(self._style_changed)
        self.z_top = QDoubleSpinBox(decimals=2, minimum=-100.0, maximum=20.0, suffix=" km")
        self.z_bottom = QDoubleSpinBox(decimals=1, minimum=0.1, maximum=3000.0, suffix=" km")
        for w in (self.z_top, self.z_bottom):
            w.editingFinished.connect(self._style_changed)
        self.vaxis = QComboBox()
        self.vaxis.addItem("depth (km below sea level)", "depth")
        self.vaxis.addItem("elevation (km)", "elevation")
        self.vaxis.currentIndexChanged.connect(self._style_changed)
        form.addRow(self.auto_top)
        form.addRow("top elevation", self.z_top)
        form.addRow("bottom depth", self.z_bottom)
        form.addRow("axis", self.vaxis)
        left.addWidget(box)

        # ---- rendering --------------------------------------------------- #
        box = QGroupBox("Rendering")
        form = QFormLayout(box)
        self.show_interp = QCheckBox("show interpolated  (off = true cells)")
        self.show_interp.toggled.connect(self._style_changed)
        self.method = QComboBox()
        for key, label in SEC.INTERP_METHODS.items():
            self.method.addItem(label, key)
        self.method.currentIndexChanged.connect(self._method_changed)
        self.nx = QSpinBox(minimum=20, maximum=5000, singleStep=50)
        self.nz = QSpinBox(minimum=20, maximum=5000, singleStep=50)
        for w in (self.nx, self.nz):
            w.editingFinished.connect(self._style_changed)
        grow = QHBoxLayout()
        grow.addWidget(self.nx)
        grow.addWidget(QLabel("×"))
        grow.addWidget(self.nz)
        self.rbf_box = QWidget()
        rf = QFormLayout(self.rbf_box)
        rf.setContentsMargins(0, 0, 0, 0)
        self.rbf_kernel = QComboBox()
        self.rbf_kernel.addItems(_RBF_KERNELS)
        self.rbf_kernel.currentIndexChanged.connect(self._style_changed)
        self.rbf_smooth = float_edit(0.0)
        self.rbf_eps = float_edit(1.0)
        self.rbf_eps.setToolTip("Shape parameter (in grid-node units) for multiquadric / "
                                "gaussian kernels; ignored by thin_plate_spline")
        self.rbf_nb = QSpinBox(minimum=4, maximum=2000, value=64)
        for w in (self.rbf_smooth, self.rbf_eps):
            w.editingFinished.connect(self._style_changed)
        self.rbf_nb.editingFinished.connect(self._style_changed)
        rf.addRow("kernel", self.rbf_kernel)
        rf.addRow("smoothing", self.rbf_smooth)
        rf.addRow("epsilon", self.rbf_eps)
        rf.addRow("neighbours", self.rbf_nb)
        self.gauss = QDoubleSpinBox(decimals=1, minimum=0.1, maximum=100.0, suffix=" nodes")
        self.gauss.editingFinished.connect(self._style_changed)
        self.render_status = QLabel("")
        self.render_status.setStyleSheet("color: gray;")
        form.addRow(self.show_interp)
        form.addRow("method", self.method)
        form.addRow("grid nodes", grow)
        form.addRow(self.rbf_box)
        form.addRow("Gaussian σ", self.gauss)
        form.addRow(self.render_status)
        left.addWidget(box)

        # ---- colour ------------------------------------------------------ #
        self.color = ColorBox()
        self.color.changed.connect(self._style_changed)
        self.cbar = QComboBox()
        for key in ("right", "bottom", "none"):
            self.cbar.addItem(key, key)
        self.cbar.currentIndexChanged.connect(self._style_changed)
        self.color.layout().addRow("colour bar", self.cbar)
        left.addWidget(self.color)

        # ---- annotations ------------------------------------------------- #
        box = QGroupBox("Stations + annotations")
        form = QFormLayout(box)
        self.st_show = QCheckBox("stations")
        self.st_labels = QCheckBox("names")
        self.surf = QCheckBox("surface line")
        self.ends = QCheckBox("A / A′")
        crow = QHBoxLayout()
        for w in (self.st_show, self.st_labels, self.surf, self.ends):
            w.toggled.connect(self._style_changed)
            crow.addWidget(w)
        self.st_marker = QComboBox()
        self.st_marker.addItems(_MARKERS)
        self.st_size = QDoubleSpinBox(decimals=1, minimum=1.0, maximum=400.0)
        self.st_color = QComboBox()
        self.st_color.setEditable(True)
        self.st_color.addItems(["white", "black", "magenta", "0.5"])
        self.st_marker.currentIndexChanged.connect(self._style_changed)
        self.st_size.editingFinished.connect(self._style_changed)
        self.st_color.currentTextChanged.connect(self._style_changed)
        mrow = QHBoxLayout()
        mrow.addWidget(self.st_marker)
        mrow.addWidget(QLabel("size"))
        mrow.addWidget(self.st_size)
        mrow.addWidget(self.st_color, 1)
        self.st_offset = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=1000.0, suffix=" km")
        self.st_offset.setToolTip("N-S / E-W / path sections: show stations at most this far "
                                  "from the line, projected onto it")
        self.st_offset.editingFinished.connect(self._style_changed)
        form.addRow(crow)
        form.addRow("marker", mrow)
        form.addRow("max offset", self.st_offset)
        left.addWidget(box)

        # ---- figure + export --------------------------------------------- #
        box = QGroupBox("Figure")
        form = QFormLayout(box)
        self.title_edit = QLineEdit()
        self.title_edit.setToolTip("Placeholders: {description} {ve_text} {label} {ve} {kind}. "
                                   "Blank = no title.")
        self.title_edit.editingFinished.connect(self._style_changed)
        self.w_in = QDoubleSpinBox(decimals=2, minimum=1.0, maximum=40.0, singleStep=0.25, suffix=" in")
        self.h_in = QDoubleSpinBox(decimals=2, minimum=0.0, maximum=40.0, singleStep=0.25, suffix=" in")
        self.h_in.setSpecialValueText("auto (from VE)")
        for w in (self.w_in, self.h_in):
            w.editingFinished.connect(self._style_changed)
        srow = QHBoxLayout()
        srow.addWidget(self.w_in)
        srow.addWidget(QLabel("×"))
        srow.addWidget(self.h_in)
        self.font = QDoubleSpinBox(decimals=1, minimum=4.0, maximum=30.0, suffix=" pt")
        self.font.editingFinished.connect(self._style_changed)
        self.dpi = QSpinBox(minimum=50, maximum=1200, singleStep=50)
        self.prof_h = QDoubleSpinBox(decimals=2, minimum=0.3, maximum=10.0, singleStep=0.1, suffix=" in")
        self.prof_h.setToolTip("Height of the panel above the section for raster profiles (tab 2)")
        self.prof_h.editingFinished.connect(self._style_changed)
        self.legend_chk = QCheckBox("legend for layer lines / markers")
        self.legend_chk.toggled.connect(self._style_changed)
        self.topo_chk = QCheckBox("topography (model surface) in the profile panel")
        self.topo_chk.setToolTip("Adds the model's surface elevation along the section as a\n"
                                 "profile with its own colour-coded axis (raster profiles from\n"
                                 "tab 2 get theirs too: auto = left, then right)")
        self.topo_chk.toggled.connect(self._style_changed)
        self.topo_color = QComboBox()
        self.topo_color.setEditable(True)
        self.topo_color.addItems(["tab:brown", "black", "tab:green", "0.4"])
        self.topo_color.currentTextChanged.connect(self._style_changed)
        form.addRow("title", self.title_edit)
        form.addRow("size", srow)
        form.addRow("font", self.font)
        form.addRow("raster dpi", self.dpi)
        form.addRow("profile panel", self.prof_h)
        trow = QHBoxLayout()
        trow.addWidget(self.topo_chk, 1)
        trow.addWidget(self.topo_color)
        form.addRow(trow)
        form.addRow(self.legend_chk)
        left.addWidget(box)

        box = QGroupBox("Index map + export")
        form = QFormLayout(box)
        self.map_bg = QComboBox()
        for key, label in _MAP_BG:
            self.map_bg.addItem(label, key)
        self.map_bg.currentIndexChanged.connect(lambda _i: (self._store_style(), self._remap.poke()))
        self.map_pad = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=5000.0, suffix=" km")
        self.map_pad.editingFinished.connect(self._style_changed)
        self.inset_chk = QCheckBox("inset index map on section figures")
        self.inset_chk.setToolTip("A small map in a corner of each section: stations, the other "
                                  "sections of its sub-tab in grey,\nthis one in red. Not drawn in "
                                  "multi-panel layouts (use an index-map panel there).")
        self.inset_chk.toggled.connect(self._style_changed)
        self.inset_corner = QComboBox()
        for key in render.INSET_CORNERS:
            self.inset_corner.addItem(key, key)
        self.inset_corner.currentIndexChanged.connect(self._style_changed)
        self.inset_size = QDoubleSpinBox(decimals=2, minimum=0.3, maximum=6.0, singleStep=0.1,
                                         suffix=" in")
        self.inset_size.setToolTip("Longer side of the inset (shrunk to fit a flat section)")
        self.inset_size.editingFinished.connect(self._style_changed)
        irow = QHBoxLayout()
        irow.addWidget(self.inset_corner, 1)
        irow.addWidget(QLabel("size"))
        irow.addWidget(self.inset_size)
        self.out_dir = PathRow("folder for 'Export all'", kinds=("folder",))
        self.template = QLineEdit()
        self.template.setToolTip("Placeholders: {label} {kind} {model}. Extension = format.")
        one = QPushButton("Export this...")
        one.clicked.connect(self._export_one)
        self.all_btn = QPushButton("Export all")
        self.all_btn.setToolTip("Every section on this sub-tab")
        self.all_btn.clicked.connect(self._export_all)
        idx = QPushButton("Export index map...")
        idx.clicked.connect(self._export_index)
        brow = QHBoxLayout()
        for b in (one, self.all_btn, idx):
            brow.addWidget(b)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        form.addRow("map background", self.map_bg)
        form.addRow("map padding", self.map_pad)
        form.addRow(self.inset_chk)
        form.addRow("inset", irow)
        form.addRow("folder", self.out_dir)
        form.addRow("file name", self.template)
        form.addRow(brow)
        form.addRow(self.progress)
        left.addWidget(box)
        left.addStretch(1)
        root.addWidget(scroll_column(col))

        # ---- right: map over section ------------------------------------- #
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        mode_row = QHBoxLayout()
        self.mode_lbl = QLabel("")
        self.mode_lbl.setStyleSheet("color: #b45309; font-weight: bold;")
        self.finish_btn = QPushButton("Finish")
        self.finish_btn.clicked.connect(self._finish_mode)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel_mode)
        mode_row.addWidget(self.mode_lbl, 1)
        mode_row.addWidget(self.finish_btn)
        mode_row.addWidget(self.cancel_btn)
        rv.addLayout(mode_row)
        split = QSplitter(Qt.Vertical)
        self.map = FigurePreview()
        self.map.set_size_inches(7.0, 5.0)
        self.preview = FigurePreview()
        sec_area = QWidget()
        sv = QVBoxLayout(sec_area)
        sv.setContentsMargins(0, 0, 0, 0)
        nav = QHBoxLayout()
        self.prev_btn, self.next_btn = QPushButton("◀"), QPushButton("▶")
        for b, step, tip in ((self.prev_btn, -1, "previous section (Ctrl+Left / Page Up)"),
                             (self.next_btn, 1, "next section (Ctrl+Right / Page Down)")):
            b.setMaximumWidth(40)
            b.setToolTip(tip)
            b.clicked.connect(lambda _=False, d=step: self._step_section(d))
        self.nav_lbl = QLabel("")
        self.nav_lbl.setStyleSheet("font-weight: bold;")
        nav.addWidget(self.prev_btn)
        nav.addWidget(self.nav_lbl, 1)
        nav.addWidget(self.next_btn)
        sv.addLayout(nav)
        sv.addWidget(self.preview, 1)
        for keys, step in (("Ctrl+Left", -1), ("Ctrl+Right", 1), ("PgUp", -1), ("PgDown", 1)):
            sc = QShortcut(QKeySequence(keys), self)
            sc.setContext(Qt.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda d=step: self._step_section(d))
        split.addWidget(self.map)
        split.addWidget(sec_area)
        split.setSizes([380, 520])
        rv.addWidget(split, 1)
        root.addWidget(right, 1)
        self.map.canvas.mpl_connect("button_press_event", self._on_map_click)
        self._show_mode()

    # ------------------------------------------------------------------ #
    @property
    def sec(self) -> SectionInfo | None:
        secs = self.project.sections
        return secs[self._current] if 0 <= self._current < len(secs) else None

    def refresh(self) -> None:
        self._loading = True
        st = self.project.section_style
        self.auto_top.setChecked(st.auto_top)
        self.z_top.setValue(st.z_top_m / 1000.0)
        self.z_bottom.setValue(st.depth_bottom_m / 1000.0)
        self.vaxis.setCurrentIndex(max(self.vaxis.findData(st.vaxis), 0))
        self.show_interp.setChecked(st.show_interp)
        self.method.setCurrentIndex(max(self.method.findData(st.interp_method), 0))
        self.nx.setValue(st.interp_nx)
        self.nz.setValue(st.interp_nz)
        self.rbf_kernel.setCurrentText(st.rbf_kernel)
        self.rbf_smooth.setText(f"{st.rbf_smoothing:g}")
        self.rbf_eps.setText(f"{st.rbf_epsilon:g}")
        self.rbf_nb.setValue(st.rbf_neighbors)
        self.gauss.setValue(st.gauss_sigma_px)
        self.color.load(st.color)
        linked = self.project.house.enabled and self.project.house.link_colors
        self.color.setEnabled(not linked)
        self.color.setTitle("Colour scale (house style, tab 6)" if linked else "Colour scale")
        self.cbar.setCurrentIndex(max(self.cbar.findData(st.colorbar), 0))
        self.st_show.setChecked(st.show_stations)
        self.st_labels.setChecked(st.station_labels)
        self.surf.setChecked(st.show_surface)
        self.ends.setChecked(st.end_labels)
        self.st_marker.setCurrentText(st.station_marker)
        self.st_size.setValue(st.station_size)
        self.st_color.setCurrentText(st.station_color)
        self.st_offset.setValue(st.station_max_offset_m / 1000.0)
        self.title_edit.setText(st.title)
        self.w_in.setValue(st.width_in)
        self.h_in.setValue(st.height_in)
        self.font.setValue(st.font_size)
        self.dpi.setValue(int(st.dpi))
        self.prof_h.setValue(st.profile_height_in)
        self.legend_chk.setChecked(st.legend)
        self.topo_chk.setChecked(st.topo_profile)
        self.topo_color.setCurrentText(st.topo_color)
        self.map_bg.blockSignals(True)
        self.map_bg.clear()
        for key, label in _MAP_BG:
            self.map_bg.addItem(label, key)
        for lay in self.project.layers:
            if lay.kind in ("raster", "surface"):
                self.map_bg.addItem(f"layer: {lay.name}", f"layer:{lay.name}")
        self.map_bg.setCurrentIndex(max(self.map_bg.findData(st.map_background), 0))
        self.map_bg.blockSignals(False)
        self.map_pad.setValue(st.map_padding_m / 1000.0)
        self.inset_chk.setChecked(st.inset_map)
        self.inset_corner.setCurrentIndex(max(self.inset_corner.findData(st.inset_corner), 0))
        self.inset_size.setValue(st.inset_size_in)
        self.out_dir.setText(st.out_dir)
        self.template.setText(st.file_template)
        self._sync_method_widgets()
        group = self.project.section_group if self.project.section_group in SEC.SECTION_GROUPS else "all"
        self._group = group
        self._group_current = {}
        self.group_bar.blockSignals(True)
        self.group_bar.setCurrentIndex(list(SEC.SECTION_GROUPS).index(group))
        self.group_bar.blockSignals(False)
        self._sync_group_buttons()
        self._fill_list(self.project.current_section)
        self._loading = False
        self._remap.poke()
        self._redraw.poke()

    def on_shown(self) -> None:
        self._remap.poke()
        self._redraw.poke()

    def _store(self) -> None:
        self._store_section()
        self._store_style()

    def _store_style(self) -> None:
        st = self.project.section_style
        st.auto_top = self.auto_top.isChecked()
        st.z_top_m = self.z_top.value() * 1000.0
        st.depth_bottom_m = self.z_bottom.value() * 1000.0
        st.vaxis = self.vaxis.currentData()
        st.show_interp = self.show_interp.isChecked()
        st.interp_method = self.method.currentData()
        st.interp_nx = self.nx.value()
        st.interp_nz = self.nz.value()
        st.rbf_kernel = self.rbf_kernel.currentText()
        st.rbf_smoothing = read_float(self.rbf_smooth, st.rbf_smoothing)
        st.rbf_epsilon = read_float(self.rbf_eps, st.rbf_epsilon)
        st.rbf_neighbors = self.rbf_nb.value()
        st.gauss_sigma_px = self.gauss.value()
        self.color.store(st.color)
        st.colorbar = self.cbar.currentData()
        st.show_stations = self.st_show.isChecked()
        st.station_labels = self.st_labels.isChecked()
        st.show_surface = self.surf.isChecked()
        st.end_labels = self.ends.isChecked()
        st.station_marker = self.st_marker.currentText()
        st.station_size = self.st_size.value()
        st.station_color = self.st_color.currentText().strip() or "white"
        st.station_max_offset_m = self.st_offset.value() * 1000.0
        st.title = self.title_edit.text()
        st.width_in = self.w_in.value()
        st.height_in = self.h_in.value()
        st.font_size = self.font.value()
        st.dpi = self.dpi.value()
        st.profile_height_in = self.prof_h.value()
        st.legend = self.legend_chk.isChecked()
        st.topo_profile = self.topo_chk.isChecked()
        st.topo_color = self.topo_color.currentText().strip() or "tab:brown"
        st.map_background = self.map_bg.currentData() or "surface"
        st.map_padding_m = self.map_pad.value() * 1000.0
        st.inset_map = self.inset_chk.isChecked()
        st.inset_corner = self.inset_corner.currentData() or "upper right"
        st.inset_size_in = self.inset_size.value()
        st.out_dir = self.out_dir.text()
        st.file_template = self.template.text().strip() or "section_{label}.png"

    def _store_section(self) -> None:
        sec = self.sec
        if sec is None or self._loading:
            return
        sec.label = self.label.text().strip() or sec.label
        if sec.kind in ("ns", "ew"):
            sec.coord_m = self.coord.value() * 1000.0
            sec.start_m = self.span0.value() * 1000.0
            sec.end_m = self.span1.value() * 1000.0
        elif sec.kind == "stations":
            sec.stations = [t.strip() for t in self.st_list.text().split(",") if t.strip()]
            sec.station_path = self.st_path.currentData()
            sec.extend_m = self.extend.value() * 1000.0
        sec.reverse = self.reverse.isChecked()
        sec.x_axis = self.x_axis.currentData() or "distance"
        sec.ve = self.ve.value()
        self.project.current_section = self._current

    # ---- sub-tabs ------------------------------------------------------- #
    def _group_sections(self) -> list:
        return [self.project.sections[i] for i in self._rows]

    def _sync_group_buttons(self) -> None:
        for key, buttons in self.add_btns.items():
            for b in buttons:
                b.setVisible(self._group in (key, "all"))

    def _group_changed(self, idx: int) -> None:
        group = self.group_bar.tabData(idx) or "all"
        if group == self._group:
            return
        if self._mode is not None:
            self._cancel_mode()
        self._store_section()
        if self.sec is not None:
            self._group_current[self._group] = self.sec.label
        self._group = self.project.section_group = group
        self._sync_group_buttons()
        labels = [x.label for x in self.project.sections]
        want = self._group_current.get(group)
        self._fill_list(labels.index(want) if want in labels else self._current)
        self._remap.poke()
        self._redraw.poke()

    def _show_group_of(self, secs) -> None:
        """New sections that the current sub-tab doesn't list: switch to theirs."""
        if all(SEC.in_group(x, self._group) for x in secs):
            return
        groups = {SEC.section_group(x) for x in secs}
        group = groups.pop() if len(groups) == 1 else "all"
        self.group_bar.blockSignals(True)
        self.group_bar.setCurrentIndex(list(SEC.SECTION_GROUPS).index(group))
        self.group_bar.blockSignals(False)
        self._group = self.project.section_group = group
        self._sync_group_buttons()

    # ------------------------------------------------------------------ #
    def _fill_list(self, select: int) -> None:
        """Rebuild the list for the current sub-tab; *select* is an index into
        ``project.sections`` (the nearest section of this sub-tab is taken)."""
        secs = self.project.sections
        self._rows = [i for i, x in enumerate(secs) if SEC.in_group(x, self._group)]
        self.list.blockSignals(True)
        self.list.clear()
        for i in self._rows:
            self.list.addItem(self._list_text(secs[i]))
        self.list.blockSignals(False)
        self._current = -1
        row = -1
        if self._rows:
            if select in self._rows:
                row = self._rows.index(select)
            else:                           # the last one before it, else the first
                row = max(sum(i < select for i in self._rows) - 1, 0)
        self._load_section(self._rows[row] if row >= 0 else -1)
        if row >= 0:
            self.list.blockSignals(True)
            self.list.setCurrentRow(row)
            self.list.blockSignals(False)

    def _row(self) -> int:
        return self._rows.index(self._current) if self._current in self._rows else -1

    @staticmethod
    def _list_text(sec) -> str:
        return f"{sec.label} – {render.section_description(sec)}"

    def _select(self, row: int) -> None:
        i = self._rows[row] if 0 <= row < len(self._rows) else -1
        if self._loading or i == self._current:
            return
        self._store_section()
        self._load_section(i)
        self._remap.poke()
        self._redraw.poke()

    def _step_section(self, step: int) -> None:
        n = len(self._rows)
        if n < 2:
            return
        self.list.setCurrentRow(min(max(self._row() + step, 0), n - 1))

    def _update_nav(self) -> None:
        n, i, sec = len(self._rows), self._row(), self.sec
        self.prev_btn.setEnabled(i > 0)
        self.next_btn.setEnabled(0 <= i < n - 1)
        self.nav_lbl.setText(f"{sec.label}  ({i + 1} of {n})  –  {render.section_description(sec)}"
                             if sec is not None else
                             f"no {SEC.SECTION_GROUPS[self._group]} sections yet"
                             if self._group != "all" else "no sections yet")

    def _load_section(self, i: int) -> None:
        was = self._loading
        self._loading = True
        self._current = i
        if i >= 0:
            self.project.current_section = i
        self._update_nav()
        sec = self.sec
        self.sec_box.setEnabled(sec is not None)
        if sec is not None:
            self.label.setText(sec.label)
            if sec.kind in ("ns", "ew"):
                self.kind_pages.setCurrentIndex(0)
                self.coord_lbl.setText("easting" if sec.kind == "ns" else "northing")
                self.span_lbl.setText("northing range" if sec.kind == "ns" else "easting range")
                self.coord.setValue(sec.coord_m / 1000.0)
                self.span0.setValue(sec.start_m / 1000.0)
                self.span1.setValue(sec.end_m / 1000.0)
            elif sec.kind == "path":
                self.kind_pages.setCurrentIndex(1)
                self.path_info.setText(f"{len(sec.vertices)} vertices")
            else:
                self.kind_pages.setCurrentIndex(2)
                self.st_list.setText(", ".join(sec.stations))
                self.st_path.setCurrentIndex(max(self.st_path.findData(sec.station_path), 0))
                self.extend.setValue(sec.extend_m / 1000.0)
            self.reverse.setChecked(sec.reverse)
            self.x_axis.clear()
            for key in render.allowed_x_axes(sec):
                self.x_axis.addItem(render.X_AXES[key], key)
            self.x_axis.setCurrentIndex(max(self.x_axis.findData(sec.x_axis), 0))
            self.ve.setValue(sec.ve)
            self.sec_info.setText(render.section_description(sec))
        self._loading = was

    def _sec_changed(self, *_):
        if self._loading or self.sec is None:
            return
        self._store_section()
        item = self.list.item(self._row())
        if item is not None:
            item.setText(self._list_text(self.sec))
        self._update_nav()
        self._remap.poke()
        self._redraw.poke()

    def _style_changed(self, *_):
        if self._loading:
            return
        self._store_style()
        self._redraw.poke()

    def _method_changed(self, *_):
        self._sync_method_widgets()
        self._style_changed()

    def _sync_method_widgets(self):
        m = self.method.currentData()
        self.rbf_box.setVisible(m == "rbf")
        self.gauss.setEnabled(m == "gaussian")

    # ---- adding sections --------------------------------------------- #
    def _need_ready(self) -> bool:
        if not self.session.ready:
            QMessageBox.information(self, "Sections", "Load and georeference a model on tab 1 first.")
            return False
        return True

    def _append(self, secs) -> None:
        self._store_section()
        self.project.sections.extend(secs)
        self._show_group_of(secs)
        self._fill_list(len(self.project.sections) - 1)
        self._remap.poke()
        self._redraw.poke()

    def _new_label(self) -> str:
        return SEC.next_label(s.label for s in self.project.sections)

    def _add_lines(self, kind: str) -> None:
        if not self._need_ready():
            return
        st = self.session.stations_display()
        centre = (np.mean(st[:, 0] if kind == "ns" else st[:, 1]) / 1000.0) if len(st) else 0.0
        dlg = LineSeriesDialog(kind, round(float(centre)), self)
        if not dlg.exec_():
            return
        try:
            coords = dlg.coords_m()
        except ValueError:
            QMessageBox.warning(self, "Sections", "Positions must be numbers (km).")
            return
        a, b = dlg.span_m()
        new = []
        for c in coords:
            lab = SEC.next_label([s.label for s in self.project.sections] + [s.label for s in new])
            new.append(SectionInfo(label=lab, kind=kind, coord_m=c, start_m=a, end_m=b))
        self._append(new)

    def _add_path(self) -> None:
        if not self._need_ready():
            return
        self._start_mode("draw", None)

    def _add_stations(self) -> None:
        if not self._need_ready():
            return
        box = QMessageBox(self)
        box.setWindowTitle("Section along stations")
        box.setText("Load the station order from a .lst file, or click the stations on the map?")
        lst = box.addButton("Load .lst...", QMessageBox.AcceptRole)
        pick = box.addButton("Pick on map", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.exec_()
        if box.clickedButton() is lst:
            path, _ = QFileDialog.getOpenFileName(self, "Station list", self.project.workdir,
                                                  "site list (*.lst *.txt);;all files (*)")
            if path:
                self._append([SectionInfo(label=self._new_label(), kind="stations",
                                          stations=SEC.read_station_list(path))])
        elif box.clickedButton() is pick:
            self._start_mode("pick", None)

    def _add_from_file(self) -> None:
        if not self._need_ready():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Sections from file", self.project.workdir,
            "section set, path or station list (*.json *.csv *.txt *.xy *.lst);;"
            "section set (*.json);;E/N or lon/lat columns (*.csv *.txt *.xy);;"
            "station list (*.lst);;all files (*)")
        if not path:
            return
        if SEC.is_section_set(path):
            self._load_section_set(path)
            return
        try:
            if path.lower().endswith(".lst"):
                sec = SectionInfo(label=self._new_label(), kind="stations",
                                  stations=SEC.read_station_list(path))
            else:
                verts = SEC.read_path_file(path, self.session.maptr)
                sec = SectionInfo(label=self._new_label(), kind="path", vertices=verts.tolist())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Sections", str(exc))
            return
        self._append([sec])
        self.log(f"section {sec.label} from {os.path.basename(path)}")

    def _load_section_set(self, path: str) -> None:
        try:
            secs, notes = SEC.read_section_set(path, self.session.maptr,
                                               [x.label for x in self.project.sections])
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Sections", str(exc))
            return
        self._append(secs)
        self.log(f"{len(secs)} section(s) from {os.path.basename(path)}: "
                 + ", ".join(x.label for x in secs))
        for n in notes:
            self.log(f"  {n}")
        if any("CRS" in n for n in notes):
            QMessageBox.information(self, "Sections", "\n".join(notes))

    def _save_to_file(self) -> None:
        secs = self._group_sections()
        if not secs:
            QMessageBox.information(self, "Sections", "No sections on this sub-tab to save.")
            return
        self._store_section()
        tag = {"ns": "ns", "ew": "ew", "other": "paths", "all": "all"}[self._group]
        start = os.path.join(self.project.section_style.out_dir or self.project.workdir
                             or os.getcwd(), f"sections_{tag}.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save sections", start,
                                              "section set (*.json);;all files (*)")
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".json"
        try:
            SEC.write_section_set(path, secs, self.session.maptr if self.session.ready else None)
        except OSError as exc:
            QMessageBox.warning(self, "Sections", str(exc))
            return
        self.log(f"saved {len(secs)} section(s) to {path}")

    def _duplicate(self) -> None:
        if self.sec is None:
            return
        self._store_section()
        dup = copy.deepcopy(self.sec)
        dup.label = self._new_label()
        self._append([dup])

    def _delete(self) -> None:
        if self.sec is None:
            return
        i = self._current
        del self.project.sections[i]
        self._fill_list(i - 1 if i > 0 else 0)
        self._remap.poke()
        self._redraw.poke()

    def _redraw_path(self) -> None:
        if self.sec is not None and self._need_ready():
            self._start_mode("draw", self._current)

    def _repick_stations(self) -> None:
        if self.sec is not None and self._need_ready():
            self._start_mode("pick", self._current)

    # ---- map interaction --------------------------------------------- #
    def _start_mode(self, mode: str, target) -> None:
        self._mode, self._target, self._draft = mode, target, []
        self._show_mode()
        self._draw_draft()

    def _show_mode(self) -> None:
        texts = {"draw": "Drawing a path: left-click to add points, right-click or Finish to end.",
                 "pick": "Picking stations: left-click stations in order, right-click or Finish "
                         "to end."}
        on = self._mode is not None
        self.mode_lbl.setText(texts.get(self._mode, "Index map. Select a section on the left; "
                                                     "add one with the buttons."))
        self.mode_lbl.setStyleSheet("color: #b45309; font-weight: bold;" if on else "color: gray;")
        self.finish_btn.setVisible(on)
        self.cancel_btn.setVisible(on)

    def _cancel_mode(self) -> None:
        self._mode, self._draft = None, []
        self._show_mode()
        self._remap.poke()

    def _finish_mode(self) -> None:
        mode, draft, target = self._mode, list(self._draft), self._target
        self._cancel_mode()
        if mode == "draw":
            if len(draft) < 2:
                self.log("a path needs at least two points")
                return
            if target is None:
                self._append([SectionInfo(label=self._new_label(), kind="path",
                                          vertices=[list(map(float, p)) for p in draft])])
            else:
                self.project.sections[target].vertices = [list(map(float, p)) for p in draft]
                self._load_section(target)
                self._sec_changed()
        elif mode == "pick":
            if len(draft) < 2:
                self.log("pick at least two stations")
                return
            if target is None:
                self._append([SectionInfo(label=self._new_label(), kind="stations", stations=draft)])
            else:
                self.project.sections[target].stations = draft
                self._load_section(target)
                self._sec_changed()

    def _on_map_click(self, event) -> None:
        if self._mode is None or event.inaxes is not self._map_ax or event.xdata is None:
            return
        if self.map.toolbar.mode:          # zoom / pan active: leave the click to it
            return
        if event.button == 3:
            self._finish_mode()
            return
        if event.button != 1:
            return
        if self._mode == "draw":
            self._draft.append((event.xdata, event.ydata))
        else:
            xy = self.session.stations_display()
            if not len(xy):
                return
            x0, x1 = self._map_ax.get_xlim()
            d = np.hypot(xy[:, 0] - event.xdata, xy[:, 1] - event.ydata)
            i = int(d.argmin())
            if d[i] > 0.03 * abs(x1 - x0):
                return
            name = self.session.receivers.names[i]
            if self._draft and self._draft[-1] == name:
                self._draft.pop()          # click the last one again to undo it
            elif name not in self._draft:
                self._draft.append(name)
            self.mode_lbl.setText(f"Picking stations ({len(self._draft)}): "
                                  + ", ".join(self._draft[-6:]))
        self._draw_draft()

    def _draw_draft(self) -> None:
        ax = self._map_ax
        if ax is None:
            return
        if self._draft_artist is not None:
            try:
                self._draft_artist.remove()
            except (ValueError, AttributeError):
                pass
            self._draft_artist = None
        pts = []
        if self._mode == "draw":
            pts = self._draft
        elif self._mode == "pick":
            look = SEC.station_lookup(self.session)
            from ..backends.georef import _norm_name
            pts = [look[_norm_name(n)] for n in self._draft if _norm_name(n) in look]
        if pts:
            a = np.asarray(pts)
            (self._draft_artist,) = ax.plot(a[:, 0], a[:, 1], "o-", color="#f97316", lw=2,
                                            ms=5, zorder=20)
        self.map.draw()

    def _draw_map(self) -> None:
        if self._loading:
            return
        try:
            self._map_ax = render.render_index_map(
                self.map.figure, self.session, self._group_sections(),
                self.project.section_style, self.project.plan, current=self._row(),
                layers=self.project.layers)
        except Exception as exc:  # noqa: BLE001
            self.map.figure.clear()
            self.map.figure.text(0.5, 0.5, f"could not draw map:\n{exc}", ha="center", va="center")
            self._map_ax = None
        self._draft_artist = None
        self._draw_draft()

    # ---- section preview --------------------------------------------- #
    def _draw_section(self) -> None:
        if self._loading:
            return
        fig = self.preview.figure
        sec = self.sec
        if not self.session.ready or sec is None:
            fig.clear()
            fig.text(0.5, 0.5, "load + georeference a model on tab 1" if not self.session.ready
                     else "add a section with the buttons on the left"
                     + ("" if self._group == "all" else
                        f" (this sub-tab: {SEC.SECTION_GROUPS[self._group]})"),
                     ha="center", va="center", color="0.4")
            self.preview.draw()
            return
        self._gen += 1
        gen = self._gen
        sec_c, style_c = copy.deepcopy(sec), copy.deepcopy(self.project.section_style)
        what = (SEC.INTERP_METHODS.get(style_c.interp_method, "interpolation")
                if style_c.show_interp else "cutting cells")
        self.render_status.setText(f"computing ({what})...")
        layers_c = copy.deepcopy(self.project.layers)
        others = copy.deepcopy(SEC.group_members(self.project.sections, sec))
        run_async(self, lambda: SEC.prepare(self.session, sec_c, style_c, layers_c),
                  lambda data: self._prepared(gen, sec_c, style_c, data, others),
                  lambda exc: self._prepare_failed(gen, exc))

    def _prepared(self, gen, sec, style, data, others=None) -> None:
        if gen != self._gen:
            return                          # a newer request is on its way
        msg = (f"{data.cut.n:,} cell polygons, {data.profile.length / 1000.0:.0f} km long, "
               f"{len(data.stations.names)} stations")
        warn = [f"layer '{i.name}': {e}" for i, e in data.layer_errors]
        for it in data.layer_items:
            z = np.asarray(it.values, dtype="float64") if it.kind == "line" else np.zeros(0)
            z = z[np.isfinite(z)]
            if z.size and (z.min() > data.z_hi or z.max() < data.z_lo
                           or np.ptp(z) < 1e-3 * (data.z_hi - data.z_lo) and abs(z).max() < 500.0):
                warn.append(f"'{it.info.name}' lies outside / squashed at the top of the plotted "
                            f"depths ({-z.max():.4g} to {-z.min():.4g} m) -- check its depth units "
                            "on tab 2")
        if warn:
            msg += "<br><span style='color:#b91c1c'>" + "<br>".join(warn) + "</span>"
        self.render_status.setText(msg)
        self.preview.set_size_inches(*render.section_size(sec, style, data, self.session))
        try:
            render.render_section(self.preview.figure, self.session, sec, style, data,
                                  others=others)
        except Exception as exc:  # noqa: BLE001
            self._prepare_failed(gen, exc)
            return
        self.preview.draw()

    def _prepare_failed(self, gen, exc) -> None:
        if gen != self._gen:
            return
        self.render_status.setText(f"<span style='color:#b91c1c'>{exc}</span>")
        fig = self.preview.figure
        fig.clear()
        fig.text(0.5, 0.5, f"could not draw:\n{exc}", ha="center", va="center", color="#b91c1c")
        self.preview.draw()
        self.log(f"section failed: {exc}")

    # ---- export ------------------------------------------------------- #
    def _export_one(self) -> None:
        if not self.session.ready or self.sec is None:
            return
        self._store()
        st = self.project.section_style
        start = os.path.join(st.out_dir or self.project.workdir or os.getcwd(),
                             render.section_file_name(self.sec, st, self.session))
        path, _ = QFileDialog.getSaveFileName(
            self, "Export section", start,
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;TIFF (*.tif);;all files (*)")
        if not path:
            return
        try:
            render.export_section(path, self.session, self.sec, st, layers=self.project.layers,
                                  others=SEC.group_members(self.project.sections, self.sec))
        except Exception as exc:  # noqa: BLE001
            self.log(f"export failed: {exc}")
            return
        self.log(f"wrote {path}")

    def _export_all(self) -> None:
        if not self.session.ready or not self._rows:
            return
        self._store()
        st = copy.deepcopy(self.project.section_style)
        secs = copy.deepcopy(self._group_sections())
        pool = copy.deepcopy(self.project.sections)
        layers = copy.deepcopy(self.project.layers)
        out = st.out_dir or QFileDialog.getExistingDirectory(self, "Export all sections to")
        if not out:
            return
        self.project.section_style.out_dir = out
        self.out_dir.setText(out)
        self.all_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(secs))

        def done(paths):
            self.all_btn.setEnabled(True)
            self.progress.setVisible(False)
            self.log(f"wrote {len(paths)} sections to {out}")

        def failed(exc):
            self.all_btn.setEnabled(True)
            self.progress.setVisible(False)
            self.log(f"batch export failed: {exc}")

        def prog(i, n, msg):
            self.progress.setValue(i)
            self.progress.setFormat(f"{i}/{n}  {msg}")

        run_async_progress(self, lambda cb: render.export_sections_all(
            out, self.session, secs, st, cb, layers=layers, all_sections=pool), done, failed, prog)

    def _export_index(self) -> None:
        if not self.session.ready:
            return
        self._store()
        st = self.project.section_style
        tag = "" if self._group == "all" else "_" + {"ns": "ns", "ew": "ew", "other": "paths"}[self._group]
        start = os.path.join(st.out_dir or self.project.workdir or os.getcwd(),
                             f"section_index_map{tag}.png")
        path, _ = QFileDialog.getSaveFileName(self, "Export index map", start,
                                              "PNG (*.png);;PDF (*.pdf);;SVG (*.svg);;all files (*)")
        if not path:
            return
        try:
            render.export_index_map(path, self.session, self._group_sections(), st, self.project.plan,
                                    layers=self.project.layers)
        except Exception as exc:  # noqa: BLE001
            self.log(f"export failed: {exc}")
            return
        self.log(f"wrote {path}")
