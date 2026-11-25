"""Tab 7 -- refine an existing (usually inverted) model.

This tab has its own inputs (``Project.refine``), unlike the others: the model it
works on is a set of files that may predate this project.  It reads the GoFEM Files
tab's inversion settings when writing ``.prm`` files, and nothing else.
"""
from __future__ import annotations

import math
import os

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox, QTableWidget,
    QTableWidgetItem, QToolBox, QVBoxLayout, QWidget,
)

from ..backends import gofem_files, refine
from ..plotting.gui.widgets import PathRow
from .mesh_panel import _fmt_list, _parse_list
from .mesh_viewer import MeshViewer
from .panel_base import BasePanel
from .worker import run_async

_MODES = [("Keep the existing stations", "none"),
          ("Add new stations alongside them", "add"),
          ("Replace them with a new dataset", "replace")]
_FRAMES = [("Fit to the existing receivers' EDIs", "fit"),
           ("From a setup project (.mtinv.json)", "project"),
           ("Enter by hand", "manual")]
_KIND_LABELS = {"box": "Box", "ellipse": "Ellipse", "polygon": "Polygon", "stations": "Around stations"}
_PICKING_STYLE = "QPushButton { background-color: #c0392b; color: white; font-weight: bold; }"


def _spin(lo: float, hi: float, value: float = 0.0, decimals: int = 2, suffix: str = " km") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSuffix(suffix)
    s.setValue(value)
    return s


def _scrolled(widget: QWidget) -> QScrollArea:
    sc = QScrollArea()
    sc.setWidgetResizable(True)
    sc.setWidget(widget)
    return sc


def _wrapped(text: str = "") -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    return lab


def rule_shells(rule: dict) -> list:
    """(buffer km, depth km) per shell, or none if the rule's values aren't usable."""
    try:
        return refine.rule_shells(refine.rule_from_dict(rule))
    except ValueError:
        return []


def rule_label(rule: dict) -> str:
    kind = _KIND_LABELS.get(rule.get("kind"), rule.get("kind", "?"))
    floor = rule.get("min_size_km") or 0
    limit = f", min {floor:g} km" if floor else ""
    if rule.get("kind") == "stations":
        where = f"{rule.get('stations', 'new')} stations, r = {rule.get('radius_km', 0):g} km"
    elif rule.get("kind") == "polygon":
        where = f"{len(rule.get('polygon_km') or [])} vertices"
        if (rule.get("pnorm") or 1.0) > 1:
            where += f", rounded bottom (pnorm {rule['pnorm']:g})"
    else:
        where = f"{2 * rule.get('half_north_km', 0):g} x {2 * rule.get('half_east_km', 0):g} km"
    off = "" if rule.get("enabled", True) else "(off) "
    if rule.get("action") == "coarsen":
        side = "outside" if rule.get("outside") else "inside"
        margin = f" + {rule.get('buffer_km', 0):g} km" if rule.get("buffer_km") else ""
        cap = f", up to {rule['max_size_km']:g} km cells" if rule.get("max_size_km") else ""
        return f"{off}COARSEN {side} {kind.lower()}{margin}: {where}, {rule.get('levels', 1)} levels{cap}"
    shells = rule_shells(rule)
    buffers = [b for b, _d in shells]
    grown = f", buffer {buffers[0]:g} to {buffers[-1]:g} km" if buffers and max(buffers) > 0 else ""
    return f"{off}{kind}: {where}, {len(shells)} levels{grown}{limit}"


def rule_outline(rule: dict, buffer_km: float = 0.0) -> list:
    """The rule's outline in plan, mesh frame (north, east) metres, for the map."""
    kind = rule.get("kind")
    if kind == "polygon":
        if buffer_km <= 0:
            return [[a * 1e3, b * 1e3] for a, b in (rule.get("polygon_km") or [])]
        if len(rule.get("polygon_km") or []) < 3:
            return []
        grown = refine.rule_polygon(refine.rule_from_dict(rule), buffer_km)
        return [list(p) for p in grown.exterior.coords] if grown.geom_type == "Polygon" else []
    if kind not in ("box", "ellipse"):
        return []
    n, e = rule.get("center_north_km", 0.0) * 1e3, rule.get("center_east_km", 0.0) * 1e3
    hn = (rule.get("half_north_km", 0.0) + buffer_km) * 1e3
    he = (rule.get("half_east_km", 0.0) + buffer_km) * 1e3
    if kind == "box":
        return [[n - hn, e - he], [n - hn, e + he], [n + hn, e + he], [n + hn, e - he], [n - hn, e - he]]
    return [[n + hn * math.cos(t), e + he * math.sin(t)] for t in np.linspace(0, 2 * math.pi, 73)]


def shell_outlines(rule: dict) -> list:
    """An outline per distinct buffer, outer to inner, for the map (a coarsening rule's margin)."""
    if rule.get("action") == "coarsen":
        return [rule_outline(rule, rule["buffer_km"])] if rule.get("buffer_km") else []
    buffers = sorted({b for b, _d in rule_shells(rule) if b > 0}, reverse=True)
    return [rule_outline(rule, b) for b in buffers]


class RefinePanel(BasePanel):
    title = "7 - Refine Model"

    def build(self) -> None:
        self._loading = False
        self._surf = None            # SurfaceFaces of the loaded model (plain numpy)
        self._model_xy = np.zeros((0, 2))
        self._busy_flag = False
        self._rule_row = -1
        self._dem_check = None       # (key, DemCheck) of the last frame check
        self._dem_lonlat = {}        # DEM path -> whether it's in lon/lat
        self._checked = None         # (key, xy, is_new) of the last Check Stations
        self._cells = None           # ModelCells on show in the slice
        self._slice_note = ""
        self._slice_depths = np.zeros(0)

        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.toolbox = QToolBox()
        self.toolbox.addItem(_scrolled(self._page_model()), "1 · Model")
        self.toolbox.addItem(_scrolled(self._page_stations()), "2 · Stations")
        self.toolbox.addItem(_scrolled(self._page_rules()), "3 · Refinement Rules")
        self.toolbox.addItem(_scrolled(self._page_topo()), "4 · Topography")
        self.toolbox.addItem(_scrolled(self._page_output()), "5 · Output")
        left.addWidget(self.toolbox, 1)

        row = QHBoxLayout()
        self.estimate_btn = QPushButton("Estimate")
        self.estimate_btn.setToolTip("Run the refinement without saving anything, to get the real cell count.")
        self.estimate_btn.clicked.connect(lambda: self._start(write=False))
        self.build_btn = QPushButton("Build Refined Model")
        self.build_btn.clicked.connect(lambda: self._start(write=True))
        row.addWidget(self.estimate_btn)
        row.addWidget(self.build_btn)
        left.addLayout(row)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.hide()
        left.addWidget(self.busy)
        self.result_lbl = _wrapped("-")
        left.addWidget(self.result_lbl)

        holder = QWidget()
        holder.setLayout(left)
        holder.setFixedWidth(520)
        root.addWidget(holder)
        self.viewer = MeshViewer()
        self.viewer.point_picked.connect(self._on_point_picked)
        self.viewer.plan_scrolled.connect(self._on_plan_scrolled)
        right = QVBoxLayout()
        right.addWidget(self._slice_controls())
        right.addWidget(self.viewer, 1)
        root.addLayout(right, 1)

    def _slice_controls(self) -> QWidget:
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        self.slice_on = QCheckBox("Model Slice")
        self.slice_on.setChecked(True)
        self.slice_on.setToolTip("A plan view through the model, on the station map.")
        self.slice_on.toggled.connect(self._on_slice_toggled)
        self.slice_depth = QSlider(Qt.Horizontal)
        self.slice_depth.setToolTip("Depth below sea level. The mouse wheel over the map does this too.")
        self.slice_depth.valueChanged.connect(self._on_slice_moved)
        self.slice_lbl = QLabel("")
        self.slice_lbl.setMinimumWidth(300)
        row.addWidget(self.slice_on)
        row.addWidget(self.slice_depth, 1)
        row.addWidget(self.slice_lbl)
        self.slice_host = w
        w.hide()
        return w

    # ---- pages -------------------------------------------------------- #
    def _page_model(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.tria = PathRow("the model to refine, e.g. model_ZK_31.tria",
                            file_filter="deal.II triangulation (*.tria);;all files (*)")
        self.tria.changed.connect(self._on_tria_changed)
        self.materials = PathRow("its .materials file", file_filter="materials (*.materials);;all files (*)")
        self.materials.changed.connect(self._store)
        self.receivers = PathRow("receivers_projected.csv", file_filter="receivers (*.csv);;all files (*)")
        self.receivers.changed.connect(self._store)
        form.addRow("Model", self.tria)
        form.addRow("Materials", self.materials)
        form.addRow("Receivers", self.receivers)
        self.load_btn = QPushButton("Load Model")
        self.load_btn.clicked.connect(self._load)
        form.addRow(self.load_btn)
        self.model_lbl = _wrapped("no model loaded")
        form.addRow(self.model_lbl)
        return w

    def _page_stations(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.mode = QComboBox()
        for label, value in _MODES:
            self.mode.addItem(label, value)
        self.mode.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Stations", self.mode)
        self.edi = PathRow("folder of EDIs, or a .lst", kinds=("folder", "file"))
        self.edi.changed.connect(self._store)
        form.addRow("New stations", self.edi)
        self.frame = QComboBox()
        for label, value in _FRAMES:
            self.frame.addItem(label, value)
        self.frame.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Model frame", self.frame)
        self.fit_edis = PathRow("EDIs of the existing receivers", kinds=("folder", "file"))
        self.fit_edis.changed.connect(self._store)
        form.addRow("Existing EDIs", self.fit_edis)
        self.frame_project = PathRow("the project the model was built with",
                                     file_filter="setup project (*.mtinv.json);;all files (*)")
        self.frame_project.changed.connect(self._store)
        form.addRow("Setup project", self.frame_project)
        self.frame_crs = QLineEdit()
        self.frame_crs.setPlaceholderText("e.g. EPSG:32609")
        self.frame_crs.editingFinished.connect(self._store)
        form.addRow("CRS", self.frame_crs)
        self.frame_x0 = _spin(-1e9, 1e9, 0.0, 3, " m")
        self.frame_y0 = _spin(-1e9, 1e9, 0.0, 3, " m")
        self.frame_x0.editingFinished.connect(self._store)
        self.frame_y0.editingFinished.connect(self._store)
        form.addRow("Origin easting", self.frame_x0)
        self.frame_rows = {"fit": [self.fit_edis], "project": [self.frame_project],
                           "manual": [self.frame_crs, self.frame_x0, self.frame_y0]}
        self.stations_form = form
        form.addRow("Origin northing", self.frame_y0)
        lay.addLayout(form)
        self.check_btn = QPushButton("Check Stations")
        self.check_btn.setToolTip("Place the new stations in the model and show the cell size under each.")
        self.check_btn.clicked.connect(lambda: self._check_stations())
        lay.addWidget(self.check_btn)
        self.station_lbl = _wrapped("")
        lay.addWidget(self.station_lbl)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Station", "Status", "Surface cell (km)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        lay.addWidget(self.table, 1)
        return w

    def _page_rules(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(_wrapped("Rules run in order. A rule refines once per level, outer shell first, as "
                               "in the Mesh tab: each shell splits every cell that overlaps the region "
                               "grown by its buffer, down to its depth, and never below the minimum cell "
                               "size. A rule can coarsen instead: it merges split cells back into their "
                               "parents, inside its region or outside it (the padding), never past the "
                               "model's original grid. Depths are below the local surface; north and east "
                               "are in the model's frame."))
        row = QHBoxLayout()
        for kind, label in _KIND_LABELS.items():
            b = QPushButton("+ " + label)
            b.clicked.connect(lambda _=False, k=kind: self._add_rule(k))
            row.addWidget(b)
        lay.addLayout(row)
        self.rule_list = QListWidget()
        self.rule_list.setMaximumHeight(110)
        self.rule_list.currentRowChanged.connect(self._on_rule_selected)
        lay.addWidget(self.rule_list)
        row = QHBoxLayout()
        self.remove_btn = QPushButton("Remove Rule")
        self.remove_btn.clicked.connect(self._remove_rule)
        self.up_btn = QPushButton("\u25b2")
        self.up_btn.setToolTip("Run this rule earlier")
        self.up_btn.clicked.connect(lambda: self._move_rule(-1))
        self.down_btn = QPushButton("\u25bc")
        self.down_btn.setToolTip("Run this rule later")
        self.down_btn.clicked.connect(lambda: self._move_rule(1))
        row.addWidget(self.remove_btn, 1)
        row.addWidget(self.up_btn)
        row.addWidget(self.down_btn)
        lay.addLayout(row)

        self.rule_form_host = QWidget()
        form = QFormLayout(self.rule_form_host)
        self.rule_form = form
        self.r_enabled = QCheckBox("Use this rule")
        self.r_action = QComboBox()
        self.r_action.addItem("Refine (split cells)", "refine")
        self.r_action.addItem("Coarsen (merge split cells)", "coarsen")
        self.r_action.setToolTip(
            "Coarsen merges cells that were split back into their parents (8 into 1), up to Levels\n"
            "times. Inverted cells merge into a new cell with the volume-weighted mean of their\n"
            "log10 conductivity; air, sea and fixed ids only merge with their own kind. It can't\n"
            "go below the model's original grid.")
        self.r_outside = QComboBox()
        self.r_outside.addItem("Inside the region", False)
        self.r_outside.addItem("Outside the region (the padding)", True)
        self.r_outside.setToolTip(
            "Outside: every cell whose parent stays clear of the region (grown by the margin, down to its\n"
            "depth) is merged, so the region keeps its cells. Inside: cells whose parent touches it.")
        self.r_maxsize = _spin(0, 1e5, 0, 2)
        self.r_maxsize.setSpecialValueText("no limit")
        self.r_maxsize.setToolTip("Don't merge into cells wider than this.")
        self.r_levels = QSpinBox()
        self.r_levels.setRange(1, 12)
        self.r_min = _spin(0, 1e5, 0, 3)
        self.r_min.setSpecialValueText("no limit")
        self.r_mindz = _spin(0, 1e5, 0, 3)
        self.r_mindz.setSpecialValueText("no limit")
        self.r_cn = _spin(-1e5, 1e5)
        self.r_ce = _spin(-1e5, 1e5)
        self.r_hn = _spin(0, 1e5)
        self.r_he = _spin(0, 1e5)
        self.r_top = _spin(0, 1e5)
        self.r_bottom = _spin(0, 1e5)
        self.r_which = QComboBox()
        self.r_which.addItem("New stations", "new")
        self.r_which.addItem("All stations", "all")
        self.r_radius = _spin(0, 1e5)
        self.r_pick = QPushButton("Pick On Map")
        self.r_pick.setCheckable(True)
        self.r_pick.toggled.connect(self._on_pick_toggled)
        self.r_pnorm = _spin(1.0, 10.0, 1.0, 2, "")
        self.r_pnorm.setSingleStep(0.1)
        self.r_pnorm.setToolTip(
            "The shape of the bottom, as the Mesh tab's polygon pnorm (pyGoFEM refine_within_polygon).\n"
            "1 = flat: the whole polygon down to the depth.\n"
            "Above 1 = rounded: a bowl centred under the polygon, reaching the full depth in the middle\n"
            "and the top depth at the circle round the polygon. 2 is round; higher is flatter,\n"
            "closer to 1 more pointed.  Each shell has its own bowl.")
        self.r_clear = QPushButton("Clear Vertices")
        self.r_clear.clicked.connect(self._clear_polygon)
        self.r_buffer = _spin(0, 1e5)
        self.r_buffer_shrink = _spin(0.25, 0.75, 0.5, 2, "")
        self.r_buffer_shrink.setSingleStep(0.05)
        self.r_buffer_override = QCheckBox("Override Shell Buffers")
        self.r_buffer_list = QLineEdit()
        self.r_buffer_list.setPlaceholderText("outer → inner, km, e.g. 20, 8, 3")
        self.r_taper = QCheckBox("Taper Refined Depth Each Level")
        self.r_depth_shrink = _spin(0.25, 0.75, 0.55, 2, "")
        self.r_depth_shrink.setSingleStep(0.05)
        self.r_depth_override = QCheckBox("Override Shell Depths")
        self.r_depth_list = QLineEdit()
        self.r_depth_list.setPlaceholderText("outer → inner, km, e.g. 100, 55, 30")
        self.r_shells_lbl = _wrapped("")
        self.rule_rows = {
            "action": ("Action", self.r_action), "outside": ("Where", self.r_outside),
            "maxsize": ("Coarsen Up to Cell Size", self.r_maxsize),
            "levels": ("Refinement Levels", self.r_levels), "min": ("Min Cell Size", self.r_min),
            "mindz": ("Min Cell Thickness", self.r_mindz), "cn": ("Centre North", self.r_cn),
            "ce": ("Centre East", self.r_ce), "hn": ("Half Width North", self.r_hn),
            "he": ("Half Width East", self.r_he), "which": ("Refine Around", self.r_which),
            "radius": ("Radius", self.r_radius), "buffer": ("Outer Buffer (Shell 0)", self.r_buffer),
            "buffer_shrink": ("Shell Shrink Factor (Lateral)", self.r_buffer_shrink),
            "buffer_override": ("", self.r_buffer_override), "buffer_list": ("Shell Buffers", self.r_buffer_list),
            "top": ("Top Depth", self.r_top), "bottom": ("Refine Down to Depth (Shell 0)", self.r_bottom),
            "pnorm": ("Bottom Shape (pnorm)", self.r_pnorm), "taper": ("", self.r_taper), "depth_shrink": ("Shell Shrink Factor (Depth)", self.r_depth_shrink),
            "depth_override": ("", self.r_depth_override), "depth_list": ("Shell Depths", self.r_depth_list),
        }
        form.addRow(self.r_enabled)
        for label, widget in self.rule_rows.values():
            form.addRow(label, widget)
        form.addRow(self.r_pick)
        form.addRow(self.r_clear)
        form.addRow(self.r_shells_lbl)
        lay.addWidget(self.rule_form_host)
        for widget in (self.r_levels, self.r_min, self.r_mindz, self.r_cn, self.r_ce, self.r_hn,
                       self.r_he, self.r_top, self.r_bottom, self.r_radius, self.r_buffer,
                       self.r_buffer_shrink, self.r_depth_shrink, self.r_pnorm, self.r_maxsize):
            widget.valueChanged.connect(self._store_rule)
        for box in (self.r_enabled, self.r_buffer_override, self.r_taper, self.r_depth_override):
            box.toggled.connect(self._store_rule)
        self.r_buffer_list.editingFinished.connect(self._store_rule)
        self.r_depth_list.editingFinished.connect(self._store_rule)
        self.r_which.currentIndexChanged.connect(self._store_rule)
        self.r_outside.currentIndexChanged.connect(self._store_rule)
        self.r_action.currentIndexChanged.connect(self._on_action_changed)
        lay.addWidget(self._smoothing_box())
        lay.addStretch(1)
        return w

    def _smoothing_box(self) -> QGroupBox:
        box = QGroupBox("Smooth the Refined Cells")
        outer = QVBoxLayout(box)
        # a wrapped label in a form row gets its last line cut off: keep it out of the form
        outer.addWidget(_wrapped(
            "Refined cells start with their old cell's conductivity, so the old blocks show in "
            "the new cells. This Gaussian-filters the model as it was before refining "
            "(log10 conductivity) and gives each refined cell the filtered value at its centre, "
            "under a new material id. Other cells, air, sea and the fixed ids on page 5 don't "
            "change. Lengths are the Gaussian's standard deviation."))
        host = QWidget()
        outer.addWidget(host)
        form = QFormLayout(host)
        form.setContentsMargins(0, 0, 0, 0)
        self.smooth = QCheckBox("Smooth the refined cells")
        self.smooth.toggled.connect(self._store)
        form.addRow(self.smooth)
        self.smooth_mode = QComboBox()
        self.smooth_mode.addItem("Relative to the original cell's size", "relative")
        self.smooth_mode.addItem("Fixed lengths (km)", "km")
        self.smooth_mode.setToolTip(
            "Relative: each refined cell is smoothed over a fraction of the cell it came from, so the "
            "smoothing follows the blocks being smoothed at every depth. Fixed: the same everywhere.")
        self.smooth_mode.currentIndexChanged.connect(self._on_smooth_mode)
        form.addRow("Lengths", self.smooth_mode)
        self.smooth_h = _spin(0.01, 1e4, 0.25, 2, "")
        self.smooth_h.editingFinished.connect(self._store)
        form.addRow("Across", self.smooth_h)
        self.smooth_v = _spin(0.01, 1e4, 0.25, 2, "")
        self.smooth_v.editingFinished.connect(self._store)
        form.addRow("Down", self.smooth_v)
        self.smooth_note = _wrapped("")
        outer.addWidget(self.smooth_note)
        return box

    def _on_smooth_mode(self, *_a) -> None:
        if self._loading:
            return
        # keep the meaning roughly: 0.25 of a cell <-> a few km
        relative = self.smooth_mode.currentData() == "relative"
        self.smooth_h.setValue(0.25 if relative else 5.0)
        self.smooth_v.setValue(0.25 if relative else 1.0)
        self._store()

    def _page_topo(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.addRow(_wrapped("Refining alone leaves the surface exactly where it is. A refit moves it "
                             "onto a DEM by the difference from where it is now, so it can't apply "
                             "the relief twice. Only the area the DEM covers changes, so a detailed "
                             "regional DEM is fine on a larger model. Use a Topography .npz (lon/lat, "
                             "placed using the Model Frame on the Stations page), or a Project DEM .npz "
                             "made in the model's frame. Not available for models with sea cells."))
        self.refit = QCheckBox("Refit the surface to a DEM")
        self.refit.toggled.connect(self._on_dem_changed)
        form.addRow(self.refit)
        self.dem = PathRow("DEM (.npz)", file_filter="DEM (*.npz);;all files (*)")
        self.dem.changed.connect(self._on_dem_changed)
        form.addRow("DEM", self.dem)
        self.dem_lbl = _wrapped("")
        form.addRow(self.dem_lbl)
        self.blend = _spin(0, 1e5, 0)
        self.blend.setSpecialValueText("automatic")
        self.blend.setToolTip("How far inside the DEM's edge the refit fades to nothing.")
        self.blend.editingFinished.connect(self._store)
        form.addRow("Fade at DEM Edge", self.blend)
        return w

    def _page_output(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        self.out_dir = PathRow("folder for the refined model", kinds=("folder",))
        self.out_dir.changed.connect(self._store)
        form.addRow("Output Folder", self.out_dir)
        self.stem = QLineEdit()
        self.stem.editingFinished.connect(self._store)
        form.addRow("Model Name", self.stem)
        self.max_cells = QSpinBox()
        self.max_cells.setRange(1000, 2_000_000_000)
        self.max_cells.setSingleStep(500_000)
        self.max_cells.setGroupSeparatorShown(True)
        self.max_cells.setToolTip("Stop rather than go past this many cells.")
        self.max_cells.editingFinished.connect(self._store)
        form.addRow("Cell Limit", self.max_cells)
        self.write_mask = QCheckBox("Write a free-domain mask")
        self.write_mask.setToolTip("Not needed if your inversion uses an Active domain box.")
        self.write_mask.toggled.connect(self._store)
        form.addRow(self.write_mask)
        self.fixed_ids = QLineEdit()
        self.fixed_ids.setToolTip("Material ids that aren't inverted (used by the mask and by smoothing). "
                                  "Loading a model suggests these.")
        self.fixed_ids.editingFinished.connect(self._store)
        form.addRow("Fixed Material IDs", self.fixed_ids)
        self.write_gofem = QCheckBox("Write frequencies, data and .prm files")
        self.write_gofem.setToolTip("Modelling and inversion settings come from the GoFEM Files tab.")
        self.write_gofem.toggled.connect(self._store)
        form.addRow(self.write_gofem)
        self.old_data = PathRow("the model's existing data", file_filter="GoFEM data (*.gdat *.dat);;all files (*)")
        self.old_data.changed.connect(self._store)
        form.addRow("Existing Data", self.old_data)
        self.old_freq = PathRow("its frequencies file")
        self.old_freq.changed.connect(self._store)
        form.addRow("Existing Frequencies", self.old_freq)
        self.domain_box = QLineEdit()
        self.domain_box.setPlaceholderText("north0, east0, z0, north1, east1, z1 (m)")
        self.domain_box.setToolTip("Cells outside this box aren't inverted. Loading a model fills it in "
                                   "from an input_inversion*.prm near it, if there is one.")
        self.domain_box.editingFinished.connect(self._store)
        form.addRow("Active Domain Box", self.domain_box)
        self.data_note = _wrapped("")
        form.addRow(self.data_note)
        return w

    # ---- state -------------------------------------------------------- #
    def refresh(self) -> None:
        self._loading = True
        try:
            r = self.project.refine
            self.tria.setText(r.tria)
            self.materials.setText(r.materials)
            self.receivers.setText(r.receivers)
            self.mode.setCurrentIndex(max(self.mode.findData(r.station_mode), 0))
            self.edi.setText(r.stations_edi)
            self.frame.setCurrentIndex(max(self.frame.findData(r.frame), 0))
            self.fit_edis.setText(r.frame_fit_edis)
            self.frame_project.setText(r.frame_project)
            self.frame_crs.setText(r.frame_crs)
            self.frame_x0.setValue(r.frame_x0)
            self.frame_y0.setValue(r.frame_y0)
            self.refit.setChecked(r.refit)
            self.dem.setText(r.refit_dem)
            self.blend.setValue(r.blend_km)
            self.out_dir.setText(r.out_dir)
            self.stem.setText(r.stem)
            self.max_cells.setValue(r.max_cells)
            self.smooth.setChecked(r.smooth)
            self.smooth_mode.setCurrentIndex(max(self.smooth_mode.findData(r.smooth_mode), 0))
            self.smooth_h.setValue(r.smooth_horizontal)
            self.smooth_v.setValue(r.smooth_vertical)
            self.write_mask.setChecked(r.write_mask)
            self.fixed_ids.setText(", ".join(str(i) for i in r.fixed_ids))
            self.write_gofem.setChecked(r.write_gofem)
            self.old_data.setText(r.existing_data)
            self.old_freq.setText(r.existing_frequencies)
            self.domain_box.setText(r.active_domain_box)
            self._reload_rule_list()
        finally:
            self._loading = False
        self._update_visibility()

    def _store(self, *_a) -> None:
        if self._loading:
            return
        r = self.project.refine
        r.tria, r.materials, r.receivers = self.tria.text(), self.materials.text(), self.receivers.text()
        r.station_mode = self.mode.currentData()
        r.stations_edi = self.edi.text()
        r.frame = self.frame.currentData()
        r.frame_fit_edis, r.frame_project = self.fit_edis.text(), self.frame_project.text()
        r.frame_crs = self.frame_crs.text().strip()
        r.frame_x0, r.frame_y0 = self.frame_x0.value(), self.frame_y0.value()
        r.refit, r.refit_dem, r.blend_km = self.refit.isChecked(), self.dem.text(), self.blend.value()
        r.out_dir, r.stem = self.out_dir.text(), self.stem.text().strip() or "model_refined"
        r.max_cells = self.max_cells.value()
        r.smooth, r.smooth_mode = self.smooth.isChecked(), self.smooth_mode.currentData()
        r.smooth_horizontal, r.smooth_vertical = self.smooth_h.value(), self.smooth_v.value()
        r.write_mask = self.write_mask.isChecked()
        try:
            r.fixed_ids = [int(v) for v in self.fixed_ids.text().replace(",", " ").split()]
        except ValueError:
            self.log("fixed material ids must be whole numbers")
        r.write_gofem = self.write_gofem.isChecked()
        r.existing_data, r.existing_frequencies = self.old_data.text(), self.old_freq.text()
        r.active_domain_box = self.domain_box.text().strip()
        self._update_visibility()
        self.notify_changed()

    def _set_row_visible(self, form: QFormLayout, widget: QWidget, on: bool) -> None:
        label = form.labelForField(widget)
        if label is not None:
            label.setVisible(on)
        widget.setVisible(on)

    def _update_visibility(self) -> None:
        mode = self.mode.currentData()
        new_stations = mode != "none"
        needs_frame = new_stations or (self.refit.isChecked() and self._dem_is_lonlat())
        frame = self.frame.currentData()
        form = self.stations_form
        self._set_row_visible(form, self.edi, new_stations)
        self._set_row_visible(form, self.frame, needs_frame)
        for kind, widgets in self.frame_rows.items():
            for w in widgets:
                self._set_row_visible(form, w, needs_frame and kind == frame)
        self.check_btn.setVisible(new_stations)
        self.dem.setEnabled(self.refit.isChecked())
        self.blend.setEnabled(self.refit.isChecked())
        self.fixed_ids.setEnabled(self.write_mask.isChecked() or self.smooth.isChecked())
        smooth = self.smooth.isChecked()
        for widget in (self.smooth_mode, self.smooth_h, self.smooth_v):
            widget.setEnabled(smooth)
        suffix = "" if self.smooth_mode.currentData() == "relative" else " km"
        self.smooth_h.setSuffix(suffix)
        self.smooth_v.setSuffix(suffix)
        self.smooth_note.setText(
            ("0.25 is light: across an old cell's face the change spreads over about two thirds of "
             "its width. 0.5 blurs a pattern of single cells out almost entirely."
             if self.smooth_mode.currentData() == "relative" else
             "Too long for fine cells can mean billions of cell pairs: the build stops if so.")
            if smooth else "")
        gofem = self.write_gofem.isChecked()
        use_old = gofem and mode != "replace"
        self._set_row_visible(self._output_form(), self.old_data, use_old)
        self._set_row_visible(self._output_form(), self.old_freq, use_old)
        self._set_row_visible(self._output_form(), self.domain_box, gofem)
        notes = {"none": "The existing data and frequencies are carried over.",
                 "add": "New stations' data is added to the existing data, and frequencies combined. "
                        "Without an existing data file, only the new data is written.",
                 "replace": "Data and frequencies come from the new stations' EDIs only."}
        self.data_note.setText(notes.get(mode, "") if gofem else "")

    def _dem_is_lonlat(self) -> bool:
        path = self.dem.text()
        if path not in self._dem_lonlat:
            try:
                self._dem_lonlat[path] = refine.is_lonlat_dem(path)
            except (OSError, ValueError):       # no such file, or not a DEM: the check reports it
                return False
        return self._dem_lonlat[path]

    def _output_form(self) -> QFormLayout:
        return self.out_dir.parentWidget().layout()

    def _on_mode_changed(self, *_a) -> None:
        self._store()
        if not self._loading:
            self._show_stations()
            self._check_dem()

    def _on_tria_changed(self, path: str) -> None:
        guess = refine.guess_companions(path) if path else {}
        if guess.get("materials") and not self.materials.text():
            self.materials.setText(guess["materials"])
        if guess.get("receivers") and not self.receivers.text():
            self.receivers.setText(guess["receivers"])
        self._store()

    # ---- DEM frame check ---------------------------------------------- #
    def _frame_key(self) -> tuple:
        r = self.project.refine
        return (r.frame, r.frame_fit_edis, r.frame_project, r.frame_crs, r.frame_x0, r.frame_y0)

    def _dem_key(self) -> tuple:
        r = self.project.refine
        return (r.tria, r.refit_dem, *self._frame_key())

    def _on_dem_changed(self, *_a) -> None:
        self._store()
        if not self._loading:
            self._check_dem()

    def _show_dem_check(self) -> None:
        check = self._dem_check[1] if self._dem_check and self._dem_check[0] == self._dem_key() else None
        if check is None or not self.project.refine.refit:
            self.dem_lbl.setText("")
        elif check.is_problem:
            self.dem_lbl.setText(f"<span style='color:#c0392b'><b>Check this DEM.</b> {check.message}</span>")
        else:
            self.dem_lbl.setText(check.message)

    def _check_dem(self, then=None) -> None:
        """Compare the chosen DEM with the model's surface, in the background.  *then* runs
        afterwards, whatever the result."""
        r = self.project.refine
        self._show_dem_check()
        if self._busy_flag and then is None:      # a build checks it again before starting
            return
        if not (r.refit and r.refit_dem and os.path.isfile(r.refit_dem)):
            if then:
                then()
            return
        if self._dem_check and self._dem_check[0] == self._dem_key():
            if then:
                then()
            return
        key, surf, tria_path, dem_path = self._dem_key(), self._surf, r.tria, r.refit_dem
        spec = refine.spec_from_info(r)
        if surf is None and not (tria_path and os.path.isfile(tria_path)):
            self.dem_lbl.setText("load the model (page 1) to check this DEM against it")
            if then:
                then()
            return

        def job():
            found = surf
            if found is None:
                tria = refine.load_tria(tria_path)
                found = refine.surface_faces(tria)
            georef = refine.resolve_georef(spec)[0] if refine.is_lonlat_dem(dem_path) else None
            return key, refine.check_dem_frame(found, dem_path, georef=georef)

        def done(res):
            self._set_busy(False)
            self._dem_check = res
            self._show_dem_check()
            self.log(("WARNING: " if res[1].is_problem else "") + res[1].message)
            if then:
                then()

        def failed(exc):
            self._set_busy(False)
            self.dem_lbl.setText(f"<b>Couldn't check this DEM:</b> {exc}")
            self.log(f"refine: couldn't check the DEM: {exc}")

        self._set_busy(True)
        self.dem_lbl.setText("checking the DEM against the model's surface...")
        run_async(self, job, done, failed)

    # ---- rules -------------------------------------------------------- #
    def _reload_rule_list(self) -> None:
        keep = self.rule_list.currentRow()
        self.rule_list.blockSignals(True)
        self.rule_list.clear()
        for rule in self.project.refine.rules:
            self.rule_list.addItem(rule_label(rule))
        self.rule_list.blockSignals(False)
        rows = self.rule_list.count()
        self.rule_list.setCurrentRow(min(max(keep, 0), rows - 1) if rows else -1)
        self._on_rule_selected(self.rule_list.currentRow())

    def _add_rule(self, kind: str) -> None:
        self._stop_picking()
        rule = refine.RefineRule(kind=kind, taper=True)   # as a new Mesh tab stage starts
        if kind in ("box", "ellipse"):
            centre = self._station_centre()
            if centre is None:
                self.log("this rule is centred on (0, 0) until the stations are known: load the "
                         "model, or press Check Stations, then add it")
            else:
                rule.center_north_km, rule.center_east_km = centre
        if kind == "stations":
            rule.stations = "new" if self.mode.currentData() != "none" else "all"
            rule.depth_bottom_km = 10.0
            rule.levels = 2
        self.project.refine.rules.append(rule.asdict())
        self._reload_rule_list()
        self.rule_list.setCurrentRow(self.rule_list.count() - 1)
        self.notify_changed()

    def _remove_rule(self) -> None:
        row = self.rule_list.currentRow()
        if row < 0:
            return
        self._stop_picking()
        del self.project.refine.rules[row]
        self._reload_rule_list()
        self.notify_changed()

    def _move_rule(self, step: int) -> None:
        row, rules = self.rule_list.currentRow(), self.project.refine.rules
        if not (0 <= row < len(rules) and 0 <= row + step < len(rules)):
            return
        self._stop_picking()
        rules[row], rules[row + step] = rules[row + step], rules[row]
        self._reload_rule_list()
        self.rule_list.setCurrentRow(row + step)
        self.notify_changed()

    def _station_centre(self) -> tuple[float, float] | None:
        """(north, east) km at the middle of the stations a new rule is most likely for: the
        new ones if they're known, or else the model's own."""
        mode, xy = self.mode.currentData(), None
        checked = self._current_check()
        if mode != "none" and checked is not None and checked[2].any():
            xy = checked[1][checked[2]]
        elif mode != "replace" and len(self._model_xy):
            xy = self._model_xy
        if xy is None:
            return None
        middle = 0.5 * (xy.min(axis=0) + xy.max(axis=0)) / 1e3
        return round(float(middle[0]), 2), round(float(middle[1]), 2)

    def _current_rule(self) -> dict | None:
        row = self.rule_list.currentRow()
        rules = self.project.refine.rules
        return rules[row] if 0 <= row < len(rules) else None

    def _on_rule_selected(self, _row: int) -> None:
        self._stop_picking()
        rule = self._current_rule()
        self.rule_form_host.setVisible(rule is not None)
        self.remove_btn.setEnabled(rule is not None)
        row = self.rule_list.currentRow()
        self.up_btn.setEnabled(rule is not None and row > 0)
        self.down_btn.setEnabled(rule is not None and row < self.rule_list.count() - 1)
        if rule is None:
            self.viewer.set_preview_polygon([])
            return
        was = self._loading
        self._loading = True
        try:
            self.r_enabled.setChecked(rule.get("enabled", True))
            self.r_action.setCurrentIndex(max(self.r_action.findData(rule.get("action", "refine")), 0))
            self.r_outside.setCurrentIndex(1 if rule.get("outside") else 0)
            self.r_maxsize.setValue(rule.get("max_size_km", 0.0))
            self.r_levels.setValue(int(rule.get("levels", 1)))
            self.r_min.setValue(rule.get("min_size_km", 0.0))
            self.r_mindz.setValue(rule.get("min_dz_km", 0.0))
            self.r_cn.setValue(rule.get("center_north_km", 0.0))
            self.r_ce.setValue(rule.get("center_east_km", 0.0))
            self.r_hn.setValue(rule.get("half_north_km", 50.0))
            self.r_he.setValue(rule.get("half_east_km", 50.0))
            self.r_top.setValue(rule.get("depth_top_km", 0.0))
            self.r_bottom.setValue(rule.get("depth_bottom_km", 50.0))
            self.r_which.setCurrentIndex(max(self.r_which.findData(rule.get("stations", "new")), 0))
            self.r_radius.setValue(rule.get("radius_km", 5.0))
            self.r_pnorm.setValue(rule.get("pnorm", 1.0))
            self.r_buffer.setValue(rule.get("buffer_km", 0.0))
            self.r_buffer_shrink.setValue(rule.get("buffer_shrink", 0.5))
            self.r_buffer_override.setChecked(bool(rule.get("buffer_override", False)))
            self.r_buffer_list.setText(_fmt_list(rule.get("buffer_km_list") or []))
            self.r_taper.setChecked(bool(rule.get("taper", False)))
            self.r_depth_shrink.setValue(rule.get("depth_shrink", 0.55))
            self.r_depth_override.setChecked(bool(rule.get("depth_override", False)))
            self.r_depth_list.setText(_fmt_list(rule.get("depth_km_list") or []))
        finally:
            self._loading = was
        kind = rule.get("kind")
        coarsen = rule.get("action") == "coarsen"
        show = {key: True for key in self.rule_rows}
        show.update({"cn": kind in ("box", "ellipse"), "ce": kind in ("box", "ellipse"),
                     "hn": kind in ("box", "ellipse"), "he": kind in ("box", "ellipse"),
                     "which": kind == "stations", "radius": kind == "stations",
                     "top": kind != "stations", "pnorm": kind == "polygon",
                     "outside": coarsen, "maxsize": coarsen})
        if coarsen:     # no shells: one region, grown by the margin
            for key in ("min", "mindz", "buffer_shrink", "buffer_override", "buffer_list", "taper",
                        "depth_shrink", "depth_override", "depth_list"):
                show[key] = False
        for key, (_label, widget) in self.rule_rows.items():
            self._set_row_visible(self.rule_form, widget, show[key])
        self.rule_form.labelForField(self.r_levels).setText("Levels" if coarsen else "Refinement Levels")
        self.rule_form.labelForField(self.r_buffer).setText("Margin" if coarsen else "Outer Buffer (Shell 0)")
        self.rule_form.labelForField(self.r_which).setText("Around" if coarsen else "Refine Around")
        self.rule_form.labelForField(self.r_bottom).setText(
            ("Depth Radius" if coarsen else "Depth Radius (Shell 0)") if kind == "stations"
            else ("Region Down to Depth" if coarsen else "Refine Down to Depth (Shell 0)"))
        self.r_pick.setVisible(kind == "polygon")
        self.r_clear.setVisible(kind == "polygon")
        self._show_rule_shells(rule)

    def _show_rule_shells(self, rule: dict) -> None:
        """The controls and the map, for this rule's shells."""
        buffer_list = self.r_buffer_override.isChecked()
        depth_list = self.r_depth_override.isChecked()
        taper = self.r_taper.isChecked()
        self.r_levels.setEnabled(not (buffer_list and rule.get("buffer_km_list")))
        self.r_buffer.setEnabled(not buffer_list)
        self.r_buffer_shrink.setEnabled(not buffer_list)
        self.r_bottom.setEnabled(not depth_list)
        self.r_taper.setEnabled(not depth_list)
        self.r_depth_shrink.setEnabled(taper and not depth_list)
        self._set_row_visible(self.rule_form, self.r_buffer_list, buffer_list)
        self._set_row_visible(self.rule_form, self.r_depth_list, depth_list)
        shells = rule_shells(rule)
        if rule.get("action") == "coarsen":
            self.r_shells_lbl.setText("")
            self.r_levels.setEnabled(True)
            self.r_buffer.setEnabled(True)
            self.r_bottom.setEnabled(True)
        else:
            self.r_shells_lbl.setText("<br>".join(
                f"Shell {k}: buffer {b:g} km, down to {d:g} km" for k, (b, d) in enumerate(shells)))
        self.viewer.set_preview_polygon(rule_outline(rule), shell_outlines(rule))

    def _read_list(self, edit: QLineEdit, previous: list) -> list:
        """The numbers typed in *edit*, or *previous* if they aren't all usable."""
        try:
            values = _parse_list(edit.text())
            if any(v < 0 for v in values):
                raise ValueError("can't be negative")
        except ValueError as exc:
            self.log(f"refine: ignoring shell list {edit.text()!r}: {exc}")
            edit.setText(_fmt_list(previous or []))
            return previous or []
        return values

    def _store_rule(self, *_a) -> None:
        if self._loading:
            return
        rule = self._current_rule()
        if rule is None:
            return
        rule.update(enabled=self.r_enabled.isChecked(), levels=self.r_levels.value(),
                    action=self.r_action.currentData(), outside=bool(self.r_outside.currentData()),
                    max_size_km=self.r_maxsize.value(),
                    min_size_km=self.r_min.value(), min_dz_km=self.r_mindz.value(),
                    center_north_km=self.r_cn.value(), center_east_km=self.r_ce.value(),
                    half_north_km=self.r_hn.value(), half_east_km=self.r_he.value(),
                    depth_top_km=self.r_top.value(), depth_bottom_km=self.r_bottom.value(),
                    stations=self.r_which.currentData(), radius_km=self.r_radius.value(),
                    pnorm=self.r_pnorm.value(),
                    buffer_km=self.r_buffer.value(), buffer_shrink=self.r_buffer_shrink.value(),
                    buffer_override=self.r_buffer_override.isChecked(),
                    buffer_km_list=self._read_list(self.r_buffer_list, rule.get("buffer_km_list")),
                    taper=self.r_taper.isChecked(), depth_shrink=self.r_depth_shrink.value(),
                    depth_override=self.r_depth_override.isChecked(),
                    depth_km_list=self._read_list(self.r_depth_list, rule.get("depth_km_list")))
        self.rule_list.item(self.rule_list.currentRow()).setText(rule_label(rule))
        self._show_rule_shells(rule)
        self.notify_changed()

    def _on_action_changed(self, *_a) -> None:
        if self._loading:
            return
        rule = self._current_rule()
        if rule is not None and self.r_action.currentData() == "coarsen" and rule.get("action") != "coarsen":
            self.r_outside.setCurrentIndex(1)       # the usual use: thin out the padding
        self._store_rule()
        self._on_rule_selected(self.rule_list.currentRow())

    def _stop_picking(self) -> None:
        if self.r_pick.isChecked():
            self.r_pick.setChecked(False)

    def _on_pick_toggled(self, on: bool) -> None:
        self.viewer.set_picking(on)
        self.r_pick.setText("STOP Picking" if on else "Pick On Map")
        self.r_pick.setStyleSheet(_PICKING_STYLE if on else "")

    def _on_point_picked(self, north_m: float, east_m: float) -> None:
        rule = self._current_rule()
        if rule is None or not self.r_pick.isChecked():
            return
        rule.setdefault("polygon_km", []).append([north_m / 1e3, east_m / 1e3])
        self.rule_list.item(self.rule_list.currentRow()).setText(rule_label(rule))
        self.viewer.set_preview_polygon(rule_outline(rule), shell_outlines(rule))
        self.notify_changed()

    def _clear_polygon(self) -> None:
        rule = self._current_rule()
        if rule is not None:
            rule["polygon_km"] = []
            self.rule_list.item(self.rule_list.currentRow()).setText(rule_label(rule))
            self.viewer.set_preview_polygon([])
            self.notify_changed()

    # ---- work --------------------------------------------------------- #
    def _set_busy(self, on: bool) -> None:
        self._busy_flag = on
        self.busy.setVisible(on)
        for b in (self.load_btn, self.check_btn, self.estimate_btn, self.build_btn):
            b.setEnabled(not on)

    def _error(self, exc: Exception) -> None:
        self._set_busy(False)
        self.log(f"refine: {exc}")
        self.result_lbl.setText(f"<b>Stopped:</b> {exc}")

    def _load(self) -> None:
        self._store()
        r = self.project.refine
        if not r.tria or not os.path.isfile(r.tria):
            self.model_lbl.setText("choose the model's .tria file first")
            return
        self._set_busy(True)
        tria_path, mats_path, rec_path = r.tria, r.materials, r.receivers

        def job():
            tria = refine.load_tria(tria_path)
            mats = refine.read_materials(mats_path) if mats_path and os.path.isfile(mats_path) else {}
            info = refine.inspect(tria, mats)
            surf = refine.surface_faces(tria, [refine.AIR_ID, *info.water_ids])
            names, xyz = refine.read_receivers(rec_path) if rec_path and os.path.isfile(rec_path) else ([], np.zeros((0, 3)))
            return info, surf, names, xyz, refine.model_cells(tria, mats)

        run_async(self, job, self._after_load, self._error)

    def _after_load(self, res) -> None:
        self._set_busy(False)
        info, surf, names, xyz, cells = res
        self._surf = surf
        self._set_cells(cells)
        self._model_names, self._model_xy = names, xyz[:, :2]
        lines = [f"<b>{info.n_cells:,}</b> cells ({info.n_air:,} air), {info.n_ids:,} material ids",
                 f"cell diagonals {info.smallest_m / 1e3:.2f} to {info.largest_m / 1e3:.0f} km",
                 f"surface elevation {info.elevation_m[0]:.0f} to {info.elevation_m[1]:.0f} m",
                 f"{len(names)} receivers" if names else "no receivers"]
        if len(names):
            sizes = refine.station_cell_sizes(surf, xyz[:, :2])
            if np.isfinite(sizes).any():
                lines.append(f"surface cell under the receivers: {np.nanmin(sizes) / 1e3:.2f} to "
                             f"{np.nanmax(sizes) / 1e3:.2f} km")
        if info.water_ids:
            lines.append("has sea cells: stations go on the seafloor; a topography refit isn't available")
        if not info.materials:
            lines.append("<b>no .materials file:</b> can't check it matches this model")
        if info.missing_ids:
            lines.append(f"<b>{len(info.missing_ids):,} ids in the .tria aren't in the .materials</b>: "
                         "are these from the same model and iteration?")
        self.model_lbl.setText("<br>".join(lines))
        if not self.project.refine.fixed_ids:
            self.project.refine.fixed_ids = list(info.fixed_ids)
            self.fixed_ids.setText(", ".join(str(i) for i in info.fixed_ids))
        if not self.domain_box.text():
            box, source = refine.guess_domain_box(self.tria.text())
            if box:
                self.domain_box.setText(box)
                self.project.refine.active_domain_box = box
                self.log(f"active domain box from {os.path.basename(source)}: {box}")
        self.log(f"loaded model: {info.n_cells:,} cells, {len(names)} receivers")
        self._show_stations()
        self.notify_changed()
        self._dem_check = None                 # a surface from this load is now available
        if self.mode.currentData() != "none" and self.edi.text():
            self._check_stations(then=self._check_dem)
        else:
            self._check_dem()

    # ---- model slice -------------------------------------------------- #
    def _set_cells(self, cells, note: str = "") -> None:
        """Show this model in the slice, staying near the current depth."""
        was = self._slice_depths[self.slice_depth.value()] if len(self._slice_depths) else 0.0
        self._cells, self._slice_note = cells, note
        self._slice_depths = refine.slice_depths(cells)
        self.slice_depth.blockSignals(True)
        self.slice_depth.setRange(0, max(len(self._slice_depths) - 1, 0))
        if len(self._slice_depths):
            self.slice_depth.setValue(int(np.argmin(np.abs(self._slice_depths - was))))
        self.slice_depth.blockSignals(False)
        self.slice_host.setVisible(True)
        self._show_slice()

    def _show_slice(self) -> None:
        if self._cells is None or not self.slice_on.isChecked() or not len(self._slice_depths):
            self.viewer.set_model_slice(None)
            self.slice_lbl.setText("")
            return
        depth = float(self._slice_depths[self.slice_depth.value()])
        rects, values = refine.plan_slice(self._cells, depth)
        title = f"{depth / 1e3:.2f} km below sea level"
        text = f"{title}, {len(rects):,} cells"
        if len(rects):
            text += f", from {min((rects[:, 1] - rects[:, 0]).min(), (rects[:, 3] - rects[:, 2]).min()) / 1e3:.2f} km"
        self.slice_lbl.setText(text + self._slice_note)
        self.viewer.set_model_slice(rects, values, title + self._slice_note)

    def _on_slice_toggled(self, on: bool) -> None:
        self.slice_depth.setEnabled(on)
        self.viewer._show_2d()
        self._show_slice()

    def _on_slice_moved(self, _value: int) -> None:
        self.viewer._show_2d()
        self._show_slice()

    def _on_plan_scrolled(self, step: int) -> None:
        if self._cells is not None and self.slice_on.isChecked():
            self.slice_depth.setValue(self.slice_depth.value() + step)

    # ---- stations ----------------------------------------------------- #
    def _stations_key(self) -> tuple:
        r = self.project.refine
        return (r.tria, r.receivers, r.station_mode, r.stations_edi, *self._frame_key())

    def _current_check(self):
        """The last Check Stations, if nothing it depends on has changed."""
        return self._checked if self._checked and self._checked[0] == self._stations_key() else None

    def _show_stations(self, xy=None) -> None:
        """Plot the stations in play: all receivers, or the given set."""
        if xy is None:
            checked = self._current_check()
            if checked is not None:
                xy = checked[1]
            else:
                xy = self._model_xy if self.mode.currentData() != "replace" else np.zeros((0, 2))
        self.viewer.set_stations(np.asarray(xy).tolist())
        self.viewer.set_preview_polygon(rule_outline(self._current_rule() or {}))

    def _check_stations(self, then=None) -> None:
        """Place the new stations in the model, in the background.  *then* runs afterwards,
        whatever the result."""
        self._store()
        r = self.project.refine
        if self._surf is None:
            self.station_lbl.setText("load the model first (page 1)")
            if then:
                then()
            return
        if not r.stations_edi:
            self.station_lbl.setText("choose the new stations' EDIs")
            if then:
                then()
            return
        self._set_busy(True)
        spec = refine.spec_from_info(r)
        old_names, old_xy = list(getattr(self, "_model_names", [])), self._model_xy
        key = self._stations_key()

        def job():
            from ..backends import edi_io

            georef, note = refine.resolve_georef(spec)
            stations = edi_io.read_edi_headers(spec["stations_edi"])
            new_names, new_xy = refine.stations_in_model_frame(stations, georef)
            names, xy, is_new, notes = refine.merge_stations(old_names, old_xy, new_names, new_xy,
                                                             spec["station_mode"])
            return key, note, names, xy, is_new, notes

        def done(res):
            self._after_check(res)
            if then:
                then()

        def failed(exc):
            self._error(exc)
            if then:
                then()

        run_async(self, job, done, failed)

    def _after_check(self, res) -> None:
        self._set_busy(False)
        key, note, names, xy, is_new, notes = res
        self._checked = (key, xy, is_new)
        sizes = refine.station_cell_sizes(self._surf, xy)
        outside = int(np.isnan(sizes).sum())
        order = np.r_[np.flatnonzero(is_new), np.flatnonzero(~is_new)]
        self.table.setRowCount(len(names))
        for row, i in enumerate(order):
            status = "new" if is_new[i] else "kept"
            size = "outside the model" if np.isnan(sizes[i]) else f"{sizes[i] / 1e3:.2f}"
            for col, text in enumerate((names[i], status, size)):
                self.table.setItem(row, col, QTableWidgetItem(text))
        text = f"Model frame {note}.<br>{int(is_new.sum())} new, {int((~is_new).sum())} kept."
        if outside:
            text += f"<br><b>{outside} stations are outside the model.</b>"
        text += "".join(f"<br>{n}" for n in notes)
        self.station_lbl.setText(text)
        self._show_stations(xy)
        self.viewer._show_2d()

    def _start(self, write: bool) -> None:
        self._store()
        self._stop_picking()
        r = self.project.refine
        if not r.tria or not os.path.isfile(r.tria):
            self.result_lbl.setText("choose the model's .tria file on page 1")
            return
        out_dir = r.out_dir
        if write and not out_dir:
            self.result_lbl.setText("choose an output folder on page 5")
            return
        if r.refit and r.refit_dem and not (self._dem_check and self._dem_check[0] == self._dem_key()):
            self._check_dem(then=lambda: self._start(write))    # comes back here with the answer
            return
        if write:
            check = self._dem_check[1] if r.refit and r.refit_dem and self._dem_check else None
            if check is not None and check.is_problem:
                answer = QMessageBox.warning(self, "Check this DEM",
                                             check.message + "\n\nBuild with this DEM anyway?",
                                             QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if answer != QMessageBox.Yes:
                    self.toolbox.setCurrentIndex(3)
                    return
            existing = refine.existing_outputs(out_dir, r.stem)
            if existing:
                names = "\n".join(os.path.basename(p) for p in existing)
                answer = QMessageBox.question(self, "Overwrite files?",
                                              f"These are already in the output folder:\n\n{names}\n\nOverwrite them?")
                if answer != QMessageBox.Yes:
                    return
        spec = refine.spec_from_info(r)
        gofem = write and r.write_gofem
        project = self.project
        self._set_busy(True)
        self.result_lbl.setText("building..." if write else "estimating...")

        def job():
            if gofem:
                gofem_files.check_refined_inputs(project, spec)    # before the long part
            result = refine.run(spec, out_dir, log=self.log, write=write)
            if gofem:
                result["gofem"] = gofem_files.write_refined_inputs(project, spec, result, log=self.log)
            return result

        run_async(self, job, self._after_run, self._error)

    def _after_run(self, res: dict) -> None:
        self._set_busy(False)
        lines = [f"<b>{res['n_cells_before']:,} → {res['n_cells']:,} cells</b>, "
                 f"cell diagonals {res['smallest_m'] / 1e3:.2f} to {res['largest_m'] / 1e3:.0f} km"]
        rules = self.project.refine.rules
        for item in res["rules"]:
            if item.get("action") == "coarsen":
                lines.append(f"rule {item['rule'] + 1}: coarsened, {item['added']:+,} cells "
                             f"({item['merged']:,} merged)")
                if not item["merged"]:
                    lines[-1] += " (nothing merged: no split cells there, or the size limit)"
                continue
            done = f"{item['passes']} of {item['shells']} levels"
            lines.append(f"rule {item['rule'] + 1}: {done}, +{item['added']:,} cells")
            if item["passes"] == 0 and item["rule"] < len(rules):
                lines[-1] += " (nothing qualified: check its region and minimum size)"
        check = res.get("dem_check")
        if check is not None and check.is_problem:
            lines.append(f"<span style='color:#c0392b'><b>Check this DEM:</b> {check.message}</span>")
        if res.get("smoothed") and res["smoothed"]["cells"]:
            change = res["smoothed"]["change"]
            lines.append(f"smoothed {res['smoothed']['cells']:,} refined cells: log10 conductivity changed "
                         f"by {np.median(change):.2f} (median), {np.percentile(change, 99):.2f} (99%), "
                         f"{change.max():.2f} (most)")
        if res.get("refit"):
            lines.append("refit: misfit to the DEM {rms_before_m:.1f} → {rms_after_m:.1f} m rms, "
                         "largest shift {max_shift_m:.0f} m".format(**res["refit"]))
        for name, dn, de in res.get("off_face", []):
            lines.append(f"<span style='color:#b9770e'>{name} was on a cell face, where GoFEM may not find "
                         f"it: moved {dn:+.1f} m north, {de:+.1f} m east</span>")
        if res["names"]:
            line = f"{len(res['names'])} stations checked against the mesh"
            if res["receivers_moved_m"] is not None:
                line += f"; existing ones moved {res['receivers_moved_m']:.3f} m"
            lines.append(line)
            new = np.asarray(res["is_new"], dtype=bool)
            if new.any():
                before, after = res["station_size_before_m"][new], res["station_size_m"][new]
                lines.append(f"surface cell under new stations: {np.nanmedian(before) / 1e3:.2f} → "
                             f"{np.nanmedian(after) / 1e3:.2f} km (median)")
        self._set_cells(res["cells"], ", refined" if "paths" in res else ", refined (not saved)")
        if "paths" in res:
            lines.append("written to <b>" + os.path.dirname(res["paths"]["tria"]) + "</b>")
            self._show_model(res)
        else:
            lines.append("nothing saved (estimate)")
        self.result_lbl.setText("<br>".join(lines))
        self.log(f"refine: {res['n_cells_before']:,} -> {res['n_cells']:,} cells"
                 + (", written" if "paths" in res else " (estimate)"))

    def _show_model(self, res: dict) -> None:
        names, xyz = refine.read_receivers(res["paths"]["receivers"]) if "receivers" in res["paths"] \
            else ([], np.zeros((0, 3)))
        if len(names):
            self._show_stations(xyz[:, :2])
        xy = xyz[:, :2] if len(names) else np.zeros((1, 2))
        self.viewer.extent.setCurrentText("Tight to Survey" if len(names) else "Whole Model")
        self.viewer.show_result({
            "step": 0, "n_total_steps": 0, "n_cells": res["n_cells"],
            "smallest_cell_m": res["smallest_m"], "largest_cell_m": res["largest_m"],
            "vtu": res["paths"]["vtu"],
            "station_x_min": float(xy[:, 0].min()), "station_x_max": float(xy[:, 0].max()),
            "station_y_min": float(xy[:, 1].min()), "station_y_max": float(xy[:, 1].max()),
        })
        self.viewer.info.setText(f"{res['n_cells']:,} cells   |   smallest {res['smallest_m']:.0f} m   "
                                 f"largest {res['largest_m']:.0f} m   |   {res['paths']['vtu']}")
