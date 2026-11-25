"""Step 1 -- load EDI files, review stations, set the local frame.

This panel is fully functional in the skeleton: it uses the dependency-free
header reader in :mod:`mtinv_gui.backends.edi_io`, so it works before the
scientific stack is installed.
"""
from __future__ import annotations

import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget, QTableWidgetItem, QTabWidget,
    QVBoxLayout, QWidget,
)

from ..backends import edi_io
from .mpl_canvas import MplCanvas
from .panel_base import BasePanel

# Dropdown order: UTM always first (by far the most common choice), Custom EPSG
# always last, everything between sorted by worst-case scale error once stations
# are loaded.  Each combo row stores its index into this list as item data, so
# the on-screen order can change without touching which preset is selected.
# (label, proj_kind, epsg, local_proj) -- every entry is a map projection to
# metres.  epsg None means "read it from the EPSG box"; local_proj is only used
# for proj_kind == "local" (see backends/projection.LOCAL_PROJECTIONS).
_PROJ_PRESETS = [
    ("UTM  (zone auto from survey centre)", "utm", 0, ""),
    ("Local Transverse Mercator  (centred on survey)", "local", 0, "tmerc"),
    ("Local Oblique Stereographic  (centred on survey)", "local", 0, "stere"),
    ("Local Azimuthal Equidistant  (centred on survey)", "local", 0, "aeqd"),
    ("Local Lambert Conformal Conic  (std. parallels from survey)", "local", 0, "lcc"),
    ("Canada Atlas Lambert  EPSG:3979", "epsg", 3979, ""),
    ("Statistics Canada Lambert (NAD83)  EPSG:3347", "epsg", 3347, ""),
    ("North America Albers Equal Area  EPSG:5070", "epsg", 5070, ""),
    ("Custom EPSG...", "epsg", None, ""),
]

_LOCAL_DESCRIPTIONS = {
    "tmerc": "Transverse Mercator, central meridian through the origin. Conformal; "
             "scale error grows east-west, so it suits north-south elongated surveys.",
    "stere": "Oblique Stereographic at the origin. Conformal; scale error grows "
             "with distance from the origin, so it suits roughly round surveys.",
    "aeqd": "Azimuthal Equidistant at the origin. True distance from the origin "
            "in every direction, but not conformal (shapes stretch tangentially).",
    "lcc": "Lambert Conformal Conic, standard parallels at 1/6 and 5/6 of the "
           "survey's latitude span. Conformal; suits east-west elongated surveys.",
}


class SurveyPanel(BasePanel):
    title = "1 - Survey / EDIs"

    def build(self) -> None:
        root = QHBoxLayout(self)

        # ---- left: controls + table ------------------------------------ #
        left = QVBoxLayout()

        pick = QHBoxLayout()
        self.dir_edit = QLineEdit(self.project.survey.edi_dir)
        self.dir_edit.setPlaceholderText("folder of *.edi  OR  a .lst site list")
        folder_btn = QPushButton("Folder...")
        folder_btn.clicked.connect(self._browse_folder)
        list_btn = QPushButton("List (.lst)...")
        list_btn.clicked.connect(self._browse_list)
        load = QPushButton("Load")
        load.clicked.connect(self._load)
        pick.addWidget(QLabel("EDIs:"))
        pick.addWidget(self.dir_edit, 1)
        pick.addWidget(folder_btn)
        pick.addWidget(list_btn)
        pick.addWidget(load)
        left.addLayout(pick)

        self.summary = QLabel("no EDIs loaded")
        self.summary.setWordWrap(True)
        left.addWidget(self.summary)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["use", "station", "lon", "lat", "elev", "periods (s)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemChanged.connect(self._on_item_changed)
        left.addWidget(self.table, 1)

        # ---- local frame -------------------------------------------------#
        frame_box = QGroupBox("Local coordinate frame")
        form = QFormLayout(frame_box)
        self.proj_combo = QComboBox()
        for i, preset in enumerate(_PROJ_PRESETS):
            self.proj_combo.addItem(preset[0], i)
        self.proj_combo.currentIndexChanged.connect(self._on_proj_kind_changed)
        self.zone_edit = QLineEdit()
        self.hemi_check = QCheckBox("northern hemisphere")
        self.epsg_edit = QLineEdit()
        self.epsg_edit.setReadOnly(True)
        self.crs_lbl = QLabel("")
        self.crs_lbl.setWordWrap(True)
        self.crs_lbl.setStyleSheet("color: gray; font-size: 11px;")
        self.dist_lbl = QLabel("")
        self.dist_lbl.setWordWrap(True)
        self.dist_lbl.setStyleSheet("font-size: 11px;")
        self.origin_lon = QLineEdit()
        self.origin_lat = QLineEdit()
        self.origin_lock = QCheckBox("lock origin (don't re-centre on load)")
        self.origin_lock.toggled.connect(
            lambda on: (on or self._suggest_frame()))   # unlock -> snap back to centre
        suggest = QPushButton("Origin = survey centre")
        suggest.clicked.connect(self._suggest_frame_clicked)
        best_btn = QPushButton("Pick lowest distortion")
        best_btn.setToolTip(
            "Evaluate every projection in the list over your stations and select "
            "the one with the smallest worst-case scale error, then apply it.\n"
            "Note this ranks by scale only: a non-conformal projection (azimuthal "
            "equidistant) can win while stretching shapes -- check the angular "
            "distortion in the readout.")
        best_btn.clicked.connect(self._pick_lowest_distortion)
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._apply_frame)
        form.addRow("projection", self.proj_combo)
        form.addRow("UTM zone", self.zone_edit)
        form.addRow("", self.hemi_check)
        form.addRow("EPSG", self.epsg_edit)
        form.addRow("", self.crs_lbl)
        form.addRow("distortion", self.dist_lbl)
        form.addRow("origin lon", self.origin_lon)
        form.addRow("origin lat", self.origin_lat)
        form.addRow("", self.origin_lock)
        row = QHBoxLayout()
        row.addWidget(suggest)
        row.addWidget(best_btn)
        row.addWidget(apply_btn)
        form.addRow(row)
        left.addWidget(frame_box)

        root.addLayout(left, 3)

        # ---- right: station map (geographic | projected) ---------------- #
        self.map_tabs = QTabWidget()
        self.map = MplCanvas(figsize=(5, 6))              # Lat / Lon
        self.map_proj = MplCanvas(figsize=(5, 6))         # projected
        self.map_tabs.addTab(self.map, "Lat / Lon")
        self.map_tabs.addTab(self.map_proj, "Projected")
        self.map_tabs.currentChanged.connect(self._on_map_tab)
        root.addWidget(self.map_tabs, 2)

        self._stations: list[edi_io.Station] = []
        self._loaded_edi_dir: str | None = None
        self._proj_map_drawn = False
        self._on_proj_kind_changed()

    # ------------------------------------------------------------------ #
    def _browse_folder(self):
        start = self.dir_edit.text() or os.getcwd()
        if os.path.isfile(start):
            start = os.path.dirname(start)
        d = QFileDialog.getExistingDirectory(self, "Select EDI folder", start)
        if d:
            self.dir_edit.setText(d)
            self._load()

    def _browse_list(self):
        start = self.dir_edit.text() or os.getcwd()
        if os.path.isfile(start):
            start = os.path.dirname(start)
        f, _ = QFileDialog.getOpenFileName(
            self, "Select site list", start, "site list (*.lst);;all files (*)")
        if f:
            self.dir_edit.setText(f)
            self._load()

    def _load(self):
        """User-initiated (re)load: Browse/.lst pick.  Reads from disk *and*
        tells sibling panels the project changed.  Forces a fresh read (past
        the process-wide EDI cache) since this is the one place a real
        on-disk change -- picking the same path again after editing the
        folder -- should actually be picked up."""
        if self._load_from(self.dir_edit.text().strip(), force=True):
            self.notify_changed()

    def _load_from(self, path: str, force: bool = False) -> bool:
        """Read EDIs at *path* into the panel's widgets/project.  Returns
        whether anything was loaded.  Deliberately does NOT notify -- this is
        also called from ``refresh()``, which must stay silent (BasePanel's
        contract: refresh() never emits, or a sibling's unrelated project
        change re-triggers this panel's own notify_changed(), which then
        re-triggers every panel's refresh() -- including this one's original
        caller, resetting whatever transient UI state it had, e.g. the Mesh
        tab's polygon-picking toggle).

        Skips the reload (and the table/map rebuild) entirely when *path*
        matches what's already loaded and a fresh read wasn't requested --
        ``refresh()`` calls this unconditionally on *every* notify_changed()
        fired anywhere in the app, so without this a spin box in the Mesh tab
        was rebuilding this panel's whole station table and map every click.
        """
        if not force and path == self._loaded_edi_dir and self._stations:
            return True
        if not os.path.exists(path):
            self.log(f"path not found: {path}")
            return False
        self._stations = edi_io.read_edi_headers(path, force=force)
        fell_back = edi_io.drain_reader_fallbacks()
        if fell_back:
            self.log("⚠ " + edi_io.summarize_fallbacks(fell_back))
        if not self._stations:
            self.log("no readable EDI files found (empty folder, or .lst names "
                     "don't resolve to .edi files next to the list)")
            return False

        s = self.project.survey
        s.edi_dir = path
        s.edi_files = [st.path for st in self._stations]
        self._recompute_summary()
        self._fill_table()
        self._suggest_frame()
        # push the (possibly re-centred) frame to the project -- quietly, since
        # _load_from is also the silent refresh() path.  The caller (_load) does
        # the notify.
        self._write_frame_to_project()
        self._refresh_distortion()
        self._draw_maps()
        self._loaded_edi_dir = path
        self.log(f"loaded {len(self._stations)} EDI files from {path}")
        return True

    def _recompute_summary(self):
        s = self.project.survey
        excl = set(s.excluded)
        bbox = edi_io.bounding_box(self._stations, excl)
        cov = edi_io.period_coverage(self._stations, excl)
        s.n_stations = sum(1 for st in self._stations if st.station_id not in excl)
        s.min_lon, s.max_lon = bbox["MinLon"], bbox["MaxLon"]
        s.min_lat, s.max_lat = bbox["MinLat"], bbox["MaxLat"]
        s.period_min_s = cov["period_min_s"]
        s.period_max_s = cov["period_max_s"]
        s.n_periods = cov["n_periods"]
        self.summary.setText(
            f"{s.n_stations} stations in use  |  "
            f"lon [{s.min_lon:.4f}, {s.max_lon:.4f}]  "
            f"lat [{s.min_lat:.4f}, {s.max_lat:.4f}]  |  "
            f"periods ~[{s.period_min_s:.3g}, {s.period_max_s:.3g}] s "
            f"(<= {s.n_periods} per site)")

    def _fill_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._stations))
        excl = set(self.project.survey.excluded)
        for r, st in enumerate(self._stations):
            use = QTableWidgetItem()
            use.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            use.setCheckState(Qt.Unchecked if st.station_id in excl else Qt.Checked)
            self.table.setItem(r, 0, use)
            self.table.setItem(r, 1, QTableWidgetItem(st.station_id))
            self.table.setItem(r, 2, QTableWidgetItem(f"{st.lon:.5f}"))
            self.table.setItem(r, 3, QTableWidgetItem(f"{st.lat:.5f}"))
            self.table.setItem(r, 4, QTableWidgetItem(f"{st.elev:.1f}"))
            self.table.setItem(
                r, 5, QTableWidgetItem(f"{st.period_min_s:.3g} .. {st.period_max_s:.3g}"))
        self.table.blockSignals(False)

    def _on_item_changed(self, item):
        if item.column() != 0:
            return
        st = self._stations[item.row()]
        excl = set(self.project.survey.excluded)
        if item.checkState() == Qt.Checked:
            excl.discard(st.station_id)
        else:
            excl.add(st.station_id)
        self.project.survey.excluded = sorted(excl)
        self._recompute_summary()
        self._draw_maps()
        self.notify_changed()

    # ---- local frame ------------------------------------------------- #
    def _current_proj(self):
        """``(label, kind, epsg)`` for the selected projection preset."""
        return _PROJ_PRESETS[self._preset_idx()][:3]

    def _current_local_proj(self) -> str:
        return _PROJ_PRESETS[self._preset_idx()][3] or "tmerc"

    def _preset_idx(self) -> int:
        """Index into _PROJ_PRESETS of the selected row (the row order varies)."""
        i = self.proj_combo.currentData()
        return int(i) if i is not None else 0

    def _select_preset(self, idx: int):
        row = self.proj_combo.findData(idx)
        if row >= 0:
            self.proj_combo.setCurrentIndex(row)

    def _resort_proj_combo(self):
        """UTM first, Custom EPSG last, the rest by ascending worst-case scale
        error (unevaluable ones after the evaluable).  Keeps the selection."""
        res = getattr(self, "_dist_results", {})
        utm = next(i for i, pr in enumerate(_PROJ_PRESETS) if pr[1] == "utm")
        custom = next(i for i, pr in enumerate(_PROJ_PRESETS)
                      if pr[1] == "epsg" and pr[2] is None)
        middle = [i for i in range(len(_PROJ_PRESETS)) if i not in (utm, custom)]
        middle.sort(key=lambda i: (res.get(i) is None,
                                   res[i]["max_err_pct"] if res.get(i) else 0.0, i))
        keep = self._preset_idx()
        self.proj_combo.blockSignals(True)
        self.proj_combo.clear()
        for i in [utm] + middle + [custom]:
            d = res.get(i)
            suffix = f"   [max scale error {d['max_err_pct']:.2f}%]" if d else ""
            self.proj_combo.addItem(_PROJ_PRESETS[i][0] + suffix, i)
        self.proj_combo.setCurrentIndex(self.proj_combo.findData(keep))
        self.proj_combo.blockSignals(False)

    def _on_proj_kind_changed(self):
        _label, kind, epsg = self._current_proj()
        is_utm = kind == "utm"
        is_custom = kind == "epsg" and epsg is None
        self.zone_edit.setEnabled(is_utm)
        self.hemi_check.setEnabled(is_utm)
        self.epsg_edit.setEnabled(kind != "local")
        self.epsg_edit.setReadOnly(not is_custom)
        if kind == "local":
            self.epsg_edit.setText("")
        elif kind == "epsg" and epsg is not None:
            self.epsg_edit.setText(str(epsg))
        elif is_utm:
            try:
                z = int(self.zone_edit.text())
                self.epsg_edit.setText(
                    str((32600 if self.hemi_check.isChecked() else 32700) + z))
            except ValueError:
                pass
        self._update_crs_label()
        self._refresh_distortion()

    def _update_crs_label(self):
        _label, kind, _epsg = self._current_proj()
        if kind == "local":
            self.crs_lbl.setText(
                _LOCAL_DESCRIPTIONS.get(self._current_local_proj(), "")
                + " Model origin (0,0) at the survey centre.")
            return
        txt = self.epsg_edit.text().strip()
        if not txt.isdigit():
            self.crs_lbl.setText("")
            return
        from ..backends import projection as P
        self.crs_lbl.setText(P.epsg_name(int(txt))
                             + " — model = this grid minus the origin.")

    # ---- distortion --------------------------------------------------- #
    def _dist_points(self):
        excl = set(self.project.survey.excluded)
        used = [st for st in self._stations if st.station_id not in excl]
        return [st.lon for st in used], [st.lat for st in used]

    def _frame_for_preset(self, idx: int):
        """LocalFrame for preset *idx* using the origin currently in the widgets
        (so the comparison reflects what Apply would do), or None if the preset
        can't be evaluated (bad EPSG / zone / origin)."""
        from ..backends import projection as P

        _label, kind, epsg, local_proj = _PROJ_PRESETS[idx]
        s = self.project.survey
        try:
            olon = float(self.origin_lon.text())
            olat = float(self.origin_lat.text())
        except ValueError:
            olon, olat = 0.5 * (s.min_lon + s.max_lon), 0.5 * (s.min_lat + s.max_lat)
        l1, l2 = P.standard_parallels(s.min_lat, s.max_lat)
        try:
            if kind == "utm":
                try:
                    z = int(self.zone_edit.text())
                except ValueError:
                    z = edi_io.suggest_utm(olon, olat)[0]
                epsg = (32600 if self.hemi_check.isChecked() else 32700) + z
            elif kind == "epsg" and epsg is None:
                if self._preset_idx() != idx:       # only meaningful once chosen + typed
                    return None
                epsg = int(self.epsg_edit.text())
            return P.make_local_frame(kind, epsg or 0, olon, olat,
                                      local_proj or "tmerc", l1, l2)
        except Exception:  # noqa: BLE001
            return None

    def _refresh_distortion(self):
        """Per-option worst-case scale error in the dropdown, plus the full
        scale / rotation readout for the selected option."""
        if not hasattr(self, "dist_lbl"):
            return
        if not self._stations:
            self.dist_lbl.setText("")
            return
        from ..backends import projection as P

        lons, lats = self._dist_points()
        if not lons:
            self.dist_lbl.setText("")
            return
        sv = self.project.survey
        key = (len(lons), self.origin_lon.text(), self.origin_lat.text(),
               self.zone_edit.text(), self.hemi_check.isChecked(),
               (self._preset_idx(), self.epsg_edit.text())
               if self._current_proj()[2] is None else "",
               round(sv.min_lat, 4), round(sv.max_lat, 4))
        if key != getattr(self, "_dist_key", None):
            self._dist_key = key
            self._dist_results = {}
            for i in range(len(_PROJ_PRESETS)):
                fr = self._frame_for_preset(i)
                self._dist_results[i] = P.distortion(fr, lons, lats) if fr else None
            self._resort_proj_combo()
        d = self._dist_results.get(self._preset_idx())
        if not d:
            self.dist_lbl.setText("(cannot evaluate this projection yet)")
            return
        e = d["max_err_pct"]
        color = "#2e7d32" if e < 0.1 else "#b26a00" if e < 0.5 else "#c62828"
        self.dist_lbl.setText(
            f"<span style='color:{color}'><b>worst-case scale error {e:.2f}%</b> "
            f"({d['m_per_km']:.1f} m per km)</span> over {len(lons)} stations "
            f"(scale {d['scale_min']:.4f}–{d['scale_max']:.4f}); "
            f"shape (angular) distortion {d['angular_deg']:.2f}°"
            + (" — conformal" if d["angular_deg"] < 0.01 else " — NOT conformal")
            + f"; grid-vs-true-north rotation {d['conv_min']:+.1f}° to {d['conv_max']:+.1f}°")

    def _pick_lowest_distortion(self):
        if not self._stations:
            self.log("load EDIs first")
            return
        self._dist_key = None                      # force a fresh evaluation
        self._refresh_distortion()
        ranked = sorted((d["max_err_pct"], i)
                        for i, d in self._dist_results.items() if d)
        if not ranked:
            self.log("could not evaluate any projection")
            return
        err, best = ranked[0]
        self._select_preset(best)
        self.log("lowest distortion: " + _PROJ_PRESETS[best][0]
                 + f" (worst-case scale error {err:.2f}%)")
        self._apply_frame()

    def _set_proj_combo_from(self, p):
        idx = 0
        if p.proj_kind == "local":
            idx = next((i for i, pr in enumerate(_PROJ_PRESETS)
                        if pr[1] == "local" and pr[3] == (p.local_proj or "tmerc")), 0)
        elif p.proj_kind == "utm":
            idx = next(i for i, pr in enumerate(_PROJ_PRESETS) if pr[1] == "utm")
        elif p.proj_kind == "epsg" and p.epsg:
            match = next((i for i, pr in enumerate(_PROJ_PRESETS)
                          if pr[1] == "epsg" and pr[2] == p.epsg), None)
            idx = match if match is not None else len(_PROJ_PRESETS) - 1   # -> Custom
        self.proj_combo.blockSignals(True)
        self._select_preset(idx)
        self.proj_combo.blockSignals(False)
        self._on_proj_kind_changed()

    def _suggest_frame_clicked(self):
        # explicit button press: also clears the lock
        self.origin_lock.blockSignals(True)
        self.origin_lock.setChecked(False)
        self.origin_lock.blockSignals(False)
        self._suggest_frame()
        self._write_frame_to_project()
        self._refresh_distortion()
        if self.map_tabs.currentIndex() == 1:
            self._draw_map_projected()
        self.notify_changed()

    def _suggest_frame(self):
        """Fill the frame widgets from the survey.  Re-centres the origin on the
        survey bbox centre unless the user has locked it."""
        s = self.project.survey
        clon = 0.5 * (s.min_lon + s.max_lon)
        clat = 0.5 * (s.min_lat + s.max_lat)
        zone, north, epsg = edi_io.suggest_utm(clon, clat)
        self.zone_edit.setText(str(zone))
        self.hemi_check.setChecked(north)
        if self._current_proj()[1] == "utm":
            self.epsg_edit.setText(str(epsg))
        self._update_crs_label()
        if not self.origin_lock.isChecked() or not self.origin_lon.text():
            self.origin_lon.setText(f"{clon:.6f}")
            self.origin_lat.setText(f"{clat:.6f}")

    def _write_frame_to_project(self) -> bool:
        """Widgets -> project.projection.  No logging / notify (callable from
        the silent load path).  Returns whether the values parsed."""
        from ..backends import projection as P

        p = self.project.projection
        _label, kind, preset_epsg = self._current_proj()
        try:
            if kind == "utm":
                p.utm_zone = int(self.zone_edit.text())
                p.northern_hemisphere = self.hemi_check.isChecked()
                p.epsg = (32600 if p.northern_hemisphere else 32700) + p.utm_zone
            elif kind == "epsg":
                epsg = preset_epsg if preset_epsg is not None else int(self.epsg_edit.text())
                if epsg <= 0:
                    raise ValueError("EPSG code must be a positive integer")
                p.epsg = epsg
            else:                                    # local
                p.epsg = 0
                p.local_proj = self._current_local_proj()
            p.proj_kind = kind
            p.origin_lon = float(self.origin_lon.text())
            p.origin_lat = float(self.origin_lat.text())
            p.origin_locked = self.origin_lock.isChecked()
            s = self.project.survey
            p.local_lat1, p.local_lat2 = P.standard_parallels(s.min_lat, s.max_lat)
        except ValueError as exc:
            self.log(f"bad frame value: {exc}")
            return False
        if kind != "local":
            self.epsg_edit.setText(str(p.epsg))
        self._update_crs_label()
        return True

    def _apply_frame(self):
        if not self._write_frame_to_project():
            return
        from ..backends import projection as P

        p = self.project.projection
        self.log("model frame set: " + P.frame_label(P.frame_from_projection(p))
                 + ("  [origin locked]" if p.origin_locked else ""))
        self._refresh_distortion()
        if self.map_tabs.currentIndex() == 1:
            self._draw_map_projected()
        self.notify_changed()

    # ---- map ------------------------------------------------------------ #
    def _on_map_tab(self, idx: int):
        if idx == 1:
            self._draw_map_projected()

    def _draw_maps(self):
        """Redraw the geographic map always; the projected one only when its
        tab is showing (it re-draws itself on tab switch otherwise)."""
        self._draw_map()
        if self.map_tabs.currentIndex() == 1:
            self._draw_map_projected()

    def _draw_map(self):
        if getattr(self.map, "ax", None) is None:
            return
        self.map.clear()
        ax = self.map.ax
        excl = set(self.project.survey.excluded)
        used = [(st.lon, st.lat) for st in self._stations if st.station_id not in excl]
        drop = [(st.lon, st.lat) for st in self._stations if st.station_id in excl]
        if used:
            ax.scatter([x for x, _ in used], [y for _, y in used],
                       s=18, c="tab:blue", label="in use")
        if drop:
            ax.scatter([x for x, _ in drop], [y for _, y in drop],
                       s=18, c="0.7", marker="x", label="excluded")
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title("stations")
        if used or drop:
            ax.legend(loc="best", fontsize=8)
        self.map.redraw()

    def _draw_map_projected(self):
        if getattr(self.map_proj, "ax", None) is None:
            return
        self.map_proj.clear()
        ax = self.map_proj.ax
        p = self.project.projection

        def _msg(text, color="gray"):
            ax.text(0.5, 0.5, text, ha="center", va="center",
                    transform=ax.transAxes, color=color, fontsize=9, wrap=True)
            self.map_proj.redraw()

        kind = p.proj_kind or "local"
        if not (p.origin_lon or p.origin_lat) or (kind != "local" and not p.epsg):
            return _msg("set a local frame below, then Apply")
        if not self._stations:
            return _msg("load EDIs to see the projected layout")
        try:
            from ..backends import projection as P
        except Exception:   # noqa: BLE001 -- pyproj not installed yet
            return _msg("install pyproj to preview the projected layout")

        excl = set(self.project.survey.excluded)
        used = [st for st in self._stations if st.station_id not in excl]
        drop = [st for st in self._stations if st.station_id in excl]
        try:
            frame = P.frame_from_projection(p)
            if used:
                ux, uy = P.project_lonlat([s.lon for s in used], [s.lat for s in used], frame)
                ax.scatter(ux / 1e3, uy / 1e3, s=18, c="tab:blue", label="in use")
            if drop:
                dx, dy = P.project_lonlat([s.lon for s in drop], [s.lat for s in drop], frame)
                ax.scatter(dx / 1e3, dy / 1e3, s=18, c="0.7", marker="x", label="excluded")
        except Exception as exc:   # noqa: BLE001 -- bad EPSG, transform blows up
            self.map_proj.clear()
            return _msg(f"projection failed:\n{exc}", color="crimson")

        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.axvline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_xlabel("easting - origin (km)")
        ax.set_ylabel("northing - origin (km)")
        ax.set_aspect("equal", adjustable="datalim")
        import textwrap
        ax.set_title(textwrap.fill(P.frame_label(frame), 58), fontsize=8)
        if used or drop:
            ax.legend(loc="best", fontsize=8)
        self.map_proj.redraw()

    # ------------------------------------------------------------------ #
    def refresh(self) -> None:
        p = self.project.projection
        self.origin_lock.blockSignals(True)
        self.origin_lock.setChecked(bool(p.origin_locked))
        self.origin_lock.blockSignals(False)
        self._set_proj_combo_from(p)
        self.dir_edit.setText(self.project.survey.edi_dir)
        if self.project.survey.edi_dir and os.path.exists(self.project.survey.edi_dir):
            self._load_from(self.project.survey.edi_dir)   # silent -- see _load_from's docstring
        if p.utm_zone:
            self.zone_edit.setText(str(p.utm_zone))
            self.hemi_check.setChecked(p.northern_hemisphere)
        if p.epsg:
            self.epsg_edit.setText(str(p.epsg))
        if p.origin_lon or p.origin_lat:
            self.origin_lon.setText(f"{p.origin_lon:.6f}")
            self.origin_lat.setText(f"{p.origin_lat:.6f}")
        self._update_crs_label()
        self._refresh_distortion()
        if self.map_tabs.currentIndex() == 1:
            self._draw_map_projected()
