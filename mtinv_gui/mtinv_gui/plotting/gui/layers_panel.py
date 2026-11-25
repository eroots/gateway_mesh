"""Tab 2 -- other datasets: rasters, depth surfaces, vector layers, CSV points.

Each layer says how it is read, how it shows on maps (plan views, the
section index map) and on sections.  The plan-view and section tabs draw every
enabled layer per these settings.  The editor shows only the settings that
apply to the selected layer's kind; the preview draws the layer alone over the
stations to check it lines up.
"""
from __future__ import annotations

import copy
import os
import re

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from ...gui.worker import run_async
from ..backends import layers as L
from ..backends import render
from ..project import LayerInfo
from .widgets import Debounce, FigurePreview, PathRow, PlotPanel, float_edit, read_float, scroll_column

_ALL = ("raster", "surface", "vector", "points")
_GRID = ("raster", "surface")
_FILE_FILTER = ("all supported (*.tif *.tiff *.nc *.grd *.npz *.asc *.shp *.gpkg *.kml *.kmz "
                "*.geojson *.json *.csv *.txt *.dat *.xyz);;rasters (*.tif *.tiff *.nc *.grd *.npz *.asc);;"
                "vectors (*.shp *.gpkg *.kml *.kmz *.geojson *.json);;tables (*.csv *.txt *.dat *.xyz);;"
                "all files (*)")


class _Field:
    """One editor row: widget + get/set + the layer kinds it applies to."""

    def __init__(self, attr, label, widget, kinds, get, set_, signal):
        self.attr, self.label, self.widget, self.kinds = attr, QLabel(label), widget, kinds
        self.get, self.set = get, set_
        self.signal = signal

    def show_for(self, kind: str) -> None:
        on = kind in self.kinds
        self.label.setVisible(on)
        self.widget.setVisible(on)


def _combo(items=(), editable=False):
    w = QComboBox()
    w.setEditable(editable)
    for it in items:
        if isinstance(it, tuple):
            w.addItem(it[1], it[0])
        else:
            w.addItem(it, it)
    return w


def _combo_get(w):
    if w.isEditable():
        return w.currentText().strip()
    return w.currentData()


def _combo_set(w, v):
    if w.isEditable():
        w.setCurrentText(str(v or ""))
    else:
        w.setCurrentIndex(max(w.findData(v), 0))


class LayersPanel(PlotPanel):
    title = "2 - Layers"

    def build(self) -> None:
        self._loading = False
        self._current = -1
        self._gen = 0
        self._changed_later = Debounce(self._commit, 400, self)

        root = QHBoxLayout(self)
        col = QWidget()
        left = QVBoxLayout(col)

        box = QGroupBox("Layers (drawn in this order, top = first)")
        v = QVBoxLayout(box)
        self.list = QListWidget()
        self.list.setMaximumHeight(150)
        self.list.currentRowChanged.connect(self._select)
        self.list.itemChanged.connect(self._item_checked)
        v.addWidget(self.list)
        row = QHBoxLayout()
        for text, slot in (("Add files...", self._add_files), ("Add .gdb...", self._add_folder),
                           ("Duplicate", self._duplicate), ("Remove", self._remove),
                           ("▲", lambda: self._move(-1)), ("▼", lambda: self._move(1))):
            b = QPushButton(text)
            if len(text) == 1:
                b.setMaximumWidth(30)
            b.clicked.connect(slot)
            row.addWidget(b)
        v.addLayout(row)
        left.addWidget(box)

        self.fields: list[_Field] = []
        self.editor = QWidget()
        ev = QVBoxLayout(self.editor)
        ev.setContentsMargins(0, 0, 0, 0)
        self.forms = {}
        for key, title in (("read", "Reading"), ("map", "On maps"), ("section", "On sections")):
            g = QGroupBox(title)
            self.forms[key] = QFormLayout(g)
            ev.addWidget(g)
        self._build_fields()
        left.addWidget(self.editor)
        self.info = QLabel("")
        self.info.setWordWrap(True)
        self.info.setStyleSheet("color: gray;")
        left.addWidget(self.info)
        left.addStretch(1)
        root.addWidget(scroll_column(col))

        self.preview = FigurePreview()
        self.preview.set_size_inches(7.0, 6.0)
        root.addWidget(self.preview, 1)

    # ------------------------------------------------------------------ #
    def _add(self, group, attr, label, widget, kinds, get=None, set_=None, signal=None):
        if get is None:
            if isinstance(widget, QComboBox):
                get, set_ = _combo_get, _combo_set
                signal = widget.currentTextChanged
            elif isinstance(widget, QCheckBox):
                get, set_, signal = (lambda w: w.isChecked()), (lambda w, v: w.setChecked(bool(v))), widget.toggled
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                get, set_, signal = (lambda w: w.value()), (lambda w, v: w.setValue(v)), widget.valueChanged
            elif isinstance(widget, PathRow):
                get, set_, signal = (lambda w: w.text()), (lambda w, v: w.setText(v)), widget.changed
            else:                                          # QLineEdit
                get, set_, signal = (lambda w: w.text().strip()), (lambda w, v: w.setText(str(v))), \
                    widget.editingFinished
        f = _Field(attr, label, widget, kinds, get, set_, signal)
        self.forms[group].addRow(f.label, widget)
        signal.connect(lambda *_a: self._edited(attr))
        self.fields.append(f)
        return widget

    def _build_fields(self):
        num = lambda w: (lambda ww: read_float(ww, 0.0))      # noqa: E731
        fset = lambda ww, v: ww.setText(f"{v:g}")              # noqa: E731
        km_get = lambda w: w.value() * 1000.0                  # noqa: E731
        km_set = lambda w, v: w.setValue(float(v) / 1000.0)    # noqa: E731

        self.name_edit = self._add("read", "name", "name", QLineEdit(), _ALL)
        self.kind_combo = self._add("read", "kind", "kind", _combo(list(L.KINDS.items())), _ALL)
        self._add("read", "path", "file", PathRow("file", kinds=("file",), file_filter=_FILE_FILTER), _ALL)
        crs = QLineEdit()
        crs.setPlaceholderText("from the file")
        crs.setToolTip("Override the CRS, e.g. EPSG:3578. Tables: blank = lon/lat if the values "
                       "fit, else the display CRS.")
        self._add("read", "crs", "CRS", crs, _ALL)
        self._add("read", "band", "band", QSpinBox(minimum=1, maximum=1000), _GRID)
        self.sublayer = self._add("read", "sublayer", "layer in file", _combo(editable=True), ("vector",))
        for attr, label, tip in (("valid_min", "valid min", "raw values below this are nodata"),
                                 ("valid_max", "valid max", "raw values above this are nodata")):
            e = float_edit(0.0, 140)
            e.setToolTip(tip)
            self._add("read", attr, label, e, ("raster", "surface", "points"), num(e), fset,
                      e.editingFinished)
        e = float_edit(1.0, 140)
        e.setToolTip("Multiply values after the validity test (e.g. to change a raster's units).\n"
                     "For depths in km, set 'depth units' instead -- that keeps km on maps.")
        self._add("read", "scale", "scale", e, ("raster", "surface", "points"), num(e), fset,
                  e.editingFinished)
        self._add("read", "units", "units", QLineEdit(), ("raster", "surface", "points"))
        self._add("read", "surface_ref", "values are", _combo(list(L.SURFACE_REFS.items())),
                  ("surface", "points"))
        du = _combo(list(L.DEPTH_UNITS.items()))
        du.setToolTip("Unit of the depths (depth surface values / the CSV depth column).\n"
                      "Maps and profiles show the layer in this unit; sections convert to metres.\n"
                      "Auto: below ~300 -> km, else m (the preview says which it picked).")
        self._add("read", "depth_unit", "depth units", du, ("surface", "points"))
        self.col_widgets = {}
        for attr, label in (("x_col", "x / lon column"), ("y_col", "y / lat column"),
                            ("value_col", "value column"), ("label_col", "label column"),
                            ("depth_col", "depth column")):
            w = _combo(editable=True)
            w.lineEdit().setPlaceholderText("guess" if attr in ("x_col", "y_col") else "none")
            self.col_widgets[attr] = self._add("read", attr, label, w, ("points",))
        self.label_field = self._add("read", "label_field", "label field", _combo(editable=True),
                                     ("vector",))

        self.map_mode = self._add("map", "map_mode", "show as", _combo(["none", "image", "contours", "show"]), _ALL)
        cm = _combo(["turbo", "viridis", "RdBu_r", "Spectral_r", "gray", "terrain", "cmc.vik",
                     "cmc.batlow"], editable=True)
        self._add("map", "cmap", "colour map", cm, ("raster", "surface", "points"))
        self._add("map", "auto_range", "auto range (2-98 %)", QCheckBox(), ("raster", "surface", "points"))
        for attr in ("vmin", "vmax"):
            e = float_edit(0.0, 140)
            self._add("map", attr, attr, e, ("raster", "surface", "points"), num(e), fset, e.editingFinished)
        self._add("map", "alpha", "opacity", QDoubleSpinBox(decimals=2, minimum=0.0, maximum=1.0,
                                                            singleStep=0.1), _ALL)
        e = float_edit(0.0, 140)
        e.setToolTip("0 = about 10 levels over the range")
        self._add("map", "contour_interval", "contour interval", e, _GRID, num(e), fset, e.editingFinished)
        self._add("map", "contour_labels", "label contours", QCheckBox(), _GRID)
        self._add("map", "color", "colour", _combo(["black", "white", "0.3", "tab:orange", "#7c3aed",
                                                     "#db2777", "tab:blue"], editable=True), _ALL)
        self._add("map", "linewidth", "line width", QDoubleSpinBox(decimals=2, minimum=0.1, maximum=10.0,
                                                                   singleStep=0.2),
                  ("raster", "surface", "vector"))
        self._add("map", "marker", "marker", _combo(["o", ".", "v", "^", "s", "D", "*", "+", "x", "|"]),
                  ("vector", "points"))
        self._add("map", "size", "marker size", QDoubleSpinBox(decimals=1, minimum=0.5, maximum=500.0),
                  ("vector", "points"))
        self._add("map", "label_size", "label size", QDoubleSpinBox(decimals=1, minimum=2.0, maximum=30.0,
                                                                    suffix=" pt"),
                  ("raster", "surface", "vector", "points"))

        self.section_mode = self._add("section", "section_mode", "show as",
                                      _combo(["none", "profile", "line", "markers"]), _ALL)
        self._add("section", "profile_axis", "profile axis",
                  _combo([("auto", "auto (left, then right)"), ("left", "left"), ("right", "right")]),
                  _GRID)
        off = QDoubleSpinBox(decimals=1, minimum=0.0, maximum=1000.0, suffix=" km")
        off.setToolTip("Points within this distance of a section are projected onto it; lines "
                       "and polygon edges are marked where they cross it")
        self._add("section", "section_offset_m", "max distance", off, ("vector", "points"),
                  km_get, km_set, off.valueChanged)

    # ------------------------------------------------------------------ #
    @property
    def layer(self) -> LayerInfo | None:
        ls = self.project.layers
        return ls[self._current] if 0 <= self._current < len(ls) else None

    def refresh(self) -> None:
        self._fill_list(self._current if self._current >= 0 else 0)

    def on_shown(self) -> None:
        self._preview()

    def _store(self) -> None:
        lay = self.layer
        if lay is None or self._loading:
            return
        for f in self.fields:
            if f.attr == "name":
                continue
            try:
                setattr(lay, f.attr, f.get(f.widget))
            except (TypeError, ValueError):
                pass
        new = self.name_edit.text().strip()
        if new and new != lay.name:
            lay.name = self._unique(new, exclude=lay)

    def _fill_list(self, select: int) -> None:
        self._loading = True
        self.list.blockSignals(True)
        self.list.clear()
        for lay in self.project.layers:
            it = QListWidgetItem(self._item_text(lay))
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if lay.enabled else Qt.Unchecked)
            self.list.addItem(it)
        self.list.blockSignals(False)
        n = len(self.project.layers)
        self._current = min(max(select, 0), n - 1) if n else -1
        if n:
            self.list.blockSignals(True)
            self.list.setCurrentRow(self._current)
            self.list.blockSignals(False)
        self._loading = False
        self._load_layer()

    @staticmethod
    def _item_text(lay) -> str:
        return f"{lay.name}  –  {L.KINDS.get(lay.kind, lay.kind)}"

    def _select(self, row: int) -> None:
        if self._loading or row == self._current:
            return
        self._store()
        self._current = row
        self._load_layer()

    def _item_checked(self, item) -> None:
        row = self.list.row(item)
        if 0 <= row < len(self.project.layers):
            self.project.layers[row].enabled = item.checkState() == Qt.Checked
            self._changed_later.poke()

    def _load_layer(self) -> None:
        lay = self.layer
        self.editor.setEnabled(lay is not None)
        if lay is None:
            self.info.setText("Add rasters (GeoTIFF / NetCDF / npz), vector files (shp / gpkg / "
                              "kml) or tables (CSV) with 'Add files...'.")
            self.preview.figure.clear()
            self.preview.draw()
            return
        self._loading = True
        if lay.kind == "vector":
            self._fill_combo(self.sublayer, L.list_sublayers(lay.path), lay.sublayer)
            self._fill_combo(self.label_field, [""] + L.vector_fields(lay.path, lay.sublayer),
                             lay.label_field)
        elif lay.kind == "points":
            cols = [""] + L.table_columns(lay.path)
            for attr, w in self.col_widgets.items():
                self._fill_combo(w, cols, getattr(lay, attr))
        for f in self.fields:
            try:
                f.set(f.widget, getattr(lay, f.attr))
            except (TypeError, ValueError):
                pass
            f.show_for(lay.kind)
        self._sync_mode_combos(lay)
        self._loading = False
        self._preview()

    @staticmethod
    def _fill_combo(w, items, current):
        w.blockSignals(True)
        w.clear()
        w.addItems([str(i) for i in items])
        w.setCurrentText(str(current or ""))
        w.blockSignals(False)

    def _sync_mode_combos(self, lay) -> None:
        """Offer only the map / section modes this kind supports."""
        for w, modes, attr in ((self.map_mode, L.MAP_MODES[lay.kind], "map_mode"),
                               (self.section_mode, L.SECTION_MODES[lay.kind], "section_mode")):
            w.blockSignals(True)
            w.clear()
            for m in modes:
                w.addItem(m, m)
            cur = getattr(lay, attr)
            if cur not in modes:
                cur = modes[1] if len(modes) > 1 else modes[0]
                setattr(lay, attr, cur)
            w.setCurrentIndex(w.findData(cur))
            w.blockSignals(False)

    # ------------------------------------------------------------------ #
    def _edited(self, attr: str) -> None:
        if self._loading or self.layer is None:
            return
        self._store()
        lay = self.layer
        if attr == "kind":
            self._loading = True
            for f in self.fields:
                f.show_for(lay.kind)
            self._sync_mode_combos(lay)
            self._loading = False
        if attr in ("path", "sublayer", "kind"):
            self._load_layer()
        item = self.list.item(self._current)
        if item is not None:
            item.setText(self._item_text(lay))
        self._changed_later.poke()

    def _commit(self) -> None:
        """Settled edit: redraw the preview and let the figure tabs redraw."""
        self._preview()
        self.notify_changed()

    def _unique(self, name: str, exclude=None) -> str:
        taken = {l.name for l in self.project.layers if l is not exclude}
        base, i, out = name, 2, name
        while out in taken:
            out = f"{base} ({i})"
            i += 1
        return out

    def _new_layer(self, path: str) -> LayerInfo:
        kind = L.guess_kind(path)
        name = os.path.splitext(os.path.basename(path.rstrip("/\\")))[0]
        name = re.sub(r"[_\-. ]?EPSG[_:]?\d+", "", name, flags=re.I).strip("_-. ") or name
        lay = LayerInfo(name=self._unique(name), kind=kind, path=path)
        if kind == "vector":
            lay.map_mode, lay.section_mode, lay.color = "show", "markers", "#7c3aed"
        elif kind == "points":
            lay.map_mode, lay.section_mode, lay.color, lay.size = "show", "none", "#db2777", 8.0
        else:
            low = name.lower()
            if any(k in low for k in ("curie", "cpd", "lab", "moho", "basement", "depth")):
                lay.kind, lay.section_mode = "surface", "line"
        return lay

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Add layers", self.project.workdir, _FILE_FILTER)
        self._append([self._new_layer(p) for p in paths])

    def _add_folder(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "File geodatabase (.gdb) or folder", self.project.workdir)
        if p:
            self._append([self._new_layer(p)])

    def _append(self, new) -> None:
        if not new:
            return
        self._store()
        self.project.layers.extend(new)
        self._fill_list(len(self.project.layers) - 1)
        for lay in new:
            self.log(f"layer '{lay.name}' ({lay.kind}) from {lay.path}")
        self._changed_later.poke()

    def _duplicate(self) -> None:
        if self.layer is not None:
            self._store()
            dup = copy.deepcopy(self.layer)
            dup.name = self._unique(dup.name)
            self._append([dup])

    def _remove(self) -> None:
        if self.layer is None:
            return
        del self.project.layers[self._current]
        self._fill_list(max(self._current - 1, 0))
        self._changed_later.poke()

    def _move(self, step: int) -> None:
        i, ls = self._current, self.project.layers
        j = i + step
        if self.layer is None or not 0 <= j < len(ls):
            return
        self._store()
        ls[i], ls[j] = ls[j], ls[i]
        self._fill_list(j)
        self._changed_later.poke()

    # ------------------------------------------------------------------ #
    def _preview(self) -> None:
        lay = self.layer
        fig = self.preview.figure
        if lay is None or not lay.path:
            return
        if not self.session.ready:
            fig.clear()
            fig.text(0.5, 0.5, "load + georeference a model on tab 1 to preview layers",
                     ha="center", va="center", color="0.4")
            self.preview.draw()
            return
        self._gen += 1
        gen = self._gen
        snap = copy.deepcopy(lay)
        self.info.setText("loading...")
        run_async(self, lambda: self.session.layer(snap),
                  lambda loaded: self._loaded(gen, snap, loaded),
                  lambda exc: self._failed(gen, exc))

    def _loaded(self, gen, lay, loaded) -> None:
        if gen != self._gen:
            return
        try:
            summary = render.render_layer_preview(self.preview.figure, self.session, lay, loaded)
        except Exception as exc:  # noqa: BLE001
            self._failed(gen, exc)
            return
        self.info.setText(summary)
        self.preview.draw()

    def _failed(self, gen, exc) -> None:
        if gen != self._gen:
            return
        self.info.setText(f"<span style='color:#b91c1c'>could not read: {exc}</span>")
        fig = self.preview.figure
        fig.clear()
        fig.text(0.5, 0.5, f"could not read:\n{exc}", ha="center", va="center", color="#b91c1c",
                 wrap=True)
        self.preview.draw()
