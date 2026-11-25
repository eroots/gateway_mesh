"""Tab 6 -- figure production: house style, saved figure recipes, layouts,
batch export.

A recipe snapshots the settings of a plan view / section / index map (or holds
a multi-panel layout) when it is added; "Update from tabs" re-snapshots, "Open
in tabs" loads it back for editing.  The model and layers stay live, so
"Export all" after pointing tab 1 at a newer iteration redraws every figure --
the same thing ``python -m mtinv_gui.plotting.batch`` does.
"""
from __future__ import annotations

import copy
import os

from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QMessageBox, QProgressBar, QPushButton, QSpinBox, QTableWidget,
    QVBoxLayout, QWidget,
)

from ...gui.worker import run_async
from ..backends import figures as F
from ..backends import render
from ..project import LayoutPanel, copy_into
from .widgets import ColorBox, Debounce, FigurePreview, PathRow, PlotPanel, run_async_progress, scroll_column

_FONTS = ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans", "Calibri", "Times New Roman",
          "DejaVu Serif"]
_PANEL_KINDS = [("plan", "plan view"), ("section", "section"), ("index", "index map"), ("empty", "empty")]


class LayoutSizeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New layout")
        form = QFormLayout(self)
        self.rows = QSpinBox(minimum=1, maximum=8, value=2)
        self.cols = QSpinBox(minimum=1, maximum=6, value=2)
        self.fill = QComboBox()
        self.fill.addItem("plan views at the plan depth list", "plan")
        self.fill.addItem("the sections, in order", "section")
        self.fill.addItem("empty (choose per panel)", "empty")
        form.addRow("rows", self.rows)
        form.addRow("columns", self.cols)
        form.addRow("fill with", self.fill)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)


class FiguresPanel(PlotPanel):
    title = "6 - Figures"

    def build(self) -> None:
        self._loading = False
        self._current = -1
        self._gen = 0
        self._redraw = Debounce(self._draw, 300, self)
        root = QHBoxLayout(self)
        col = QWidget()
        left = QVBoxLayout(col)

        # ---- house style ------------------------------------------------- #
        box = QGroupBox("House style (all figures)")
        form = QFormLayout(box)
        self.h_on = QCheckBox("use it")
        self.h_on.setToolTip("On: every figure (all tabs, recipes, exports) uses this font, "
                             "width, line width and dpi.")
        self.h_font = QComboBox()
        self.h_font.setEditable(True)
        self.h_font.addItems(_FONTS)
        self.h_font.setToolTip("Fonts matplotlib doesn't have are looked for in the Windows / "
                               "system font folders")
        self.h_size = QDoubleSpinBox(decimals=1, minimum=4.0, maximum=24.0, suffix=" pt")
        self.h_lw = QDoubleSpinBox(decimals=2, minimum=0.1, maximum=3.0, singleStep=0.1, suffix=" pt")
        self.h_preset = QComboBox()
        self.h_preset.addItem("custom", 0.0)
        for label, mm in F.WIDTH_PRESETS_MM.items():
            self.h_preset.addItem(label, mm)
        self.h_width = QDoubleSpinBox(decimals=1, minimum=20.0, maximum=600.0, suffix=" mm")
        self.h_dpi = QSpinBox(minimum=72, maximum=2400, singleStep=100)
        self.h_text = QCheckBox("editable text in PDF / SVG (not outlines)")
        self.h_link = QCheckBox("one colour scale for every figure:")
        self.h_color = ColorBox("Shared colour scale")
        wrow = QHBoxLayout()
        wrow.addWidget(self.h_preset, 1)
        wrow.addWidget(self.h_width)
        form.addRow(self.h_on)
        form.addRow("font", self.h_font)
        form.addRow("font size", self.h_size)
        form.addRow("line width", self.h_lw)
        form.addRow("width", wrow)
        form.addRow("raster dpi", self.h_dpi)
        form.addRow(self.h_text)
        form.addRow(self.h_link)
        form.addRow(self.h_color)
        self.h_preset.currentIndexChanged.connect(self._preset)
        for w in (self.h_on, self.h_text, self.h_link):
            w.toggled.connect(self._house_changed)
        self.h_font.currentTextChanged.connect(self._house_changed)
        for w in (self.h_size, self.h_lw, self.h_width, self.h_dpi):
            w.editingFinished.connect(self._house_changed)
        self.h_color.changed.connect(self._house_changed)
        left.addWidget(box)

        # ---- recipes ----------------------------------------------------- #
        box = QGroupBox("Figure recipes")
        v = QVBoxLayout(box)
        self.list = QListWidget()
        self.list.setMaximumHeight(150)
        self.list.currentRowChanged.connect(self._select)
        v.addWidget(self.list)
        r1, r2 = QHBoxLayout(), QHBoxLayout()
        for text, slot, row in (("+ Plan view", self._add_plan, r1), ("+ Section", self._add_section, r1),
                                ("+ Index map", self._add_index, r1), ("+ Layout...", self._add_layout, r1),
                                ("Update from tabs", self._update_from_tabs, r2),
                                ("Open in tabs", self._open_in_tabs, r2),
                                ("Duplicate", self._duplicate, r2), ("Remove", self._remove, r2)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        v.addLayout(r1)
        v.addLayout(r2)
        self.r_box = QWidget()
        rf = QFormLayout(self.r_box)
        rf.setContentsMargins(0, 0, 0, 0)
        self.r_name = QLineEdit()
        self.r_name.editingFinished.connect(self._recipe_edited)
        self.r_kind = QLabel("")
        self.r_path = PathRow("file (relative = next to the project)", kinds=("save",),
                              file_filter="PDF (*.pdf);;PNG (*.png);;SVG (*.svg);;TIFF (*.tif)")
        self.r_path.changed.connect(self._recipe_edited)
        rf.addRow("name", self.r_name)
        rf.addRow("kind", self.r_kind)
        rf.addRow("file", self.r_path)
        v.addWidget(self.r_box)
        left.addWidget(box)

        # ---- layout editor ----------------------------------------------- #
        self.l_box = QGroupBox("Layout")
        lf = QFormLayout(self.l_box)
        self.l_rows = QSpinBox(minimum=1, maximum=8)
        self.l_cols = QSpinBox(minimum=1, maximum=6)
        grid = QHBoxLayout()
        grid.addWidget(self.l_rows)
        grid.addWidget(QLabel("rows ×"))
        grid.addWidget(self.l_cols)
        grid.addWidget(QLabel("columns"))
        for w in (self.l_rows, self.l_cols):
            w.valueChanged.connect(self._grid_changed)
        self.l_cbar = QComboBox()
        for key in ("bottom", "right", "none"):
            self.l_cbar.addItem({"none": "none (each panel its own)"}.get(key, key), key)
        self.l_labels = QComboBox()
        for st in F.LABEL_STYLES:
            self.l_labels.addItem(st, st)
        self.l_height = QDoubleSpinBox(decimals=2, minimum=0.0, maximum=40.0, singleStep=0.5, suffix=" in")
        self.l_height.setSpecialValueText("auto")
        self.l_width = QDoubleSpinBox(decimals=2, minimum=1.0, maximum=40.0, singleStep=0.5, suffix=" in")
        self.l_width.setToolTip("Ignored while the house style is on (its width wins)")
        size = QHBoxLayout()
        size.addWidget(self.l_width)
        size.addWidget(QLabel("×"))
        size.addWidget(self.l_height)
        self.l_space = QDoubleSpinBox(decimals=3, minimum=0.0, maximum=0.3, singleStep=0.01)
        self.l_share = QCheckBox("hide repeated axes (left column / bottom row only)")
        self.l_share.setToolTip("A panel drops its y-axis labels when the panel to its left shows\n"
                                "the same y-range, and its x-axis labels when the panel below\n"
                                "shows the same x-range (e.g. a grid of plan views).")
        self.l_share.toggled.connect(self._layout_edited)
        for w in (self.l_cbar, self.l_labels):
            w.currentIndexChanged.connect(self._layout_edited)
        for w in (self.l_height, self.l_width, self.l_space):
            w.editingFinished.connect(self._layout_edited)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["panel", "depth (km)", "main image", "section", "title"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(170)
        lf.addRow("grid", grid)
        lf.addRow("colour bar", self.l_cbar)
        lf.addRow("panel labels", self.l_labels)
        lf.addRow("size", size)
        lf.addRow("spacing", self.l_space)
        lf.addRow(self.l_share)
        lf.addRow(self.table)
        lf.addRow(QLabel("Plan panels use the layout's plan look, section panels its section "
                         "look ('Update from tabs' re-copies them from tabs 3 / 4)."))
        left.addWidget(self.l_box)

        # ---- export ------------------------------------------------------ #
        box = QGroupBox("Export")
        form = QFormLayout(box)
        self.out_dir = PathRow("blank = each recipe's own file", kinds=("folder",))
        one = QPushButton("Export selected")
        one.clicked.connect(lambda: self._export(selected=True))
        self.all_btn = QPushButton("Export all recipes")
        self.all_btn.clicked.connect(lambda: self._export(selected=False))
        brow = QHBoxLayout()
        brow.addWidget(one)
        brow.addWidget(self.all_btn)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.cli = QLineEdit()
        self.cli.setReadOnly(True)
        self.cli.setToolTip("Same export from a terminal (mtinv env, in the mtinv_gui folder); "
                            "add --model <newer.vtu> to redraw on another iteration")
        form.addRow("into folder", self.out_dir)
        form.addRow(brow)
        form.addRow(self.progress)
        form.addRow("command line", self.cli)
        left.addWidget(box)
        left.addStretch(1)
        root.addWidget(scroll_column(col, 470))

        self.preview = FigurePreview()
        root.addWidget(self.preview, 1)

    # ------------------------------------------------------------------ #
    @property
    def recipe(self):
        rs = self.project.recipes
        return rs[self._current] if 0 <= self._current < len(rs) else None

    def refresh(self) -> None:
        self._loading = True
        h = self.project.house
        self.h_on.setChecked(h.enabled)
        self.h_font.setCurrentText(h.font_family)
        self.h_size.setValue(h.font_size)
        self.h_lw.setValue(h.line_width)
        self.h_width.setValue(h.width_mm)
        i = self.h_preset.findData(h.width_mm)
        self.h_preset.setCurrentIndex(i if i > 0 else 0)
        self.h_dpi.setValue(int(h.dpi))
        self.h_text.setChecked(h.editable_text)
        self.h_link.setChecked(h.link_colors)
        self.h_color.load(h.color)
        self.h_color.setEnabled(h.link_colors)
        self._loading = False
        self._fill_list(self._current if self._current >= 0 else 0)
        self._update_cli()

    def on_shown(self) -> None:
        self._update_cli()
        self._redraw.poke()

    def _store(self) -> None:
        self._store_house()
        self._store_recipe()

    def _store_house(self) -> None:
        h = self.project.house
        h.enabled = self.h_on.isChecked()
        h.font_family = self.h_font.currentText().strip() or "DejaVu Sans"
        h.font_size = self.h_size.value()
        h.line_width = self.h_lw.value()
        h.width_mm = self.h_width.value()
        h.dpi = self.h_dpi.value()
        h.editable_text = self.h_text.isChecked()
        h.link_colors = self.h_link.isChecked()
        self.h_color.store(h.color)

    def _store_recipe(self) -> None:
        r = self.recipe
        if r is None or self._loading:
            return
        r.name = self.r_name.text().strip() or r.name
        r.out_path = self.r_path.text()
        if r.kind == "layout":
            lay = r.layout
            lay.rows, lay.cols = self.l_rows.value(), self.l_cols.value()
            lay.shared_colorbar = self.l_cbar.currentData()
            lay.panel_labels = self.l_labels.currentData()
            lay.height_in = self.l_height.value()
            lay.width_in = self.l_width.value()
            lay.wspace = lay.hspace = self.l_space.value()
            lay.hide_shared_axes = self.l_share.isChecked()
            lay.panels = [self._read_row(i) for i in range(self.table.rowCount())]

    # ---- house style ---------------------------------------------------- #
    def _preset(self, _i) -> None:
        mm = self.h_preset.currentData()
        if mm:
            self.h_width.setValue(mm)
            self._house_changed()

    def _house_changed(self, *_):
        if self._loading:
            return
        self._store_house()
        self.h_color.setEnabled(self.h_link.isChecked())
        h = self.project.house
        if h.enabled and not render.ensure_font(h.font_family):
            self.log(f"font '{h.font_family}' not found -- matplotlib will substitute "
                     f"(looked in {', '.join(d for d in render.FONT_DIRS if os.path.isdir(d))})")
        self.session.house = h
        self.notify_changed()
        self._redraw.poke()

    # ---- recipe list ---------------------------------------------------- #
    def _fill_list(self, select: int) -> None:
        self._loading = True
        self.list.blockSignals(True)
        self.list.clear()
        for r in self.project.recipes:
            self.list.addItem(f"{r.name}  –  {F.RECIPE_KINDS.get(r.kind, r.kind)}")
        self.list.blockSignals(False)
        n = len(self.project.recipes)
        self._current = min(max(select, 0), n - 1) if n else -1
        if n:
            self.list.blockSignals(True)
            self.list.setCurrentRow(self._current)
            self.list.blockSignals(False)
        self._loading = False
        self._load_recipe()

    def _select(self, row: int) -> None:
        if self._loading or row == self._current:
            return
        self._store_recipe()
        self._current = row
        self._load_recipe()

    def _load_recipe(self) -> None:
        r = self.recipe
        self.r_box.setEnabled(r is not None)
        self.l_box.setVisible(r is not None and r.kind == "layout")
        if r is None:
            self.r_name.setText("")
            self.r_kind.setText("add a figure with the + buttons")
            self.r_path.setText("")
            self._redraw.poke()
            return
        self._loading = True
        self.r_name.setText(r.name)
        self.r_kind.setText(F.RECIPE_KINDS.get(r.kind, r.kind))
        self.r_path.setText(r.out_path)
        if r.kind == "layout":
            lay = r.layout
            self.l_rows.setValue(int(lay.rows))
            self.l_cols.setValue(int(lay.cols))
            self.l_cbar.setCurrentIndex(max(self.l_cbar.findData(lay.shared_colorbar), 0))
            self.l_labels.setCurrentIndex(max(self.l_labels.findData(lay.panel_labels), 0))
            self.l_height.setValue(lay.height_in)
            self.l_width.setValue(lay.width_in)
            self.l_space.setValue(lay.wspace)
            self.l_share.setChecked(lay.hide_shared_axes)
            self._fill_table(lay)
        self._loading = False
        self._redraw.poke()

    def _recipe_edited(self, *_):
        if self._loading or self.recipe is None:
            return
        self._store_recipe()
        item = self.list.item(self._current)
        if item is not None:
            r = self.recipe
            item.setText(f"{r.name}  –  {F.RECIPE_KINDS.get(r.kind, r.kind)}")
        self._update_cli()

    def _append(self, recipe) -> None:
        self._store_recipe()
        names = {r.name for r in self.project.recipes}
        base, i = recipe.name, 2
        while recipe.name in names:
            recipe.name = f"{base} ({i})"
            i += 1
        if not recipe.out_path:
            recipe.out_path = os.path.join("figures", F._safe(recipe.name) + ".pdf")
        self.project.recipes.append(recipe)
        self._fill_list(len(self.project.recipes) - 1)
        self.log(f"figure recipe '{recipe.name}' added")

    def _sync_tabs(self) -> None:
        """Pull unsaved edits from the other tabs before snapshotting them."""
        win = self.window()
        sync = getattr(win, "_sync_widgets", None)
        if sync:
            sync()

    def _add_plan(self) -> None:
        self._sync_tabs()
        self._append(F.recipe_from_plan(self.project))

    def _add_section(self) -> None:
        self._sync_tabs()
        secs = self.project.sections
        if not secs:
            QMessageBox.information(self, "Figures", "Add a section on tab 4 first.")
            return
        i = min(max(self.project.current_section, 0), len(secs) - 1)
        self._append(F.recipe_from_section(self.project, secs[i]))

    def _add_index(self) -> None:
        self._sync_tabs()
        self._append(F.recipe_index_map(self.project))

    def _add_layout(self) -> None:
        self._sync_tabs()
        dlg = LayoutSizeDialog(self)
        if not dlg.exec_():
            return
        r = F.new_layout_recipe(self.project, dlg.rows.value(), dlg.cols.value())
        fill = dlg.fill.currentData()
        n = dlg.rows.value() * dlg.cols.value()
        if fill == "section":
            labels = [s.label for s in self.project.sections][:n]
            r.layout.panels = ([LayoutPanel(kind="section", section=l) for l in labels]
                               + [LayoutPanel(kind="empty") for _ in range(n - len(labels))])
        elif fill == "empty":
            r.layout.panels = [LayoutPanel(kind="empty") for _ in range(n)]
        self._append(r)

    def _update_from_tabs(self) -> None:
        r = self.recipe
        if r is None:
            return
        self._sync_tabs()
        p = self.project
        if r.kind == "plan":
            copy_into(r.plan, p.plan)
        elif r.kind == "section":
            secs = [s for s in p.sections if s.label == r.section.label] or p.sections[p.current_section:
                                                                                         p.current_section + 1]
            if secs:
                copy_into(r.section, secs[0])
            copy_into(r.section_style, p.section_style)
        elif r.kind == "index":
            copy_into(r.plan, p.plan)
            copy_into(r.section_style, p.section_style)
            r.section_group = p.section_group
        else:
            copy_into(r.layout.plan, p.plan)
            copy_into(r.layout.section_style, p.section_style)
        self.log(f"recipe '{r.name}' updated from the plan / section tabs")
        self._redraw.poke()

    def _open_in_tabs(self) -> None:
        r = self.recipe
        if r is None:
            return
        p = self.project
        if r.kind in ("plan", "index"):
            copy_into(p.plan, r.plan)
        if r.kind == "layout":
            copy_into(p.plan, r.layout.plan)
            copy_into(p.section_style, r.layout.section_style)
        if r.kind in ("section", "index"):
            copy_into(p.section_style, r.section_style)
        if r.kind == "index":
            p.section_group = r.section_group
        if r.kind == "section":
            for i, s in enumerate(p.sections):
                if s.label == r.section.label:
                    p.current_section = i
        self.log(f"recipe '{r.name}' loaded into the plan / section tabs -- edit there, then "
                 "'Update from tabs'")
        self.notify_changed()

    def _duplicate(self) -> None:
        if self.recipe is not None:
            self._store_recipe()
            dup = copy.deepcopy(self.recipe)
            dup.out_path = ""
            self._append(dup)

    def _remove(self) -> None:
        if self.recipe is None:
            return
        del self.project.recipes[self._current]
        self._fill_list(max(self._current - 1, 0))

    # ---- layout table --------------------------------------------------- #
    def _fill_table(self, lay) -> None:
        panels = F._panels(lay)
        self.table.setRowCount(len(panels))
        rasters = [""] + [l.name for l in self.project.layers if l.kind in ("raster", "surface")]
        labels = [s.label for s in self.project.sections]
        for i, p in enumerate(panels):
            r, c = divmod(i, max(int(lay.cols), 1))
            kind = QComboBox()
            for key, text in _PANEL_KINDS:
                kind.addItem(f"{r + 1},{c + 1}: {text}", key)
            kind.setCurrentIndex(max(kind.findData(p.kind), 0))
            depth = QDoubleSpinBox(decimals=2, minimum=-10.0, maximum=3000.0, singleStep=1.0)
            depth.setValue(p.depth_m / 1000.0)
            main = QComboBox()
            for name in rasters:
                main.addItem(name or "model", name)
            main.setCurrentIndex(max(main.findData(p.main_layer), 0))
            sec = QComboBox()
            sec.addItems(labels or [""])
            sec.setCurrentText(p.section)
            title = QLineEdit(p.title)
            title.setPlaceholderText("default")
            for col, w in enumerate((kind, depth, main, sec, title)):
                self.table.setCellWidget(i, col, w)
            kind.currentIndexChanged.connect(self._layout_edited)
            depth.editingFinished.connect(self._layout_edited)
            main.currentIndexChanged.connect(self._layout_edited)
            sec.currentIndexChanged.connect(self._layout_edited)
            title.editingFinished.connect(self._layout_edited)

    def _read_row(self, i: int) -> LayoutPanel:
        w = [self.table.cellWidget(i, c) for c in range(5)]
        return LayoutPanel(kind=w[0].currentData(), depth_m=w[1].value() * 1000.0,
                           main_layer=w[2].currentData() or "", section=w[3].currentText(),
                           title=w[4].text().strip())

    def _grid_changed(self, *_):
        if self._loading or self.recipe is None:
            return
        self._store_recipe()
        lay = self.recipe.layout
        lay.panels = F._panels(lay)
        self._loading = True
        self._fill_table(lay)
        self._loading = False
        self._redraw.poke()

    def _layout_edited(self, *_):
        if self._loading or self.recipe is None:
            return
        self._store_recipe()
        self._redraw.poke()

    # ---- preview + export ---------------------------------------------- #
    def _draw(self) -> None:
        fig = self.preview.figure
        r = self.recipe
        if r is None or not self.session.ready:
            fig.clear()
            fig.text(0.5, 0.5, "load + georeference a model on tab 1" if not self.session.ready
                     else "add a figure with the + buttons", ha="center", va="center", color="0.4")
            self.preview.draw()
            return
        self._store_recipe()
        self._gen += 1
        gen = self._gen
        snap = copy.deepcopy(r)
        run_async(self, lambda: F.warm(self.session, self.project, snap),
                  lambda size: self._warmed(gen, snap, size),
                  lambda exc: self._failed(gen, exc))

    def _warmed(self, gen, recipe, size) -> None:
        if gen != self._gen:
            return
        self.preview.set_size_inches(*size)
        try:
            F.render_recipe(self.preview.figure, self.session, self.project, recipe)
        except Exception as exc:  # noqa: BLE001
            self._failed(gen, exc)
            return
        self.preview.draw()

    def _failed(self, gen, exc) -> None:
        if gen != self._gen:
            return
        fig = self.preview.figure
        fig.clear()
        fig.text(0.5, 0.5, f"could not draw:\n{exc}", ha="center", va="center", color="#b91c1c",
                 wrap=True)
        self.preview.draw()
        self.log(f"figure preview failed: {exc}")

    def _update_cli(self) -> None:
        win = self.window()
        path = getattr(win, "_saved_path", None)
        if path:
            import shutil

            cmd = "mtplot-batch" if shutil.which("mtplot-batch") else "python -m mtinv_gui.plotting.batch"
            self.cli.setText(f'{cmd} "{path}"')
        else:
            self.cli.setText("(save the project first: Project > Save)")

    def _export(self, selected: bool) -> None:
        if not self.session.ready:
            return
        self._store()
        recipes = [self.recipe] if selected else list(self.project.recipes)
        recipes = [copy.deepcopy(r) for r in recipes if r is not None]
        if not recipes:
            return
        out = self.out_dir.text()
        project = copy.deepcopy(self.project)
        self.all_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(recipes))

        def done(results):
            self.all_btn.setEnabled(True)
            self.progress.setVisible(False)
            bad = [(n, e) for n, e in results if isinstance(e, Exception)]
            for name, res in results:
                self.log(f"  {name}: {'FAILED ' + str(res) if isinstance(res, Exception) else res}")
            self.log(f"exported {len(results) - len(bad)} of {len(results)} figure(s)")
            if bad:
                QMessageBox.warning(self, "Export", "\n".join(f"{n}: {e}" for n, e in bad))

        def failed(exc):
            self.all_btn.setEnabled(True)
            self.progress.setVisible(False)
            self.log(f"export failed: {exc}")

        def prog(i, n, msg):
            self.progress.setValue(i)
            self.progress.setFormat(f"{i}/{n}  {msg}")

        run_async_progress(self, lambda cb: F.export_recipes(self.session, project, recipes, out, cb),
                           done, failed, prog)

