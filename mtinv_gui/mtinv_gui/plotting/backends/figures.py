"""Figure production: multi-panel layouts, saved figure recipes, batch export.

* **Layouts** put plan views, sections and index maps on one page.  Each panel
  is a matplotlib ``SubFigure`` drawn by the same ``render_*`` function as a
  single figure, so a panel looks exactly like the figure on its own; the
  layout adds (a), (b), ... labels and one shared colour bar.
* **Recipes** (``project.recipes``) are named figures whose settings were
  snapshotted when they were added.  The model, georeferencing, section
  geometry and layers stay live, so pointing tab 1 at a newer inversion
  iteration and exporting the recipes redraws the whole figure set.
* :func:`export_recipes` is what both the GUI's "Export all" and the
  command line (``python -m mtinv_gui.plotting.batch``) run.
"""
from __future__ import annotations

import copy
import os
import string

import numpy as np

from . import render
from . import sections as SEC

#: offered in the GUI; widths are typical, check your journal's guide
WIDTH_PRESETS_MM = {"single column (≈ 85 mm)": 85.0, "1.5 column (≈ 130 mm)": 130.0,
                    "double column (≈ 175 mm)": 175.0, "A4 portrait text (≈ 160 mm)": 160.0}
LABEL_STYLES = ("(a)", "a)", "a", "A", "(A)", "none")
RECIPE_KINDS = {"plan": "plan view", "section": "section", "index": "section index map",
                "layout": "multi-panel layout"}


def panel_label(i: int, style: str) -> str:
    if style == "none":
        return ""
    letter = string.ascii_lowercase[i % 26]
    if style in ("A", "(A)"):
        letter = letter.upper()
    return {"(a)": f"({letter})", "a)": f"{letter})", "a": letter, "A": letter,
            "(A)": f"({letter})"}.get(style, f"({letter})")


def _section_by_label(sections_list, label):
    for sec in sections_list:
        if sec.label == label:
            return sec
    raise ValueError(f"no section labelled {label!r}")


# --------------------------------------------------------------------------- #
# layouts
# --------------------------------------------------------------------------- #
def _panels(layout):
    n = int(layout.rows) * int(layout.cols)
    panels = list(layout.panels)[:n]
    from ..project import LayoutPanel

    return panels + [LayoutPanel(kind="empty") for _ in range(n - len(panels))]


def _panel_style(layout, panel, shared: bool):
    """Per-panel copies of the layout's plan / section settings."""
    plan = copy.deepcopy(layout.plan)
    style = copy.deepcopy(layout.section_style)
    if shared:
        plan.colorbar = "none"
        style.colorbar = "none"
    if panel.kind == "plan":
        plan.depth_m = float(panel.depth_m)
        plan.main_layer = panel.main_layer
        if panel.title:
            plan.title = panel.title
    if panel.kind == "section" and panel.title:
        style.title = panel.title
    return plan, style


def _panel_aspect(session, project, layout, panel, col_w_in: float) -> float:
    """Height (in) a panel wants at width *col_w_in*."""
    if panel.kind == "plan":
        x0, x1, y0, y1 = session.extent(layout.plan.extent)
        # map box = column minus tick + axis labels; plus title + x labels above / below
        return max(col_w_in - 0.75, 0.5) * (y1 - y0) / max(x1 - x0, 1.0) + 0.8
    if panel.kind == "section":
        sec = _section_by_label(project.sections, panel.section)
        style = copy.deepcopy(layout.section_style)
        style.width_in, style.height_in = col_w_in, 0.0
        data = SEC.prepare(session, sec, style, project.layers)
        return render.section_size(sec, style, data, None)[1]
    if panel.kind == "index":
        return 0.86 * col_w_in
    return 0.3 * col_w_in


def layout_size(session, project, layout) -> tuple[float, float, list]:
    """``(width, height, row_heights)`` in inches."""
    w = render.width_of(session, layout.width_in)
    cols, rows = max(int(layout.cols), 1), max(int(layout.rows), 1)
    col_w = w / cols - 0.1
    panels = _panels(layout)
    keys = [_x_key(session, project, layout, p) for p in panels]
    # x tick labels + axis label that a hidden repeated x-axis gives back
    label_room = 0.15 + 2.4 * render.font_size_of(session, layout.plan.font_size) / 72.0
    row_h = []
    for r in range(rows):
        hs = []
        for c in range(cols):
            i = r * cols + c
            p = panels[i]
            try:
                h = _panel_aspect(session, project, layout, p, col_w)
            except ValueError:
                h = 0.3 * col_w
            below = i + cols
            if (layout.hide_shared_axes and below < len(panels) and keys[i] is not None
                    and keys[i] == keys[below]):
                h -= label_room
            hs.append(h)
        row_h.append(max(hs))
    if layout.height_in > 0:
        scale = float(layout.height_in) / max(sum(row_h), 1e-6)
        row_h = [h * scale for h in row_h]
    h = sum(row_h) + (_cbar_row_in(session, layout) if layout.shared_colorbar == "bottom" else 0.0)
    return w, float(h), row_h


def _x_key(session, project, layout, panel):
    """What decides a panel's x-range, to predict (before drawing) which x-axes
    will be hidden as repeats: plan views and index maps share the layout's
    extent; sections share when they are equally long on the same x-axis."""
    if panel.kind in ("plan", "index"):
        return (panel.kind,)
    if panel.kind == "section":
        try:
            sec = _section_by_label(project.sections, panel.section)
            prof = SEC.build_profile(sec, session, layout.section_style)
        except ValueError:
            return None
        return ("section", render._x_mode(sec), round(prof.length, 3)) if \
            render._x_mode(sec) == "distance" else ("section", sec.x_axis, tuple(
                np.round(render._x_from_s(render._x_mode(sec), prof,
                                          np.array([0.0, prof.length])), 3)))
    return None


def _cbar_row_in(session, layout) -> float:
    """Height of the shared colour-bar row: bar + tick labels + label."""
    return 0.3 + 3.4 * render.font_size_of(session, layout.plan.font_size) / 72.0


def render_layout(fig, session, project, layout) -> list:
    """Draw *layout* onto *fig*; returns the panels' main axes (None for empty)."""
    import matplotlib as mpl

    _w, _h, row_h = layout_size(session, project, layout)
    rows, cols = max(int(layout.rows), 1), max(int(layout.cols), 1)
    shared = layout.shared_colorbar in ("bottom", "right")
    fs = render.font_size_of(session, layout.plan.font_size)
    rc = render.style_rc(session, layout.plan.font_size)
    rc["axes.titlesize"] = "medium"           # panel titles: a grid of "large" titles collides
    with mpl.rc_context(rc):
        fig.clear()
        fig.set_layout_engine("constrained", w_pad=0.02, h_pad=0.02,
                              wspace=float(layout.wspace), hspace=float(layout.hspace))
        fig.__dict__["_mt_post_layout"] = []
        extra_r = 1 if layout.shared_colorbar == "bottom" else 0
        extra_c = 1 if layout.shared_colorbar == "right" else 0
        hr = list(row_h) + ([_cbar_row_in(session, layout)] if extra_r else [])
        wr = [1.0] * cols + ([0.08] if extra_c else [])
        gs = fig.add_gridspec(rows + extra_r, cols + extra_c, height_ratios=hr, width_ratios=wr)
        axes, mappable, cb_label, n_label, extend = [], None, render.RHO_LABEL, 0, "neither"
        for i, panel in enumerate(_panels(layout)):
            r, c = divmod(i, cols)
            sub = fig.add_subfigure(gs[r, c])
            if panel.kind == "empty":
                axes.append(None)
                continue
            plan, style = _panel_style(layout, panel, shared)
            try:
                if panel.kind == "plan":
                    ax = render.render_plan(sub, session, plan, plan.depth_m, project.layers)
                elif panel.kind == "section":
                    sec = _section_by_label(project.sections, panel.section)
                    data = SEC.prepare(session, sec, style, project.layers)
                    ax = render.render_section(sub, session, sec, style, data)
                else:
                    secs = [s for s in project.sections
                            if s.label in {p.section for p in layout.panels if p.kind == "section"}] \
                        or list(project.sections)
                    ax = render.render_index_map(sub, session, secs, style, plan,
                                                 layers=project.layers)
            except Exception as exc:  # noqa: BLE001 -- show the problem in its panel
                sub.clear()
                sub.text(0.5, 0.5, f"{panel.kind}: {exc}", ha="center", va="center",
                         color="#b91c1c", wrap=True)
                axes.append(None)
                continue
            ax._mt_label = panel_label(n_label, layout.panel_labels)
            n_label += 1
            axes.append(ax)
            if mappable is None and panel.kind in ("plan", "section") and not (
                    panel.kind == "plan" and panel.main_layer):
                mappable = getattr(ax, "_mt_mappable", None)
                cb_label = getattr(ax, "_mt_cb_label", cb_label)
                extend = getattr(ax, "_mt_extend", "neither")
        if shared and mappable is not None:
            if layout.shared_colorbar == "bottom":
                cbf = fig.add_subfigure(gs[rows, :cols])
                bar_h = 0.12 / max(_cbar_row_in(session, layout), 0.2)    # 0.12 in tall bar
                cax = cbf.add_axes([0.25, 1.0 - bar_h - 0.05, 0.5, bar_h])
                cb = fig.colorbar(mappable, cax=cax, orientation="horizontal", extend=extend)
            else:
                cbf = fig.add_subfigure(gs[:rows, cols])
                cax = cbf.add_axes([0.15, 0.25, 0.3, 0.5])
                cb = fig.colorbar(mappable, cax=cax, orientation="vertical", extend=extend)
            cb.set_label(cb_label)
        kinds = [p.kind for p in _panels(layout)]
        if layout.hide_shared_axes:
            _hide_repeated_axes(axes, kinds, rows, cols)      # before layout: space is reclaimed
        fig.get_layout_engine().execute(fig)
        for cb_fn in fig.__dict__.pop("_mt_post_layout", []):
            cb_fn()
        fig.set_layout_engine("none")
        if layout.hide_shared_axes:
            _hide_repeated_profile_axes(axes, kinds, rows, cols)
        for ax in axes:
            if ax is None or not ax._mt_label:
                continue
            head = getattr(ax, "_mt_head", ax)
            title = head.get_title()
            if title:
                # "(a) 5 km below sea level": a separate label beside a centred
                # title collides as soon as panels get narrow
                head.set_title(f"{ax._mt_label} {title}")
            else:
                head.set_title(ax._mt_label, loc="left", fontweight="bold", fontsize=1.1 * fs)
    return axes


def _same_range(a, b, which: str) -> bool:
    la = np.asarray(a.get_xlim() if which == "x" else a.get_ylim(), dtype="float64")
    lb = np.asarray(b.get_xlim() if which == "x" else b.get_ylim(), dtype="float64")
    return bool(np.allclose(la, lb, rtol=0.0, atol=1e-6 * max(abs(la[1] - la[0]), 1e-12)))


def _neighbour(axes, kinds, rows, cols, r, c, dr, dc):
    """The axes at (r + dr, c + dc) if it is the same kind of panel, else None."""
    rr, cc = r + dr, c + dc
    if not (0 <= rr < rows and 0 <= cc < cols):
        return None
    j = rr * cols + cc
    return axes[j] if axes[j] is not None and kinds[j] == kinds[r * cols + c] else None


def shared_flags(axes, kinds, rows, cols):
    """Per panel: ``(hide_x, hide_y, hide_top, hide_right)`` -- an axis's
    labels are repeated when the same kind of panel next to it (below for x,
    left for y; above / right for a plan view's lon/lat labels) shows exactly
    the same range."""
    out = []
    for i, ax in enumerate(axes):
        if ax is None:
            out.append((False, False, False, False))
            continue
        r, c = divmod(i, cols)
        below = _neighbour(axes, kinds, rows, cols, r, c, 1, 0)
        left = _neighbour(axes, kinds, rows, cols, r, c, 0, -1)
        above = _neighbour(axes, kinds, rows, cols, r, c, -1, 0)
        right = _neighbour(axes, kinds, rows, cols, r, c, 0, 1)
        out.append((below is not None and _same_range(ax, below, "x"),
                    left is not None and _same_range(ax, left, "y"),
                    above is not None and _same_range(ax, above, "x"),
                    right is not None and _same_range(ax, right, "y")))
    return out


def _hide_repeated_axes(axes, kinds, rows, cols) -> None:
    from matplotlib.axes._secondary_axes import SecondaryAxis

    for ax, (hx, hy, ht, hr) in zip(axes, shared_flags(axes, kinds, rows, cols)):
        if ax is None:
            continue
        if hx:
            ax.tick_params(axis="x", labelbottom=False)
            ax.set_xlabel("")
        if hy:
            ax.tick_params(axis="y", labelleft=False)
            ax.set_ylabel("")
        for child in getattr(ax, "child_axes", []):          # lon / lat graticule labels
            if isinstance(child, SecondaryAxis):
                orient = getattr(child, "_orientation", "")
                if (orient == "x" and ht) or (orient == "y" and hr):
                    child.tick_params(labeltop=False, labelright=False)


def _hide_repeated_profile_axes(axes, kinds, rows, cols) -> None:
    """Sections' profile panels (added after the layout): left-hand profile
    axes repeat when the section to the left is shown the same way, right-hand
    ones when the section to the right is."""
    flags = shared_flags(axes, kinds, rows, cols)
    for ax, (_hx, hy, _ht, hr) in zip(axes, flags):
        for a in getattr(ax, "_mt_profile_axes", []) if ax is not None else []:
            side = a.yaxis.get_label_position()
            if (side == "left" and hy) or (side == "right" and hr):
                a.tick_params(axis="y", labelleft=False, labelright=False)
                a.set_ylabel("")


# --------------------------------------------------------------------------- #
# recipes
# --------------------------------------------------------------------------- #
def recipe_from_plan(project, name: str = ""):
    from ..project import RecipeInfo

    p = copy.deepcopy(project.plan)
    return RecipeInfo(name=name or f"plan {p.depth_m / 1000.0:g} km", kind="plan", plan=p)


def recipe_from_section(project, sec, name: str = ""):
    from ..project import RecipeInfo

    return RecipeInfo(name=name or f"section {sec.label}", kind="section",
                      section=copy.deepcopy(sec), section_style=copy.deepcopy(project.section_style))


def recipe_index_map(project, name: str = ""):
    from ..project import RecipeInfo

    group = getattr(project, "section_group", "all") or "all"
    return RecipeInfo(name=name or ("section index map" if group == "all" else
                                    f"section index map ({SEC.SECTION_GROUPS.get(group, group)})"),
                      kind="index", section_group=group,
                      plan=copy.deepcopy(project.plan),
                      section_style=copy.deepcopy(project.section_style))


def new_layout_recipe(project, rows: int = 2, cols: int = 2, name: str = ""):
    """A layout pre-filled from the project: plan views at the plan depth list,
    or the sections, whichever fills the grid."""
    from ..project import LayoutInfo, LayoutPanel, RecipeInfo

    lay = LayoutInfo(rows=rows, cols=cols, plan=copy.deepcopy(project.plan),
                     section_style=copy.deepcopy(project.section_style))
    n = rows * cols
    depths = list(project.plan.depths_m)[:n]
    lay.panels = [LayoutPanel(kind="plan", depth_m=d) for d in depths]
    lay.panels += [LayoutPanel(kind="empty") for _ in range(n - len(lay.panels))]
    return RecipeInfo(name=name or f"layout {rows}x{cols}", kind="layout", layout=lay)


def recipe_size(session, project, recipe) -> tuple[float, float]:
    if recipe.kind == "plan":
        return render.plan_size(recipe.plan, session)
    if recipe.kind == "section":
        sec = _live_section(project, recipe)
        data = SEC.prepare(session, sec, recipe.section_style, project.layers)
        return render.section_size(sec, recipe.section_style, data, session)
    if recipe.kind == "index":
        w = render.width_of(session, 7.0)
        return w, 6.0 * w / 7.0
    w, h, _ = layout_size(session, project, recipe.layout)
    return w, h


def _live_section(project, recipe):
    """A section recipe's geometry: the project's current section with that
    label (so moving the line updates the recipe), else the snapshot."""
    try:
        live = _section_by_label(project.sections, recipe.section.label)
    except ValueError:
        return recipe.section
    sec = copy.deepcopy(live)
    sec.ve, sec.x_axis, sec.reverse = recipe.section.ve, recipe.section.x_axis, recipe.section.reverse
    return sec


def render_recipe(fig, session, project, recipe):
    if recipe.kind == "plan":
        return render.render_plan(fig, session, recipe.plan, recipe.plan.depth_m, project.layers)
    if recipe.kind == "section":
        sec = _live_section(project, recipe)
        data = SEC.prepare(session, sec, recipe.section_style, project.layers)
        return render.render_section(fig, session, sec, recipe.section_style, data,
                                     others=SEC.group_members(project.sections, sec))
    if recipe.kind == "index":
        secs = [s for s in project.sections if SEC.in_group(s, recipe.section_group)]
        return render.render_index_map(fig, session, secs, recipe.section_style,
                                       recipe.plan, layers=project.layers)
    return render_layout(fig, session, project, recipe.layout)


def recipe_dpi(recipe) -> int:
    if recipe.kind == "plan":
        return int(recipe.plan.dpi)
    if recipe.kind == "layout":
        return int(recipe.layout.plan.dpi)
    return int(recipe.section_style.dpi)


def recipe_path(project, recipe, out_dir: str = "") -> str:
    """Where a recipe is written: its own path, made absolute against the
    project folder, or ``<out_dir>/<name>.pdf``."""
    path = recipe.out_path
    if out_dir:
        base = os.path.basename(path) if path else _safe(recipe.name) + ".pdf"
        return os.path.join(out_dir, base)
    if not path:
        path = _safe(recipe.name) + ".pdf"
    if not os.path.isabs(path):
        path = os.path.join(project.workdir or os.getcwd(), path)
    return path


def _safe(name: str) -> str:
    keep = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name.strip())
    return keep or "figure"


def export_recipe(session, project, recipe, path: str = "") -> str:
    path = path or recipe_path(project, recipe)
    fig = render.new_figure(*recipe_size(session, project, recipe))
    render_recipe(fig, session, project, recipe)
    return render.save_figure(fig, path, session, recipe_dpi(recipe))


def export_recipes(session, project, recipes=None, out_dir: str = "", progress=None) -> list:
    """Every recipe (or *recipes*) to its file; returns ``[(name, path or exception)]``.
    One failing recipe doesn't stop the rest."""
    recipes = list(project.recipes if recipes is None else recipes)
    out = []
    for i, r in enumerate(recipes):
        if progress:
            progress(i, len(recipes), r.name)
        try:
            out.append((r.name, export_recipe(session, project, r, recipe_path(project, r, out_dir))))
        except Exception as exc:  # noqa: BLE001
            out.append((r.name, exc))
    if progress:
        progress(len(recipes), len(recipes), "done")
    return out


def open_session(project):
    """A session for *project*: model loaded, georeferenced, house style set.
    (What the command line uses; the GUI builds its own.)"""
    from .session import PlotSession

    s = PlotSession()
    s.load_model(project.model)
    if project.georef.crs:
        s.apply_georef(project.georef)
    else:
        s.solve_georef(project.georef)
    s.house = project.house
    return s


__all__ = ["WIDTH_PRESETS_MM", "LABEL_STYLES", "RECIPE_KINDS", "render_layout", "layout_size",
           "recipe_from_plan", "recipe_from_section", "recipe_index_map", "new_layout_recipe",
           "render_recipe", "recipe_size", "export_recipe", "export_recipes", "open_session"]


def warm(session, project, recipe) -> tuple[float, float]:
    """Do a recipe's slow, cached work -- mesh cuts, interpolations, layer
    loading -- so drawing it afterwards is quick.  Safe off the GUI thread
    (no matplotlib).  Returns the figure size."""
    from . import slicing

    size = recipe_size(session, project, recipe)
    session.active_layers(project.layers)
    plans = []
    if recipe.kind in ("plan", "index"):
        plans = [(recipe.plan, recipe.plan.depth_m)]
    elif recipe.kind == "layout":
        plans = [(recipe.layout.plan, p.depth_m) for p in _panels(recipe.layout) if p.kind == "plan"]
    for plan, depth in plans:
        slicing.plan_slice(session.model, session.maptr, float(depth), plan.reference, plan.crinkle)
    return size
