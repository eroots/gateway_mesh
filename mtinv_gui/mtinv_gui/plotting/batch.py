"""Redraw a project's saved figures without the GUI.

Installed as the ``mtplot-batch`` command (``pip install -e .``, see
pyproject.toml); ``python -m mtinv_gui.plotting.batch`` works too.

    python -m mtinv_gui.plotting.batch plots.mtplot.json
    python -m mtinv_gui.plotting.batch plots.mtplot.json --model ZK/model_ZK_40.vtu --out figs_it40
    python -m mtinv_gui.plotting.batch plots.mtplot.json --only "section A" --only "layout 2x2"
    python -m mtinv_gui.plotting.batch plots.mtplot.json --list

``--model`` swaps in another model file (e.g. a newer inversion iteration,
same mesh frame) without editing the project; the georeferencing stored in the
project is reused.  Run from the ``mtinv`` environment.
"""
from __future__ import annotations

import argparse
import os
import sys


def main(argv=None) -> int:
    called = os.path.basename(sys.argv[0])
    prog = "mtplot-batch" if called.startswith("mtplot-batch") else "python -m mtinv_gui.plotting.batch"
    ap = argparse.ArgumentParser(prog=prog,
                                 description="Export the figure recipes saved in a *.mtplot.json")
    ap.add_argument("project", help="plotting project (*.mtplot.json)")
    ap.add_argument("--model", help="use this model file instead of the project's")
    ap.add_argument("--out", default="", help="write every figure into this folder "
                                              "(default: each recipe's own path)")
    ap.add_argument("--only", action="append", default=[], metavar="NAME",
                    help="export only this recipe (repeatable)")
    ap.add_argument("--list", action="store_true", help="list the recipes and exit")
    args = ap.parse_args(argv)

    from .backends import figures as F
    from .project import PlotProject

    project = PlotProject.load(args.project)
    if args.list:
        for r in project.recipes:
            print(f"{r.name:30s} {F.RECIPE_KINDS.get(r.kind, r.kind):22s} "
                  f"{F.recipe_path(project, r, args.out)}")
        return 0
    recipes = project.recipes
    if args.only:
        wanted = set(args.only)
        recipes = [r for r in recipes if r.name in wanted]
        missing = wanted - {r.name for r in recipes}
        if missing:
            print(f"no recipe named: {', '.join(sorted(missing))}", file=sys.stderr)
            return 2
    if not recipes:
        print("nothing to export (add recipes on the Figures tab)", file=sys.stderr)
        return 1
    if args.model:
        project.model.model_path = os.path.abspath(args.model)
    print(f"loading {project.model.model_path}")
    session = F.open_session(project)
    failed = 0
    for name, result in F.export_recipes(
            session, project, recipes, args.out,
            progress=lambda i, n, msg: print(f"[{i + 1}/{n}] {msg}") if i < n else None):
        if isinstance(result, Exception):
            failed += 1
            print(f"  FAILED {name}: {result}", file=sys.stderr)
        else:
            print(f"  wrote {result}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
