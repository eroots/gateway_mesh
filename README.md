# gateway_mesh

Desktop tools for 3-D magnetotelluric (MT) inversion with
GoFEM: from a folder of EDI files to a mesh and
input files, and from a finished inversion model to maps, sections and GeoTIFFs.

| command | what |
|---------|------|
| `mtinv` | inversion setup: EDIs, topography and bathymetry, mesh design and build (deal.II), GoFEM data and `.prm` files, refining an inverted model |
| `mtplot` | model plotting: plan views, vertical sections, other datasets as layers, GeoTIFFs, multi-panel figures |
| `mtplot-batch` | redraw an `mtplot` project's saved figures from the terminal |
| `jupyter lab notebooks` | step-by-step notebooks that do the same with the code in view |

The environment also installs [gateway_mt](https://github.com/eroots/gateway_mt), for
viewing and editing MT data, and its `mt_launcher`, a small window that starts gateway_mt,
`mtinv` and `mtplot` on a project folder.

## Requirements

**Linux or Windows WSL2, x86-64.** Meshes are built with deal.II 9.3, which conda-forge
provides for Linux only. On Windows 11, WSL2 shows the GUIs without extra setup; on
Windows 10 you need an X server (see [mtinv_gui/README.md](mtinv_gui/README.md#install)).

gateway_mt on its own also runs natively on Windows: see its repository.

## Install

You need `git` and conda.

**On Windows**, set up WSL2 first: in PowerShell, run `wsl --install` (it installs
Ubuntu; restart when asked), then open Ubuntu from the Start menu and run everything
below there. Ubuntu comes with `git`.

**If you don't have conda**, [Miniforge](https://github.com/conda-forge/miniforge) works
well. Install it, then close and reopen the terminal:

```bash
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
```

Then:

```bash
git clone https://github.com/eroots/gateway_mesh.git
cd gateway_mesh/mtinv_gui
conda env create -f environment.yml
conda activate mtinv
python tests/test_install.py
```

Every package in `environment.yml` is pinned to the build that was tested, so the
install is quick and repeatable. Details, including the optional natural-neighbour
interpolation, are in [mtinv_gui/README.md](mtinv_gui/README.md#install).

## Getting started

- [mtinv_gui/README.md](mtinv_gui/README.md): the setup GUI, tab by tab, with a walkthrough
  on the sample stations in [`edis/`](edis).
- [mtinv_gui/PLOTTING.md](mtinv_gui/PLOTTING.md): the plotting GUI.
- [mtinv_gui/notebooks/](mtinv_gui/notebooks/README.md): the notebooks, which run on the
  included sample data.

## Status

GoFEM output is in use for real inversions. Some parts are newer and not yet confirmed
on a real inversion run:

- mesh output for **FEMTIC** (with its data and control files) and **SimPEG** (an octree);
- in the Refine Model tab: level balancing, smoothing of the refined cells, and moving
  stations off cell faces.

Versions are `0.0.x`: file formats and settings may still change. Projects
(`*.mtinv.json`, `*.mtplot.json`) saved by older versions open in newer ones.

## Built on

[deal.II](https://www.dealii.org) 9.3,
[pyGoFEM](https://github.com/eroots/pyGoFEM/tree/dealii-9.3) (a fork kept compatible with
deal.II 9.3 and mtpy 1), [mtpy](https://github.com/MTgeophysics/mtpy) 1.1.5,
[PyVista](https://pyvista.org), [pyproj](https://pyproj4.github.io/pyproj),
[rasterio](https://rasterio.readthedocs.io), [GeoPandas](https://geopandas.org),
[discretize](https://discretize.simpeg.xyz). Topography from
[GMRT](https://www.gmrt.org) or [OpenTopography](https://opentopography.org).

## License

GPL-3.0: see [LICENSE](LICENSE).
