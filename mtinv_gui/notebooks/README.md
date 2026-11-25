# Notebooks

Jupyter notebooks that do what the two GUIs do, one step at a time, with the code in
view. Use them when a job needs something the GUIs don't offer: a refinement of your own,
a different way of handling the ocean, a figure the plotting GUI doesn't make. Each
notebook reads and writes ordinary files, so you can switch between notebooks and GUIs
from one step to the next.

```bash
conda activate mtinv
cd notebooks
jupyter lab
```

Start with `00_start_here.ipynb`: it explains how to run a notebook and checks the
environment.

| notebook | does | GUI |
|---|---|---|
| `00_start_here` | how to use them, environment check, conventions | |
| `01_interpolate_edis` | every station onto one set of periods | setup tab 0 |
| `02_survey_and_topography` | check stations, choose the projection, get a DEM (detailed + regional), project it | setup tabs 1-3 |
| `03_mesh_and_receivers` | the mesh, with each refinement as a pyGoFEM call; topography, thin-sheet or real ocean; receivers | setup tab 5 |
| `04_refine_inverted_model` | refine a finished model, add stations | setup tab 7 |
| `05_plot_model` | plan views, sections, map layers | `mtplot` |
| `06_model_to_geotiff` | depth slices as GeoTIFFs | `mtplot`, GeoTIFF tab |

**No data files.** Choosing the data needs a careful look through it, so select it and write
the GoFEM data file in `gateway_mt`, with the receiver file from notebook 03 or 04.

**The worked example** runs out of the box: `sample_data/edis` holds 31 synthetic stations
beside Monterey Bay, California (dummy responses at real positions, so the ocean and
topography are real), `sample_data/edis_infill` 9 more to add to a finished model, and
`sample_data/synthetic_model.py` turns notebook 03's mesh into a stand-in inversion result.
Notebook 02 downloads the topography from GMRT, so it needs an internet connection.
Everything is written to `work/` (not tracked); delete it to start again. The whole chain
takes about 3 minutes.

**For your own data**, change the paths in each notebook's Settings cell. Save a copy of a
notebook before changing it for a project.

`tests/test_notebooks.py` runs all of them on the sample data, so a change to the
backends that breaks a notebook fails the tests. Commit notebooks without outputs
(*Edit → Clear Outputs of All Cells*, then save); the test checks that too.
