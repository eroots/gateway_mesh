# MT Inversion Setup GUI

Part of [gateway_mesh](../README.md).

> **Plotting finished models:** see [PLOTTING.md](PLOTTING.md). It's a sibling
> app (the `mtplot` command, same environment) for plan views, sections,
> GeoTIFFs and publication figures.
>
> **Notebooks:** [notebooks/](notebooks/README.md) does what both apps do, step by
> step, with the code in view: for meshes and figures the GUIs don't cover.

A desktop tool that turns a folder of **EDI files** into the input files a 3-D
magnetotelluric inversion needs. First target: **GoFEM**. The mesh can also be
written for **FEMTIC** (its DHEXA hexahedral mesh: the same cells), with its data and
control files, and as a **SimPEG** octree (a discretize `TreeMesh`). The structure leaves
room for **ModEM**.

## Install

You need Linux or Windows WSL2 (x86-64), `git`, and conda
([Miniforge](https://github.com/conda-forge/miniforge) works well). From this
folder:

```bash
conda env create -f environment.yml
conda activate mtinv
python tests/test_install.py
```

The last line checks the install. After that, with `mtinv` active, from any folder:

| command | what |
|---------|------|
| `mtinv` | inversion setup GUI |
| `mtplot` | model plotting GUI |
| `mtplot-batch` | redraw saved figures from the terminal (see [PLOTTING.md](PLOTTING.md)) |
| `jupyter lab notebooks` | the step-by-step notebooks (see [notebooks/README.md](notebooks/README.md)) |
| `gateway_mt` | MT data viewing and editing ([gateway_mt](https://github.com/eroots/gateway_mt), installed with the environment) |
| `mt_launcher` | a window that starts `gateway_mt`, `mtinv` and `mtplot` on a project folder |

The environment includes deal.II, mtpy and pyGoFEM, so nothing has to be built
by hand. Two pins in `environment.yml` are deliberate:

- **deal.II 9.3.2.** GoFEM installations built against deal.II 9.3 can't read
  meshes from newer versions.
- **[A pyGoFEM fork](https://github.com/eroots/pyGoFEM/tree/dealii-9.3).** Current
  upstream pyGoFEM expects a newer deal.II and mtpy 2. The fork stays on the
  compatible code and adds rounded polygon refinement, which the Mesh tab needs.

**On Windows:** WSL2 on Windows 11 can show GUI apps out of the box. On Windows
10, install an X server (VcXsrv or X410) and set `DISPLAY` to point at it.

**Optional: natural-neighbour interpolation** for the Sections tab. This one
package compiles C++ when installed, so it needs a compiler (`sudo apt install
g++`) and isn't part of the environment. Every other interpolation method works
without it.

```bash
pip install "git+https://github.com/eroots/natural-neighbor-interpolation@7296adde70e778f9f19d9255b2c27fd6dadb9c30"
```

## Try it with the sample data

The `edis` folder at the top of this repository holds 121 synthetic stations,
about 100 km across, near 49°N 95°W. To go from those to GoFEM input files in
the `mtinv` GUI, without downloading topography:

1. **Survey / EDIs:** choose the `edis` folder and load it.
2. **Mesh:** press **Recompute From Data**. On page 6, untick **Fit Mesh to the
   Projected DEM**, then press **Build Full Mesh → Files**.
3. **GoFEM files:** write the frequencies, data and `.prm` files.

## Status

| Step | Panel | Backend | State |
|------|-------|---------|-------|
| 0 | Interpolate EDIs | `backends/edi_interp.py` | **functional**, optional -- resample a folder of EDIs onto one shared log-period grid before Survey / EDIs; skip it if your EDIs are already on a common grid |
| 1 | Survey / EDIs | `backends/edi_io.py` | **functional** — EDI read is mtpy-first with a text-parser fallback (parity: `tests/test_edi_readers.py`); folder of `*.edi` or a `.lst`; model frame = a map projection to metres you choose explicitly (survey-centred Transverse Mercator / Oblique Stereographic / Azimuthal Equidistant / Lambert Conformal Conic with origin 0,0, or a UTM zone / EPSG e.g. 3979), with a live distortion readout and a "pick lowest distortion" button; Lat-Lon + Projected station maps |
| 2 | Topography | `backends/topography.py` | **functional** (GMRT / OpenTopography / local; stations drawn on the DEM) |
| 3 | Project DEM | `backends/projection.py` | **functional** (pyproj) |
| 4 | Data | `backends/edi_z.py` | **functional** -- SSQ ρ_a + C-response depth, editable period window |
| 5 | Mesh | `backends/mesh_design.py` + `backends/mesh.py` | **functional** -- guided flow, data-driven sizing, deal.II build |
| 6 | GoFEM files | `backends/gofem_files.py` | **functional**; EDIs mtpy 1.1.5 can't read as written (a tipper block with no variance block, or a coordinate with 60 seconds) are read from a fixed copy and the log says how many. Those tippers' errors are the error floor |
| 7 | Refine Model | `backends/refine.py` | **functional**, see [below](#the-refine-model-tab) |

### The Mesh tab (redesigned)

One guided **QToolBox** flow, biggest → smallest, with the 3-D preview on the
right. "Recompute from data" pulls footprint, nearest-neighbour spacing, period
range, the **log-mean SSQ resistivity curve** (`backends/edi_z.py`) and DEM
relief, then:

1. **Model Size** — depth & edge padding as multiples of δ_max; optional 20 Ω·m
   mantle transition below 410 km (fixed, not inverted), which caps the model
   depth at 1000 km.
2. **Ocean** — how the sea enters the model (needs the projected DEM). **None**
   ignores bathymetry (flat sea level). **Thin Sheet**
   keeps the surface flat and gives the top *X* km over the ocean the
   *conductance* of the real water column (`depth × σ_sea ÷ X`, never below the
   background); cells that only partly overlap the sheet get the thickness-
   weighted mix so the column conductance is exact whatever the cell sizes, and
   are binned into ~100 log-spaced conductivity classes (material ids from 5).
   **Real Ocean** puts the seafloor into the fitted surface: cells between sea
   level and the seabed become water (material 3, σ = 1/sea resistivity).
   Its accuracy is limited by the cell size near the surface -- add an
   *Interface* and an *Along Coastline* stage; the build logs what fraction of
   the DEM's water volume ended up resolved. Sea and thin-sheet cells are
   fixed (not inverted). Sea resistivity defaults to 1/3.2 S/m.
   **Sea Is** decides which DEM points below 0 m are sea. *Connected* (default):
   only those joined to the open sea (the DEM's edge), across gaps of up to
   **Joined Across Gaps Up To** (10 km; a DEM doesn't resolve narrow straits or fjord
   mouths). Land below sea level elsewhere (Death Valley, polders, lake beds) stays
   land at its own elevation, with air above it, in every ocean mode. *Everything
   Below Sea Level* is the old behaviour. **Always Land** / **Always Sea** take
   polygon files (shapefile, GeoPackage, GeoJSON, KML) where the rule gets it wrong.
   The page reports how much land below sea level the model holds.
3. **Base Cells** — **not** sized to fit between stations, that is what survey
   refinement is for. Base cell size defaults to whatever spans the padded
   half-width in 20 cells (editable N, or type an exact size); vertical growth
   factor; uniform/graded dx; round-to-clean; Advanced: floor, max-layer.
   `dz_earth` is floored at `0.5·min(dx)` so the base grid stays near-isotropic.
   An **"Override cell lists"** escape hatch lets you type the raw dx/dz lists
   directly. The base first layer is **always** at least `0.5x` the base cell.
   The info line reports the current grid: base dx cell, the dz range and layer
   count, base cell total, the surface dz, and -- for context -- the data's
   **C-response depth** `Re{C}` at the lowest *selected* period (the shallowest
   depth the data resolves) with how many refinement levels it would take to
   bring the surface cell down to it. At the bottom, the panel **suggests how
   many survey-area refinement levels** get you *N cells between sites* (minimum
   or median spacing); that count seeds step 3 until you hand-edit the level
   there (a *Recompute From Data* re-seeds it).
4. **Refine Survey Area** — levels (pre-filled from the base-cells suggestion) /
   buffer / depth / taper / shape. The polygon is **automatically the convex
   hull of the stations + buffer** (no more undefined "polygon tokens").
5. **Local Refinement** — add "around stations", "rectangular zone",
   **"along coastline"** or **"interface"** stages, each a small form. Optional.
   *Along Coastline* refines around the edge of the sea (see **Sea Is**). Before it is
   sampled (one point per radius of arc length) the contour is cleaned up:
   wiggles smaller than **Simplify Tolerance** are straightened
   (Douglas-Peucker) and islands / lakes enclosing less than **Drop Loops
   Smaller Than** are ignored, so detail far below the cell size does not cost
   thousands of refinement points. **Preview Coastline** shows raw vs. cleaned
   coast over the DEM with live controls, and outlines land below sea level in purple.
6. **Topography & materials** — fit to DEM, background ρ (auto = SSQ log-mean).

Refinement is stored as form **stages** (`mesh_design.RefinementStage`);
`mesh.run` turns them into deal.II ops once it knows the projected footprint.
The build bar previews the mesh **to any step** (`mesh.run(...,
stop_after_refinement=N)`, writing `model_step<N>.vtu`) or builds the full
mesh + files. The viewer renders that `.vtu` **in 3-D with PyVista**
(`gui/mesh_viewer.py`, `Mesh3DView`) -- a box clipped to **whole model /
within N skin depths**, by default cut open through its centre (**Cutaway**,
any axis) to expose the interior, or an **Interactive Slice** you drag
through the mesh by hand. Before any preview exists the same viewer shows a
plain 2-D station plot instead -- also the click target for the Refine
Survey Area step's Custom Polygon picker, which needs a literal plan-view
click, not a 3-D one.

`mesh.run` (base grid → materials → refinements → fit-to-topography → ocean
materials → receiver projection → mask + file writing) includes the ocean modes
above (offshore stations are projected onto the seafloor in real-ocean mode). The pure ocean / coastline logic is covered by
`tests/test_ocean.py`; the deal.II side is not. `_mu()` in `backends/mesh.py`
resolves each pyGoFEM helper across candidate submodules and raises a named
error if the API differs.

### Walkthrough

1. Survey tab: load a folder of EDIs (or a `.lst`) and choose the model frame.
   The default is a **UTM** zone, kept in that grid minus the origin. Survey-centred
   projections and any EPSG code (Canada Atlas Lambert EPSG:3979, etc.) are also
   available, and the tab shows each option's worst-case scale error. Stations,
   DEM and the deal.II mesh all end up in this frame. The origin re-centres on
   the survey automatically when a new dataset loads, unless you tick "lock
   origin". The station map has a **Lat / Lon** and a **Projected** view.
2. Topography → Project DEM: GMRT download, resample, project (or skip for a
   flat interface). Add a **Regional DEM** that reaches across the mesh padding: the
   detailed DEM is used wherever it reaches, the regional one everywhere else. Without
   it the mesh repeats the detailed DEM's edge values out to its own edge, and with an
   ocean the sea floor at that edge fills the padding. The Mesh tab warns, above the
   build buttons, whenever the DEM stops short of the mesh.
3. Data tab: "Compute SSQ resistivity + C-response depth"; set the period window.
4. Mesh tab: "Recompute from data", walk the QToolBox pages, use "preview to
   step" to watch it refine in the in-app 3-D viewer, then "Build full mesh →
   files". **Mesh For** "GoFEM + FEMTIC" also writes `mesh.dat` and
   `resistivity_block_iter0.dat` (air = block 0, the sea and a mantle fixed, one free block
   per earth cell); FEMTIC takes no ocean or the real one, not the thin sheet.
   **Mesh For** "GoFEM + SimPEG" also writes an octree for SimPEG (`octree_mesh.txt`, UBC
   format: `discretize.TreeMesh.read_UBC`), with UBC model files and `simpeg_model.npz`
   (starting conductivity, cell type, active = inverted cells, receivers; discretize's cell
   order) and `receivers_simpeg.csv`. SimPEG's frame: x east, y north, z up, origin at the
   survey centre. An octree can't bend to the topography, so the GoFEM mesh, fitted as
   usual, tells it how fine to be where: every place gets octree cells no wider than the
   GoFEM cells there, so the fine cells follow the real surface, the stations and the coast.
   Each octree cell is earth, air, sea or mantle by where its centre is relative to the DEM,
   and each station sits on the top of the earth (or the seafloor) under it. All octree
   cells have one shape: **cell width / height** (next to Mesh For) makes them flatter, for
   topography and a shallow sea; the build warns when the octree loses much of the sea. The
   octree's finest cells are as wide as the narrowest GoFEM cells, so its box (a power of 2 of
   them) is usually bigger than the GoFEM model, up to twice as wide and deep; the margin gets
   only a few coarse cells, with the DEM's edge values for the surface. It is centred on the
   survey and on sea level. No ocean or the real one, as for FEMTIC.
5. GoFEM files tab: "Write frequencies + data + .prm files". With a mesh built for
   FEMTIC it also writes a FEMTIC run folder, `<output dir>/femtic/`: copies of the two
   mesh files, `observe.dat` (the same stations, frequencies and error floor as the GoFEM
   data, in FEMTIC's SI units and e<sup>-iωt</sup> convention; `observe_sites.csv` maps its
   station ids to names) and `control.dat` (FEMTIC's recommended first settings; 0
   iterations = a forward run).

`Project > Environment report...` shows which packages the GUI can find.
**Project > Settings** can point at development copies of deal.II, pyGoFEM or
mtpy instead of the installed ones; leave those paths blank otherwise.

### The Refine Model tab

Refines an existing, usually inverted, model. It starts from files, not from the
other tabs, so it works on a model from any project. The files it needs are the
`.tria` and its `.materials`, plus usually its receivers. Cells keep their material
ids when split, so they start from the inverted conductivity, and GoFEM treats each
cell as its own parameter.

1. **Model:** load the files. The tab reports cell counts and sizes and checks that
   the `.tria` and `.materials` belong together.
2. **Stations:** keep the existing ones, **add** new ones alongside them, or
   **replace** them with a new dataset (for example, to use a larger model as the
   starting model for a regional study). New stations need the model's coordinate
   frame, which can be fitted from the existing receivers' EDIs, read from the setup
   project the model was built with, or typed in. A setup project is refused if its
   frame doesn't fit the model's receivers. **Check Stations** shows the surface cell size under each.
3. **Refinement Rules:** a box, ellipse, polygon or ellipsoids around stations. A rule
   refines once per level, in shells from outer to inner, like the Mesh tab's survey
   area: each shell splits every cell that overlaps the region grown by its buffer,
   down to its depth, and never below the rule's minimum cell size. Buffers and depths
   shrink by a factor per level, or can be typed as lists, and the map shows each shell
   (except around stations, where the buffer adds to the radius). A polygon's bottom can be
   rounded with **Bottom Shape (pnorm)**, as in the Mesh tab: above 1 it refines a bowl, full
   depth under the polygon's middle and shallower towards its edges. With no buffer and no
   taper, every level uses the region itself. Depths are below the local surface. Rules
   run in order, and the arrows change it. A new box or ellipse starts centred on the
   new stations if there are any, or else on the model's own. **Smooth the Refined Cells**
   (optional, below the rules) is described after this list.
4. **Topography (optional):** refining leaves the surface exactly where it is. To
   follow a DEM more closely, the refit moves the surface by the difference from where
   it is now. It can't apply the relief twice, and it only changes the area the DEM
   covers, fading out at the DEM's edge, so a detailed regional DEM can be used on a
   larger model. Junctions between coarse and fine cells keep their shape, because a
   vertex there must stay on the coarse cell's face. It isn't available for models
   with sea cells. The DEM can be the Topography tab's lon/lat `.npz`, which the tab
   places using the model's frame, or a Project DEM `.npz` made in that frame. The tab compares the DEM with the model's surface as soon as you
   choose one, and warns if it looks offset, has swapped axes, or is from another
   area. A build with such a DEM asks for confirmation first.
5. **Output:** **Estimate** runs the real refinement and saves nothing.
   **Build Refined Model** writes `model_refined.tria`, `.vtu`, `.materials`, the
   receivers and, if you ask, a mask, plus frequencies, data and `.prm` files.
   Inversion settings come from the GoFEM Files tab, and the Active Domain Box is
   filled in from an `input_inversion*.prm` near the model when there is one.

The station map shows a plan-view slice through the model, as true cells coloured by
resistivity. Move through depths with the slider, or the mouse wheel over the map. Depth
is below sea level. **Estimate** and **Build** switch it to the refined model, so you can
see what a rule did before saving anything.

Every station is checked against the finished mesh before anything is saved: earth at
its depth (0.1 m below the surface), and air or water just above it. If one fails, the
build stops without writing.

A station exactly on a vertical cell face belongs to two cells. GoFEM splits the mesh
between processes, and when those two cells belong to different ones, neither may take the
station: the run stops with *Not all cells for receiver points were found*, or not, depending
on how the mesh was split. GoFEM's *Refine cells around receivers* splits each station's cell,
so a station on one of its halving lines ends up on a face too. Any station within 1 m of its
cell's faces, or of the lines that splitting the cell up to 3 times would add, is moved 2 m
into the cell, and the build says which. Mesh-tab builds do the same.

After the rules, the model is **balanced**: wherever cells meeting at a corner or along an
edge are two levels apart (deal.II only keeps face neighbours within one level), the coarser
ones are split, as p4est meshes are. It adds very few cells and doesn't move the surface. With
no rules (new stations only), nothing is balanced and the mesh is written as it was.

**Smoothing the refined cells.** A split cell's children keep its conductivity, so the old
blocks show in the finer mesh. With **Smooth the refined cells** on, the model as it was before
refining is Gaussian-filtered (log10 conductivity) and each refined cell gets the filtered value
at its centre, under a new material id of its own, appended to the `.materials`. Each original
cell counts by the Gaussian's integral over it, so the result is an exact filter of the blocky
model, like `scipy.ndimage.gaussian_filter` on a fine grid but without the grid. Lengths are
the Gaussian's standard deviation across and down: **relative** to the size of the cell each
refined cell came from (so a 20 km cell at depth is smoothed more than a 1 km one near the
surface), or fixed in km. Relative 0.25 is light: across an old cell's face the change spreads
over about two thirds of its width; 0.5 blurs a pattern of single cells out almost entirely.
Unrefined cells, air, sea and the **Fixed Material IDs** (page 5) don't change, and values
can't leave the range the model already has, so GoFEM's BOUNDED limits still hold. Cells the
inversion pushed to those limits are averaged with their neighbours like any other.

## Topography providers

- **GMRT** GridServer -- keyless, global topography + bathymetry, **NetCDF**
  (COARDS, EPSG:4326). Default.
- **OpenTopography** `globaldem` API -- GeoTIFF; needs a free key
  (Project > Settings). `demtype` selectable (SRTMGL1/3, COP30/90, GEBCOIceTopo,
  NASADEM).
- **local** -- ingest your own NetCDF or GeoTIFF.

Before anything is fetched, the tab estimates each DEM's size: the download (nodes,
posting, file size) and the resampled `.npz` (nodes, file size, memory to make it).
The download covers the survey box plus the pad at the provider's own posting, so
*target posting* only changes the `.npz`. GMRT's "default" resolution keeps each side
under about 1000 nodes, so its files stay at a few MB but get coarser as the box grows
(the estimate follows GMRT's levels and is marked `~`). OpenTopography uses the
dataset's posting (1″ for SRTMGL1 / NASADEM / COP30: a 1° pad around a 4° x 3° survey is
~0.8-1.6 GB) and caps the area per request (450,000 km² at 1″, 4,050,000 km² at 3″,
125,000,000 km² for GEBCO). The tab warns in red when a download or `.npz` is large,
when a request is over that cap (it is then not sent), and when the download is
coarser than the target posting; Download / Clip + resample ask before large jobs.

The Topography tab takes two DEMs: the detailed one (pad and posting in arc-seconds)
and an optional **regional** one (pad in km, coarse posting) for the mesh padding.
One grid fine enough for the survey and wide enough for the padding would be far too
big, and downloads of large areas come at a coarse resolution anyway (GMRT gives
about 3 km for a 600 km pad). The Project DEM tab grids both into
`DEM_projected_gridded.npz`: `arr_0` is the detailed grid as before, `regional` the
coarse one, with its own posting. Its hint line says how far the mesh reaches, so
the pad can be set to cover it.

## Sessions

`Project ▸ Save` / `Save As` (Ctrl+S) writes the whole build state to a
`*.mtinv.json`. The most recently saved/opened project is remembered in
`~/.mtinv_gui/settings.json` and **reopened automatically on the next launch**
(toggle: `Project ▸ Reopen last project on startup`). On exit the current
session file is re-saved so a restart resumes exactly where you left off.

`mtinv project.mtinv.json` opens that project instead, and `mtinv --new` starts
a blank one in the current folder.

## Architecture

- `mtinv_gui/project.py` - the `Project` dataclass tree. One JSON file per
  project (`*.mtinv.json`) in a working directory; big artefacts (DEM `.npz`,
  meshes, `.prm`) are written next to it and referenced by path.
- `mtinv_gui/backends/` - plain functions, one module per step. The GUI
  never imports the scientific stack directly, so a backend can later be swapped
  for a subprocess / WSL implementation without touching the widgets.
- `mtinv_gui/gui/` - PyQt5. One `BasePanel` subclass per workflow step; panels
  share state through `Project` and re-`refresh()` on `project_changed`.

## License

GNU General Public License v3.0; see [LICENSE](../LICENSE).
