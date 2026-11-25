# MT Model Plotting GUI

A sibling app to the inversion-setup GUI. It turns a finished inversion model
into plan views, sections, GeoTIFFs and publication figures.

Both GUIs share one environment; see [Install](README.md#install) in the README.
For figures this GUI doesn't make, notebooks `05_plot_model` and `06_model_to_geotiff` in
[notebooks/](notebooks/README.md) use the same drawing functions from code.

```bash
conda activate mtinv
mtplot
```

| command | what |
|---------|------|
| `mtplot` | model plotting GUI |
| `mtplot-batch project.mtplot.json ...` | export a project's figure recipes (see below) |
| `mtinv` | inversion setup GUI |

The install is *editable*: the commands run the code in this folder, so edits
take effect without reinstalling. `python run_plot.py` / `python run.py` in this
folder work too.


## Status

| Phase | What | State |
|-------|------|-------|
| 1 | Load model; georeference; plan-view slices; GeoTIFF export | **functional**, tested (`tests/test_plotting.py`) |
| 2 | N-S / E-W / arbitrary-path / station sections, index map | **functional**, tested |
| 3 | Other data: rasters, depth surfaces, vector layers, CSV points | **functional**, tested |
| 4 | Figure recipes, house style, batch export, multi-panel layouts | **functional**, tested |

## Tabs

1. **Model.** Pick the code (GoFEM only for now) and the model `.vtu`. The
   receivers file is found automatically (`receivers_projected.csv` next to
   the model or one level up). Then **georeference** it in one of three ways:
   * **Match receivers to EDI coordinates.** Point it at the EDI folder or a
     `.lst`. The receiver names are matched to the EDIs and the origin is
     solved by least squares. `auto` tries the survey's UTM zones and the
     setup GUI's local projections and keeps the lowest-residual one. If none
     fits to within a metre, it also tries the registered national and regional
     grids that cover the survey. The residual tells you whether it's right:
     centimetres if so, kilometres if the CRS is wrong.
   * **From a setup-GUI project.** Reads the frame straight from a
     `.mtinv.json`.
   * **Manual.** Enter the CRS plus the CRS coordinates of the model's (0, 0).

   The **display CRS** is what all figures and GeoTIFFs are drawn in. Blank
   means the model's own CRS. Set it to e.g. `EPSG:32609` to match your other
   rasters, and the model is reprojected for you (no more QGIS warp step).
2. **Layers.** Other datasets, drawn on the plan views, the section index map
   and the sections. **Add files...** works out the kind from the file:
   * **raster**: GeoTIFF, GMT `.grd`, NetCDF, or the setup GUI's `.npz`
     (projected DEM or lon/lat/z);
   * **depth surface**: a raster of depths or elevations, such as Curie depth,
     LAB or basement (guessed from names like "curie" or "depth");
   * **vector**: shp, gpkg, kml, geojson, or a `.gdb` folder via **Add .gdb...**;
   * **point table**: CSV or whitespace text with lon/lat or E/N columns
     (guessed from the column names, or picked), plus optional value, label
     and depth columns.

   Fixed-width text that writes zeros as blanks (`-130. 22103`) is read
   correctly. Every layer is reprojected to the display CRS, so the file's own
   CRS doesn't matter (override it if the file has none). The editor shows
   only the settings for the selected kind:
   * **Reading:** band, layer within the file, valid min / max (anything
     outside is nodata), scale (a multiplier on the values), units, what a
     depth surface's values mean (depth below sea level, depth below surface,
     or elevation), and its **depth units**: m, km, ft, or *auto*.
     - Maps and profiles show the layer in its own unit (km stays km). Only
       its line on a section is converted to metres.
     - *Auto* reads values that are mostly below 300 as km and anything larger
       as m. The preview says which it picked.
     - If a depth line ends up outside the plotted depths, or squashed at the
       top, the section status shows a red warning to check the units. That is
       the km-as-m symptom. CSV depth columns follow the same rule.
   * **On maps:** an image or contours (interval, labels) for rasters; features
     for vectors and points (marker, size, colour, labels from an attribute or
     column). Colour range is auto (2–98 %) or fixed.
   * **On sections:** rasters as a **profile** in a panel above the section.
     Every profile gets its **own y-axis, colour-coded** (spine, ticks and
     label) to match its line, e.g. magnetics left in black,
     gravity right in orange. *Profile axis* = auto takes the left axis, then
     the right. More curves get extra axes offset outward. A curve whose colour
     is already taken gets the next free one, so two default-black layers stay
     distinguishable. Depth surfaces as a **line**. Vectors and points as
     **markers**: points within *max distance* are projected onto the line,
     and lines or polygon edges are marked where they cross it. Points with a
     depth column are plotted at that depth.

   The preview draws the selected layer alone over the stations to check it
   lines up. Profile axes use the layer's whole range, not each profile's, so
   a set of sections shares the same scales.
3. **Plan view.** Depth slices as true cell polygons, never interpolated.
   * Depth is measured either below sea level (negative depths are above sea
     level) or below the model's surface. The second option follows the
     topography.
   * *Crinkle slice* (as in ParaView) draws every cell the depth passes
     through as its whole map footprint, flat, instead of cutting the cells.
     The exact cut can leave slivers, gaps and overlaps where cells are warped
     by the topography fit or meet at hanging nodes; footprints always tile
     the map. Where two cells of a column both reach the depth (a tilted
     boundary), the one holding the depth at its centre is drawn on top; on a
     face, the cell below. Boundaries that cross the depth at a slant come out
     as cell steps: on slices above sea level, cells that only partly rise
     above the depth are drawn whole, so the land outline is blockier and a
     little larger. The setting carries into layouts, recipes and the index
     map's plan background. GeoTIFFs sample points and are unaffected.
   * Options: extent (around the stations, whole model, custom, or "use the
     current preview view"), colour map, discrete colours, resistivity range,
     colour-bar ends (square by default; tick *pointed ends* for the
     beyond-range arrows),
     colour bar, stations (marker, size, colour, names), title template,
     figure size and font.
   * The optional lon/lat grid is labelled on the top and right axes.
   * **Main image**: the model, or any raster layer (e.g. a magnetics map with
     the same stations, extent and overlays). Layers are drawn per their map
     settings.
   * The preview is drawn at the exported figure size in inches, so it looks
     like the saved file. Save one depth, or **Export all depths** using a file
     name template. The file extension picks the format (.png / .pdf / .svg /
     .tif).
4. **Sections.** Sub-tabs along the top split the sections: **N-S**, **E-W**,
   **Paths + stations** and **All**. Each sub-tab has its own list, index map,
   ◀ / ▶ order, *Export all* and *Save to file*, and shows only the buttons
   that add its kind; **All** has every section together, as one combined
   index map. Labels (A, B, ...) are unique across all of them. The look below
   is shared by every sub-tab. Add sections with the buttons:
   * **N-S... / E-W...** One section per value, e.g. `260, 280, 300` km. The
     list fills itself as you change start / step / count (count 1 = a single
     section). The range defaults to the stations plus the map padding.
   * **Draw path.** Left-click points on the index map; right-click or
     **Finish** to end.
   * **Stations...** Load the order from a `.lst`, or click stations on the
     map in order (click the last one again to undo it).
   * **From file...** Read a saved section set (`.json`), a CSV of E/N or
     lon/lat (worked out from the value ranges), or a `.lst`.
   * **Save to file...** Writes the sections on the current sub-tab (their full
     definitions: kind, position, range, stations, direction, x axis, VE) to a
     `.json` section set, for *From file...* here or in another project. Labels
     already in use are renamed on loading. If the display CRS differs, drawn
     paths are converted (they are stored as lon/lat too); N-S / E-W positions
     are kept as numbers, with a warning.

   Step through the sections with **◀ / ▶** above the preview, or Ctrl+← /
   Ctrl+→ and Page Up / Page Down. The selected line is drawn red on the
   index map.

   A station section can run as straight segments, a smooth spline, or a
   best-fit straight line through the stations, extended at both ends. Each
   section has its own label, direction (reverse swaps A / A′), x axis
   (distance, or Easting / Northing where that makes sense) and vertical
   exaggeration. The rest is shared by every section, so a set of figures
   matches:
   * vertical extent, and a depth or elevation axis;
   * colours, stations (projected from up to *max offset* away), surface line
     and A / A′ labels;
   * title and size (height *auto* follows the VE).

   **Rendering** is the true cells by default. *Show interpolated* switches to
   one of: linear, cubic, nearest (Voronoi), RBF (kernel / smoothing /
   epsilon / neighbours), natural neighbour (an
   [optional install](README.md#install)), or Gaussian-smoothed cells. Each
   result is computed once in the background and cached, so flipping back and
   forth is instant. Interpolations are clipped to where the model has earth
   and pinned to the true cells at both section ends.
   Layers (tab 2) add profiles in a panel above the section, sized by
   *profile panel*, plus depth lines and markers (with an optional legend).
   Tick *topography* to add the model surface's elevation along the section
   to that panel, with its own colour-coded axis.
   **Export this / Export all / Export index map.** The index map shows the
   sub-tab's section lines over the model surface, the plan view or a raster
   layer, with the map layers on top. Tick *inset index map* to put a small map
   in a corner of every section figure (preview and exports): the stations,
   the other sections of the same kind (N-S, E-W, or paths + stations) in grey,
   and this one in red, labelled at its A end. *size* is its longer side; it
   shrinks to fit a very flat section. Layout panels leave it out (use an
   index-map panel there). A *+ Index map* recipe on the Figures tab keeps the
   sub-tab it was added from.
5. **GeoTIFF.** One file per depth (`depth_slice_<depth>m.tif`) or one
   multi-band stack. Values can be log10 ρ, ρ, log10 σ or σ.
   Each pixel takes the value of the cell that contains its centre. Pixels in
   air or off the mesh are NaN (nodata). Rasters are written in the display
   CRS, north-up.

6. **Figures.** Figure production:
   * **House style** (tick *use it*): one font, font size, line width,
     width and dpi for every figure in every tab and export. The width presets
     are typical single, 1.5 and double column widths, so check your
     journal's guide. *Editable text* keeps PDF / SVG text as real text
     (TrueType) rather than outlines, so it can be edited in Illustrator or
     Inkscape. *One colour scale* makes plan views, sections, index maps and
     layouts share a single resistivity scale (the per-tab colour boxes grey
     out). If matplotlib doesn't have a font, the Windows font folder (from WSL)
     and the usual Linux font folders are searched, so Arial and Times work.
   * **Figure recipes:** **+ Plan view / + Section / + Index map** save the
     current tab-3 / tab-4 settings as a named figure. **+ Layout...** makes a
     multi-panel page filled with plan views at the depth list, with the
     sections, or with nothing.
     - A recipe keeps its snapshot of the settings. The model, the
       georeferencing, the section lines and the layers stay live.
     - *Update from tabs* re-copies the settings, and *Open in tabs* loads a
       recipe back for editing.
     - Each recipe has an output file. Relative paths resolve next to the
       project, and the extension picks the format.
   * **Layouts:** set rows × columns, then per panel its kind (plan /
     section / index map / empty), depth, main image (model or a raster
     layer), section, and title. Options are a shared colour bar (bottom /
     right / per panel), panel labels ((a), a), A, ...) added before each
     panel's title, size, and spacing.
     **Hide repeated axes** (on by default): a panel drops its y-axis labels
     when the panel to its left shows the same y-range, and its x-axis labels
     when the panel below shows the same x-range. In a grid of plan views
     only the left column and bottom row are labelled, and lon/lat labels only
     the top row and right column. Sections of different lengths, or a section
     next to a map, keep their own axes. Each panel is drawn by the same code as
     the single figure, including the sections' profile panels.
   * **Export selected / Export all recipes** (optionally into one folder).
     The *command line* box shows the equivalent terminal command.

`Project ▸ Save` writes a `*.mtplot.json` holding every setting. The last
project reopens on launch. `mtplot project.mtplot.json` opens that project
instead, and `mtplot --new` starts a blank one in the current folder.

## Redrawing figures for a new iteration (command line)

```bash
conda activate mtinv
mtplot-batch plots.mtplot.json --list
mtplot-batch plots.mtplot.json                       # every recipe, own paths
mtplot-batch plots.mtplot.json --model ZK/model_ZK_61.vtu --out figs_it61
mtplot-batch plots.mtplot.json --only "section A" --only "depths 2x2"
```

`--model` swaps in another model file from the same mesh frame without
editing the project; the stored georeferencing is reused. One failing recipe
doesn't stop the rest, and the exit code is non-zero if any failed.

## Architecture

```
mtinv_gui/plotting/
  project.py              PlotProject -> *.mtplot.json (figure recipes)
  backends/               no Qt
    model_source.py       ModelSource interface; GoFEMVtuSource.  Canonical local
                          frame: x = E, y = N, z = up, cell_data["log10_rho"].
                          Add FEMTIC/ModEM by subclassing + MODEL_SOURCES.
    georef.py             Georef (crs, x0, y0); fits; MapTransform local <-> display
    slicing.py            plan cuts (flat or terrain-following; exact or crinkle), point sampling, LRU cache
    sections.py           section paths, exact per-segment cuts, interpolations, station projection
    layers.py             rasters / surfaces / vectors / CSV points: loading, sampling, on-section items
    geotiff.py            depth lists, RasterGrid, export
    render.py             settings -> matplotlib Figure (GUI preview + export + batch);
                          house style, panel-aware (layout cells are SubFigures)
    figures.py            multi-panel layouts, figure recipes, batch export
    session.py            runtime state: model, receivers, map transform
  gui/                    PyQt5: one PlotPanel (BasePanel + session) per tab
  batch.py                command-line export of a project's recipes
```

GoFEM `.vtu` files are deal.II output: X = North, Y = East, Z = depth (down),
every cell owns its 8 points, and conductivity is point data that is constant
per cell. The loader maps (x, y, z) → (y, x, −z), which is a proper rotation,
so cell orientation is kept, and turns the conductivity into one value per
cell. Sections are cut one path segment at a time: a vertical plane through
the segment, clipped to the segment's two ends. Neighbouring strips meet
exactly at each bend, and a vertex's distance along the path comes straight
from the segment geometry. VTK's `slice_along_line` is not used: it runs on
past the path's ends and creates false surfaces behind bends. The model
surface is found by probing just above each earth cell for
air. That stays correct across the hanging-node faces of an adaptively refined
mesh.
