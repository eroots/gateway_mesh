"""Model plotting: turn a finished inversion model into maps, sections,
GeoTIFFs and publication figures.

Sibling application to the inversion-setup GUI (``run_plot.py``).  It shares
``mtinv_gui``'s Qt helpers, projection code and EDI readers, but runs in its own
environment (Python 3.12 + PyVista / rasterio / geopandas) because it never
needs deal.II.

Layout mirrors the setup GUI:

  * :mod:`.project`   -- ``PlotProject`` dataclasses -> ``*.mtplot.json``
  * :mod:`.backends`  -- plain functions, no Qt (model I/O, georeferencing,
    slicing, GeoTIFF export, matplotlib rendering)
  * :mod:`.gui`       -- one ``BasePanel`` per tab
"""
