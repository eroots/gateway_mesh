"""MT inversion setup GUI.

A desktop tool for turning a folder of EDI files into the input files required
by a 3-D magnetotelluric inversion.  The first supported target is GoFEM; the
architecture leaves room for ModEM and FEMTIC writers later.

Workflow:

  1. Load EDI files -> station table, survey bounding box, period coverage.
  2. Download topography / bathymetry for the survey region.            (task 1)
  3. Project + interpolate the DEM into the model's local UTM frame.    (task 2)
  4. Design the inversion mesh.                                         (task 3)
  5. Write the solver input files (frequencies, data, parameter files).
"""

__version__ = "0.0.69"
