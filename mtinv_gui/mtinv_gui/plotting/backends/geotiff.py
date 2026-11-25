"""Model -> GeoTIFF depth slices.

Three things to know:

  * pixels are sampled from the cell that *contains* each pixel centre
    (:func:`.slicing.sample_points`) -- exact, no interpolation;
  * the raster is laid out in the **display CRS** directly (rows north ->
    south, no transposes), so a model built in UTM 8N can be written straight
    to UTM 9N instead of being warped afterwards in QGIS;
  * depth can be below sea level or below the model surface.

Outside the earth (air, off the mesh) is NaN = nodata.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import slicing

#: value written per pixel: key -> (label, transform of log10(rho))
QUANTITIES: dict[str, tuple[str, Callable]] = {
    "log10_rho": ("log10 resistivity (log10 ohm-m)", lambda v: v),
    "rho": ("resistivity (ohm-m)", lambda v: 10.0 ** v),
    "log10_sigma": ("log10 conductivity (log10 S/m)", lambda v: -v),
    "sigma": ("conductivity (S/m)", lambda v: 10.0 ** (-v)),
}

LAYOUTS = {"per_depth": "one file per depth", "stack": "one multi-band file"}


# --------------------------------------------------------------------------- #
def parse_depths(text: str) -> list[float]:
    """``"0, 100 250; 1e3"`` -> ``[0.0, 100.0, 250.0, 1000.0]`` (metres)."""
    out = []
    for tok in text.replace(";", ",").replace("\n", ",").replace(" ", ",").split(","):
        tok = tok.strip()
        if tok:
            out.append(float(tok))
    return out


def format_depths(depths) -> str:
    return ", ".join(f"{d:g}" for d in depths)


def generate_depths(start: float, stop: float, n: int, spacing: str = "log",
                    round_to: float = 0.0) -> list[float]:
    """*n* depths from *start* to *stop* (inclusive), linearly or
    logarithmically spaced, optionally rounded to multiples of *round_to*.

    Log spacing needs ``0 < start < stop``; a 0 m slice can be added by hand.
    """
    n = max(int(n), 1)
    if spacing == "log":
        if start <= 0 or stop <= 0:
            raise ValueError("log spacing needs positive start and stop depths")
        vals = np.logspace(math.log10(start), math.log10(stop), n)
    else:
        vals = np.linspace(start, stop, n)
    if round_to > 0:
        vals = np.round(vals / round_to) * round_to
    return sorted(set(float(v) for v in vals))


def depth_tag(depth_m: float) -> str:
    """File-name fragment: ``-130``, ``1000``, ``12p5``."""
    return f"{depth_m:g}".replace(".", "p")


# --------------------------------------------------------------------------- #
@dataclass
class RasterGrid:
    """Pixel grid in the display CRS (north-up)."""
    x_min: float
    y_max: float
    pixel: float
    nx: int
    ny: int

    @classmethod
    def from_bounds(cls, x_min, x_max, y_min, y_max, pixel) -> "RasterGrid":
        pixel = float(pixel)
        if pixel <= 0:
            raise ValueError("pixel size must be positive")
        # snap to whole pixels so neighbouring exports line up
        x0 = math.floor(x_min / pixel) * pixel
        x1 = math.ceil(x_max / pixel) * pixel
        y0 = math.floor(y_min / pixel) * pixel
        y1 = math.ceil(y_max / pixel) * pixel
        nx, ny = int(round((x1 - x0) / pixel)), int(round((y1 - y0) / pixel))
        if nx * ny > 50_000_000:
            raise ValueError(f"{nx} x {ny} pixels is too large -- increase the pixel size "
                             "or shrink the extent")
        return cls(x0, y1, pixel, max(nx, 1), max(ny, 1))

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.x_min + (np.arange(self.nx) + 0.5) * self.pixel
        ys = self.y_max - (np.arange(self.ny) + 0.5) * self.pixel
        return np.meshgrid(xs, ys)

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """``(left, right, bottom, top)`` for ``imshow``."""
        return (self.x_min, self.x_min + self.nx * self.pixel,
                self.y_max - self.ny * self.pixel, self.y_max)

    def affine(self):
        from rasterio.transform import from_origin

        return from_origin(self.x_min, self.y_max, self.pixel, self.pixel)


def sample_slice(model, maptr, grid: RasterGrid, depth_m: float,
                 reference: str = "sea_level") -> np.ndarray:
    """``(ny, nx)`` log10(rho) at one depth (NaN = nodata)."""
    E, N = grid.centres()
    return slicing.sample_depth_grid(model, maptr, depth_m, reference, E, N)


def export(model, maptr, depths, grid: RasterGrid, out_dir: str, *,
           reference: str = "sea_level", quantity: str = "log10_rho",
           layout: str = "per_depth", prefix: str = "depth_slice",
           progress: Callable[[int, int, str], None] | None = None) -> list[str]:
    """Write the GeoTIFF(s); returns the paths written.

    ``per_depth`` -> ``<prefix>_<depth>m.tif`` per depth;
    ``stack`` -> ``<prefix>.tif`` with one band per depth, each band
    described by its depth.
    """
    import rasterio

    if quantity not in QUANTITIES:
        raise ValueError(f"unknown quantity {quantity!r}")
    depths = [float(d) for d in depths]
    if not depths:
        raise ValueError("no depths given")
    os.makedirs(out_dir, exist_ok=True)
    label, fn = QUANTITIES[quantity]
    profile = dict(driver="GTiff", width=grid.nx, height=grid.ny, dtype="float32",
                   crs=maptr.display_crs.to_wkt(), transform=grid.affine(),
                   nodata=np.nan, compress="deflate", predictor=3, tiled=True,
                   blockxsize=256, blockysize=256)
    if grid.nx < 256 or grid.ny < 256:
        profile.update(tiled=False)
        profile.pop("blockxsize")
        profile.pop("blockysize")
    tags = dict(quantity=label, depth_reference=slicing.REFERENCES.get(reference, reference),
                source_model=os.path.basename(model.source_path or ""),
                model_crs=maptr.georef.crs, model_origin=f"{maptr.georef.x0:.3f},{maptr.georef.y0:.3f}")

    written: list[str] = []
    total = len(depths)
    if layout == "stack":
        path = os.path.join(out_dir, f"{prefix}.tif")
        with rasterio.open(path, "w", count=total, **profile) as dst:
            dst.update_tags(**tags, depths_m=",".join(f"{d:g}" for d in depths))
            for b, d in enumerate(depths, start=1):
                if progress:
                    progress(b - 1, total, f"{d:g} m")
                dst.write(fn(sample_slice(model, maptr, grid, d, reference)).astype("float32"), b)
                dst.set_band_description(b, f"{slicing.depth_label(d, reference, 'm')}")
        written.append(path)
    else:
        for i, d in enumerate(depths):
            if progress:
                progress(i, total, f"{d:g} m")
            path = os.path.join(out_dir, f"{prefix}_{depth_tag(d)}m.tif")
            with rasterio.open(path, "w", count=1, **profile) as dst:
                dst.update_tags(**tags, depth_m=f"{d:g}")
                dst.write(fn(sample_slice(model, maptr, grid, d, reference)).astype("float32"), 1)
                dst.set_band_description(1, slicing.depth_label(d, reference, "m"))
            written.append(path)
    if progress:
        progress(total, total, "done")
    return written
