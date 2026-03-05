#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyCrown Simplified – Simplified tree crown segmentation using CHM.

Copyright (C) 2025 Igor Pawelec

This file is part of PyCrown Simplified.

PyCrown Simplified is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

PyCrown Simplified is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details:
<https://www.gnu.org/licenses/>.
"""

__author__    = "Igor Pawelec"
__copyright__ = "Copyright (C) 2025 Igor Pawelec"
__license__   = "GPLv3"

import os
import numpy as np


# ── Lazy imports ──────────────────────────────────────────────────────

def _ensure_fiona():
    try:
        import fiona
        return fiona
    except ImportError as e:
        raise ImportError(
            "fiona is required for vector I/O (GeoPackage export).\n"
            "Install with:  conda install -c conda-forge fiona\n"
            "Or:            pip install fiona"
        ) from e
    except OSError as e:
        raise OSError(
            "fiona found but failed to load (DLL/shared library error).\n"
            "Fix:  conda install -c conda-forge fiona --force-reinstall\n"
            f"Original error: {e}"
        ) from e


def _ensure_rasterio_features():
    try:
        from rasterio.features import shapes
        return shapes
    except ImportError as e:
        raise ImportError(
            "rasterio is required for raster vectorization.\n"
            "Install with:  conda install -c conda-forge rasterio"
        ) from e


def _ensure_morphology():
    from skimage.morphology import binary_closing, disk
    return binary_closing, disk


def save_segments(segments: np.ndarray,
                  out_path: str,
                  fname: str,
                  transform,
                  crs_wkt: str,
                  chm_array: np.ndarray,
                  closing_radius: int = 0) -> None:
    """
    Zapis segmentów koron:
     - RAW (.bin + .vrt)
     - GeoPackage z poligonami i atrybutami:
         id, max_height, area_m2, crown_diameter
    Jeśli closing_radius>0, na każdy segment nakładamy binary_closing
    z elementem strukturalnym disk(closing_radius), żeby wygładzić krawędzie.
    """
    fiona = _ensure_fiona()
    shapes = _ensure_rasterio_features()

    # --- 1) RAW dump (.bin) + VRT ---
    bin_path = os.path.join(out_path, f"{fname}.bin")
    vrt_path = os.path.join(out_path, f"{fname}.vrt")
    segments.astype(np.int32).tofile(bin_path)

    rows, cols = segments.shape
    pixel_area = abs(transform.a * transform.e)
    gtx = (transform.c, transform.a, 0, transform.f, 0, -transform.e)
    with open(vrt_path, "w") as f:
        f.write(f'<VRTDataset rasterXSize="{cols}" rasterYSize="{rows}">\n')
        f.write(f'  <SRS>{crs_wkt}</SRS>\n')
        f.write(f'  <GeoTransform>{",".join(map(str, gtx))}</GeoTransform>\n')
        f.write('  <VRTRasterBand dataType="Int32" band="1">\n')
        f.write(f'    <SourceFilename relativeToVRT="1">{fname}.bin</SourceFilename>\n')
        f.write('    <ImageOffset>0</ImageOffset>\n')
        f.write(f'    <PixelOffset>4</PixelOffset>\n')
        f.write(f'    <LineOffset>{4*cols}</LineOffset>\n')
        f.write('  </VRTRasterBand>\n')
        f.write('</VRTDataset>\n')

    # --- 2) Wektorowanie + atrybuty ---
    gpkg_path = os.path.join(out_path, f"{fname}.gpkg")
    schema = {
        'geometry': 'Polygon',
        'properties': {
            'id': 'int',
            'max_height': 'float',
            'area_m2': 'float',
            'crown_diameter': 'float'
        }
    }

    segment_ids = np.unique(segments)
    segment_ids = segment_ids[segment_ids != 0]

    # Lazy morphology — only if needed
    if closing_radius > 0:
        binary_closing, disk = _ensure_morphology()

    with fiona.open(
        gpkg_path,
        'w',
        driver='GPKG',
        crs_wkt=crs_wkt,
        schema=schema
    ) as dst:
        for seg_id in segment_ids:
            seg_mask = (segments == seg_id)

            if closing_radius > 0:
                seg_mask = binary_closing(seg_mask, disk(closing_radius))

            arr = np.where(seg_mask, seg_id, 0).astype(np.int32)

            for geom, val in shapes(arr, mask=seg_mask, transform=transform):
                if int(val) != seg_id:
                    continue

                max_h = float(chm_array[seg_mask].max())
                area  = float(seg_mask.sum() * pixel_area)
                diam  = float(2 * np.sqrt(area / np.pi))

                props = {
                    'id': seg_id,
                    'max_height': round(max_h, 2),
                    'area_m2':    round(area,   2),
                    'crown_diameter': round(diam, 2)
                }

                dst.write({
                    'geometry': geom,
                    'properties': props
                })


def save_tree_tops(corrected_tops: np.ndarray,
                   out_path: str,
                   fname: str,
                   transform,
                   crs_wkt: str,
                   chm: np.ndarray) -> None:
    """
    Zapis skorygowanych tree-tops do GeoPackage z kolumnami: id, height.
    """
    fiona = _ensure_fiona()

    gpkg_path = os.path.join(out_path, fname + "_treetops.gpkg")
    schema = {
        'geometry': 'Point',
        'properties': {
            'id': 'int',
            'height': 'float'
        }
    }

    coords = np.array(corrected_tops, dtype=float)

    rows = coords[:, 0].astype(int)
    cols = coords[:, 1].astype(int)
    heights = chm[rows, cols]
    heights = np.round(heights, 2)

    with fiona.open(
        gpkg_path,
        'w',
        driver='GPKG',
        crs_wkt=crs_wkt,
        schema=schema
    ) as dst:
        for idx, (r, c) in enumerate(coords):
            x, y = transform * (float(c), float(r))
            geom = {"type": "Point", "coordinates": (x, y)}
            dst.write({
                'geometry': geom,
                'properties': {
                    'id': idx,
                    'height': float(heights[idx])
                }
            })
