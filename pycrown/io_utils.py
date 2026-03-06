#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyCrown Simplified – I/O utilities for crown segments and tree tops.

Vectorization uses only rasterio.features.shapes + fiona.
No geopandas or shapely required.

Copyright (C) 2025 Igor Pawelec
Licence: GPLv3
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
            "fiona is required for vector I/O.\n"
            "Install with:  conda install -c conda-forge fiona"
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
    from skimage.morphology import closing, disk
    return closing, disk


# ── Public API ────────────────────────────────────────────────────────

def save_segments(segments: np.ndarray,
                  out_path: str,
                  fname: str,
                  transform,
                  crs_wkt: str,
                  chm_array: np.ndarray,
                  closing_radius: int = 0,
                  driver: str = "ESRI Shapefile") -> None:
    """
    Save crown segments as vector file + raw raster dump.

    Uses single-pass vectorization with precomputed per-crown statistics.
    Much faster than per-segment iteration for large numbers of crowns.

    Outputs
    -------
    - RAW raster: {fname}.bin + {fname}.vrt
    - Vector file with attributes: id, max_height, area_m2, crown_diameter
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

    # --- 2) Optional morphological closing on the FULL raster ---
    seg_data = segments.astype(np.int32)
    if closing_radius > 0:
        morph_closing, disk = _ensure_morphology()
        # Close each segment mask — but do it efficiently via label dilation
        # For small numbers of crowns, per-label closing is acceptable
        # For large numbers, we close the binary mask then re-label
        closed = np.zeros_like(seg_data)
        seg_ids = np.unique(seg_data)
        seg_ids = seg_ids[seg_ids != 0]
        selem = disk(closing_radius)
        for sid in seg_ids:
            m = morph_closing(seg_data == sid, selem)
            closed[m & (closed == 0)] = sid
        seg_data = closed

    # --- 3) Precompute per-crown stats (one pass) ---
    max_id = int(seg_data.max()) + 1 if seg_data.max() > 0 else 1
    pixel_counts = np.bincount(seg_data.ravel(), minlength=max_id)

    # Max height per crown — use np.maximum.at (in-place, no copy)
    max_heights = np.full(max_id, -np.inf, dtype=np.float32)
    flat_seg = seg_data.ravel()
    flat_chm = chm_array.ravel()
    if flat_chm.dtype != np.float32:
        flat_chm = flat_chm.astype(np.float32)
    np.maximum.at(max_heights, flat_seg, flat_chm)
    max_heights[0] = 0.0  # background

    # --- 4) Single-pass vectorization ---
    ext_map = {"ESRI Shapefile": ".shp", "GPKG": ".gpkg", "GeoJSON": ".geojson"}
    ext = ext_map.get(driver, ".shp")
    vec_path = os.path.join(out_path, f"{fname}{ext}")

    schema = {
        'geometry': 'Polygon',
        'properties': {
            'id': 'int',
            'max_height': 'float',
            'area_m2': 'float',
            'crown_diameter': 'float'
        }
    }

    seg_mask = seg_data > 0
    with fiona.open(
        vec_path,
        'w',
        driver=driver,
        crs_wkt=crs_wkt,
        schema=schema
    ) as dst:
        for geom, val in shapes(seg_data, mask=seg_mask, transform=transform):
            seg_id = int(val)
            if seg_id <= 0:
                continue

            n_px = int(pixel_counts[seg_id])
            area = n_px * pixel_area
            diam = 2.0 * np.sqrt(area / np.pi)
            max_h = float(max_heights[seg_id])

            dst.write({
                'geometry': geom,
                'properties': {
                    'id': seg_id,
                    'max_height': round(max_h, 2),
                    'area_m2':    round(area, 2),
                    'crown_diameter': round(diam, 2)
                }
            })


def save_tree_tops(corrected_tops: np.ndarray,
                   out_path: str,
                   fname: str,
                   transform,
                   crs_wkt: str,
                   chm: np.ndarray,
                   driver: str = "ESRI Shapefile") -> None:
    """
    Save corrected tree tops as point vector file.

    Parameters
    ----------
    corrected_tops : ndarray (n, 2)
        Tree top positions as (row, col) pixel coordinates.
    out_path, fname, transform, crs_wkt, chm : see save_segments.
    driver : str
        Fiona driver: "ESRI Shapefile" (default), "GPKG", "GeoJSON".
    """
    fiona = _ensure_fiona()

    ext_map = {"ESRI Shapefile": ".shp", "GPKG": ".gpkg", "GeoJSON": ".geojson"}
    ext = ext_map.get(driver, ".shp")
    vec_path = os.path.join(out_path, fname + f"_treetops{ext}")

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
        vec_path,
        'w',
        driver=driver,
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
