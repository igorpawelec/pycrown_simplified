#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyCrown Simplified – Simplified tree crown segmentation using CHM.

Copyright (C) 2025 Igor Pawelec

This file is part of PyCrown Simplified.

PyCrown Simplified is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
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
__version__   = "0.1"

#%% LIBRARIES

import os
import rasterio
import fiona
import numpy as np
import matplotlib.pyplot as plt
from rasterio.features import shapes
from rasterio.plot import show
from skimage.measure import find_contours
from fiona.crs import CRS
from pycrown import PyCrown
from pycrown.io_utils import save_segments, save_tree_tops

#%% READ CHM

#=== PATHS ===#
chm_path = "D:/PyCrown_simplified/test_data/chm_42_2014.tif"
file_name = os.path.basename(chm_path)
OUT_PATH = "D:/PyCrown_simplified/test_data/results"

# === READ RASTER === #
with rasterio.open(chm_path) as src:
    chm = src.read(1)
    meta = src.meta.copy()
    transform = src.transform
    crs = src.crs

# === RASTER INFO === #
print("CHM metadata:")
print(f"- Dimensions: {meta['width']} x {meta['height']}")
print(f"- CRS: {meta['crs']}")
print(f"- Resolution: {transform[0]} x {-transform[4]}")
print(f"- Data type: {meta['dtype']}")
print(f"- NoData value: {meta.get('nodata', 'brak info')}")

# === SWITCH TO NoData (optional) === #
#chm = np.where(chm < 0, np.nan, chm)

# === PLOT === #
plt.figure(figsize=(6, 6))
show(chm, cmap='viridis', title=f'CHM: {file_name}')
plt.show()

# === CREATE PyCrown OBJECT === #
pc = PyCrown(chm_path)

#%% SMOOTH CHM

smoothed = pc.smooth_chm(ws=3, method="median")

# === PLOT === # 
fig, axs = plt.subplots(1, 2, figsize=(10, 6))

# original CHM
axs[0].imshow(chm, cmap='viridis')
axs[0].set_title("Oryginalny CHM")
axs[0].axis('off')

# smoothed CHM
axs[1].imshow(smoothed, cmap='viridis')
axs[1].set_title("Wygładzony CHM")
axs[1].axis('off')

plt.tight_layout()
plt.show()

#%% TREE TOPS DETECTION

# zmień parametry hmin (minimalna wysokość) oraz ws (window size)
tree_tops = pc.tree_detection(hmin=7, ws=5)

# tree_tops to array (row, col)
tree_tops = np.array(tree_tops)
rows = tree_tops[:, 0]
cols = tree_tops[:, 1]

# === PLOT === # 
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(smoothed, cmap='viridis')
ax.scatter(cols, rows, color='red', marker='o', s=20)
ax.set_title(f"CHM: {file_name} z wykrytymi wierzcholkami")
ax.axis('off')
plt.show()

#%% TREE TOPS CORRECTION

# ustaw threshold
corrected_tops = pc.correct_tree_tops(distance_threshold=5.0)

# === PLOT === # 
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(smoothed, cmap='viridis')
ax.scatter(tree_tops[:, 1], tree_tops[:, 0], color='blue', marker='x', s=50, label="Oryginalne")
ax.scatter(corrected_tops[:, 1], corrected_tops[:, 0], color='red', marker='o', s=20, label="Skorygowane")
ax.set_title("Wykryte i skorygowane wierzcholki")
ax.axis('off')
ax.legend()
plt.show()

#%% FILTER SMALL TREES

# wybierz wysokosc [h tree] od ktorej wyfiltrowac korony drzew
hmin = 10.0
corrected_tops = pc.screen_small_trees(hmin=hmin)
corrected_tops = np.array(pc.tree_tops)

# === PLOT === # 
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(smoothed, cmap='viridis')
ax.scatter(corrected_tops[:, 1], corrected_tops[:, 0],
           color='red', marker='o', s=20, label=f"≥ {hmin} m")
ax.set_title(f"Tree tops ≥ {hmin} m")
ax.axis('off')
ax.legend()
plt.show()
#%% WATERSHED (Dalponte 2016) SEGMENTATION

# 'standard' lub 'circ'. Tryb "circ" zwykle daje gładkie korony.
crowns = pc.crown_delineation(mode="circ", th_seed=0.45, th_crown=0.55, th_tree=2.0, max_crown=10.0)
crs_wkt = src.crs.to_wkt()

# === PLOT === # 
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(smoothed, cmap='viridis')

labels = np.unique(crowns)
labels = labels[labels != 0]

for lab in labels:
    mask = (crowns == lab)
    # znajdź kontury tej jednej korony
    contours = find_contours(mask.astype(float), level=0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

ax.set_title("Obrysy koron drzew")
ax.axis('off')
plt.tight_layout()
plt.show()

#%% HIERARCHICAL REGION-GROWING (Pawelec 2025) SEGMENTATION 

crowns = pc.hierarchical_crown_delineation(
    variance_thresh=2.0,
    mask_thresh=9.0
)
crs_wkt = src.crs.to_wkt()

# === PLOT === # 
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(smoothed, cmap='viridis')

labels = np.unique(crowns)
labels = labels[labels != 0]

for lab in labels:
    mask = (crowns == lab)
    # znajdź kontury tej jednej korony
    contours = find_contours(mask.astype(float), level=0.5)
    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], color='red', linewidth=1)

ax.set_title("Obrysy koron drzew")
ax.axis('off')
plt.tight_layout()
plt.show()

#%% SAVE RESULTS

base = file_name + "_crowns"
save_segments(
    segments=crowns,
    out_path=OUT_PATH,
    fname=base,
    transform=transform,
    crs_wkt=crs_wkt,
    chm_array=chm,
    closing_radius=2
)
save_tree_tops(
    corrected_tops=corrected_tops,
    out_path=OUT_PATH,
    fname=file_name,
    transform=transform,
    crs_wkt=crs_wkt,
    chm=chm
)

print("✅ Wyniki zapisane w:", OUT_PATH)
