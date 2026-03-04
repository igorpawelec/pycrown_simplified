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

import numpy as np
import rasterio
import fiona
import matplotlib.pyplot as plt
from rasterio.features import shapes
from rasterio.plot import show
from skimage.measure import find_contours
from fiona.crs import CRS
from pycrown import PyCrown
from pycrown.io_utils import save_segments, save_tree_tops

#%% PATHS (portable)

from pathlib import Path
import argparse

BASE = Path(__file__).parent
DEFAULT_CHM = BASE / "test_data" / "chm_42_2014.tif"
DEFAULT_OUT = BASE / "test_data" / "results"

parser = argparse.ArgumentParser(description="PyCrown Simplified – test script")
parser.add_argument("--chm", type=Path, default=DEFAULT_CHM, help="Path to CHM raster (.tif)")
parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory for results")
args = parser.parse_args()

args.out.mkdir(parents=True, exist_ok=True)
chm_path = str(args.chm)
file_name = args.chm.name
OUT_PATH = str(args.out)

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
crs_wkt = crs.to_wkt()

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

#%% HIERARCHICAL REGION-GROWING (Pawelec 2025) SEGMENTATION — v2

# ═══════════════════════════════════════════════════════════════════
# Nowe parametry v2 (wszystkie mają domyślne wartości kompatybilne z v1):
#
#   morpho_radius  — promień dysku do morpho-czyszczenia maski.
#                    0 = brak (v1), 2-3 = zalecane dla CHM 1m
#   alpha, beta, gamma — wagi krawędzi RAG:
#                    w(a,b) = α·|Δμ| + β·|Δσ| + γ·1/(border+1)
#   anneal_lambda  — czynnik wygaszania progu wariancji.
#                    1.0 = stały próg (v1), <1.0 = zaostrzanie
#   max_iters      — maks. iteracji na seed (domyślnie 200)
#   n_jobs         — liczba procesów równoległych.
#                    1 = sekwencyjnie (v1), -1 = wszystkie rdzenie
# ═══════════════════════════════════════════════════════════════════

crowns = pc.hierarchical_crown_delineation(
    variance_thresh=2.0,       # próg wariancji (σ²)
    mask_thresh=9.0,           # odcięcie gruntu (CHM > 9m)
    morpho_radius=2,           # [NOWE] morpho cleaning maski
    alpha=1.0,                 # [NOWE] waga różnicy średnich
    beta=0.5,                  # [NOWE] waga różnicy odchyleń std
    gamma=0.1,                 # [NOWE] waga granicy wspólnej
    anneal_lambda=0.9,         # [NOWE] wygaszanie progu (simulated annealing)
    max_iters=200,             # [NOWE] maks. iteracji grow
    n_jobs=1                   # [NOWE] procesy równoległe (-1 = all cores)
)
crs_wkt = crs.to_wkt()

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