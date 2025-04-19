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

import numpy as np
import rasterio
import networkx as nx
from rasterio.features import shapes
from collections import defaultdict

class HierarchicalRegionGrower:
    """
    Implements a hybrid watershed-based, graph-driven, hierarchical region grows
    from seed markers (tree tops) on CHM raster to delineate individual crowns.
    """
    def __init__(self, chm_path, smoothing=None):
        # load CHM
        with rasterio.open(chm_path) as src:
            self.chm = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs
        # optional smoothing (e.g., median or gaussian)
        if smoothing:
            self.chm = smoothing(self.chm)
        self.watershed_labels = None
        self.graph = None
        self.region_attrs = {}

    def detect_markers(self, tree_tops_pixels):
        """
        tree_tops_pixels: list of (row, col) indices of seed points
        """
        # empty placeholder: seeds are pixel coordinates
        self.seed_pixels = tree_tops_pixels

    def initial_watershed(self, markers, mask=None):
        """
        Run a simple watershed on -CHM using given markers array
        markers: ndarray of same shape, zero background, unique ints for each seed
        mask: bool mask where watershed applies (e.g., chm > threshold)
        """
        from skimage.segmentation import watershed
        elev = -self.chm
        labels = watershed(elev, markers=markers, mask=mask)
        self.watershed_labels = labels
        return labels

    def build_adjacency_graph(self):
        """
        Buduje graf przyległości segmentów bez użycia Shapely/GeoPandas.
        Dla każdej pary sąsiadujących pikseli o różnych etykietach
        dodajemy krawędź w grafie.
        """
        labels = self.watershed_labels
        nrows, ncols = labels.shape
        G = nx.Graph()

        # Dodajemy wszystkie węzły
        region_ids = np.unique(labels)
        region_ids = region_ids[region_ids != 0]  # pomijamy tło = 0
        G.add_nodes_from(region_ids)

        # Dla przyspieszenia zbieramy krawędzie w zbiór
        edges = set()

        # Sprawdzamy sąsiedztwo 4‑kierunkowe
        for dr, dc in [(0,1),(1,0)]:
            # przesunięcie obrazu
            shifted = np.roll(labels, shift=-dr, axis=0) if dr else labels
            shifted = np.roll(shifted, shift=-dc, axis=1) if dc else shifted

            # maska pikseli, gdzie etykiety są różne i różne od zera
            mask = (labels != shifted) & (labels != 0) & (shifted != 0)

            # wyciągamy pary (min, max) aby uniknąć duplikatów (a,b) i (b,a)
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                a = int(labels[y, x])
                b = int(shifted[y, x])
                edge = (min(a,b), max(a,b))
                edges.add(edge)

        # Dodajemy krawędzie do grafu
        G.add_edges_from(edges)

        # Obliczamy atrybuty regionów (średnia, sigma, pole)
        pixel_area = abs(self.transform.a * self.transform.e)
        for rid in G.nodes:
            mask = (labels == rid)
            vals = self.chm[mask]
            self.region_attrs[rid] = {
                'mean': float(vals.mean()),
                'std':  float(vals.std()),
                'area': float(mask.sum() * pixel_area)
            }

        self.graph = G
        return G

    def hierarchical_grow(self, seed_id, var_threshold=2.0, max_iters=100):
        """
        Grow a region from the seed watershed node seed_id,
        adding neighbors iteratively based on similarity until variance <= var_threshold.
        """
        G = self.graph
        R = {seed_id}
        for _ in range(max_iters):
            # compute current region variance
            vals = []
            mask = np.isin(self.watershed_labels, list(R))
            vals = self.chm[mask]
            if vals.std() <= var_threshold:
                # try to add neighbors
                candidates = set()
                for rid in R:
                    candidates |= set(G.neighbors(rid))
                candidates -= R
                if not candidates:
                    break
                # select candidate with minimal mean difference
                diffs = {c: abs(self.region_attrs[c]['mean'] - vals.mean()) for c in candidates}
                best = min(diffs, key=diffs.get)
                R.add(best)
            else:
                # remove worst contributor: region in R with max variance
                parts = dict()
                for rid in R:
                    vals_r = self.chm[self.watershed_labels==rid]
                    parts[rid] = vals_r.std()
                worst = max(parts, key=parts.get)
                if worst == seed_id:
                    break
                R.remove(worst)
        # build final mask
        mask_final = np.isin(self.watershed_labels, list(R))
        return mask_final

    def run_all(self, tree_tops_pixels, variance_thresh=2.0):
        """
        Full pipeline: detect markers, watershed, build graph, grow regions
        returns list of binary masks for each tree
        """
        # assign marker labels from tree_tops_pixels
        markers = np.zeros_like(self.chm, dtype=np.int32)
        for idx, (r, c) in enumerate(tree_tops_pixels, 1):
            markers[int(r), int(c)] = idx
        # apply watershed
        labels = self.initial_watershed(markers, mask=self.chm>0)
        # build adjacency graph
        self.build_adjacency_graph()
        crowns = []
        for seed_idx in range(1, len(tree_tops_pixels)+1):
            crowns.append(self.hierarchical_grow(seed_idx, var_threshold=variance_thresh))
        return crowns
