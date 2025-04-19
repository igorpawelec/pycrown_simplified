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

class HierarchicalRegionGrower:
    """
    Hybrid watershed + graph + iterative region-growing from seedów (tree-tops).
    """
    def __init__(self, chm_path: str, smoothing=None):
        # 1) wczytaj CHM
        with rasterio.open(chm_path) as src:
            self.chm = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs

        # 2) opcjonalne wygładzenie
        if smoothing is not None:
            self.chm = smoothing(self.chm)

        self.watershed_labels = None
        self.graph = None
        self.region_attrs = {}

    def initial_watershed(self, markers: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Prosty watershed na -CHM z zadanymi markerami.
        """
        from skimage.segmentation import watershed
        labels = watershed(-self.chm, markers=markers, mask=mask)
        self.watershed_labels = labels
        return labels

    def build_adjacency_graph(self) -> nx.Graph:
        """
        Tworzy graf przyległości segmentów na podstawie etykiet z watershed.
        """
        labels = self.watershed_labels
        G = nx.Graph()

        # 1) dodaj węzły (pomijamy etykietę 0 = tło)
        region_ids = np.unique(labels)
        region_ids = region_ids[region_ids != 0]
        G.add_nodes_from(region_ids)

        # 2) znajdź krawędzie między sąsiednimi regionami (4‑sąsiedztwo)
        edges = set()
        for dr, dc in [(0,1),(1,0)]:
            shifted = np.roll(labels, -dr, axis=0)
            shifted = np.roll(shifted, -dc, axis=1)
            mask = (labels != shifted) & (labels != 0) & (shifted != 0)
            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                a, b = int(labels[y, x]), int(shifted[y, x])
                edges.add((min(a,b), max(a,b)))

        G.add_edges_from(edges)

        # 3) policz cechy każdego regionu
        pixel_area = abs(self.transform.a * self.transform.e)
        for rid in G.nodes:
            region_mask = (labels == rid)
            vals = self.chm[region_mask]
            self.region_attrs[rid] = {
                "mean": float(vals.mean()),
                "std":  float(vals.std()),
                "area": float(region_mask.sum() * pixel_area)
            }

        self.graph = G
        return G

    def hierarchical_grow(self, seed_id: int, var_threshold: float = 2.0, max_iters: int = 100) -> np.ndarray:
        """
        Iteracyjnie dołączaj/usuń sąsiednie regiony na podstawie wariancji i średniej.
        """
        G = self.graph
        R = {seed_id}

        for _ in range(max_iters):
            # 1) oblicz wariancję dla bieżącego zbioru R
            mask_R = np.isin(self.watershed_labels, list(R))
            vals_R = self.chm[mask_R]
            std_R = vals_R.std()

            if std_R <= var_threshold:
                # 2) spróbuj dodać kandydata
                nbrs = set().union(*(G.neighbors(r) for r in R)) - R
                if not nbrs:
                    break

                # wybierz tylko te, dla których wariancja nie wzrośnie ponad threshold
                safe = []
                for c in nbrs:
                    mask_c = mask_R | (self.watershed_labels == c)
                    std_c = self.chm[mask_c].std()
                    if std_c <= var_threshold:
                        mean_diff = abs(self.region_attrs[c]["mean"] - vals_R.mean())
                        safe.append((c, mean_diff))

                if not safe:
                    break

                best = min(safe, key=lambda x: x[1])[0]
                R.add(best)
            else:
                # usuń region o największej wariancji
                worst = max(
                    R,
                    key=lambda rid: self.chm[self.watershed_labels == rid].std()
                )
                if worst == seed_id:
                    break
                R.remove(worst)

        return np.isin(self.watershed_labels, list(R))

    def run_all(self,
                tree_tops_pixels: list[tuple[int,int]],
                variance_thresh: float = 2.0,
                mask_thresh: float = 0.0) -> list[np.ndarray]:
        """
        Pełny pipeline:
          1) watershed(z maską=CHM>mask_thresh),
          2) build graph,
          3) hierarchiczne rozrastanie dla każdego seed_id.
        """
        # 1) stwórz marker-array
        markers = np.zeros_like(self.chm, dtype=np.int32)
        for idx, (r, c) in enumerate(tree_tops_pixels, start=1):
            markers[r, c] = idx

        # 2) watershed
        self.initial_watershed(markers, mask=(self.chm > mask_thresh))

        # 3) graf
        self.build_adjacency_graph()

        # 4) hierarchiczne grow dla każdego drzewa
        return [
            self.hierarchical_grow(seed_id=i, var_threshold=variance_thresh)
            for i in range(1, len(tree_tops_pixels) + 1)
        ]
