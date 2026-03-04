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

import numpy as np
import rasterio
import scipy.ndimage as ndimage
from skimage.segmentation import watershed

from ._crown_dalponte_numba import _crown_dalponte
from ._crown_dalponteCIRC_numba import _crown_dalponteCIRC
from ._crown_hierarchical_region_growing import HierarchicalRegionGrower
from scipy.spatial.distance import cdist


class PyCrown:
    def __init__(self, chm_file):
        """
        Inicjalizuje obiekt PyCrown poprzez wczytanie CHM z podanej ścieżki.

        Parameters
        ----------
        chm_file : str
            Ścieżka do pliku CHM (Canopy Height Model)
        """
        self.chm_file = chm_file
        with rasterio.open(chm_file) as src:
            self.chm = src.read(1)
            self.transform = src.transform
            self.crs = src.crs
        self.smoothed_chm = None
        self.tree_tops = None
        self.crowns = None
        self._hrg = None

    def smooth_chm(self, ws=3, method="median"):
        """
        Wygładza CHM przy użyciu różnych metod filtrowania.

        Parameters
        ----------
        ws : int, optional
            Rozmiar okna filtra (domyślnie 3). Dla filtru gaussowskiego
            sigma zostanie ustawione jako ws/3.
        method : str, optional
            Metoda wygładzania. Dostępne opcje:
              - "median": filtr medianowy (odporny na szumy),
              - "mean" lub "average": filtr średniej (uniform filter),
              - "gaussian": filtr gaussowski,
              - "maximum": filtr maksymalny.

        Returns
        -------
        ndarray
            Wygładzony CHM.
        """
        if method == "median":
            self.smoothed_chm = ndimage.median_filter(self.chm, size=ws)
        elif method in ("mean", "average"):
            self.smoothed_chm = ndimage.uniform_filter(self.chm, size=ws)
        elif method == "gaussian":
            sigma = ws / 3.0
            self.smoothed_chm = ndimage.gaussian_filter(self.chm, sigma=sigma)
        elif method == "maximum":
            self.smoothed_chm = ndimage.maximum_filter(self.chm, size=ws)
        else:
            raise ValueError(f"Nieznana metoda wygładzania: {method}")

        return self.smoothed_chm

    def tree_detection(self, hmin=2, ws=3):
        """
        Wykrywa wierzchołki drzew w wygładzonym CHM.

        Parameters
        ----------
        hmin : float, optional
            Minimalna wartość CHM, aby piksel mógł być drzewem (domyślnie 2)
        ws : int, optional
            Rozmiar okna lokalnego filtru (domyślnie 3)

        Returns
        -------
        list of tuples
            Lista współrzędnych (wiersz, kolumna) wykrytych tree tops.
        """
        if self.smoothed_chm is None:
            self.smooth_chm(ws=ws)
        local_max = ndimage.maximum_filter(self.smoothed_chm, size=ws)
        detected = (self.smoothed_chm == local_max) & (self.smoothed_chm > hmin)
        labels, num = ndimage.label(detected)
        centers = ndimage.center_of_mass(self.smoothed_chm, labels, range(1, num + 1))
        self.tree_tops = centers
        return centers

    def correct_tree_tops(self, distance_threshold=5.0):
        """
        Korekta pozycji tree tops przez uśrednianie pozycji lokalnych maximów
        w obrębie jednego drzewa.

        Parameters
        ----------
        distance_threshold : float, optional
            Maksymalna odległość (w pikselach) między punktami, które będą
            uznane za należące do tego samego drzewa. Domyślnie 5.0.

        Returns
        -------
        corrected_tops : ndarray, shape (m, 2)
            Skorygowane pozycje tree tops, gdzie m ≤ n.
        """
        tree_tops = self.tree_tops
        if not isinstance(tree_tops, np.ndarray):
            tree_tops = np.array(tree_tops)

        if tree_tops.shape[0] < 2:
            return tree_tops

        groups = []
        used = np.zeros(tree_tops.shape[0], dtype=bool)
        for i in range(tree_tops.shape[0]):
            if used[i]:
                continue
            group = [i]
            used[i] = True
            expanded = True
            while expanded:
                expanded = False
                current_indices = np.array(group)
                current_points = tree_tops[current_indices]
                remaining_indices = np.where(~used)[0]
                if remaining_indices.size == 0:
                    break
                remaining_points = tree_tops[remaining_indices]
                dists = cdist(remaining_points, current_points)
                close_points = remaining_indices[np.any(dists < distance_threshold, axis=1)]
                if close_points.size > 0:
                    for idx in close_points:
                        group.append(idx)
                        used[idx] = True
                    expanded = True
            groups.append(group)

        corrected_list = []
        for group in groups:
            pts = tree_tops[group]
            mean_pt = np.mean(pts, axis=0)
            corrected_list.append(mean_pt)
        corrected_tops = np.array(corrected_list)
        self.tree_tops = corrected_tops
        return corrected_tops

    def crown_delineation(self, mode="standard", th_seed=0.7, th_crown=0.55,
                          th_tree=15.0, max_crown=10.0):
        """
        Segmentuje korony drzew metodą Dalponte.

        Parameters
        ----------
        mode : str
            "standard" lub "circ" (wersja kołowa).
        th_seed : float
            Threshold dla seed pixel.
        th_crown : float
            Threshold dla średniej wysokości korony.
        th_tree : float
            Minimalna wysokość drzewa.
        max_crown : float
            Maksymalny promień korony w pikselach.

        Returns
        -------
        ndarray[int32]
            Raster koron drzew.
        """
        if self.smoothed_chm is None:
            raise ValueError("CHM must be smoothed first using smooth_chm().")

        if self.tree_tops is None:
            self.tree_tops = self.tree_detection()

        arr = np.array(self.tree_tops)
        Trees = np.vstack((np.floor(arr[:, 0]),
                           np.floor(arr[:, 1]))).astype(np.int32)
        chm = self.smoothed_chm.astype(np.float32)

        if mode == "standard":
            self.crowns = _crown_dalponte(
                chm, Trees, float(th_seed), float(th_crown),
                float(th_tree), float(max_crown))
        elif mode == "circ":
            self.crowns = _crown_dalponteCIRC(
                chm, Trees, float(th_seed), float(th_crown),
                float(th_tree), float(max_crown))
        else:
            raise ValueError("Mode must be 'standard' or 'circ'")
        return self.crowns

    def screen_small_trees(self, hmin: float = 2.0):
        """
        Usuwa drzewa niższe niż hmin.

        Parameters
        ----------
        hmin : float
            Minimalna wysokość drzewa.

        Returns
        -------
        tuple
            (tree_tops, crowns)
        """
        kept = []
        for pt in self.tree_tops:
            r, c = int(pt[0]), int(pt[1])
            if self.smoothed_chm[r, c] >= hmin:
                kept.append(pt)
        self.tree_tops = np.array(kept)

        if self.crowns is not None:
            new_label = 1
            new_crowns = np.zeros_like(self.crowns, dtype=np.int32)
            for old_label, pt in enumerate(kept, start=1):
                mask = self.crowns == old_label
                if mask.any():
                    new_crowns[mask] = new_label
                    new_label += 1
            self.crowns = new_crowns

        return self.tree_tops, self.crowns

    def hierarchical_crown_delineation(
            self,
            variance_thresh: float = 2.0,
            mask_thresh: float = 0.0,
            morpho_radius: int = 0,
            alpha: float = 1.0,
            beta: float = 0.5,
            gamma: float = 0.1,
            anneal_lambda: float = 1.0,
            max_iters: int = 200,
            n_jobs: int = 1
    ) -> np.ndarray:
        """
        Hierarchical watershed + weighted RAG + Welford region-growing (v2).

        This method implements all 7 improvements from the planning document:
          1. Online statistics (Welford) — scalar merge, no array scans
          2. Priority queue — O(log k) candidate selection
          3. Parallel grows — optional multiprocessing per seed
          4. Weighted RAG edges — α·|Δμ| + β·|Δσ| + γ·1/(border+1)
          5. Morphological mask — binary opening/closing to clean mask
          6. Variance annealing — λ-based threshold tightening
          7. Numba @njit — accelerated graph build and statistics

        Parameters
        ----------
        variance_thresh : float
            Maximum allowed variance (σ²) within a grown region.
            Higher = more permissive merging. Default 2.0.
        mask_thresh : float
            Minimum CHM height for initial mask (ground rejection).
            Default 0.0.
        morpho_radius : int
            Disk radius for morphological mask cleaning.
            0 = no morphology (v1 compatible behavior).
            Recommended: 2–3 for 1m resolution CHM.
        alpha : float
            Weight for mean height difference in RAG edges. Default 1.0.
        beta : float
            Weight for std deviation difference in RAG edges. Default 0.5.
        gamma : float
            Weight for inverse shared border length. Default 0.1.
        anneal_lambda : float
            Variance threshold annealing factor per iteration.
            1.0 = constant threshold (v1 behavior).
            <1.0 (e.g. 0.95) = threshold tightens, starting permissive
            then consolidating. Default 1.0.
        max_iters : int
            Maximum grow iterations per seed. Default 200.
        n_jobs : int
            Number of parallel processes.
            1 = sequential (v1 behavior).
            -1 = use all CPU cores.

        Returns
        -------
        crowns : ndarray[int32]
            Label image: each crown gets value 1..N, background = 0.

        Notes
        -----
        For v1-compatible behavior, use default parameters (all improvements
        are backward-compatible: morpho_radius=0, anneal_lambda=1.0, n_jobs=1).

        For maximum performance on large rasters:
            hierarchical_crown_delineation(
                morpho_radius=2,
                anneal_lambda=0.95,
                n_jobs=-1
            )
        """
        if self.smoothed_chm is None:
            self.smooth_chm(ws=3, method="median")

        seeds = [(int(r), int(c)) for r, c in self.tree_tops]

        self._hrg = HierarchicalRegionGrower(
            chm_path=self.chm_file,
            smoothing=lambda arr: self.smoothed_chm
        )
        masks = self._hrg.run_all(
            tree_tops_pixels=seeds,
            variance_thresh=variance_thresh,
            mask_thresh=mask_thresh,
            morpho_radius=morpho_radius,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            anneal_lambda=anneal_lambda,
            max_iters=max_iters,
            n_jobs=n_jobs
        )

        h, w = self.smoothed_chm.shape
        lbl = np.zeros((h, w), dtype=np.int32)
        for i, m in enumerate(masks, start=1):
            lbl[m] = i

        self.crowns = lbl
        return lbl
