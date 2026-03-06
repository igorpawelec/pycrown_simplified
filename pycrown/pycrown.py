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

# ── Lazy imports ──────────────────────────────────────────────────────
# Heavy / C-extension deps are imported on first use, not at module load.
# This prevents "DLL load failed" or "ModuleNotFoundError" from crashing
# the entire package when only a subset of functionality is needed.

_rasterio = None
_ndimage = None
_watershed = None
_cdist = None


def _ensure_rasterio():
    global _rasterio
    if _rasterio is None:
        try:
            import rasterio as _rio
            _rasterio = _rio
        except ImportError as e:
            raise ImportError(
                "rasterio is required for reading CHM files.\n"
                "Install with:  conda install -c conda-forge rasterio\n"
                "Or:            pip install rasterio"
            ) from e
        except OSError as e:
            raise OSError(
                "rasterio found but failed to load (DLL/shared library error).\n"
                "This usually means a version mismatch between rasterio, "
                "GDAL, and PROJ.\n"
                "Fix:  conda install -c conda-forge rasterio --force-reinstall\n"
                f"Original error: {e}"
            ) from e
    return _rasterio


def _ensure_scipy():
    global _ndimage, _cdist
    if _ndimage is None:
        import scipy.ndimage as ndi
        _ndimage = ndi
    if _cdist is None:
        from scipy.spatial.distance import cdist
        _cdist = cdist
    return _ndimage, _cdist


def _ensure_watershed():
    global _watershed
    if _watershed is None:
        from skimage.segmentation import watershed
        _watershed = watershed
    return _watershed


# ── Numba crown algorithms — also lazy ───────────────────────────────
_crown_dalponte_fn = None
_crown_dalponteCIRC_fn = None
_HierarchicalRegionGrower = None


def _ensure_dalponte():
    global _crown_dalponte_fn
    if _crown_dalponte_fn is None:
        from ._crown_dalponte_numba import _crown_dalponte
        _crown_dalponte_fn = _crown_dalponte
    return _crown_dalponte_fn


def _ensure_dalponteCIRC():
    global _crown_dalponteCIRC_fn
    if _crown_dalponteCIRC_fn is None:
        from ._crown_dalponteCIRC_numba import _crown_dalponteCIRC
        _crown_dalponteCIRC_fn = _crown_dalponteCIRC
    return _crown_dalponteCIRC_fn


def _ensure_hrg():
    global _HierarchicalRegionGrower
    if _HierarchicalRegionGrower is None:
        from ._crown_hierarchical_region_growing import HierarchicalRegionGrower
        _HierarchicalRegionGrower = HierarchicalRegionGrower
    return _HierarchicalRegionGrower


# ── Main class ────────────────────────────────────────────────────────

class PyCrown:
    def __init__(self, chm_file=None, chm_array=None, transform=None, crs=None):
        """
        Inicjalizuje obiekt PyCrown.

        Dwa tryby:
          1) Z pliku:   PyCrown("path/to/chm.tif")     — wymaga rasterio
          2) Z tablicy: PyCrown(chm_array=arr, transform=t, crs=c)  — bez I/O

        Parameters
        ----------
        chm_file : str, optional
            Ścieżka do pliku CHM (Canopy Height Model).
        chm_array : ndarray, optional
            CHM jako numpy array (2D, float32/float64).
        transform : affine.Affine, optional
            Geotransformacja (wymagana przy chm_array jeśli chcesz
            eksportować wyniki).
        crs : rasterio.crs.CRS or str, optional
            Układ współrzędnych.
        """
        if chm_file is not None and chm_array is not None:
            raise ValueError("Podaj albo chm_file albo chm_array, nie oba.")

        self.chm_file = chm_file

        if chm_file is not None:
            rasterio = _ensure_rasterio()
            with rasterio.open(chm_file) as src:
                self.chm = src.read(1)
                self.transform = src.transform
                self.crs = src.crs
        elif chm_array is not None:
            self.chm = np.asarray(chm_array)
            self.transform = transform
            self.crs = crs
        else:
            raise ValueError("Musisz podać chm_file lub chm_array.")

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
            Metoda wygładzania: "median", "mean"/"average", "gaussian",
            "maximum".

        Returns
        -------
        ndarray
            Wygładzony CHM.
        """
        ndimage, _ = _ensure_scipy()

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
        ndimage, _ = _ensure_scipy()

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

        Uses scipy.spatial.cKDTree for O(n log n) neighbor queries instead
        of O(n²) cdist.

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

        # KDTree approach: query_ball_point gives all neighbors within radius
        from scipy.spatial import cKDTree
        kd = cKDTree(tree_tops)
        neighbor_lists = kd.query_ball_point(tree_tops, r=distance_threshold)

        # Union-Find for fast grouping
        n = tree_tops.shape[0]
        parent = np.arange(n)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i, neighbors in enumerate(neighbor_lists):
            for j in neighbors:
                if j > i:
                    union(i, j)

        # Group by root and average
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            groups[find(i)].append(i)

        corrected_list = []
        for indices in groups.values():
            pts = tree_tops[indices]
            corrected_list.append(np.mean(pts, axis=0))

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
            _crown_dalponte = _ensure_dalponte()
            self.crowns = _crown_dalponte(
                chm, Trees, float(th_seed), float(th_crown),
                float(th_tree), float(max_crown))
        elif mode == "circ":
            _crown_dalponteCIRC = _ensure_dalponteCIRC()
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
        mask_thresh : float
            Minimum CHM height for initial mask.
        morpho_radius : int
            Disk radius for morphological mask cleaning. 0 = off.
        alpha, beta, gamma : float
            RAG edge weight coefficients.
        anneal_lambda : float
            Variance threshold annealing factor. 1.0 = constant.
        max_iters : int
            Maximum grow iterations per seed.
        n_jobs : int
            Parallel processes. 1 = sequential, -1 = all cores.

        Returns
        -------
        crowns : ndarray[int32]
            Label image: each crown gets value 1..N, background = 0.
        """
        HierarchicalRegionGrower = _ensure_hrg()

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
