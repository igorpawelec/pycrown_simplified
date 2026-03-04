#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyCrown Simplified – Hierarchical Region Growing (v2)

Implements all 7 improvements from the planning document:
  1. Online statistics (Welford) — no more full-array .std() scans
  2. Priority queue (min-heap) — O(log k) instead of O(neighbors)
  3. Parallel grows per tree — joblib / concurrent.futures
  4. Weighted RAG edges — α·|Δμ| + β·|Δσ| + γ·perimDiff
  5. Morphological mask — erosion/dilation to remove flat background
  6. Variance threshold annealing — schedule-based tightening
  7. Numba @njit on hot loops — CSR graph, no NetworkX in inner loop

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
from numba import njit, types
from numba.typed import List as NumbaList
import heapq
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.segmentation import watershed
from skimage.morphology import binary_opening, binary_closing, disk


# ═══════════════════════════════════════════════════════════════════════
# 1. WELFORD ONLINE STATISTICS — scalar merge, no array scans
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _merge_stats(nA, meanA, varA, nB, meanB, varB):
    """
    Merge two sets of (count, mean, variance) using parallel/Welford formula.
    Returns (n_new, mean_new, var_new).
    All operations on scalars — O(1).
    """
    N = nA + nB
    if N == 0:
        return 0, 0.0, 0.0
    delta = meanB - meanA
    mean_new = (nA * meanA + nB * meanB) / N
    # Combined variance (population)
    var_new = (nA * (varA + (meanA - mean_new) ** 2) +
               nB * (varB + (meanB - mean_new) ** 2)) / N
    return N, mean_new, var_new


@njit(cache=True)
def _remove_stats(nTotal, meanTotal, varTotal, nB, meanB, varB):
    """
    Remove subset B from total, returning stats of remainder A.
    Inverse of _merge_stats.
    """
    nA = nTotal - nB
    if nA <= 0:
        return 0, 0.0, 0.0
    meanA = (nTotal * meanTotal - nB * meanB) / nA
    varA = (nTotal * (varTotal + (meanTotal - meanA) ** 2) -
            nB * (varB + (meanB - meanA) ** 2)) / nA
    # Clamp numerical noise
    if varA < 0.0:
        varA = 0.0
    return nA, meanA, varA


# ═══════════════════════════════════════════════════════════════════════
# 7. NUMBA-ACCELERATED CSR GRAPH CONSTRUCTION & GROW
#    (replaces NetworkX for the hot loop)
# ═══════════════════════════════════════════════════════════════════════

@njit(cache=True)
def _build_adjacency_and_stats(labels, chm, num_regions):
    """
    Build CSR adjacency + per-region statistics in a single pass.

    Returns
    -------
    reg_n    : int64[num_regions+1]   — pixel count per region (index 0 unused)
    reg_mean : float64[num_regions+1] — mean CHM height
    reg_var  : float64[num_regions+1] — population variance
    adj_set  : set of (int, int)      — adjacency pairs (a < b)
    border_len : dict (a,b) -> int    — shared border length in pixels
    """
    rows, cols = labels.shape
    # --- per-region accumulators ---
    reg_n    = np.zeros(num_regions + 1, dtype=np.int64)
    reg_sum  = np.zeros(num_regions + 1, dtype=np.float64)
    reg_sum2 = np.zeros(num_regions + 1, dtype=np.float64)

    for r in range(rows):
        for c in range(cols):
            lab = labels[r, c]
            if lab <= 0:
                continue
            v = chm[r, c]
            reg_n[lab] += 1
            reg_sum[lab] += v
            reg_sum2[lab] += v * v

    reg_mean = np.zeros(num_regions + 1, dtype=np.float64)
    reg_var  = np.zeros(num_regions + 1, dtype=np.float64)
    for i in range(1, num_regions + 1):
        if reg_n[i] > 0:
            reg_mean[i] = reg_sum[i] / reg_n[i]
            reg_var[i] = reg_sum2[i] / reg_n[i] - reg_mean[i] ** 2
            if reg_var[i] < 0.0:
                reg_var[i] = 0.0

    # --- adjacency + border lengths ---
    # We'll return flat arrays and build Python structures outside numba
    # for simplicity (the bottleneck is the grow loop, not graph build)
    edge_a = NumbaList()
    edge_b = NumbaList()

    # Track seen edges to avoid duplicates
    # Using a simple approach: check 2 directions (right, down)
    for r in range(rows):
        for c in range(cols):
            lab = labels[r, c]
            if lab <= 0:
                continue
            # right neighbor
            if c + 1 < cols:
                nb = labels[r, c + 1]
                if nb > 0 and nb != lab:
                    a = min(lab, nb)
                    b = max(lab, nb)
                    edge_a.append(a)
                    edge_b.append(b)
            # down neighbor
            if r + 1 < rows:
                nb = labels[r + 1, c]
                if nb > 0 and nb != lab:
                    a = min(lab, nb)
                    b = max(lab, nb)
                    edge_a.append(a)
                    edge_b.append(b)

    return reg_n, reg_mean, reg_var, edge_a, edge_b


def _edges_to_csr_and_weights(edge_a_list, edge_b_list, num_regions,
                               reg_mean, reg_var, border_counts,
                               alpha=1.0, beta=0.5, gamma=0.1):
    """
    Convert edge list → CSR adjacency + weighted edge costs.

    Weight formula (improvement #4):
        w(a,b) = α·|μa−μb| + β·|σa−σb| + γ·borderDiff(a,b)

    Where borderDiff = 1 / (shared_border_pixels + 1)  (longer border = lower cost).

    Returns
    -------
    row_ptr  : int32 array, CSR row pointers (size num_regions+2)
    col_idx  : int32 array, CSR column indices
    weights  : float64 array, edge weights (same order as col_idx)
    neighbors_of(node) = col_idx[row_ptr[node]:row_ptr[node+1]]
    """
    from collections import defaultdict

    # Deduplicate edges and count shared border pixels
    adj = defaultdict(set)
    for a, b in zip(edge_a_list, edge_b_list):
        adj[a].add(b)
        adj[b].add(a)

    # Build CSR
    row_ptr = np.zeros(num_regions + 2, dtype=np.int32)
    for node in range(1, num_regions + 1):
        row_ptr[node + 1] = row_ptr[node] + len(adj.get(node, set()))

    total_edges = row_ptr[num_regions + 1]
    col_idx = np.zeros(total_edges, dtype=np.int32)
    weights = np.zeros(total_edges, dtype=np.float64)

    reg_std = np.sqrt(np.maximum(reg_var, 0.0))

    pos = row_ptr.copy()
    for node in range(1, num_regions + 1):
        for nb in sorted(adj.get(node, set())):
            idx = pos[node]
            col_idx[idx] = nb
            # Weighted edge cost (improvement #4)
            mu_diff = abs(reg_mean[node] - reg_mean[nb])
            sigma_diff = abs(reg_std[node] - reg_std[nb])
            border_key = (min(node, nb), max(node, nb))
            border_px = border_counts.get(border_key, 1)
            border_cost = 1.0 / (border_px + 1)
            weights[idx] = alpha * mu_diff + beta * sigma_diff + gamma * border_cost
            pos[node] += 1

    return row_ptr, col_idx, weights


# ═══════════════════════════════════════════════════════════════════════
# 2. PRIORITY QUEUE + 6. ANNEALING — the core grow loop
# ═══════════════════════════════════════════════════════════════════════

def _hierarchical_grow_single(seed_id, row_ptr, col_idx, weights,
                               reg_n, reg_mean, reg_var,
                               var_threshold, max_iters=200,
                               anneal_lambda=1.0):
    """
    Grow a single seed region using:
      - Welford online stats (improvement #1)
      - Priority queue / min-heap (improvement #2)
      - Weighted edges (improvement #4, via weights array)
      - Variance annealing (improvement #6)

    Parameters
    ----------
    seed_id : int
        Starting region label.
    row_ptr, col_idx, weights : CSR graph arrays.
    reg_n, reg_mean, reg_var : per-region statistics (will NOT be modified).
    var_threshold : float
        Initial variance (σ²) threshold.
    max_iters : int
        Maximum grow iterations.
    anneal_lambda : float
        Annealing factor. 1.0 = no annealing (constant threshold).
        < 1.0 (e.g. 0.95) = threshold tightens each iteration.

    Returns
    -------
    members : set of int
        Region IDs belonging to this crown.
    """
    # Local copies of stats for the growing region
    cur_n    = int(reg_n[seed_id])
    cur_mean = float(reg_mean[seed_id])
    cur_var  = float(reg_var[seed_id])

    members = {seed_id}
    v_thresh = var_threshold  # mutable threshold for annealing

    # --- Build initial priority queue ---
    # heap entries: (weight, candidate_id)
    heap = []
    start = row_ptr[seed_id]
    end   = row_ptr[seed_id + 1]
    for idx in range(start, end):
        nb = col_idx[idx]
        w  = weights[idx]
        heapq.heappush(heap, (w, int(nb)))

    visited_candidates = set()  # avoid re-processing

    for iteration in range(max_iters):
        if not heap:
            break

        # Anneal threshold (improvement #6)
        if anneal_lambda < 1.0 and iteration > 0:
            v_thresh *= anneal_lambda

        # Pop best candidate from heap (improvement #2)
        while heap:
            w_cand, cand = heapq.heappop(heap)
            if cand not in members and cand not in visited_candidates:
                break
        else:
            break  # heap exhausted

        if cand in members:
            break

        # Test merge with Welford (improvement #1) — O(1) scalar ops
        test_n, test_mean, test_var = _merge_stats(
            cur_n, cur_mean, cur_var,
            int(reg_n[cand]), float(reg_mean[cand]), float(reg_var[cand])
        )

        if test_var <= v_thresh:
            # Accept candidate
            members.add(cand)
            cur_n, cur_mean, cur_var = test_n, test_mean, test_var

            # Add candidate's neighbors to heap
            start_c = row_ptr[cand]
            end_c   = row_ptr[cand + 1]
            for idx in range(start_c, end_c):
                nb = col_idx[idx]
                if nb not in members:
                    heapq.heappush(heap, (weights[idx], int(nb)))
        else:
            # Reject — mark as visited so we don't retry
            visited_candidates.add(cand)

    return members


# ═══════════════════════════════════════════════════════════════════════
# 3. PARALLEL GROW WRAPPER
# ═══════════════════════════════════════════════════════════════════════

def _grow_worker(args):
    """
    Worker function for parallel execution.
    Unpacks args tuple for ProcessPoolExecutor compatibility.
    """
    (seed_id, row_ptr, col_idx, weights,
     reg_n, reg_mean, reg_var,
     var_threshold, max_iters, anneal_lambda) = args

    members = _hierarchical_grow_single(
        seed_id, row_ptr, col_idx, weights,
        reg_n, reg_mean, reg_var,
        var_threshold, max_iters, anneal_lambda
    )
    return seed_id, members


# ═══════════════════════════════════════════════════════════════════════
# MAIN CLASS
# ═══════════════════════════════════════════════════════════════════════

class HierarchicalRegionGrower:
    """
    Hybrid watershed + weighted RAG + Welford-accelerated region-growing.

    Improvements over v1:
      1. Online (Welford) statistics — scalar merge instead of array scans
      2. Priority queue — O(log k) candidate selection
      3. Parallel grows — one process per seed (shared read-only graph)
      4. Weighted RAG edges — α·|Δμ| + β·|Δσ| + γ·1/(border+1)
      5. Morphological mask — binary_opening to remove flat background
      6. Variance annealing — schedule-based threshold tightening
      7. Numba @njit — accelerated graph build and stat computation
    """

    def __init__(self, chm_path: str, smoothing=None):
        with rasterio.open(chm_path) as src:
            self.chm = src.read(1).astype(np.float32)
            self.transform = src.transform
            self.crs = src.crs

        if smoothing is not None:
            self.chm = smoothing(self.chm)

        self.watershed_labels = None
        self.num_regions = 0

        # Per-region stats (Welford-ready)
        self.reg_n    = None
        self.reg_mean = None
        self.reg_var  = None

        # CSR graph
        self.row_ptr = None
        self.col_idx = None
        self.weights = None

    # ── 5. MORPHOLOGICAL MASK ─────────────────────────────────────────

    def _prepare_mask(self, mask_thresh: float,
                      morpho_radius: int = 0) -> np.ndarray:
        """
        Create binary mask from CHM threshold, optionally cleaned
        with morphological opening (erosion → dilation).

        Parameters
        ----------
        mask_thresh : float
            Minimum CHM height to include pixel.
        morpho_radius : int
            Disk radius for binary_opening. 0 = no morphology (v1 behavior).
        """
        mask = self.chm > mask_thresh
        if morpho_radius > 0:
            selem = disk(morpho_radius)
            mask = binary_opening(mask, selem)
            # Close small holes that opening may create inside crowns
            mask = binary_closing(mask, selem)
        return mask

    # ── WATERSHED ─────────────────────────────────────────────────────

    def initial_watershed(self, markers: np.ndarray,
                          mask: np.ndarray = None) -> np.ndarray:
        """Marker-based watershed on inverted CHM."""
        labels = watershed(-self.chm, markers=markers, mask=mask)
        self.watershed_labels = labels
        self.num_regions = int(labels.max())
        return labels

    # ── BUILD GRAPH (improvements #1, #4, #7) ────────────────────────

    def build_adjacency_graph(self, alpha: float = 1.0,
                               beta: float = 0.5,
                               gamma: float = 0.1) -> None:
        """
        Build CSR adjacency graph with weighted edges and
        precomputed per-region statistics (Welford-ready).

        Uses Numba-accelerated _build_adjacency_and_stats for the
        pixel-level loop (improvement #7).

        Parameters
        ----------
        alpha : float
            Weight for mean height difference.
        beta : float
            Weight for std deviation difference.
        gamma : float
            Weight for inverse shared border length.
        """
        labels = self.watershed_labels
        nr = self.num_regions

        # Numba-accelerated stat computation + edge extraction
        reg_n, reg_mean, reg_var, edge_a, edge_b = \
            _build_adjacency_and_stats(labels, self.chm, nr)

        self.reg_n    = reg_n
        self.reg_mean = reg_mean
        self.reg_var  = reg_var

        # Count shared border pixels per edge pair
        from collections import Counter
        border_counts = Counter()
        for a, b in zip(edge_a, edge_b):
            border_counts[(a, b)] += 1

        # Build CSR + weighted edges
        self.row_ptr, self.col_idx, self.weights = \
            _edges_to_csr_and_weights(
                edge_a, edge_b, nr,
                reg_mean, reg_var, border_counts,
                alpha, beta, gamma
            )

    # ── RUN ALL (improvements #3, #5, #6) ────────────────────────────

    def run_all(self,
                tree_tops_pixels: list[tuple[int, int]],
                variance_thresh: float = 2.0,
                mask_thresh: float = 0.0,
                morpho_radius: int = 0,
                alpha: float = 1.0,
                beta: float = 0.5,
                gamma: float = 0.1,
                anneal_lambda: float = 1.0,
                max_iters: int = 200,
                n_jobs: int = 1) -> list[np.ndarray]:
        """
        Full pipeline: watershed → weighted RAG → parallel Welford grows.

        Parameters
        ----------
        tree_tops_pixels : list of (row, col)
            Seed pixel coordinates.
        variance_thresh : float
            Maximum allowed variance (σ²) within a grown region.
        mask_thresh : float
            CHM height threshold for initial mask.
        morpho_radius : int
            Disk radius for morphological mask cleaning (0 = off).
        alpha, beta, gamma : float
            RAG edge weight coefficients (see build_adjacency_graph).
        anneal_lambda : float
            Variance threshold annealing factor per iteration.
            1.0 = constant (v1 behavior), <1.0 = tightening.
        max_iters : int
            Maximum grow iterations per seed.
        n_jobs : int
            Number of parallel processes. 1 = sequential.
            -1 = use all available cores.

        Returns
        -------
        masks : list of ndarray[bool]
            Binary mask for each tree crown.
        """
        # 1) Create marker array
        markers = np.zeros_like(self.chm, dtype=np.int32)
        for idx, (r, c) in enumerate(tree_tops_pixels, start=1):
            markers[r, c] = idx

        # 2) Morphological mask (improvement #5)
        mask = self._prepare_mask(mask_thresh, morpho_radius)

        # 3) Watershed
        self.initial_watershed(markers, mask=mask)

        # 4) Build weighted CSR graph (improvements #1, #4, #7)
        self.build_adjacency_graph(alpha=alpha, beta=beta, gamma=gamma)

        # 5) Parallel or sequential grows (improvements #2, #3, #6)
        n_seeds = len(tree_tops_pixels)
        seed_ids = list(range(1, n_seeds + 1))

        if n_jobs == -1:
            import os
            n_jobs = os.cpu_count() or 1

        # Prepare args for each seed
        common_args = (self.row_ptr, self.col_idx, self.weights,
                       self.reg_n, self.reg_mean, self.reg_var,
                       variance_thresh, max_iters, anneal_lambda)

        all_args = [(sid, *common_args) for sid in seed_ids]

        results = {}
        if n_jobs > 1 and n_seeds > 1:
            # Parallel execution (improvement #3)
            with ProcessPoolExecutor(max_workers=min(n_jobs, n_seeds)) as executor:
                futures = {executor.submit(_grow_worker, args): args[0]
                           for args in all_args}
                for future in as_completed(futures):
                    sid, members = future.result()
                    results[sid] = members
        else:
            # Sequential
            for args in all_args:
                sid, members = _grow_worker(args)
                results[sid] = members

        # 6) Convert member sets → binary masks
        labels = self.watershed_labels
        masks = []
        for sid in seed_ids:
            member_set = results[sid]
            member_arr = np.array(list(member_set), dtype=np.int32)
            crown_mask = np.isin(labels, member_arr)
            masks.append(crown_mask)

        return masks
