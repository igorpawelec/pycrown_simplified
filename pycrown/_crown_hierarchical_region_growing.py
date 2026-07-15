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
from numba import njit
from numba.typed import List as NumbaList
import heapq
from concurrent.futures import ProcessPoolExecutor
from skimage.segmentation import watershed
from skimage.morphology import opening, closing, disk


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
    mean_new = (nA * meanA + nB * meanB) / N
    # Combined variance (population)
    var_new = (nA * (varA + (meanA - mean_new) ** 2) +
               nB * (varB + (meanB - mean_new) ** 2)) / N
    return N, mean_new, var_new


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
    reg_cy   : float64[num_regions+1] — centroid row coordinate
    reg_cx   : float64[num_regions+1] — centroid column coordinate
    edge_a   : NumbaList[int]         — edge endpoints (a < b), with duplicates
    edge_b   : NumbaList[int]         — duplicate count = shared border length
    """
    rows, cols = labels.shape
    # --- per-region accumulators ---
    reg_n     = np.zeros(num_regions + 1, dtype=np.int64)
    reg_sum   = np.zeros(num_regions + 1, dtype=np.float64)
    reg_sum2  = np.zeros(num_regions + 1, dtype=np.float64)
    reg_sumr  = np.zeros(num_regions + 1, dtype=np.float64)
    reg_sumc  = np.zeros(num_regions + 1, dtype=np.float64)

    for r in range(rows):
        for c in range(cols):
            lab = labels[r, c]
            if lab <= 0:
                continue
            v = chm[r, c]
            reg_n[lab] += 1
            reg_sum[lab] += v
            reg_sum2[lab] += v * v
            reg_sumr[lab] += r
            reg_sumc[lab] += c

    reg_mean = np.zeros(num_regions + 1, dtype=np.float64)
    reg_var  = np.zeros(num_regions + 1, dtype=np.float64)
    reg_cy   = np.zeros(num_regions + 1, dtype=np.float64)
    reg_cx   = np.zeros(num_regions + 1, dtype=np.float64)
    for i in range(1, num_regions + 1):
        if reg_n[i] > 0:
            reg_mean[i] = reg_sum[i] / reg_n[i]
            reg_var[i] = reg_sum2[i] / reg_n[i] - reg_mean[i] ** 2
            if reg_var[i] < 0.0:
                reg_var[i] = 0.0
            reg_cy[i] = reg_sumr[i] / reg_n[i]
            reg_cx[i] = reg_sumc[i] / reg_n[i]

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

    return reg_n, reg_mean, reg_var, reg_cy, reg_cx, edge_a, edge_b


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
                              anneal_lambda=1.0, retry_rejected=False):
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
    retry_rejected : bool
        Whether a candidate rejected once may be reconsidered if it is
        reached again from another accepted region.

        Rejection is not permanent in principle: merging a large,
        homogeneous neighbour can *lower* the region's variance, which
        can bring a previously rejected candidate back under the
        threshold. Blocking rejected candidates (the default) is
        therefore an approximation — it trades that recall for speed and
        keeps each candidate to a single test.

        True searches more thoroughly at the cost of extra merge tests.
        The effect is confined to intermediate variance thresholds, where
        crowns actively merge: on a 1105-tree synthetic scene it changed
        nothing at var_threshold 2 or 8 (rejections are final anyway) and
        nothing at 120 (everything merges regardless), but changed ~17%
        of the raster at 20. The direction is not predictable — the crown
        count moved both up and down — because absorbing more regions
        also shifts which crown wins each contested region.
        Default False.

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

    # Candidates already tested and rejected. Kept out of the search
    # unless retry_rejected is set — see the docstring for why this is
    # an approximation.
    rejected = set()

    for iteration in range(max_iters):
        if not heap:
            break

        # Anneal threshold (improvement #6)
        if anneal_lambda < 1.0 and iteration > 0:
            v_thresh *= anneal_lambda

        # Pop best candidate from heap (improvement #2)
        cand = -1
        while heap:
            _, c = heapq.heappop(heap)
            if c in members:
                continue
            if c in rejected and not retry_rejected:
                continue
            cand = c
            break
        if cand < 0:
            break  # heap exhausted

        # Test merge with Welford (improvement #1) — O(1) scalar ops
        test_n, test_mean, test_var = _merge_stats(
            cur_n, cur_mean, cur_var,
            int(reg_n[cand]), float(reg_mean[cand]), float(reg_var[cand])
        )

        if test_var <= v_thresh:
            # Accept candidate
            members.add(cand)
            rejected.discard(cand)
            cur_n, cur_mean, cur_var = test_n, test_mean, test_var

            # Add candidate's neighbors to heap
            start_c = row_ptr[cand]
            end_c   = row_ptr[cand + 1]
            for idx in range(start_c, end_c):
                nb = col_idx[idx]
                if nb not in members:
                    heapq.heappush(heap, (weights[idx], int(nb)))
        else:
            rejected.add(cand)

    return members


# ═══════════════════════════════════════════════════════════════════════
# 3. PARALLEL GROW WRAPPER
# ═══════════════════════════════════════════════════════════════════════

# The CSR graph and per-region stats are identical for every seed and are
# never written to during growing. Passing them inside each task would
# re-pickle the whole graph once per seed — for a few thousand trees that
# is hundreds of MB of IPC and makes n_jobs>1 slower than sequential.
# Instead each worker process receives them once, via the pool
# initializer, and tasks carry only seed ids.
_WORKER_CTX = {}


def _init_worker(row_ptr, col_idx, weights, reg_n, reg_mean, reg_var,
                 var_threshold, max_iters, anneal_lambda, retry_rejected):
    """Pool initializer: stash the read-only grow context in this worker."""
    _WORKER_CTX["args"] = (row_ptr, col_idx, weights,
                           reg_n, reg_mean, reg_var,
                           var_threshold, max_iters, anneal_lambda,
                           retry_rejected)


def _grow_chunk(seed_chunk):
    """Grow a batch of seeds using the context stored by _init_worker."""
    args = _WORKER_CTX["args"]
    return [(int(sid), _hierarchical_grow_single(int(sid), *args))
            for sid in seed_chunk]


# ═══════════════════════════════════════════════════════════════════════
# CONFLICT RESOLUTION — deterministic assignment of contested regions
# ═══════════════════════════════════════════════════════════════════════

CONFLICT_RULES = ("height", "distance", "similarity")


def _resolve_conflicts(results, seed_ids, seed_rc, seed_height,
                       reg_mean, reg_cy, reg_cx,
                       rule="height", protect_seeds=False):
    """
    Assign each watershed region to exactly one crown.

    Marker-based watershed yields one region per tree top, so the graph
    nodes *are* the detected trees and growing merges neighbouring trees
    into a single crown. This is how the algorithm corrects over-detected
    tree tops, and it means two crowns routinely claim each other's
    regions. Whichever crown loses a mutual claim is absorbed and
    disappears from the output.

    Growing runs independently per seed, so these claims must be
    arbitrated. Doing it by write order (last writer wins) makes the
    result depend on tree-top ordering; this function decides it on the
    data instead:

    - ``'height'``     — the taller tree wins (higher CHM at its seed
      pixel). Dominant trees overtop their neighbours, so ambiguous
      canopy — and any absorbed tree — is attributed to the taller crown.
    - ``'distance'``   — the nearest seed wins (Euclidean distance from
      region centroid to seed pixel, in pixels). Classic ITC behaviour.
    - ``'similarity'`` — the seed whose apex height is closest to the
      region's mean height wins, i.e. a 20 m region joins the 20 m tree
      rather than the 30 m one. In the spirit of the variance criterion
      that drives the growing itself. (Anchoring on the seed rather than
      on the crown's aggregate mean is deliberate: crowns that claim the
      same regions have the same aggregate mean by construction, which
      would leave the rule undefined exactly when it is needed.)

    Ties within a rule are broken by lower crown id, so the output is
    fully reproducible.

    Parameters
    ----------
    results : dict[int, set[int]]
        seed_id → set of watershed region labels claimed by that seed.
    seed_ids : list[int]
        Seed ids in crown-id order (crown_id = index + 1).
    seed_rc : list[tuple[int, int]]
        Seed pixel (row, col), same order as *seed_ids*.
    seed_height : ndarray[float]
        CHM height at each seed pixel, same order as *seed_ids*.
    reg_mean, reg_cy, reg_cx : ndarray
        Per-region mean height and centroid, indexed by region label.
    rule : {'height', 'distance', 'similarity'}
        Arbitration rule for contested regions.
    protect_seeds : bool
        If True, the region holding a tree top always stays with that
        tree top's crown, so no tree is ever absorbed and every input
        tree top yields a crown. Disables merging.

    Returns
    -------
    assignment : dict[int, int]
        watershed region label → crown id (1-based).
    n_contested : int
        Number of regions claimed by more than one crown.
    """
    if rule not in CONFLICT_RULES:
        raise ValueError(
            f"rule must be one of {CONFLICT_RULES}, got {rule!r}"
        )

    from collections import defaultdict

    # region label → list of crown ids claiming it
    claims = defaultdict(list)
    for crown_id, sid in enumerate(seed_ids, start=1):
        for ws_label in results[sid]:
            claims[ws_label].append(crown_id)

    # Region holding the seed of each crown.
    owner_of_region = {sid: crown_id
                       for crown_id, sid in enumerate(seed_ids, start=1)}

    assignment = {}
    n_contested = 0

    for ws_label, crown_list in claims.items():
        if len(crown_list) == 1:
            assignment[ws_label] = crown_list[0]
            continue

        n_contested += 1

        # Optional: a tree top's own region is never taken from it.
        if protect_seeds:
            owner = owner_of_region.get(ws_label)
            if owner is not None and owner in crown_list:
                assignment[ws_label] = owner
                continue

        # Arbitration. The crown id sits in the sort key so equal scores
        # resolve to the lower id rather than to insertion order.
        if rule == "height":
            # taller seed wins → maximise height, hence the negation
            best = min((-seed_height[cid - 1], cid) for cid in crown_list)
        elif rule == "distance":
            cy, cx = reg_cy[ws_label], reg_cx[ws_label]
            best = min(
                ((cy - seed_rc[cid - 1][0]) ** 2 +
                 (cx - seed_rc[cid - 1][1]) ** 2, cid)
                for cid in crown_list
            )
        else:  # similarity
            rm = reg_mean[ws_label]
            best = min(
                (abs(seed_height[cid - 1] - rm), cid) for cid in crown_list
            )

        assignment[ws_label] = best[1]

    return assignment, n_contested


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

    Seeds grow independently, so two crowns may claim the same watershed
    region. Contested regions are arbitrated explicitly (see the
    ``conflict_rule`` argument of :meth:`run_all`), which makes the output
    reproducible regardless of seed order.
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

        # Per-region centroids (used by conflict rule 'distance')
        self.reg_cy = None
        self.reg_cx = None

        # Number of regions claimed by >1 crown in the last run_all()
        self.n_contested = 0

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
            mask = opening(mask, selem)
            # Close small holes that opening may create inside crowns
            mask = closing(mask, selem)
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
        reg_n, reg_mean, reg_var, reg_cy, reg_cx, edge_a, edge_b = \
            _build_adjacency_and_stats(labels, self.chm, nr)

        self.reg_n    = reg_n
        self.reg_mean = reg_mean
        self.reg_var  = reg_var
        self.reg_cy   = reg_cy
        self.reg_cx   = reg_cx

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
                conflict_rule: str = "height",
                protect_seeds: bool = False,
                retry_rejected: bool = False,
                n_jobs: int = 1) -> np.ndarray:
        """
        Full pipeline: watershed → weighted RAG → Welford grows → arbitration.

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
        conflict_rule : {'height', 'distance', 'similarity'}
            How to arbitrate watershed regions claimed by more than one
            crown. Watershed gives one region per tree top, so growing
            merges neighbouring trees and the loser of a mutual claim is
            absorbed — this is how over-detected tree tops get corrected,
            and it means the crown count can end up below the tree-top
            count.

            - ``'height'`` (default) — the taller tree wins.
            - ``'distance'`` — the nearest seed wins.
            - ``'similarity'`` — the seed whose height best matches the
              region's mean height wins.

            Ties are broken by lower crown id, so results are reproducible.
            The number of contested regions is stored in
            ``self.n_contested`` after the run.
        protect_seeds : bool
            If True, no tree is ever absorbed: every input tree top keeps
            its own region and yields its own crown. Use when the tree
            tops are trusted (e.g. field-measured) and merging is not
            wanted. Default False.
        retry_rejected : bool
            Allow a rejected region to be reconsidered if reached again
            from another accepted region. Default False, which is faster
            but is an approximation — see
            :func:`_hierarchical_grow_single`.
        n_jobs : int
            Number of parallel processes. 1 = sequential. -1 = all cores.
            The grow is cheap per seed, so this pays off only for large
            scenes; the context is shared per worker rather than per
            task, but process start-up still costs ~0.1 s each.

        Returns
        -------
        crown_raster : ndarray[int32]
            Crown id per pixel (0 = background), same shape as the CHM.
            Crown ids follow the order of *tree_tops_pixels* (1-based);
            absorbed trees leave no pixels, so ids can be missing.
        """
        if conflict_rule not in CONFLICT_RULES:
            raise ValueError(
                f"conflict_rule must be one of {CONFLICT_RULES}, "
                f"got {conflict_rule!r}"
            )

        rows, cols = self.chm.shape
        for r, c in tree_tops_pixels:
            if not (0 <= r < rows and 0 <= c < cols):
                raise ValueError(
                    f"tree top ({r}, {c}) lies outside the CHM "
                    f"({rows}×{cols})"
                )
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

        # Read-only context shared by every seed.
        common_args = (self.row_ptr, self.col_idx, self.weights,
                       self.reg_n, self.reg_mean, self.reg_var,
                       variance_thresh, max_iters, anneal_lambda,
                       retry_rejected)

        results = {}
        if n_jobs > 1 and n_seeds > 1:
            n_workers = min(n_jobs, n_seeds)
            # Several chunks per worker: enough to even out uneven grow
            # times, few enough to keep task overhead negligible.
            n_chunks = min(n_workers * 4, n_seeds)
            chunks = [c for c in np.array_split(np.array(seed_ids), n_chunks)
                      if len(c)]
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=common_args,
            ) as executor:
                for chunk_result in executor.map(_grow_chunk, chunks):
                    for sid, members in chunk_result:
                        results[sid] = members
        else:
            # Sequential — call the grow directly, no worker context needed.
            for sid in seed_ids:
                results[sid] = _hierarchical_grow_single(sid, *common_args)

        # 6) Arbitrate regions claimed by more than one crown.
        # Growing is independent per seed, so overlaps are expected; without
        # this step the assignment would depend on iteration order.
        seed_height = np.array(
            [self.chm[r, c] for r, c in tree_tops_pixels], dtype=np.float64
        )
        assignment, n_contested = _resolve_conflicts(
            results, seed_ids, tree_tops_pixels, seed_height,
            self.reg_mean, self.reg_cy, self.reg_cx,
            rule=conflict_rule, protect_seeds=protect_seeds,
        )
        self.n_contested = n_contested

        # 7) Convert the assignment → label raster in one vectorized pass.
        labels = self.watershed_labels
        max_wlabel = int(labels.max()) + 1
        ws_to_crown = np.zeros(max_wlabel, dtype=np.int32)
        for ws_label, crown_id in assignment.items():
            if 0 <= ws_label < max_wlabel:
                ws_to_crown[ws_label] = crown_id

        crown_raster = ws_to_crown[labels]

        return crown_raster
