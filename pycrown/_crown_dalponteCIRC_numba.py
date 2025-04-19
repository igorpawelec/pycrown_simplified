# _crown_dalponteCIRC_numba.py
"""
PyCrown - Fast raster-based individual tree segmentation for LiDAR data
Circular Dalponte algorithm (using Numba)
Copyright: 2018, Jan Zörner (modified for Python 3.12 & Numba)
Licence: GNU GPLv3
"""

import numpy as np
from numba import jit, int32, float32, float64
from numba.typed import List

@jit(nopython=True)
def get_neighbourhood(radius):
    size = 2 * radius + 1
    # Utwórz macierz odległości
    kernel = np.empty((size, size), dtype=np.int32)
    for i in range(size):
        for j in range(size):
            kernel[i, j] = (i - radius) ** 2 + (j - radius) ** 2

    flat_kernel = kernel.flatten()
    flat_sorted = np.sort(flat_kernel)
    unique = List()
    unique.append(flat_sorted[0])
    for i in range(1, flat_sorted.shape[0]):
        if flat_sorted[i] != flat_sorted[i - 1]:
            unique.append(flat_sorted[i])
    
    # Tworzymy listę unikalnych wartości poza pierwszą
    nums = List()
    for i in range(1, len(unique)):
        nums.append(unique[i])
    
    ne_x = List()
    ne_y = List()
    breaks = List()
    
    for u in nums:
        if u >= (radius * radius):
            continue
        count = 0
        for i in range(size):
            for j in range(size):
                if kernel[i, j] == u:
                    ne_y.append(i - radius)  # wierszowy offset
                    ne_x.append(j - radius)  # kolumnowy offset
                    count += 1
        breaks.append(count)
    
    # Ręczna konwersja typowanych list na NumPy arrays.
    l = len(ne_x)
    ne_x_arr = np.empty(l, dtype=np.int32)
    for i in range(l):
        ne_x_arr[i] = ne_x[i]
    
    l2 = len(ne_y)
    ne_y_arr = np.empty(l2, dtype=np.int32)
    for i in range(l2):
        ne_y_arr[i] = ne_y[i]
    
    lb = len(breaks)
    breaks_arr = np.empty(lb, dtype=np.int32)
    for i in range(lb):
        breaks_arr[i] = breaks[i]
    
    return ne_x_arr, ne_y_arr, breaks_arr

# The remainder of the file (e.g. the _crown_dalponteCIRC function)
# remains unchanged from your earlier implementation.
@jit(int32[:, :](float32[:, :], int32[:, :], float64, float64, float64, float64),
     nopython=True, nogil=True, parallel=False)
def _crown_dalponteCIRC(Chm, Trees, th_seed, th_crown, th_tree, max_crown):
    """
    Circular Dalponte crown delineation algorithm using Numba.

    Parameters
    ----------
    Chm : float32[:, :]
          CHM raster.
    Trees : int32[:, :]
            Tree top pixel coordinates as a 2 x n array.
    th_seed : float64
              Threshold for the seed pixel.
    th_crown : float64
               Threshold for the crown mean height.
    th_tree : float64
              Minimum height required for a pixel to be considered a tree.
    max_crown : float64
                Maximum crown radius in pixels.

    Returns
    -------
    int32[:, :]
         Raster of tree crowns.
    """
    ntops = Trees.shape[1]
    npixel = np.ones(ntops, dtype=np.float32)
    tidx_x = np.floor(Trees[0]).astype(np.int32)
    tidx_y = np.floor(Trees[1]).astype(np.int32)
    nrows = Chm.shape[0]
    ncols = Chm.shape[1]
    Crowns = np.zeros((nrows, ncols), dtype=np.int32)
    sum_height = np.zeros(ntops, dtype=np.float64)

    for i in range(ntops):
        Crowns[tidx_y[i], tidx_x[i]] = i + 1
        sum_height[i] = Chm[tidx_y[i], tidx_x[i]]

    tree_idx = np.arange(ntops)
    # Preallocate a 4x2 array for local neighbours (will be reused inside loop)
    neighbours = np.zeros((4, 2), dtype=np.int32)

    # Precompute circular neighbourhood offsets.
    ne_x, ne_y, breaks = get_neighbourhood(int(max_crown))
    step = 0
    for k in range(breaks.shape[0]):
        n_neighbours = breaks[k]
        grown = False
        for tidx in tree_idx:
            seed_y = tidx_y[tidx]
            seed_x = tidx_x[tidx]
            seed_h = Chm[seed_y, seed_x]
            mh_crown = sum_height[tidx] / npixel[tidx]
            for n in range(n_neighbours):
                nb_x = seed_x + ne_x[step + n]
                nb_y = seed_y + ne_y[step + n]
                if nb_x < 0 or nb_x >= ncols or nb_y < 0 or nb_y >= nrows:
                    continue
                nb_h = Chm[nb_y, nb_x]
                if (nb_h > th_tree and 
                    Crowns[nb_y, nb_x] == 0 and 
                    nb_h > (seed_h * th_seed) and 
                    nb_h > (mh_crown * th_crown) and 
                    nb_h <= (seed_h * 1.05) and 
                    abs(seed_x - nb_x) < max_crown and 
                    abs(seed_y - nb_y) < max_crown):
                    # Define coordinates for adjacent neighbours
                    neighbours[0, 0] = nb_y - 1; neighbours[0, 1] = nb_x
                    neighbours[1, 0] = nb_y;     neighbours[1, 1] = nb_x - 1
                    neighbours[2, 0] = nb_y;     neighbours[2, 1] = nb_x + 1
                    neighbours[3, 0] = nb_y + 1; neighbours[3, 1] = nb_x
                    for j in range(4):
                        if Crowns[neighbours[j, 0], neighbours[j, 1]] == tidx + 1:
                            Crowns[nb_y, nb_x] = tidx + 1
                            npixel[tidx] += 1
                            sum_height[tidx] += nb_h
                            grown = True
                            break
        step += n_neighbours
        if not grown:
            break

    return Crowns
