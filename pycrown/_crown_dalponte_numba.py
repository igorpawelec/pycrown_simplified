# _crown_dalponte_numba.py
"""
PyCrown - Fast raster-based individual tree segmentation for LiDAR data
Standard Dalponte algorithm implementation (using Numba)
Copyright: 2018, Jan Zörner (modified for Python 3.12 & Numba)
Licence: GNU GPLv3
"""

import numpy as np
from numba import jit, float32, int32, float64

@jit(int32[:, :](float32[:, :], int32[:, :], float64, float64, float64, float64),
     nopython=True, nogil=True, parallel=False)
def _crown_dalponte(Chm, Trees, th_seed, th_crown, th_tree, max_crown):
    """
    Standard Dalponte crown delineation using Numba.
    
    Parameters
    ----------
    Chm : float32[:, :]
          Canopy Height Model raster.
    Trees : int32[:, :]
            Tree top pixel coordinates (2 x n array).  
    th_seed : float64
              Threshold multiplier for seed pixel.
    th_crown : float64
               Threshold multiplier for mean crown height.
    th_tree : float64
              Minimum height to qualify as tree.
    max_crown : float64
                Maximum crown radius in pixels.

    Returns
    -------
    int32[:, :]
         Raster with tree crowns, each labelled with a unique integer.
    """
    grown = True
    nrow = Chm.shape[0]
    ncol = Chm.shape[1]
    ntops = Trees.shape[1]
    # npixel will track number of pixels added per crown
    npixel = np.ones(ntops, dtype=np.float32)
    # Convert tree coordinates to integers (pixel indices)
    tidx_x = np.floor(Trees[0]).astype(np.int32)
    tidx_y = np.floor(Trees[1]).astype(np.int32)
    Crowns = np.zeros((nrow, ncol), dtype=np.int32)
    sum_height = np.zeros(ntops, dtype=np.float64)
    
    # Initialize tree seeds
    for i in range(ntops):
        Crowns[tidx_y[i], tidx_x[i]] = i + 1
        sum_height[i] = Chm[tidx_y[i], tidx_x[i]]
    CrownsTemp = Crowns.copy()
    
    while grown:
        grown = False
        for row in range(1, nrow - 1):
            for col in range(1, ncol - 1):
                if Crowns[row, col] != 0:
                    tidx = Crowns[row, col] - 1
                    seed_y = tidx_y[tidx]
                    seed_x = tidx_x[tidx]
                    seed_h = Chm[seed_y, seed_x]
                    mh_crown = sum_height[tidx] / npixel[tidx]
                    
                    # Define coordinates for 4-connected neighbours
                    # (up, left, right, down)
                    if Crowns[row - 1, col] == 0:
                        nb_h = Chm[row - 1, col]
                        if (nb_h > th_tree and 
                            nb_h > (seed_h * th_seed) and 
                            nb_h > (mh_crown * th_crown) and 
                            nb_h <= (seed_h * 1.05) and 
                            abs(seed_x - col) < max_crown and 
                            abs(seed_y - (row - 1)) < max_crown):
                            CrownsTemp[row - 1, col] = Crowns[row, col]
                            npixel[tidx] += 1
                            sum_height[tidx] += nb_h
                            grown = True
                    if Crowns[row, col - 1] == 0:
                        nb_h = Chm[row, col - 1]
                        if (nb_h > th_tree and 
                            nb_h > (seed_h * th_seed) and 
                            nb_h > (mh_crown * th_crown) and 
                            nb_h <= (seed_h * 1.05) and 
                            abs(seed_x - (col - 1)) < max_crown and 
                            abs(seed_y - row) < max_crown):
                            CrownsTemp[row, col - 1] = Crowns[row, col]
                            npixel[tidx] += 1
                            sum_height[tidx] += nb_h
                            grown = True
                    if Crowns[row, col + 1] == 0:
                        nb_h = Chm[row, col + 1]
                        if (nb_h > th_tree and 
                            nb_h > (seed_h * th_seed) and 
                            nb_h > (mh_crown * th_crown) and 
                            nb_h <= (seed_h * 1.05) and 
                            abs(seed_x - (col + 1)) < max_crown and 
                            abs(seed_y - row) < max_crown):
                            CrownsTemp[row, col + 1] = Crowns[row, col]
                            npixel[tidx] += 1
                            sum_height[tidx] += nb_h
                            grown = True
                    if Crowns[row + 1, col] == 0:
                        nb_h = Chm[row + 1, col]
                        if (nb_h > th_tree and 
                            nb_h > (seed_h * th_seed) and 
                            nb_h > (mh_crown * th_crown) and 
                            nb_h <= (seed_h * 1.05) and 
                            abs(seed_x - col) < max_crown and 
                            abs(seed_y - (row + 1)) < max_crown):
                            CrownsTemp[row + 1, col] = Crowns[row, col]
                            npixel[tidx] += 1
                            sum_height[tidx] += nb_h
                            grown = True
        Crowns[:, :] = CrownsTemp.copy()
    
    return Crowns
