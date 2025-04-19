# PyCrown Simplified
<img src="https://raw.githubusercontent.com/igorpawelec/pycrown_simplified/main/www/pycrown_logo.png" align="right" width="120"/>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**Author:** Igor Pawelec  
**Original implementation:** Dr. Jan Zörner et al. (2018) under GNU GPLv3  
**This fork:** adds a novel hybrid watershed‑plus‑graph hierarchical region‑growing approach  

---

## Summary

PyCrown is a Python toolkit for delineating individual tree crowns from a Canopy Height Model (CHM).  
It implements:

1. **CHM smoothing** (median, mean, Gaussian, maximum filters)  
2. **Tree‑top detection** via local maxima + optional center‑of‑mass correction  
3. **Dalponte crown delineation** in two modes:
   - **standard**: raster‑based region growing  
   - **circ**: circular (“CIRC”) variant for smoother crowns  
4. **New hierarchical segmentation**: a hybrid watershed → region adjacency graph → region‑growing algorithm for especially challenging canopies  
5. **I/O utilities** to export:
   - raw segmentation mask (`.bin` + `.vrt`)  
   - vectorized crowns & attributes to GeoPackage  
   - tree‑top points to GeoPackage  

## Features

- **Flexible smoothing**: choose among four filters to pre‑process your CHM  
- **Robust peak detection**: cluster nearby maxima to correct false tree‑tops  
- **Fast crown growing**: Cython + Numba back‑ends for both standard and circular Dalponte methods  
- **Hierarchical region‑growing**  
  - Watershed segmentation from seed markers  
  - Build a Region Adjacency Graph (RAG) without external GIS libs  
  - Iteratively merge/split basins to meet variance/compactness thresholds  
- **Easy export**: save results as rasters or GeoPackage with crown attributes (max height, area, diameter)  

## Installation

```bash
# From PyPI
pip install pycrown-simplified

# Or from source
git clone https://github.com/yourusername/pycrown_simplified.git
cd pycrown_simplified
pip install -e .
```

## Dependencies:

- Python ≥ 3.12
- numpy ≥ 1.23
- scipy ≥ 1.9
- scikit‑image ≥ 0.20
- rasterio ≥ 1.3
- numba ≥ 0.60
- fiona ≥ 1.9

You can also use Conda:
```
conda env create -f environment.yaml
conda activate pycrown
```
## Quickstart
```
import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np

from pycrown import PyCrown
from pycrown.io_utils import save_segments, save_tree_tops

# Paths
chm_path = "path/to/chm.tif"
out_dir  = "results"
os.makedirs(out_dir, exist_ok=True)

# 1. Initialize and read CHM
pc = PyCrown(chm_path)

# 2. Smooth CHM
smoothed = pc.smooth_chm(ws=5, method="gaussian")

# 3. Detect and correct tree‑tops
raw_tops   = pc.tree_detection(hmin=2.0, ws=5)
tops = pc.correct_tree_tops(distance_threshold=5.0)

# 4. (Optional) Screen small trees
pc.screen_small_trees(hmin=3.0)

# 5. Delineate crowns (use "circ" for smooth crowns)
crowns = pc.crown_delineation(mode="circ",
                              th_seed=0.7,
                              th_crown=0.55,
                              th_tree=2.0,
                              max_crown=10.0)

# 6. Save outputs
with rasterio.open(chm_path) as src:
    transform = src.transform
    crs_wkt    = src.crs.to_wkt()
    chm_array  = src.read(1)

save_segments(crowns, out_dir, "crowns", transform, crs_wkt, chm_array)
save_tree_tops(tops, out_dir, "treetops", transform, crs_wkt, chm_array)
```
## API

PyCrown(chm_file: str)

    smooth_chm(ws: int = 3, method: str = "median") → np.ndarray

    tree_detection(hmin: float = 2.0, ws: int = 3) → list[tuple]

    correct_tree_tops(distance_threshold: float = 5.0) → np.ndarray

    screen_small_trees(hmin: float = 2.0) → (np.ndarray, np.ndarray)

    crown_delineation(mode: str, th_seed: float, th_crown: float, th_tree: float, max_crown: float) → np.ndarray

    hierarchical_crown_delineation(variance_thresh: float, mask_thresh: float) → np.ndarray

See the docstrings for full details.

## Citation

If you use PyCrown in your research, please cite the original implementation:

    Zörner J., Dymond J.R., Shepherd J.D., Wiser S.K., Bunting P., Jolly B.
    PyCrown – Fast raster-based individual tree segmentation for LiDAR data.
    Landcare Research NZ Ltd. 2018. https://doi.org/10.7931/M0SR-DN55

And acknowledge this fork’s novel hierarchical region‑growing approach:

    Pawelec I. (2025) PyCrown Simplified: hybrid watershed + graph hierarchical segmentation.

## License

This project is licensed under the GNU General Public License v3.0. See LICENSE for details.