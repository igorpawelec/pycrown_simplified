# PyCrown Simplified

<img src="https://raw.githubusercontent.com/igorpawelec/pycrown_simplified/main/www/pycrown_logo.png" align="right" width="120"/>

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE.md)

**Individual tree crown segmentation from Canopy Height Models (CHM).**

A simplified and extended Python toolkit for delineating individual tree crowns from LiDAR-derived CHM data. Implements the Dalponte & Coomes (2016) region-growing algorithm via Numba JIT, plus a novel hierarchical watershed + graph-based region-growing approach.

## Background

This package builds on two prior works:

1. **The Dalponte & Coomes (2016) algorithm** for tree-centric crown delineation using CHM region growing:

   > Dalponte, M. & Coomes, D.A. (2016). Tree-centric mapping of forest carbon density from airborne laser scanning and hyperspectral data. *Methods in Ecology and Evolution*, 7, 1236–1245. https://doi.org/10.1111/2041-210X.12575

2. **The PyCrown implementation** by Jan Zörner et al. (2018), which provided the original Python/Cython/Numba code for both the standard and circular Dalponte variants:

   > Zörner, J., Dymond, J.R., Shepherd, J.D., Wiser, S.K., Bunting, P., & Jolly, B. (2018). PyCrown – Fast raster-based individual tree segmentation for LiDAR data. Landcare Research NZ Ltd. https://doi.org/10.7931/M0SR-DN55

   Original repository: https://github.com/manaakiwhenua/pycrown

**PyCrown Simplified** (this package) is a fork that:
- Removes legacy dependencies (GDAL, laspy, pyximport, Cython)
- Replaces Cython with pure Numba `@njit` for portability
- Adds a novel **Hierarchical Region Growing (HRG v2)** algorithm
- Provides simplified I/O utilities (GeoPackage / Shapefile export via fiona)

## Features

- **CHM smoothing:** median, mean, Gaussian, maximum filters
- **Tree-top detection:** local maxima + center-of-mass correction + KDTree-based grouping
- **Dalponte crown delineation** in two modes:
  - `standard` — raster-based 4-connected region growing
  - `circ` — circular variant for smoother crown boundaries
- **Hierarchical Region Growing (HRG v2):**
  - Watershed segmentation from seed markers
  - Weighted Region Adjacency Graph (no external GIS libs)
  - Welford online statistics for O(1) merge decisions
  - Priority queue + variance annealing
  - Optional parallel processing
- **Flexible I/O:** export crowns and tree tops to Shapefile, GeoPackage, or GeoJSON

## Installation

**Recommended (conda + pip):**

```bash
# 1. Install native dependencies via conda
conda install -c conda-forge numpy numba scipy scikit-image rasterio fiona

# 2. Install pycrown-simplified
pip install --no-deps .          # from cloned repo
# or
pip install --no-deps git+https://github.com/igorpawelec/pycrown_simplified.git
```

> **Note:** Dependencies are installed via conda to avoid conflicts with native libraries. The `--no-deps` flag prevents pip from overwriting conda packages.

## Quick start

```python
from pycrown import PyCrown
from pycrown.io_utils import save_segments, save_tree_tops
import rasterio

# 1. Load CHM
pc = PyCrown("path/to/chm.tif")

# 2. Smooth
pc.smooth_chm(ws=5, method="median")

# 3. Detect and correct tree tops
pc.tree_detection(hmin=2.0, ws=5)
pc.correct_tree_tops(distance_threshold=5.0)
pc.screen_small_trees(hmin=3.0)

# 4. Delineate crowns
crowns = pc.crown_delineation(
    mode="circ",            # or "standard"
    th_seed=0.7,
    th_crown=0.55,
    th_tree=2.0,
    max_crown=10.0
)

# 5. Save results
with rasterio.open("path/to/chm.tif") as src:
    transform = src.transform
    crs_wkt = src.crs.to_wkt()
    chm_array = src.read(1)

save_segments(crowns, "results", "crowns", transform, crs_wkt, chm_array)
save_tree_tops(pc.tree_tops, "results", "chm", transform, crs_wkt, chm_array)
```

### Hierarchical Region Growing (HRG v2)

```python
crowns_hrg = pc.hierarchical_crown_delineation(
    variance_thresh=2.0,
    mask_thresh=9.0,
    morpho_radius=2,
    alpha=1.0,
    beta=0.5,
    gamma=0.1,
    anneal_lambda=0.9,
    max_iters=200,
    n_jobs=1              # -1 for all CPU cores
)
```

### Array-based (no rasterio needed)

```python
import numpy as np
chm = np.random.uniform(0, 30, (500, 500)).astype(np.float32)
pc = PyCrown(chm_array=chm)
```

### Windowed read (large rasters)

```python
# Load only a 2000×2000 px region starting at col=1000, row=500
pc = PyCrown("large_chm.tif", window=(1000, 500, 2000, 2000))
```

## API

| Method | Description |
|---|---|
| `PyCrown(chm_file)` or `PyCrown(chm_array=...)` | Initialize from file or array |
| `PyCrown(..., window=(col, row, w, h))` | Windowed read for large rasters |
| `smooth_chm(ws, method)` | Smooth CHM |
| `tree_detection(hmin, ws)` | Detect tree tops |
| `correct_tree_tops(distance_threshold)` | Merge nearby tops (KDTree) |
| `screen_small_trees(hmin)` | Remove short trees |
| `crown_delineation(mode, ...)` | Dalponte standard/circ |
| `hierarchical_crown_delineation(...)` | HRG v2 |
| `save_segments(...)` | Export crowns to SHP/GPKG |
| `save_tree_tops(...)` | Export tree tops to SHP/GPKG |
| `save_crowns_raster(...)` | Export crowns as GeoTIFF |

See the docstrings for full parameter details.

## Testing

```bash
pip install pytest
pytest tests/ -v
```

## Repository structure

```
pycrown_simplified/
├── pycrown/                              # Package source
│   ├── __init__.py                       # Public API (lazy imports)
│   ├── pycrown.py                        # Main PyCrown class
│   ├── _crown_dalponte_numba.py          # Standard Dalponte (Numba)
│   ├── _crown_dalponteCIRC_numba.py      # Circular Dalponte (Numba)
│   ├── _crown_hierarchical_region_growing.py  # HRG v2
│   └── io_utils.py                       # I/O: vector (fiona) + raster (rasterio)
├── tests/                                # Pytest suite
├── test_data/                            # Sample CHM rasters
├── www/                                  # Logo & assets
├── pycrown_test.py                       # Example usage script
├── pyproject.toml
├── environment.yaml
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
└── LICENSE
```

## Requirements

- Python ≥ 3.12
- NumPy ≥ 1.23
- SciPy ≥ 1.9
- scikit-image ≥ 0.20
- Numba ≥ 0.60
- Rasterio ≥ 1.3
- Fiona ≥ 1.9

## Citation

If you use this software in your research, please cite:

1. **This implementation:**

   > Pawelec, I. (2025). PyCrown Simplified: hybrid watershed + graph hierarchical segmentation [Software]. https://github.com/igorpawelec/pycrown_simplified

2. **The original PyCrown:**

   > Zörner, J., Dymond, J.R., Shepherd, J.D., Wiser, S.K., Bunting, P., & Jolly, B. (2018). PyCrown – Fast raster-based individual tree segmentation for LiDAR data. Landcare Research NZ Ltd. https://doi.org/10.7931/M0SR-DN55

3. **The crown delineation algorithm:**

   > Dalponte, M. & Coomes, D.A. (2016). Tree-centric mapping of forest carbon density from airborne laser scanning and hyperspectral data. *Methods in Ecology and Evolution*, 7, 1236–1245. https://doi.org/10.1111/2041-210X.12575

See also [CITATION.cff](CITATION.cff).

## License

GNU General Public License v3.0 — see [LICENSE.md](LICENSE.md).

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).
