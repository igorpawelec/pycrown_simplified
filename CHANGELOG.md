# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-04-19

### Added
- Initial release of PyCrown Simplified
- CHM smoothing (median, mean, Gaussian, maximum filters)
- Tree-top detection via local maxima + center-of-mass correction
- Dalponte crown delineation in standard and circular (CIRC) modes
- Novel hierarchical watershed + RAG + region-growing segmentation
- I/O utilities: raw raster export (.bin + .vrt), GeoPackage crowns & tree-tops
- Numba JIT acceleration for both Dalponte variants
