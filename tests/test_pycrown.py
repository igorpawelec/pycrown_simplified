"""
pytest suite for PyCrown Simplified.
Run: pytest tests/test_pycrown.py -v
"""

import numpy as np
import pytest


@pytest.fixture
def synthetic_chm():
    """50x50 CHM with some peaks."""
    np.random.seed(42)
    return np.random.uniform(0, 30, (50, 50)).astype(np.float32)


@pytest.fixture
def pc(synthetic_chm):
    from pycrown import PyCrown
    pc = PyCrown(chm_array=synthetic_chm)
    pc.smooth_chm(ws=3, method="median")
    return pc


class TestPyCrownInit:

    def test_from_array(self, synthetic_chm):
        from pycrown import PyCrown
        pc = PyCrown(chm_array=synthetic_chm)
        assert pc.chm.shape == (50, 50)
        assert pc.transform is None

    def test_from_array_with_transform(self, synthetic_chm):
        from pycrown import PyCrown
        from affine import Affine
        t = Affine(0.5, 0, 500000, 0, -0.5, 300000)
        pc = PyCrown(chm_array=synthetic_chm, transform=t)
        assert pc.transform == t

    def test_both_raises(self, synthetic_chm):
        from pycrown import PyCrown
        with pytest.raises(ValueError):
            PyCrown(chm_file="fake.tif", chm_array=synthetic_chm)

    def test_neither_raises(self):
        from pycrown import PyCrown
        with pytest.raises(ValueError):
            PyCrown()


class TestSmoothing:

    @pytest.mark.parametrize("method", ["median", "mean", "gaussian", "maximum"])
    def test_methods(self, synthetic_chm, method):
        from pycrown import PyCrown
        pc = PyCrown(chm_array=synthetic_chm)
        smoothed = pc.smooth_chm(ws=3, method=method)
        assert smoothed.shape == synthetic_chm.shape

    def test_invalid_method(self, synthetic_chm):
        from pycrown import PyCrown
        pc = PyCrown(chm_array=synthetic_chm)
        with pytest.raises(ValueError):
            pc.smooth_chm(ws=3, method="invalid")


class TestTreeDetection:

    def test_detect_trees(self, pc):
        tops = pc.tree_detection(hmin=10, ws=5)
        assert len(tops) > 0
        assert isinstance(tops, np.ndarray)
        assert tops.shape[1] == 2  # (row, col)

    def test_hmin_filters(self, pc):
        tops_low = pc.tree_detection(hmin=5, ws=5)
        # Reset
        pc.tree_tops = None
        tops_high = pc.tree_detection(hmin=25, ws=5)
        assert len(tops_low) >= len(tops_high)


class TestCorrectTreeTops:

    def test_correct_reduces(self, pc):
        pc.tree_detection(hmin=10, ws=3)
        n_before = len(pc.tree_tops)
        corrected = pc.correct_tree_tops(distance_threshold=5.0)
        assert len(corrected) <= n_before

    def test_correct_uses_kdtree(self, pc):
        """Verify KDTree implementation works."""
        pc.tree_detection(hmin=10, ws=3)
        corrected = pc.correct_tree_tops(distance_threshold=5.0)
        assert isinstance(corrected, np.ndarray)
        assert corrected.shape[1] == 2


class TestScreenSmallTrees:

    def test_screen(self, pc):
        pc.tree_detection(hmin=5, ws=3)
        pc.correct_tree_tops(distance_threshold=5.0)
        n_before = len(pc.tree_tops)
        pc.screen_small_trees(hmin=15)
        assert len(pc.tree_tops) <= n_before


class TestCrownDelineation:

    @pytest.mark.parametrize("mode", ["standard", "circ"])
    def test_dalponte_modes(self, pc, mode):
        pc.tree_detection(hmin=10, ws=5)
        pc.correct_tree_tops(distance_threshold=5.0)
        crowns = pc.crown_delineation(
            mode=mode, th_seed=0.45, th_crown=0.55,
            th_tree=2.0, max_crown=15.0
        )
        assert crowns.shape == pc.chm.shape
        assert crowns.dtype == np.int32
        n_crowns = len(np.unique(crowns)) - 1
        assert n_crowns > 0


class TestImports:

    def test_import(self):
        import pycrown
        assert hasattr(pycrown, '__version__')

    def test_import_io(self):
        from pycrown.io_utils import save_segments, save_tree_tops, save_crowns_raster
        assert callable(save_segments)
        assert callable(save_tree_tops)
        assert callable(save_crowns_raster)

    def test_quiet_mode(self):
        chm = np.random.uniform(0, 30, (50, 50)).astype(np.float32)
        from affine import Affine
        pc = PyCrown(chm_array=chm, transform=Affine.identity(), quiet=True)
        pc.smooth_chm(ws=3)
        pc.tree_detection(hmin=5, ws=3)
        # Should run without printing anything
        assert pc.tree_tops is not None
