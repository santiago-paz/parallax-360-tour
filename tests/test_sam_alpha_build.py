"""Tests for build_alpha_masks."""
from __future__ import annotations

import numpy as np

from layered_360 import SamObject, build_alpha_masks


def _full_gate(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=np.float32)


def _square(h: int, w: int, y0: int, y1: int, x0: int, x1: int, median: float) -> SamObject:
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    o = SamObject(mask=m, area=int(m.sum()), stability=0.9)
    o.median_depth = median
    return o


def test_alpha_per_bin_unions_objects() -> None:
    h, w = 20, 20
    a = _square(h, w, 2, 6, 2, 6, median=0.7)
    b = _square(h, w, 10, 14, 10, 14, median=0.7)
    fg_bins = [[a, b]]
    alphas = build_alpha_masks(fg_bins, _full_gate(h, w), image_shape=(h, w), feather_px=0)
    assert len(alphas) == 1
    assert alphas[0].dtype == np.float32
    assert alphas[0][3, 3] > 0.9 and alphas[0][12, 12] > 0.9
    assert alphas[0][0, 0] == 0


def test_overlap_with_closer_bin_is_subtracted() -> None:
    h, w = 20, 20
    near = _square(h, w, 5, 10, 5, 10, median=0.8)
    far = _square(h, w, 7, 12, 7, 12, median=0.4)
    fg_bins = [[near], [far]]
    alphas = build_alpha_masks(fg_bins, _full_gate(h, w), image_shape=(h, w), feather_px=0)
    assert alphas[0][8, 8] > 0.9
    assert alphas[1][8, 8] == 0


def test_returns_float32_in_range() -> None:
    h, w = 10, 10
    obj = _square(h, w, 2, 6, 2, 6, median=0.7)
    alphas = build_alpha_masks([[obj]], _full_gate(h, w), image_shape=(h, w), feather_px=0)
    assert alphas[0].dtype == np.float32
    assert alphas[0].min() >= 0.0
    assert alphas[0].max() <= 1.0
