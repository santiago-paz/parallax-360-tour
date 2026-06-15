"""Tests for the horizontal seam handling in sam_segment_objects."""
from __future__ import annotations

import numpy as np

from layered_360 import merge_seam_masks


def _mask(h: int, w: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[:, x0:x1] = True
    return m


def test_two_halves_at_seam_merge_into_one() -> None:
    h, w = 8, 20
    left = _mask(h, w, 0, 4)
    right = _mask(h, w, 16, 20)
    merged = merge_seam_masks([left, right])
    assert len(merged) == 1
    out = merged[0]
    assert out[:, 0].all() and out[:, w - 1].all()


def test_non_seam_masks_pass_through() -> None:
    h, w = 8, 20
    a = _mask(h, w, 4, 8)
    b = _mask(h, w, 12, 16)
    merged = merge_seam_masks([a, b])
    assert len(merged) == 2


def test_single_seam_object_only_touches_one_edge_does_not_merge() -> None:
    h, w = 8, 20
    only_left = _mask(h, w, 0, 4)
    middle = _mask(h, w, 8, 12)
    merged = merge_seam_masks([only_left, middle])
    assert len(merged) == 2
