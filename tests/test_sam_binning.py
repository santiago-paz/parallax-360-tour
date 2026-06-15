"""Tests for bin_objects_by_depth (K-means + bg split)."""
from __future__ import annotations

import numpy as np

from layered_360 import SamObject, bin_objects_by_depth


def _obj(median: float, label: str = "") -> SamObject:
    mask = np.ones((1, 1), dtype=bool)
    o = SamObject(mask=mask, area=1, stability=0.9)
    o.median_depth = median
    return o


def test_bg_threshold_separates_far_objects() -> None:
    objs = [_obj(0.70), _obj(0.55), _obj(0.40), _obj(0.10)]
    fg_bins, bg_objs = bin_objects_by_depth(objs, k=3, bg_threshold=0.20)
    assert len(bg_objs) == 1
    assert bg_objs[0].median_depth == 0.10


def test_three_clusters_sorted_near_first() -> None:
    objs = [_obj(m) for m in [0.71, 0.72, 0.55, 0.54, 0.38, 0.39]]
    fg_bins, _ = bin_objects_by_depth(objs, k=3, bg_threshold=0.20)
    assert len(fg_bins) == 3
    medians = [np.median([o.median_depth for o in b]) for b in fg_bins]
    assert medians[0] > medians[1] > medians[2]
    assert abs(medians[0] - 0.715) < 0.01
    assert abs(medians[2] - 0.385) < 0.01


def test_k_reduced_when_fewer_objects_than_k() -> None:
    objs = [_obj(0.7), _obj(0.4)]
    fg_bins, _ = bin_objects_by_depth(objs, k=5, bg_threshold=0.20)
    assert len(fg_bins) == 2
    assert fg_bins[0][0].median_depth == 0.7


def test_empty_fg_returns_empty_bins() -> None:
    objs = [_obj(0.05), _obj(0.10)]
    fg_bins, bg_objs = bin_objects_by_depth(objs, k=3, bg_threshold=0.20)
    assert fg_bins == []
    assert len(bg_objs) == 2


def test_deterministic_runs() -> None:
    objs = [_obj(m) for m in [0.71, 0.55, 0.40, 0.72, 0.54, 0.39]]
    a, _ = bin_objects_by_depth(objs, k=3, bg_threshold=0.20)
    b, _ = bin_objects_by_depth(objs, k=3, bg_threshold=0.20)
    a_meds = [sorted(o.median_depth for o in bin_) for bin_ in a]
    b_meds = [sorted(o.median_depth for o in bin_) for bin_ in b]
    assert a_meds == b_meds
