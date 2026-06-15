"""Tests for assign_depth_to_objects."""
from __future__ import annotations

import numpy as np

from layered_360 import SamObject, assign_depth_to_objects


def test_assigns_median_depth_under_mask() -> None:
    h, w = 10, 10
    depth = np.full((h, w), 0.1, dtype=np.float32)
    depth[3:6, 3:6] = 0.8
    mask = np.zeros((h, w), dtype=bool)
    mask[3:6, 3:6] = True
    obj = SamObject(mask=mask, area=int(mask.sum()), stability=0.9)
    assign_depth_to_objects([obj], depth, metric_depth=None)
    assert abs(obj.median_depth - 0.8) < 1e-6
    assert obj.median_metric is None


def test_assigns_median_metric_when_provided() -> None:
    h, w = 10, 10
    depth = np.zeros((h, w), dtype=np.float32)
    metric = np.full((h, w), 5.0, dtype=np.float32)
    metric[2:4, 2:4] = 2.5
    mask = np.zeros((h, w), dtype=bool)
    mask[2:4, 2:4] = True
    obj = SamObject(mask=mask, area=int(mask.sum()), stability=0.9)
    assign_depth_to_objects([obj], depth, metric_depth=metric)
    assert abs(obj.median_metric - 2.5) < 1e-6
