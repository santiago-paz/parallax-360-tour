"""Smoke tests for the SAM availability detector and SamObject dataclass."""
from __future__ import annotations

import numpy as np

from layered_360 import SamObject, is_sam_available


def test_sam_object_defaults() -> None:
    mask = np.ones((4, 4), dtype=bool)
    obj = SamObject(mask=mask, area=int(mask.sum()), stability=0.93)
    assert obj.median_depth == 0.0
    assert obj.median_metric is None
    assert obj.area == 16


def test_is_sam_available_returns_bool() -> None:
    assert isinstance(is_sam_available(), bool)
