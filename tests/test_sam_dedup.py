"""Tests for dedup_and_filter (gate, area cap, NMS, containment)."""
from __future__ import annotations

import numpy as np

from layered_360 import SamObject, dedup_and_filter


def _rect_obj(h: int, w: int, y0: int, y1: int, x0: int, x1: int, stab: float = 0.9) -> SamObject:
    m = np.zeros((h, w), dtype=bool)
    m[y0:y1, x0:x1] = True
    return SamObject(mask=m, area=int(m.sum()), stability=stab)


def _full_gate(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=np.float32)


def test_drops_object_too_large() -> None:
    h, w = 100, 100
    huge = _rect_obj(h, w, 0, 80, 0, 80)
    kept = _rect_obj(h, w, 0, 10, 0, 10)
    out = dedup_and_filter([huge, kept], _full_gate(h, w))
    assert len(out) == 1
    assert np.array_equal(out[0].mask, kept.mask)


def test_nms_keeps_higher_stability_when_iou_above_threshold() -> None:
    h, w = 100, 100
    a = _rect_obj(h, w, 10, 30, 10, 30, stab=0.92)
    b = _rect_obj(h, w, 11, 31, 11, 31, stab=0.95)  # ~0.82 IoU vs a
    out = dedup_and_filter([a, b], _full_gate(h, w))
    assert len(out) == 1
    assert out[0].stability == 0.95


def test_containment_keeps_larger() -> None:
    h, w = 100, 100
    big = _rect_obj(h, w, 10, 50, 10, 50)
    small = _rect_obj(h, w, 20, 30, 20, 30)
    out = dedup_and_filter([big, small], _full_gate(h, w))
    assert len(out) == 1
    assert np.array_equal(out[0].mask, big.mask)


def test_drops_object_centroid_outside_gate_when_low_in_gate_coverage() -> None:
    h, w = 100, 100
    gate = np.zeros((h, w), dtype=np.float32)
    gate[40:60, :] = 1.0
    outside = _rect_obj(h, w, 0, 10, 0, 10)
    out = dedup_and_filter([outside], gate)
    assert out == []


def test_keeps_object_with_60pct_inside_gate() -> None:
    h, w = 100, 100
    gate = np.zeros((h, w), dtype=np.float32)
    gate[40:100, :] = 1.0
    obj = _rect_obj(h, w, 30, 60, 0, 10)
    out = dedup_and_filter([obj], gate)
    assert len(out) == 1
