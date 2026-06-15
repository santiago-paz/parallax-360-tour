"""Contract: with use_sam=True and a mocked SAM, return shape/dtype matches the threshold path."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np

from layered_360 import SamObject, make_layered_masks


def test_sam_path_returns_same_tuple_shape() -> None:
    h, w = 64, 128
    depth = np.zeros((h, w), dtype=np.float32)
    depth[20:40, 20:40] = 0.8
    depth[20:40, 70:90] = 0.5
    img_bgr = np.zeros((h, w, 3), dtype=np.uint8)

    near_mask = np.zeros((h, w), dtype=bool); near_mask[20:40, 20:40] = True
    mid_mask = np.zeros((h, w), dtype=bool); mid_mask[20:40, 70:90] = True

    def fake_segment(img, device="cpu"):
        return [
            SamObject(mask=near_mask, area=int(near_mask.sum()), stability=0.95),
            SamObject(mask=mid_mask, area=int(mid_mask.sum()), stability=0.95),
        ]

    with patch("layered_360.is_sam_available", return_value=True), \
         patch("layered_360.sam_segment_objects", side_effect=fake_segment):
        alphas, union_hard = make_layered_masks(
            depth, thresholds_desc=[0.65, 0.40], feather=0.06,
            exclude_top=0.0, exclude_bottom=0.0,
            img_bgr=img_bgr, use_sam=True, sam_k=2,
        )
    assert isinstance(alphas, list)
    assert all(a.dtype == np.float32 for a in alphas)
    assert union_hard.dtype == np.uint8
    assert union_hard.shape == (h, w)
    assert all(a.shape == (h, w) for a in alphas)
    assert alphas[0][30, 30] > 0.5
    assert alphas[0][30, 80] == 0
    assert alphas[1][30, 80] > 0.5


def test_sam_unavailable_falls_back_to_threshold_path() -> None:
    h, w = 32, 32
    depth = np.zeros((h, w), dtype=np.float32)
    depth[5:15, 5:15] = 0.9
    img_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    with patch("layered_360.is_sam_available", return_value=False):
        alphas, union_hard = make_layered_masks(
            depth, thresholds_desc=[0.5], feather=0.06,
            exclude_top=0.0, exclude_bottom=0.0,
            img_bgr=img_bgr, use_sam=True, sam_k=3,
        )
    assert len(alphas) == 1
    assert alphas[0][10, 10] > 0.5
    assert union_hard.shape == (h, w)
