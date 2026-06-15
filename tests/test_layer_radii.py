import numpy as np

from layered_360 import compute_layer_radii


def _alphas(*masks):
    return [m.astype(np.float32) for m in masks]


def test_radii_proporcionales_a_la_distancia():
    # slab 0 at 3 m, slab 1 at 6 m, bg at 7.5 m → r = 10·(3/7.5)=4.0 and 10·(6/7.5)=8.0
    metric = np.zeros((4, 4), np.float32)
    a0 = np.zeros((4, 4)); a0[0, :] = 1.0
    a1 = np.zeros((4, 4)); a1[1, :] = 1.0
    metric[0, :] = 3.0
    metric[1, :] = 6.0
    metric[2:, :] = 7.5
    union = ((a0 + a1) > 0).astype(np.uint8) * 255
    assert compute_layer_radii(metric, _alphas(a0, a1), union) == [4.0, 8.0]


def test_clamp_inferior_y_superior():
    metric = np.zeros((4, 4), np.float32)
    a0 = np.zeros((4, 4)); a0[0, :] = 1.0   # 0.1 m → r=0.1 → clamp 2.5
    a1 = np.zeros((4, 4)); a1[1, :] = 1.0   # 9.9 m → r=9.9 → clamp 9.0
    metric[0, :] = 0.1
    metric[1, :] = 9.9
    metric[2:, :] = 10.0
    union = ((a0 + a1) > 0).astype(np.uint8) * 255
    assert compute_layer_radii(metric, _alphas(a0, a1), union) == [2.5, 9.0]


def test_orden_ascendente_se_fuerza():
    # Medians inverted by noise: the nearest slab measures FARTHER than the next.
    metric = np.zeros((4, 4), np.float32)
    a0 = np.zeros((4, 4)); a0[0, :] = 1.0
    a1 = np.zeros((4, 4)); a1[1, :] = 1.0
    metric[0, :] = 6.0   # raw r 7.5
    metric[1, :] = 5.0   # raw r 6.25
    metric[2:, :] = 8.0
    union = ((a0 + a1) > 0).astype(np.uint8) * 255
    radii = compute_layer_radii(metric, _alphas(a0, a1), union)
    assert radii == [6.0, 6.2]  # r1=round(6.25)→6.2; r0=min(7.5, 6.2−0.2)=6.0


def test_slab_vacio_da_none():
    metric = np.full((4, 4), 5.0, np.float32)
    a0 = np.zeros((4, 4))                    # never exceeds 0.5 → None
    a1 = np.zeros((4, 4)); a1[1, :] = 1.0
    metric[1, :] = 2.5
    union = (a1 > 0).astype(np.uint8) * 255
    radii = compute_layer_radii(metric, _alphas(a0, a1), union)
    assert radii[0] is None
    assert radii[1] == 5.0


def test_bg_vacio_da_todos_none():
    metric = np.full((2, 2), 3.0, np.float32)
    a0 = np.ones((2, 2))
    union = np.full((2, 2), 255, np.uint8)   # union covers everything → no bg
    assert compute_layer_radii(metric, _alphas(a0), union) == [None]


def test_orden_se_fuerza_a_traves_de_slab_vacio():
    # An empty intermediate slab must not break the ordering guarantee:
    # layer 0 (6 m) inverted relative to layer 2 (4 m), layer 1 empty.
    metric = np.zeros((4, 4), np.float32)
    a0 = np.zeros((4, 4)); a0[0, :] = 1.0
    a1 = np.zeros((4, 4))
    a2 = np.zeros((4, 4)); a2[1, :] = 1.0
    metric[0, :] = 6.0
    metric[1, :] = 4.0
    metric[2:, :] = 8.0
    union = ((a0 + a2) > 0).astype(np.uint8) * 255
    radii = compute_layer_radii(metric, _alphas(a0, a1, a2), union)
    assert radii == [4.8, None, 5.0]


def test_cascada_en_tres_capas_clampadas():
    # Three slabs equally close: the clamp pulls them to r_min and the cascade
    # separates them downward (may end up slightly below r_min).
    metric = np.zeros((4, 4), np.float32)
    a0 = np.zeros((4, 4)); a0[0, :] = 1.0
    a1 = np.zeros((4, 4)); a1[1, :] = 1.0
    a2 = np.zeros((4, 4)); a2[2, :] = 1.0
    metric[:3, :] = 0.5
    metric[3, :] = 10.0
    union = ((a0 + a1 + a2) > 0).astype(np.uint8) * 255
    radii = compute_layer_radii(metric, _alphas(a0, a1, a2), union)
    assert radii == [2.1, 2.3, 2.5]


def test_nan_en_metric_depth_no_contamina():
    # Non-finite pixels (e.g. model-invalids) are excluded from the
    # medians instead of producing radius: nan in the snippet.
    metric = np.full((4, 4), 8.0, np.float32)
    a0 = np.zeros((4, 4)); a0[0, :] = 1.0
    metric[0, :2] = np.nan   # half of the slab is invalid
    metric[0, 2:] = 4.0      # valid half → median 4.0 → r = 5.0
    union = (a0 > 0).astype(np.uint8) * 255
    radii = compute_layer_radii(metric, [a0.astype(np.float32)], union)
    assert radii == [5.0]
