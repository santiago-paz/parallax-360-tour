import numpy as np
import pytest

from parallax_360 import metric_to_disparity


def test_invierte_monotonicamente():
    d = np.array([[0.02, 0.5, 1.0]], np.float32)
    disp = metric_to_disparity(d)
    assert disp.dtype == np.float32
    assert disp[0, 0] > disp[0, 1] > disp[0, 2]


def test_clip_evita_division_explosiva():
    # 0.0 se clipea al min_clip → mismo valor que 0.01; approx porque
    # 0.01 no es exacto en float32 (1/0.01f ≈ 100.000002).
    d = np.array([[0.0, 0.01]], np.float32)
    disp = metric_to_disparity(d)
    assert np.isfinite(disp).all()
    assert disp[0, 0] == disp[0, 1]
    assert disp[0, 0] == pytest.approx(100.0)


def test_min_clip_personalizado():
    d = np.array([[0.0]], np.float32)
    assert metric_to_disparity(d, min_clip=0.1)[0, 0] == pytest.approx(10.0)
