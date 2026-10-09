"""NumPy inference building blocks (no torch needed)."""
import numpy as np
import pytest

from app import nn


def test_softmax_and_sigmoid_are_stable():
    assert nn.softmax(np.array([1000.0, 0.0, -1000.0])).sum() == pytest.approx(1.0)
    assert nn.sigmoid(-1e6) == pytest.approx(0.0) and nn.sigmoid(1e6) == pytest.approx(1.0)


def test_layer_norm_matches_definition():
    x = np.random.default_rng(0).normal(size=(4, 16))
    out = nn._layer_norm(x, np.ones(16), np.zeros(16))
    assert np.allclose(out.mean(axis=-1), 0, atol=1e-9)
    assert np.allclose(out.var(axis=-1), 1, atol=1e-4)


def test_isotonic_curve_matches_sklearn_clip():
    iso = pytest.importorskip("sklearn.isotonic")
    rng = np.random.default_rng(1)
    x = rng.random(500)
    y = (rng.random(500) < x).astype(int)
    ref = iso.IsotonicRegression(out_of_bounds="clip").fit(x, y)
    curve = nn.IsotonicCurve(ref.X_thresholds_, ref.y_thresholds_)
    grid = np.linspace(-0.2, 1.2, 1000)           # includes out-of-range values
    assert np.allclose(ref.predict(grid), curve(grid), atol=1e-12)
