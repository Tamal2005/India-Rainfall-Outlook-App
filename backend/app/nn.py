"""Pure-NumPy inference for the PyTorch networks trained in the notebook.

Why NumPy: the server then needs neither torch nor scikit-learn (torch alone is
~700 MB installed and can blow a 512 MB PaaS instance). The architectures are
tiny residual MLPs, so a forward pass is a handful of matrix products.

Weights come from `scripts/export_weights.py`, which converts the notebook's
`.pt` checkpoints to `.npz` (same key names as the PyTorch state_dict) and the
scikit-learn isotonic calibrators to plain threshold arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LN_EPS = 1e-5  # torch.nn.LayerNorm default


def _linear(x, w, b):
    return x @ w.T + b


def _layer_norm(x, w, b):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)          # biased, like torch
    return (x - mu) / np.sqrt(var + LN_EPS) * w + b


def _silu(x):
    return x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _softplus(x):
    return np.logaddexp(0.0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)))


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: np.asarray(z[k], dtype=np.float64) for k in z.files}


@dataclass
class Scaler:
    """JSON-stored StandardScaler: (x - mean) / scale, columns picked by name."""
    cols: list[str]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def from_json(cls, info: dict, prefix: str = "feature") -> "Scaler":
        return cls(
            cols=list(info["feature_cols"]),
            mean=np.asarray(info[f"{prefix}_mean"], dtype=np.float64),
            scale=np.asarray(info[f"{prefix}_scale"], dtype=np.float64),
        )

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.cols].to_numpy(dtype=np.float64)
        return (x - self.mean) / self.scale


class ResidualMLP:
    """RainGateNet / RainSeverityNet / ClimateMultiTaskNet from the notebook:
    encoder(Linear-LN-SiLU, N x residual block, LN) -> head(Linear-SiLU-Linear).
    Dropout is a no-op at inference."""

    def __init__(self, path: Path):
        self.p = _load_npz(path)
        ids = {int(k.split(".")[2]) for k in self.p if k.startswith("encoder.blocks.")}
        self.n_blocks = (max(ids) + 1) if ids else 0

    @property
    def n_in(self) -> int:
        return self.p["encoder.input.0.weight"].shape[1]

    @property
    def n_out(self) -> int:
        return self.p["head.3.weight"].shape[0]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        h = _linear(x, p["encoder.input.0.weight"], p["encoder.input.0.bias"])
        h = _silu(_layer_norm(h, p["encoder.input.1.weight"], p["encoder.input.1.bias"]))
        for i in range(self.n_blocks):
            k = f"encoder.blocks.{i}."
            z = _layer_norm(h, p[k + "norm.weight"], p[k + "norm.bias"])
            z = _silu(_linear(z, p[k + "fc1.weight"], p[k + "fc1.bias"]))
            z = _linear(z, p[k + "fc2.weight"], p[k + "fc2.bias"])
            h = h + z
        h = _layer_norm(h, p["encoder.out_norm.weight"], p["encoder.out_norm.bias"])
        h = _silu(_linear(h, p["head.0.weight"], p["head.0.bias"]))
        return _linear(h, p["head.3.weight"], p["head.3.bias"])


class QuantileMLP:
    """QuantileMLP from Phase B: returns (N, n_targets, 3) = p10 <= p50 <= p90
    in the scaled target space."""

    def __init__(self, path: Path):
        self.p = _load_npz(path)

    @property
    def n_in(self) -> int:
        return self.p["net.0.weight"].shape[1]

    @property
    def n_targets(self) -> int:
        return self.p["net.9.weight"].shape[0] // 3

    def __call__(self, x: np.ndarray) -> np.ndarray:
        p = self.p
        h = _linear(x, p["net.0.weight"], p["net.0.bias"])
        h = _silu(_layer_norm(h, p["net.1.weight"], p["net.1.bias"]))
        h = _silu(_linear(h, p["net.4.weight"], p["net.4.bias"]))
        h = _silu(_linear(h, p["net.7.weight"], p["net.7.bias"]))
        raw = _linear(h, p["net.9.weight"], p["net.9.bias"]).reshape(len(x), -1, 3)
        p10 = raw[..., 0]
        p50 = p10 + _softplus(raw[..., 1])
        p90 = p50 + _softplus(raw[..., 2])
        return np.stack([p10, p50, p90], axis=-1)


@dataclass
class IsotonicCurve:
    """sklearn IsotonicRegression(out_of_bounds='clip') reduced to its threshold
    arrays; np.interp reproduces .predict() exactly (linear between thresholds,
    clamped at both ends)."""
    x: np.ndarray
    y: np.ndarray

    def __call__(self, v):
        return np.interp(v, self.x, self.y)
