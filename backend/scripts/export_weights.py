"""Convert the notebook's PyTorch artifacts into the files the backend loads.

Run this ONCE, on the machine that has torch + scikit-learn + joblib (your
training PC / Colab). The server itself never needs those packages.

    python scripts/export_weights.py \
        --src "D:/coding playground/python/College Project/models_pytorch" \
        --dst ../data/models

Reads  (from --src):   rain_gate.pt, rain_severity.pt, climate_multitask.pt,
                       quantile_mlp.pt (optional), rain_gate_calibrator.pkl,
                       severity_calibrators.pkl, rain_gate_scaler.json,
                       climate_multitask_scalers.json, quantile_mlp_scalers.json
                       (optional), precip_bin_stats.json
Writes (to --dst):     rain_gate.npz, rain_severity.npz, climate_multitask.npz,
                       quantile_mlp.npz, calibrators.npz + the JSON files copied over

Every converted model is checked against the original PyTorch / scikit-learn
object on random inputs; the script aborts if they disagree.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.nn import IsotonicCurve, QuantileMLP as NpQuantile, ResidualMLP as NpResidual  # noqa: E402

TOL = 1e-4


# ---- architectures: identical copies of the notebook definitions (parity check only) ----
class ResidualBlock(nn.Module):
    def __init__(self, width, dropout=0.12):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width * 2)
        self.fc2 = nn.Linear(width * 2, width)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        z = self.norm(x)
        z = self.act(self.fc1(z))
        z = self.dropout(z)
        z = self.dropout(self.fc2(z))
        return x + z


class WeatherEncoder(nn.Module):
    def __init__(self, n_features, width=256, blocks=4, dropout=0.12):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(n_features, width), nn.LayerNorm(width),
                                   nn.SiLU(), nn.Dropout(dropout))
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(blocks)])
        self.out_norm = nn.LayerNorm(width)

    def forward(self, x):
        return self.out_norm(self.blocks(self.input(x)))


class HeadNet(nn.Module):
    """RainGateNet (n_out=1), RainSeverityNet (n_out=5), ClimateMultiTaskNet (n_out=targets)."""
    def __init__(self, n_features, n_out, width, blocks, dropout, hidden):
        super().__init__()
        self.encoder = WeatherEncoder(n_features, width, blocks, dropout)
        self.head = nn.Sequential(nn.Linear(width, hidden), nn.SiLU(), nn.Dropout(0.10),
                                  nn.Linear(hidden, n_out))

    def forward(self, x):
        return self.head(self.encoder(x))


class QuantileMLP(nn.Module):
    def __init__(self, n_features, n_targets, hidden=128):
        super().__init__()
        self.n_targets = n_targets
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(0.10),
            nn.Linear(hidden, hidden // 2), nn.SiLU(),
            nn.Linear(hidden // 2, n_targets * 3))

    def forward(self, x):
        raw = self.net(x).view(-1, self.n_targets, 3)
        p10 = raw[:, :, 0]
        p50 = p10 + nn.functional.softplus(raw[:, :, 1])
        p90 = p50 + nn.functional.softplus(raw[:, :, 2])
        return torch.stack([p10, p50, p90], dim=-1)


def _cfg(ck, key, default):
    return ck.get(key, ck.get("config", {}).get(key, default))


def convert_resnet(src: Path, dst: Path, n_out_key: str | None, fixed_out: int | None) -> None:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    n_features = int(_cfg(ck, "feature_count", sd["encoder.input.0.weight"].shape[1]))
    n_out = fixed_out if fixed_out else int(_cfg(ck, n_out_key, sd["head.3.weight"].shape[0]))
    width, blocks, dropout = int(_cfg(ck, "width", 256)), int(_cfg(ck, "blocks", 4)), float(_cfg(ck, "dropout", 0.12))
    hidden = sd["head.0.weight"].shape[0]
    net = HeadNet(n_features, n_out, width, blocks, dropout, hidden)
    net.load_state_dict(sd)
    net.eval()
    np.savez_compressed(dst, **{k: v.cpu().numpy() for k, v in sd.items()})

    x = np.random.default_rng(0).normal(size=(256, n_features)).astype(np.float32)
    with torch.no_grad():
        ref = net(torch.from_numpy(x)).numpy()
    got = NpResidual(dst)(x.astype(np.float64))
    if ref.ndim == 1:
        ref = ref[:, None]
    diff = float(np.abs(ref - got).max())
    print(f"  {dst.name}: max|torch - numpy| = {diff:.2e}")
    if diff > TOL:
        raise SystemExit(f"Parity check FAILED for {dst.name}")


def convert_quantile(src: Path, dst: Path) -> None:
    ck = torch.load(src, map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"]
    n_features, n_targets = int(ck["feature_count"]), int(ck["target_count"])
    net = QuantileMLP(n_features, n_targets, int(ck.get("hidden", 128)))
    net.load_state_dict(sd)
    net.eval()
    np.savez_compressed(dst, **{k: v.cpu().numpy() for k, v in sd.items()})
    x = np.random.default_rng(1).normal(size=(256, n_features)).astype(np.float32)
    with torch.no_grad():
        ref = net(torch.from_numpy(x)).numpy()
    diff = float(np.abs(ref - NpQuantile(dst)(x.astype(np.float64))).max())
    print(f"  {dst.name}: max|torch - numpy| = {diff:.2e}")
    if diff > TOL:
        raise SystemExit(f"Parity check FAILED for {dst.name}")


def convert_calibrators(src: Path, dst: Path) -> None:
    gate = joblib.load(src / "rain_gate_calibrator.pkl")
    sev = joblib.load(src / "severity_calibrators.pkl")
    arrays = {"gate_x": gate.X_thresholds_, "gate_y": gate.y_thresholds_}
    for i, c in enumerate(sev):
        arrays[f"sev_x_{i}"], arrays[f"sev_y_{i}"] = c.X_thresholds_, c.y_thresholds_
    np.savez_compressed(dst, **arrays)

    grid = np.random.default_rng(2).uniform(-0.1, 1.1, 2000)
    pairs = [(gate, IsotonicCurve(arrays["gate_x"], arrays["gate_y"]))]
    pairs += [(c, IsotonicCurve(arrays[f"sev_x_{i}"], arrays[f"sev_y_{i}"])) for i, c in enumerate(sev)]
    diff = max(float(np.abs(c.predict(grid) - curve(grid)).max()) for c, curve in pairs)
    print(f"  {dst.name}: max|sklearn - numpy| = {diff:.2e}  ({1 + len(sev)} calibrators)")
    if diff > TOL:
        raise SystemExit("Parity check FAILED for calibrators")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="notebook MODEL_OUT folder (models_pytorch)")
    ap.add_argument("--dst", type=Path, required=True, help="backend MODEL_DIR (data/models)")
    args = ap.parse_args()
    src, dst = args.src, args.dst
    dst.mkdir(parents=True, exist_ok=True)

    print("Converting models...")
    convert_resnet(src / "rain_gate.pt", dst / "rain_gate.npz", None, 1)
    convert_resnet(src / "rain_severity.pt", dst / "rain_severity.npz", "n_classes", None)
    convert_resnet(src / "climate_multitask.pt", dst / "climate_multitask.npz", "target_count", None)
    if (src / "quantile_mlp.pt").exists():
        convert_quantile(src / "quantile_mlp.pt", dst / "quantile_mlp.npz")
    else:
        print("  quantile_mlp.pt not found: skipping (API will omit p10/p90)")
    convert_calibrators(src, dst / "calibrators.npz")

    for name in ["rain_gate_scaler.json", "climate_multitask_scalers.json",
                 "quantile_mlp_scalers.json", "precip_bin_stats.json"]:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)
            print(f"  copied {name}")
        elif name != "quantile_mlp_scalers.json":
            raise SystemExit(f"Required file missing in --src: {name}")
    print(f"\nDone. Backend MODEL_DIR = {dst.resolve()}")


if __name__ == "__main__":
    main()
