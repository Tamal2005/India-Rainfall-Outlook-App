"""Generate a tiny, synthetic set of artifacts with the SAME file layout and
schema as the real notebook output. Used by the test-suite and for trying the
web app before your real models are copied in.

    python scripts/make_mock_artifacts.py --out ../data_mock

The predictions are meaningless; only the plumbing is real.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config  # noqa: E402
from app.knnr import ROTATION_24  # noqa: E402

BASE_FEATURES = ["latitude", "longitude", "elevation", "doy_sin", "doy_cos", "is_monsoon"]
FEATURES = BASE_FEATURES + config.REGIME_FEATURE_COLS      # == notebook FEATURE_COLS (18)
RAIN_CLASSES = config.RAIN_CLASSES
CLIMATE_TARGETS = config.CLIMATE_TARGETS_REQUIRED

STATES = {
    "KERALA": ["Kasaragod", "Kannur", "Wayanad", "Kozhikode", "Thrissur", "Idukki", "Alappuzha"],
    "ASSAM": ["Kamrup", "Dibrugarh", "Cachar", "Jorhat", "Nagaon"],
    "RAJASTHAN": ["Jaipur", "Jaisalmer", "Bikaner", "Udaipur", "Barmer"],
    "TRIPURA": ["West Tripura", "North Tripura", "Gomati", "Dhalai"],
    "ANDAMAN AND NICOBAR ISLANDS": ["Nicobars", "South Andaman"],
}


def make_lookup(rng: np.random.Generator) -> pd.DataFrame:
    rows, code = [], 100.0
    for state, districts in tqdm(STATES.items(), desc="Districts"):
        for d in districts:
            code += 1
            lat = float(rng.uniform(8, 32)); lon = float(rng.uniform(68, 94))
            dist = float(rng.uniform(1, 900))
            mean = float(rng.uniform(0.5, 9))
            rows.append({
                "state_name": state, "district_name": d, "district_code": code,
                "latitude": lat, "longitude": lon, "elevation": float(rng.uniform(0, 1500)),
                "distance_to_coast_km": dist, "log_distance_to_coast_km": float(np.log1p(dist)),
                "rainfall_mean": mean, "rainfall_std": mean * 2.5, "rainfall_p95": mean * 6,
                "rainfall_p99": mean * 12, "rain_rate_ge20": 0.05, "rain_rate_ge50": 0.01,
                "rain_rate_ge100": 0.002, "rain_rate_ge150": 0.0005,
                "rainfall_extreme_ratio": 0.1, "tail_risk_score": float(rng.uniform(0, 1)),
            })
    return pd.DataFrame(rows)


def make_training_frame(lookup: pd.DataFrame, rng: np.random.Generator, n: int = 3000) -> pd.DataFrame:
    base = lookup.sample(n, replace=True, random_state=1).reset_index(drop=True)
    doy = rng.integers(1, 366, n)
    df = base[["latitude", "longitude", "elevation"] + config.REGIME_FEATURE_COLS].copy()
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    month = ((doy - 1) // 30.5 + 1).astype(int)
    df["is_monsoon"] = np.isin(month, [6, 7, 8, 9]).astype(int)
    return df[FEATURES]


def _ln(rng, w):  # LayerNorm params near identity
    return {"weight": 1 + 0.05 * rng.normal(size=w), "bias": 0.05 * rng.normal(size=w)}


def _lin(rng, n_out, n_in):
    return {"weight": rng.normal(size=(n_out, n_in)) / np.sqrt(n_in), "bias": 0.05 * rng.normal(size=n_out)}


def random_resnet(rng, n_in, n_out, width=32, blocks=2, hidden=16) -> dict:
    """Random weights with the EXACT state_dict keys of the notebook's networks."""
    sd = {}

    def put(prefix, d):
        for k, v in d.items():
            sd[f"{prefix}.{k}"] = v.astype(np.float32)

    put("encoder.input.0", _lin(rng, width, n_in))
    put("encoder.input.1", _ln(rng, width))
    for i in range(blocks):
        put(f"encoder.blocks.{i}.norm", _ln(rng, width))
        put(f"encoder.blocks.{i}.fc1", _lin(rng, width * 2, width))
        put(f"encoder.blocks.{i}.fc2", _lin(rng, width, width * 2))
    put("encoder.out_norm", _ln(rng, width))
    put("head.0", _lin(rng, hidden, width))
    put("head.3", _lin(rng, n_out, hidden))
    return sd


def random_quantile(rng, n_in, n_targets, hidden=32) -> dict:
    sd = {}
    for name, d in [("net.0", _lin(rng, hidden, n_in)), ("net.1", _ln(rng, hidden)),
                    ("net.4", _lin(rng, hidden, hidden)), ("net.7", _lin(rng, hidden // 2, hidden)),
                    ("net.9", _lin(rng, n_targets * 3, hidden // 2))]:
        for k, v in d.items():
            sd[f"{name}.{k}"] = v.astype(np.float32)
    return sd


def make_models(lookup: pd.DataFrame, out: Path, rng: np.random.Generator) -> None:
    """Writes the same files scripts/export_weights.py produces from the notebook."""
    out.mkdir(parents=True, exist_ok=True)
    n_f = len(FEATURES)

    def scaler_json(extra: dict) -> dict:
        return {"feature_cols": FEATURES, "feature_mean": [0.0] * n_f, "feature_scale": [1.0] * n_f, **extra}

    np.savez_compressed(out / "rain_gate.npz", **random_resnet(rng, n_f, 1))
    np.savez_compressed(out / "rain_severity.npz", **random_resnet(rng, n_f, len(RAIN_CLASSES)))
    (out / "rain_gate_scaler.json").write_text(json.dumps(scaler_json({"threshold": 0.5})))

    base_value = {"temperature": 26, "temperature_max": 32, "temperature_min": 20,
                  "humidity": 70, "wind_speed": 2.5, "surface_pressure": 980}
    t_mean = [float(base_value[t]) for t in CLIMATE_TARGETS]
    t_scale = [2.0] * len(CLIMATE_TARGETS)
    np.savez_compressed(out / "climate_multitask.npz", **random_resnet(rng, n_f, len(CLIMATE_TARGETS)))
    (out / "climate_multitask_scalers.json").write_text(json.dumps(scaler_json({
        "target_cols": CLIMATE_TARGETS, "target_mean": t_mean, "target_scale": t_scale})))
    np.savez_compressed(out / "quantile_mlp.npz", **random_quantile(rng, n_f, len(CLIMATE_TARGETS)))
    (out / "quantile_mlp_scalers.json").write_text(json.dumps(scaler_json({
        "target_cols": CLIMATE_TARGETS, "quantiles": [0.1, 0.5, 0.9],
        "target_mean": t_mean, "target_scale": t_scale})))

    # monotone isotonic-style curves (thresholds, like sklearn's X_/y_thresholds_)
    cal = {"gate_x": np.array([0.0, 0.5, 1.0]), "gate_y": np.array([0.02, 0.45, 0.95])}
    for i in range(len(RAIN_CLASSES)):
        cal[f"sev_x_{i}"] = np.array([0.0, 0.3, 1.0])
        cal[f"sev_y_{i}"] = np.array([0.01, 0.25, 0.9])
    np.savez_compressed(out / "calibrators.npz", **cal)

    stats = {"no_rain": (0.09, 0.0), "light": (4.1, 3.0), "medium": (14.1, 13.0),
             "moderate": (29.3, 27.0), "heavy": (65.0, 60.0), "extreme": (127.5, 118.0)}
    (out / "precip_bin_stats.json").write_text(json.dumps({
        k: {"mean": m, "median": med, "p25": med * 0.7, "p75": med * 1.4} for k, (m, med) in stats.items()
    }, indent=2))


def make_knnr(lookup: pd.DataFrame, out: Path, rng: np.random.Generator) -> None:
    out.mkdir(parents=True, exist_ok=True)
    dates = pd.date_range("2015-01-01", "2023-12-31", freq="D")
    for window in tqdm((3, 5, 7), desc="KNNR windows"):
        rows = []
        for code in tqdm(lookup["district_code"], desc=f"{window}-day fragments", leave=False):
            for d in rng.choice(dates, 220, replace=False):
                total = float(rng.gamma(2.0, 6.0)) + 0.5
                shape = rng.dirichlet(np.full(24, 0.3))
                rotated = ROTATION_24 @ (shape * total)     # fragment stored in rotated space
                d = pd.Timestamp(d)
                row = {"district_code": code, "date": d.strftime("%Y-%m-%d"),
                       "day_of_year": d.dayofyear, "pattern": f"p{rng.integers(1, 4)}"}
                row.update({f"h{h}": float(rotated[h]) for h in range(24)})
                rows.append(row)
        pd.DataFrame(rows).to_parquet(out / f"knnr_{window}day.parquet", index=False)


def generate(root: Path, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    lookup = make_lookup(rng)
    (root / "prepped").mkdir(parents=True, exist_ok=True)
    lookup.to_parquet(root / "prepped" / "district_lookup.parquet", index=False)
    make_models(lookup, root / "models", rng)
    make_knnr(lookup, root / "disaggregation_knnr", rng)
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    print("Wrote mock artifacts to", generate(args.out.resolve()))
