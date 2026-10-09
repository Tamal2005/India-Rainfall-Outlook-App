"""Loads every model / lookup artifact exactly once at server start.

Models are the PyTorch networks from the training notebook, converted by
`scripts/export_weights.py` to .npz and run with plain NumPy (see app/nn.py)."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config
from .nn import IsotonicCurve, QuantileMLP, ResidualMLP, Scaler, _load_npz

log = logging.getLogger("rainfall.assets")


def progress(iterable, **kwargs):
    """tqdm wrapper that can be turned off via SHOW_PROGRESS=0."""
    kwargs.setdefault("disable", not config.SHOW_PROGRESS)
    return tqdm(iterable, **kwargs)


@dataclass
class QuantileBundle:
    scaler: Scaler
    net: QuantileMLP
    target_cols: list[str]
    target_mean: np.ndarray
    target_scale: np.ndarray


@dataclass
class Assets:
    lookup: pd.DataFrame                     # district lookup + upper-cased key columns
    rain_classes: list[str]
    precip_bin_stats: dict[str, dict[str, float]]
    # rain gate + severity (both use the gate's feature scaler, as in training)
    gate_scaler: Scaler
    gate_net: ResidualMLP
    gate_calibrator: IsotonicCurve
    severity_net: ResidualMLP
    severity_calibrators: list[IsotonicCurve]
    # shared multi-task climate regression
    climate_scaler: Scaler
    climate_net: ResidualMLP
    climate_target_cols: list[str]
    climate_target_mean: np.ndarray
    climate_target_scale: np.ndarray
    # optional p10/p50/p90 model
    quantile: Optional[QuantileBundle] = None


def required_files() -> list:
    d, m = config.DATA_DIR, config.MODEL_DIR
    return [
        d / "district_lookup.parquet",
        m / "rain_gate.npz", m / "rain_gate_scaler.json",
        m / "rain_severity.npz", m / "calibrators.npz",
        m / "climate_multitask.npz", m / "climate_multitask_scalers.json",
        m / "precip_bin_stats.json",
    ]


def _read_json(path):
    with open(path) as f:
        return json.load(f)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def load_assets() -> Assets:
    missing = [str(p) for p in required_files() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing model/data files (check DATA_ROOT / MODEL_DIR / DATA_DIR; "
            "run scripts/export_weights.py to convert the notebook's .pt files):\n  "
            + "\n  ".join(missing)
        )

    lookup = pd.read_parquet(config.DATA_DIR / "district_lookup.parquet")
    absent = [c for c in config.REGIME_FEATURE_COLS if c not in lookup.columns]
    if absent:
        raise ValueError(
            f"district_lookup.parquet is missing regime columns: {absent}. "
            "Run backend/scripts/rebuild_district_lookup.py."
        )
    lookup = lookup.copy()
    lookup["state_u"] = lookup["state_name"].astype(str).str.strip().str.upper()
    lookup["district_u"] = lookup["district_name"].astype(str).str.strip().str.upper()

    m = config.MODEL_DIR
    classes = list(config.RAIN_CLASSES)

    gate_scaler = Scaler.from_json(_read_json(m / "rain_gate_scaler.json"))
    gate_net = ResidualMLP(m / "rain_gate.npz")
    severity_net = ResidualMLP(m / "rain_severity.npz")
    _check(gate_net.n_in == len(gate_scaler.cols), "rain_gate.npz input size != scaler feature count")
    _check(gate_net.n_out == 1, "rain_gate.npz must have a single output logit")
    _check(severity_net.n_in == len(gate_scaler.cols), "rain_severity.npz input size != gate scaler feature count")
    _check(severity_net.n_out == len(classes), f"rain_severity.npz must have {len(classes)} outputs")

    cal = _load_npz(m / "calibrators.npz")
    gate_cal = IsotonicCurve(cal["gate_x"], cal["gate_y"])
    sev_cals = [IsotonicCurve(cal[f"sev_x_{i}"], cal[f"sev_y_{i}"]) for i in range(len(classes))]

    clim_info = _read_json(m / "climate_multitask_scalers.json")
    climate_scaler = Scaler.from_json(clim_info)
    climate_net = ResidualMLP(m / "climate_multitask.npz")
    target_cols = list(clim_info["target_cols"])
    _check(climate_net.n_in == len(climate_scaler.cols), "climate_multitask.npz input size != scaler feature count")
    _check(climate_net.n_out == len(target_cols), "climate_multitask.npz outputs != target_cols")

    quantile = None
    if (m / "quantile_mlp.npz").exists() and (m / "quantile_mlp_scalers.json").exists():
        qi = _read_json(m / "quantile_mlp_scalers.json")
        qnet = QuantileMLP(m / "quantile_mlp.npz")
        qcols = list(qi["target_cols"])
        _check(qnet.n_targets == len(qcols), "quantile_mlp.npz targets != quantile_mlp_scalers.json")
        quantile = QuantileBundle(
            scaler=Scaler.from_json(qi), net=qnet, target_cols=qcols,
            target_mean=np.asarray(qi["target_mean"], dtype=np.float64),
            target_scale=np.asarray(qi["target_scale"], dtype=np.float64),
        )
    else:
        log.warning("quantile_mlp.* not found: climate p10/p90 intervals will be omitted")

    log.info("Loaded %d districts, %d climate targets, quantile model: %s",
             len(lookup), len(target_cols), quantile is not None)
    return Assets(
        lookup=lookup,
        rain_classes=classes,
        precip_bin_stats=_read_json(m / "precip_bin_stats.json"),
        gate_scaler=gate_scaler, gate_net=gate_net, gate_calibrator=gate_cal,
        severity_net=severity_net, severity_calibrators=sev_cals,
        climate_scaler=climate_scaler, climate_net=climate_net,
        climate_target_cols=target_cols,
        climate_target_mean=np.asarray(clim_info["target_mean"], dtype=np.float64),
        climate_target_scale=np.asarray(clim_info["target_scale"], dtype=np.float64),
        quantile=quantile,
    )
