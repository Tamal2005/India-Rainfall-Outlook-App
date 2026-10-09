"""Location resolution, feature construction and the rain-gate / severity cascade
and climate prediction using the PyTorch models (NumPy inference)."""
from __future__ import annotations

import difflib
from datetime import datetime

import numpy as np
import pandas as pd

from . import config
from .assets import Assets
from .nn import sigmoid, softmax


class LocationNotFound(ValueError):
    pass


def resolve_location(a: Assets, state_name: str, district_name: str) -> dict:
    lut = a.lookup
    s, d = state_name.strip().upper(), district_name.strip().upper()
    match = lut[(lut["state_u"] == s) & (lut["district_u"] == d)]
    if len(match) == 0:
        candidates = lut.loc[lut["state_u"] == s, "district_u"].unique().tolist()
        suggestions = difflib.get_close_matches(d, candidates, n=3, cutoff=0.5)
        raise LocationNotFound(
            f"No match for state='{state_name}', district='{district_name}'. "
            f"Closest district matches: {suggestions}"
        )
    row = match.iloc[0]
    loc = {
        "state_name": str(row["state_name"]),
        "district_name": str(row["district_name"]),
        "district_code": row["district_code"],
        "latitude": float(row["latitude"]),
        "longitude": float(row["longitude"]),
        "elevation": float(row["elevation"]),
        "is_unlisted_location": False,
    }
    for col in config.REGIME_FEATURE_COLS:
        loc[col] = float(row[col])
    return loc


def nearest_regime_location(a: Assets, latitude: float, longitude: float,
                            elevation: float, distance_to_coast_km: float) -> dict:
    """For a location missing from the lookup, borrow the 10 rainfall-statistic
    regime features from the closest known district (normalised space)."""
    cols = ["latitude", "longitude", "elevation", "distance_to_coast_km"]
    ref = a.lookup[cols].astype(float)
    means, stds = ref.mean(), ref.std().replace(0, 1.0)
    query = pd.Series({
        "latitude": latitude, "longitude": longitude,
        "elevation": elevation, "distance_to_coast_km": distance_to_coast_km,
    })
    dist = np.sqrt((((ref - means) / stds - (query - means) / stds) ** 2).sum(axis=1))
    idx = dist.idxmin()
    near = a.lookup.loc[idx]

    loc = {
        "state_name": "Custom location",
        "district_name": "Unlisted district (manual input)",
        "district_code": near["district_code"],
        "latitude": float(latitude),
        "longitude": float(longitude),
        "elevation": float(elevation),
        "distance_to_coast_km": float(distance_to_coast_km),
        "log_distance_to_coast_km": float(np.log1p(distance_to_coast_km)),
        "is_unlisted_location": True,
        "matched_from": {
            "district_name": str(near["district_name"]),
            "state_name": str(near["state_name"]),
            "similarity_distance": float(dist.loc[idx]),
        },
    }
    for col in config.REGIME_FEATURE_COLS:
        if col not in ("distance_to_coast_km", "log_distance_to_coast_km"):
            loc[col] = float(near[col])
    return loc


def is_monsoon(month: int) -> int:
    return int(month in (6, 7, 8, 9))


def build_feature_row(date_str: str, loc: dict) -> pd.DataFrame:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    doy = dt.timetuple().tm_yday
    row = {
        "latitude": loc["latitude"],
        "longitude": loc["longitude"],
        "elevation": loc["elevation"],
        "doy_sin": np.sin(2 * np.pi * doy / 365.25),
        "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "is_monsoon": is_monsoon(dt.month),
    }
    for col in config.REGIME_FEATURE_COLS:
        row[col] = loc[col]
    return pd.DataFrame([row])


def predict(a: Assets, date_str: str, loc: dict) -> dict:
    X = build_feature_row(date_str, loc)

    # Rain gate and severity share the gate's feature scaler (as in training).
    xg = a.gate_scaler.transform(X)
    raw_rain = float(sigmoid(a.gate_net(xg)[0, 0]))
    p_rain = float(np.clip(a.gate_calibrator(raw_rain), 0.0, 1.0))

    raw_sev = softmax(a.severity_net(xg)[0])
    cal = np.array([a.severity_calibrators[i](raw_sev[i]) for i in range(len(a.rain_classes))])
    cal = np.clip(cal, 1e-7, None)
    cal = cal / cal.sum()

    severity = {"no_rain": 1.0 - p_rain}
    for i, cls in enumerate(a.rain_classes):
        severity[cls] = p_rain * float(cal[i])

    stats = a.precip_bin_stats
    expected_mean = sum(severity[c] * stats[c]["mean"] for c in severity)
    expected_median = sum(severity[c] * stats[c]["median"] for c in severity)
    top = max(severity, key=severity.get)

    xc = a.climate_scaler.transform(X)
    values = a.climate_net(xc)[0] * a.climate_target_scale + a.climate_target_mean

    intervals: dict[str, tuple[float, float]] = {}
    if a.quantile is not None:
        q = a.quantile
        xq = q.scaler.transform(X)
        pred = q.net(xq)[0] * q.target_scale[:, None] + q.target_mean[:, None]   # (targets, 3)
        intervals = {t: (float(pred[i, 0]), float(pred[i, 2])) for i, t in enumerate(q.target_cols)}

    climate = []
    for i, target in enumerate(a.climate_target_cols):
        label, unit = config.CLIMATE_LABELS.get(target, (target.replace("_", " ").title(), ""))
        lo, hi = intervals.get(target, (None, None))
        climate.append({"key": target, "label": label, "unit": unit,
                        "value": float(values[i]), "p10": lo, "p90": hi})

    return {
        "date": date_str,
        "rain_probability": p_rain,
        "severity_probabilities": [
            {"key": c, "probability": severity.get(c, 0.0)}
            for c in config.SEVERITY_ORDER if c in severity or c == "no_rain"
        ],
        "expected_precipitation_mm": {
            "weighted_mean": expected_mean,
            "weighted_median": expected_median,
            "most_likely": {
                "bin": top,
                "probability": severity[top],
                "p25": stats[top]["p25"],
                "p75": stats[top]["p75"],
            },
        },
        "climate": climate,
    }
