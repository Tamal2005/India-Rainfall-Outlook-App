"""KNNR (k-nearest-neighbour resampling) daily -> hourly rainfall disaggregation
and its leave-one-out validation. Ported from the Streamlit app, with the
Streamlit caches replaced by functools.lru_cache."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .assets import progress

WINDOW_FILES = {3: "knnr_3day.parquet", 5: "knnr_5day.parquet", 7: "knnr_7day.parquet"}
WINDOWS = (3, 5, 7)


def library_available() -> bool:
    d = config.KNNR_DIR
    return d.exists() and any(d.rglob("*.parquet"))


def available_windows() -> list[int]:
    return [w for w, f in WINDOW_FILES.items() if (config.KNNR_DIR / f).exists()]


# ----------------------------------------------------------------- helpers
def hour_columns(df: pd.DataFrame) -> list[str]:
    found = []
    for c in df.columns:
        name = str(c).lower()
        if name and name[0] in ("h", "y") and name[1:].isdigit() and 0 <= int(name[1:]) <= 23:
            found.append((int(name[1:]), c))
    found.sort(key=lambda x: x[0])
    return [c for _, c in found] if len(found) == 24 else []


def _date_column(df: pd.DataFrame):
    for name in ("date", "target_date"):
        if name in df.columns:
            return name
    return None


def _code_str(value) -> str:
    s = str(value)
    return s[:-2] if s.endswith(".0") else s


def circular_day_distance(a, b):
    diff = np.abs(np.asarray(a, dtype=float) - float(b))
    return np.minimum(diff, 365.25 - diff)


def _build_rotation(dimension: int) -> np.ndarray:
    """Orthogonal matrix whose last row is the (1,...,1)/sqrt(n) 'total rainfall' axis."""
    direction = np.ones(dimension) / np.sqrt(dimension)
    basis: list[np.ndarray] = []
    for i in range(dimension - 1):
        v = np.zeros(dimension)
        v[i] = 1.0
        for q in basis:
            v -= np.dot(v, q) * q
        v -= np.dot(v, direction) * direction
        norm = np.linalg.norm(v)
        if norm < 1e-12:
            for j in range(dimension):
                cand = np.zeros(dimension)
                cand[j] = 1.0
                for q in basis:
                    cand -= np.dot(cand, q) * q
                cand -= np.dot(cand, direction) * direction
                if np.linalg.norm(cand) > 1e-12:
                    v, norm = cand, np.linalg.norm(cand)
                    break
            else:
                raise RuntimeError("Could not construct orthogonal basis.")
        basis.append(v / norm)

    Q, _ = np.linalg.qr(np.vstack(basis).T)
    R = np.vstack([Q[:, : dimension - 1].T, direction])
    if not np.allclose(R @ R.T, np.eye(dimension), atol=1e-8):
        raise RuntimeError("Rotation matrix is not orthogonal.")
    return R


ROTATION_24 = _build_rotation(24)


def inverse_rotation(fragment, target_total: float) -> np.ndarray:
    y = np.nan_to_num(np.asarray(fragment, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if len(y) != 24:
        raise ValueError("Rotated hourly fragment must contain exactly 24 values.")
    y[-1] = target_total / np.sqrt(24.0)
    out = ROTATION_24.T @ y
    out[out < 0] = 0.0
    total = out.sum()
    if total <= 0:
        out[:] = target_total / 24.0
    else:
        out *= target_total / total
    return out


# ----------------------------------------------------------------- library
@lru_cache(maxsize=4)
def load_window(directory: str, window: int) -> pd.DataFrame | None:
    filename = WINDOW_FILES.get(window)
    path = Path(directory) / filename if filename else None
    if path is None or not path.exists():
        return None
    df = pd.read_parquet(path)
    dc = _date_column(df)
    if dc is not None:
        df["_date_parsed"] = pd.to_datetime(df[dc], errors="coerce")
    if "district_code" in df.columns:
        df["_district_code_str"] = df["district_code"].astype(str).str.replace(r"\.0$", "", regex=True)
    return df


def warm_up() -> None:
    """Pre-load the KNNR files at startup so the first request is not slow."""
    for w in progress(available_windows(), desc="Loading KNNR windows"):
        load_window(str(config.KNNR_DIR), w)


# ----------------------------------------------------------------- core
def disaggregate(df, district_code, target_date, daily_total_mm, n_draws=50, top_k=10, seed=42):
    """Returns (matrix[n_draws, 24], analog_dates, match_info) or None."""
    info = {"pattern_column": False, "seasonal_only": True, "target_in_library": False}
    if df is None:
        return None
    if daily_total_mm <= 0:
        return np.zeros((n_draws, 24)), [], info

    hcols = hour_columns(df)
    if len(hcols) != 24:
        return None

    if "_district_code_str" in df.columns:
        work = df[df["_district_code_str"] == _code_str(district_code)].copy()
    else:
        work = df.copy()
    if len(work) == 0:
        return None

    target_dt = pd.to_datetime(target_date)
    target_row = None
    if "_date_parsed" in work.columns:
        hit = work[work["_date_parsed"] == target_dt]
        if len(hit) > 0:
            target_row = hit.iloc[0]
            info["target_in_library"] = True

    if "day_of_year" in work.columns:
        work["_doy_distance"] = circular_day_distance(work["day_of_year"].to_numpy(), target_dt.dayofyear)
        seasonal = work[work["_doy_distance"] <= 45]
        if len(seasonal) >= top_k:
            work = seasonal.copy()
        work["_distance"] = work["_doy_distance"].astype(float)
    else:
        work["_distance"] = 0.0

    if "pattern" in work.columns and target_row is not None:
        pat = target_row.get("pattern")
        if pat is not None and pd.notna(pat):
            same = work[work["pattern"] == pat]
            if "_date_parsed" in same.columns:
                same = same[same["_date_parsed"] != target_dt]
            if len(same) >= min(top_k, 3):
                work = same.copy()
                info["pattern_column"] = True
                info["seasonal_only"] = False

    if "_date_parsed" in work.columns:
        work = work[work["_date_parsed"] != target_dt]
    if len(work) == 0:
        return None

    nearest = work.sort_values("_distance").head(min(top_k, len(work)))
    weights = 1.0 / np.arange(1, len(nearest) + 1, dtype=float)
    weights /= weights.sum()

    rng = np.random.RandomState(seed)
    fragments = nearest[hcols].to_numpy(dtype=float)
    dates = nearest["_date_parsed"].to_numpy() if "_date_parsed" in nearest.columns else None

    matrix = np.zeros((n_draws, 24))
    analogs = []
    for i in range(n_draws):
        idx = rng.choice(len(nearest), p=weights)
        frag = fragments[idx].copy()
        frag[~np.isfinite(frag)] = 0.0
        matrix[i] = inverse_rotation(frag, daily_total_mm)
        analogs.append(None if dates is None else str(pd.Timestamp(dates[idx]).date()))
    return matrix, analogs, info


def ensemble(district_code, target_date, daily_total_mm, n_draws=50, top_k=10):
    """Combine the 3/5/7-day libraries into one ensemble."""
    results = {}
    for w in WINDOWS:
        df = load_window(str(config.KNNR_DIR), w)
        if df is None:
            continue
        try:
            out = disaggregate(df, district_code, target_date, daily_total_mm,
                               n_draws=n_draws, top_k=top_k, seed=1000 + w)
        except Exception:
            continue
        if out is not None:
            results[w] = out
    if not results:
        return None

    matrices, analogs = [], []
    for w in WINDOWS:
        if w in results:
            matrices.append(results[w][0])
            analogs.extend(results[w][1])
    combined = np.maximum(np.vstack(matrices), 0.0)
    totals = combined.sum(axis=1, keepdims=True)
    safe = np.where(totals > 0, totals, 1.0)
    combined = np.where(totals > 0, combined * (daily_total_mm / safe), daily_total_mm / 24.0)

    infos = {w: results[w][2] for w in results}
    return {
        "matrix": combined,
        "analogs": analogs,
        "windows": list(results),
        "match_info": infos,
        "any_pattern_matching": any(not i["seasonal_only"] for i in infos.values()),
        "target_in_library": any(i["target_in_library"] for i in infos.values()),
    }


def summarise_ensemble(res: dict) -> dict:
    m = res["matrix"]
    mean = m.mean(axis=0)
    peak = int(np.argmax(mean))
    pct = lambda q: np.percentile(m, q, axis=0)
    hours = [
        {"hour": h, "mean": float(mean[h]), "p10": float(pct(10)[h]), "p25": float(pct(25)[h]),
         "p75": float(pct(75)[h]), "p90": float(pct(90)[h])}
        for h in range(24)
    ]
    return {
        "hours": hours,
        "daily_total": float(mean.sum()),
        "peak_hour": peak,
        "peak_value": float(mean[peak]),
        "simulations": int(len(m)),
        "windows": res["windows"],
        "any_pattern_matching": res["any_pattern_matching"],
        "target_in_library": res["target_in_library"],
        "match_info": [
            {"window": w, "matched_on": "pattern-label column" if i["pattern_column"]
             else "seasonal (day-of-year) only"}
            for w, i in sorted(res["match_info"].items())
        ],
        "analogue_dates": sorted({d for d in res["analogs"] if d}),
    }


# ----------------------------------------------------------------- validation
@lru_cache(maxsize=16)
def _candidate_pool(directory: str, window: int, district_code: str | None):
    df = load_window(directory, window)
    if df is None:
        return None
    hcols = hour_columns(df)
    if len(hcols) != 24 or "_date_parsed" not in df.columns:
        return None
    work = df
    if district_code is not None and "_district_code_str" in work.columns:
        work = work[work["_district_code_str"] == district_code]
    work = work.dropna(subset=["_date_parsed"])
    totals = work[hcols].sum(axis=1)
    work = work[totals > 0].copy()
    work["_daily_total"] = totals[totals > 0]
    return work.reset_index(drop=True)


def _score(actual: np.ndarray, pred: np.ndarray):
    mae = float(np.mean(np.abs(actual - pred)))
    peak_hit = abs(int(np.argmax(actual)) - int(np.argmax(pred))) <= 1
    ss_res = float(np.sum((actual - pred) ** 2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    nse = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    return mae, bool(peak_hit), nse


def run_validation(window: int, district_code, n_days: int, n_draws: int, top_k: int, seed: int = 42):
    code = None if district_code is None else _code_str(district_code)
    pool = _candidate_pool(str(config.KNNR_DIR), window, code)
    if pool is None or len(pool) == 0:
        return None
    hcols = hour_columns(pool)
    library = load_window(str(config.KNNR_DIR), window)
    sample = pool.sample(min(n_days, len(pool)), random_state=seed)

    rows, detail = [], []
    for _, r in progress(sample.iterrows(), total=len(sample), desc=f"Validating {window}-day"):
        date = r["_date_parsed"].strftime("%Y-%m-%d")
        total = float(r["_daily_total"])
        actual = r[hcols].to_numpy(dtype=float)
        out = disaggregate(library, r.get("district_code", district_code), date, total,
                           n_draws=n_draws, top_k=top_k, seed=seed)
        if out is None:
            continue
        matrix, _, info = out
        pred = matrix.mean(axis=0)
        naive = np.full(24, total / 24.0)
        mae, hit, nse = _score(actual, pred)
        n_mae, n_hit, n_nse = _score(actual, naive)
        rows.append({
            "date": date, "daily_total_mm": total,
            "mae": mae, "peak_hour_hit": hit, "nse": nse,
            "naive_mae": n_mae, "naive_peak_hour_hit": n_hit, "naive_nse": n_nse,
            "match_type": "seasonal-only" if info["seasonal_only"] else "pattern",
        })
        detail.append({
            "date": date, "daily_total_mm": total,
            "actual": actual.tolist(), "predicted": pred.tolist(),
            "p10": np.percentile(matrix, 10, axis=0).tolist(),
            "p90": np.percentile(matrix, 90, axis=0).tolist(),
        })
    if not rows:
        return None

    df = pd.DataFrame(rows)
    mean = lambda c: float(df[c].mean())
    med = lambda c: float(df[c].median())
    nanmean = lambda c: None if df[c].dropna().empty else float(df[c].dropna().mean())
    nanmed = lambda c: None if df[c].dropna().empty else float(df[c].dropna().median())
    summary = {
        "n_days": len(df),
        "knnr": {"mean_mae": mean("mae"), "median_mae": med("mae"),
                 "peak_hit_rate": mean("peak_hour_hit"),
                 "mean_nse": nanmean("nse"), "median_nse": nanmed("nse")},
        "naive": {"mean_mae": mean("naive_mae"), "median_mae": med("naive_mae"),
                  "peak_hit_rate": mean("naive_peak_hour_hit"),
                  "mean_nse": nanmean("naive_nse"), "median_nse": nanmed("naive_nse")},
        "beats_naive": bool(mean("mae") < mean("naive_mae")),
        "match_counts": df["match_type"].value_counts().to_dict(),
    }
    return {"window": window, "summary": summary, "rows": rows, "detail": detail}
