"""One-time fix: rebuild district_lookup.parquet so it contains ALL regime
features, sourced from the exact regime CSV used during Phase 0 training.

    python scripts/rebuild_district_lookup.py --regime-path /path/to/spatial_regime_features.csv

Unlike the old script this no longer imports the app (which used to execute the
whole Streamlit UI as a side effect); it only reads app/config.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import config  # noqa: E402

REGIME_COLS = ["latitude", "longitude", *config.REGIME_FEATURE_COLS]

# lat/lon are rounded before joining so float-precision differences between the
# two files (21.15 vs 21.150000000000002) cannot cause silent non-matches.
COORD_ROUND_DP = 5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=config.DATA_DIR,
                    help=f"folder containing district_lookup.parquet (default: {config.DATA_DIR})")
    ap.add_argument("--regime-path", type=Path, required=True,
                    help="the EXACT regime CSV used in phase0_pipeline.py")
    args = ap.parse_args()

    lookup_path = args.data_dir.resolve() / "district_lookup.parquet"
    regime_path = args.regime_path.resolve()
    print(f"district_lookup: {lookup_path}\nregime CSV:      {regime_path}\n")
    for p in (lookup_path, regime_path):
        if not p.exists():
            sys.exit(f"ERROR: {p} does not exist.")

    lookup = pd.read_parquet(lookup_path)
    regime = pd.read_csv(regime_path, usecols=REGIME_COLS)
    print(f"{len(lookup)} districts, {len(regime)} regime locations")

    steps = tqdm(["round coordinates", "check duplicates", "merge", "validate", "write"], desc="Rebuild")
    for step in steps:
        if step == "round coordinates":
            for df in (lookup, regime):
                df["_lat_r"] = df["latitude"].round(COORD_ROUND_DP)
                df["_lon_r"] = df["longitude"].round(COORD_ROUND_DP)
        elif step == "check duplicates":
            dup = int(regime.duplicated(subset=["_lat_r", "_lon_r"]).sum())
            if dup:
                sys.exit(f"ERROR: {dup} duplicate (lat, lon) pairs in the regime CSV after rounding; "
                         "merging would fan out rows.")
        elif step == "merge":
            base = lookup.drop(columns=["distance_to_coast_km"], errors="ignore")
            slim = regime.drop(columns=["latitude", "longitude"])
            enriched = base.merge(slim, on=["_lat_r", "_lon_r"], how="left").drop(columns=["_lat_r", "_lon_r"])
        elif step == "validate":
            missing = enriched["distance_to_coast_km"].isna()
            if missing.any():
                print(enriched.loc[missing, ["state_name", "district_name", "latitude", "longitude"]].head(20))
                sys.exit(f"ERROR: {int(missing.sum())} districts have no regime match; refusing to write a "
                         "corrupt lookup. Check COORD_ROUND_DP or --regime-path.")
        elif step == "write":
            enriched.to_parquet(lookup_path, index=False)

    print(f"\nRebuilt {lookup_path} with {len(enriched.columns)} columns.")
    print("Restart the server so it re-loads the new file.")


if __name__ == "__main__":
    main()
