"""
02b_csv_to_parquet.py
Convert a manually downloaded Gaia SSO CSV to the parquet format expected by step 03.
Usage: python pipeline/02b_csv_to_parquet.py <path_to_csv>
"""

import sys
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "sso_observations.parquet"

G_MAG_ERROR_MAX = 0.1

REQUIRED_COLS = [
    "source_id", "number_mp", "denomination", "epoch_utc",
    "g_mag", "g_mag_error", "phase_angle",
    "heliocentric_distance", "geocentric_distance",
]


def main():
    if len(sys.argv) < 2:
        print("Usage: python pipeline/02b_csv_to_parquet.py <path_to_csv>")
        sys.exit(1)

    csv_path = Path(sys.argv[1])
    print(f"\n  Reading {csv_path} …")
    df = pd.read_csv(csv_path, comment="#", low_memory=False)
    print(f"  Rows: {len(df):,}  Cols: {list(df.columns)}")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        print(f"\n  ❌ Missing columns: {missing}")
        print("  Check CSV column names against REQUIRED_COLS above.")
        sys.exit(1)

    df = df[REQUIRED_COLS]
    before = len(df)
    df = df.dropna(subset=["g_mag", "g_mag_error", "phase_angle",
                            "heliocentric_distance", "geocentric_distance"])
    print(f"  After null-drop: {len(df):,}  (dropped {before - len(df):,})")

    before = len(df)
    df = df[df["g_mag_error"] <= G_MAG_ERROR_MAX]
    print(f"  After g_mag_error ≤ {G_MAG_ERROR_MAX}: {len(df):,}  (dropped {before - len(df):,})")

    n_obj = df["number_mp"].nunique()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, compression="snappy")
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n  ✅  Saved: {OUT_PATH}  ({size_mb:.1f} MB)")
    print(f"  {len(df):,} observations  ·  {n_obj:,} unique asteroids\n")


if __name__ == "__main__":
    main()
