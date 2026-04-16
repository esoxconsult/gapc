"""
02_download_sso.py
GAPC — Download Gaia DR3 sso_observation table via TAP.

Expected output: data/raw/sso_observations.parquet
Expected size:   ~3–5M rows, ~200 MB
Runtime:         10–30 min depending on TAP load
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "sso_observations.parquet"

# ── TAP Query ─────────────────────────────────────────────────────────────────
QUERY = """
SELECT
    source_id,
    number_mp,
    denomination,
    epoch_utc,
    g_mag,
    g_mag_error,
    phase_angle,
    heliocentric_distance,
    geocentric_distance,
    ecl_lon,
    ecl_lat,
    ra,
    dec_,
    x_gaia,
    y_gaia,
    z_gaia
FROM gaiadr3.sso_observation
WHERE g_mag          IS NOT NULL
  AND g_mag_error    IS NOT NULL
  AND phase_angle    IS NOT NULL
  AND heliocentric_distance IS NOT NULL
  AND geocentric_distance   IS NOT NULL
ORDER BY number_mp, epoch_utc
"""

# ── Quality thresholds applied at download ────────────────────────────────────
G_MAG_ERROR_MAX = 0.1    # drop obviously bad photometry early


def astropy_to_df(table: Table) -> pd.DataFrame:
    """Convert astropy Table to pandas DataFrame, handling masked arrays."""
    df = table.to_pandas()
    # astropy sometimes returns object columns for masked floats
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def main():
    print("\n" + "=" * 55)
    print("  GAPC Step 2 — Download sso_observation")
    print("=" * 55)

    if OUT_PATH.exists():
        df_exist = pd.read_parquet(OUT_PATH, columns=["number_mp"])
        print(f"\n  Output already exists: {OUT_PATH}")
        print(f"  Rows: {len(df_exist):,} — delete to re-download.\n")
        return

    print(f"\n  Launching async TAP job (may take 10–30 min) …")
    t0 = time.time()

    Gaia.ROW_LIMIT = -1  # no row limit
    job = Gaia.launch_job_async(QUERY, verbose=True)

    print("  Waiting for job …")
    results = job.get_results()
    elapsed = time.time() - t0
    print(f"  TAP returned {len(results):,} rows in {elapsed/60:.1f} min")

    # ── Convert & light pre-filter ─────────────────────────────────────────
    df = astropy_to_df(results)
    print(f"  Rows after astropy→pandas: {len(df):,}")

    # Drop rows with critical nulls (safety net — TAP WHERE should handle this)
    before = len(df)
    df = df.dropna(subset=["g_mag", "g_mag_error", "phase_angle",
                            "heliocentric_distance", "geocentric_distance"])
    print(f"  Rows after null-drop: {len(df):,}  (dropped {before - len(df):,})")

    # Early g_mag_error filter
    before = len(df)
    df = df[df["g_mag_error"] <= G_MAG_ERROR_MAX]
    print(f"  Rows after g_mag_error ≤ {G_MAG_ERROR_MAX}: {len(df):,}  "
          f"(dropped {before - len(df):,})")

    n_obj = df["number_mp"].nunique()
    print(f"  Unique asteroids: {n_obj:,}")

    # ── Save ────────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, compression="snappy")
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n  ✅  Saved: {OUT_PATH}  ({size_mb:.1f} MB)")
    print(f"  {len(df):,} observations  ·  {n_obj:,} unique asteroids\n")


if __name__ == "__main__":
    main()
