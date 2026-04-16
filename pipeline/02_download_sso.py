"""
02_download_sso.py
GAPC — Download Gaia DR3 sso_observation table via TAP.

Expected output: data/raw/sso_observations.parquet
Expected size:   ~3–5M rows, ~200 MB
Runtime:         10–30 min depending on TAP load

Column name notes (Gaia DR3 TAP, verified against Galluccio et al. 2022):
  - epoch_utc  (not 'epoch')
  - heliocentric_distance, geocentric_distance  (full names)
  - dec (not dec_) — but Gaia TAP returns it as 'dec' in results
  ORDER BY removed: server-side sort on 5M rows stresses the archive.
  Downstream code sorts by number_mp if needed.
"""

import time
from pathlib import Path

import pandas as pd
from astroquery.gaia import Gaia
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "sso_observations.parquet"

# ── TAP Query ─────────────────────────────────────────────────────────────────
# Minimal columns required for HG1G2 fitting.
# ORDER BY omitted — reduces server load; step 04 groupby doesn't need sort.
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
    geocentric_distance
FROM gaiadr3.sso_observation
WHERE g_mag               IS NOT NULL
  AND g_mag_error         IS NOT NULL
  AND phase_angle         IS NOT NULL
  AND heliocentric_distance IS NOT NULL
  AND geocentric_distance   IS NOT NULL
"""

# ── Quality threshold applied at download ─────────────────────────────────────
G_MAG_ERROR_MAX = 0.1    # drop obviously bad photometry early

# ── Retry config ──────────────────────────────────────────────────────────────
MAX_RETRIES    = 40      # persistent: up to ~7 h of retrying
RETRY_DELAY_S  = 600     # 10 min between retries


def astropy_to_df(table: Table) -> pd.DataFrame:
    """Convert astropy Table to pandas DataFrame, handling masked arrays."""
    df = table.to_pandas()
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def launch_with_retry() -> Table:
    """Launch async TAP job with retries on server errors."""
    Gaia.ROW_LIMIT = -1
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  Attempt {attempt}/{MAX_RETRIES} — launching async TAP job …")
            job = Gaia.launch_job_async(QUERY, verbose=False)
            print("  Job submitted, waiting for results …")
            results = job.get_results()
            return results
        except Exception as e:
            last_exc = e
            print(f"  ⚠️  Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"  Waiting {RETRY_DELAY_S}s before retry …")
                time.sleep(RETRY_DELAY_S)
    raise RuntimeError(
        f"All {MAX_RETRIES} TAP attempts failed. Last error: {last_exc}"
    )


def main():
    print("\n" + "=" * 55)
    print("  GAPC Step 2 — Download sso_observation")
    print("=" * 55)

    if OUT_PATH.exists():
        df_exist = pd.read_parquet(OUT_PATH, columns=["number_mp"])
        print(f"\n  Output already exists: {OUT_PATH}")
        print(f"  Rows: {len(df_exist):,} — delete to re-download.\n")
        return

    t0 = time.time()
    results = launch_with_retry()
    elapsed = time.time() - t0
    print(f"  TAP returned {len(results):,} rows in {elapsed/60:.1f} min")
    print(f"  Columns: {list(results.colnames)}")

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
