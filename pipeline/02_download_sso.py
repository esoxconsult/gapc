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
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from pathlib import Path

import pandas as pd
from astroquery.utils.tap.core import TapPlus
from astropy.table import Table

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "sso_observations.parquet"

# ── TAP endpoints (primary = ARI Heidelberg mirror, fallback = ESA) ───────────
TAP_ENDPOINTS = [
    ("ARI Heidelberg", "https://gaia.ari.uni-heidelberg.de/tap"),
    ("ESA Gaia",       "https://gea.esac.esa.int/tap-server/tap"),
]

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
    geocentric_distance
FROM gaiadr3.sso_observation
WHERE g_mag               IS NOT NULL
  AND g_mag_error         IS NOT NULL
  AND phase_angle         IS NOT NULL
  AND heliocentric_distance IS NOT NULL
  AND geocentric_distance   IS NOT NULL
"""

# ── Quality threshold applied at download ─────────────────────────────────────
G_MAG_ERROR_MAX = 0.1

# ── Retry config ──────────────────────────────────────────────────────────────
MAX_RETRIES    = 10      # per endpoint
RETRY_DELAY_S  = 300     # 5 min between retries
JOB_TIMEOUT_S  = 900     # 15 min hard timeout per TAP job attempt


def astropy_to_df(table: Table) -> pd.DataFrame:
    df = table.to_pandas()
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _run_tap_job(url: str) -> Table:
    tap = TapPlus(url=url, verbose=False)
    job = tap.launch_job_async(QUERY, verbose=False)
    return job.get_results()


def try_endpoint(name: str, url: str) -> Table:
    """Try one TAP endpoint with retries and a hard per-attempt timeout."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  [{name}] Attempt {attempt}/{MAX_RETRIES} — async TAP job …")
            # Must NOT use `with` executor — its __exit__ calls shutdown(wait=True)
            # which blocks until the thread finishes, defeating the timeout entirely.
            ex = ThreadPoolExecutor(max_workers=1)
            future = ex.submit(_run_tap_job, url)
            try:
                results = future.result(timeout=JOB_TIMEOUT_S)
            except FuturesTimeout:
                ex.shutdown(wait=False)
                raise TimeoutError(f"TAP job exceeded {JOB_TIMEOUT_S}s")
            ex.shutdown(wait=False)
            print(f"  [{name}] ✅ Got {len(results):,} rows")
            return results
        except TimeoutError as e:
            last_exc = e
            print(f"  [{name}] ⚠️  Attempt {attempt} timed out after {JOB_TIMEOUT_S}s")
        except Exception as e:
            last_exc = e
            print(f"  [{name}] ⚠️  Attempt {attempt} failed: {e}")
        if attempt < MAX_RETRIES:
            print(f"  [{name}] Waiting {RETRY_DELAY_S}s before retry …")
            time.sleep(RETRY_DELAY_S)
    raise RuntimeError(f"[{name}] All {MAX_RETRIES} attempts failed. Last: {last_exc}")


def launch_with_retry() -> Table:
    """Try each TAP endpoint in order; return first success."""
    for name, url in TAP_ENDPOINTS:
        try:
            return try_endpoint(name, url)
        except RuntimeError as e:
            print(f"\n  Switching endpoint: {e}\n")
    raise RuntimeError("All TAP endpoints exhausted.")


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
