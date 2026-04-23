"""
02c_horizons_geometry.py
GAPC — Replace Keplerian geometry with JPL Horizons ephemerides for
GASP-matched asteroids in sso_filtered.parquet.

Queries JPL Horizons (geocenter, location='500') for precise r, delta,
phase_angle at each Gaia observation epoch.  Error vs ESA-exact geometry:
  ~0.009 mag (Gaia–geocenter offset ~0.01 AU for MBAs at 2.5 AU)
vs Keplerian error:
  ~1–2 mag systematic

Overwrites geometry columns for GASP-matched rows only; backs up originals.
Also recomputes v_reduced = g_mag - 5*log10(r*delta).

Output:  data/interim/sso_filtered.parquet  (updated in-place)
         data/interim/horizons_geometry_backup.parquet  (original Kepler values)
Runtime: ~2 hours  (18K requests, 3 parallel threads, 1 req/sec)
Resumable: yes — caches each asteroid's result under data/interim/_hz_cache/
"""

import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
FITS_PATH  = ROOT / "data" / "final"   / "gapc_catalog_v2.parquet"
OBS_PATH   = ROOT / "data" / "interim" / "sso_filtered.parquet"
BACKUP     = ROOT / "data" / "interim" / "horizons_geometry_backup.parquet"
CACHE_DIR  = ROOT / "data" / "interim" / "_hz_cache"

JD_J2010    = 2455197.5
N_WORKERS   = 3          # parallel Horizons threads
REQ_DELAY   = 3.0        # seconds per thread between requests → ~1 req/sec total
EPOCH_CHUNK = 50         # max JDs per Horizons request (avoid 414 URI-too-long)


def _horizons_chunk(number_mp: int, jds: list[float]) -> pd.DataFrame:
    """Single Horizons request for ≤ EPOCH_CHUNK JDs. Raises on failure."""
    from astroquery.jplhorizons import Horizons
    obj = Horizons(id=str(number_mp), location="500", epochs=jds)
    eph = obj.ephemerides()
    return pd.DataFrame({
        "epoch_jd": eph["datetime_jd"].data.data.astype(float),
        "r_hz":     eph["r"].data.data.astype(float),
        "delta_hz": eph["delta"].data.data.astype(float),
        "phase_hz": eph["alpha"].data.data.astype(float),
    })


_RETRY_DELAYS = [30, 90, 270]   # backoff seconds for 5xx errors


def fetch_one(number_mp: int, jds: list[float]) -> pd.DataFrame | None:
    """Query Horizons for one asteroid; batches large epoch lists, retries 5xx."""
    cache_file = CACHE_DIR / f"{number_mp}.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)

    time.sleep(REQ_DELAY)
    chunks = [jds[i:i + EPOCH_CHUNK] for i in range(0, len(jds), EPOCH_CHUNK)]

    for attempt, delay in enumerate([0] + _RETRY_DELAYS):
        if delay:
            log.info(f"  [{number_mp}] retry {attempt} in {delay}s …")
            time.sleep(delay)
        try:
            parts = []
            for chunk in chunks:
                parts.append(_horizons_chunk(number_mp, chunk))
                if len(chunks) > 1:
                    time.sleep(1.0)
            df = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
            df.to_parquet(cache_file, index=False)
            return df
        except Exception as e:
            msg = str(e)
            if any(c in msg for c in ("502", "503", "504", "Gateway")):
                log.warning(f"  [{number_mp}] server error (attempt {attempt+1}): {msg[:80]}")
            else:
                log.warning(f"  [{number_mp}] Horizons failed: {msg[:120]}")
                return None   # non-5xx errors won't improve with retries

    log.warning(f"  [{number_mp}] gave up after {len(_RETRY_DELAYS)+1} attempts")
    return None


def main():
    log.info("\n" + "=" * 55)
    log.info("  GAPC Step 2c — Horizons geometry for GASP-matched")
    log.info("=" * 55)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load target asteroid list ─────────────────────────────────────────
    fits = pd.read_parquet(FITS_PATH, columns=["number_mp", "gasp_match"])
    gasp_nums = set(fits.loc[fits["gasp_match"] == True, "number_mp"].astype(int))
    log.info(f"\n  GASP-matched asteroids: {len(gasp_nums):,}")

    # ── Load observations ────────────────────────────────────────────────
    log.info(f"  Loading {OBS_PATH.name} …")
    obs = pd.read_parquet(OBS_PATH)
    log.info(f"  {len(obs):,} observations  ·  {obs['number_mp'].nunique():,} asteroids")

    # ── Back up original geometry ────────────────────────────────────────
    if not BACKUP.exists():
        geo_cols = ["number_mp", "epoch_utc",
                    "heliocentric_distance", "geocentric_distance", "phase_angle",
                    "v_reduced"]
        obs[geo_cols].to_parquet(BACKUP, index=False)
        log.info(f"  Backed up original geometry → {BACKUP.name}")

    # ── Build per-asteroid JD lists ──────────────────────────────────────
    gasp_obs = obs[obs["number_mp"].isin(gasp_nums)].copy()
    gasp_obs["epoch_jd"] = gasp_obs["epoch_utc"] + JD_J2010

    asteroid_jds: dict[int, list[float]] = {}
    for num, grp in gasp_obs.groupby("number_mp"):
        asteroid_jds[int(num)] = grp["epoch_jd"].tolist()

    already_cached = sum(1 for n in asteroid_jds if (CACHE_DIR / f"{n}.parquet").exists())
    log.info(f"  {already_cached:,} / {len(asteroid_jds):,} already cached")

    # ── Parallel Horizons queries ────────────────────────────────────────
    log.info(f"\n  Querying JPL Horizons ({N_WORKERS} threads) …")
    results: dict[int, pd.DataFrame] = {}
    failed = []

    t0 = time.time()
    todo = [(n, jds) for n, jds in asteroid_jds.items()
            if not (CACHE_DIR / f"{n}.parquet").exists()]
    log.info(f"  Requests remaining: {len(todo):,}")

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(fetch_one, n, jds): n for n, jds in todo}
        done = 0
        for fut in as_completed(futures):
            num = futures[fut]
            df = fut.result()
            done += 1
            if df is None:
                failed.append(num)
            if done % 500 == 0:
                elapsed = time.time() - t0
                rate = done / elapsed
                remaining = (len(todo) - done) / rate if rate > 0 else 0
                log.info(f"  {done:,}/{len(todo):,}  "
                         f"({elapsed/60:.0f} min elapsed, "
                         f"~{remaining/60:.0f} min remaining)")

    # Load all cached results
    for num in asteroid_jds:
        cf = CACHE_DIR / f"{num}.parquet"
        if cf.exists():
            results[num] = pd.read_parquet(cf)

    log.info(f"\n  Horizons results: {len(results):,} success  ·  {len(failed):,} failed")

    # ── Merge Horizons geometry into obs ─────────────────────────────────
    log.info("  Merging Horizons geometry …")

    hz_frames = []
    for num, df in results.items():
        df = df.copy()
        df["number_mp"] = num
        hz_frames.append(df)

    if not hz_frames:
        log.error("  No Horizons results — aborting patch")
        return

    hz_all = pd.concat(hz_frames, ignore_index=True)
    hz_all["epoch_jd"] = hz_all["epoch_jd"].round(6)

    # Attach epoch_jd to obs for merging
    obs["epoch_jd"] = (obs["epoch_utc"] + JD_J2010).round(6)

    obs_merged = obs.merge(
        hz_all[["number_mp", "epoch_jd", "r_hz", "delta_hz", "phase_hz"]],
        on=["number_mp", "epoch_jd"],
        how="left",
    )

    # Patch: use Horizons values where available, keep Keplerian otherwise
    patched = obs_merged["r_hz"].notna()
    log.info(f"  Patching {patched.sum():,} / {len(obs):,} observations "
             f"({100*patched.mean():.1f}%)")

    obs_merged.loc[patched, "heliocentric_distance"] = obs_merged.loc[patched, "r_hz"]
    obs_merged.loc[patched, "geocentric_distance"]   = obs_merged.loc[patched, "delta_hz"]
    obs_merged.loc[patched, "phase_angle"]           = obs_merged.loc[patched, "phase_hz"]

    # Recompute v_reduced for patched rows
    obs_merged.loc[patched, "v_reduced"] = (
        obs_merged.loc[patched, "g_mag"]
        - 5 * np.log10(
            obs_merged.loc[patched, "heliocentric_distance"]
            * obs_merged.loc[patched, "geocentric_distance"]
        )
    )

    # Drop helper columns
    obs_merged.drop(columns=["epoch_jd", "r_hz", "delta_hz", "phase_hz"],
                    errors="ignore", inplace=True)

    obs_merged.to_parquet(OBS_PATH, index=False, compression="snappy")
    size_mb = OBS_PATH.stat().st_size / 1e6
    log.info(f"\n  ✅  Updated {OBS_PATH.name}  ({size_mb:.0f} MB)")
    log.info(f"  {patched.sum():,} rows have Horizons geometry; "
             f"{(~patched).sum():,} retain Keplerian\n")


if __name__ == "__main__":
    main()
