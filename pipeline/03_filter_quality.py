"""
03_filter_quality.py
GAPC — Quality filtering and geometric reduction.

Applies:
  - Per-observation photometric quality cuts
  - Reduced magnitude computation: V_r = g_mag - 5*log10(r_h * r_geo)
  - Per-object sample requirements (min obs, phase range)

Input:  data/raw/sso_observations.parquet
Output: data/interim/sso_filtered.parquet
"""

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN_PATH  = ROOT / "data" / "raw"       / "sso_observations.parquet"
OUT_PATH = ROOT / "data" / "interim"   / "sso_filtered.parquet"

# ── Filter thresholds ─────────────────────────────────────────────────────────
G_MAG_BRIGHT    = 12.0   # Gaia saturation limit
G_MAG_FAINT     = 21.0   # below this: unreliable
G_MAG_ERR_MAX   = 0.05   # tighter than download filter
PHASE_MIN       = 0.5    # deg — near-zero phase unreliable
PHASE_MAX       = 120.0  # deg — Gaia coverage ceiling
R_HELIO_MAX     = 5.5    # AU — beyond Jupiter belt
R_GEO_MAX       = 6.0    # AU

# Per-object requirements
MIN_OBS         = 5      # minimum observations per asteroid
MIN_PHASE_RANGE = 5.0    # deg — minimum phase angle coverage for meaningful fit


def reduce_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute reduced magnitude (standard geometry: r_h = r_geo = 1 AU).
    V_r = g_mag - 5 * log10(r_helio * r_geo)
    """
    df = df.copy()
    df["v_reduced"] = (
        df["g_mag"]
        - 5.0 * np.log10(df["heliocentric_distance"] * df["geocentric_distance"])
    )
    return df


def main():
    print("\n" + "=" * 55)
    print("  GAPC Step 3 — Quality Filtering")
    print("=" * 55)

    df = pd.read_parquet(IN_PATH)
    print(f"\n  Input: {len(df):,} observations  ·  "
          f"{df['number_mp'].nunique():,} asteroids")

    # ── Observation-level cuts ─────────────────────────────────────────────
    # Each mask is computed from the current (already-filtered) df to avoid
    # stale-index issues when reassigning df in a loop.
    cuts = [
        ("g_mag bright limit",  lambda d: d["g_mag"] >= G_MAG_BRIGHT),
        ("g_mag faint limit",   lambda d: d["g_mag"] <= G_MAG_FAINT),
        ("g_mag_error ≤ 0.05",  lambda d: d["g_mag_error"] <= G_MAG_ERR_MAX),
        ("phase_angle range",   lambda d: d["phase_angle"].between(PHASE_MIN, PHASE_MAX)),
        ("r_helio ≤ 5.5 AU",   lambda d: d["heliocentric_distance"] <= R_HELIO_MAX),
        ("r_geo ≤ 6.0 AU",     lambda d: d["geocentric_distance"] <= R_GEO_MAX),
    ]
    for label, mask_fn in cuts:
        before = len(df)
        df = df[mask_fn(df)]
        print(f"  After {label}: {len(df):,}  (dropped {before - len(df):,})")

    # ── Reduced magnitude ─────────────────────────────────────────────────
    df = reduce_magnitude(df)
    print(f"\n  v_reduced computed — "
          f"range: {df['v_reduced'].min():.2f} – {df['v_reduced'].max():.2f} mag")

    # ── Per-object requirements ────────────────────────────────────────────
    stats = df.groupby("number_mp").agg(
        n_obs=("g_mag", "count"),
        phase_range=("phase_angle", lambda x: x.max() - x.min()),
    )

    valid_ids = stats[
        (stats["n_obs"] >= MIN_OBS) &
        (stats["phase_range"] >= MIN_PHASE_RANGE)
    ].index

    before_obj = df["number_mp"].nunique()
    df = df[df["number_mp"].isin(valid_ids)]
    after_obj = df["number_mp"].nunique()

    print(f"\n  Per-object filter (≥{MIN_OBS} obs, phase range ≥{MIN_PHASE_RANGE}°):")
    print(f"  {before_obj:,} → {after_obj:,} asteroids  "
          f"(dropped {before_obj - after_obj:,})")
    print(f"  {len(df):,} observations remain")

    # ── Observation count distribution ────────────────────────────────────
    obs_counts = df.groupby("number_mp")["g_mag"].count()
    print(f"\n  Obs/asteroid — median: {obs_counts.median():.0f}  "
          f"mean: {obs_counts.mean():.1f}  "
          f"max: {obs_counts.max()}")
    for thresh in [5, 10, 20, 30]:
        n = (obs_counts >= thresh).sum()
        print(f"    ≥{thresh:2d} obs: {n:,} asteroids")

    # ── Save ──────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, compression="snappy")
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n  ✅  Saved: {OUT_PATH}  ({size_mb:.1f} MB)")
    print(f"  {len(df):,} observations  ·  {after_obj:,} asteroids\n")


if __name__ == "__main__":
    main()
