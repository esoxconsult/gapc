"""
05_crossmatch_gasp.py
GAPC — Cross-match GAPC fits with GASP catalog.

Produces the combined spectral + phase curve catalog.

Input:  data/interim/hg1g2_fits.parquet
        (GASP) gasp_catalog_v1.parquet — auto-detected or via GASP_CATALOG env
Output: data/final/gapc_catalog_v1.parquet
"""

import os
import pandas as pd
from pathlib import Path

ROOT     = Path(__file__).resolve().parents[1]
IN_FITS  = ROOT / "data" / "interim" / "hg1g2_fits.parquet"
OUT_PATH = ROOT / "data" / "final"   / "gapc_catalog_v1.parquet"

# GASP catalog search path (override via env var GASP_CATALOG)
GASP_CANDIDATES = [
    os.environ.get("GASP_CATALOG", ""),
    str(ROOT.parent / "GASP" / "data" / "final" / "gasp_catalog_v1.parquet"),
    str(Path.home() / "gasp" / "data" / "final" / "gasp_catalog_v1.parquet"),
]


def find_gasp() -> str | None:
    for p in GASP_CANDIDATES:
        if p and Path(p).exists():
            return p
    return None


def main():
    print("\n" + "=" * 55)
    print("  GAPC Step 5 — Cross-match with GASP")
    print("=" * 55)

    # ── Load GAPC fits ────────────────────────────────────────────────────
    fits = pd.read_parquet(IN_FITS)
    print(f"\n  GAPC fits: {len(fits):,} asteroids  "
          f"(ok: {fits['fit_ok'].sum():,})")

    # ── Locate & load GASP ───────────────────────────────────────────────
    gasp_path = find_gasp()
    if gasp_path is None:
        print("\n  ⚠️   GASP catalog not found.")
        print("       Set GASP_CATALOG=/path/to/gasp_catalog_v1.parquet")
        print("       Saving GAPC-only catalog (no spectral columns).")
        gasp = None
    else:
        gasp = pd.read_parquet(gasp_path)
        print(f"  GASP catalog: {gasp_path}")
        print(f"  GASP rows: {len(gasp):,}  "
              f"columns: {list(gasp.columns)[:6]} …")

    # ── Add gasp_match flag to all GAPC rows ─────────────────────────────
    if gasp is not None:
        gasp_ids = set(gasp["number_mp"].values)
        fits["gasp_match"] = fits["number_mp"].isin(gasp_ids)
    else:
        fits["gasp_match"] = False

    n_match = fits["gasp_match"].sum()
    print(f"\n  GAPC asteroids with GASP match: {n_match:,}  "
          f"({100*n_match/len(fits):.1f}%)")

    # ── Merge ─────────────────────────────────────────────────────────────
    if gasp is not None:
        # Prefix GASP columns (except join key) to avoid collisions
        gasp_cols = [c for c in gasp.columns if c != "number_mp"]
        gasp_renamed = gasp.rename(
            columns={c: f"gasp_{c}" for c in gasp_cols
                     if not c.startswith("gasp_")}
        )

        catalog = fits.merge(gasp_renamed, on="number_mp", how="left")
    else:
        catalog = fits.copy()

    # ── Column ordering ───────────────────────────────────────────────────
    core_cols = [
        "number_mp", "denomination", "n_obs", "phase_min", "phase_max",
        "phase_range", "H", "G1", "G2", "sigma_H", "sigma_G1", "sigma_G2",
        "chi2_reduced", "fit_ok", "flag_unphysical", "fit_method",
        "gasp_match",
    ]
    other_cols = [c for c in catalog.columns if c not in core_cols]
    catalog = catalog[core_cols + other_cols]

    # ── Save ──────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_parquet(OUT_PATH, index=False, compression="snappy")
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n  ✅  Saved: {OUT_PATH}  ({size_mb:.1f} MB)")
    print(f"  {len(catalog):,} rows  ·  {len(catalog.columns)} columns\n")

    print("  Column summary:")
    for col in catalog.columns[:25]:
        nn = catalog[col].notna().sum()
        print(f"    {col:30s}  {nn:>8,} non-null")
    if len(catalog.columns) > 25:
        print(f"    … and {len(catalog.columns)-25} more columns")


if __name__ == "__main__":
    main()
