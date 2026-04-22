"""
06_validate.py
GAPC — Validation against MPC and NEOWISE H magnitudes.

Analogous to ECAS validation in GASP.
Produces:
  - Pearson r and RMS vs. MPC H
  - Pearson r and RMS vs. NEOWISE H
  - plots/validation_H_mpc.png
  - plots/validation_H_neowise.png
  - logs/06_validation_stats.txt

Input:  data/final/gapc_catalog_v1.parquet
        data/raw/mpc_orbital_classes.parquet  (from GASP)
        data/raw/neowise_masiero2017.parquet  (from GASP)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

ROOT     = Path(__file__).resolve().parents[1]
IN_PATH  = ROOT / "data" / "final" / "gapc_catalog_v1.parquet"
PLOT_DIR = ROOT / "plots"

# Reference catalog paths (check GASP data dir if not local)
def _find(local_rel, gasp_rel=None):
    p = ROOT / local_rel
    if p.exists():
        return p
    if gasp_rel:
        p2 = ROOT.parent / "GASP" / gasp_rel
        if p2.exists():
            return p2
        p3 = Path.home() / "gasp" / gasp_rel
        if p3.exists():
            return p3
    return None

# mpc_h_magnitudes.parquet is produced by 02b_mpc_h.py (H from MPCORB.DAT).
# mpc_orbital_classes.parquet (from GASP) has orbit_code/a_au but no H column.
MPC_H_PATH   = _find("data/raw/mpc_h_magnitudes.parquet")
MPC_PATH     = _find("data/raw/mpc_orbital_classes.parquet",
                      "data/raw/mpc_orbital_classes.parquet")
NEOWISE_PATH = _find("data/raw/neowise_masiero2017.parquet",
                      "data/raw/neowise_masiero2017.parquet")


def compute_stats(gapc_H, ref_H, label):
    mask = np.isfinite(gapc_H) & np.isfinite(ref_H)
    n = mask.sum()
    if n < 10:
        print(f"  {label}: only {n} valid pairs — skipping")
        return {}
    g = gapc_H[mask]
    r = ref_H[mask]
    rms = np.sqrt(np.mean((g - r) ** 2))
    bias = np.mean(g - r)
    corr, pval = pearsonr(g, r)
    stats = dict(n=n, pearson_r=corr, p_value=pval, rms=rms, bias=bias)
    print(f"\n  {label}  (n={n:,})")
    print(f"    Pearson r = {corr:.4f}  (p={pval:.2e})")
    print(f"    RMS       = {rms:.4f} mag")
    print(f"    Bias      = {bias:+.4f} mag  (GAPC − {label})")
    return stats


def plot_comparison(gapc_H, ref_H, label, out_path, units="mag"):
    mask = np.isfinite(gapc_H) & np.isfinite(ref_H)
    if mask.sum() < 10:
        return
    g, r = gapc_H[mask], ref_H[mask]
    rms = np.sqrt(np.mean((g - r) ** 2))
    corr, _ = pearsonr(g, r)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"GAPC H vs {label}", fontsize=13)

    # Scatter
    ax = axes[0]
    ax.scatter(r, g, s=2, alpha=0.3, color="steelblue", rasterized=True)
    lo, hi = min(r.min(), g.min()), max(r.max(), g.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="1:1")
    ax.set_xlabel(f"H ({label}) [{units}]")
    ax.set_ylabel(f"H (GAPC) [{units}]")
    ax.set_title(f"r={corr:.3f}  RMS={rms:.3f} mag  n={mask.sum():,}")
    ax.legend(fontsize=8)

    # Residuals
    ax2 = axes[1]
    res = g - r
    ax2.hist(res, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
    ax2.axvline(0, color="k", lw=0.8, linestyle="--")
    ax2.axvline(np.mean(res), color="red", lw=1.2, linestyle="-",
                label=f"bias={np.mean(res):+.3f}")
    ax2.set_xlabel(f"GAPC − {label} H [mag]")
    ax2.set_ylabel("Count")
    ax2.set_title("Residual distribution")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Plot → {out_path}")


def main():
    print("\n" + "=" * 55)
    print("  GAPC Step 6 — Validation")
    print("=" * 55)

    catalog = pd.read_parquet(IN_PATH)
    ok = catalog[catalog["fit_ok"]].copy()
    print(f"\n  Catalog: {len(catalog):,} total  ·  {len(ok):,} fitted")

    all_stats = {}

    # ── MPC validation ────────────────────────────────────────────────────
    # Prefer mpc_h_magnitudes.parquet (from 02b_mpc_h.py, H_mpc column).
    # Fall back to mpc_orbital_classes.parquet if it ever gains an H column.
    mpc_h_source = None
    if MPC_H_PATH and MPC_H_PATH.exists():
        mpc_h_source = pd.read_parquet(MPC_H_PATH).rename(columns={"H_mpc": "h_mpc"})
        print(f"\n  MPC H: {len(mpc_h_source):,} rows (from mpc_h_magnitudes.parquet)")
    elif MPC_PATH and MPC_PATH.exists():
        mpc_raw = pd.read_parquet(MPC_PATH)
        h_col = next((c for c in mpc_raw.columns
                      if c.lower() in ["h", "h_mag", "absolute_magnitude"]), None)
        if h_col:
            mpc_h_source = mpc_raw.rename(columns={h_col: "h_mpc"})
            print(f"\n  MPC H: {len(mpc_h_source):,} rows (from mpc_orbital_classes.parquet)")

    if mpc_h_source is not None:
        merged = ok.merge(mpc_h_source[["number_mp", "h_mpc"]], on="number_mp", how="left")
        stats = compute_stats(merged["H"].values, merged["h_mpc"].values, "MPC")
        all_stats["mpc"] = stats
        plot_comparison(
            merged["H"].values, merged["h_mpc"].values,
            "MPC", PLOT_DIR / "validation_H_mpc.png",
        )
    else:
        print("\n  ⚠️   MPC H not available — run pipeline/02b_mpc_h.py first")

    # ── NEOWISE validation ────────────────────────────────────────────────
    if NEOWISE_PATH and NEOWISE_PATH.exists():
        neo = pd.read_parquet(NEOWISE_PATH)
        print(f"\n  NEOWISE: {len(neo):,} rows, columns: {list(neo.columns)}")

        # Try direct H column first, fall back to deriving from D + p_V
        h_col = next((c for c in neo.columns
                      if c.lower() in ["h", "h_mag", "hmag", "h_v"]), None)
        if h_col:
            neo = neo.rename(columns={h_col: "h_neowise"})
            print(f"  Using NEOWISE H column: '{h_col}'")
        else:
            d_col  = next((c for c in neo.columns
                           if c.lower() in ["d", "diam", "diameter", "d_km", "diameter_km"]), None)
            pv_col = next((c for c in neo.columns
                           if c.lower() in ["p_v", "pv", "albedo", "pv_mean",
                                            "p_v_mean", "geometric_albedo"]), None)
            if d_col and pv_col:
                print(f"  Deriving NEOWISE H from D='{d_col}', p_V='{pv_col}'")
                neo["h_neowise"] = np.nan
                mask = (neo[d_col] > 0) & (neo[pv_col] > 0)
                neo.loc[mask, "h_neowise"] = -5.0 * np.log10(
                    (neo.loc[mask, d_col] / 1329.0)
                    * np.sqrt(neo.loc[mask, pv_col])
                )
                print(f"    Derived H for {mask.sum():,} / {len(neo):,} rows")
            else:
                print(f"  ⚠️   NEOWISE: no H column and no D/p_V columns found")

        if "h_neowise" in neo.columns:
            merged = ok.merge(neo[["number_mp", "h_neowise"]], on="number_mp", how="left")
            stats = compute_stats(merged["H"].values, merged["h_neowise"].values, "NEOWISE")
            all_stats["neowise"] = stats
            plot_comparison(
                merged["H"].values, merged["h_neowise"].values,
                "NEOWISE", PLOT_DIR / "validation_H_neowise.png",
            )
    else:
        print("\n  ⚠️   NEOWISE catalog not found — skipping NEOWISE validation")

    # ── Internal consistency ──────────────────────────────────────────────
    print(f"\n  Internal H distribution (fitted objects):")
    for q in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]:
        print(f"    p{int(q*100):02d}: {ok['H'].quantile(q):.2f} mag")

    print(f"\n  chi2_reduced distribution:")
    chi2 = ok["chi2_reduced"]
    print(f"    median={chi2.median():.2f}  "
          f"mean={chi2.mean():.2f}  "
          f"p95={chi2.quantile(0.95):.2f}")
    print(f"    Fraction with chi2_red > 5: "
          f"{(chi2 > 5).mean()*100:.1f}%")

    # ── Save stats ────────────────────────────────────────────────────────
    stats_path = ROOT / "logs" / "06_validation_stats.txt"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        f.write("GAPC Validation Statistics\n")
        f.write("=" * 40 + "\n\n")
        for source, s in all_stats.items():
            f.write(f"vs {source.upper()}:\n")
            for k, v in s.items():
                f.write(f"  {k}: {v}\n")
            f.write("\n")

    print(f"\n  ✅  Stats saved: {stats_path}\n")


if __name__ == "__main__":
    main()
