"""
14_variability_flag.py
GAPC — Photometric variability candidates from chi2_reduced outliers.

Asteroids with chi2_reduced >> median (for their n_obs bin) may be
photometrically variable: tumbling objects, contact binaries, or strongly
elongated bodies. This script flags candidates and computes a variability index.

Method:
  For each n_obs quartile, compute median chi2_red and flag objects at
  chi2_red > threshold_sigma * IQR above median as variability candidates.

Outputs:
  data/final/gapc_catalog_v3_var.parquet  (adds var_flag, var_chi2_zscore)
  plots/14_variability_candidates.png
  logs/14_variability_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import median_abs_deviation

ROOT      = Path(__file__).resolve().parents[1]
# prefer v3 if available (with diameter), fall back to v2
CAT_V3    = ROOT / "data" / "final" / "gapc_catalog_v3.parquet"
CAT_V2    = ROOT / "data" / "final" / "gapc_catalog_v2.parquet"
CAT_PATH  = CAT_V3 if CAT_V3.exists() else CAT_V2
OUT_CAT   = ROOT / "data" / "final" / "gapc_catalog_v3_var.parquet"
PLOT_DIR  = ROOT / "plots"
LOG_DIR   = ROOT / "logs"

N_OBS_BINS  = [0, 50, 100, 150, 250, 800]    # n_obs bin edges
SIGMA_THRESH = 5.0                             # MAD-sigma threshold for variability flag


def robust_zscore(values, ref_median, ref_mad):
    """Robust z-score using median and MAD."""
    if ref_mad == 0:
        return np.zeros(len(values))
    return (values - ref_median) / (ref_mad * 1.4826)


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 14 — Photometric variability candidates")
    print("=" * 60)

    df = pd.read_parquet(CAT_PATH)
    n  = len(df)
    print(f"\n  Input: {CAT_PATH.name}  ({n:,} rows)")

    chi2 = df["chi2_reduced"].values
    n_obs = df["n_obs"].values

    var_flag   = np.zeros(n, dtype=bool)
    var_zscore = np.full(n, np.nan)

    print(f"\n  {'n_obs bin':15s}  {'N':>6s}  {'chi2 median':>12s}  "
          f"{'chi2 MAD':>10s}  {'threshold':>10s}  {'n_flagged':>10s}")

    rows = []
    for i in range(len(N_OBS_BINS)-1):
        lo, hi = N_OBS_BINS[i], N_OBS_BINS[i+1]
        mask = (n_obs >= lo) & (n_obs < hi)
        chi2_bin = chi2[mask]
        valid = np.isfinite(chi2_bin)
        if valid.sum() < 10:
            continue
        med  = np.median(chi2_bin[valid])
        mad  = median_abs_deviation(chi2_bin[valid], scale="normal")
        thresh = med + SIGMA_THRESH * mad
        zscore = robust_zscore(chi2_bin, med, mad if mad > 0 else 1.0)
        var_zscore[mask] = zscore
        flagged = valid & (chi2_bin > thresh)
        var_flag[np.where(mask)[0][flagged]] = True
        n_flagged = flagged.sum()
        row = dict(n_obs_bin=f"{lo}-{hi}", n=mask.sum(), chi2_median=med,
                   chi2_mad=mad, threshold=thresh, n_flagged=n_flagged,
                   flag_pct=n_flagged/mask.sum()*100)
        rows.append(row)
        print(f"  {lo:3d}–{hi:3d}             "
              f"{mask.sum():6,}  {med:12.2f}  {mad:10.2f}  "
              f"{thresh:10.2f}  {n_flagged:10,} ({n_flagged/mask.sum()*100:.1f}%)")

    stats_df = pd.DataFrame(rows)
    total_flagged = var_flag.sum()
    print(f"\n  Total variability candidates: {total_flagged:,}  "
          f"({total_flagged/n*100:.2f}% of all objects)")

    # Top candidates
    cands = df[var_flag].copy()
    cands["var_zscore"] = var_zscore[var_flag]
    cands_sorted = cands.sort_values("var_zscore", ascending=False)
    print(f"\n  Top 20 variability candidates:")
    print(f"  {'number_mp':>10s}  {'denomination':20s}  "
          f"{'chi2_red':>10s}  {'zscore':>8s}  {'n_obs':>6s}  {'H_V':>6s}")
    for _, row in cands_sorted.head(20).iterrows():
        denom = str(row.get("denomination", ""))[:20] if pd.notna(row.get("denomination")) else ""
        hv    = f"{row['H_V']:.2f}" if pd.notna(row.get("H_V")) else "—"
        print(f"  {int(row['number_mp']):10d}  {denom:20s}  "
              f"{row['chi2_reduced']:10.1f}  {row['var_zscore']:8.1f}  "
              f"{int(row['n_obs']):6d}  {hv:>6s}")

    # --- Save updated catalog ---
    cat_out = df.copy()
    cat_out["var_flag"]        = var_flag
    cat_out["var_chi2_zscore"] = np.round(var_zscore, 3)
    cat_out.to_parquet(OUT_CAT, index=False, compression="snappy")
    mb = OUT_CAT.stat().st_size / 1e6
    print(f"\n  Saved: {OUT_CAT.name}  ({len(cat_out):,} rows · {mb:.1f} MB)")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("Photometric variability candidates (chi2_red outliers)", fontsize=13)

    ax = axes[0, 0]
    chi2_plot = chi2[np.isfinite(chi2) & (chi2 < 200)]
    ax.hist(chi2_plot, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
    ax.axvline(np.nanmedian(chi2), color="red", lw=1.2,
               label=f"median={np.nanmedian(chi2):.1f}")
    ax.set_xlabel("chi2_reduced")
    ax.set_ylabel("Count")
    ax.set_title("chi2_reduced distribution (< 200)")
    ax.legend()

    ax2 = axes[0, 1]
    ax2.scatter(df.loc[~var_flag, "n_obs"].clip(0, 800),
                chi2[~var_flag].clip(0, 500),
                s=1, alpha=0.05, color="steelblue", rasterized=True, label="normal")
    ax2.scatter(df.loc[var_flag, "n_obs"],
                chi2[var_flag].clip(0, 500),
                s=4, alpha=0.5, color="red", rasterized=True, label=f"variable ({total_flagged:,})")
    ax2.set_xlabel("n_obs")
    ax2.set_ylabel("chi2_reduced (clipped at 500)")
    ax2.set_title(f"Variability flag ({SIGMA_THRESH:.0f}σ threshold)")
    ax2.legend(fontsize=8)

    ax3 = axes[1, 0]
    ax3.hist(var_zscore[np.isfinite(var_zscore)].clip(-5, 50), bins=100,
             color="coral", edgecolor="none", alpha=0.8)
    ax3.axvline(SIGMA_THRESH, color="k", lw=1.2, linestyle="--",
                label=f"flag threshold = {SIGMA_THRESH}σ")
    ax3.set_xlabel("chi2_red robust z-score")
    ax3.set_ylabel("Count")
    ax3.set_title("Variability z-score distribution")
    ax3.legend()

    ax4 = axes[1, 1]
    hv_all  = df.loc[~var_flag, "H_V"].dropna()
    hv_var  = df.loc[ var_flag, "H_V"].dropna()
    bins_hv = np.arange(8, 21, 0.3)
    ax4.hist(hv_all, bins=bins_hv, color="steelblue", alpha=0.6,
             density=True, label="non-variable")
    ax4.hist(hv_var, bins=bins_hv, color="red",      alpha=0.6,
             density=True, label=f"variable candidates ({len(hv_var):,})")
    ax4.set_xlabel("H_V [mag]")
    ax4.set_ylabel("Density")
    ax4.set_title("H_V distribution: variable vs normal")
    ax4.legend(fontsize=9)

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "14_variability_candidates.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → plots/14_variability_candidates.png")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(LOG_DIR / "14_variability_stats.csv", index=False, float_format="%.3f")
    cands_sorted.head(100)[["number_mp", "denomination", "chi2_reduced",
                             "var_zscore", "n_obs", "H_V", "G",
                             "G_uncertain"]].to_csv(
        LOG_DIR / "14_variability_top100.csv", index=False, float_format="%.3f"
    )
    print(f"  Log  → logs/14_variability_stats.csv, 14_variability_top100.csv\n")


if __name__ == "__main__":
    main()
