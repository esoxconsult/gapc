"""
11_spectral_slope.py
GAPC — GASP spectral slope vs phase curve G-parameter correlation.

Computes the spectral slope S' (% per 100 nm, normalized at 550 nm) from
GASP reflectance spectra and tests correlation with HG slope G.

Scientific motivation: spectral slope traces space weathering and composition;
if G correlates with S' this links photometric behavior to surface state.

Outputs:
  plots/11_spectral_slope_vs_G.png
  logs/11_spectral_slope_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v2.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

NORM_WL = 550   # normalisation wavelength [nm]


def compute_spectral_slope(df):
    """Linear slope fit to reflectance vs wavelength, normalized at NORM_WL."""
    refl_cols = sorted(
        [c for c in df.columns if c.startswith("gasp_refl_")],
        key=lambda c: int(c.split("_")[-1])
    )
    wavelengths = np.array([int(c.split("_")[-1]) for c in refl_cols], dtype=float)
    print(f"  Reflectance channels: {len(refl_cols)}  "
          f"({wavelengths[0]:.0f}–{wavelengths[-1]:.0f} nm)")

    # Subset: GASP-matched with enough valid reflectance points
    gasp = df[df["gasp_match"]].copy()
    refl = gasp[refl_cols].values.astype(float)

    # Normalize each spectrum at NORM_WL (interpolate if exact channel missing)
    norm_idx = np.argmin(np.abs(wavelengths - NORM_WL))
    norm_wl_actual = wavelengths[norm_idx]
    print(f"  Normalisation channel: {norm_wl_actual:.0f} nm (target {NORM_WL} nm)")

    norm_vals = refl[:, norm_idx]
    valid_norm = (norm_vals > 0) & np.isfinite(norm_vals)
    refl_norm = np.where(
        valid_norm[:, None],
        refl / norm_vals[:, None],
        np.nan
    )

    # Linear fit S' per spectrum (% reflectance / 100 nm)
    slopes = np.full(len(gasp), np.nan)
    n_valid_min = max(5, len(refl_cols) // 3)

    for i in range(len(gasp)):
        row = refl_norm[i]
        ok  = np.isfinite(row)
        if ok.sum() < n_valid_min:
            continue
        wl_ok  = wavelengths[ok]
        ref_ok = row[ok]
        # Reject obvious bad points (reflectance < 0.3 or > 3.0 after normalization)
        good = (ref_ok > 0.3) & (ref_ok < 3.0)
        if good.sum() < n_valid_min:
            continue
        coef = np.polyfit(wl_ok[good], ref_ok[good], 1)
        # slope in units of (reflectance/nm); convert to %/100nm relative to ref
        slopes[i] = coef[0] * 100 / 1.0  # already relative since normalized

    gasp = gasp.copy()
    gasp["spectral_slope"] = slopes
    valid = gasp["spectral_slope"].notna() & gasp["G"].notna()
    print(f"  Objects with valid slope + G: {valid.sum():,} / {len(gasp):,}")
    return gasp, valid


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 11 — Spectral slope vs G-parameter")
    print("=" * 60)

    df = pd.read_parquet(CAT_PATH)
    gasp, valid = compute_spectral_slope(df)

    S = gasp.loc[valid, "spectral_slope"].values
    G = gasp.loc[valid, "G"].values

    r_p, p_p = pearsonr(S, G)
    r_s, p_s = spearmanr(S, G)
    print(f"\n  Pearson r  = {r_p:.4f}  (p={p_p:.2e})")
    print(f"  Spearman ρ = {r_s:.4f}  (p={p_s:.2e})")

    # --- By taxonomy ---
    tax_col = "gasp_taxonomy_final"
    if tax_col in gasp.columns:
        print(f"\n  Correlation by taxonomy class:")
        for cls in sorted(gasp.loc[valid, tax_col].dropna().unique()):
            m = valid & (gasp[tax_col] == cls)
            if m.sum() < 20:
                continue
            sv, gv = gasp.loc[m, "spectral_slope"].values, gasp.loc[m, "G"].values
            ok = np.isfinite(sv) & np.isfinite(gv)
            if ok.sum() < 10:
                continue
            rr, pp = pearsonr(sv[ok], gv[ok])
            print(f"    {cls:3s}  n={ok.sum():5,}  r={rr:+.3f}  p={pp:.2e}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("GASP spectral slope vs Gaia HG slope G", fontsize=13)

    ax = axes[0]
    ax.scatter(S, G, s=3, alpha=0.2, color="steelblue", rasterized=True)
    ax.set_xlabel("Spectral slope S' [reflectance/100 nm, norm. 550 nm]")
    ax.set_ylabel("G (HG phase slope)")
    ax.set_title(f"n={valid.sum():,}  Pearson r={r_p:.3f}  p={p_p:.1e}")

    # 2D hexbin for density
    ax2 = axes[1]
    hb = ax2.hexbin(S, G, gridsize=40, cmap="Blues", mincnt=1)
    plt.colorbar(hb, ax=ax2, label="count")
    ax2.set_xlabel("Spectral slope S'")
    ax2.set_ylabel("G")
    ax2.set_title("Density (hexbin)")

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "11_spectral_slope_vs_G.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/11_spectral_slope_vs_G.png")

    # --- Save stats ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "11_spectral_slope_stats.txt", "w") as f:
        f.write(f"n_valid: {valid.sum()}\n")
        f.write(f"Pearson r:  {r_p:.6f}  p={p_p:.4e}\n")
        f.write(f"Spearman r: {r_s:.6f}  p={p_s:.4e}\n")
        f.write(f"S' mean:    {np.nanmean(S):.4f}\n")
        f.write(f"S' median:  {np.nanmedian(S):.4f}\n")
        f.write(f"S' std:     {np.nanstd(S):.4f}\n")
    print(f"  Log  → logs/11_spectral_slope_stats.txt\n")


if __name__ == "__main__":
    main()
