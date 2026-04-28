"""
48_gapc_gasp_crossmatch.py
GAPC — G × Gaia-spectrum analysis for the ~10K GAPC×GASP overlap.

Objects with both a GAPC phase curve (G) and GASP Gaia reflectance spectrum
(16 bands + 4 slopes) allow us to test whether spectral features — beyond
the coarse taxonomy class — correlate with G.

Hypotheses:
  H1: rho(G, s1) < 0  — steeper (redder) VIS slope → more weathered → lower G
      (opposite to "redder = more weathered = higher G" for outer S-types)
  H2: rho(G, refl_550/refl_900) — band ratio proxy for olivine/pyroxene
  H3: rho(G, NUV) — UV drop correlates with hydration (C-types) or saturation (S)
  H4: After controlling for size, do any spectral features still predict G?

Outputs:
  plots/48_gapc_gasp_crossmatch.png
  logs/48_gapc_gasp_crossmatch_stats.txt
  (v8 NOT modified — read-only)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from scipy.stats import rankdata

ROOT    = Path(__file__).resolve().parents[1]
V8_PATH = ROOT / "data" / "final" / "gapc_catalog_v8.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

BANDS = [374, 418, 462, 506, 550, 594, 638, 682,
         726, 770, 814, 858, 902, 946, 990, 1034]
REFL_COLS  = [f"gasp_refl_{b}" for b in BANDS]
SLOPE_COLS = ["gasp_s1", "gasp_s2", "gasp_s3", "gasp_s4"]


def partial_spearman(x, y, z):
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    bx = np.cov(xr, zr)[0, 1] / np.var(zr)
    by = np.cov(yr, zr)[0, 1] / np.var(zr)
    return pearsonr(xr - bx * zr, yr - by * zr)


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 48 — G × Gaia spectrum (GAPC×GASP crossmatch)")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V8_PATH.exists():
        print(f"\n  ERROR: {V8_PATH} not found"); return

    gapc = pd.read_parquet(V8_PATH)
    print(f"\n  v8 loaded: {len(gapc):,} rows, {len(gapc.columns)} cols")

    # ── Crossmatch subset ─────────────────────────────────────────────────────
    all_feat = REFL_COLS + SLOPE_COLS
    available = [c for c in all_feat if c in gapc.columns]
    if not available:
        print("  No gasp_refl_ columns — check crossmatch step"); return

    cross = gapc[gapc["G"].notna() & gapc[available[0]].notna()].copy()
    # also require all refl columns
    refl_avail = [c for c in REFL_COLS if c in gapc.columns]
    if refl_avail:
        cross = cross[cross[refl_avail].notna().all(axis=1)]

    print(f"\n  Crossmatch with G + spectrum: {len(cross):,} objects")

    # ── G × spectral slopes ───────────────────────────────────────────────────
    print(f"\n  1. G × spectral slopes:")
    slope_results = {}
    for col in SLOPE_COLS:
        if col not in cross.columns:
            continue
        sub = cross[cross[col].notna()]
        if len(sub) < 20:
            continue
        rho, p = spearmanr(sub["G"], sub[col])
        print(f"    {col}: rho={rho:+.4f}  p={p:.2e}  n={len(sub):,}")
        slope_results[col] = dict(rho=rho, p=p, n=len(sub))

    # ── G × each reflectance band ─────────────────────────────────────────────
    print(f"\n  2. G × reflectance band (rho, n={len(cross):,}):")
    band_results = {}
    for col, band in zip(refl_avail, BANDS):
        sub = cross[cross[col].notna()]
        if len(sub) < 20:
            continue
        rho, p = spearmanr(sub["G"], sub[col])
        sig = "*" if p < 0.05 else " "
        print(f"    {band} nm: rho={rho:+.4f}  p={p:.2e}  {sig}")
        band_results[band] = dict(rho=rho, p=p, n=len(sub))

    # ── Partial: G × s1 | logD ────────────────────────────────────────────────
    print(f"\n  3. Partial correlations (controlling for size):")
    ctrl = cross[cross["D_km"].notna() & (cross["D_km"] > 0)].copy()
    ctrl["log_D"] = np.log10(ctrl["D_km"])
    for col, label in [(SLOPE_COLS[0], "s1"), (SLOPE_COLS[1], "s2")]:
        if col not in ctrl.columns:
            continue
        sub = ctrl[ctrl[col].notna()]
        if len(sub) < 20:
            continue
        r_part, p_part = partial_spearman(sub["G"], sub[col], sub["log_D"])
        r_sD,   p_sD   = partial_spearman(sub["G"], sub["log_D"], sub[col])
        print(f"    r(G, {label} | logD) = {r_part:+.4f}  p={p_part:.2e}  n={len(sub):,}")
        print(f"    r(G, logD | {label}) = {r_sD:+.4f}  p={p_sD:.2e}")

    # ── By taxonomy ───────────────────────────────────────────────────────────
    print(f"\n  4. rho(G, s1) by taxonomy_refined:")
    tax = "taxonomy_refined"
    s1_col = "gasp_s1"
    if s1_col in cross.columns and tax in cross.columns:
        for t in ["S", "C", "X", "E", "M"]:
            sub = cross[(cross[tax] == t) & cross[s1_col].notna() & cross["G"].notna()]
            if len(sub) < 20:
                continue
            rho, p = spearmanr(sub["G"], sub[s1_col])
            print(f"    {t}: rho={rho:+.4f}  p={p:.2e}  n={len(sub):,}")

    # ── NUV correlation ───────────────────────────────────────────────────────
    nuv_col = "gasp_nuv_correction_applied"
    refl374 = "gasp_refl_374"
    if refl374 in cross.columns:
        sub374 = cross[cross[refl374].notna()]
        if len(sub374) >= 20:
            rho_uv, p_uv = spearmanr(sub374["G"], sub374[refl374])
            print(f"\n  G × refl_374 (NUV proxy): rho={rho_uv:+.4f}  p={p_uv:.2e}  "
                  f"n={len(sub374):,}")
    else:
        rho_uv, p_uv = np.nan, np.nan

    # ── Band ratio: 550/900 (olivine/pyroxene proxy) ─────────────────────────
    r550 = "gasp_refl_550"; r900 = "gasp_refl_902"
    if r550 in cross.columns and r900 in cross.columns:
        ratio = cross[cross[r550].notna() & cross[r900].notna() & (cross[r900] > 0)].copy()
        ratio["band_ratio"] = ratio[r550] / ratio[r900]
        rho_br, p_br = spearmanr(ratio["G"], ratio["band_ratio"])
        print(f"  G × refl550/refl902 ratio:  rho={rho_br:+.4f}  p={p_br:.2e}  "
              f"n={len(ratio):,}")
    else:
        rho_br, p_br = np.nan, np.nan

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"G × Gaia spectrum  (GAPC×GASP crossmatch, n={len(cross):,})", fontsize=12)

    # rho(G, band) across all 16 bands
    ax = axes[0, 0]
    bands_list  = sorted(band_results.keys())
    rhos_list   = [band_results[b]["rho"] for b in bands_list]
    ps_list     = [band_results[b]["p"]   for b in bands_list]
    colors_b    = ["#d62728" if p < 0.05 else "steelblue" for p in ps_list]
    ax.bar(range(len(bands_list)), rhos_list, color=colors_b, alpha=0.85)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(bands_list)))
    ax.set_xticklabels([str(b) for b in bands_list], rotation=45, fontsize=7)
    ax.set_xlabel("Wavelength [nm]"); ax.set_ylabel("Spearman rho(G, refl)")
    ax.set_title("G vs reflectance by band  (red = p<0.05)")
    ax.grid(alpha=0.2, axis="y")

    # G vs s1 scatter (colour by taxonomy)
    ax = axes[0, 1]
    if "gasp_s1" in cross.columns and tax in cross.columns:
        colors_t = {"S": "#e07b39", "C": "#5c85d6", "X": "#9c27b0",
                    "E": "#e74c3c", "M": "#8e44ad"}
        for t, col in colors_t.items():
            sub = cross[(cross[tax] == t) & cross["gasp_s1"].notna()]
            if len(sub) < 5:
                continue
            smp = sub.sample(min(3000, len(sub)), random_state=42)
            ax.scatter(smp["gasp_s1"], smp["G"], s=3, alpha=0.3,
                       color=col, rasterized=True, label=f"{t} (n={len(sub):,})")
        ax.set_xlabel("s1 (spectral slope 374–506 nm)")
        ax.set_ylabel("G (phase slope)")
        ax.set_title("G vs s1 by taxonomy")
        ax.legend(fontsize=8, markerscale=3)
        ax.grid(alpha=0.2)

    # G vs refl_550 scatter
    ax = axes[1, 0]
    if r550 in cross.columns:
        smp_r = cross[cross[r550].notna()].sample(
            min(8000, len(cross)), random_state=42)
        ax.scatter(smp_r[r550], smp_r["G"], s=2, alpha=0.2,
                   color="steelblue", rasterized=True)
        ax.set_xlabel("refl_550 (normalised at 550 nm → 1)")
        ax.set_ylabel("G")
        ax.set_title("G vs reflectance at 550 nm")
        ax.grid(alpha=0.2)

    # rho(G, slope) bar chart
    ax = axes[1, 1]
    if slope_results:
        slbl = [c.replace("gasp_", "") for c in slope_results]
        srho = [slope_results[c]["rho"] for c in slope_results]
        sp   = [slope_results[c]["p"]   for c in slope_results]
        cols_sl = ["#d62728" if p < 0.05 else "steelblue" for p in sp]
        ax.bar(range(len(slbl)), srho, color=cols_sl, alpha=0.85)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(slbl)))
        ax.set_xticklabels(slbl, fontsize=9)
        ax.set_ylabel("Spearman rho(G, slope)")
        ax.set_title("G vs spectral slopes (red = p<0.05)")
        ax.grid(alpha=0.2, axis="y")
        for i, (rho, n) in enumerate(
                zip(srho, [slope_results[c]["n"] for c in slope_results])):
            ax.text(i, rho + 0.003 * np.sign(rho + 1e-9),
                    f"n={n:,}", ha="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "48_gapc_gasp_crossmatch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/48_gapc_gasp_crossmatch.png")

    with open(LOG_DIR / "48_gapc_gasp_crossmatch_stats.txt", "w") as f:
        f.write("GAPC Step 48 — G × Gaia spectrum (GAPC×GASP crossmatch)\n")
        f.write("=" * 60 + "\n")
        f.write(f"n_crossmatch: {len(cross):,}\n\n")
        f.write("G × reflectance band:\n")
        for b, r in band_results.items():
            f.write(f"  {b} nm: rho={r['rho']:+.4f}  p={r['p']:.2e}\n")
        f.write("\nG × spectral slopes:\n")
        for c, r in slope_results.items():
            f.write(f"  {c}: rho={r['rho']:+.4f}  p={r['p']:.2e}\n")
    print(f"  Log  → logs/48_gapc_gasp_crossmatch_stats.txt\n")


if __name__ == "__main__":
    main()
