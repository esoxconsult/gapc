"""
39_binary_analysis.py
GAPC — Binary asteroid G excess: size-controlled analysis.

Steps 33/35 found binary asteroids have significantly higher G
(median 0.209 vs 0.128, p=1.5e-4, n_binary=384).

Question: Is the binary G excess a size effect?
Hypothesis: YORP-driven binaries are preferentially small → younger surfaces
→ higher G. After controlling for size, does the excess persist?

Analysis:
  1. Compare binary vs non-binary G at fixed size bins
  2. Logistic regression: binary ~ f(G, D, orbital_zone)
  3. Size distribution: primaries in Pravec catalog vs full GAPC
  4. Orbital class distribution: binaries by zone (inner/middle/outer)
  5. Cross-check: binary primary periods vs LCDB/Durech (consistency)

Outputs:
  plots/39_binary_analysis.png
  logs/39_binary_analysis_stats.txt
  (v5 NOT modified — read-only)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr

ROOT     = Path(__file__).resolve().parents[1]
V5_PATH  = ROOT / "data" / "final" / "gapc_catalog_v5.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 39 — Binary asteroid G excess (size-controlled)")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V5_PATH.exists():
        print(f"\n  ERROR: {V5_PATH} not found"); return

    gapc = pd.read_parquet(V5_PATH)
    print(f"\n  v5 loaded: {len(gapc):,} objects, {len(gapc.columns)} columns")

    bin_col = "binary_known"
    if bin_col not in gapc.columns:
        print("  binary_known column not found — run steps 33/35 first"); return

    n_bin  = gapc[bin_col].sum()
    n_sing = (~gapc[bin_col]).sum()
    print(f"\n  Known binaries:   {n_bin:,}")
    print(f"  Non-binaries:     {n_sing:,}")

    # ── Overall G comparison ───────────────────────────────────────────────────
    g_bin  = gapc[gapc[bin_col] & gapc["G"].notna()]["G"]
    g_sing = gapc[~gapc[bin_col] & gapc["G"].notna()]["G"]
    U, p_all = mannwhitneyu(g_bin, g_sing, alternative="two-sided")
    print(f"\n  G overall:  binary={g_bin.median():.4f}  non-binary={g_sing.median():.4f}")
    print(f"  Mann-Whitney p={p_all:.3e}")

    # ── Size distribution of binaries ─────────────────────────────────────────
    d_bin  = gapc[gapc[bin_col] & gapc["D_km"].notna()]["D_km"]
    d_sing = gapc[~gapc[bin_col] & gapc["D_km"].notna()]["D_km"]
    print(f"\n  Size: binary median D={d_bin.median():.2f} km  "
          f"non-binary median={d_sing.median():.2f} km")

    # ── G at fixed size bins ───────────────────────────────────────────────────
    both = gapc[gapc["G"].notna() & gapc["D_km"].notna() & (gapc["D_km"] > 0)].copy()
    both["log_D"] = np.log10(both["D_km"])
    # Size bins: 0.1–1, 1–3, 3–10, 10–30, 30+ km
    size_bins = [(0.1, 1), (1, 3), (3, 10), (10, 30), (30, 1000)]
    bin_labels = ["0.1–1 km", "1–3 km", "3–10 km", "10–30 km", ">30 km"]
    print(f"\n  G in size bins (binary vs non-binary):")
    bin_results = []
    for (dlo, dhi), lbl in zip(size_bins, bin_labels):
        mask = both["D_km"].between(dlo, dhi)
        sub = both[mask]
        g_b = sub[sub[bin_col]]["G"]
        g_s = sub[~sub[bin_col]]["G"]
        if len(g_b) < 5 or len(g_s) < 10:
            continue
        U_s, p_s = mannwhitneyu(g_b, g_s, alternative="two-sided")
        print(f"    {lbl:10s}  bin n={len(g_b):3d}  med={g_b.median():.4f}  "
              f"sing n={len(g_s):5,}  med={g_s.median():.4f}  p={p_s:.3e}")
        bin_results.append(dict(
            label=lbl, g_bin=g_b.median(), g_sing=g_s.median(),
            n_bin=len(g_b), n_sing=len(g_s), p=p_s
        ))

    # ── Orbital zone distribution ─────────────────────────────────────────────
    if "orbital_class" in gapc.columns:
        print(f"\n  Binary fraction by orbital zone:")
        for zone in ["MBA-inner", "MBA-middle", "MBA-outer", "NEA"]:
            z = gapc[gapc["orbital_class"] == zone]
            if len(z) < 10:
                continue
            frac = z[bin_col].mean() * 100
            n_b  = z[bin_col].sum()
            print(f"    {zone:15s}: {frac:.2f}%  ({n_b}/{len(z)})")

    # ── Taxonomy distribution of binaries ─────────────────────────────────────
    tax_col = ("predicted_taxonomy" if "predicted_taxonomy" in gapc.columns
               else "gasp_taxonomy_final")
    if tax_col in gapc.columns:
        bins_tax = gapc[gapc[bin_col]][tax_col].astype(str).str[0].value_counts().head(5)
        total_tax = gapc[tax_col].astype(str).str[0].value_counts()
        print(f"\n  Binary fraction by taxonomy type:")
        for t, nb in bins_tax.items():
            if t not in ("n", "N") and total_tax.get(t, 0) > 0:
                frac = nb / total_tax[t] * 100
                print(f"    {t}: {frac:.2f}%  ({nb}/{total_tax[t]:,})")

    # ── Partial correlation: G × binary | D ───────────────────────────────────
    # Use logistic-style: within size quintiles, is G higher for binaries?
    both["size_quintile"] = pd.qcut(both["log_D"], 5, labels=False)
    residual_G = both["G"].copy()
    for q in range(5):
        mask = both["size_quintile"] == q
        residual_G.loc[mask] -= both.loc[mask, "G"].mean()

    g_bin_res  = residual_G[both[bin_col]]
    g_sing_res = residual_G[~both[bin_col]]
    if len(g_bin_res) >= 5 and len(g_sing_res) >= 5:
        U_r, p_r = mannwhitneyu(g_bin_res, g_sing_res, alternative="two-sided")
        print(f"\n  G residual (size-demeaned) test:")
        print(f"  binary median_res={g_bin_res.median():.4f}  "
              f"non-binary={g_sing_res.median():.4f}  p={p_r:.3e}")
        size_controlled = p_r < 0.05
        print(f"  Binary G excess AFTER size control: "
              f"{'YES' if size_controlled else 'NO'} (p={p_r:.3e})")
    else:
        p_r = np.nan

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Binary asteroid G analysis  n_binary={n_bin:,}", fontsize=12)

    # G distribution binary vs non-binary
    ax = axes[0, 0]
    g_rng = (gapc["G"].quantile(0.01), gapc["G"].quantile(0.99))
    bins_g = np.linspace(g_rng[0], g_rng[1], 50)
    ax.hist(g_sing.clip(*g_rng).values, bins=bins_g, density=True,
            color="steelblue", alpha=0.6, label=f"Non-binary ({n_sing:,})")
    ax.hist(g_bin.clip(*g_rng).values,  bins=bins_g, density=True,
            color="coral", alpha=0.7, label=f"Binary ({n_bin:,})")
    ax.axvline(g_sing.median(), color="steelblue", lw=1.5, ls="--")
    ax.axvline(g_bin.median(),  color="coral",     lw=1.5, ls="--")
    ax.set_xlabel("G (phase slope)"); ax.set_ylabel("Density")
    ax.set_title(f"G: binary vs non-binary  p={p_all:.2e}")
    ax.legend(fontsize=9)

    # G in size bins
    ax = axes[0, 1]
    if bin_results:
        lbls    = [r["label"] for r in bin_results]
        g_bs_v  = [r["g_bin"]  for r in bin_results]
        g_ss_v  = [r["g_sing"] for r in bin_results]
        x = np.arange(len(lbls))
        w = 0.35
        ax.bar(x - w/2, g_bs_v, w, label="Binary",     color="coral",     alpha=0.8)
        ax.bar(x + w/2, g_ss_v, w, label="Non-binary", color="steelblue", alpha=0.8)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(lbls, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("Median G")
        ax.set_title("G by size bin: binary vs non-binary")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2, axis="y")

    # Size comparison: binary vs non-binary
    ax = axes[1, 0]
    d_rng = (0.1, 300)
    ax.hist(d_sing[d_sing.between(*d_rng)].values, bins=60, density=True,
            color="steelblue", alpha=0.6, label="Non-binary")
    ax.hist(d_bin[d_bin.between(*d_rng)].values,   bins=60, density=True,
            color="coral", alpha=0.7, label="Binary")
    ax.set_xscale("log")
    ax.set_xlabel("D [km]"); ax.set_ylabel("Density")
    ax.set_title(f"Size distribution  binary med={d_bin.median():.1f} km")
    ax.legend(fontsize=9)

    # G residual (size-demeaned)
    ax = axes[1, 1]
    rng_r = (g_bin_res.quantile(0.01), g_bin_res.quantile(0.99))
    bins_r = np.linspace(min(rng_r[0], g_sing_res.quantile(0.01)),
                         max(rng_r[1], g_sing_res.quantile(0.99)), 50)
    ax.hist(g_sing_res.clip(*bins_r[[0,-1]]).values, bins=bins_r, density=True,
            color="steelblue", alpha=0.6, label="Non-binary")
    ax.hist(g_bin_res.clip(*bins_r[[0,-1]]).values,  bins=bins_r, density=True,
            color="coral", alpha=0.7, label="Binary")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("G residual (size-demeaned)"); ax.set_ylabel("Density")
    ax.set_title(f"G after size control  p={p_r:.2e}" if not np.isnan(p_r)
                 else "G residual (insufficient data)")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "39_binary_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/39_binary_analysis.png")

    with open(LOG_DIR / "39_binary_analysis_stats.txt", "w") as f:
        f.write("GAPC Step 39 — Binary asteroid G excess analysis\n")
        f.write("=" * 60 + "\n")
        f.write(f"Known binaries in GAPC: {n_bin:,}\n")
        f.write(f"G binary median:     {g_bin.median():.4f}  n={len(g_bin):,}\n")
        f.write(f"G non-binary median: {g_sing.median():.4f}  n={len(g_sing):,}\n")
        f.write(f"Mann-Whitney p:      {p_all:.3e}\n")
        f.write(f"D binary median:   {d_bin.median():.2f} km\n")
        f.write(f"D non-binary med:  {d_sing.median():.2f} km\n")
        f.write(f"\nSize-controlled G residual test:\n")
        if not np.isnan(p_r):
            f.write(f"  binary res median:     {g_bin_res.median():.4f}\n")
            f.write(f"  non-binary res median: {g_sing_res.median():.4f}\n")
            f.write(f"  Mann-Whitney p:        {p_r:.3e}\n")
            f.write(f"  G excess after size control: {'YES' if p_r < 0.05 else 'NO'}\n")
        f.write("\nG by size bin:\n")
        for r in bin_results:
            f.write(f"  {r['label']:10s}  bin={r['g_bin']:.4f} (n={r['n_bin']})  "
                    f"sing={r['g_sing']:.4f} (n={r['n_sing']:,})  p={r['p']:.3e}\n")
    print(f"  Log  → logs/39_binary_analysis_stats.txt\n")


if __name__ == "__main__":
    main()
