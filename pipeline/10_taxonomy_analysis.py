"""
10_taxonomy_analysis.py
GAPC — Taxonomy class × phase curve parameter analysis.

Tests whether different taxonomic classes (S, C, X, ...) show systematically
different G-slope distributions — as predicted by surface composition models.

Reference: Penttilä et al. 2016, Planet. Space Sci. 123 (taxonomy-dependent G)

Outputs:
  plots/10_taxonomy_G_boxplot.png
  plots/10_taxonomy_bias_boxplot.png
  logs/10_taxonomy_stats.txt
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kruskal

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v2.parquet"
MPC_PATH = ROOT / "data" / "raw"   / "mpc_h_magnitudes.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

MIN_PER_CLASS = 30   # minimum objects for a class to appear in plots


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 10 — Taxonomy × phase curve analysis")
    print("=" * 60)

    df  = pd.read_parquet(CAT_PATH)
    mpc = pd.read_parquet(MPC_PATH)

    # GASP-matched subset with taxonomy
    gasp = df[df["gasp_match"]].copy()
    tax_col = "gasp_taxonomy_final"
    gasp = gasp[gasp[tax_col].notna()].copy()
    print(f"\n  GASP-matched with taxonomy: {len(gasp):,}")

    # Merge MPC H for bias computation
    gasp = gasp.merge(mpc[["number_mp", "H_mpc"]], on="number_mp", how="left")
    gasp["bias_HV"] = gasp["H_V"] - gasp["H_mpc"]

    # Per-class stats
    classes = (gasp[tax_col].value_counts()
               .loc[lambda x: x >= MIN_PER_CLASS].index.tolist())
    classes.sort()
    print(f"\n  Classes with ≥{MIN_PER_CLASS} objects: {classes}")

    rows = []
    for cls in classes:
        sub = gasp[gasp[tax_col] == cls]
        g   = sub["G"].dropna()
        bv  = sub["bias_HV"].dropna()
        row = dict(
            class_=cls,
            n=len(sub),
            G_median=g.median(),
            G_std=g.std(),
            G_uncertain_pct=(sub["G_uncertain"].sum()/len(sub)*100),
            bias_median=bv.median(),
            bias_std=bv.std(),
            n_mpc=len(bv),
        )
        rows.append(row)
        print(f"  {cls:3s}  n={len(sub):5,}  "
              f"G median={g.median():.3f}±{g.std():.3f}  "
              f"G_uncertain={sub['G_uncertain'].mean()*100:.0f}%  "
              f"bias={bv.median():+.3f} (n={len(bv):,})")

    stats_df = pd.DataFrame(rows).rename(columns={"class_": "class"})

    # Kruskal-Wallis test: does G differ significantly across classes?
    groups = [gasp.loc[gasp[tax_col] == cls, "G"].dropna().values for cls in classes]
    H_stat, p_kw = kruskal(*groups)
    print(f"\n  Kruskal-Wallis (G across {len(classes)} classes): "
          f"H={H_stat:.2f}  p={p_kw:.2e}")

    # --- Plot 1: G distribution by taxonomy (boxplot) ---
    fig, ax = plt.subplots(figsize=(max(8, len(classes)*0.9), 5))
    data_g = [gasp.loc[gasp[tax_col] == cls, "G"].dropna().values for cls in classes]
    bp = ax.boxplot(data_g, labels=classes, patch_artist=True,
                    medianprops={"color": "red", "lw": 1.5},
                    flierprops={"marker": ".", "ms": 2, "alpha": 0.3})
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Taxonomy class (GASP ML)")
    ax.set_ylabel("G (HG slope)")
    ax.set_title(f"Phase slope G by taxonomy  ·  KW p={p_kw:.1e}  (n_class≥{MIN_PER_CLASS})")
    ax.axhline(df["G"].median(), color="gray", lw=0.8, linestyle="--",
               label=f"All-object median G={df['G'].median():.3f}")
    ax.legend(fontsize=9)
    for i, cls in enumerate(classes):
        n = (gasp[tax_col] == cls).sum()
        ax.text(i+1, ax.get_ylim()[0]*0.97, f"n={n}", ha="center", fontsize=7, color="gray")
    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "10_taxonomy_G_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/10_taxonomy_G_boxplot.png")

    # --- Plot 2: H_V - H_MPC bias by taxonomy ---
    classes_bias = [c for c in classes
                    if gasp.loc[gasp[tax_col] == c, "bias_HV"].dropna().shape[0] >= 10]
    if classes_bias:
        fig, ax = plt.subplots(figsize=(max(8, len(classes_bias)*0.9), 5))
        data_b = [gasp.loc[gasp[tax_col] == cls, "bias_HV"].dropna().clip(-2, 2).values
                  for cls in classes_bias]
        bp = ax.boxplot(data_b, labels=classes_bias, patch_artist=True,
                        medianprops={"color": "red", "lw": 1.5},
                        flierprops={"marker": ".", "ms": 2, "alpha": 0.3})
        colors = plt.cm.tab20(np.linspace(0, 1, len(classes_bias)))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(0,    color="k",    lw=0.8, linestyle="--")
        ax.axhline(gasp["bias_HV"].median(), color="gray", lw=0.8, linestyle=":",
                   label=f"Overall median={gasp['bias_HV'].median():+.3f}")
        ax.set_xlabel("Taxonomy class (GASP ML)")
        ax.set_ylabel("H_V − H_MPC [mag]")
        ax.set_title("Bias H_V − H_MPC by taxonomy class")
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / "10_taxonomy_bias_boxplot.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot → plots/10_taxonomy_bias_boxplot.png")

    # --- Save stats ---
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(LOG_DIR / "10_taxonomy_stats.csv", index=False, float_format="%.4f")
    with open(LOG_DIR / "10_taxonomy_stats.txt", "w") as f:
        f.write(stats_df.to_string(index=False))
        f.write(f"\n\nKruskal-Wallis: H={H_stat:.3f}  p={p_kw:.3e}\n")
    print(f"  Log  → logs/10_taxonomy_stats.txt\n")


if __name__ == "__main__":
    main()
