"""
17_g1g2_space.py
GAPC — C2: G1/G2 parameter space analysis for the 5,938 HG1G2-fitted objects.

The HG1G2 system (Muinonen et al. 2010) provides a 2D photometric parameter
space where different surface types occupy distinct loci.

This is the first empirical test of these loci using Gaia DR3 sparse photometry
at this scale (~6K objects vs. ~100 objects in prior ground-based studies).

Theoretical loci from Muinonen et al. (2010, Icarus 209):
  "Low-G" regime  (G1 < 0.2, G2 < 0.2): dark, C/B-type
  "High-G" regime (G1 > 0.4, G2 > 0.2): bright, S-type

Outputs:
  plots/17_g1g2_space.png
  plots/17_g1g2_by_taxonomy.png
  logs/17_g1g2_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kruskal, pearsonr

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v3_var.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

# Muinonen et al. 2010 — theoretical loci for the G1-G2 plane
# Mean + sigma for major surface types (approximate from Penttilä 2016 Fig. 4)
THEORETICAL_LOCI = {
    "C-type":  {"G1": (0.15, 0.06), "G2": (0.09, 0.05), "color": "royalblue"},
    "S-type":  {"G1": (0.53, 0.08), "G2": (0.32, 0.07), "color": "coral"},
    "X-type":  {"G1": (0.26, 0.10), "G2": (0.16, 0.08), "color": "goldenrod"},
}


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 17 — G1/G2 parameter space (C2)")
    print("=" * 60)

    df = pd.read_parquet(CAT_PATH)

    # Filter to HG1G2-fitted objects
    mask = df["G1"].notna() & df["G2"].notna() & df["fit_ok"]
    hg12 = df[mask].copy()
    print(f"\n  HG1G2-fitted objects: {len(hg12):,} / {len(df):,} total")
    print(f"  G1 range: {hg12['G1'].min():.3f} – {hg12['G1'].max():.3f}")
    print(f"  G2 range: {hg12['G2'].min():.3f} – {hg12['G2'].max():.3f}")

    # Physical constraint check: G1 + G2 <= 1
    phy_ok = (hg12["G1"] + hg12["G2"] <= 1.0)
    print(f"  G1+G2 ≤ 1 (physical): {phy_ok.sum():,} ({phy_ok.mean()*100:.1f}%)")

    # Basic statistics
    print(f"\n  G1 stats: median={hg12['G1'].median():.3f}  "
          f"mean={hg12['G1'].mean():.3f}  std={hg12['G1'].std():.3f}")
    print(f"  G2 stats: median={hg12['G2'].median():.3f}  "
          f"mean={hg12['G2'].mean():.3f}  std={hg12['G2'].std():.3f}")

    r, p = pearsonr(hg12["G1"].values, hg12["G2"].values)
    print(f"  G1–G2 Pearson correlation: r={r:.4f}  p={p:.2e}")

    # chi2_red for HG1G2 objects
    c = hg12["chi2_reduced"].dropna()
    print(f"  chi2_red: median={c.median():.2f}  mean={c.mean():.2f}")

    # --- By taxonomy ---
    tax_col = "gasp_taxonomy_final"
    hg12_tax = hg12[hg12[tax_col].notna()]
    classes = (hg12_tax[tax_col].value_counts()
               .loc[lambda x: x >= 10].index.tolist())
    print(f"\n  With taxonomy + HG1G2: {len(hg12_tax):,}")
    print(f"\n  {'Class':5s}  {'N':>5s}  {'G1 med':>8s}  {'G2 med':>8s}  "
          f"{'σ(G1)':>7s}  {'σ(G2)':>7s}")
    rows = []
    for cls in sorted(classes):
        sub = hg12_tax[hg12_tax[tax_col] == cls]
        row = dict(cls=cls, n=len(sub),
                   G1_med=sub["G1"].median(), G2_med=sub["G2"].median(),
                   G1_std=sub["G1"].std(), G2_std=sub["G2"].std())
        rows.append(row)
        print(f"  {cls:5s}  {len(sub):5,}  {sub['G1'].median():8.3f}  "
              f"{sub['G2'].median():8.3f}  {sub['G1'].std():7.3f}  {sub['G2'].std():7.3f}")

    # KW test on G1 across classes
    if len(classes) >= 2:
        groups_g1 = [hg12_tax.loc[hg12_tax[tax_col]==cls,"G1"].dropna().values
                     for cls in classes]
        groups_g1 = [g for g in groups_g1 if len(g) >= 5]
        if len(groups_g1) >= 2:
            H1, p1 = kruskal(*groups_g1)
            print(f"\n  KW test G1 across taxonomy classes: H={H1:.2f}  p={p1:.2e}")

    # --- Plot 1: G1 vs G2 full scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle(f"HG1G2 parameter space  (n={len(hg12):,})", fontsize=13)

    ax = axes[0]
    # Density hexbin for all objects
    hb = ax.hexbin(hg12["G1"], hg12["G2"], gridsize=40, cmap="Blues",
                   mincnt=1, extent=[0, 1, 0, 1])
    plt.colorbar(hb, ax=ax, label="count")

    # Theoretical loci (ellipses)
    theta = np.linspace(0, 2*np.pi, 100)
    for label, locus in THEORETICAL_LOCI.items():
        mx, sx = locus["G1"]; my, sy = locus["G2"]; col = locus["color"]
        ax.plot(mx + 2*sx*np.cos(theta), my + 2*sy*np.sin(theta),
                color=col, lw=1.5, linestyle="--", label=f"{label} 2σ")
        ax.plot(mx, my, "x", color=col, ms=8, mew=2)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.plot([0, 1], [1, 0], "k:", lw=0.8, alpha=0.4, label="G1+G2=1")
    ax.set_xlabel("G1"); ax.set_ylabel("G2")
    ax.set_title("All HG1G2 objects (hexbin density)")
    ax.legend(fontsize=8)

    # G1 vs G2 by taxonomy
    ax2 = axes[1]
    colors_tax = {"S": "coral", "C": "royalblue", "X": "goldenrod",
                  "V": "purple", "D": "green", "B": "teal"}
    if len(hg12_tax) > 0:
        # Background: non-matched
        ax2.scatter(hg12.loc[~mask & df.index.isin(hg12.index), "G1"],
                    hg12.loc[~mask & df.index.isin(hg12.index), "G2"],
                    s=3, alpha=0.1, color="gray", label="No taxonomy")
        # Taxonomy-labeled
        plotted = set()
        for _, row in hg12_tax.iterrows():
            cls = row[tax_col]
            if cls not in colors_tax:
                continue
            lbl = cls if cls not in plotted else None
            ax2.scatter(row["G1"], row["G2"], s=8, alpha=0.5,
                        color=colors_tax[cls], label=lbl)
            plotted.add(cls)
    else:
        ax2.scatter(hg12["G1"], hg12["G2"], s=3, alpha=0.2, color="steelblue")

    for label, locus in THEORETICAL_LOCI.items():
        mx, sx = locus["G1"]; my, sy = locus["G2"]; col = locus["color"]
        ax2.plot(mx + 2*sx*np.cos(theta), my + 2*sy*np.sin(theta),
                 color=col, lw=1.5, linestyle="--")

    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
    ax2.plot([0, 1], [1, 0], "k:", lw=0.8, alpha=0.4)
    ax2.set_xlabel("G1"); ax2.set_ylabel("G2")
    ax2.set_title("G1-G2 by taxonomy class")
    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys(), fontsize=8)

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "17_g1g2_space.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/17_g1g2_space.png")

    # --- Plot 2: G1, G2 marginal distributions ---
    fig2, axes2 = plt.subplots(2, 2, figsize=(11, 8))
    fig2.suptitle("G1 and G2 marginal distributions by taxonomy", fontsize=12)

    for ax_idx, (param, axes_row) in enumerate(zip(["G1","G2"], [axes2[0], axes2[1]])):
        ax_all, ax_tax = axes_row
        ax_all.hist(hg12[param], bins=50, color="steelblue", edgecolor="none", alpha=0.8)
        ax_all.set_xlabel(param); ax_all.set_ylabel("Count")
        ax_all.set_title(f"{param} all HG1G2 objects (n={len(hg12):,})")
        ax_all.axvline(hg12[param].median(), color="red", lw=1.2,
                       label=f"median={hg12[param].median():.3f}")
        ax_all.legend(fontsize=8)

        for cls, col in colors_tax.items():
            sub = hg12_tax.loc[hg12_tax[tax_col]==cls, param].dropna()
            if len(sub) < 5:
                continue
            ax_tax.hist(sub, bins=30, alpha=0.5, color=col, label=f"{cls} (n={len(sub)})",
                        density=True, edgecolor="none")
        ax_tax.set_xlabel(param); ax_tax.set_ylabel("Density (normalized)")
        ax_tax.set_title(f"{param} by taxonomy")
        ax_tax.legend(fontsize=8)

    fig2.tight_layout()
    fig2.savefig(PLOT_DIR / "17_g1g2_by_taxonomy.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Plot → plots/17_g1g2_by_taxonomy.png")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_DIR / "17_g1g2_stats.txt", "w") as f:
        f.write(f"n_hg1g2: {len(hg12)}\n")
        f.write(f"G1_physical_pct: {phy_ok.mean()*100:.2f}\n")
        f.write(f"G1_G2_pearson_r: {r:.6f}\n")
        f.write(f"G1_G2_pearson_p: {p:.4e}\n\n")
        if rows:
            pd.DataFrame(rows).to_csv(f, index=False, float_format="%.4f")
    print(f"  Log  → logs/17_g1g2_stats.txt\n")


if __name__ == "__main__":
    main()
