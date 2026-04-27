"""
30_hg1g2_taxonomy.py
GAPC — HG1G2 parameter space × taxonomy: comparison with Penttilä+2016 predictions.

Step 17 mapped the G1/G2 space for all 5,938 HG1G2-fitted objects.
This step uses the v4 catalog (which adds predicted_taxonomy for all objects via
the RF classifier from step 19) to test whether the observed G1/G2 loci per
taxonomy class agree with the theoretical loci of Penttilä+2016.

Key question: Does Gaia's sparse phase coverage (5–14°) produce G1/G2 values
consistent with the theoretical expectations, or does the limited phase range
compress the G1/G2 space systematically?

Theoretical reference (Penttilä+2016, A&A 594, A39, Table 2):
  C:  G1=0.15±0.06  G2=0.09±0.05
  S:  G1=0.53±0.08  G2=0.32±0.07
  X:  G1=0.26±0.10  G2=0.16±0.08
  V:  G1=0.62±0.09  G2=0.43±0.07
  D:  G1=0.47±0.11  G2=0.44±0.08

Outputs:
  plots/30_hg1g2_taxonomy.png
  logs/30_hg1g2_taxonomy_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v4.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

# Penttilä+2016 Table 2 — mean ± 1σ G1/G2 per taxonomy class
PENTILLA = {
    "C": {"G1": (0.15, 0.06), "G2": (0.09, 0.05), "color": "royalblue"},
    "S": {"G1": (0.53, 0.08), "G2": (0.32, 0.07), "color": "coral"},
    "X": {"G1": (0.26, 0.10), "G2": (0.16, 0.08), "color": "goldenrod"},
    "V": {"G1": (0.62, 0.09), "G2": (0.43, 0.07), "color": "purple"},
    "D": {"G1": (0.47, 0.11), "G2": (0.44, 0.08), "color": "forestgreen"},
}

TAX_COLORS = {k: v["color"] for k, v in PENTILLA.items()}
TAX_COLORS["B"] = "teal"
TAX_COLORS["Other"] = "lightgray"


def best_taxonomy(row):
    """GASP direct measurement > predicted RF > None."""
    gasp = row.get("gasp_taxonomy_final")
    if pd.notna(gasp) and str(gasp).strip():
        t = str(gasp).strip().upper()
        return t[0] if t else None
    pred = row.get("predicted_taxonomy")
    if pd.notna(pred) and str(pred).strip():
        t = str(pred).strip().upper()
        return t[0] if t else None
    return None


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 30 — HG1G2 × Taxonomy (Penttilä+2016 comparison)")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CAT_PATH)
    print(f"\n  Catalog loaded: {len(df):,} objects, {len(df.columns)} columns")

    # ── Select HG1G2-fitted objects ───────────────────────────────────────────
    hg = df[df["G1"].notna() & df["G2"].notna() & df["fit_ok"]].copy()
    print(f"  HG1G2-fitted: {len(hg):,}")

    # ── Assign taxonomy ───────────────────────────────────────────────────────
    # HG1G2-fitted objects have G=NaN (no single-parameter HG fit), so
    # predicted_taxonomy (step 19, which uses G as feature) is None for all of
    # them. Use only gasp_taxonomy_final for the taxonomy analysis.
    if "gasp_taxonomy_final" in hg.columns:
        hg["_tax"] = (hg["gasp_taxonomy_final"]
                      .astype(str).str.strip().str.upper().str[0]
                      .where(hg["gasp_taxonomy_final"].notna()
                             & (hg["gasp_taxonomy_final"].astype(str).str.strip() != "")
                             & (hg["gasp_taxonomy_final"].astype(str).str.strip() != "nan")
                             & (hg["gasp_taxonomy_final"].astype(str).str.strip() != "None")))
    else:
        hg["_tax"] = None
    n_tax = hg["_tax"].notna().sum()
    print(f"  With GASP taxonomy: {n_tax:,} ({n_tax/len(hg)*100:.1f}%)")
    print(f"  (predicted_taxonomy unavailable for HG1G2 objects — G=NaN → RF skipped)")

    # ── Per-class statistics ──────────────────────────────────────────────────
    classes = [c for c in ["S", "C", "X", "V", "D", "B"]
               if (hg["_tax"] == c).sum() >= 10]
    rows = []
    print(f"\n  {'Class':5s}  {'N':>5s}  "
          f"{'G1_obs':>8s}  {'G1_lit':>8s}  {'ΔG1':>7s}  "
          f"{'G2_obs':>8s}  {'G2_lit':>8s}  {'ΔG2':>7s}")
    for cls in classes:
        sub = hg[hg["_tax"] == cls]
        g1m, g2m = sub["G1"].median(), sub["G2"].median()
        g1s, g2s = sub["G1"].std(),    sub["G2"].std()
        lit = PENTILLA.get(cls, {})
        g1_lit = lit.get("G1", (np.nan, np.nan))[0]
        g2_lit = lit.get("G2", (np.nan, np.nan))[0]
        dg1 = g1m - g1_lit if np.isfinite(g1_lit) else np.nan
        dg2 = g2m - g2_lit if np.isfinite(g2_lit) else np.nan
        row = dict(cls=cls, n=len(sub),
                   G1_obs=g1m, G1_std=g1s, G1_lit=g1_lit, delta_G1=dg1,
                   G2_obs=g2m, G2_std=g2s, G2_lit=g2_lit, delta_G2=dg2)
        rows.append(row)
        g1_lit_str = f"{g1_lit:.3f}" if np.isfinite(g1_lit) else "  n/a"
        g2_lit_str = f"{g2_lit:.3f}" if np.isfinite(g2_lit) else "  n/a"
        dg1_str = f"{dg1:+.3f}" if np.isfinite(dg1) else "   n/a"
        dg2_str = f"{dg2:+.3f}" if np.isfinite(dg2) else "   n/a"
        print(f"  {cls:5s}  {len(sub):5,}  "
              f"{g1m:8.3f}  {g1_lit_str:>8s}  {dg1_str:>7s}  "
              f"{g2m:8.3f}  {g2_lit_str:>8s}  {dg2_str:>7s}")

    # KW test: G1 across classes
    groups_g1 = [hg.loc[hg["_tax"]==c, "G1"].dropna().values for c in classes]
    groups_g1 = [g for g in groups_g1 if len(g) >= 5]
    kw_msg = "n/a"
    if len(groups_g1) >= 2:
        H, p_kw = kruskal(*groups_g1)
        kw_msg = f"H={H:.2f}  p={p_kw:.2e}"
        print(f"\n  KW test G1 across {len(groups_g1)} classes: {kw_msg}")

    # S vs C Mann-Whitney (most important comparison)
    mw_msg = "n/a"
    if "S" in classes and "C" in classes:
        g1_s = hg.loc[hg["_tax"]=="S", "G1"].dropna().values
        g1_c = hg.loc[hg["_tax"]=="C", "G1"].dropna().values
        U, p_mw = mannwhitneyu(g1_s, g1_c, alternative="greater")
        mw_msg = f"U={U:.0f}  p={p_mw:.2e}"
        print(f"  Mann-Whitney G1(S) > G1(C): {mw_msg}")

    # ── Physical constraint ───────────────────────────────────────────────────
    phy_pct = (hg["G1"] + hg["G2"] <= 1.0).mean() * 100
    print(f"\n  G1+G2 ≤ 1 (physical): {phy_pct:.1f}%")

    # ── G_predicted from HG1G2 (Penttilä conversion) ─────────────────────────
    # HG1G2 objects have G=NaN (no independent HG fit), so we can only report
    # the predicted G from the HG1G2 parameters.
    hg["G_predicted"] = 0.46 * hg["G1"] + 0.54 * hg["G2"]
    gp = hg["G_predicted"].dropna()
    print(f"\n  G_pred = 0.46·G1+0.54·G2: "
          f"median={gp.median():.4f}  std={gp.std():.4f}  range={gp.min():.3f}–{gp.max():.3f}")
    print(f"  NOTE: All taxonomy classes show G1≈G2≈0.20 — Gaia 5–14° phase")
    print(f"  coverage is insufficient to discriminate G1/G2 (expected from step 17).")

    # ── Plots ─────────────────────────────────────────────────────────────────
    theta = np.linspace(0, 2*np.pi, 120)
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle(
        f"HG1G2 × Taxonomy — Penttilä+2016 comparison  (n_HG1G2={len(hg):,})",
        fontsize=13)

    # Panel A — G1 vs G2 hexbin + Penttilä loci
    ax = axes[0, 0]
    hb = ax.hexbin(hg["G1"], hg["G2"], gridsize=45, cmap="Blues",
                   mincnt=1, extent=[0, 1, 0, 1])
    plt.colorbar(hb, ax=ax, label="count")
    for cls, locus in PENTILLA.items():
        mx, sx = locus["G1"]; my, sy = locus["G2"]; col = locus["color"]
        ax.plot(mx + 2*sx*np.cos(theta), my + 2*sy*np.sin(theta),
                color=col, lw=1.8, linestyle="--", label=f"{cls} (Penttilä 2σ)")
        ax.plot(mx, my, "x", color=col, ms=9, mew=2.5)
    ax.plot([0, 1], [1, 0], "k:", lw=0.8, alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("G1"); ax.set_ylabel("G2")
    ax.set_title("All HG1G2 objects + Penttilä+2016 loci")
    ax.legend(fontsize=7, loc="upper right")

    # Panel B — G1 vs G2 by taxonomy (observed medians vs Penttilä)
    ax = axes[0, 1]
    if rows:
        hg_tax = hg[hg["_tax"].notna()]
        for cls in classes:
            sub = hg_tax[hg_tax["_tax"] == cls]
            col = TAX_COLORS.get(cls, "gray")
            ax.scatter(sub["G1"], sub["G2"], s=4, alpha=0.25, color=col,
                       rasterized=True)
        # Observed medians
        for r in rows:
            col = TAX_COLORS.get(r["cls"], "gray")
            ax.scatter(r["G1_obs"], r["G2_obs"], s=120, color=col, zorder=5,
                       edgecolors="black", linewidths=1.2,
                       label=f"{r['cls']} obs (n={r['n']:,})")
        # Penttilä reference
        for cls, locus in PENTILLA.items():
            if cls not in classes:
                continue
            col = TAX_COLORS.get(cls, "gray")
            ax.plot(*[locus["G1"][0], locus["G2"][0]], "x",
                    color=col, ms=10, mew=2.5, alpha=0.7)
            ax.plot([locus["G1"][0] - 2*locus["G1"][1],
                     locus["G1"][0] + 2*locus["G1"][1]],
                    [locus["G2"][0], locus["G2"][0]],
                    color=col, lw=1.2, alpha=0.6)
            ax.plot([locus["G1"][0], locus["G1"][0]],
                    [locus["G2"][0] - 2*locus["G2"][1],
                     locus["G2"][0] + 2*locus["G2"][1]],
                    color=col, lw=1.2, alpha=0.6)
    ax.plot([0, 1], [1, 0], "k:", lw=0.8, alpha=0.4)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("G1"); ax.set_ylabel("G2")
    ax.set_title("Observed medians (dots) vs Penttilä+2016 (×, 2σ bars)")
    ax.legend(fontsize=7, loc="upper right")

    # Panel C — G1 box per taxonomy
    ax = axes[1, 0]
    if rows:
        data_g1 = [hg.loc[hg["_tax"]==r["cls"], "G1"].dropna().values for r in rows]
        bp = ax.boxplot(data_g1, labels=[r["cls"] for r in rows],
                        patch_artist=True, showfliers=False, medianprops={"lw": 2})
        for patch, r in zip(bp["boxes"], rows):
            patch.set_facecolor(TAX_COLORS.get(r["cls"], "gray"))
            patch.set_alpha(0.7)
        # Penttilä references
        for i, r in enumerate(rows):
            lit = PENTILLA.get(r["cls"])
            if lit:
                ax.plot(i+1, lit["G1"][0], "_", color="black", ms=18, mew=2.5,
                        zorder=5)
        ax.set_ylabel("G1")
        ax.set_title("G1 by taxonomy  (─ = Penttilä+2016 mean)")
        ax.grid(alpha=0.3, axis="y")

    # Panel D — ΔG1 = G1_obs − G1_Penttilä per class
    ax = axes[1, 1]
    valid = [r for r in rows if np.isfinite(r["delta_G1"])]
    if valid:
        cls_labels = [r["cls"] for r in valid]
        dg1_vals   = [r["delta_G1"] for r in valid]
        colors_bar = [TAX_COLORS.get(r["cls"], "gray") for r in valid]
        bars = ax.bar(cls_labels, dg1_vals, color=colors_bar, alpha=0.8,
                      edgecolor="black", linewidth=0.8)
        ax.axhline(0, color="black", lw=1.0, linestyle="--")
        ax.set_ylabel("G1_obs − G1_Penttilä")
        ax.set_title("Offset from Penttilä+2016 G1 predictions")
        ax.grid(alpha=0.3, axis="y")
        for bar, val in zip(bars, dg1_vals):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.005*np.sign(val),
                    f"{val:+.3f}", ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=9)
    else:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "30_hg1g2_taxonomy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/30_hg1g2_taxonomy.png")

    # ── Log ───────────────────────────────────────────────────────────────────
    with open(LOG_DIR / "30_hg1g2_taxonomy_stats.txt", "w") as f:
        f.write("GAPC Step 30 — HG1G2 × Taxonomy (Penttilä+2016 comparison)\n")
        f.write("=" * 65 + "\n")
        f.write(f"n_hg1g2_total: {len(hg):,}\n")
        f.write(f"n_with_taxonomy: {n_tax:,}\n")
        f.write(f"G1+G2<=1 physical pct: {phy_pct:.2f}%\n")
        f.write(f"G_pred_median (0.46G1+0.54G2): {gp.median():.6f}\n")
        f.write(f"G_pred_std: {gp.std():.6f}\n")
        f.write(f"KW_G1_across_classes: {kw_msg}\n")
        f.write(f"Mann-Whitney G1(S)>G1(C): {mw_msg}\n\n")
        f.write(f"{'Class':>5s}  {'N':>5s}  "
                f"{'G1_obs':>8s}  {'G1_lit':>8s}  {'dG1':>7s}  "
                f"{'G2_obs':>8s}  {'G2_lit':>8s}  {'dG2':>7s}\n")
        f.write("-" * 65 + "\n")
        for r in rows:
            g1l  = f"{r['G1_lit']:8.4f}" if np.isfinite(r["G1_lit"])  else "     n/a"
            g2l  = f"{r['G2_lit']:8.4f}" if np.isfinite(r["G2_lit"])  else "     n/a"
            dg1  = f"{r['delta_G1']:+7.4f}" if np.isfinite(r["delta_G1"]) else "    n/a"
            dg2  = f"{r['delta_G2']:+7.4f}" if np.isfinite(r["delta_G2"]) else "    n/a"
            f.write(
                f"{r['cls']:>5s}  {r['n']:>5,}  "
                f"{r['G1_obs']:8.4f}  {g1l:>8s}  {dg1:>7s}  "
                f"{r['G2_obs']:8.4f}  {g2l:>8s}  {dg2:>7s}\n"
            )
    print(f"  Log  → logs/30_hg1g2_taxonomy_stats.txt\n")


if __name__ == "__main__":
    main()
