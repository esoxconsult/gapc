"""
54_heliocentric_gradient.py
GAPC — G × heliocentric distance: test for solar wind weathering gradient.

Physical hypotheses:
  H1 (solar wind dominates): solar wind flux ∝ 1/a² → inner belt more
     weathered → LOWER G at smaller a (for same size, same taxonomy)
  H2 (micrometeorite impacts dominate): impact rate higher in inner belt
     → more gardening / surface refreshing → HIGHER G at smaller a
  H3 (no gradient): neither mechanism creates a detectable a-dependence
     beyond what taxonomy and size already explain

We test rho(G, a | logD) for S-types and each major type separately,
and compare G distributions in inner / middle / outer belt bins.

Belt definitions (semi-major axis):
  Inner:  2.10 – 2.50 AU  (Flora, Vesta region)
  Middle: 2.50 – 2.82 AU  (Eunomia, Koronis region)
  Outer:  2.82 – 3.28 AU  (Themis, Hygiea region)

Inputs:
  data/final/gapc_catalog_v8.parquet
  data/interim/proper_elements.parquet

Outputs:
  plots/54_heliocentric_gradient.png
  logs/54_heliocentric_gradient_stats.txt
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, mannwhitneyu, kruskal
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
V8_PATH  = ROOT / "data" / "final"   / "gapc_catalog_v8.parquet"
PE_PATH  = ROOT / "data" / "interim" / "proper_elements.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

BELT_BINS = [
    (2.10, 2.50, "Inner",  "#e74c3c"),
    (2.50, 2.82, "Middle", "#e67e22"),
    (2.82, 3.28, "Outer",  "#2980b9"),
]
TYPES = ["S", "C", "M", "P"]


def partial_spearman(x, y, z):
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    bx = np.cov(xr, zr)[0, 1] / np.var(zr)
    by = np.cov(yr, zr)[0, 1] / np.var(zr)
    return pearsonr(xr - bx * zr, yr - by * zr)[0]


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 54 — Heliocentric gradient in G")
    print("=" * 65)

    gapc = pd.read_parquet(V8_PATH)
    pe   = pd.read_parquet(PE_PATH)

    # ── Merge proper elements ─────────────────────────────────────────────────
    num_col = next(c for c in pe.columns if c.lower() in
                   ("number_mp","num","number","asteroid"))
    a_col   = next(c for c in pe.columns if c.lower() in
                   ("a_p","ap","a_proper","proper_a"))
    pe = pe.rename(columns={num_col: "number_mp", a_col: "a_p"})

    df = gapc.merge(pe[["number_mp","a_p"]].drop_duplicates("number_mp"),
                    on="number_mp", how="left")

    base = df[df["G"].notna() & df["a_p"].notna() &
              df["D_km"].notna() & (df["D_km"] > 0) &
              df["a_p"].between(2.0, 3.4)].copy()
    base["logD"] = np.log10(base["D_km"])
    print(f"\n  Working sample (G + a_p + D, 2.0–3.4 AU): {len(base):,}")
    print(f"  a_p range: [{base['a_p'].min():.3f}, {base['a_p'].max():.3f}]")

    # ── 1. Global rho(G, a) ───────────────────────────────────────────────────
    rho_ga, p_ga = spearmanr(base["G"], base["a_p"])
    r_part_a  = partial_spearman(base["G"].values,
                                 base["a_p"].values,
                                 base["logD"].values)
    r_part_D2 = partial_spearman(base["G"].values,
                                 base["logD"].values,
                                 base["a_p"].values)
    print(f"\n  1. Global:")
    print(f"     rho(G, a) = {rho_ga:+.4f}  p={p_ga:.3e}")
    print(f"     r(G, a | logD) = {r_part_a:+.4f}")
    print(f"     r(G, logD | a) = {r_part_D2:+.4f}")

    # ── 2. Per taxonomy ───────────────────────────────────────────────────────
    print(f"\n  2. Partial r(G, a | logD) by taxonomy:")
    tax_results = {}
    for t in TYPES:
        sub = base[base["taxonomy_refined"] == t]
        if len(sub) < 100:
            continue
        rho_t, p_t = spearmanr(sub["G"], sub["a_p"])
        r_pt = partial_spearman(sub["G"].values,
                                sub["a_p"].values,
                                sub["logD"].values)
        print(f"    {t} (n={len(sub):,}): rho={rho_t:+.4f}  p={p_t:.2e}  "
              f"partial r={r_pt:+.4f}")
        tax_results[t] = dict(n=len(sub), rho=rho_t, p=p_t,
                               r_partial=r_pt, sub=sub)

    # ── 3. Belt bins: inner / middle / outer ──────────────────────────────────
    print(f"\n  3. G by belt region:")
    belt_data = {}
    for a0, a1, label, col in BELT_BINS:
        sub = base[(base["a_p"] >= a0) & (base["a_p"] < a1)]
        tax_dist = sub["taxonomy_refined"].value_counts().head(3).to_dict()
        print(f"    {label:7s} [{a0:.2f},{a1:.2f}) AU: n={len(sub):6,}  "
              f"G_med={sub['G'].median():.4f}  D_med={sub['D_km'].median():.1f}km  "
              f"tax={tax_dist}")
        belt_data[label] = dict(a0=a0, a1=a1, color=col,
                                n=len(sub), G=sub["G"], D=sub["D_km"],
                                sub=sub)

    # S-type only, belt comparison
    print(f"\n  3b. S-type only, by belt:")
    belt_S = {}
    for a0, a1, label, col in BELT_BINS:
        sub = base[(base["a_p"] >= a0) & (base["a_p"] < a1) &
                   (base["taxonomy_refined"] == "S")]
        if len(sub) < 20:
            continue
        print(f"    {label:7s}: n={len(sub):6,}  G_med={sub['G'].median():.4f}  "
              f"D_med={sub['D_km'].median():.1f} km")
        belt_S[label] = dict(a0=a0, a1=a1, color=col, n=len(sub),
                              G=sub["G"], D=sub["D_km"], sub=sub)

    # Kruskal-Wallis across belts (S-type)
    if len(belt_S) >= 3:
        kw_stat, kw_p = kruskal(*[v["G"] for v in belt_S.values()])
        print(f"\n     KW test (inner vs middle vs outer S-type): p={kw_p:.3e}")

    # MW inner vs outer S-type
    if "Inner" in belt_S and "Outer" in belt_S:
        U_io, p_io = mannwhitneyu(belt_S["Inner"]["G"],
                                  belt_S["Outer"]["G"],
                                  alternative="two-sided")
        print(f"     MW inner vs outer S-type: p={p_io:.3e}  "
              f"ΔG={belt_S['Inner']['G'].median()-belt_S['Outer']['G'].median():+.4f}")

    # ── 4. Size-controlled belt comparison ────────────────────────────────────
    print(f"\n  4. Size-controlled (rank residuals), S-type:")
    s_base = base[base["taxonomy_refined"] == "S"].copy()
    lD = s_base["logD"].values
    G  = s_base["G"].values
    Gr = rankdata(G); Dr = rankdata(lD)
    b  = np.cov(Gr, Dr)[0, 1] / np.var(Dr)
    s_base["G_resid"] = Gr - b * Dr

    belt_S_ctrl = {}
    for a0, a1, label, col in BELT_BINS:
        sub = s_base[(s_base["a_p"] >= a0) & (s_base["a_p"] < a1)]
        if len(sub) < 20:
            continue
        belt_S_ctrl[label] = dict(G_resid=sub["G_resid"], n=len(sub), color=col)
        print(f"    {label:7s}: n={len(sub):6,}  G_resid_mean={sub['G_resid'].mean():.2f}")

    if "Inner" in belt_S_ctrl and "Outer" in belt_S_ctrl:
        U2, p2 = mannwhitneyu(belt_S_ctrl["Inner"]["G_resid"],
                               belt_S_ctrl["Outer"]["G_resid"],
                               alternative="two-sided")
        print(f"     MW size-controlled inner vs outer: p={p2:.3e}")
    else:
        p2 = np.nan

    # ── 5. Narrow size bins: G vs a gradient ─────────────────────────────────
    print(f"\n  5. rho(G, a) within S-type size bins:")
    size_bins_a = [(1,3),(3,10),(10,30),(30,100)]
    for d0, d1 in size_bins_a:
        sub = base[(base["taxonomy_refined"] == "S") &
                   (base["D_km"] >= d0) & (base["D_km"] < d1)]
        if len(sub) < 50:
            continue
        rho_b, p_b = spearmanr(sub["G"], sub["a_p"])
        print(f"    D={d0:3d}-{d1:3d} km: n={len(sub):6,}  "
              f"rho(G,a)={rho_b:+.4f}  p={p_b:.2e}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("G × heliocentric distance (proper semi-major axis)\n"
                 "Test: solar wind vs micrometeorite weathering gradient",
                 fontsize=11)

    # G vs a scatter (S-type, subsampled)
    ax = axes[0, 0]
    s_smp = base[base["taxonomy_refined"] == "S"].sample(
        min(20000, (base["taxonomy_refined"] == "S").sum()), random_state=42)
    sc = ax.scatter(s_smp["a_p"], s_smp["G"], s=2, alpha=0.1,
                    c=np.log10(s_smp["D_km"]), cmap="viridis", rasterized=True)
    plt.colorbar(sc, ax=ax, label="log D [km]")
    for a0, a1, label, col in BELT_BINS:
        ax.axvline(a0, color="k", lw=0.7, ls="--", alpha=0.4)
    ax.axvline(BELT_BINS[-1][1], color="k", lw=0.7, ls="--", alpha=0.4)
    ax.set_xlabel("Proper semi-major axis a [AU]")
    ax.set_ylabel("G")
    ax.set_title(f"S-type G vs a  rho={tax_results.get('S',{}).get('rho',np.nan):+.3f}  "
                 f"partial r={tax_results.get('S',{}).get('r_partial',np.nan):+.3f}")
    ax.grid(alpha=0.2)
    for a0, a1, label, col in BELT_BINS:
        ax.text((a0+a1)/2, ax.get_ylim()[1]*0.95 if ax.get_ylim()[1] > 0 else 0.95,
                label, ha="center", fontsize=8, color=col)

    # G distributions by belt (S-type)
    ax = axes[0, 1]
    g_rng = (base[base["taxonomy_refined"]=="S"]["G"].quantile(0.01),
             base[base["taxonomy_refined"]=="S"]["G"].quantile(0.99))
    bins_g = np.linspace(*g_rng, 50)
    for label, res in belt_S.items():
        col = next(c for a0,a1,lb,c in BELT_BINS if lb==label)
        ax.hist(res["G"].clip(*g_rng), bins=bins_g, density=True,
                histtype="step", lw=2, color=col,
                label=f"{label} (n={res['n']:,}  G={res['G'].median():.3f})")
    ax.set_xlabel("G"); ax.set_ylabel("Density")
    ax.set_title("S-type G distribution by belt")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Median G vs belt bin (all types)
    ax = axes[1, 0]
    belt_labels = [lb for _, _, lb, _ in BELT_BINS]
    belt_a_mid  = [(a0+a1)/2 for a0, a1, _, _ in BELT_BINS]
    tax_colors2 = {"S":"#e07b39","C":"#5c85d6","M":"#8e44ad","P":"#27ae60"}
    for t in TYPES:
        meds = []
        for a0, a1, label, _ in BELT_BINS:
            sub = base[(base["a_p"] >= a0) & (base["a_p"] < a1) &
                       (base["taxonomy_refined"] == t)]
            meds.append(sub["G"].median() if len(sub) >= 20 else np.nan)
        if not all(np.isnan(meds)):
            ax.plot(belt_a_mid, meds, "o-", color=tax_colors2.get(t,"gray"),
                    label=t, lw=2, ms=8)
    ax.set_xlabel("Belt center [AU]")
    ax.set_ylabel("Median G")
    ax.set_title("Median G vs heliocentric zone by taxonomy")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Partial r by taxonomy
    ax = axes[1, 1]
    if tax_results:
        taxa_t = list(tax_results.keys())
        rhos_t = [tax_results[t]["rho"]       for t in taxa_t]
        rpar_t = [tax_results[t]["r_partial"]  for t in taxa_t]
        ns_t   = [tax_results[t]["n"]          for t in taxa_t]
        x = np.arange(len(taxa_t))
        w = 0.35
        ax.bar(x - w/2, rhos_t, w, label="rho(G, a)",
               color=[tax_colors2.get(t,"gray") for t in taxa_t], alpha=0.8)
        ax.bar(x + w/2, rpar_t, w, label="partial r(G, a | logD)",
               color=[tax_colors2.get(t,"gray") for t in taxa_t], alpha=0.4,
               hatch="//")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n(n={n:,})" for t, n in zip(taxa_t, ns_t)],
                           fontsize=9)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel("Correlation with a [AU]")
        ax.set_title("G–a correlation by taxonomy\n(solid=raw, hatch=size-controlled)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "54_heliocentric_gradient.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/54_heliocentric_gradient.png")

    # ── Log ───────────────────────────────────────────────────────────────────
    with open(LOG_DIR / "54_heliocentric_gradient_stats.txt", "w") as f:
        f.write("GAPC Step 54 — Heliocentric gradient in G\n")
        f.write("=" * 60 + "\n")
        f.write(f"n (2.0-3.4 AU, G+a+D): {len(base):,}\n\n")
        f.write(f"Global:\n")
        f.write(f"  rho(G, a) = {rho_ga:+.4f}  p={p_ga:.3e}\n")
        f.write(f"  r(G, a | logD) = {r_part_a:+.4f}\n")
        f.write(f"  r(G, logD | a) = {r_part_D2:+.4f}\n\n")
        f.write("By taxonomy:\n")
        for t, res in tax_results.items():
            f.write(f"  {t} (n={res['n']:,}): rho={res['rho']:+.4f}  "
                    f"p={res['p']:.3e}  partial_r={res['r_partial']:+.4f}\n")
        f.write("\nS-type by belt:\n")
        for label, res in belt_S.items():
            f.write(f"  {label:7s}: n={res['n']:,}  "
                    f"G_med={res['G'].median():.4f}  "
                    f"D_med={res['D'].median():.1f} km\n")
        if "Inner" in belt_S and "Outer" in belt_S:
            f.write(f"\nMW inner vs outer S-type: p={p_io:.3e}\n")
            f.write(f"MW size-controlled: p={p2:.3e}\n")
    print(f"  Log  → logs/54_heliocentric_gradient_stats.txt\n")


if __name__ == "__main__":
    main()
