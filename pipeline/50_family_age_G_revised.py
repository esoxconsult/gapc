"""
50_family_age_G_revised.py
GAPC — Family age × G, revised with taxonomy_refined (v8) and proper elements.

Step 24b was limited by: (a) few proper-element matches, (b) no taxonomy breakdown.
This step uses:
  - Proper element family membership from data/interim/family_membership_proper.parquet
    (computed in step 24b using the Zappala 1994 velocity metric)
  - taxonomy_refined from v8 to separate S-type families (space weathering expected)
    from C-type families (less weathering signal expected)
  - Literature family ages (Nesvorny+2015, Spoto+2015) with uncertainties

Scientific question: does G decrease with family age for S-types?
  Young S-families (Flora 950 Myr, Vesta 1 Gyr) should have HIGHER G
  (fresher, less weathered surfaces) than old S-families (Koronis 2.9 Gyr).
  This would be a direct observational constraint on the space weathering timescale.

Outputs:
  plots/50_family_age_G_revised.png
  logs/50_family_age_G_revised_stats.txt
  (v8 NOT modified — read-only)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu

ROOT     = Path(__file__).resolve().parents[1]
V8_PATH  = ROOT / "data" / "final"   / "gapc_catalog_v8.parquet"
FAM_PATH = ROOT / "data" / "interim" / "family_membership_proper.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

# Family ages and taxonomy from Nesvorny+2015 / Spoto+2015 / Broz+2013
# (name, age_Myr, age_err_Myr, dominant_tax, color)
FAMILY_META = {
    "Veritas":   (8,    2,   "C", "#5c85d6"),
    "Vesta":     (1000, 500, "V", "#9c27b0"),
    "Flora":     (950,  300, "S", "#e07b39"),
    "Nysa":      (2000, 500, "S", "#f4a582"),
    "Eunomia":   (2500, 500, "S", "#d6604d"),
    "Hygiea":    (2000, 500, "C", "#74add1"),
    "Themis":    (2500, 500, "C", "#4393c3"),
    "Eos":       (1300, 200, "K", "#a6d96a"),
    "Koronis":   (2900, 500, "S", "#c0392b"),
}


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 50 — Family age × G (revised with taxonomy_refined)")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V8_PATH.exists():
        print(f"\n  ERROR: {V8_PATH} not found"); return
    if not FAM_PATH.exists():
        print(f"\n  ERROR: {FAM_PATH} not found — run step 24b first"); return

    gapc = pd.read_parquet(V8_PATH)
    fam  = pd.read_parquet(FAM_PATH)

    print(f"\n  v8 loaded:    {len(gapc):,} rows")
    print(f"  Family file:  {len(fam):,} rows")
    print(f"  Family file columns: {list(fam.columns)}")

    # ── Merge family membership ───────────────────────────────────────────────
    fam_col = next((c for c in fam.columns if "family" in c.lower()), None)
    if fam_col is None:
        print("  ERROR: no family column in family_membership file")
        # fallback: use gasp_family (GASP crossmatch only, fewer objects)
        gapc["family_gapc"] = gapc.get("gasp_family", pd.Series(np.nan, index=gapc.index))
    else:
        merge = gapc.merge(fam[["number_mp", fam_col]].drop_duplicates("number_mp"),
                           on="number_mp", how="left")
        merge = merge.rename(columns={fam_col: "family_gapc"})
        gapc = merge

    n_with_family = gapc["family_gapc"].notna().sum()
    print(f"\n  Objects with family assignment: {n_with_family:,}")
    fam_counts = gapc["family_gapc"].value_counts()
    print(f"  Top families: {fam_counts.head(8).to_dict()}")

    tax = "taxonomy_refined"

    # ── Per-family G statistics ───────────────────────────────────────────────
    print(f"\n  G by family and taxonomy_refined:")
    family_results = {}
    for fam_name, (age, age_err, dom_tax, col) in FAMILY_META.items():
        # match by name (flexible: exact or startswith)
        mask = gapc["family_gapc"].str.startswith(fam_name, na=False)
        sub  = gapc[mask & gapc["G"].notna()]
        if len(sub) < 10:
            print(f"    {fam_name:12s}: n={len(sub)} (too few)")
            continue
        g_med = sub["G"].median()
        g_std = sub["G"].std()
        # S-type only
        s_sub = sub[sub[tax].astype(str).str.startswith("S")]
        g_s   = s_sub["G"].median() if len(s_sub) >= 5 else np.nan
        # C-type only
        c_sub = sub[sub[tax].astype(str).str.startswith("C")]
        g_c   = c_sub["G"].median() if len(c_sub) >= 5 else np.nan

        print(f"    {fam_name:12s}  age={age:5d} Myr  "
              f"n={len(sub):5,}  G={g_med:.4f}  "
              f"G_S={g_s:.4f}  G_C={g_c:.4f}")
        family_results[fam_name] = dict(
            age=age, age_err=age_err, dom_tax=dom_tax, color=col,
            n=len(sub), G_med=g_med, G_std=g_std,
            G_S=g_s, G_C=g_c,
            n_S=len(s_sub), n_C=len(c_sub)
        )

    # ── Correlation: age × G ──────────────────────────────────────────────────
    fam_ages  = [family_results[f]["age"]   for f in family_results]
    fam_G_all = [family_results[f]["G_med"] for f in family_results]
    fam_G_S   = [family_results[f]["G_S"]   for f in family_results]
    fam_G_C   = [family_results[f]["G_C"]   for f in family_results]

    if len(fam_ages) >= 3:
        rho_all, p_all = spearmanr(fam_ages, fam_G_all)
        print(f"\n  rho(age, G) all families: {rho_all:+.4f}  p={p_all:.3e}  "
              f"n={len(fam_ages)}")
        # S-type families only
        s_fams = [(f, r) for f, r in family_results.items()
                  if not np.isnan(r["G_S"]) and r["dom_tax"] in ("S", "K")]
        if len(s_fams) >= 3:
            s_ages = [family_results[f]["age"] for f, _ in s_fams]
            s_Gs   = [family_results[f]["G_S"] for f, _ in s_fams]
            rho_S, p_S = spearmanr(s_ages, s_Gs)
            print(f"  rho(age, G_S) S-families: {rho_S:+.4f}  p={p_S:.3e}  "
                  f"n={len(s_fams)}")
        else:
            rho_S, p_S = np.nan, np.nan
    else:
        rho_all = p_all = rho_S = p_S = np.nan

    # ── Young vs old S-type families ─────────────────────────────────────────
    young_cut = 1200  # Myr
    young_fams = [f for f, r in family_results.items()
                  if r["age"] <= young_cut and r["dom_tax"] in ("S", "K")]
    old_fams   = [f for f, r in family_results.items()
                  if r["age"] > young_cut  and r["dom_tax"] in ("S", "K")]

    def get_G(fam_list):
        parts = []
        for f in fam_list:
            mask = gapc["family_gapc"].str.startswith(f, na=False)
            sub  = gapc[mask & gapc["G"].notna() &
                        gapc[tax].astype(str).str.startswith("S")]
            parts.append(sub["G"])
        return pd.concat(parts) if parts else pd.Series(dtype=float)

    g_young = get_G(young_fams)
    g_old   = get_G(old_fams)
    if len(g_young) >= 10 and len(g_old) >= 10:
        U, p_yo = mannwhitneyu(g_young, g_old, alternative="greater")
        print(f"\n  Young S-families (≤{young_cut} Myr): "
              f"G_med={g_young.median():.4f}  n={len(g_young):,}")
        print(f"  Old   S-families (>{young_cut} Myr): "
              f"G_med={g_old.median():.4f}  n={len(g_old):,}")
        print(f"  MW G(young > old): p={p_yo:.3e}")
    else:
        p_yo = np.nan

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Family age × G (space weathering timescale)", fontsize=12)

    # Left: G_S vs family age
    ax = axes[0]
    for fam_name, r in family_results.items():
        if np.isnan(r["G_S"]):
            continue
        ax.errorbar(r["age"] / 1000, r["G_S"],
                    xerr=r["age_err"] / 1000,
                    fmt="o", ms=8, color=r["color"],
                    capsize=4, lw=1.5, zorder=3)
        ax.text(r["age"] / 1000, r["G_S"] + 0.006, fam_name,
                fontsize=8, ha="center", va="bottom")
    ax.set_xlabel("Family age [Gyr]")
    ax.set_ylabel("Median G (S-type members)")
    ax.set_title(f"G vs age (S-type in each family)\n"
                 f"rho={rho_S:+.3f}  p={p_S:.2e}" if not np.isnan(rho_S)
                 else "G vs family age (S-types)")
    ax.grid(alpha=0.3)

    # Right: G distribution young vs old S-families
    ax = axes[1]
    if len(g_young) >= 10 and len(g_old) >= 10:
        g_rng = (min(g_young.min(), g_old.min()),
                 max(g_young.quantile(0.98), g_old.quantile(0.98)))
        bins_g = np.linspace(g_rng[0], g_rng[1], 40)
        ax.hist(g_young.clip(*g_rng).values, bins=bins_g, density=True,
                histtype="step", lw=2, color="#e07b39",
                label=f"Young ≤{young_cut} Myr  ({', '.join(young_fams)})\n"
                      f"n={len(g_young):,}  G_med={g_young.median():.4f}")
        ax.hist(g_old.clip(*g_rng).values, bins=bins_g, density=True,
                histtype="step", lw=2, color="#c0392b",
                label=f"Old >{young_cut} Myr  ({', '.join(old_fams)})\n"
                      f"n={len(g_old):,}  G_med={g_old.median():.4f}")
        ax.set_xlabel("G"); ax.set_ylabel("Density")
        ax.set_title(f"G distribution: young vs old S-families\n"
                     f"MW p(young>old) = {p_yo:.3e}"
                     if not np.isnan(p_yo) else "G: young vs old S-families")
        ax.legend(fontsize=8)
        ax.axvline(g_young.median(), color="#e07b39", lw=1.2, ls="--")
        ax.axvline(g_old.median(),   color="#c0392b", lw=1.2, ls="--")
    else:
        ax.text(0.5, 0.5, "Insufficient data for young/old comparison",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "50_family_age_G_revised.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/50_family_age_G_revised.png")

    with open(LOG_DIR / "50_family_age_G_revised_stats.txt", "w") as f:
        f.write("GAPC Step 50 — Family age × G (revised)\n")
        f.write("=" * 60 + "\n")
        for fam_name, r in family_results.items():
            f.write(f"{fam_name:12s}: age={r['age']:5d}±{r['age_err']:.0f} Myr  "
                    f"n={r['n']:5,}  G_med={r['G_med']:.4f}  "
                    f"G_S={r['G_S']:.4f}(n={r['n_S']})  "
                    f"G_C={r['G_C']:.4f}(n={r['n_C']})\n")
        f.write(f"\nrho(age, G_all): {rho_all:+.4f}  p={p_all:.3e}\n")
        if not np.isnan(rho_S):
            f.write(f"rho(age, G_S):   {rho_S:+.4f}  p={p_S:.3e}\n")
        if not np.isnan(p_yo):
            f.write(f"MW young vs old S: p={p_yo:.3e}\n"
                    f"  young: {young_fams}  G_med={g_young.median():.4f}\n"
                    f"  old:   {old_fams}   G_med={g_old.median():.4f}\n")
    print(f"  Log  → logs/50_family_age_G_revised_stats.txt\n")


if __name__ == "__main__":
    main()
