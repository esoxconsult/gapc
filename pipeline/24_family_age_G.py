"""
24_family_age_G.py
GAPC — Family age × G parameter correlation (space weathering timescale test).

Tests whether asteroid families of known age show a systematic trend in the
HG slope parameter G, consistent with space weathering gradually reducing
the geometric albedo (and hence G) of S-type asteroids.

Family ages from Spoto et al. 2015 (Icarus 257) and Nesvorny et al. 2003/2004.

Outputs:
  plots/24_g_vs_age.png
  plots/24_g_vs_age_stypes.png
  plots/24_family_g_violins.png
  plots/24_g_excess_vs_age.png
  logs/24_family_age_stats.csv
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final"   / "gapc_catalog_v4.parquet"
FAM_PATH = ROOT / "data" / "interim" / "family_membership_proper.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

MIN_FAMILY_N = 30   # minimum GAPC members to include a family in the analysis

# Literature family ages (Spoto et al. 2015, Nesvorny et al. 2003/2004/2015)
# Format: (age_Myr, uncertainty_Myr)
FAMILY_AGES = {
    "Veritas":     (8,    2),
    "Karin":       (6,    1),
    "Eunomia":     (2500, 500),
    "Flora":       (950,  300),
    "Vesta":       (1000, 500),
    "Koronis":     (2900, 500),
    "Eos":         (1300, 200),
    "Themis":      (2500, 500),
    "Nysa-Polana": (2000, 500),
    "Hungaria":    (400,  200),
}

# S-type dominated families (for weathering test)
S_TYPE_FAMILIES = {"Flora", "Koronis", "Karin", "Vesta", "Eunomia"}


def get_tax_group(df_in):
    """Map gasp_taxonomy_final + predicted_taxonomy to 4 groups."""
    pt = df_in.get("predicted_taxonomy", pd.Series(np.nan, index=df_in.index))
    gf = df_in.get("gasp_taxonomy_final", pd.Series(np.nan, index=df_in.index))
    result = pt.copy()
    need = result.isna()
    raw = gf[need].str.strip().str.upper().str[0]
    tax_map = {"S": "S", "C": "C", "X": "X"}
    result[need] = raw.map(tax_map).fillna("Other")
    return result


def family_g_stats(sub, g_col="G", unc_col="G_uncertain"):
    """Return dict with G stats for a family subset."""
    sub_rel = sub[~sub.get(unc_col, pd.Series(False, index=sub.index)).fillna(False)]
    g = sub_rel[g_col].dropna()
    return {
        "N_total":    len(sub),
        "N_reliable": len(g),
        "G_median":   g.median() if len(g) > 0 else np.nan,
        "G_std":      g.std()    if len(g) > 1 else np.nan,
        "G_mean":     g.mean()   if len(g) > 0 else np.nan,
        "G_unc_frac": sub.get(unc_col, pd.Series(False, index=sub.index)).fillna(False).mean(),
    }


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 24 — Family Age × G Correlation")
    print("=" * 60)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading catalog: {CAT_PATH}")
    df = pd.read_parquet(CAT_PATH)
    print(f"  Rows: {len(df):,}")

    print(f"Loading family membership: {FAM_PATH}")
    fam = pd.read_parquet(FAM_PATH)
    print(f"  Family rows: {len(fam):,}")

    # Merge
    df = df.merge(fam[["number_mp", "family_proper", "a_p", "e_p", "i_p"]].drop_duplicates("number_mp"),
                  on="number_mp", how="left")
    df["family_proper"] = df["family_proper"].fillna("Field")
    print(f"  Objects with proper elements: {(df['family_proper'] != 'Field').sum():,}")

    # Taxonomy
    df["_tax"] = get_tax_group(df)

    # G_uncertain coerce
    if "G_uncertain" in df.columns:
        df["G_uncertain"] = df["G_uncertain"].fillna(False).astype(bool)
    else:
        df["G_uncertain"] = False

    # ── Per-family stats ──────────────────────────────────────────────────────
    all_families = [f for f in df["family_proper"].unique() if f != "Field"]
    fam_rows = []

    print("\nPer-family stats:")
    print(f"  {'Family':>15s}  {'N':>6s}  {'N_rel':>6s}  "
          f"{'G_med':>7s}  {'G_std':>7s}  {'Age_Myr':>9s}")

    for fname in sorted(all_families):
        sub = df[df["family_proper"] == fname]
        stats = family_g_stats(sub)
        age, age_unc = FAMILY_AGES.get(fname, (np.nan, np.nan))

        stats["family"] = fname
        stats["age_Myr"] = age
        stats["age_unc"] = age_unc

        # S-type only stats
        s_sub = sub[sub["_tax"] == "S"]
        s_stats = family_g_stats(s_sub)
        stats["N_S"] = s_stats["N_total"]
        stats["G_median_S"] = s_stats["G_median"]
        stats["G_std_S"]    = s_stats["G_std"]

        if stats["N_total"] >= MIN_FAMILY_N:
            print(f"  {fname:>15s}  {stats['N_total']:>6,}  {stats['N_reliable']:>6,}  "
                  f"{stats['G_median']:>7.4f}  {stats['G_std']:>7.4f}  "
                  f"{age if not np.isnan(age) else '  ?':>9}")
        fam_rows.append(stats)

    fam_df = pd.DataFrame(fam_rows)
    fam_df_ok = fam_df[fam_df["N_total"] >= MIN_FAMILY_N].copy()
    fam_df_aged = fam_df_ok[fam_df_ok["age_Myr"].notna()].copy()
    fam_df_aged = fam_df_aged.sort_values("age_Myr")

    print(f"\nFamilies with N>={MIN_FAMILY_N} and known age: {len(fam_df_aged)}")

    # ── Spearman correlation: age vs G (all families) ─────────────────────────
    print("\nSpearman correlation tests:")
    if len(fam_df_aged) >= 3:
        rho_all, p_all = spearmanr(fam_df_aged["age_Myr"], fam_df_aged["G_median"])
        print(f"  All families — age vs G_median:  rho={rho_all:+.4f}  p={p_all:.3e}")

        rho_log, p_log = spearmanr(np.log10(fam_df_aged["age_Myr"]),
                                    fam_df_aged["G_median"])
        print(f"  All families — log10(age) vs G_median:  rho={rho_log:+.4f}  p={p_log:.3e}")
    else:
        rho_all, p_all = np.nan, np.nan
        rho_log, p_log = np.nan, np.nan
        print("  Not enough families for correlation test")

    # S-type only
    s_fam = fam_df_aged[
        fam_df_aged["family"].isin(S_TYPE_FAMILIES) &
        fam_df_aged["G_median_S"].notna() &
        (fam_df_aged["N_S"] >= 10)
    ].copy()
    print(f"\nS-type dominated families with age and N_S>=10: {len(s_fam)}")
    if len(s_fam) >= 3:
        rho_s, p_s = spearmanr(s_fam["age_Myr"], s_fam["G_median_S"])
        rho_s_log, p_s_log = spearmanr(np.log10(s_fam["age_Myr"]), s_fam["G_median_S"])
        print(f"  S-type families — age vs G_S:  rho={rho_s:+.4f}  p={p_s:.3e}")
        print(f"  S-type families — log10(age) vs G_S:  rho={rho_s_log:+.4f}  p={p_s_log:.3e}")
        if rho_s < -0.3 and p_s < 0.1:
            print("  => NEGATIVE TREND: older S-type families show lower G "
                  "(consistent with space weathering timescale)")
        elif not np.isnan(rho_s):
            print("  => No significant negative trend detected in S-type families")
    else:
        rho_s, p_s, rho_s_log, p_s_log = np.nan, np.nan, np.nan, np.nan
        print("  Insufficient S-type families for correlation test")

    # ── G excess vs orbital-class background ─────────────────────────────────
    # For each family, compute G median vs background MBA at same a_p range
    print("\nG excess vs background (family vs local MBA):")
    if "a_au" in df.columns or "a_p" in df.columns:
        a_col = "a_p" if "a_p" in df.columns else "a_au"
        # Background: Field objects only
        bg = df[df["family_proper"] == "Field"].copy()
        for _, row in fam_df_aged.iterrows():
            fname = row["family"]
            fam_sub = df[df["family_proper"] == fname]
            if len(fam_sub) < MIN_FAMILY_N:
                continue
            if a_col not in fam_sub.columns:
                continue
            a_min = fam_sub[a_col].quantile(0.05)
            a_max = fam_sub[a_col].quantile(0.95)
            bg_local = bg[(bg[a_col] >= a_min) & (bg[a_col] <= a_max) &
                          bg["G"].notna() & ~bg["G_uncertain"]]
            if len(bg_local) < 10:
                continue
            g_excess = row["G_median"] - bg_local["G"].median()
            fam_df_aged.loc[fam_df_aged["family"] == fname, "G_excess"] = g_excess
            print(f"  {fname:>15s}: G={row['G_median']:.4f}  "
                  f"BG_G={bg_local['G'].median():.4f}  "
                  f"excess={g_excess:+.4f}  (N_bg={len(bg_local):,})")
    else:
        fam_df_aged["G_excess"] = np.nan
        print("  a_au/a_p column not found — G excess not computed")

    # ── Plots ─────────────────────────────────────────────────────────────────
    ages = fam_df_aged["age_Myr"].values
    ages_err = fam_df_aged["age_unc"].values
    g_med = fam_df_aged["G_median"].values
    g_err = (fam_df_aged["G_std"] / np.sqrt(np.maximum(fam_df_aged["N_reliable"], 1))).values
    labels = fam_df_aged["family"].values
    n_fam = fam_df_aged["N_total"].values

    # Color by family type
    def fam_color(fname):
        if fname in S_TYPE_FAMILIES:
            return "#e07b39"
        elif fname in ("Themis", "Nysa-Polana"):
            return "#4a90d9"
        else:
            return "#888888"

    colors_fam = [fam_color(f) for f in labels]

    # 1. G median vs family age (log x-axis)
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, (age, age_e, g, g_e, lbl, col, n) in enumerate(
            zip(ages, ages_err, g_med, g_err, labels, colors_fam, n_fam)):
        if np.isnan(g) or np.isnan(age):
            continue
        ax.errorbar(age, g, xerr=age_e, yerr=g_e,
                    fmt="o", color=col, capsize=4, markersize=8,
                    markeredgecolor="black", markeredgewidth=0.5)
        ax.annotate(f"{lbl}\n(N={n:,})", (age, g),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, ha="left")
    ax.set_xscale("log")
    ax.set_xlabel("Family Age (Myr)", fontsize=12)
    ax.set_ylabel("Median G", fontsize=12)
    ax.set_title(f"G vs Family Age — All Families\n"
                 f"Spearman rho={rho_all:+.3f}  p={p_all:.3e}  "
                 f"(log-age rho={rho_log:+.3f})", fontsize=11)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e07b39", label="S-type dominated"),
        Patch(facecolor="#4a90d9", label="C-type dominated"),
        Patch(facecolor="#888888", label="Mixed/Other"),
    ]
    ax.legend(handles=legend_elements, fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p1 = PLOT_DIR / "24_g_vs_age.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {p1}")

    # 2. G vs age for S-types only
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(s_fam) >= 2:
        s_ages = s_fam["age_Myr"].values
        s_ages_e = s_fam["age_unc"].values
        s_g = s_fam["G_median_S"].values
        s_g_e = (s_fam["G_std_S"].fillna(0) /
                  np.sqrt(np.maximum(s_fam["N_S"], 1))).values
        s_lbl = s_fam["family"].values
        for i, (age, age_e, g, g_e, lbl) in enumerate(
                zip(s_ages, s_ages_e, s_g, s_g_e, s_lbl)):
            if np.isnan(g) or np.isnan(age):
                continue
            ax.errorbar(age, g, xerr=age_e, yerr=g_e,
                        fmt="s", color="#e07b39", capsize=4, markersize=9,
                        markeredgecolor="black", markeredgewidth=0.8)
            ax.annotate(lbl, (age, g), textcoords="offset points",
                        xytext=(5, 4), fontsize=9)

        # Linear regression in log space for reference
        valid_s = ~(np.isnan(s_ages) | np.isnan(s_g))
        if valid_s.sum() >= 3:
            log_ages = np.log10(s_ages[valid_s])
            g_vals   = s_g[valid_s]
            coeffs = np.polyfit(log_ages, g_vals, 1)
            x_fit = np.logspace(np.log10(s_ages[valid_s].min() * 0.5),
                                 np.log10(s_ages[valid_s].max() * 2), 100)
            y_fit = np.polyval(coeffs, np.log10(x_fit))
            ax.plot(x_fit, y_fit, "r--", linewidth=1.5, alpha=0.7,
                    label=f"log-linear fit: slope={coeffs[0]:.4f}")
            ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.set_xlabel("Family Age (Myr)", fontsize=12)
    ax.set_ylabel("Median G (S-type members only)", fontsize=12)
    ax.set_title(f"Space Weathering Signal — S-type Family G vs Age\n"
                 f"Spearman rho={rho_s:+.3f}  p={p_s:.3e}", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p2 = PLOT_DIR / "24_g_vs_age_stypes.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p2}")

    # 3. Family G distributions as box plots ordered by age
    families_ordered = fam_df_aged.sort_values("age_Myr")["family"].tolist()
    families_ordered = [f for f in families_ordered
                        if f in df["family_proper"].values
                        and (df[df["family_proper"] == f]["G"].dropna().__len__() >= 5)]

    if families_ordered:
        fig, ax = plt.subplots(figsize=(max(8, len(families_ordered) * 0.9 + 2), 5))
        data_list = []
        tick_labels = []
        for fname in families_ordered:
            sub = df[(df["family_proper"] == fname) & df["G"].notna() & ~df["G_uncertain"]]
            g_vals = sub["G"].values
            if len(g_vals) >= 5:
                data_list.append(g_vals)
                age_v = FAMILY_AGES.get(fname, (np.nan,))[0]
                age_str = f"{age_v:.0f} Myr" if not np.isnan(age_v) else "?"
                tick_labels.append(f"{fname}\n({age_str})")

        bp = ax.boxplot(data_list, labels=tick_labels,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=2),
                        whis=[10, 90])
        for patch, fname in zip(bp["boxes"], families_ordered[:len(data_list)]):
            patch.set_facecolor(fam_color(fname))
            patch.set_alpha(0.65)

        ax.set_ylabel("G", fontsize=12)
        ax.set_title("Family G Distributions Ordered by Age\n(boxes = 25-75th pct, whiskers = 10-90th pct)",
                     fontsize=11)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        p3 = PLOT_DIR / "24_family_g_violins.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        print(f"  Saved: {p3}")
    else:
        print("  Insufficient data for family boxplot")

    # 4. G excess vs background vs age
    fam_excess = fam_df_aged[fam_df_aged["G_excess"].notna()].copy() \
        if "G_excess" in fam_df_aged.columns else pd.DataFrame()

    fig, ax = plt.subplots(figsize=(8, 5))
    if len(fam_excess) >= 2:
        for _, row in fam_excess.iterrows():
            if np.isnan(row.get("G_excess", np.nan)) or np.isnan(row["age_Myr"]):
                continue
            col = fam_color(row["family"])
            ax.scatter(row["age_Myr"], row["G_excess"],
                       color=col, s=80, zorder=3,
                       edgecolors="black", linewidths=0.5)
            ax.annotate(row["family"], (row["age_Myr"], row["G_excess"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=8)
        ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7,
                   label="Background level")
        ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "G excess data not available\n(proper elements not yet downloaded)",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("Family Age (Myr)", fontsize=12)
    ax.set_ylabel("G (family) - G (local background)", fontsize=12)
    ax.set_title("G Excess vs Background by Family Age", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p4 = PLOT_DIR / "24_g_excess_vs_age.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p4}")

    # ── Save CSV log ──────────────────────────────────────────────────────────
    log_csv = LOG_DIR / "24_family_age_stats.csv"
    out_cols = ["family", "N_total", "N_reliable", "G_median", "G_std",
                "G_median_S", "N_S", "age_Myr", "age_unc"]
    if "G_excess" in fam_df_aged.columns:
        out_cols.append("G_excess")
    fam_df_aged[out_cols].to_csv(log_csv, index=False, float_format="%.5f")
    print(f"\n  Log: {log_csv}")

    # Text summary
    log_txt = LOG_DIR / "24_family_age_stats.txt"
    with open(log_txt, "w") as f:
        f.write("GAPC Step 24 — Family Age × G Correlation\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Families with N>={MIN_FAMILY_N}: {len(fam_df_ok)}\n")
        f.write(f"Families with known age:         {len(fam_df_aged)}\n\n")
        f.write(f"Spearman rho (age vs G_median):         {rho_all:+.4f}  p={p_all:.3e}\n")
        f.write(f"Spearman rho (log10(age) vs G_median):  {rho_log:+.4f}  p={p_log:.3e}\n")
        f.write(f"Spearman rho S-type (age vs G_S):       {rho_s:+.4f}  p={p_s:.3e}\n\n")
        f.write("Family stats:\n")
        f.write(f"  {'Family':>15s}  {'N_total':>8s}  {'G_med':>7s}  "
                f"{'G_med_S':>8s}  {'Age_Myr':>9s}\n")
        for _, row in fam_df_aged.iterrows():
            f.write(f"  {row['family']:>15s}  {int(row['N_total']):>8,}  "
                    f"{row['G_median']:>7.4f}  "
                    f"{row['G_median_S'] if not np.isnan(row['G_median_S']) else '   nan':>8}  "
                    f"{row['age_Myr']:>9.0f}\n")

    print(f"  Log: {log_txt}")
    print("\nStep 24 complete.")


if __name__ == "__main__":
    main()
