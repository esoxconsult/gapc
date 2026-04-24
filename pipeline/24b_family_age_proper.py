"""
24b_family_age_proper.py
GAPC — Family age × G using proper orbital elements.

Replaces step 24 (which used rectangular boxes in osculating element space).
Proper elements (from AstDys all.syn, now in data/interim/proper_elements.parquet)
give accurate, drift-free family membership via the velocity-equivalent metric.

Method (Zappala et al. 1994):
  d_v = n*a * sqrt( (da/a)^2 + c_e*(de)^2 + c_i*(d_sinI)^2 )

  where:
    n  = mean motion ≈ 2π/sqrt(a^3) [rad/yr]  (Kepler, a in AU)
    a  = family center semi-major axis
    c_e = c_i = (2/3)^2 / (1 - e_c^2)^2  (simplified)

  Objects within d_v < v_cut are considered family members.
  We use published v_cut values from Nesvorny+2015 / Spoto+2015.

Family centers: proper elements of the largest/defining body in each family
  (from AstDys proper element file, looked up by asteroid number).

Outputs:
  data/interim/family_membership_proper.parquet
  plots/24b_family_age_proper.png
  logs/24b_family_age_proper_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

ROOT      = Path(__file__).resolve().parents[1]
CAT_PATH  = ROOT / "data" / "final"   / "gapc_catalog_v4.parquet"
PROP_PATH = ROOT / "data" / "interim" / "proper_elements.parquet"
PLOT_DIR  = ROOT / "plots"
LOG_DIR   = ROOT / "logs"
DATA_INT  = ROOT / "data" / "interim"

# ── Family definitions ────────────────────────────────────────────────────────
# Center proper elements (a_p AU, e_p, sin_I_p) and v_cut (m/s)
# Sources: Nesvorny+2015 Table 1, Spoto+2015, Broz+2013, Milani+2014
# Center: asteroid number of the defining body
FAMILIES = {
    # name:  (center_num, a_p,    e_p,    sinI_p, v_cut_ms, age_Myr, age_unc_Myr, tax)
    # Proper element centers from AstDys all.syn (Knezevic 2024).
    # v_cut at 150 m/s captures "extended family region"; Veritas tight (young).
    # Note: Flora poorly recovered with this metric (dispersed in proper space).
    "Koronis": (158,   2.8688, 0.0452, 0.0375, 150,  2900, 500, "S"),
    "Eos":     (221,   3.0124, 0.0726, 0.1712, 150,  1300, 200, "K"),
    "Themis":  (24,    3.1345, 0.1528, 0.0189, 150,  2500, 500, "C"),
    "Flora":   (8,     2.2014, 0.1449, 0.0971, 150,   950, 300, "S"),
    "Eunomia": (15,    2.6437, 0.1486, 0.2266, 150,  2500, 500, "S"),
    "Vesta":   (4,     2.3615, 0.0988, 0.1113, 150,  1000, 500, "V"),
    "Hygiea":  (10,    3.1418, 0.1356, 0.0890, 150,  2000, 500, "C"),
    "Veritas":  (490,  3.1740, 0.0656, 0.1593,  60,     8,   2, "C"),
    "Nysa":    (44,    2.4227, 0.1740, 0.0534, 150,  2000, 500, "S/E"),
}

TWO_THIRDS_SQ = (2.0 / 3.0) ** 2


def velocity_metric(a1, e1, si1, a_c, e_c, si_c):
    """
    Velocity-equivalent distance (m/s) between object and family center.
    Zappala+1994 metric: d_v = n_c*a_c * sqrt(5/4*(da/a)^2 + 2*de^2 + 2*dsinI^2)
    n_c*a_c = orbital velocity = sqrt(GM/a) ≈ 2π/sqrt(a) AU/yr * 4740 m/s per AU/yr
    """
    v_orb = 2 * np.pi / np.sqrt(a_c) * 4740.0   # orbital velocity in m/s (a in AU)
    d_v = v_orb * np.sqrt(
        1.25 * ((a1 - a_c) / a_c) ** 2 +
        2.0  * (e1 - e_c) ** 2 +
        2.0  * (si1 - si_c) ** 2
    )
    return d_v


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 24b — Family age × G (proper elements)")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_INT.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    gapc = pd.read_parquet(CAT_PATH)
    if not PROP_PATH.exists():
        print(f"\n  ERROR: {PROP_PATH} not found.")
        print("  Run 23b_proper_elements_download.py first, or place all.syn in data/raw/")
        return

    prop = pd.read_parquet(PROP_PATH)
    print(f"  GAPC catalog: {len(gapc):,}  proper elements: {len(prop):,}")

    # Merge
    merged = gapc.merge(prop, on="number_mp", how="inner")
    print(f"  Objects with proper elements: {len(merged):,}")

    # Quality filter
    if "fit_ok" in merged.columns:
        merged = merged[merged["fit_ok"]]
    if "G_uncertain" in merged.columns:
        merged = merged[~merged["G_uncertain"].fillna(False)]
    merged = merged[merged["G"].notna()]
    print(f"  After quality filter: {len(merged):,}")

    # Taxonomy
    if "predicted_taxonomy" in merged.columns:
        merged["_tax"] = merged["predicted_taxonomy"].fillna("Other")
    else:
        merged["_tax"] = "Other"
    if "gasp_taxonomy_final" in merged.columns:
        mask = merged["_tax"] == "Other"
        raw = merged.loc[mask, "gasp_taxonomy_final"].str.strip().str.upper().str[0]
        merged.loc[mask, "_tax"] = raw.map({"S":"S","C":"C","X":"X"}).fillna("Other")

    # ── Assign family membership ───────────────────────────────────────────────
    print(f"\n  Assigning family membership via velocity metric …")
    merged["family_name"] = "field"
    merged["family_dv"]   = np.inf

    fam_records = []
    for fam_name, (cnum, a_c, e_c, si_c, v_cut, age, age_unc, fam_tax) in FAMILIES.items():
        dv = velocity_metric(
            merged["a_proper"].values,
            merged["e_proper"].values,
            merged["sinI_proper"].values,
            a_c, e_c, si_c
        )
        in_fam = dv < v_cut
        # Only assign if closer than current best (handle overlaps)
        better = in_fam & (dv < merged["family_dv"])
        merged.loc[better, "family_name"] = fam_name
        merged.loc[better, "family_dv"]   = dv[better]
        fam_records.append(dict(family=fam_name, age_Myr=age,
                                age_unc=age_unc, tax=fam_tax, v_cut=v_cut))

    # Print exclusive membership counts
    print()
    for fam_name in FAMILIES:
        n_exc = (merged["family_name"] == fam_name).sum()
        v_cut = FAMILIES[fam_name][4]
        print(f"  {fam_name:12s}  v_cut={v_cut:3d} m/s  exclusive members={n_exc:5,}")
    print(f"\n  Field (no family): {(merged['family_name']=='field').sum():,}")

    # ── Save membership ────────────────────────────────────────────────────────
    mem_path = DATA_INT / "family_membership_proper.parquet"
    merged[["number_mp", "family_name", "family_dv"]].to_parquet(mem_path, index=False)
    print(f"  Membership saved → {mem_path.relative_to(ROOT)}")

    # ── G vs family age ────────────────────────────────────────────────────────
    print(f"\n  G statistics per family:")
    age_g_rows = []
    for fam_name, (cnum, a_c, e_c, si_c, v_cut, age, age_unc, fam_tax) in FAMILIES.items():
        sub = merged[merged["family_name"] == fam_name]
        if len(sub) < 15:
            continue
        g = sub["G"].dropna()
        s_sub = sub[sub["_tax"] == "S"]
        g_s = s_sub["G"].dropna()
        print(f"  {fam_name:12s}  n={len(sub):5,}  G_med={g.median():.4f}±{g.std():.4f}  "
              f"S-type n={len(s_sub):,}")
        age_g_rows.append(dict(
            family=fam_name, age_Myr=age, age_unc=age_unc,
            G_median=g.median(), G_std=g.std(), G_sem=g.std()/np.sqrt(len(g)),
            n=len(sub), n_S=len(s_sub),
            G_S_median=g_s.median() if len(g_s) > 5 else np.nan,
            G_S_sem=g_s.std()/np.sqrt(max(len(g_s),1)) if len(g_s) > 5 else np.nan,
        ))

    if not age_g_rows:
        print("\n  No families with sufficient members — check v_cut or data availability.")
        rho_all = rho_s = p_all = p_s = np.nan
        res = pd.DataFrame(columns=["family","age_Myr","age_unc","G_median","G_std",
                                     "G_sem","n","n_S","G_S_median","G_S_sem"])
    else:
        res = pd.DataFrame(age_g_rows).dropna(subset=["G_median", "age_Myr"])

    # Spearman correlation age vs G
    if len(res) >= 4:
        rho_all, p_all = spearmanr(res["age_Myr"], res["G_median"])
        print(f"\n  Spearman rho(age, G) all tax:  rho={rho_all:+.4f}  p={p_all:.3e}  n={len(res)}")
        res_s = res.dropna(subset=["G_S_median"])
        if len(res_s) >= 4:
            rho_s, p_s = spearmanr(res_s["age_Myr"], res_s["G_S_median"])
            print(f"  Spearman rho(age, G) S-types:  rho={rho_s:+.4f}  p={p_s:.3e}  n={len(res_s)}")
        else:
            rho_s, p_s = np.nan, np.nan
            print(f"  S-type: too few families ({len(res_s)}) for Spearman test")
    else:
        rho_all, p_all, rho_s, p_s = np.nan, np.nan, np.nan, np.nan
        print("  Too few families for correlation test.")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("G vs Family Age (proper element membership)", fontsize=13)

    # Panel A: G median vs age, all taxonomy
    ax = axes[0, 0]
    if len(res) > 0:
        ax.errorbar(res["age_Myr"] / 1000, res["G_median"],
                    xerr=res["age_unc"] / 1000,
                    yerr=res["G_sem"],
                    fmt="o", ms=7, capsize=4, color="steelblue", linewidth=1.5)
        for _, row in res.iterrows():
            ax.annotate(row["family"], (row["age_Myr"]/1000, row["G_median"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
    if np.isfinite(rho_all):
        ax.set_title(f"All taxonomy  rho={rho_all:+.3f}  p={p_all:.3e}")
    else:
        ax.set_title("All taxonomy (insufficient data)")
    ax.set_xlabel("Family age [Gyr]"); ax.set_ylabel("Median G")
    ax.grid(alpha=0.3)

    # Panel B: G median vs age, S-types only
    ax = axes[0, 1]
    res_s2 = res.dropna(subset=["G_S_median"])
    if len(res_s2) > 0:
        ax.errorbar(res_s2["age_Myr"] / 1000, res_s2["G_S_median"],
                    xerr=res_s2["age_unc"] / 1000,
                    yerr=res_s2["G_S_sem"],
                    fmt="s", ms=7, capsize=4, color="#e07b39", linewidth=1.5)
        for _, row in res_s2.iterrows():
            ax.annotate(row["family"], (row["age_Myr"]/1000, row["G_S_median"]),
                        textcoords="offset points", xytext=(4, 4), fontsize=8)
    if np.isfinite(rho_s):
        ax.set_title(f"S-type only  rho={rho_s:+.3f}  p={p_s:.3e}")
    else:
        ax.set_title("S-type only (insufficient data)")
    ax.set_xlabel("Family age [Gyr]"); ax.set_ylabel("Median G (S-type members)")
    ax.grid(alpha=0.3)

    # Panel C: family membership in proper element space
    ax = axes[1, 0]
    field = merged[merged["family_name"] == "field"]
    ax.scatter(field["a_proper"], field["e_proper"],
               s=1, alpha=0.05, color="lightgray", rasterized=True)
    cmap = plt.cm.Set1
    for i, (fam_name, _) in enumerate(FAMILIES.items()):
        sub = merged[merged["family_name"] == fam_name]
        if len(sub) < 5:
            continue
        ax.scatter(sub["a_proper"], sub["e_proper"],
                   s=3, alpha=0.4, color=cmap(i % 9), label=fam_name, rasterized=True)
    ax.set_xlabel("a_proper [AU]"); ax.set_ylabel("e_proper")
    ax.set_title("Family membership in proper element space")
    ax.legend(fontsize=6, ncol=2); ax.grid(alpha=0.3)

    # Panel D: G distribution per family (boxplot)
    ax = axes[1, 1]
    fam_names_sorted = sorted(res["family"].tolist(), key=lambda f: res.loc[res["family"]==f, "age_Myr"].values[0])
    data_boxes = []
    labels_box = []
    for fam in fam_names_sorted:
        sub = merged[(merged["family_name"] == fam) & merged["G"].notna()]
        if len(sub) >= 5:
            data_boxes.append(sub["G"].dropna().values)
            age_v = res.loc[res["family"] == fam, "age_Myr"].values[0]
            labels_box.append(f"{fam}\n({age_v/1000:.1f}Ga)")
    if data_boxes:
        bp = ax.boxplot(data_boxes, tick_labels=labels_box, patch_artist=True,
                        medianprops=dict(color="red", lw=2))
        for patch, color in zip(bp["boxes"], plt.cm.Set1(np.linspace(0, 1, len(data_boxes)))):
            patch.set_facecolor(color); patch.set_alpha(0.6)
    ax.set_ylabel("G parameter")
    ax.set_title("G distribution by family age")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = PLOT_DIR / "24b_family_age_proper.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → {out.relative_to(ROOT)}")

    # ── Log ───────────────────────────────────────────────────────────────────
    log_path = LOG_DIR / "24b_family_age_proper_stats.txt"
    with open(log_path, "w") as f:
        f.write("GAPC Step 24b — Family age × G (proper elements)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Proper element coverage: {len(prop):,} objects\n")
        f.write(f"GAPC merged: {len(merged):,}\n\n")
        f.write(f"Spearman rho(age, G) all tax: {rho_all:+.4f}  p={p_all:.3e}\n")
        f.write(f"Spearman rho(age, G) S-only:  {rho_s:+.4f}  p={p_s:.3e}\n\n")
        f.write(f"{'Family':12s}  {'N':6s}  {'G_med':7s}  {'G_std':7s}  {'Age_Gyr':8s}\n")
        for _, row in res.iterrows():
            f.write(f"{row['family']:12s}  {int(row['n']):6,}  {row['G_median']:7.4f}  "
                    f"{row['G_std']:7.4f}  {row['age_Myr']/1000:8.2f}\n")
    print(f"  Log  → {log_path.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
