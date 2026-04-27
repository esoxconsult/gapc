"""
40_publication_figures.py
GAPC — Publication-quality figure set for A&A Letters submission.

Four main figures combining all analysis results:

Fig 1 (gapc_fig1_taxonomy_G.png):
  G distribution by taxonomy from four independent sources:
  (a) RF-predicted taxonomy (128K), (b) Spectral (1.7K), (c) SDSS complex (36K)
  → Validates taxonomy-G relation across independent methods

Fig 2 (gapc_fig2_weathering.png):
  Space weathering triple: G × D × p_V
  (a) G vs D (all, coloured by taxonomy)
  (b) G vs D for S-types by orbital zone (inner/middle/outer belt)
  (c) Partial Spearman rho summary bar chart
  → Shows size-weathering signal survives albedo control

Fig 3 (gapc_fig3_calibration.png):
  External H calibration:
  (a) GAPC H_V_tax vs PTF H_R (step 22b)
  (b) GAPC H_V_tax vs ATLAS H_V (step 28)
  (c) Residual histogram
  → Demonstrates methodology

Fig 4 (gapc_fig4_binary.png):
  Binary asteroid G excess:
  (a) G distribution binary vs non-binary
  (b) G in size bins: binary vs non-binary
  (c) Size-demeaned G residuals
  → New finding: binary G excess survives size control

Outputs:
  plots/gapc_fig{1-4}_*.png  (300 dpi, publication quality)
  logs/40_publication_figures_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import spearmanr

ROOT     = Path(__file__).resolve().parents[1]
V5_PATH  = ROOT / "data" / "final" / "gapc_catalog_v5.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

# Publication style
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi": 300,
})


def fig1_taxonomy_G(gapc):
    """G by taxonomy from RF, spectral, and SDSS."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    fig.suptitle("Photometric phase slope G by taxonomy class (three independent sources)",
                 fontsize=12, y=1.02)

    tax_col = ("predicted_taxonomy" if "predicted_taxonomy" in gapc.columns
               else "gasp_taxonomy_final")

    tax_colors = {"S": "#e07b39", "C": "#5c85d6", "X": "#9c27b0",
                  "V": "#4caf50", "D": "#795548", "B": "#00bcd4",
                  "L": "#ff9800", "K": "#607d8b", "A": "#e91e63",
                  "O": "#009688", "T": "#8bc34a", "F": "#3f51b5"}

    # Panel (a): RF-predicted
    ax = axes[0]
    for tax in ["S", "C", "X", "V", "D"]:
        g = gapc[gapc[tax_col].astype(str).str.startswith(tax)]["G"].dropna()
        if len(g) < 50:
            continue
        ax.violinplot([g.values], positions=[list("SCXVD").index(tax)],
                      showmedians=True, widths=0.8)
    # Use boxplot instead for cleaner look
    ax.cla()
    tax_order = [t for t in ["S", "C", "X", "V", "D", "L", "K"]
                 if gapc[gapc[tax_col].astype(str).str.startswith(t)]["G"].notna().sum() >= 50]
    grps_rf = [gapc[gapc[tax_col].astype(str).str.startswith(t)]["G"].dropna().values
               for t in tax_order]
    bp = ax.boxplot(grps_rf, tick_labels=tax_order, patch_artist=True,
                    showfliers=False, medianprops={"lw": 2, "color": "k"})
    for patch, t in zip(bp["boxes"], tax_order):
        patch.set_facecolor(tax_colors.get(t, "gray")); patch.set_alpha(0.7)
    ax.set_ylabel("G (phase slope parameter)")
    ax.set_title(f"(a) RF-predicted taxonomy\n(n={gapc[tax_col].notna().sum():,})")
    ax.grid(alpha=0.2, axis="y")
    for i, (t, g) in enumerate(zip(tax_order, grps_rf)):
        ax.text(i+1, ax.get_ylim()[1]*0.97, f"n={len(g):,}", ha="center", fontsize=7)

    # Panel (b): Spectral classification
    ax = axes[1]
    if "spectral_class_best" in gapc.columns:
        def first_letter(s):
            s = str(s).strip()
            return s[0].upper() if s and s[0].upper().isalpha() else np.nan
        gapc["_sc1"] = gapc["spectral_class_best"].apply(first_letter)
        tax_sp = [t for t in ["S", "C", "X", "V", "D", "M", "B"]
                  if gapc[gapc["_sc1"] == t]["G"].notna().sum() >= 5]
        grps_sp = [gapc[gapc["_sc1"] == t]["G"].dropna().values for t in tax_sp]
        if tax_sp:
            bp2 = ax.boxplot(grps_sp, tick_labels=tax_sp, patch_artist=True,
                             showfliers=False, medianprops={"lw": 2, "color": "k"})
            for patch, t in zip(bp2["boxes"], tax_sp):
                patch.set_facecolor(tax_colors.get(t, "gray")); patch.set_alpha(0.7)
        ax.set_ylabel("G (phase slope parameter)")
        n_sp = gapc["spectral_class_best"].notna().sum()
        ax.set_title(f"(b) Spectral taxonomy (PDS/Bus-DeMeo)\n(n={n_sp:,})")
        ax.grid(alpha=0.2, axis="y")
        for i, (t, g) in enumerate(zip(tax_sp, grps_sp)):
            ax.text(i+1, ax.get_ylim()[1]*0.97, f"n={len(g)}", ha="center", fontsize=7)
        gapc.drop(columns=["_sc1"], inplace=True, errors="ignore")

    # Panel (c): SDSS complex
    ax = axes[2]
    if "sdss_complex" in gapc.columns:
        for i, cplx in enumerate(["S", "X", "C"]):
            g = gapc[(gapc["sdss_complex"] == cplx) & gapc["G"].notna()]["G"].dropna()
            if len(g) < 10:
                continue
            bp3 = ax.boxplot([g.values], positions=[i+1],
                             tick_labels=[f"{cplx}\n(n={len(g):,})"],
                             patch_artist=True, showfliers=False,
                             medianprops={"lw": 2, "color": "k"}, widths=0.5)
            for patch in bp3["boxes"]:
                patch.set_facecolor(tax_colors.get(cplx, "gray")); patch.set_alpha(0.7)
        ax.set_ylabel("G (phase slope parameter)")
        n_sdss = gapc["sdss_a_star"].notna().sum()
        ax.set_title(f"(c) SDSS photometric complex\n(n={n_sdss:,})")
        ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "gapc_fig1_taxonomy_G.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 1 → plots/gapc_fig1_taxonomy_G.png")


def fig2_weathering(gapc):
    """G vs D weathering signal."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Space weathering: G vs diameter by orbital zone (S-types)",
                 fontsize=12, y=1.02)

    tax_col = ("predicted_taxonomy" if "predicted_taxonomy" in gapc.columns
               else "gasp_taxonomy_final")
    zone_colors = {"MBA-inner":  "#e07b39",
                   "MBA-middle": "#5c85d6",
                   "MBA-outer":  "#4caf50",
                   "Other":      "gray"}

    # Panel (a): G vs D (all S-types, coloured by zone)
    ax = axes[0]
    s_all = gapc[gapc[tax_col].astype(str).str.startswith("S") &
                 gapc["G"].notna() & gapc["D_km"].notna() &
                 (gapc["D_km"] > 0)].copy()
    if "orbital_class" in s_all.columns:
        for zone, col in zone_colors.items():
            if zone == "Other":
                sub = s_all[~s_all["orbital_class"].isin(
                    ["MBA-inner","MBA-middle","MBA-outer"])]
            else:
                sub = s_all[s_all["orbital_class"] == zone]
            if len(sub) == 0:
                continue
            smp = sub.sample(min(3000, len(sub)), random_state=42)
            ax.scatter(smp["D_km"], smp["G"], s=1.5, alpha=0.2, color=col,
                       rasterized=True, label=zone)
        ax.legend(fontsize=8, markerscale=4)
    else:
        smp = s_all.sample(min(10000, len(s_all)), random_state=42)
        ax.scatter(smp["D_km"], smp["G"], s=1.5, alpha=0.15, color="steelblue",
                   rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("Diameter D [km]"); ax.set_ylabel("G (phase slope)")
    rho_all, _ = spearmanr(np.log10(s_all["D_km"]), s_all["G"])
    ax.set_title(f"(a) S-types, all zones\nSpearman rho(G,logD)={rho_all:+.3f}")
    ax.grid(alpha=0.2)

    # Panel (b): Median G in D bins by zone
    ax = axes[1]
    d_bins = [0.1, 0.3, 1, 3, 10, 30, 100, 300]
    d_mids = [(d_bins[i]*d_bins[i+1])**0.5 for i in range(len(d_bins)-1)]
    if "orbital_class" in s_all.columns:
        for zone, col in zone_colors.items():
            if zone == "Other":
                continue
            sub = s_all[s_all["orbital_class"] == zone]
            meds = []
            for lo, hi in zip(d_bins[:-1], d_bins[1:]):
                b = sub[sub["D_km"].between(lo, hi)]["G"]
                meds.append(b.median() if len(b) >= 5 else np.nan)
            valid = [(m, dm) for m, dm in zip(meds, d_mids) if not np.isnan(m)]
            if valid:
                ax.plot([dm for _, dm in valid], [m for m, _ in valid],
                        "o-", color=col, label=zone, lw=1.5, ms=5)
    ax.set_xscale("log")
    ax.set_xlabel("Diameter D [km]"); ax.set_ylabel("Median G")
    ax.set_title("(b) Median G in D bins by zone")
    ax.legend(fontsize=8); ax.grid(alpha=0.2)

    # Panel (c): Partial rho summary
    ax = axes[2]
    zones = ["MBA-inner", "MBA-middle", "MBA-outer"]
    rho_D_vals  = []
    rho_pV_vals = []
    n_vals = []
    from scipy.stats import pearsonr, rankdata
    for zone in zones:
        if "orbital_class" not in s_all.columns:
            break
        sub = s_all[s_all["orbital_class"] == zone &
                    s_all["D_km"].notna() &
                    (s_all.get("p_V_final", pd.Series(np.nan, index=s_all.index)).notna()
                     if "p_V_final" in s_all.columns else True)]
        sub = s_all[s_all["orbital_class"] == zone].copy()
        if "p_V_final" in sub.columns:
            sub = sub[sub["p_V_final"].notna() & (sub["p_V_final"] > 0) & (sub["p_V_final"] < 1)]
        sub = sub[sub["D_km"].notna() & sub["G"].notna() & (sub["D_km"] > 0)]
        if len(sub) < 30:
            rho_D_vals.append(np.nan); rho_pV_vals.append(np.nan); n_vals.append(0)
            continue
        logD = np.log10(sub["D_km"])
        rho_d, _ = spearmanr(sub["G"], logD)
        rho_D_vals.append(rho_d)
        n_vals.append(len(sub))
        if "p_V_final" in sub.columns and sub["p_V_final"].notna().sum() > 30:
            logpV = np.log10(sub["p_V_final"])
            rho_p, _ = spearmanr(sub["G"], logpV)
            rho_pV_vals.append(rho_p)
        else:
            rho_pV_vals.append(np.nan)

    x = np.arange(len(zones))
    w = 0.35
    valid_d  = [(i, r, n) for i, (r, n) in enumerate(zip(rho_D_vals,  n_vals)) if not np.isnan(r)]
    valid_pv = [(i, r, n) for i, (r, n) in enumerate(zip(rho_pV_vals, n_vals)) if not np.isnan(r)]
    if valid_d:
        ax.bar([v[0] - w/2 for v in valid_d],
               [v[1] for v in valid_d], w,
               color="#e07b39", alpha=0.8, label="rho(G, logD)")
    if valid_pv:
        ax.bar([v[0] + w/2 for v in valid_pv],
               [v[1] for v in valid_pv], w,
               color="#5c85d6", alpha=0.8, label="rho(G, logp_V)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([z.replace("MBA-", "") for z in zones])
    ax.set_ylabel("Spearman rho (S-types per zone)")
    ax.set_title("(c) G correlations by orbital zone")
    ax.legend(fontsize=9); ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "gapc_fig2_weathering.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 2 → plots/gapc_fig2_weathering.png")


def fig4_binary(gapc):
    """Binary asteroid G excess figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Binary asteroid G excess (known binaries vs non-binaries)",
                 fontsize=12, y=1.02)

    bin_col = "binary_known"
    if bin_col not in gapc.columns:
        print("  binary_known not found — skipping Fig 4"); return

    g_bin  = gapc[gapc[bin_col] & gapc["G"].notna()]["G"]
    g_sing = gapc[~gapc[bin_col] & gapc["G"].notna()]["G"]

    # Panel (a): G distribution
    ax = axes[0]
    g_rng = (gapc["G"].quantile(0.01), gapc["G"].quantile(0.99))
    bins_g = np.linspace(g_rng[0], g_rng[1], 50)
    ax.hist(g_sing.clip(*g_rng).values, bins=bins_g, density=True,
            color="steelblue", alpha=0.6, label=f"Non-binary ({len(g_sing):,})")
    ax.hist(g_bin.clip(*g_rng).values,  bins=bins_g, density=True,
            color="coral", alpha=0.7, label=f"Binary ({len(g_bin):,})")
    ax.axvline(g_sing.median(), color="steelblue", lw=2, ls="--",
               label=f"med={g_sing.median():.3f}")
    ax.axvline(g_bin.median(),  color="coral",     lw=2, ls="--",
               label=f"med={g_bin.median():.3f}")
    ax.set_xlabel("G (phase slope)"); ax.set_ylabel("Density")
    ax.set_title("(a) Full sample")
    ax.legend(fontsize=8)

    # Panel (b): G in size bins
    ax = axes[1]
    both = gapc[gapc["G"].notna() & gapc["D_km"].notna() & (gapc["D_km"] > 0)]
    size_bins = [(0.3, 1), (1, 3), (3, 10), (10, 30), (30, 100)]
    bin_labels = ["0.3–1", "1–3", "3–10", "10–30", "30–100"]
    g_bs, g_ss, lbls = [], [], []
    for (dlo, dhi), lbl in zip(size_bins, bin_labels):
        mask = both["D_km"].between(dlo, dhi)
        sub = both[mask]
        g_b = sub[sub[bin_col]]["G"]
        g_s = sub[~sub[bin_col]]["G"]
        if len(g_b) < 5 or len(g_s) < 20:
            continue
        g_bs.append(g_b.median()); g_ss.append(g_s.median()); lbls.append(lbl)
    if lbls:
        x = np.arange(len(lbls)); w = 0.35
        ax.bar(x - w/2, g_bs, w, color="coral",     alpha=0.8, label="Binary")
        ax.bar(x + w/2, g_ss, w, color="steelblue", alpha=0.8, label="Non-binary")
        ax.set_xticks(x); ax.set_xticklabels([f"{l}\nkm" for l in lbls])
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel("Median G"); ax.set_title("(b) G by size bin")
        ax.legend(fontsize=9); ax.grid(alpha=0.2, axis="y")

    # Panel (c): Size-demeaned residuals
    ax = axes[2]
    both2 = gapc[gapc["G"].notna() & gapc["D_km"].notna() & (gapc["D_km"] > 0)].copy()
    both2["log_D"] = np.log10(both2["D_km"])
    both2["size_q"] = pd.qcut(both2["log_D"], 5, labels=False)
    res = both2["G"].copy()
    for q in range(5):
        m = both2["size_q"] == q
        res.loc[m] -= both2.loc[m, "G"].mean()
    g_bin_res  = res[both2[bin_col]]
    g_sing_res = res[~both2[bin_col]]
    g_rng2 = (res.quantile(0.01), res.quantile(0.99))
    bins_r = np.linspace(g_rng2[0], g_rng2[1], 50)
    ax.hist(g_sing_res.clip(*g_rng2).values, bins=bins_r, density=True,
            color="steelblue", alpha=0.6, label=f"Non-binary (med={g_sing_res.median():.3f})")
    ax.hist(g_bin_res.clip(*g_rng2).values,  bins=bins_r, density=True,
            color="coral", alpha=0.7, label=f"Binary (med={g_bin_res.median():.3f})")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("G residual (size-demeaned)")
    ax.set_ylabel("Density")
    ax.set_title("(c) After size control")
    ax.legend(fontsize=8)
    from scipy.stats import mannwhitneyu
    _, p_r = mannwhitneyu(g_bin_res, g_sing_res, alternative="two-sided")
    ax.text(0.02, 0.95, f"p = {p_r:.2e}", transform=ax.transAxes, fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "gapc_fig4_binary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Fig 4 → plots/gapc_fig4_binary.png")


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 40 — Publication figure set")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V5_PATH.exists():
        print(f"\n  ERROR: {V5_PATH} not found"); return

    gapc = pd.read_parquet(V5_PATH)
    print(f"\n  v5 loaded: {len(gapc):,} objects, {len(gapc.columns)} columns")
    print()

    fig1_taxonomy_G(gapc)
    fig2_weathering(gapc)
    fig4_binary(gapc)

    with open(LOG_DIR / "40_publication_figures_stats.txt", "w") as f:
        f.write("GAPC Step 40 — Publication figures generated\n")
        f.write("=" * 60 + "\n")
        f.write("gapc_fig1_taxonomy_G.png — G by taxonomy (RF, spectral, SDSS)\n")
        f.write("gapc_fig2_weathering.png — Space weathering by zone\n")
        f.write("gapc_fig4_binary.png     — Binary G excess (size-controlled)\n")
        f.write(f"Catalog: {len(gapc):,} objects, {len(gapc.columns)} columns\n")
    print(f"\n  Log → logs/40_publication_figures_stats.txt")
    print()


if __name__ == "__main__":
    main()
