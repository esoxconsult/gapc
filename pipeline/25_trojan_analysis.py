"""
25_trojan_analysis.py
GAPC — Jupiter Trojan asteroid characterization (Lucy mission context).

Jupiter Trojans are primitive, dark (D/P/C-type) asteroids sharing Jupiter's
orbit at L4 (leading) and L5 (trailing) Lagrange points. NASA's Lucy mission
(launched 2021) will fly by seven Trojan targets 2027–2033.

Analysis:
  1. Basic Trojan statistics in GAPC
  2. G distribution: Trojans vs outer MBA (dark vs dark comparison)
  3. Variability (chi2_reduced, var_flag) — Trojans are known rotators
  4. Size distribution (D_km)
  5. Taxonomy (expect D/P dominated)
  6. Lucy mission targets in catalog

Outputs:
  plots/25_trojan_g_distribution.png
  plots/25_trojan_h_distribution.png
  plots/25_trojan_diameter.png
  plots/25_trojan_chi2.png
  plots/25_lucy_targets.png  (if any found)
  logs/25_trojan_stats.txt
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from scipy.stats import ks_2samp, mannwhitneyu
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final"   / "gapc_catalog_v4.parquet"
PE_PATH  = ROOT / "data" / "raw"     / "proper_elements.parquet"
OC_PATH  = ROOT / "data" / "interim" / "mpcorb_orbital_class.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

# Lucy mission primary targets (number_mp, name)
LUCY_TARGETS = {
    3548:  "Eurybates",
    15094: "Polymele",
    11351: "Leucus",
    21900: "Orus",
    617:   "Patroclus",
    # Donaldjohanson (MBA flyby, not Trojan): 52246
}

# Trojan a_au range from osculating elements (step 15)
TROJAN_A_MIN = 5.05
TROJAN_A_MAX = 5.40

# Outer MBA for comparison
MBA_OUTER_A_MIN = 2.82
MBA_OUTER_A_MAX = 3.27


def get_tax_group(df_in):
    """Map taxonomy to 4 groups: S/C/X/Other."""
    pt = df_in.get("predicted_taxonomy", pd.Series(np.nan, index=df_in.index))
    gf = df_in.get("gasp_taxonomy_final", pd.Series(np.nan, index=df_in.index))
    result = pt.copy()
    need = result.isna()
    raw = gf[need].str.strip().str.upper().str[0]
    tax_map = {"S": "S", "C": "C", "X": "X"}
    result[need] = raw.map(tax_map).fillna("Other")
    return result


def kde_plot(ax, data, label, color, bw_method="scott"):
    """Plot KDE + histogram for a 1D dataset."""
    data = data[np.isfinite(data)]
    if len(data) < 5:
        return
    ax.hist(data, bins=30, density=True, alpha=0.3, color=color, label=label)
    if HAS_SCIPY and len(data) >= 10:
        try:
            kde = gaussian_kde(data, bw_method=bw_method)
            x = np.linspace(data.min() - 0.05, data.max() + 0.05, 300)
            ax.plot(x, kde(x), color=color, linewidth=2)
        except Exception:
            pass


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 25 — Trojan Asteroid Analysis")
    print("=" * 60)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading catalog: {CAT_PATH}")
    df = pd.read_parquet(CAT_PATH)
    print(f"  Rows: {len(df):,}")

    # Merge orbital elements for a_au if not already present
    a_col = None
    if "a_au" in df.columns:
        a_col = "a_au"
    else:
        print(f"Loading orbital elements: {OC_PATH}")
        try:
            oc = pd.read_parquet(OC_PATH)
            df = df.merge(oc[["number_mp", "a_au"]].drop_duplicates("number_mp"),
                          on="number_mp", how="left")
            a_col = "a_au"
            print(f"  a_au merged: {df['a_au'].notna().sum():,}")
        except FileNotFoundError:
            print(f"  Warning: {OC_PATH} not found. Using orbital_class column if available.")

    # Proper elements for L4/L5 distinction (optional)
    use_proper = False
    if PE_PATH.exists():
        try:
            pe = pd.read_parquet(PE_PATH)
            df = df.merge(pe[["number_mp", "a_p"]].drop_duplicates("number_mp"),
                          on="number_mp", how="left")
            use_proper = True
            print(f"  Proper a_p merged: {df['a_p'].notna().sum():,}")
        except Exception as e:
            print(f"  Proper elements merge failed: {e}")

    # ── Filter Trojans ────────────────────────────────────────────────────────
    if "orbital_class" in df.columns:
        trojan_mask_oc = df["orbital_class"].str.contains("Trojan", na=False, case=False)
    else:
        trojan_mask_oc = pd.Series(False, index=df.index)

    if a_col:
        trojan_mask_a = (df[a_col] >= TROJAN_A_MIN) & (df[a_col] <= TROJAN_A_MAX)
    else:
        trojan_mask_a = pd.Series(False, index=df.index)

    trojan_mask = trojan_mask_oc | trojan_mask_a
    trojans = df[trojan_mask].copy()
    print(f"\nTrojans in GAPC: {len(trojans):,}")
    if len(trojans) == 0:
        print("  No Trojans found. Check that orbital_class or a_au columns are present.")
        print("  Run step 15 (orbital_class) first.")

    # Outer MBA for comparison
    if a_col:
        mba_outer_mask = (
            (df[a_col] >= MBA_OUTER_A_MIN) &
            (df[a_col] <= MBA_OUTER_A_MAX) &
            ~trojan_mask
        )
    elif "orbital_class" in df.columns:
        mba_outer_mask = df["orbital_class"].str.contains("MBA-III", na=False)
    else:
        mba_outer_mask = pd.Series(False, index=df.index)

    mba_outer = df[mba_outer_mask].copy()
    print(f"Outer MBA (comparison): {len(mba_outer):,}")

    # G_uncertain coerce
    for sub_df in [trojans, mba_outer, df]:
        if "G_uncertain" in sub_df.columns:
            sub_df["G_uncertain"] = sub_df["G_uncertain"].fillna(False).astype(bool)

    # ── 1. Basic stats ────────────────────────────────────────────────────────
    hv_col = "H_V_tax" if "H_V_tax" in trojans.columns else "H_V"
    print(f"\n  Using {hv_col} for H_V")
    print(f"\nBasic Trojan stats:")
    print(f"  N total:            {len(trojans):,}")

    g_rel = trojans[~trojans.get("G_uncertain", pd.Series(False, index=trojans.index)).fillna(False)]
    print(f"  N reliable G:       {g_rel['G'].notna().sum():,}")
    if len(trojans) > 0 and trojans[hv_col].notna().any():
        print(f"  H_V range:          [{trojans[hv_col].min():.2f}, {trojans[hv_col].max():.2f}]")
        print(f"  H_V median:          {trojans[hv_col].median():.2f}")
    if "n_obs" in trojans.columns:
        print(f"  n_obs median:        {trojans['n_obs'].median():.0f}")
    if "D_km" in trojans.columns and trojans["D_km"].notna().any():
        d = trojans["D_km"].dropna()
        print(f"  D_km range:         [{d.min():.1f}, {d.max():.1f}] km")
        print(f"  D_km median:         {d.median():.1f} km")

    # ── 2. G distribution comparison ─────────────────────────────────────────
    g_troj = trojans.loc[
        ~trojans.get("G_uncertain", pd.Series(False, index=trojans.index)).fillna(False),
        "G"
    ].dropna().values

    g_mba = mba_outer.loc[
        ~mba_outer.get("G_uncertain", pd.Series(False, index=mba_outer.index)).fillna(False),
        "G"
    ].dropna().values

    print(f"\nG distribution:")
    if len(g_troj) > 0:
        print(f"  Trojans (N={len(g_troj):,}):    median={np.median(g_troj):.4f}  "
              f"std={np.std(g_troj):.4f}")
    if len(g_mba) > 0:
        print(f"  MBA-outer (N={len(g_mba):,}):  median={np.median(g_mba):.4f}  "
              f"std={np.std(g_mba):.4f}")
    if HAS_SCIPY and len(g_troj) >= 5 and len(g_mba) >= 5:
        ks_stat, ks_p = ks_2samp(g_troj, g_mba)
        mw_stat, mw_p = mannwhitneyu(g_troj, g_mba, alternative="two-sided")
        print(f"  KS test: D={ks_stat:.4f}  p={ks_p:.3e}")
        print(f"  Mann-Whitney: p={mw_p:.3e}")
    else:
        ks_stat, ks_p, mw_p = np.nan, np.nan, np.nan

    # ── 3. Variability ────────────────────────────────────────────────────────
    print("\nVariability stats:")
    for name, sub in [("Trojans", trojans), ("MBA-outer", mba_outer)]:
        if "chi2_reduced" in sub.columns:
            chi2 = sub["chi2_reduced"].dropna()
            print(f"  {name} chi2_reduced median: {chi2.median():.3f} (N={len(chi2):,})")
        if "var_flag" in sub.columns:
            vf = sub["var_flag"].notna()
            print(f"  {name} var_flag set: {vf.sum():,} / {len(sub):,} "
                  f"({100*vf.mean():.1f}%)")

    # ── 4. Taxonomy ──────────────────────────────────────────────────────────
    trojans["_tax"] = get_tax_group(trojans)
    print("\nTaxonomy of Trojans (predicted + GASP):")
    tax_counts = trojans["_tax"].value_counts()
    for t, n in tax_counts.items():
        print(f"  {t:>8s}: {n:,}")

    # GASP taxonomy directly
    if "gasp_taxonomy_final" in trojans.columns:
        gasp_tax = trojans[trojans["gasp_taxonomy_final"].notna()]["gasp_taxonomy_final"]
        if len(gasp_tax) > 0:
            print("\n  GASP taxonomy (direct, GASP-matched only):")
            for t, n in gasp_tax.value_counts().head(10).items():
                print(f"    {t:>8s}: {n:,}")

    # ── 5. Lucy targets ──────────────────────────────────────────────────────
    lucy_found = df[df["number_mp"].isin(LUCY_TARGETS)].copy()
    print(f"\nLucy mission targets in GAPC catalog ({len(lucy_found)}/{len(LUCY_TARGETS)}):")
    if len(lucy_found) > 0:
        for _, row in lucy_found.iterrows():
            name = LUCY_TARGETS.get(int(row["number_mp"]), "?")
            hv_v = row.get(hv_col, np.nan)
            g_v  = row.get("G", np.nan)
            n_v  = row.get("n_obs", np.nan)
            d_v  = row.get("D_km", np.nan)
            print(f"  ({int(row['number_mp']):>6d}) {name:>15s}:  "
                  f"H_V={hv_v:.3f}  G={g_v:.3f}  n_obs={n_v:.0f}  "
                  f"D={d_v:.1f if not np.isnan(d_v) else '?'} km")
    else:
        print("  None found.")

    # ── Plots ─────────────────────────────────────────────────────────────────

    # 1. G distribution: Trojans vs MBA-outer
    fig, ax = plt.subplots(figsize=(8, 5))
    kde_plot(ax, g_troj,  f"Trojans (N={len(g_troj):,})", "#e07b39")
    kde_plot(ax, g_mba,   f"MBA-outer (N={len(g_mba):,})", "#4a90d9")
    if len(g_troj) > 0:
        ax.axvline(np.median(g_troj), color="#e07b39", linestyle="--",
                   linewidth=1.5, alpha=0.8)
    if len(g_mba) > 0:
        ax.axvline(np.median(g_mba), color="#4a90d9", linestyle="--",
                   linewidth=1.5, alpha=0.8)
    ax.set_xlabel("G (phase slope parameter)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    title_ks = f"\nKS p={ks_p:.3e}" if not np.isnan(ks_p) else ""
    ax.set_title(f"G Distribution: Jupiter Trojans vs Outer MBA{title_ks}", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.1, 1.0)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p1 = PLOT_DIR / "25_trojan_g_distribution.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {p1}")

    # 2. H_V distribution Trojans
    fig, ax = plt.subplots(figsize=(7, 4))
    hv_troj = trojans[hv_col].dropna().values if len(trojans) > 0 else np.array([])
    if len(hv_troj) > 0:
        ax.hist(hv_troj, bins=30, color="#e07b39", alpha=0.8,
                edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(hv_troj), color="red", linestyle="--",
                   label=f"Median = {np.median(hv_troj):.2f}")
        ax.legend(fontsize=10)
    ax.set_xlabel(f"{hv_col} (mag)", fontsize=12)
    ax.set_ylabel("N", fontsize=12)
    ax.set_title(f"H_V Distribution — Jupiter Trojans (N={len(hv_troj):,})", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p2 = PLOT_DIR / "25_trojan_h_distribution.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p2}")

    # 3. D_km distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    if "D_km" in trojans.columns:
        d_troj = trojans["D_km"].dropna().values
        d_troj = d_troj[d_troj > 0]
        if len(d_troj) > 0:
            ax.hist(np.log10(d_troj), bins=25, color="#9ccc65", alpha=0.8,
                    edgecolor="white", linewidth=0.3)
            ax.axvline(np.log10(np.median(d_troj)), color="red", linestyle="--",
                       label=f"Median = {np.median(d_troj):.1f} km")
            ax.legend(fontsize=10)
            # Set x-tick labels in km
            x_ticks = ax.get_xticks()
            ax.set_xticklabels([f"{10**x:.1f}" for x in x_ticks], fontsize=9)
    else:
        ax.text(0.5, 0.5, "D_km column not available", ha="center", va="center",
                transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("Diameter (km)", fontsize=12)
    ax.set_ylabel("N", fontsize=12)
    ax.set_title(f"Diameter Distribution — Jupiter Trojans", fontsize=12)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p3 = PLOT_DIR / "25_trojan_diameter.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p3}")

    # 4. chi2_reduced comparison Trojans vs MBA-outer
    fig, ax = plt.subplots(figsize=(8, 5))
    if "chi2_reduced" in df.columns:
        chi2_troj = trojans["chi2_reduced"].dropna().values
        chi2_mba  = mba_outer["chi2_reduced"].dropna().values
        # Clip for display
        clip_hi = 20
        kde_plot(ax, np.clip(chi2_troj, 0, clip_hi),
                 f"Trojans (N={len(chi2_troj):,})", "#e07b39")
        kde_plot(ax, np.clip(chi2_mba, 0, clip_hi),
                 f"MBA-outer (N={len(chi2_mba):,})", "#4a90d9")
        ax.axvline(1.0, color="black", linestyle=":", alpha=0.5, label="chi2=1")
        ax.set_xlim(0, clip_hi)
    else:
        ax.text(0.5, 0.5, "chi2_reduced column not available",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
    ax.set_xlabel("chi2_reduced (HG phase curve fit)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Phase Curve Fit Quality: Trojans vs MBA-outer", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p4 = PLOT_DIR / "25_trojan_chi2.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p4}")

    # 5. Lucy targets in G vs H_V scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    if len(trojans) > 0 and trojans["G"].notna().any():
        ax.scatter(trojans[hv_col], trojans["G"],
                   alpha=0.3, s=10, color="#aaaaaa", label=f"All Trojans (N={len(trojans):,})",
                   zorder=1, rasterized=True)
    if len(lucy_found) > 0:
        for _, row in lucy_found.iterrows():
            name = LUCY_TARGETS.get(int(row["number_mp"]), "?")
            hv_v = row.get(hv_col, np.nan)
            g_v  = row.get("G", np.nan)
            if not (np.isnan(hv_v) or np.isnan(g_v)):
                ax.scatter(hv_v, g_v, s=120, color="#e53935",
                           zorder=5, edgecolors="black", linewidths=0.8)
                ax.annotate(name, (hv_v, g_v),
                            textcoords="offset points", xytext=(7, 4),
                            fontsize=9, color="#e53935", fontweight="bold")
    else:
        ax.text(0.5, 0.2, "Lucy targets not found in catalog\n"
                "(may lack sufficient Gaia observations)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=10, style="italic", alpha=0.7)
    ax.set_xlabel(f"{hv_col} (mag)", fontsize=12)
    ax.set_ylabel("G (phase slope)", fontsize=12)
    ax.set_title("G vs H_V — Jupiter Trojans (Lucy targets highlighted)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    p5 = PLOT_DIR / "25_lucy_targets.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p5}")

    # ── Log file ──────────────────────────────────────────────────────────────
    log_path = LOG_DIR / "25_trojan_stats.txt"
    with open(log_path, "w") as f:
        f.write("GAPC Step 25 — Jupiter Trojan Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Catalog: {CAT_PATH}\n")
        f.write(f"H_V column: {hv_col}\n")
        f.write(f"Trojans in GAPC: {len(trojans):,}\n")
        f.write(f"Trojans with reliable G: {len(g_troj):,}\n")
        f.write(f"MBA-outer comparison sample: {len(mba_outer):,}\n\n")

        if len(g_troj) > 0:
            f.write("G distribution — Trojans:\n")
            f.write(f"  N={len(g_troj):,}  median={np.median(g_troj):.4f}  "
                    f"std={np.std(g_troj):.4f}  "
                    f"mean={np.mean(g_troj):.4f}\n\n")
        if len(g_mba) > 0:
            f.write("G distribution — MBA-outer:\n")
            f.write(f"  N={len(g_mba):,}  median={np.median(g_mba):.4f}  "
                    f"std={np.std(g_mba):.4f}\n\n")

        if not np.isnan(ks_p):
            f.write(f"KS test (Trojan G vs MBA-outer G): D={ks_stat:.4f}  p={ks_p:.3e}\n")
            f.write(f"Mann-Whitney: p={mw_p:.3e}\n\n")

        f.write("Taxonomy distribution (Trojans):\n")
        for t, n in tax_counts.items():
            f.write(f"  {t:>8s}: {n:,}\n")

        f.write("\nLucy mission targets:\n")
        if len(lucy_found) > 0:
            for _, row in lucy_found.iterrows():
                name = LUCY_TARGETS.get(int(row["number_mp"]), "?")
                f.write(f"  ({int(row['number_mp']):>6d}) {name:>15s}:  "
                        f"H_V={row.get(hv_col, np.nan):.3f}  "
                        f"G={row.get('G', np.nan):.3f}\n")
        else:
            f.write("  None found in GAPC catalog.\n")

    print(f"\n  Log: {log_path}")
    print("\nStep 25 complete.")


if __name__ == "__main__":
    main()
