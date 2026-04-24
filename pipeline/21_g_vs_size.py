"""
21_g_vs_size.py
GAPC — G parameter as a function of asteroid size (space weathering signal test).

Tests whether the HG phase slope G correlates with asteroid size (H_V proxy,
and D_km where available), and whether this trend differs between S-types
(expected: size-dependent space weathering) and C-types (expected: weaker trend).

Space weathering hypothesis (Clark et al. 2002; Nesvorny et al. 2005):
  Larger S-type asteroids → older surfaces → more space weathering →
  lower geometric albedo → flatter phase curve → lower G

Outputs:
  plots/21_g_vs_size_4panel.png
  logs/21_g_vs_size_stats.txt
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v4.parquet"
MPC_PATH = ROOT / "data" / "raw"   / "mpc_h_magnitudes.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

HV_BIN_EDGES = np.arange(9.0, 20.5, 0.5)
HV_BIN_CENTERS = (HV_BIN_EDGES[:-1] + HV_BIN_EDGES[1:]) / 2

MIN_BIN_N = 10   # minimum objects per bin to include in plot


def bin_stats(df, hv_col="H_V", g_col="G", edges=HV_BIN_EDGES, g_unc_col="G_uncertain"):
    """Return per-bin stats: center, N, median G, std G, G_uncertain fraction."""
    rows = []
    for lo, hi, cen in zip(edges[:-1], edges[1:], (edges[:-1] + edges[1:]) / 2):
        sub = df[(df[hv_col] >= lo) & (df[hv_col] < hi) & df[g_col].notna()]
        n = len(sub)
        if n == 0:
            rows.append(dict(center=cen, N=0, G_median=np.nan, G_std=np.nan,
                             G_unc_frac=np.nan))
            continue
        g_med = sub[g_col].median()
        g_std = sub[g_col].std()
        if g_unc_col in sub.columns:
            unc_frac = sub[g_unc_col].mean() if sub[g_unc_col].notna().any() else np.nan
        else:
            unc_frac = np.nan
        rows.append(dict(center=cen, N=n, G_median=g_med, G_std=g_std,
                         G_unc_frac=unc_frac))
    return pd.DataFrame(rows)


def spearman_test(df, x_col, y_col, label=""):
    """Run Spearman test, print and return (rho, pvalue)."""
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < 10:
        print(f"  {label}: too few data ({len(sub)}), skipping")
        return np.nan, np.nan
    rho, pval = spearmanr(sub[x_col], sub[y_col])
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    print(f"  {label}: rho={rho:+.4f}  p={pval:.2e}  {sig}  (N={len(sub):,})")
    return rho, pval


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 21 — G vs Size / Space Weathering Test")
    print("=" * 60)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading catalog: {CAT_PATH}")
    df = pd.read_parquet(CAT_PATH)
    print(f"  Rows: {len(df):,}")

    # Use H_V_tax if available, else fall back to H_V
    if "H_V_tax" in df.columns:
        df["_hv"] = df["H_V_tax"]
        hv_label = "H_V_tax"
    else:
        df["_hv"] = df["H_V"]
        hv_label = "H_V"
    print(f"  Using {hv_label} for H_V axis")

    # G_uncertain: coerce to bool if needed
    if "G_uncertain" in df.columns:
        df["G_uncertain"] = df["G_uncertain"].fillna(False).astype(bool)
    else:
        df["G_uncertain"] = False

    # Reliable G: fit_ok and not G_uncertain
    if "fit_ok" in df.columns:
        df_rel = df[df["fit_ok"] & ~df["G_uncertain"]].copy()
    else:
        df_rel = df[~df["G_uncertain"]].copy()
    print(f"  Reliable G objects (fit_ok & not G_uncertain): {len(df_rel):,}")

    # Define taxonomy columns: prefer predicted_taxonomy, supplement with gasp
    def get_tax_group(df_in):
        pt = df_in.get("predicted_taxonomy", pd.Series(np.nan, index=df_in.index))
        gf = df_in.get("gasp_taxonomy_final", pd.Series(np.nan, index=df_in.index))
        result = pt.copy()
        need = result.isna()
        raw = gf[need].str.strip().str.upper().str[0]
        tax_map = {"S": "S", "C": "C", "X": "X"}
        result[need] = raw.map(tax_map).fillna("Other")
        return result

    df_rel["_tax"] = get_tax_group(df_rel)

    # ── Analysis 1: G vs H_V full sample ──────────────────────────────────────
    print("\nAnalysis 1: G vs H_V full sample")
    bins_full = bin_stats(df_rel, hv_col="_hv")
    bins_full_ok = bins_full[bins_full["N"] >= MIN_BIN_N]
    rho1, p1 = spearman_test(df_rel, "_hv", "G", label="Full sample G vs H_V")

    # ── Analysis 2: S-type space weathering test ───────────────────────────────
    print("\nAnalysis 2: S-type G vs H_V (space weathering)")
    s_type = df_rel[df_rel["_tax"] == "S"].copy()
    print(f"  S-type objects: {len(s_type):,}")
    bins_S = bin_stats(s_type, hv_col="_hv")
    bins_S_ok = bins_S[bins_S["N"] >= MIN_BIN_N]
    rho2, p2 = spearman_test(s_type, "_hv", "G", label="S-type G vs H_V")

    # Direction check: negative rho = G decreases with H_V
    # H_V increases → smaller (dimmer) asteroid
    # For space weathering: larger = lower H_V → more weathered → lower G
    # So expect: G DECREASES as H_V DECREASES → positive rho (G increases with H_V)?
    # Actually: large asteroid → low H_V → old surface → low G
    #           small asteroid → high H_V → fresh surface → high G
    # => positive rho expected for weathering signal
    print(f"  Space weathering signal: rho={rho2:+.4f}")
    if not np.isnan(rho2):
        if rho2 > 0.1 and p2 < 0.05:
            print("  => POSITIVE TREND DETECTED: larger S-types show lower G "
                  "(consistent with space weathering)")
        elif rho2 < -0.1 and p2 < 0.05:
            print("  => NEGATIVE TREND: smaller S-types show lower G "
                  "(counter to space weathering hypothesis)")
        else:
            print("  => No significant trend detected")

    # ── Analysis 3: C-type comparison ────────────────────────────────────────
    print("\nAnalysis 3: C-type G vs H_V (comparison)")
    c_type = df_rel[df_rel["_tax"] == "C"].copy()
    print(f"  C-type objects: {len(c_type):,}")
    bins_C = bin_stats(c_type, hv_col="_hv")
    bins_C_ok = bins_C[bins_C["N"] >= MIN_BIN_N]
    rho3, p3 = spearman_test(c_type, "_hv", "G", label="C-type G vs H_V")

    # ── Analysis 4: G vs log10(D_km) ─────────────────────────────────────────
    print("\nAnalysis 4: G vs diameter D_km")
    diam = df_rel[df_rel["D_km"].notna() & (df_rel["D_km"] > 0)].copy()
    diam["log10_D"] = np.log10(diam["D_km"])
    print(f"  Objects with D_km: {len(diam):,}")
    rho4, p4 = spearman_test(diam, "log10_D", "G", label="G vs log10(D_km)")

    # Binned by log10(D_km)
    d_edges = np.arange(np.floor(diam["log10_D"].min() * 2) / 2,
                        np.ceil(diam["log10_D"].max() * 2) / 2 + 0.5,
                        0.5)
    if len(d_edges) < 3:
        d_edges = np.linspace(diam["log10_D"].min(), diam["log10_D"].max(), 10)
    d_centers = (d_edges[:-1] + d_edges[1:]) / 2
    bins_D = []
    for lo, hi, cen in zip(d_edges[:-1], d_edges[1:], d_centers):
        sub = diam[(diam["log10_D"] >= lo) & (diam["log10_D"] < hi)]
        if len(sub) >= MIN_BIN_N:
            bins_D.append(dict(center=cen, N=len(sub),
                               G_median=sub["G"].median(),
                               G_std=sub["G"].std()))
    bins_D = pd.DataFrame(bins_D) if bins_D else pd.DataFrame(
        columns=["center", "N", "G_median", "G_std"])

    # ── 4-panel figure ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: G median vs H_V, full sample
    ax = axes[0, 0]
    if len(bins_full_ok) > 0:
        ax.errorbar(bins_full_ok["center"], bins_full_ok["G_median"],
                    yerr=bins_full_ok["G_std"] / np.sqrt(np.maximum(bins_full_ok["N"], 1)),
                    fmt="o-", color="#5c85d6", linewidth=1.5, markersize=5,
                    capsize=3, label=f"All (N={len(df_rel):,})")
    ax.set_xlabel(f"{hv_label} (mag)", fontsize=11)
    ax.set_ylabel("Median G", fontsize=11)
    ax.set_title(f"G vs {hv_label} — Full Sample\n"
                 f"Spearman rho={rho1:+.3f}  p={p1:.2e}", fontsize=11)
    ax.axhline(0.15, color="gray", linestyle=":", alpha=0.5, label="G=0.15")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    # Panel 2: G median vs H_V, S-type vs C-type
    ax = axes[0, 1]
    if len(bins_S_ok) > 0:
        ax.errorbar(bins_S_ok["center"], bins_S_ok["G_median"],
                    yerr=bins_S_ok["G_std"] / np.sqrt(np.maximum(bins_S_ok["N"], 1)),
                    fmt="s-", color="#e07b39", linewidth=1.5, markersize=5,
                    capsize=3, label=f"S-type (N={len(s_type):,})  rho={rho2:+.3f}")
    if len(bins_C_ok) > 0:
        ax.errorbar(bins_C_ok["center"], bins_C_ok["G_median"],
                    yerr=bins_C_ok["G_std"] / np.sqrt(np.maximum(bins_C_ok["N"], 1)),
                    fmt="^-", color="#4a90d9", linewidth=1.5, markersize=5,
                    capsize=3, label=f"C-type (N={len(c_type):,})  rho={rho3:+.3f}")
    ax.set_xlabel(f"{hv_label} (mag)", fontsize=11)
    ax.set_ylabel("Median G", fontsize=11)
    ax.set_title(f"G vs {hv_label} — S-type vs C-type\n(space weathering test)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    # Panel 3: G vs log10(D_km) scatter + running median
    ax = axes[1, 0]
    if len(diam) > 0:
        ax.scatter(diam["log10_D"], diam["G"],
                   alpha=0.08, s=3, color="#888888", rasterized=True)
    if len(bins_D) > 0:
        ax.plot(bins_D["center"], bins_D["G_median"],
                "ro-", linewidth=2, markersize=6, label="Running median")
        ax.errorbar(bins_D["center"], bins_D["G_median"],
                    yerr=bins_D["G_std"] / np.sqrt(np.maximum(bins_D["N"], 1)),
                    fmt="none", color="red", capsize=3, alpha=0.6)
    ax.set_xlabel("log10(D_km)", fontsize=11)
    ax.set_ylabel("G", fontsize=11)
    ax.set_title(f"G vs Diameter\nSpearman rho={rho4:+.3f}  p={p4:.2e}  "
                 f"(N={len(diam):,})", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 4: G_uncertain fraction vs H_V (quality context)
    ax = axes[1, 1]
    unc_bins = bin_stats(df, hv_col="_hv", g_col="G")  # include all for uncertainty frac
    unc_ok = unc_bins[unc_bins["N"] >= MIN_BIN_N]
    if len(unc_ok) > 0:
        ax.bar(unc_ok["center"], unc_ok["G_unc_frac"],
               width=0.45, color="#e05c5c", alpha=0.7,
               label="G_uncertain fraction")
        ax2 = ax.twinx()
        ax2.plot(unc_ok["center"], unc_ok["N"], "k--", linewidth=1, alpha=0.5, label="N objects")
        ax2.set_ylabel("N objects per bin", fontsize=10)
        ax2.legend(loc="upper left", fontsize=9)
    ax.set_xlabel(f"{hv_label} (mag)", fontsize=11)
    ax.set_ylabel("G_uncertain fraction", fontsize=11)
    ax.set_title("G Quality Context: G_uncertain Fraction vs H_V", fontsize=11)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.invert_xaxis()

    fig.suptitle("GAPC — G Parameter vs Asteroid Size", fontsize=14, y=1.01)
    plt.tight_layout()
    p1_path = PLOT_DIR / "21_g_vs_size_4panel.png"
    fig.savefig(p1_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {p1_path}")

    # ── Log file ──────────────────────────────────────────────────────────────
    log_path = LOG_DIR / "21_g_vs_size_stats.txt"
    with open(log_path, "w") as f:
        f.write("GAPC Step 21 — G vs Size / Space Weathering Test\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Catalog: {CAT_PATH}\n")
        f.write(f"Total rows: {len(df):,}\n")
        f.write(f"Reliable G (fit_ok & not G_uncertain): {len(df_rel):,}\n")
        f.write(f"H_V column used: {hv_label}\n\n")

        f.write("Analysis 1 — Full sample G vs H_V:\n")
        f.write(f"  Spearman rho={rho1:+.4f}  p={p1:.3e}\n\n")

        f.write("Analysis 2 — S-type G vs H_V (space weathering test):\n")
        f.write(f"  N S-type objects: {len(s_type):,}\n")
        f.write(f"  Spearman rho={rho2:+.4f}  p={p2:.3e}\n")
        if not np.isnan(rho2):
            if rho2 > 0.1 and p2 < 0.05:
                f.write("  Interpretation: POSITIVE TREND — larger S-types show lower G "
                        "(space weathering signal detected)\n\n")
            elif rho2 < -0.1 and p2 < 0.05:
                f.write("  Interpretation: NEGATIVE TREND — counter to space weathering hypothesis\n\n")
            else:
                f.write("  Interpretation: No significant trend\n\n")

        f.write("Analysis 3 — C-type G vs H_V (comparison):\n")
        f.write(f"  N C-type objects: {len(c_type):,}\n")
        f.write(f"  Spearman rho={rho3:+.4f}  p={p3:.3e}\n\n")

        f.write("Analysis 4 — G vs log10(D_km):\n")
        f.write(f"  N objects with D_km: {len(diam):,}\n")
        f.write(f"  Spearman rho={rho4:+.4f}  p={p4:.3e}\n\n")

        f.write("Binned G stats — Full sample (H_V bins):\n")
        f.write(f"  {'Bin ctr':>8s}  {'N':>6s}  {'G median':>9s}  {'G std':>8s}\n")
        for _, row in bins_full_ok.iterrows():
            f.write(f"  {row['center']:>8.2f}  {int(row['N']):>6,}  "
                    f"{row['G_median']:>9.4f}  {row['G_std']:>8.4f}\n")
    print(f"  Log: {log_path}")

    print("\nStep 21 complete.")


if __name__ == "__main__":
    main()
