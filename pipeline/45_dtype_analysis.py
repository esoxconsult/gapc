"""
45_dtype_analysis.py
GAPC — D-type asteroid G analysis.

Step 44 found D-type median G=0.212, higher than S-types (0.145).
D-types are organic-rich, reddish surfaces common in the outer belt and
among Jupiter Trojans. Their G behaviour has not been systematically
studied with a large sample.

Questions:
  1. Is D-type G=0.212 significantly different from S-types at the same size?
  2. Are our D-types actually Trojans or outer-belt MBAs?
     (RF was trained on a predominantly MBA sample — Trojans could be mislabeled)
  3. Does D-type G correlate with size like S-types?

Outputs:
  plots/45_dtype_analysis.png
  logs/45_dtype_analysis_stats.txt
  (v8 NOT modified — read-only)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr, kruskal

ROOT    = Path(__file__).resolve().parents[1]
V8_PATH = ROOT / "data" / "final" / "gapc_catalog_v8.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 45 — D-type asteroid G analysis")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V8_PATH.exists():
        print(f"\n  ERROR: {V8_PATH} not found"); return

    gapc = pd.read_parquet(V8_PATH)
    print(f"\n  v8 loaded: {len(gapc):,} rows, {len(gapc.columns)} cols")

    tax = "taxonomy_refined"
    oc_col = "orbital_class" if "orbital_class" in gapc.columns else "gasp_orbital_class"
    has_oc = oc_col in gapc.columns

    # ── D-type population ─────────────────────────────────────────────────────
    d_all  = gapc[gapc[tax] == "D"]
    d_g    = d_all[d_all["G"].notna()]
    s_all  = gapc[gapc[tax] == "S"]
    s_g    = s_all[s_all["G"].notna()]

    print(f"\n  D-types: {len(d_all):,} total, {len(d_g):,} with G")
    print(f"  S-types: {len(s_all):,} total, {len(s_g):,} with G")
    print(f"\n  D median G = {d_g['G'].median():.4f}  (n={len(d_g):,})")
    print(f"  S median G = {s_g['G'].median():.4f}  (n={len(s_g):,})")
    U, p_ds = mannwhitneyu(d_g["G"], s_g["G"], alternative="two-sided")
    print(f"  Mann-Whitney D vs S: p={p_ds:.3e}")

    # ── Orbital distribution of D-types ──────────────────────────────────────
    if has_oc:
        print(f"\n  D-type orbital distribution:")
        oc_counts = d_all[oc_col].value_counts()
        for zone, n in oc_counts.items():
            print(f"    {zone:20s}: {n:3d} ({n/len(d_all)*100:.1f}%)")

    # ── D vs S at same size ───────────────────────────────────────────────────
    print(f"\n  G comparison at fixed size (D vs S):")
    size_bins = [(0.1, 1), (1, 3), (3, 10), (10, 30), (30, 999)]
    bin_labels = ["0.1–1", "1–3", "3–10", "10–30", ">30"]
    bin_results = []
    for (lo, hi), lbl in zip(size_bins, bin_labels):
        d_bin = d_g[d_g["D_km"].between(lo, hi)]["G"]
        s_bin = s_g[s_g["D_km"].between(lo, hi)]["G"]
        if len(d_bin) < 2 or len(s_bin) < 10:
            continue
        if len(d_bin) >= 2 and len(s_bin) >= 2:
            U_s, p_s = mannwhitneyu(d_bin, s_bin, alternative="two-sided")
        else:
            p_s = np.nan
        print(f"    {lbl:8s} km:  D n={len(d_bin):2d} med={d_bin.median():.4f}  "
              f"S n={len(s_bin):5,} med={s_bin.median():.4f}  p={p_s:.3e}")
        bin_results.append(dict(label=lbl, g_d=d_bin.median(), g_s=s_bin.median(),
                                n_d=len(d_bin), n_s=len(s_bin), p=p_s))

    # ── G × log D for D-types ─────────────────────────────────────────────────
    d_sz = d_g[d_g["D_km"].notna() & (d_g["D_km"] > 0)].copy()
    d_sz["log_D"] = np.log10(d_sz["D_km"])
    if len(d_sz) >= 10:
        rho_d, p_rho_d = spearmanr(d_sz["G"], d_sz["log_D"])
        print(f"\n  D-type rho(G, logD) = {rho_d:+.4f}  p={p_rho_d:.3e}  n={len(d_sz)}")
    else:
        rho_d, p_rho_d = np.nan, np.nan

    # ── Compare D with outer-belt S for space weathering context ─────────────
    if has_oc:
        s_outer = s_g[s_g[oc_col] == "MBA-outer"]["G"]
        d_outer = d_g[d_g[oc_col] == "MBA-outer"]["G"] if has_oc else d_g["G"]
        if len(d_outer) >= 5 and len(s_outer) >= 10:
            U3, p3 = mannwhitneyu(d_outer, s_outer, alternative="two-sided")
            print(f"\n  Outer belt: D G={d_outer.median():.4f} (n={len(d_outer)})  "
                  f"vs S G={s_outer.median():.4f} (n={len(s_outer):,})  p={p3:.3e}")
        else:
            p3 = np.nan
    else:
        p3 = np.nan

    # ── Compare D, S, C, X G full distributions via Kruskal-Wallis ──────────
    groups = {}
    for t in ["D", "S", "C", "X", "E", "M", "P"]:
        sub = gapc[(gapc[tax] == t) & gapc["G"].notna()]["G"]
        if len(sub) >= 5:
            groups[t] = sub.values
    if len(groups) >= 3:
        H_kw, p_kw = kruskal(*groups.values())
        print(f"\n  Kruskal-Wallis G ({list(groups.keys())}): H={H_kw:.1f}  p={p_kw:.2e}")

    # ── Taxonomy source for D-types ───────────────────────────────────────────
    if "taxonomy_source" in gapc.columns:
        src = d_all["taxonomy_source"].value_counts()
        print(f"\n  D-type taxonomy_source:")
        for s, n in src.items():
            print(f"    {s}: {n}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"D-type G analysis  (n_D={len(d_all):,})", fontsize=12)

    # G distribution: D vs S vs C
    ax = axes[0, 0]
    g_rng = gapc["G"].quantile([0.01, 0.99]).values
    bins_g = np.linspace(g_rng[0], g_rng[1], 50)
    for t, col, alpha in [("S", "steelblue", 0.5), ("C", "gray", 0.5),
                           ("D", "#c0392b", 0.85)]:
        sub = gapc[(gapc[tax] == t) & gapc["G"].notna()]["G"]
        if len(sub) < 5:
            continue
        ax.hist(sub.clip(*g_rng).values, bins=bins_g, density=True,
                histtype="stepfilled" if t == "D" else "step",
                color=col, alpha=alpha, lw=1.5, label=f"{t} (n={len(sub):,})")
        ax.axvline(sub.median(), color=col, lw=1.5, ls="--")
    ax.set_xlabel("G (phase slope)"); ax.set_ylabel("Density")
    ax.set_title(f"G: D vs S vs C  p(D≠S)={p_ds:.2e}")
    ax.legend(fontsize=9)

    # G in size bins
    ax = axes[0, 1]
    if bin_results:
        lbls = [r["label"] for r in bin_results]
        x = np.arange(len(lbls))
        w = 0.35
        ax.bar(x - w/2, [r["g_d"] for r in bin_results], w,
               color="#c0392b", alpha=0.8, label="D-type")
        ax.bar(x + w/2, [r["g_s"] for r in bin_results], w,
               color="steelblue", alpha=0.8, label="S-type")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x); ax.set_xticklabels(
            [f"{r['label']}\n(n_D={r['n_d']})" for r in bin_results],
            fontsize=8)
        ax.set_ylabel("Median G")
        ax.set_title("G by size bin: D vs S")
        ax.legend(fontsize=9); ax.grid(alpha=0.2, axis="y")

    # D-type G vs D_km
    ax = axes[1, 0]
    ax.scatter(d_sz["D_km"], d_sz["G"], s=25, color="#c0392b", alpha=0.7,
               edgecolors="k", lw=0.3, zorder=3, label="D-type")
    if len(s_g) > 0:
        smp_s = s_g[s_g["D_km"].notna()].sample(
            min(2000, len(s_g)), random_state=42)
        ax.scatter(smp_s["D_km"], smp_s["G"], s=2, color="steelblue",
                   alpha=0.15, rasterized=True, label="S-type (sample)")
    ax.set_xscale("log")
    ax.set_xlabel("D [km]"); ax.set_ylabel("G")
    ax.set_title(f"D-type G vs size  rho={rho_d:+.3f}  p={p_rho_d:.2e}")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)

    # Orbital class distribution of D-types
    ax = axes[1, 1]
    if has_oc and len(oc_counts) > 0:
        ax.bar(range(len(oc_counts)), oc_counts.values,
               color="#c0392b", alpha=0.8)
        ax.set_xticks(range(len(oc_counts)))
        ax.set_xticklabels(oc_counts.index, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("N D-type")
        ax.set_title("D-type orbital distribution")
        ax.grid(alpha=0.2, axis="y")
    else:
        ax.text(0.5, 0.5, "No orbital_class", transform=ax.transAxes,
                ha="center", va="center")
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "45_dtype_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/45_dtype_analysis.png")

    with open(LOG_DIR / "45_dtype_analysis_stats.txt", "w") as f:
        f.write("GAPC Step 45 — D-type G analysis\n")
        f.write("=" * 60 + "\n")
        f.write(f"n_D:           {len(d_all):,}\n")
        f.write(f"n_D_with_G:    {len(d_g):,}\n")
        f.write(f"D median G:    {d_g['G'].median():.4f}\n")
        f.write(f"S median G:    {s_g['G'].median():.4f}\n")
        f.write(f"MW p(D≠S):     {p_ds:.3e}\n")
        if not np.isnan(rho_d):
            f.write(f"rho(G,logD) D: {rho_d:+.4f}  p={p_rho_d:.3e}\n")
        f.write("\nG by size bin:\n")
        for r in bin_results:
            f.write(f"  {r['label']:8s}: D={r['g_d']:.4f}(n={r['n_d']})  "
                    f"S={r['g_s']:.4f}(n={r['n_s']:,})  p={r['p']:.3e}\n")
    print(f"  Log  → logs/45_dtype_analysis_stats.txt\n")


if __name__ == "__main__":
    main()
