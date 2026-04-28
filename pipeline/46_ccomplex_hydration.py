"""
46_ccomplex_hydration.py
GAPC — C-complex hydration signal in G.

Step 44 found Ch (hydrated C) G = 0.017 vs C G = 0.014 — nearly identical.
Step 38 found pIR/pV borderline significant (p=0.003).

This step investigates the C-complex more carefully:
  1. C vs Ch vs B: are these really indistinguishable in G?
  2. pIR/pV gradient within C-complex: does G decrease monotonically with pIR?
  3. Per orbital zone: outer-belt C-types vs inner-belt C-types
     (outer belt has more hydrated C-types, more space weathering)
  4. Themis family (model hydrated family) vs Hygiea family vs generic C

Outputs:
  plots/46_ccomplex_hydration.png
  logs/46_ccomplex_hydration_stats.txt
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
    print("  GAPC Step 46 — C-complex hydration signal in G")
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
    pir_col = "neowise_pIR_ratio"
    has_pir = pir_col in gapc.columns

    # ── C / Ch / B comparison ─────────────────────────────────────────────────
    print(f"\n  1. C / Ch / B G comparison:")
    ctypes = {}
    for t in ["C", "Ch", "B", "F"]:
        sub = gapc[(gapc[tax] == t) & gapc["G"].notna()]
        if len(sub) < 5:
            continue
        ctypes[t] = sub
        print(f"    {t:4s}: G_med={sub['G'].median():.4f}  "
              f"G_mean={sub['G'].mean():.4f}  n={len(sub):,}")

    if "C" in ctypes and "Ch" in ctypes:
        U1, p1 = mannwhitneyu(ctypes["C"]["G"], ctypes["Ch"]["G"],
                               alternative="two-sided")
        print(f"  MW C vs Ch: p={p1:.3e}")
    else:
        p1 = np.nan

    if len(ctypes) >= 2:
        H_kw, p_kw = kruskal(*[v["G"].values for v in ctypes.values()])
        print(f"  Kruskal-Wallis C/Ch/B: H={H_kw:.2f}  p={p_kw:.3e}")
    else:
        p_kw = np.nan

    # ── pIR/pV gradient within C-complex ─────────────────────────────────────
    pir_results = {}
    if has_pir:
        print(f"\n  2. pIR/pV gradient within C-complex:")
        c_pir = gapc[gapc[tax].isin(["C", "Ch", "B"]) &
                     gapc["G"].notna() & gapc[pir_col].notna() &
                     (gapc[pir_col] > 0) & (gapc[pir_col] < 10)]
        if len(c_pir) >= 20:
            rho_c, p_c = spearmanr(c_pir["G"], c_pir[pir_col])
            print(f"    rho(G, pIR/pV) for C-complex: {rho_c:+.4f}  "
                  f"p={p_c:.3e}  n={len(c_pir):,}")
            # Quartile breakdown
            pir_q = pd.qcut(c_pir[pir_col], 4, labels=["Q1\n(dry)", "Q2", "Q3", "Q4\n(wet)"])
            print(f"    G by pIR/pV quartile:")
            for q, grp in c_pir.groupby(pir_q, observed=True)["G"]:
                print(f"      {str(q):8s}: G_med={grp.median():.4f}  n={len(grp)}")
                pir_results[str(q)] = (grp.median(), len(grp))
        else:
            rho_c, p_c = np.nan, np.nan
            print(f"    Insufficient C-complex data with pIR/pV (n={len(c_pir)})")
    else:
        rho_c, p_c = np.nan, np.nan
        print(f"  No {pir_col} column")

    # ── Per orbital zone ──────────────────────────────────────────────────────
    zone_results = {}
    if has_oc:
        print(f"\n  3. C-complex G by orbital zone:")
        c_all = gapc[gapc[tax].isin(["C", "Ch", "B"]) & gapc["G"].notna()]
        for zone in ["MBA-inner", "MBA-middle", "MBA-outer"]:
            z = c_all[c_all[oc_col] == zone]["G"]
            if len(z) < 10:
                continue
            print(f"    {zone:15s}: G_med={z.median():.4f}  n={len(z):,}")
            zone_results[zone] = z

        # Also C vs Ch per zone
        print(f"\n  C vs Ch by zone:")
        for zone in ["MBA-inner", "MBA-middle", "MBA-outer"]:
            for t in ["C", "Ch"]:
                z = gapc[(gapc[tax] == t) & (gapc[oc_col] == zone) &
                         gapc["G"].notna()]["G"]
                if len(z) >= 10:
                    print(f"    {zone:15s} {t}: G_med={z.median():.4f}  n={len(z):,}")

    # ── Family analysis ───────────────────────────────────────────────────────
    family_results = {}
    fam_col = "family" if "family" in gapc.columns else "gasp_family"
    if fam_col in gapc.columns:
        print(f"\n  4. G by family (C-complex families):")
        c_fam = gapc[gapc[tax].isin(["C", "Ch", "B"]) &
                     gapc["G"].notna() & gapc[fam_col].notna()]
        fam_counts = c_fam[fam_col].value_counts().head(10)
        for fam, n in fam_counts.items():
            z = c_fam[c_fam[fam_col] == fam]["G"]
            print(f"    {str(fam):20s}: G_med={z.median():.4f}  n={n}")
            family_results[str(fam)] = (z.median(), n)

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("C-complex hydration signal in G", fontsize=12)

    # G distribution C vs Ch
    ax = axes[0, 0]
    g_rng = gapc["G"].quantile([0.01, 0.99]).values
    bins_g = np.linspace(g_rng[0], g_rng[1], 50)
    colors_c = {"C": "#2166ac", "Ch": "#d6604d", "B": "#4dac26"}
    for t, col in colors_c.items():
        if t not in ctypes:
            continue
        ax.hist(ctypes[t]["G"].clip(*g_rng).values, bins=bins_g, density=True,
                histtype="step", lw=2, color=col,
                label=f"{t} (n={len(ctypes[t]):,}  med={ctypes[t]['G'].median():.3f})")
        ax.axvline(ctypes[t]["G"].median(), color=col, lw=1.2, ls="--")
    ax.set_xlabel("G"); ax.set_ylabel("Density")
    ax.set_title(f"C vs Ch vs B  p(C≠Ch)={p1:.2e}")
    ax.legend(fontsize=8)

    # pIR/pV vs G scatter (C-complex)
    ax = axes[0, 1]
    if has_pir and len(c_pir) >= 20:
        ax.scatter(c_pir[pir_col], c_pir["G"], s=3, alpha=0.3, rasterized=True,
                   color="#2166ac")
        ax.axvline(1.5, color="orange", lw=1.2, ls="--", label="pIR/pV=1.5 (Ch threshold)")
        ax.set_xlabel("pIR/pV (NEOWISE)"); ax.set_ylabel("G")
        ax.set_title(f"G vs pIR/pV (C-complex)  rho={rho_c:+.3f}  p={p_c:.2e}")
        ax.set_xlim(0, 5); ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    else:
        ax.text(0.5, 0.5, "Insufficient pIR/pV data",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()

    # G by orbital zone (C-complex)
    ax = axes[1, 0]
    if zone_results:
        zone_lbls = list(zone_results.keys())
        zone_meds = [v.median() for v in zone_results.values()]
        zone_ns   = [len(v) for v in zone_results.values()]
        bars = ax.bar(range(len(zone_lbls)), zone_meds,
                      color=["#74add1", "#4393c3", "#2166ac"], alpha=0.85)
        ax.set_xticks(range(len(zone_lbls)))
        ax.set_xticklabels([f"{l}\n(n={n:,})" for l, n in zip(zone_lbls, zone_ns)],
                           fontsize=9)
        ax.set_ylabel("Median G")
        ax.set_title("C-complex G by orbital zone")
        ax.grid(alpha=0.2, axis="y")
    else:
        ax.text(0.5, 0.5, "No orbital zone data", transform=ax.transAxes,
                ha="center", va="center"); ax.set_axis_off()

    # G by family (top 8)
    ax = axes[1, 1]
    if family_results:
        fams = list(family_results.keys())[:8]
        meds = [family_results[f][0] for f in fams]
        ns   = [family_results[f][1] for f in fams]
        bars = ax.barh(range(len(fams)), meds, color="#2166ac", alpha=0.8)
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels([f"{f} (n={n})" for f, n in zip(fams, ns)], fontsize=8)
        ax.set_xlabel("Median G")
        ax.set_title("G by family (C-complex)")
        ax.axvline(0, color="k", lw=0.5)
        ax.grid(alpha=0.2, axis="x")
    else:
        ax.text(0.5, 0.5, "No family data", transform=ax.transAxes,
                ha="center", va="center"); ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "46_ccomplex_hydration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/46_ccomplex_hydration.png")

    with open(LOG_DIR / "46_ccomplex_hydration_stats.txt", "w") as f:
        f.write("GAPC Step 46 — C-complex hydration signal\n")
        f.write("=" * 60 + "\n")
        for t, sub in ctypes.items():
            f.write(f"{t}: G_med={sub['G'].median():.4f}  n={len(sub):,}\n")
        f.write(f"MW p(C≠Ch): {p1:.3e}\n")
        f.write(f"KW p(C/Ch/B): {p_kw:.3e}\n")
        if not np.isnan(rho_c):
            f.write(f"rho(G, pIR/pV) C-complex: {rho_c:+.4f}  p={p_c:.3e}  "
                    f"n={len(c_pir):,}\n")
        for zone, z in zone_results.items():
            f.write(f"{zone}: G_med={z.median():.4f}  n={len(z):,}\n")
    print(f"  Log  → logs/46_ccomplex_hydration_stats.txt\n")


if __name__ == "__main__":
    main()
