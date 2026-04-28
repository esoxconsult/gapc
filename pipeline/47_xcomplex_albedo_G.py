"""
47_xcomplex_albedo_G.py
GAPC — E/M/P subtype G × albedo × size analysis.

Step 44 found E=0.189, M=0.151, S=0.145 — all "bright" types with similar G
despite very different albedos (E: p_V>0.30, M: 0.10–0.30, S: 0.15 typical).
Step 37 showed size dominates over albedo for S-types.

Questions:
  1. Do E, M, P each show the same size-dominated G signal as S?
     (partial rho(G, logD | logpV) for E, M, P separately)
  2. Within X-complex subtypes: does p_V correlate with G once size is controlled?
  3. How does the M-type (metallic) G compare to S at same size?
     (metallic meteorites have different surface roughness → different G expected)
  4. E vs S: both high-albedo, but E achondrites vs S olivine/pyroxene.
     Does composition matter beyond albedo?

Outputs:
  plots/47_xcomplex_albedo_G.png
  logs/47_xcomplex_albedo_G_stats.txt
  (v8 NOT modified — read-only)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr, pearsonr
from scipy.stats import rankdata

ROOT    = Path(__file__).resolve().parents[1]
V8_PATH = ROOT / "data" / "final" / "gapc_catalog_v8.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"


def partial_spearman(x, y, z):
    """Partial Spearman rho(x, y | z)."""
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    bx = np.cov(xr, zr)[0, 1] / np.var(zr)
    by = np.cov(yr, zr)[0, 1] / np.var(zr)
    return pearsonr(xr - bx * zr, yr - by * zr)


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 47 — E/M/P G × albedo × size analysis")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V8_PATH.exists():
        print(f"\n  ERROR: {V8_PATH} not found"); return

    gapc = pd.read_parquet(V8_PATH)
    print(f"\n  v8 loaded: {len(gapc):,} rows, {len(gapc.columns)} cols")

    tax = "taxonomy_refined"
    pv_col = "p_V_final"
    has_pv = pv_col in gapc.columns
    results = {}

    # ── 1. G by subtype: basic comparison ─────────────────────────────────────
    print(f"\n  1. G / albedo / size by X-complex subtype:")
    subtypes = ["E", "M", "P", "S", "C"]  # include S, C as reference
    for t in subtypes:
        sub = gapc[(gapc[tax] == t) & gapc["G"].notna()]
        pv  = sub[pv_col].dropna() if has_pv else pd.Series(dtype=float)
        dk  = sub["D_km"].dropna() if "D_km" in gapc.columns else pd.Series(dtype=float)
        print(f"    {t}: G={sub['G'].median():.4f}  "
              f"pV_med={pv.median():.3f}  "
              f"D_med={dk.median():.1f} km  n={len(sub):,}")
        results[t] = dict(
            G_med=sub["G"].median(), pV_med=pv.median() if len(pv) > 0 else np.nan,
            D_med=dk.median() if len(dk) > 0 else np.nan, n=len(sub)
        )

    # ── 2. Partial correlations per subtype ───────────────────────────────────
    print(f"\n  2. Partial rho(G, logD | logpV) by subtype:")
    partial_results = {}
    for t in ["E", "M", "P", "S"]:
        sub = gapc[(gapc[tax] == t) & gapc["G"].notna() &
                   gapc["D_km"].notna() & (gapc["D_km"] > 0)]
        if not has_pv:
            continue
        sub = sub[sub[pv_col].notna() & (sub[pv_col] > 0)].copy()
        if len(sub) < 20:
            print(f"    {t}: too few (n={len(sub)})")
            continue
        sub["log_D"] = np.log10(sub["D_km"])
        sub["log_pV"] = np.log10(sub[pv_col])
        r_GD_pv, p_GD_pv = partial_spearman(sub["G"], sub["log_D"], sub["log_pV"])
        r_Gpv_D, p_Gpv_D = partial_spearman(sub["G"], sub["log_pV"], sub["log_D"])
        rho_D, p_rD = spearmanr(sub["G"], sub["log_D"])
        rho_pv, p_rpv = spearmanr(sub["G"], sub["log_pV"])
        print(f"    {t} (n={len(sub):,}):")
        print(f"      rho(G,logD)             = {rho_D:+.4f}  p={p_rD:.2e}")
        print(f"      partial r(G,logD|logpV) = {r_GD_pv:+.4f}  p={p_GD_pv:.2e}")
        print(f"      rho(G,logpV)            = {rho_pv:+.4f}  p={p_rpv:.2e}")
        print(f"      partial r(G,logpV|logD) = {r_Gpv_D:+.4f}  p={p_Gpv_D:.2e}")
        partial_results[t] = dict(
            n=len(sub), rho_D=rho_D, r_GD_pv=r_GD_pv, p_GD_pv=p_GD_pv,
            rho_pv=rho_pv, r_Gpv_D=r_Gpv_D, p_Gpv_D=p_Gpv_D
        )

    # ── 3. E vs S at fixed size ───────────────────────────────────────────────
    print(f"\n  3. E vs S at fixed size bins:")
    e_g = gapc[(gapc[tax] == "E") & gapc["G"].notna() & gapc["D_km"].notna()]
    s_g = gapc[(gapc[tax] == "S") & gapc["G"].notna() & gapc["D_km"].notna()]
    size_bins = [(1, 5), (5, 20), (20, 100)]
    bin_lbls  = ["1–5 km", "5–20 km", "20–100 km"]
    for (lo, hi), lbl in zip(size_bins, bin_lbls):
        e_b = e_g[e_g["D_km"].between(lo, hi)]["G"]
        s_b = s_g[s_g["D_km"].between(lo, hi)]["G"]
        if len(e_b) < 3 or len(s_b) < 10:
            continue
        U_es, p_es = mannwhitneyu(e_b, s_b, alternative="two-sided")
        print(f"    {lbl}: E n={len(e_b)} G={e_b.median():.4f}  "
              f"S n={len(s_b):,} G={s_b.median():.4f}  p={p_es:.3e}")

    # ── 4. M vs S at fixed size ───────────────────────────────────────────────
    print(f"\n  4. M vs S at fixed size bins:")
    m_g = gapc[(gapc[tax] == "M") & gapc["G"].notna() & gapc["D_km"].notna()]
    for (lo, hi), lbl in zip(size_bins, bin_lbls):
        m_b = m_g[m_g["D_km"].between(lo, hi)]["G"]
        s_b = s_g[s_g["D_km"].between(lo, hi)]["G"]
        if len(m_b) < 5 or len(s_b) < 10:
            continue
        U_ms, p_ms = mannwhitneyu(m_b, s_b, alternative="two-sided")
        print(f"    {lbl}: M n={len(m_b):,} G={m_b.median():.4f}  "
              f"S n={len(s_b):,} G={s_b.median():.4f}  p={p_ms:.3e}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("E/M/P subtype G × albedo × size", fontsize=12)

    # G vs pV scatter by subtype
    ax = axes[0, 0]
    colors_t = {"E": "#e74c3c", "M": "#8e44ad", "P": "#27ae60",
                "S": "steelblue", "C": "gray"}
    for t in ["S", "C", "E", "M", "P"]:
        if not has_pv:
            break
        sub = gapc[(gapc[tax] == t) & gapc["G"].notna() & gapc[pv_col].notna()]
        smp = sub.sample(min(3000, len(sub)), random_state=42)
        size = 8 if t in ("E", "M", "P") else 2
        alpha = 0.7 if t in ("E", "M", "P") else 0.1
        ax.scatter(smp[pv_col], smp["G"], s=size, alpha=alpha,
                   color=colors_t[t], rasterized=True, label=f"{t} (n={len(sub):,})")
    ax.set_xlabel("p_V (NEOWISE albedo)")
    ax.set_ylabel("G (phase slope)")
    ax.set_title("G vs albedo by subtype")
    ax.legend(fontsize=8, markerscale=3)
    ax.grid(alpha=0.2)

    # Partial rho comparison bar chart
    ax = axes[0, 1]
    if partial_results:
        taxa_p = list(partial_results.keys())
        r_D  = [partial_results[t]["r_GD_pv"] for t in taxa_p]
        r_pv = [partial_results[t]["r_Gpv_D"] for t in taxa_p]
        x = np.arange(len(taxa_p))
        w = 0.35
        ax.bar(x - w/2, r_D,  w, label="r(G,logD|logpV)",  color="#3498db", alpha=0.85)
        ax.bar(x + w/2, r_pv, w, label="r(G,logpV|logD)", color="#e74c3c", alpha=0.85)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{t}\n(n={partial_results[t]['n']:,})" for t in taxa_p])
        ax.set_ylabel("Partial Spearman r")
        ax.set_title("Partial r: size vs albedo driving G")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2, axis="y")

    # G vs D_km for E, M, P, S
    ax = axes[1, 0]
    for t in ["S", "M", "E", "P"]:
        sub = gapc[(gapc[tax] == t) & gapc["G"].notna() & gapc["D_km"].notna()].copy()
        sub = sub[sub["D_km"] > 0]
        if len(sub) < 5:
            continue
        smp = sub.sample(min(5000, len(sub)), random_state=42)
        size = 6 if t in ("E", "P") else 2
        alpha = 0.6 if t in ("E", "P") else 0.15
        ax.scatter(smp["D_km"], smp["G"], s=size, alpha=alpha,
                   color=colors_t[t], rasterized=True, label=f"{t} (n={len(sub):,})")
    ax.set_xscale("log")
    ax.set_xlabel("D [km]")
    ax.set_ylabel("G")
    ax.set_title("G vs size by X-complex subtype")
    ax.legend(fontsize=8, markerscale=4)
    ax.grid(alpha=0.2)

    # Median G by subtype bar chart
    ax = axes[1, 1]
    taxa_all = [t for t in ["E", "M", "P", "S", "C"] if t in results]
    g_meds = [results[t]["G_med"] for t in taxa_all]
    pv_meds = [results[t]["pV_med"] for t in taxa_all]
    x = np.arange(len(taxa_all))
    ax2r = ax.twinx()
    bars = ax.bar(x, g_meds, color=[colors_t[t] for t in taxa_all], alpha=0.8)
    ax2r.plot(x, pv_meds, "D--", color="black", ms=7, label="Median p_V")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n(n={results[t]['n']:,})" for t in taxa_all], fontsize=9)
    ax.set_ylabel("Median G")
    ax2r.set_ylabel("Median p_V")
    ax.set_title("Median G and p_V by subtype")
    ax2r.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "47_xcomplex_albedo_G.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/47_xcomplex_albedo_G.png")

    with open(LOG_DIR / "47_xcomplex_albedo_G_stats.txt", "w") as f:
        f.write("GAPC Step 47 — E/M/P G × albedo × size\n")
        f.write("=" * 60 + "\n")
        for t, r in results.items():
            f.write(f"{t}: G_med={r['G_med']:.4f}  pV={r['pV_med']:.3f}  "
                    f"D={r['D_med']:.1f} km  n={r['n']:,}\n")
        f.write("\nPartial correlations:\n")
        for t, r in partial_results.items():
            f.write(f"  {t}: r(G,logD|logpV)={r['r_GD_pv']:+.4f}(p={r['p_GD_pv']:.2e})  "
                    f"r(G,logpV|logD)={r['r_Gpv_D']:+.4f}  n={r['n']:,}\n")
    print(f"  Log  → logs/47_xcomplex_albedo_G_stats.txt\n")


if __name__ == "__main__":
    main()
