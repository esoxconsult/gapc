"""
52_fast_rotators_G.py
GAPC — G × rotation rate: fast vs slow rotators, size-controlled.

YORP theory: fast rotators (P < 4h) near the spin barrier shed surface
material → fresher, less weathered surfaces → higher G expected.
But fast rotators tend to be small (monolithic), and small objects
already have higher G (size law). We control for size.

Tests:
  1. rho(G, logP) — does spin rate correlate with G beyond step 38?
  2. Partial r(G, logP | logD) — spin signal after size control
  3. Mann-Whitney: G(fast) vs G(slow) in narrow size bins (3-10 km)
  4. Spin barrier objects (P < 2.2h, rubble-pile limit) separately

Outputs:
  plots/52_fast_rotators_G.png
  logs/52_fast_rotators_stats.txt
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from scipy.stats import rankdata

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
V8_PATH  = ROOT / "data" / "final" / "gapc_catalog_v8.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

PLOT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SPIN_BARRIER = 2.2   # h — rubble-pile spin barrier
FAST_CUT     = 4.0   # h — "fast" rotator threshold
SLOW_CUT     = 20.0  # h — "slow" rotator threshold


def partial_spearman(x, y, z):
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    bx = np.cov(xr, zr)[0, 1] / np.var(zr)
    by = np.cov(yr, zr)[0, 1] / np.var(zr)
    return pearsonr(xr - bx * zr, yr - by * zr)[0]


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 52 — Fast rotators × G (size-controlled)")
    print("=" * 65)

    gapc = pd.read_parquet(V8_PATH)
    print(f"\n  v8: {len(gapc):,} rows")

    # ── Working sample ────────────────────────────────────────────────────────
    base = gapc[
        gapc["G"].notna() &
        gapc["rot_period_best"].notna() &
        gapc["D_km"].notna() & (gapc["D_km"] > 0)
    ].copy()
    base["logP"] = np.log10(base["rot_period_best"])
    base["logD"] = np.log10(base["D_km"])
    print(f"  Working sample (G + P + D): {len(base):,}")

    # ── 1. Global rho(G, logP) ────────────────────────────────────────────────
    rho_gp, p_gp = spearmanr(base["G"], base["logP"])
    print(f"\n  1. rho(G, logP) = {rho_gp:+.4f}  p={p_gp:.2e}  n={len(base):,}")

    # ── 2. Partial r(G, logP | logD) ─────────────────────────────────────────
    r_part = partial_spearman(base["G"].values,
                              base["logP"].values,
                              base["logD"].values)
    print(f"  2. r(G, logP | logD) = {r_part:+.4f}")
    print(f"     r(G, logD | logP) = "
          f"{partial_spearman(base['G'].values, base['logD'].values, base['logP'].values):+.4f}")

    # ── 3. Fast vs slow, size-controlled ─────────────────────────────────────
    fast = base[base["rot_period_best"] < FAST_CUT]
    slow = base[base["rot_period_best"] > SLOW_CUT]
    mid  = base[(base["rot_period_best"] >= FAST_CUT) &
                (base["rot_period_best"] <= SLOW_CUT)]
    print(f"\n  3. Fast (<{FAST_CUT}h): n={len(fast):,}  "
          f"G_med={fast['G'].median():.4f}  D_med={fast['D_km'].median():.1f} km")
    print(f"     Mid : n={len(mid):,}   G_med={mid['G'].median():.4f}  "
          f"D_med={mid['D_km'].median():.1f} km")
    print(f"     Slow (>{SLOW_CUT}h): n={len(slow):,}  "
          f"G_med={slow['G'].median():.4f}  D_med={slow['D_km'].median():.1f} km")

    # Raw MW test
    if len(fast) >= 10 and len(slow) >= 10:
        U_raw, p_raw = mannwhitneyu(fast["G"], slow["G"], alternative="greater")
        print(f"     MW raw G(fast > slow): p={p_raw:.3e}")

    # Size-controlled: residuals from G ~ logD regression (rank-based)
    # Compute rank residuals of G after removing logD effect
    lD_all = base["logD"].values
    G_all  = base["G"].values
    Gr     = rankdata(G_all); Dr = rankdata(lD_all)
    b_GD   = np.cov(Gr, Dr)[0, 1] / np.var(Dr)
    G_resid = Gr - b_GD * Dr  # rank residual

    base = base.copy()
    base["G_resid"] = G_resid

    fast_r = base.loc[base["rot_period_best"] < FAST_CUT, "G_resid"]
    slow_r = base.loc[base["rot_period_best"] > SLOW_CUT, "G_resid"]
    if len(fast_r) >= 10 and len(slow_r) >= 10:
        U_ctrl, p_ctrl = mannwhitneyu(fast_r, slow_r, alternative="greater")
        print(f"     MW size-controlled G(fast > slow): p={p_ctrl:.3e}")
        print(f"     G_resid: fast={fast_r.mean():.3f}  slow={slow_r.mean():.3f}")
    else:
        p_ctrl = np.nan

    # ── 4. Narrow size bins ───────────────────────────────────────────────────
    print(f"\n  4. Size bins — fast vs slow G:")
    size_bins = [(1, 3, "1-3 km"), (3, 10, "3-10 km"),
                 (10, 30, "10-30 km"), (30, 100, "30-100 km")]
    bin_results = []
    for d_lo, d_hi, label in size_bins:
        sub = base[(base["D_km"] >= d_lo) & (base["D_km"] < d_hi)]
        sf = sub[sub["rot_period_best"] < FAST_CUT]
        ss = sub[sub["rot_period_best"] > SLOW_CUT]
        if len(sf) >= 5 and len(ss) >= 5:
            U, p_b = mannwhitneyu(sf["G"], ss["G"], alternative="greater")
            print(f"    {label:10s}: fast n={len(sf):4,} G={sf['G'].median():.4f}  "
                  f"slow n={len(ss):4,} G={ss['G'].median():.4f}  p={p_b:.3e}")
            bin_results.append((label, d_lo, d_hi, len(sf), len(ss),
                                sf["G"].median(), ss["G"].median(), p_b))
        else:
            print(f"    {label:10s}: too few (fast={len(sf)}, slow={len(ss)})")

    # ── 5. Spin barrier (monolithic) ─────────────────────────────────────────
    barrier = base[base["rot_period_best"] < SPIN_BARRIER]
    normal  = base[(base["rot_period_best"] >= SPIN_BARRIER) &
                   (base["rot_period_best"] < FAST_CUT)]
    g_bar = barrier["G"].median() if len(barrier) > 0 else np.nan
    d_bar = barrier["D_km"].median() if len(barrier) > 0 else np.nan
    g_nor = normal["G"].median() if len(normal) > 0 else np.nan
    print(f"\n  5. Spin barrier (<{SPIN_BARRIER}h): n={len(barrier):,}  "
          f"G_med={g_bar:.4f}  D_med={d_bar:.1f} km")
    print(f"     Fast non-barrier ({SPIN_BARRIER}-{FAST_CUT}h): n={len(normal):,}  "
          f"G_med={g_nor:.4f}")

    # ── By taxonomy ───────────────────────────────────────────────────────────
    print(f"\n  6. Partial r(G, logP | logD) by taxonomy:")
    for t in ["S", "C", "M"]:
        sub = base[base["taxonomy_refined"] == t]
        if len(sub) < 100:
            continue
        r = partial_spearman(sub["G"].values, sub["logP"].values, sub["logD"].values)
        rho_t, p_t = spearmanr(sub["G"], sub["logP"])
        print(f"    {t}: partial r={r:+.4f}  raw rho={rho_t:+.4f}  p={p_t:.2e}  n={len(sub):,}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("G × rotation period (fast vs slow rotators, size-controlled)",
                 fontsize=11)

    # G vs logP scatter
    ax = axes[0, 0]
    smp = base.sample(min(15000, len(base)), random_state=42)
    sc = ax.scatter(smp["rot_period_best"], smp["G"], s=2, alpha=0.15,
                    c=np.log10(smp["D_km"]), cmap="viridis", rasterized=True)
    plt.colorbar(sc, ax=ax, label="log D [km]")
    ax.set_xscale("log")
    ax.axvline(SPIN_BARRIER, color="red", lw=1, ls="--", alpha=0.7,
               label=f"Spin barrier {SPIN_BARRIER}h")
    ax.axvline(FAST_CUT, color="orange", lw=1, ls="--", alpha=0.7,
               label=f"Fast cut {FAST_CUT}h")
    ax.set_xlabel("Rotation period [h]")
    ax.set_ylabel("G")
    ax.set_title(f"G vs P  rho={rho_gp:+.3f}  partial r={r_part:+.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    # G distributions fast/slow
    ax = axes[0, 1]
    g_rng = (base["G"].quantile(0.01), base["G"].quantile(0.99))
    bins_g = np.linspace(*g_rng, 50)
    for grp, lbl, col in [
        (fast["G"], f"Fast <{FAST_CUT}h (n={len(fast):,})", "#d62728"),
        (slow["G"], f"Slow >{SLOW_CUT}h (n={len(slow):,})", "#1f77b4"),
    ]:
        ax.hist(grp.clip(*g_rng), bins=bins_g, density=True,
                histtype="step", lw=2, color=col, label=lbl)
    ax.set_xlabel("G"); ax.set_ylabel("Density")
    ax.set_title(f"G: fast vs slow\nMW size-controlled p={p_ctrl:.2e}"
                 if not np.isnan(p_ctrl) else "G: fast vs slow")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # G vs logD colored by fast/slow
    ax = axes[1, 0]
    for grp, lbl, col, size in [
        (fast, f"Fast <{FAST_CUT}h", "#d62728", 4),
        (slow, f"Slow >{SLOW_CUT}h", "#1f77b4", 2),
    ]:
        smp2 = grp.sample(min(3000, len(grp)), random_state=42)
        ax.scatter(smp2["D_km"], smp2["G"], s=size, alpha=0.3,
                   color=col, label=lbl, rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("D [km]"); ax.set_ylabel("G")
    ax.set_title("G vs D: fast (red) vs slow (blue)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # Size bin comparison
    ax = axes[1, 1]
    if bin_results:
        labels_b = [r[0] for r in bin_results]
        g_fast_b = [r[5] for r in bin_results]
        g_slow_b = [r[6] for r in bin_results]
        ps_b     = [r[7] for r in bin_results]
        x = np.arange(len(labels_b))
        w = 0.35
        bars_f = ax.bar(x - w/2, g_fast_b, w, label=f"Fast <{FAST_CUT}h",
                        color="#d62728", alpha=0.8)
        bars_s = ax.bar(x + w/2, g_slow_b, w, label=f"Slow >{SLOW_CUT}h",
                        color="#1f77b4", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(labels_b)
        ax.set_ylabel("Median G")
        ax.set_title("Median G by size bin (fast vs slow)")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2, axis="y")
        for xi, p_b in zip(x, ps_b):
            stars = "***" if p_b < 0.001 else ("**" if p_b < 0.01 else ("*" if p_b < 0.05 else "ns"))
            ax.text(xi, max(g_fast_b[x.tolist().index(xi)],
                            g_slow_b[x.tolist().index(xi)]) + 0.003,
                    stars, ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "52_fast_rotators_G.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/52_fast_rotators_G.png")

    with open(LOG_DIR / "52_fast_rotators_stats.txt", "w") as f:
        f.write("GAPC Step 52 — Fast rotators × G\n")
        f.write("=" * 60 + "\n")
        f.write(f"n total with G+P+D: {len(base):,}\n")
        f.write(f"rho(G, logP) = {rho_gp:+.4f}  p={p_gp:.3e}\n")
        f.write(f"r(G, logP | logD) = {r_part:+.4f}\n")
        f.write(f"MW size-controlled G(fast > slow): p={p_ctrl:.3e}\n\n")
        f.write("Size bins:\n")
        for br in bin_results:
            f.write(f"  {br[0]:10s}: fast G={br[5]:.4f}(n={br[3]})  "
                    f"slow G={br[6]:.4f}(n={br[4]})  p={br[7]:.3e}\n")
    print(f"  Log  → logs/52_fast_rotators_stats.txt\n")


if __name__ == "__main__":
    main()
