"""
32_rotation_periods.py
GAPC — Integrate rotation periods from Durech+2022 (J/A+A/675/A24).

Durech, Tonry & Carry (2022) published spin pole solutions for 8,596
asteroids from combined lightcurve + Gaia/Pan-STARRS sparse data.
Methods: CE (combined epochs), C (convex inversion), E (ellipsoid model).

8,337 of these overlap with GAPC. Rotation periods add a new physical
dimension:
  - Size-period relation (YORP effect: small asteroids spin faster)
  - G × period: do rapidly rotating asteroids show higher G?
    (Fast rotators → younger surface → less weathering → higher G)
  - Variability flag validation: known rapid rotators should be flagged

New columns added to gapc_catalog_v5:
  rot_period_h    — sidereal rotation period [hours]
  rot_pole_lambda — ecliptic longitude of spin pole [deg]
  rot_pole_beta   — ecliptic latitude of spin pole [deg]
  rot_method      — CE / C / E

Outputs:
  data/final/gapc_catalog_v5.parquet  (updated in-place, adds 4 columns)
  plots/32_rotation_periods.png
  logs/32_rotation_periods_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu

ROOT     = Path(__file__).resolve().parents[1]
V5_PATH  = ROOT / "data" / "final" / "gapc_catalog_v5.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"
DATA_RAW = ROOT / "data" / "raw"


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 32 — Rotation periods (Durech+2022)")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V5_PATH.exists():
        print(f"\n  ERROR: {V5_PATH} not found — run step 31 first"); return
    gapc = pd.read_parquet(V5_PATH)
    print(f"\n  v5 loaded: {len(gapc):,} objects")

    # ── Load Durech+2022 ──────────────────────────────────────────────────────
    dur_path = DATA_RAW / "durech2022_spins_table0.csv"
    if not dur_path.exists():
        print(f"\n  ERROR: {dur_path} not found"); return
    dur = pd.read_csv(dur_path)
    dur["number_mp"] = pd.to_numeric(dur["Number"], errors="coerce")
    dur = dur.dropna(subset=["number_mp"]).astype({"number_mp": "int64"})
    dur["P"] = pd.to_numeric(dur["P"], errors="coerce")
    # Take first solution per asteroid (Durech lists up to 2 poles per object)
    dur_unique = dur.drop_duplicates("number_mp").copy()
    print(f"  Durech+2022 spin solutions: {len(dur):,} rows → "
          f"{len(dur_unique):,} unique asteroids")
    print(f"  Methods: {dur_unique['Method'].value_counts().to_dict()}")
    p = dur_unique["P"].dropna()
    print(f"  Period range: {p.min():.2f}–{p.max():.1f} h  median={p.median():.2f} h")

    # ── Merge ─────────────────────────────────────────────────────────────────
    dur_merge = dur_unique[["number_mp","P","lambda1","beta1","Method"]].rename(
        columns={"P":"rot_period_h","lambda1":"rot_pole_lambda",
                 "beta1":"rot_pole_beta","Method":"rot_method"})
    # Clean up
    for col in ("rot_period_h","rot_pole_lambda","rot_pole_beta"):
        dur_merge[col] = pd.to_numeric(dur_merge[col], errors="coerce")

    gapc = gapc.merge(dur_merge, on="number_mp", how="left")
    n_per = gapc["rot_period_h"].notna().sum()
    print(f"\n  Matched into GAPC: {n_per:,} ({n_per/len(gapc)*100:.1f}%)")

    # ── Size-period relationship ───────────────────────────────────────────────
    sp = gapc[gapc["rot_period_h"].notna() & gapc["D_km"].notna()].copy()
    rho_dp, p_dp = spearmanr(np.log10(sp["D_km"]), np.log10(sp["rot_period_h"]))
    print(f"\n  Spearman rho(log D, log P) n={len(sp):,}: "
          f"rho={rho_dp:+.4f}  p={p_dp:.2e}")

    # ── G × rotation period ───────────────────────────────────────────────────
    gp = gapc[gapc["rot_period_h"].notna() & gapc["G"].notna()].copy()
    rho_gp, p_gp = spearmanr(gp["G"], np.log10(gp["rot_period_h"]))
    print(f"  Spearman rho(G, log P)    n={len(gp):,}: "
          f"rho={rho_gp:+.4f}  p={p_gp:.2e}")

    # By taxonomy
    for tax in ["S","C"]:
        sub = gp[gp["predicted_taxonomy"] == tax] if "predicted_taxonomy" in gp.columns else pd.DataFrame()
        if len(sub) < 20:
            continue
        r, p_ = spearmanr(sub["G"], np.log10(sub["rot_period_h"]))
        print(f"    {tax}: rho={r:+.4f}  p={p_:.2e}  n={len(sub):,}")

    # Fast rotators (P < 4h) vs slow (P > 10h) — G comparison
    fast = gapc[gapc["rot_period_h"] < 4]["G"].dropna()
    slow = gapc[gapc["rot_period_h"] > 10]["G"].dropna()
    if len(fast) >= 10 and len(slow) >= 10:
        U, p_mw = mannwhitneyu(fast, slow, alternative="greater")
        print(f"\n  G: fast rotators (P<4h, n={len(fast)}) vs slow (P>10h, n={len(slow)})")
        print(f"  median G fast={fast.median():.4f}  slow={slow.median():.4f}")
        print(f"  Mann-Whitney G(fast)>G(slow): p={p_mw:.3e}")
    else:
        p_mw = np.nan

    # ── Variability flag validation ───────────────────────────────────────────
    if "var_flag" in gapc.columns:
        has_per = gapc["rot_period_h"].notna()
        flag_rate_with = gapc.loc[has_per, "var_flag"].mean() * 100
        flag_rate_wo   = gapc.loc[~has_per, "var_flag"].mean() * 100
        print(f"\n  var_flag rate WITH period: {flag_rate_with:.1f}%")
        print(f"  var_flag rate WITHOUT period: {flag_rate_wo:.1f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Rotation periods (Durech+2022)  n={n_per:,}", fontsize=13)

    # Period histogram
    ax = axes[0, 0]
    p_vals = gapc["rot_period_h"].dropna()
    ax.hist(p_vals[p_vals < 100].values, bins=80, color="steelblue",
            edgecolor="none", alpha=0.8)
    ax.axvline(p_vals.median(), color="red", lw=1.5,
               label=f"median={p_vals.median():.2f} h")
    ax.set_xlabel("Rotation period [h]"); ax.set_ylabel("Count")
    ax.set_title("Period distribution (P < 100 h)")
    ax.legend(fontsize=9)

    # Size-period
    ax = axes[0, 1]
    sp_plot = sp[sp["D_km"].between(0.1, 500) & sp["rot_period_h"].between(1, 1000)]
    ax.scatter(sp_plot["D_km"], sp_plot["rot_period_h"],
               s=3, alpha=0.2, color="steelblue", rasterized=True)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("D [km]"); ax.set_ylabel("Period [h]")
    ax.set_title(f"Size-period  rho(logD,logP)={rho_dp:+.3f}")
    ax.grid(alpha=0.2)

    # G vs period
    ax = axes[1, 0]
    gp_plot = gp[gp["rot_period_h"].between(1, 1000) & gp["G"].between(0, 1)]
    ax.scatter(gp_plot["rot_period_h"], gp_plot["G"],
               s=3, alpha=0.2, color="coral", rasterized=True)
    ax.set_xscale("log")
    ax.set_xlabel("Period [h]"); ax.set_ylabel("G (phase slope)")
    ax.set_title(f"G vs period  rho(G,logP)={rho_gp:+.3f}  p={p_gp:.1e}")
    ax.grid(alpha=0.2)

    # Fast vs slow G boxplot
    ax = axes[1, 1]
    bins_labels = ["P<4h", "4–10h", ">10h"]
    grp = [
        gapc[gapc["rot_period_h"] < 4]["G"].dropna().values,
        gapc[gapc["rot_period_h"].between(4, 10)]["G"].dropna().values,
        gapc[gapc["rot_period_h"] > 10]["G"].dropna().values,
    ]
    grp_nonempty = [(l, g) for l, g in zip(bins_labels, grp) if len(g) > 0]
    if grp_nonempty:
        bp = ax.boxplot([g for _, g in grp_nonempty],
                        tick_labels=[l for l, _ in grp_nonempty],
                        patch_artist=True, showfliers=False,
                        medianprops={"lw": 2, "color": "red"})
        for patch in bp["boxes"]:
            patch.set_facecolor("steelblue"); patch.set_alpha(0.6)
        ax.set_ylabel("G (phase slope)")
        ax.set_title("G by rotation speed")
        ax.grid(alpha=0.3, axis="y")
        for i, (lbl, g) in enumerate(grp_nonempty):
            ax.text(i+1, ax.get_ylim()[1]*0.95, f"n={len(g):,}",
                    ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "32_rotation_periods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/32_rotation_periods.png")

    # ── Save updated v5 ───────────────────────────────────────────────────────
    gapc.to_parquet(V5_PATH, index=False)
    print(f"  Updated v5: {len(gapc.columns)} cols")

    with open(LOG_DIR / "32_rotation_periods_stats.txt", "w") as f:
        f.write("GAPC Step 32 — Rotation periods (Durech+2022)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Durech+2022 unique asteroids: {len(dur_unique):,}\n")
        f.write(f"Matched into GAPC: {n_per:,} ({n_per/len(gapc)*100:.1f}%)\n")
        f.write(f"Period median: {p.median():.2f} h\n")
        f.write(f"Spearman rho(log D, log P): {rho_dp:+.4f}  p={p_dp:.2e}\n")
        f.write(f"Spearman rho(G, log P):     {rho_gp:+.4f}  p={p_gp:.2e}\n")
        if len(fast) >= 10 and len(slow) >= 10:
            f.write(f"G fast (P<4h) median: {fast.median():.4f}  n={len(fast)}\n")
            f.write(f"G slow (P>10h) median: {slow.median():.4f}  n={len(slow)}\n")
            f.write(f"Mann-Whitney G(fast)>G(slow): p={p_mw:.3e}\n")
    print(f"  Log  → logs/32_rotation_periods_stats.txt\n")


if __name__ == "__main__":
    main()
