"""
21b_g_vs_size_controlled.py
GAPC — G vs size with compositional gradient confound controlled.

Step 21 found rho(G, D_km) = -0.144 (p≈0) — larger objects have lower G.
But this could be spurious: large objects preferentially live in the outer belt
(C-type, intrinsically low G), while small objects populate the inner belt
(S-type, higher G). The compositional gradient confounds the size-G signal.

Control strategy:
  1. Restrict to S-types (predicted_taxonomy == "S") within each orbital zone.
  2. Within each zone × taxonomy cell, run Spearman rho(G, D_km).
  3. If negative rho persists in each zone separately, the signal is real
     (larger S-type asteroids have lower G regardless of location → space
     weathering). If it vanishes or reverses, the signal was the gradient.

Orbital zones (Gradie & Tedesco 1982; DeMeo & Carry 2014):
  MBA-inner  : 2.0  ≤ a <  2.5  AU
  MBA-middle : 2.5  ≤ a <  2.82 AU
  MBA-outer  : 2.82 ≤ a <  3.27 AU
  Cybele     : 3.27 ≤ a <  3.7  AU (small N expected)

Additional test: within each zone, compare G for S-types vs C-types to
confirm the taxonomy offset is present even at matched size.

Outputs:
  plots/21b_g_size_controlled.png
  logs/21b_g_size_controlled_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

ROOT     = Path(__file__).resolve().parents[1]
CAT_PATH = ROOT / "data" / "final" / "gapc_catalog_v4.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"

ZONES = {
    "MBA-inner":  (2.00, 2.50),
    "MBA-middle": (2.50, 2.82),
    "MBA-outer":  (2.82, 3.27),
    "Cybele":     (3.27, 3.70),
}
ZONE_COLORS = {
    "MBA-inner":  "#e07b39",
    "MBA-middle": "#5c85d6",
    "MBA-outer":  "#4caf50",
    "Cybele":     "#9c27b0",
}

MIN_N = 30   # minimum per cell for a meaningful Spearman test


def spearman_report(df, x_col, y_col, label):
    sub = df[[x_col, y_col]].dropna()
    if len(sub) < MIN_N:
        print(f"  {label}: n={len(sub)} < {MIN_N} — skip")
        return dict(label=label, n=len(sub), rho=np.nan, pval=np.nan)
    rho, pval = spearmanr(sub[x_col], sub[y_col])
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
    print(f"  {label:30s}  n={len(sub):6,}  rho={rho:+.4f}  p={pval:.2e}  {sig}")
    return dict(label=label, n=len(sub), rho=rho, pval=pval)


def main():
    print("\n" + "=" * 70)
    print("  GAPC Step 21b — G vs Size (confound-controlled)")
    print("=" * 70)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CAT_PATH)
    print(f"  Catalog rows: {len(df):,}")

    # ── Column setup ──────────────────────────────────────────────────────────
    if "fit_ok" in df.columns:
        df = df[df["fit_ok"] & df["G"].notna()].copy()
    else:
        df = df[df["G"].notna()].copy()
    if "G_uncertain" in df.columns:
        df = df[~df["G_uncertain"].fillna(False)].copy()
    print(f"  Reliable-G rows: {len(df):,}")

    # Taxonomy: prefer predicted, fall back to GASP first letter
    if "predicted_taxonomy" in df.columns:
        tax = df["predicted_taxonomy"].copy()
    else:
        tax = pd.Series("Other", index=df.index)
    if "gasp_taxonomy_final" in df.columns:
        mask = tax.isna() | (tax == "Other")
        raw = df.loc[mask, "gasp_taxonomy_final"].str.strip().str.upper().str[0]
        tax_map = {"S": "S", "C": "C", "X": "X"}
        tax[mask] = raw.map(tax_map).fillna("Other")
    df["_tax"] = tax.fillna("Other")

    # Log diameter
    if "D_km" not in df.columns or df["D_km"].notna().sum() < 100:
        print("ERROR: D_km column missing or nearly empty — run step 13 first.")
        return
    df = df[df["D_km"] > 0].copy()
    df["log10_D"] = np.log10(df["D_km"])

    # Semi-major axis
    a_col = None
    for cand in ["a_au", "a", "semi_major_axis"]:
        if cand in df.columns:
            a_col = cand
            break
    if a_col is None:
        print("ERROR: No semi-major axis column found.")
        return
    df["_a"] = df[a_col]
    print(f"  Semi-major axis column: {a_col}")

    # Assign zone
    def assign_zone(a):
        for zname, (lo, hi) in ZONES.items():
            if lo <= a < hi:
                return zname
        return None
    df["_zone"] = df["_a"].map(assign_zone)
    zone_counts = df["_zone"].value_counts()
    print(f"\n  Asteroid counts per zone:")
    for z, n in zone_counts.items():
        print(f"    {z:12s}: {n:,}")

    # ── Panel 1: uncontrolled rho (reference) ────────────────────────────────
    print("\n--- Uncontrolled (all objects) ---")
    r_all = spearman_report(df, "log10_D", "G", "All objects  G vs log10(D)")

    # ── Panel 2: rho by zone (all taxonomy) ──────────────────────────────────
    print("\n--- Per orbital zone (all taxonomy) ---")
    zone_results_all = []
    for zname in ZONES:
        sub = df[df["_zone"] == zname]
        r = spearman_report(sub, "log10_D", "G", f"{zname} (all tax)")
        r["zone"] = zname
        r["tax"] = "all"
        zone_results_all.append(r)

    # ── Panel 3: S-type only per zone (key confound test) ────────────────────
    print("\n--- S-type per orbital zone (confound control) ---")
    zone_results_S = []
    for zname in ZONES:
        sub = df[(df["_zone"] == zname) & (df["_tax"] == "S")]
        r = spearman_report(sub, "log10_D", "G", f"{zname} S-type")
        r["zone"] = zname
        r["tax"] = "S"
        zone_results_S.append(r)

    # ── Panel 4: C-type only per zone (should show weaker signal) ────────────
    print("\n--- C-type per orbital zone ---")
    zone_results_C = []
    for zname in ZONES:
        sub = df[(df["_zone"] == zname) & (df["_tax"] == "C")]
        r = spearman_report(sub, "log10_D", "G", f"{zname} C-type")
        r["zone"] = zname
        r["tax"] = "C"
        zone_results_C.append(r)

    # ── Median G per zone × taxonomy ─────────────────────────────────────────
    print("\n--- Median G per zone × taxonomy ---")
    for zname in ZONES:
        for tax in ["S", "C", "X"]:
            sub = df[(df["_zone"] == zname) & (df["_tax"] == tax)]
            if len(sub) < 10:
                continue
            print(f"  {zname:12s} {tax}  n={len(sub):5,}  "
                  f"G={sub['G'].median():.4f}±{sub['G'].std():.4f}  "
                  f"D_med={sub['D_km'].median():.1f} km")

    # ── Interpretation ────────────────────────────────────────────────────────
    s_rhos = [r["rho"] for r in zone_results_S if np.isfinite(r.get("rho", np.nan))]
    s_pvs  = [r["pval"] for r in zone_results_S if np.isfinite(r.get("pval", np.nan))]
    n_sig_neg = sum(1 for rho, pv in zip(s_rhos, s_pvs) if rho < -0.05 and pv < 0.05)
    n_sig_pos = sum(1 for rho, pv in zip(s_rhos, s_pvs) if rho > 0.05 and pv < 0.05)
    print(f"\n  INTERPRETATION:")
    print(f"  Zones with significant negative S-type rho(G, D): {n_sig_neg}/{len(s_rhos)}")
    print(f"  Zones with significant positive S-type rho(G, D): {n_sig_pos}/{len(s_rhos)}")
    if n_sig_neg >= 2:
        print("  => REAL SIGNAL: G-D correlation persists within orbital zones → space weathering")
    elif n_sig_pos >= 2:
        print("  => COUNTER-SIGNAL: G increases with D within zones (puzzling)")
    else:
        print("  => CONFOUND CONFIRMED: G-D correlation disappears within zones → "
              "the global signal was the compositional gradient, not weathering")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle("G vs Diameter — Confound Control\n"
                 "(compositional gradient: inner=S-rich, outer=C-rich)", fontsize=13)

    # Panel A: scatter all, colored by zone
    ax = axes[0, 0]
    for zname, (lo, hi) in ZONES.items():
        sub = df[df["_zone"] == zname]
        if len(sub) < 5:
            continue
        ax.scatter(sub["log10_D"], sub["G"], s=2, alpha=0.05,
                   color=ZONE_COLORS[zname], rasterized=True)
    # Running median
    log_bins = np.arange(-0.5, 2.6, 0.25)
    lbc = (log_bins[:-1] + log_bins[1:]) / 2
    medians = []
    for lo, hi in zip(log_bins[:-1], log_bins[1:]):
        s = df[(df["log10_D"] >= lo) & (df["log10_D"] < hi)]["G"]
        medians.append(s.median() if len(s) >= 5 else np.nan)
    ax.plot(lbc, medians, "k-", lw=2, label=f"Running median  rho={r_all['rho']:+.3f}")
    ax.set_xlabel("log₁₀ D [km]"); ax.set_ylabel("G")
    ax.set_title("All objects (uncontrolled)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel B: running median per zone
    ax = axes[0, 1]
    for zname, (lo_z, hi_z) in ZONES.items():
        sub = df[df["_zone"] == zname]
        if len(sub) < MIN_N:
            continue
        r_z = next((r for r in zone_results_all if r["zone"] == zname), None)
        meds = []
        for lo, hi in zip(log_bins[:-1], log_bins[1:]):
            s = sub[(sub["log10_D"] >= lo) & (sub["log10_D"] < hi)]["G"]
            meds.append(s.median() if len(s) >= 5 else np.nan)
        rho_str = f"rho={r_z['rho']:+.3f}" if r_z and np.isfinite(r_z["rho"]) else ""
        ax.plot(lbc, meds, "-o", ms=4, color=ZONE_COLORS[zname],
                label=f"{zname}  {rho_str}", linewidth=1.5)
    ax.set_xlabel("log₁₀ D [km]"); ax.set_ylabel("Median G")
    ax.set_title("Per zone (all taxonomy)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel C: S-type per zone (confound test)
    ax = axes[1, 0]
    for zname, (lo_z, hi_z) in ZONES.items():
        sub = df[(df["_zone"] == zname) & (df["_tax"] == "S")]
        if len(sub) < MIN_N:
            continue
        r_z = next((r for r in zone_results_S if r["zone"] == zname), None)
        meds = []
        for lo, hi in zip(log_bins[:-1], log_bins[1:]):
            s = sub[(sub["log10_D"] >= lo) & (sub["log10_D"] < hi)]["G"]
            meds.append(s.median() if len(s) >= 5 else np.nan)
        rho_str = f"rho={r_z['rho']:+.3f}" if r_z and np.isfinite(r_z["rho"]) else ""
        ax.plot(lbc, meds, "-s", ms=4, color=ZONE_COLORS[zname],
                label=f"{zname}  {rho_str}", linewidth=1.5)
    ax.set_xlabel("log₁₀ D [km]"); ax.set_ylabel("Median G")
    ax.set_title("S-type per zone\n(confound-controlled space weathering test)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel D: rho bar chart — all vs S-type per zone
    ax = axes[1, 1]
    zone_names = [z for z in ZONES if any(r["zone"] == z and np.isfinite(r.get("rho", np.nan))
                                          for r in zone_results_S + zone_results_all)]
    x = np.arange(len(zone_names))
    w = 0.35
    rho_all_list = [next((r["rho"] for r in zone_results_all
                          if r["zone"] == z and np.isfinite(r.get("rho", np.nan))), 0.0)
                    for z in zone_names]
    rho_S_list   = [next((r["rho"] for r in zone_results_S
                          if r["zone"] == z and np.isfinite(r.get("rho", np.nan))), 0.0)
                    for z in zone_names]
    ax.bar(x - w/2, rho_all_list, w, label="All tax", color="#9e9e9e", alpha=0.8)
    ax.bar(x + w/2, rho_S_list,   w, label="S-type only", color="#e07b39", alpha=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(-0.1, color="gray", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(zone_names, rotation=15, ha="right")
    ax.set_ylabel("Spearman rho(G, log10D)")
    ax.set_title("rho comparison by zone\n(all-tax vs S-type only)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    out = PLOT_DIR / "21b_g_size_controlled.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → {out.relative_to(ROOT)}")

    # ── Log ───────────────────────────────────────────────────────────────────
    log_path = LOG_DIR / "21b_g_size_controlled_stats.txt"
    all_results = zone_results_all + zone_results_S + zone_results_C
    with open(log_path, "w") as f:
        f.write("GAPC Step 21b — G vs Size Confound Control\n")
        f.write("=" * 60 + "\n")
        f.write(f"Uncontrolled rho(G, log10D) = {r_all['rho']:+.4f}  p={r_all['pval']:.2e}\n\n")
        for r in all_results:
            if np.isfinite(r.get("rho", np.nan)):
                f.write(f"{r['label']:35s}  n={r['n']:6,}  rho={r['rho']:+.4f}  p={r['pval']:.2e}\n")
        f.write(f"\nInterpretation: {n_sig_neg} zones with significant negative S-type rho, "
                f"{n_sig_pos} with positive\n")
    print(f"  Log  → {log_path.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
