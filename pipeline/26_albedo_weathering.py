"""
26_albedo_weathering.py
GAPC — Triple space-weathering consistency check: G ↔ albedo ↔ diameter.

Space weathering prediction (Clark+2002, Nesvorny+2005):
  Larger S-type asteroid → older surface → more weathering →
    (1) lower geometric albedo p_V  (surface darkened)
    (2) redder spectrum / smaller phase slope G  (microcrater scattering)
    (3) correlation: p_V ↔ G ↔ D_km  (all three linked)

Step 21b showed G vs D_km signal persists within orbital zones (real).
This step tests whether NEOWISE albedo follows the same pattern within zones,
and whether G and albedo correlate independently of size.

If all three Spearman tests are significant within the same zone:
  rho(D, G) < 0, rho(D, p_V) < 0, rho(G, p_V) > 0
the weathering narrative is internally consistent across three independent
observables (two from Gaia phase curves, one from WISE thermal emission).

Outputs:
  plots/26_albedo_weathering.png
  logs/26_albedo_weathering_stats.txt
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
}
ZONE_COLORS = {
    "MBA-inner":  "#e07b39",
    "MBA-middle": "#5c85d6",
    "MBA-outer":  "#4caf50",
}
MIN_N = 20


def srho(x, y, label=""):
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < MIN_N:
        return np.nan, np.nan, mask.sum()
    r, p = spearmanr(x[mask], y[mask])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"    {label:35s}  n={mask.sum():5,}  rho={r:+.4f}  p={p:.2e}  {sig}")
    return r, p, mask.sum()


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 26 — Albedo × G × Diameter: Space Weathering Check")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CAT_PATH)
    print(f"  Catalog: {len(df):,} objects")

    # Quality filter
    if "fit_ok" in df.columns:
        df = df[df["fit_ok"]].copy()
    if "G_uncertain" in df.columns:
        df = df[~df["G_uncertain"].fillna(False)].copy()
    df = df[df["G"].notna()].copy()
    print(f"  Reliable-G: {len(df):,}")

    # NEOWISE albedo subset
    neo = df[df["gasp_albedo"].notna() & df["gasp_diameter_km"].notna() &
             (df["gasp_albedo"] > 0) & (df["gasp_diameter_km"] > 0)].copy()
    neo["log_D"]   = np.log10(neo["gasp_diameter_km"])
    neo["log_pV"]  = np.log10(neo["gasp_albedo"].clip(1e-3, 1.5))
    print(f"  With NEOWISE albedo+diam: {len(neo):,}")

    # Taxonomy
    if "predicted_taxonomy" in neo.columns:
        neo["_tax"] = neo["predicted_taxonomy"].fillna("Other")
    else:
        neo["_tax"] = "Other"
    if "gasp_taxonomy_final" in neo.columns:
        m = neo["_tax"] == "Other"
        raw = neo.loc[m, "gasp_taxonomy_final"].str.strip().str.upper().str[0]
        neo.loc[m, "_tax"] = raw.map({"S": "S", "C": "C", "X": "X"}).fillna("Other")

    # Assign zone
    def zone_of(a):
        for z, (lo, hi) in ZONES.items():
            if lo <= a < hi:
                return z
        return None
    neo["_zone"] = neo["a_au"].map(zone_of)

    # ── Summary: NEOWISE objects per zone × taxonomy ──────────────────────────
    print(f"\n  NEOWISE objects per zone × taxonomy:")
    for z in ZONES:
        for tax in ["S", "C", "X"]:
            n = ((neo["_zone"] == z) & (neo["_tax"] == tax)).sum()
            if n > 0:
                print(f"    {z:12s} {tax}  n={n:4,}")

    # ── Triple correlation tests ──────────────────────────────────────────────
    log_rows = []

    # All objects (uncontrolled)
    print(f"\n--- Uncontrolled (all zones, all taxonomy) ---")
    r1, p1, n1 = srho(neo["log_D"].values, neo["G"].values,       "log D vs G      (expect −)")
    r2, p2, n2 = srho(neo["log_D"].values, neo["log_pV"].values,  "log D vs log pV (expect −)")
    r3, p3, n3 = srho(neo["log_pV"].values, neo["G"].values,      "log pV vs G     (expect +)")
    log_rows.append(dict(zone="all", tax="all", pair="D-G",   n=n1, rho=r1, p=p1))
    log_rows.append(dict(zone="all", tax="all", pair="D-pV",  n=n2, rho=r2, p=p2))
    log_rows.append(dict(zone="all", tax="all", pair="pV-G",  n=n3, rho=r3, p=p3))

    # Per zone, S-type only (confound controlled)
    print(f"\n--- S-type per zone (confound-controlled) ---")
    for z in ZONES:
        sub = neo[(neo["_zone"] == z) & (neo["_tax"] == "S")]
        print(f"  {z} (n={len(sub)}):")
        ra, pa, na = srho(sub["log_D"].values, sub["G"].values,      "  log D vs G      (expect −)")
        rb, pb, nb = srho(sub["log_D"].values, sub["log_pV"].values, "  log D vs log pV (expect −)")
        rc, pc, nc = srho(sub["log_pV"].values, sub["G"].values,     "  log pV vs G     (expect +)")
        log_rows.append(dict(zone=z, tax="S", pair="D-G",   n=na, rho=ra, p=pa))
        log_rows.append(dict(zone=z, tax="S", pair="D-pV",  n=nb, rho=rb, p=pb))
        log_rows.append(dict(zone=z, tax="S", pair="pV-G",  n=nc, rho=rc, p=pc))

    # C-type comparison (should be weaker or opposite for G vs pV)
    print(f"\n--- C-type per zone (comparison) ---")
    for z in ZONES:
        sub = neo[(neo["_zone"] == z) & (neo["_tax"] == "C")]
        if len(sub) < MIN_N:
            print(f"  {z} C-type: n={len(sub)} < {MIN_N} — skip")
            continue
        print(f"  {z} C-type (n={len(sub)}):")
        ra, pa, na = srho(sub["log_D"].values, sub["G"].values,      "  log D vs G      (expect weaker)")
        rb, pb, nb = srho(sub["log_D"].values, sub["log_pV"].values, "  log D vs log pV")
        rc, pc, nc = srho(sub["log_pV"].values, sub["G"].values,     "  log pV vs G")
        log_rows.append(dict(zone=z, tax="C", pair="D-G",   n=na, rho=ra, p=pa))
        log_rows.append(dict(zone=z, tax="C", pair="D-pV",  n=nb, rho=rb, p=pb))
        log_rows.append(dict(zone=z, tax="C", pair="pV-G",  n=nc, rho=rc, p=pc))

    # ── Median albedo per zone × taxonomy ────────────────────────────────────
    print(f"\n  Median albedo and G by zone × taxonomy:")
    med_rows = []
    for z in ZONES:
        for tax in ["S", "C", "X"]:
            sub = neo[(neo["_zone"] == z) & (neo["_tax"] == tax)]
            if len(sub) < 5:
                continue
            med_rows.append(dict(zone=z, tax=tax, n=len(sub),
                                 pV_med=sub["gasp_albedo"].median(),
                                 G_med=sub["G"].median(),
                                 D_med=sub["gasp_diameter_km"].median()))
            print(f"    {z:12s} {tax}  n={len(sub):4,}  "
                  f"pV={sub['gasp_albedo'].median():.3f}  "
                  f"G={sub['G'].median():.4f}  "
                  f"D_med={sub['gasp_diameter_km'].median():.1f} km")

    # ── Figure: 3×3 grid (3 panels per zone, S-type) ─────────────────────────
    zones_list = list(ZONES.keys())
    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    fig.suptitle("Space Weathering Triangle: G ↔ Albedo ↔ Diameter (S-type, per zone)",
                 fontsize=13)

    panel_labels = [
        ("log_D",  "G",       "log₁₀(D_km)",   "G",          "log D vs G",      True),
        ("log_D",  "log_pV",  "log₁₀(D_km)",   "log₁₀(pV)",  "log D vs pV",     True),
        ("log_pV", "G",       "log₁₀(pV)",      "G",          "log pV vs G",     False),
    ]

    for row, z in enumerate(zones_list):
        sub = neo[(neo["_zone"] == z) & (neo["_tax"] == "S")]
        for col, (xcol, ycol, xlabel, ylabel, title, _) in enumerate(panel_labels):
            ax = axes[row, col]
            x = sub[xcol].values
            y = sub[ycol].values
            mask = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[mask], y[mask], s=4, alpha=0.25,
                       color=ZONE_COLORS[z], rasterized=True)
            # Running median
            x_ok, y_ok = x[mask], y[mask]
            if len(x_ok) > 20:
                xbins = np.linspace(np.nanpercentile(x_ok, 5),
                                    np.nanpercentile(x_ok, 95), 10)
                xbc = (xbins[:-1] + xbins[1:]) / 2
                meds = [np.median(y_ok[(x_ok >= lo) & (x_ok < hi)])
                        for lo, hi in zip(xbins[:-1], xbins[1:])]
                ax.plot(xbc, meds, "k-", lw=2, label="running med")
            r_row = next((row2 for row2 in log_rows
                          if row2["zone"] == z and row2["tax"] == "S"
                          and row2["pair"] == title.replace("log ", "").replace(" vs ", "-")),
                         None)
            rstr = ""
            if r_row and np.isfinite(r_row.get("rho", np.nan)):
                rstr = f" ρ={r_row['rho']:+.3f}"
            ax.set_xlabel(xlabel, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"{z}\n{title}{rstr}", fontsize=9)
            ax.grid(alpha=0.3)

    plt.tight_layout()
    out = PLOT_DIR / "26_albedo_weathering.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → {out.relative_to(ROOT)}")

    # ── Log ───────────────────────────────────────────────────────────────────
    log_path = LOG_DIR / "26_albedo_weathering_stats.txt"
    with open(log_path, "w") as f:
        f.write("GAPC Step 26 — Space Weathering Triple Check\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"NEOWISE objects: {len(neo):,}\n\n")
        f.write(f"{'Zone':12s}  {'Tax':4s}  {'Pair':6s}  {'n':6s}  {'rho':8s}  {'p':10s}\n")
        f.write("-" * 60 + "\n")
        for r in log_rows:
            if np.isfinite(r.get("rho", np.nan)):
                f.write(f"{r['zone']:12s}  {r['tax']:4s}  {r['pair']:6s}  "
                        f"{r['n']:6,}  {r['rho']:+8.4f}  {r['p']:10.3e}\n")
    print(f"  Log  → {log_path.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
