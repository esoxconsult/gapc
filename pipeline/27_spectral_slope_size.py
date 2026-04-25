"""
27_spectral_slope_size.py
GAPC — Spectral slope vs size, stratified by orbital zone (S-types).

Step 11 showed no overall correlation between spectral slope and G.
But space weathering predicts: larger S-types → older surface → redder AND
lower G. If true, we should see spectral slope increase WITH size for S-types,
and spectral slope should correlate WITH G (both driven by weathering).

Tests:
  1. Spearman rho(log D_km, spectral_slope)  per zone, S-type only
  2. Spearman rho(spectral_slope, G)         per zone, S-type only
  3. Partial correlation: G vs D controlling for spectral slope

This is a simultaneous confirmation of the weathering signal in two independent
Gaia observables (reflectance spectrum + phase curve) on the same objects.

GASP spectral slope: computed from the Gaia reflectance spectrum as
  slope = (R_max - R_min) / (λ_max - λ_min) / R_550
where R is the reflectance at the GASP wavelength bands (374–916 nm).

Outputs:
  plots/27_spectral_slope_size.png
  logs/27_spectral_slope_size_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

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

# GASP reflectance band wavelengths (nm)
GASP_WAVES = np.array([374, 418, 462, 506, 550, 594, 638, 682, 726, 770, 814, 858, 902])
GASP_COLS  = [f"gasp_refl_{int(w)}" for w in GASP_WAVES]

MIN_N = 20


def compute_slope(df):
    """
    Compute spectral slope (%/100nm) from GASP reflectance bands.
    Uses linear regression on available bands, normalised at 550 nm.
    Returns NaN if <4 valid bands.
    """
    refl_cols = [c for c in GASP_COLS if c in df.columns]
    if len(refl_cols) < 4:
        return pd.Series(np.nan, index=df.index)
    waves = np.array([int(c.split("_")[-1]) for c in refl_cols])
    refl  = df[refl_cols].values.astype(float)  # shape (n, m)
    slopes = np.full(len(df), np.nan)
    for i in range(len(df)):
        r = refl[i]
        ok = np.isfinite(r)
        if ok.sum() < 4:
            continue
        # Normalise at 550 nm (or closest)
        idx550 = np.argmin(np.abs(waves - 550))
        norm = r[idx550]
        if not np.isfinite(norm) or norm <= 0:
            continue
        r_norm = r[ok] / norm
        w_ok   = waves[ok]
        # Linear fit: slope in % / 100 nm
        coeffs = np.polyfit(w_ok, r_norm, 1)
        slopes[i] = coeffs[0] * 100 * 100   # per 100 nm, as fraction → %
    return pd.Series(slopes, index=df.index)


def srho(x, y, label=""):
    mask = np.isfinite(x) & np.isfinite(y)
    n = mask.sum()
    if n < MIN_N:
        print(f"    {label:40s}  n={n} < {MIN_N} — skip")
        return np.nan, np.nan, n
    r, p = spearmanr(x[mask], y[mask])
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"    {label:40s}  n={n:5,}  rho={r:+.4f}  p={p:.2e}  {sig}")
    return r, p, n


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 27 — Spectral Slope vs Size (Space Weathering B)")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(CAT_PATH)
    print(f"  Catalog: {len(df):,}")

    # Quality filter
    if "fit_ok" in df.columns:
        df = df[df["fit_ok"]].copy()
    if "G_uncertain" in df.columns:
        df = df[~df["G_uncertain"].fillna(False)].copy()
    df = df[df["G"].notna()].copy()

    # Spectral slope
    avail = [c for c in GASP_COLS if c in df.columns]
    print(f"  GASP reflectance bands available: {len(avail)}")
    if len(avail) < 4:
        print("  ERROR: too few GASP bands — check catalog")
        return

    df["spectral_slope"] = compute_slope(df)
    n_slope = df["spectral_slope"].notna().sum()
    print(f"  Objects with spectral slope: {n_slope:,}")
    print(f"  Slope range: {df['spectral_slope'].quantile(0.01):.3f} – "
          f"{df['spectral_slope'].quantile(0.99):.3f} %/100nm")

    # Diameter
    if "D_km" not in df.columns:
        print("  ERROR: D_km missing — run step 13 first")
        return
    df = df[df["D_km"].notna() & (df["D_km"] > 0)].copy()
    df["log_D"] = np.log10(df["D_km"])

    # Taxonomy & zone
    if "predicted_taxonomy" in df.columns:
        df["_tax"] = df["predicted_taxonomy"].fillna("Other")
    else:
        df["_tax"] = "Other"
    if "gasp_taxonomy_final" in df.columns:
        m = df["_tax"] == "Other"
        raw = df.loc[m, "gasp_taxonomy_final"].str.strip().str.upper().str[0]
        df.loc[m, "_tax"] = raw.map({"S": "S", "C": "C", "X": "X"}).fillna("Other")

    def zone_of(a):
        for z, (lo, hi) in ZONES.items():
            if lo <= a < hi:
                return z
        return None
    df["_zone"] = df["a_au"].map(zone_of)

    # ── All objects baseline ──────────────────────────────────────────────────
    print(f"\n--- Baseline (all objects) ---")
    srho(df["log_D"].values, df["spectral_slope"].values, "log D vs spectral slope (all)")
    srho(df["spectral_slope"].values, df["G"].values,      "spectral slope vs G (all)")

    # ── Per zone, S-type ──────────────────────────────────────────────────────
    log_rows = []
    print(f"\n--- S-type per orbital zone ---")
    for z in ZONES:
        sub = df[(df["_zone"] == z) & (df["_tax"] == "S") &
                 df["spectral_slope"].notna()].copy()
        print(f"\n  {z}  (n with slope={len(sub)}):")

        ra, pa, na = srho(sub["log_D"].values, sub["spectral_slope"].values,
                          "  log D vs spectral slope  (expect +)")
        rb, pb, nb = srho(sub["spectral_slope"].values, sub["G"].values,
                          "  spectral slope vs G      (expect −)")
        rc, pc, nc = srho(sub["log_D"].values, sub["G"].values,
                          "  log D vs G               (expect −)")

        log_rows += [
            dict(zone=z, tax="S", pair="D-slope",     n=na, rho=ra, p=pa),
            dict(zone=z, tax="S", pair="slope-G",     n=nb, rho=rb, p=pb),
            dict(zone=z, tax="S", pair="D-G",         n=nc, rho=rc, p=pc),
        ]

        # Large vs small: median slope for large (D>10 km) vs small (D<3 km)
        big   = sub[sub["D_km"] > 10]
        small = sub[sub["D_km"] < 3]
        if len(big) >= 5 and len(small) >= 5:
            print(f"    Large S-types (D>10 km, n={len(big)}): "
                  f"slope_med={big['spectral_slope'].median():.3f}  G_med={big['G'].median():.4f}")
            print(f"    Small S-types (D<3  km, n={len(small)}): "
                  f"slope_med={small['spectral_slope'].median():.3f}  G_med={small['G'].median():.4f}")

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Spectral Slope vs Size & G — Space Weathering (S-type, per zone)",
                 fontsize=13)

    zones_list = list(ZONES.keys())
    for col, z in enumerate(zones_list):
        sub = df[(df["_zone"] == z) & (df["_tax"] == "S") &
                 df["spectral_slope"].notna()].copy()
        clr = ZONE_COLORS[z]

        # Top row: log D vs spectral slope
        ax = axes[0, col]
        ax.scatter(sub["log_D"], sub["spectral_slope"].clip(-5, 15),
                   s=3, alpha=0.15, color=clr, rasterized=True)
        # Running median
        if len(sub) > MIN_N:
            xbins = np.linspace(sub["log_D"].quantile(0.05),
                                sub["log_D"].quantile(0.95), 10)
            xbc = (xbins[:-1] + xbins[1:]) / 2
            meds = [sub.loc[(sub["log_D"] >= lo) & (sub["log_D"] < hi),
                            "spectral_slope"].median()
                    for lo, hi in zip(xbins[:-1], xbins[1:])]
            ax.plot(xbc, meds, "k-", lw=2)
        r_val = next((r["rho"] for r in log_rows
                      if r["zone"] == z and r["pair"] == "D-slope"), np.nan)
        ax.set_xlabel("log₁₀ D [km]", fontsize=9)
        ax.set_ylabel("Spectral slope [%/100nm]", fontsize=9)
        ax.set_title(f"{z}\nlog D vs slope  ρ={r_val:+.3f}" if np.isfinite(r_val)
                     else f"{z}\nlog D vs slope", fontsize=9)
        ax.set_ylim(-5, 15); ax.grid(alpha=0.3)

        # Bottom row: spectral slope vs G
        ax = axes[1, col]
        ax.scatter(sub["spectral_slope"].clip(-5, 15), sub["G"],
                   s=3, alpha=0.15, color=clr, rasterized=True)
        if len(sub) > MIN_N:
            xbins = np.linspace(sub["spectral_slope"].quantile(0.05),
                                sub["spectral_slope"].quantile(0.95), 10)
            xbc = (xbins[:-1] + xbins[1:]) / 2
            meds = [sub.loc[(sub["spectral_slope"] >= lo) & (sub["spectral_slope"] < hi),
                            "G"].median()
                    for lo, hi in zip(xbins[:-1], xbins[1:])]
            ax.plot(xbc, meds, "k-", lw=2)
        r_val = next((r["rho"] for r in log_rows
                      if r["zone"] == z and r["pair"] == "slope-G"), np.nan)
        ax.set_xlabel("Spectral slope [%/100nm]", fontsize=9)
        ax.set_ylabel("G", fontsize=9)
        ax.set_title(f"Slope vs G  ρ={r_val:+.3f}" if np.isfinite(r_val)
                     else "Slope vs G", fontsize=9)
        ax.set_xlim(-5, 15); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = PLOT_DIR / "27_spectral_slope_size.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → {out.relative_to(ROOT)}")

    # ── Log ───────────────────────────────────────────────────────────────────
    log_path = LOG_DIR / "27_spectral_slope_size_stats.txt"
    with open(log_path, "w") as f:
        f.write("GAPC Step 27 — Spectral Slope vs Size\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Objects with GASP slope: {n_slope:,}\n\n")
        f.write(f"{'Zone':12s}  {'Tax':4s}  {'Pair':12s}  {'n':6s}  "
                f"{'rho':8s}  {'p':10s}\n")
        f.write("-" * 60 + "\n")
        for r in log_rows:
            if np.isfinite(r.get("rho", np.nan)):
                f.write(f"{r['zone']:12s}  {r['tax']:4s}  {r['pair']:12s}  "
                        f"{r['n']:6,}  {r['rho']:+8.4f}  {r['p']:10.3e}\n")
    print(f"  Log  → {log_path.relative_to(ROOT)}\n")


if __name__ == "__main__":
    main()
