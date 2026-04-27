"""
36_sdss_colors.py
GAPC — Add SDSS MOC4 photometric colors for taxonomy discrimination.

Ivezic et al. 2001 (AJ 122, 2749); SDSS Moving Object Catalog 4 (MOC4).
471,569 observations; 123,590 with numbered asteroid IDs.

Key color indices:
  a*  = 0.89(g-r) + 0.45(r-i) - 0.57   [Ivezic+2001 taxonomy proxy]
     a* > 0.1 → S-complex (silicate, rocky)
     a* < 0.0 → C/X-complex (dark, carbonaceous)
  i-z = spectral redness beyond 750 nm
     i-z > 0.15 → D-type (very red)

We aggregate multiple observations per asteroid (mean of valid detections).
Quality cut: only detections with g-band error < 0.2 mag.

New columns added to v5:
  sdss_a_star     — mean a* color index
  sdss_g_r        — mean g-r color
  sdss_i_z        — mean i-z color
  sdss_n_obs      — number of valid SDSS observations

Derived taxonomy:
  sdss_complex    — C (a*<0), S (a*≥0.1), X/ambiguous (0≤a*<0.1) — broad class

Outputs:
  data/final/gapc_catalog_v5.parquet  (updated in-place)
  plots/36_sdss_colors.png
  logs/36_sdss_colors_stats.txt
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

ROOT     = Path(__file__).resolve().parents[1]
V5_PATH  = ROOT / "data" / "final" / "gapc_catalog_v5.parquet"
PLOT_DIR = ROOT / "plots"
LOG_DIR  = ROOT / "logs"
DATA_RAW = ROOT / "data" / "raw"

SDSS_PATH = DATA_RAW / "sdss_moc4.csv"

# Column name map: from MOC4 file
NUM_COL = "ast_number"
A_COL   = "a_star"
GR_COL  = "g_r"
IZ_COL  = None  # will be computed from i-z if present
G_ERR   = "gErr"


def compute_iz(df):
    """Compute i-z if columns available."""
    if "i" in df.columns and "z" in df.columns:
        return df["i"] - df["z"]
    if "i_z" in df.columns:
        return df["i_z"]
    return pd.Series(np.nan, index=df.index)


def sdss_complex(a_star):
    """Broad taxonomy from a*."""
    if pd.isna(a_star):
        return np.nan
    if a_star >= 0.1:
        return "S"
    if a_star < 0.0:
        return "C"
    return "X"


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 36 — SDSS MOC4 photometric colors")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V5_PATH.exists():
        print(f"\n  ERROR: {V5_PATH} not found — run step 35 first"); return
    if not SDSS_PATH.exists():
        print(f"\n  ERROR: {SDSS_PATH} not found"); return

    gapc = pd.read_parquet(V5_PATH)
    print(f"\n  v5 loaded: {len(gapc):,} objects, {len(gapc.columns)} columns")

    # ── Load SDSS MOC4 ────────────────────────────────────────────────────────
    print(f"\n  Loading SDSS MOC4 …")
    sdss = pd.read_csv(SDSS_PATH, low_memory=False)
    print(f"  {len(sdss):,} total observations")

    # Keep numbered asteroids only
    sdss[NUM_COL] = pd.to_numeric(sdss[NUM_COL], errors="coerce")
    sdss = sdss[sdss[NUM_COL] > 0].copy()
    sdss[NUM_COL] = sdss[NUM_COL].astype("int64")
    print(f"  Numbered asteroid observations: {len(sdss):,}")

    # Quality filter: g-band error < 0.2
    if G_ERR in sdss.columns:
        sdss["_gErr"] = pd.to_numeric(sdss[G_ERR], errors="coerce")
        sdss = sdss[sdss["_gErr"] < 0.2]
        print(f"  After g-error < 0.2 cut: {len(sdss):,}")

    # Ensure a_star is available
    if A_COL not in sdss.columns:
        if "g_r" in sdss.columns and "r_i" in sdss.columns:
            g_r  = pd.to_numeric(sdss["g_r"],  errors="coerce")
            r_i  = pd.to_numeric(sdss["r_i"],  errors="coerce")
            sdss[A_COL] = 0.89 * g_r + 0.45 * r_i - 0.57
        else:
            print("  Cannot compute a*: required columns missing"); return
    else:
        sdss[A_COL] = pd.to_numeric(sdss[A_COL], errors="coerce")

    # g-r color
    if GR_COL in sdss.columns:
        sdss[GR_COL] = pd.to_numeric(sdss[GR_COL], errors="coerce")
    elif "g" in sdss.columns and "r" in sdss.columns:
        sdss[GR_COL] = pd.to_numeric(sdss["g"], errors="coerce") - pd.to_numeric(sdss["r"], errors="coerce")

    # i-z color
    sdss["_i_z"] = compute_iz(sdss)

    # ── Aggregate per asteroid ─────────────────────────────────────────────────
    agg = (sdss.groupby(NUM_COL)
           .agg(
               sdss_a_star=pd.NamedAgg(A_COL, lambda x: x.dropna().mean()),
               sdss_g_r   =pd.NamedAgg(GR_COL if GR_COL in sdss.columns else A_COL,
                                         lambda x: x.dropna().mean()),
               sdss_i_z   =pd.NamedAgg("_i_z", lambda x: x.dropna().mean()),
               sdss_n_obs =pd.NamedAgg(A_COL, lambda x: x.dropna().count()),
           )
           .reset_index()
           .rename(columns={NUM_COL: "number_mp"}))

    # Filter: require at least 1 valid a* measurement
    agg = agg[agg["sdss_n_obs"] >= 1]
    print(f"\n  Unique asteroids with valid a*: {len(agg):,}")
    a = agg["sdss_a_star"].dropna()
    print(f"  a* range: {a.min():.3f}–{a.max():.3f}  median={a.median():.3f}")

    # sdss_complex taxonomy label
    agg["sdss_complex"] = agg["sdss_a_star"].apply(sdss_complex)
    comp_dist = agg["sdss_complex"].value_counts()
    print(f"  SDSS complex: {comp_dist.to_dict()}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    gapc = gapc.merge(agg, on="number_mp", how="left")
    n_sdss = gapc["sdss_a_star"].notna().sum()
    print(f"\n  Matched into GAPC: {n_sdss:,} ({n_sdss/len(gapc)*100:.1f}%)")

    # ── G vs a* correlation ────────────────────────────────────────────────────
    ga = gapc[gapc["sdss_a_star"].notna() & gapc["G"].notna()].copy()
    if len(ga) > 50:
        rho_ga, p_ga = spearmanr(ga["G"], ga["sdss_a_star"])
        print(f"\n  Spearman rho(G, a*) n={len(ga):,}: "
              f"rho={rho_ga:+.4f}  p={p_ga:.2e}")
        # By complex
        for cplx in ["S", "C"]:
            sub = ga[ga["sdss_complex"] == cplx]
            if len(sub) >= 20:
                r, p_ = spearmanr(sub["G"], sub["sdss_a_star"])
                print(f"    {cplx}: rho={r:+.4f}  p={p_:.2e}  n={len(sub):,}")
    else:
        rho_ga = p_ga = np.nan

    # G by SDSS complex
    g_by_cplx = {}
    for cplx in ["S", "C", "X"]:
        sub = gapc[(gapc["sdss_complex"] == cplx) & gapc["G"].notna()]
        if len(sub) >= 10:
            g_by_cplx[cplx] = sub["G"]
            print(f"  G {cplx}: median={sub['G'].median():.4f}  "
                  f"std={sub['G'].std():.4f}  n={len(sub):,}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"SDSS MOC4 colors  n={n_sdss:,}", fontsize=13)

    # a* histogram
    ax = axes[0, 0]
    a_vals = gapc["sdss_a_star"].dropna()
    ax.hist(a_vals.clip(-0.5, 0.5).values, bins=80, color="steelblue",
            alpha=0.8, edgecolor="none")
    ax.axvline(0.0,  color="k",   lw=1, linestyle="--", label="a*=0 (C/S boundary)")
    ax.axvline(0.1,  color="red", lw=1, linestyle="--", label="a*=0.1")
    ax.set_xlabel("a* color index"); ax.set_ylabel("Count")
    ax.set_title("SDSS a* distribution")
    ax.legend(fontsize=8)

    # G vs a*
    ax = axes[0, 1]
    if len(ga) > 10:
        ax.scatter(ga["sdss_a_star"].clip(-0.4, 0.4), ga["G"],
                   s=3, alpha=0.2, color="steelblue", rasterized=True)
        ax.axvline(0, color="k", lw=0.8, linestyle="--")
        ax.set_xlabel("a* color index"); ax.set_ylabel("G (phase slope)")
        ax.set_title(f"G vs a*  rho={rho_ga:+.3f}  p={p_ga:.1e}")
        ax.grid(alpha=0.2)
    else:
        ax.set_axis_off()

    # g-r histogram
    ax = axes[1, 0]
    gr_vals = gapc["sdss_g_r"].dropna()
    if len(gr_vals) > 10:
        ax.hist(gr_vals.clip(0.1, 0.9).values, bins=60, color="coral",
                alpha=0.8, edgecolor="none")
        ax.set_xlabel("g-r color"); ax.set_ylabel("Count")
        ax.set_title("g-r distribution")
    else:
        ax.set_axis_off()

    # G boxplot by SDSS complex
    ax = axes[1, 1]
    if g_by_cplx:
        labels = list(g_by_cplx.keys())
        grps   = [g_by_cplx[c].values for c in labels]
        bp = ax.boxplot(grps, tick_labels=labels,
                        patch_artist=True, showfliers=False,
                        medianprops={"lw": 2, "color": "red"})
        colors = ["#e07b39", "#5c85d6", "#9c27b0"]
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col); patch.set_alpha(0.6)
        ax.set_ylabel("G (phase slope)")
        ax.set_title("G by SDSS complex")
        ax.grid(alpha=0.3, axis="y")
        for i, (lbl, g) in enumerate(g_by_cplx.items()):
            ax.text(i+1, ax.get_ylim()[1]*0.95, f"n={len(g):,}",
                    ha="center", fontsize=8)
    else:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "36_sdss_colors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/36_sdss_colors.png")

    gapc.to_parquet(V5_PATH, index=False)
    print(f"  Updated v5: {len(gapc.columns)} cols")

    with open(LOG_DIR / "36_sdss_colors_stats.txt", "w") as f:
        f.write("GAPC Step 36 — SDSS MOC4 photometric colors\n")
        f.write("=" * 60 + "\n")
        f.write(f"Unique numbered in MOC4: {len(agg):,}\n")
        f.write(f"Matched into GAPC:       {n_sdss:,} ({n_sdss/len(gapc)*100:.1f}%)\n")
        f.write(f"SDSS complex: {comp_dist.to_dict()}\n")
        if not np.isnan(rho_ga):
            f.write(f"Spearman rho(G, a*): {rho_ga:+.4f}  p={p_ga:.2e}  n={len(ga):,}\n")
        for cplx, g in g_by_cplx.items():
            f.write(f"G {cplx}: median={g.median():.4f}  std={g.std():.4f}  n={len(g):,}\n")
    print(f"  Log  → logs/36_sdss_colors_stats.txt\n")


if __name__ == "__main__":
    main()
