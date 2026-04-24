"""
15_orbital_class.py
GAPC — B3: Orbital class comparison of G-parameter distributions.

Derives orbital classes from MPCORB (a, e, i, q) for all 128K objects and
compares G-slope distributions between NEA sub-types, MBA zones, Trojans, etc.

Classification (Gladman et al. 2008 + MPC standard):
  Atira:   a < 1.0 AU
  Aten:    a < 1.0 AU, Q > 0.983 AU
  Apollo:  a >= 1.0, q < 1.017 AU
  Amor:    1.017 < q < 1.3 AU
  MBA-I:   2.0 < a < 2.5 AU (inner)
  MBA-II:  2.5 < a < 2.82 AU (middle)
  MBA-III: 2.82 < a < 3.27 AU (outer)
  Hungaria: 1.78 < a < 2.0 AU, e < 0.18, i > 16 deg
  Phocaea:  2.25 < a < 2.5 AU, e > 0.10, i > 18 deg
  Trojan:   5.05 < a < 5.40 AU
  Hilda:    3.70 < a < 4.20 AU
  Cybele:   3.27 < a < 3.70 AU

Cross-validates against gasp_orbital_class for 18K matched objects.

Outputs:
  data/interim/mpcorb_orbital_class.parquet
  plots/15_orbital_class_G_boxplot.png
  plots/15_orbital_class_bias_boxplot.png
  logs/15_orbital_class_stats.csv
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import kruskal

ROOT      = Path(__file__).resolve().parents[1]
CAT_PATH  = ROOT / "data" / "final"   / "gapc_catalog_v3_var.parquet"
MPC_PATH  = ROOT / "data" / "raw"     / "mpc_h_magnitudes.parquet"
MPCORB    = ROOT / "data" / "raw"     / "MPCORB.DAT"
OUT_OC    = ROOT / "data" / "interim" / "mpcorb_orbital_class.parquet"
PLOT_DIR  = ROOT / "plots"
LOG_DIR   = ROOT / "logs"

MIN_N = 30


def parse_mpcorb(path):
    """Extract (number_mp, a, e, i, q) from MPCORB.DAT."""
    records = []
    data_start = False
    with open(path, encoding="ascii", errors="ignore") as f:
        for line in f:
            if line.startswith("---"):
                data_start = True
                continue
            if not data_start or len(line) < 103:
                continue
            s = line[0:7].strip()
            if not s:
                continue
            c = s[0]
            if c.isdigit():
                try:
                    num = int(s)
                except ValueError:
                    continue
            else:
                base = (ord(c)-ord("A")+10) if c.isupper() else (ord(c)-ord("a")+36)
                try:
                    num = base * 10000 + int(s[1:].strip() or 0)
                except ValueError:
                    continue
            try:
                e = float(line[70:79])
                a = float(line[92:103])
                i = float(line[59:68])
                q = a * (1 - e)
                Q = a * (1 + e)
                records.append((num, a, e, i, q, Q))
            except (ValueError, IndexError):
                continue
    return pd.DataFrame(records, columns=["number_mp", "a_au", "ecc", "inc_deg", "q_au", "Q_au"])


def classify_orbit(row):
    a, e, i, q, Q = row["a_au"], row["ecc"], row["inc_deg"], row["q_au"], row["Q_au"]
    if pd.isna(a):
        return "Unknown"
    if a < 1.0 and Q < 0.983:
        return "Atira"
    if a < 1.0:
        return "Aten"
    if q < 1.017:
        return "Apollo"
    if q < 1.3:
        return "Amor"
    if 1.78 < a < 2.0 and e < 0.18 and i > 16:
        return "Hungaria"
    if 2.25 < a < 2.5 and e > 0.10 and i > 18:
        return "Phocaea"
    if 2.0 <= a < 2.5:
        return "MBA-inner"
    if 2.5 <= a < 2.82:
        return "MBA-middle"
    if 2.82 <= a < 3.27:
        return "MBA-outer"
    if 3.27 <= a < 3.70:
        return "Cybele"
    if 3.70 <= a < 4.20:
        return "Hilda"
    if 5.05 <= a < 5.40:
        return "Trojan"
    if a >= 5.40:
        return "Outer"
    return "Other"


def main():
    print("\n" + "=" * 60)
    print("  GAPC Step 15 — Orbital class comparison (B3)")
    print("=" * 60)

    df  = pd.read_parquet(CAT_PATH)
    mpc_h = pd.read_parquet(MPC_PATH)

    # --- Parse MPCORB ---
    print(f"\n  Parsing MPCORB.DAT …")
    orb = parse_mpcorb(MPCORB)
    print(f"  Parsed {len(orb):,} numbered asteroids")

    # --- Classify ---
    orb["orbital_class"] = orb.apply(classify_orbit, axis=1)
    class_counts = orb["orbital_class"].value_counts()
    print(f"\n  Orbital class breakdown (MPCORB, all numbered):")
    for cls, n in class_counts.items():
        print(f"    {cls:12s}: {n:7,}")

    # --- Save interim ---
    OUT_OC.parent.mkdir(parents=True, exist_ok=True)
    orb.to_parquet(OUT_OC, index=False, compression="snappy")

    # --- Merge with catalog ---
    merged = df.merge(orb[["number_mp", "orbital_class", "a_au", "ecc", "inc_deg", "q_au"]],
                      on="number_mp", how="left")
    merged = merged.merge(mpc_h[["number_mp", "H_mpc"]], on="number_mp", how="left")
    merged["bias_HV"] = merged["H_V"] - merged["H_mpc"]

    print(f"\n  Matched {merged['orbital_class'].notna().sum():,} / {len(merged):,} objects to MPCORB")

    # --- Cross-validate vs GASP orbital class ---
    gasp_mask = merged["gasp_match"] & merged["gasp_orbital_class"].notna()
    print(f"\n  Cross-validation MPCORB vs GASP (n={gasp_mask.sum():,}):")
    xv = merged.loc[gasp_mask, ["orbital_class", "gasp_orbital_class"]]
    for cls in ["MBA-inner", "MBA-middle", "MBA-outer", "Hungaria", "Phocaea",
                "Apollo", "Amor", "Aten", "Trojan"]:
        sub = xv[xv["orbital_class"] == cls]
        if len(sub) < 5:
            continue
        top_gasp = sub["gasp_orbital_class"].value_counts().head(2)
        print(f"    {cls:12s} (n={len(sub):4,}): "
              + ", ".join(f"{g}={n}" for g, n in top_gasp.items()))

    # --- Per-class statistics ---
    classes = (merged["orbital_class"].value_counts()
               .loc[lambda x: x >= MIN_N].index.tolist())
    # Sort by semi-major axis proxy
    order = ["Atira","Aten","Apollo","Amor","Hungaria","MBA-inner",
             "Phocaea","MBA-middle","MBA-outer","Cybele","Hilda","Trojan","Outer","Other"]
    classes = [c for c in order if c in classes]

    rows = []
    print(f"\n  {'Class':12s}  {'N':>7s}  {'G median':>9s}  {'G std':>7s}  "
          f"{'G_unc%':>7s}  {'bias':>7s}  {'a median':>9s}")
    for cls in classes:
        sub = merged[merged["orbital_class"] == cls]
        g   = sub["G"].dropna()
        bv  = sub["bias_HV"].dropna()
        gu  = sub["G_uncertain"].mean() * 100
        a   = sub["a_au"].median()
        row = dict(orbital_class=cls, n=len(sub), G_median=g.median(),
                   G_std=g.std(), G_uncertain_pct=gu,
                   bias_median=bv.median(), n_mpc=len(bv), a_median=a)
        rows.append(row)
        print(f"  {cls:12s}  {len(sub):7,}  {g.median():9.3f}  {g.std():7.3f}  "
              f"{gu:7.1f}%  {bv.median():+7.3f}  {a:9.3f} AU")

    stats_df = pd.DataFrame(rows)

    # --- Kruskal-Wallis test ---
    groups = [merged.loc[merged["orbital_class"] == cls, "G"].dropna().values
              for cls in classes]
    H_stat, p_kw = kruskal(*[g for g in groups if len(g) > 0])
    print(f"\n  Kruskal-Wallis (G across {len(classes)} classes): "
          f"H={H_stat:.2f}  p={p_kw:.2e}")

    # NEA group vs MBA group
    nea_classes  = ["Atira","Aten","Apollo","Amor"]
    mba_classes  = ["MBA-inner","MBA-middle","MBA-outer"]
    nea_G = merged.loc[merged["orbital_class"].isin(nea_classes), "G"].dropna()
    mba_G = merged.loc[merged["orbital_class"].isin(mba_classes), "G"].dropna()
    if len(nea_G) > 5 and len(mba_G) > 5:
        H2, p2 = kruskal(nea_G, mba_G)
        print(f"\n  NEA (n={len(nea_G):,}) vs MBA (n={len(mba_G):,}): "
              f"KW H={H2:.2f}  p={p2:.2e}")
        print(f"    NEA G median={nea_G.median():.3f}  MBA G median={mba_G.median():.3f}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle("G-slope by orbital class", fontsize=13)

    ax = axes[0]
    data_g = [merged.loc[merged["orbital_class"] == cls, "G"].dropna().values
              for cls in classes]
    bp = ax.boxplot(data_g, tick_labels=classes, patch_artist=True,
                    medianprops={"color": "red", "lw": 1.5},
                    flierprops={"marker": ".", "ms": 2, "alpha": 0.3})
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_ylabel("G (HG slope)")
    ax.set_title(f"G by orbital class  ·  KW p={p_kw:.1e}")
    ax.tick_params(axis="x", rotation=30)
    for i, cls in enumerate(classes):
        n = (merged["orbital_class"] == cls).sum()
        ax.text(i+1, 0.02, f"n={n}", ha="center", fontsize=7, color="gray")

    ax2 = axes[1]
    data_b = [merged.loc[merged["orbital_class"] == cls, "bias_HV"].dropna().clip(-2,2).values
              for cls in classes]
    bp2 = ax2.boxplot(data_b, tick_labels=classes, patch_artist=True,
                      medianprops={"color": "red", "lw": 1.5},
                      flierprops={"marker": ".", "ms": 2, "alpha": 0.3})
    for patch, color in zip(bp2["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax2.axhline(0, color="k", lw=0.8, linestyle="--")
    ax2.set_ylabel("H_V − H_MPC [mag]")
    ax2.set_title("Bias by orbital class")
    ax2.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOT_DIR / "15_orbital_class_G_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/15_orbital_class_G_boxplot.png")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(LOG_DIR / "15_orbital_class_stats.csv", index=False, float_format="%.4f")
    print(f"  Log  → logs/15_orbital_class_stats.csv\n")


if __name__ == "__main__":
    main()
