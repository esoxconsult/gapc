"""
33_lcdb_integration.py
GAPC — Integrate rotation periods and lightcurve data from LCDB.

Warner, Harris & Pravec 2009, Icarus 202, 134-146.
LCDB v2023-Oct: 34,293 numbered asteroids with lightcurve solutions.

Key data extracted (U≥2 only — probable or certain periods):
  lcdb_period_h   — rotation period [hours]
  lcdb_U_qual     — quality code (2, 2-, 2+, 3, 3+)
  lcdb_amp_min    — minimum lightcurve amplitude [mag]
  lcdb_amp_max    — maximum lightcurve amplitude [mag]
  lcdb_taxonomy   — LCDB taxonomy class
  lcdb_binary     — binary flag from LCDB (True/False)

Derived:
  rot_period_best — best period: Durech+2022 (pole model) > LCDB (empirical)

Outputs:
  data/final/gapc_catalog_v5.parquet  (updated in-place)
  plots/33_lcdb_integration.png
  logs/33_lcdb_integration_stats.txt
"""

import re
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

LCDB_PATH = DATA_RAW / "lcdb_sum.txt"

# Minimum U quality code to include (2 = probably correct, 3 = reliable)
U_MIN = 2


def _parse_float(s):
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return np.nan


def _parse_u(s):
    """Extract numeric part of U code (2-, 2, 2+, 3, 3+, 1, etc.)."""
    s = s.strip()
    if not s:
        return np.nan
    m = re.search(r"(\d)", s)
    return int(m.group(1)) if m else np.nan


def load_lcdb(path):
    records = []
    with open(path, encoding="latin-1") as f:
        for line in f:
            m = re.match(r"^\s*(\d+)", line)
            if not m:
                continue
            num = int(m.group(1))
            if num == 0:
                continue
            if len(line) < 190:
                continue

            period_str = line[145:162].strip()
            u_raw      = line[187:190].strip()
            u_val      = _parse_u(u_raw)
            # Only keep U≥2 periods
            if np.isnan(u_val) or u_val < U_MIN or not period_str:
                continue

            records.append(dict(
                number_mp   = num,
                lcdb_period_h = _parse_float(period_str),
                lcdb_U_qual   = u_raw[:3],
                lcdb_amp_min  = _parse_float(line[177:182]),
                lcdb_amp_max  = _parse_float(line[182:187]),
                lcdb_taxonomy = line[73:79].strip(),
                lcdb_binary   = bool(line[197:201].strip()),
            ))
    return pd.DataFrame(records)


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 33 — LCDB rotation periods & lightcurve data")
    print("=" * 65)

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    if not V5_PATH.exists():
        print(f"\n  ERROR: {V5_PATH} not found — run steps 31/32 first"); return
    if not LCDB_PATH.exists():
        print(f"\n  ERROR: {LCDB_PATH} not found"); return

    gapc = pd.read_parquet(V5_PATH)
    print(f"\n  v5 loaded: {len(gapc):,} objects, {len(gapc.columns)} columns")

    # ── Load LCDB ─────────────────────────────────────────────────────────────
    lcdb = load_lcdb(LCDB_PATH)
    print(f"\n  LCDB (U≥{U_MIN}): {len(lcdb):,} numbered asteroids with periods")
    p = lcdb["lcdb_period_h"].dropna()
    print(f"  Period range: {p.min():.3f}–{p.max():.1f} h  median={p.median():.3f} h")
    print(f"  Binary flag (lcdb_binary=True): {lcdb['lcdb_binary'].sum():,}")
    u_dist = lcdb["lcdb_U_qual"].value_counts()
    print(f"  U distribution: {u_dist.to_dict()}")

    # ── Merge ─────────────────────────────────────────────────────────────────
    gapc = gapc.merge(lcdb, on="number_mp", how="left")
    gapc["lcdb_binary"] = gapc["lcdb_binary"].fillna(False).astype(bool)
    n_per  = gapc["lcdb_period_h"].notna().sum()
    n_bin  = gapc["lcdb_binary"].sum()
    print(f"\n  Matched into GAPC: {n_per:,} ({n_per/len(gapc)*100:.1f}%) with period")
    print(f"  Known binaries in GAPC (LCDB): {n_bin:,}")

    # ── rot_period_best: Durech (model) > LCDB (empirical) ───────────────────
    if "rot_period_h" in gapc.columns:
        # Durech+2022 from step 32
        gapc["rot_period_best"] = gapc["rot_period_h"].copy()
        gapc["rot_period_source"] = np.where(gapc["rot_period_h"].notna(), "durech2022", pd.NA)
        # Fill from LCDB where Durech is missing
        fill_mask = gapc["rot_period_best"].isna() & gapc["lcdb_period_h"].notna()
        gapc.loc[fill_mask, "rot_period_best"]   = gapc.loc[fill_mask, "lcdb_period_h"]
        gapc.loc[fill_mask, "rot_period_source"]  = "lcdb"
        n_dur   = (gapc["rot_period_source"] == "durech2022").sum()
        n_lcdb_ = (gapc["rot_period_source"] == "lcdb").sum()
        print(f"\n  rot_period_best: {n_dur+n_lcdb_:,} total "
              f"({n_dur:,} Durech + {n_lcdb_:,} LCDB-only)")
    else:
        gapc["rot_period_best"]   = gapc["lcdb_period_h"]
        gapc["rot_period_source"] = np.where(gapc["lcdb_period_h"].notna(), "lcdb", pd.NA)
        print(f"\n  rot_period_best from LCDB only (no Durech column found)")

    n_best = gapc["rot_period_best"].notna().sum()
    print(f"  Total rot_period_best coverage: {n_best:,} ({n_best/len(gapc)*100:.1f}%)")

    # ── G vs amplitude ────────────────────────────────────────────────────────
    amp_col = "lcdb_amp_max"
    gamp = gapc[gapc[amp_col].notna() & gapc["G"].notna()].copy()
    if len(gamp) > 50:
        rho_ga, p_ga = spearmanr(gamp["G"], gamp[amp_col])
        print(f"\n  Spearman rho(G, amp_max) n={len(gamp):,}: "
              f"rho={rho_ga:+.4f}  p={p_ga:.2e}")
    else:
        rho_ga = p_ga = np.nan

    # ── G for known binaries vs non-binaries ──────────────────────────────────
    bin_g  = gapc[gapc["lcdb_binary"] == True]["G"].dropna()
    sing_g = gapc[gapc["lcdb_binary"] == False]["G"].dropna()
    if len(bin_g) >= 5 and len(sing_g) >= 5:
        from scipy.stats import mannwhitneyu
        U_mw, p_mw = mannwhitneyu(bin_g, sing_g, alternative="two-sided")
        print(f"\n  G: binaries (n={len(bin_g)}) vs singles (n={len(sing_g):.0f})")
        print(f"  median G  binary={bin_g.median():.4f}  single={sing_g.median():.4f}")
        print(f"  Mann-Whitney p={p_mw:.3e}")
    else:
        p_mw = np.nan

    # ── Plots ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"LCDB (Warner+2009)  n_period={n_per:,}  n_binary={n_bin:,}", fontsize=13)

    # Period histogram (LCDB-sourced only)
    ax = axes[0, 0]
    p_vals = gapc["lcdb_period_h"].dropna()
    p_plot = p_vals[p_vals < 100]
    ax.hist(p_plot.values, bins=80, color="steelblue", alpha=0.8, edgecolor="none")
    ax.axvline(p_vals.median(), color="red", lw=1.5,
               label=f"median={p_vals.median():.2f} h")
    ax.set_xlabel("Rotation period [h]"); ax.set_ylabel("Count")
    ax.set_title("LCDB periods (P < 100 h)")
    ax.legend(fontsize=9)

    # rot_period_best histogram
    ax = axes[0, 1]
    pb = gapc["rot_period_best"].dropna()
    pb_plot = pb[pb < 100]
    ax.hist(pb_plot.values, bins=80, color="coral", alpha=0.8, edgecolor="none")
    ax.axvline(pb.median(), color="navy", lw=1.5,
               label=f"median={pb.median():.2f} h  n={len(pb):,}")
    ax.set_xlabel("rot_period_best [h]"); ax.set_ylabel("Count")
    ax.set_title("Best period (Durech+LCDB, P < 100 h)")
    ax.legend(fontsize=9)

    # G vs amplitude
    ax = axes[1, 0]
    if len(gamp) > 10:
        ax.scatter(gamp[amp_col], gamp["G"], s=3, alpha=0.2,
                   color="steelblue", rasterized=True)
        ax.set_xlabel("Lightcurve amplitude [mag]"); ax.set_ylabel("G (phase slope)")
        ax.set_title(f"G vs amplitude  rho={rho_ga:+.3f}  p={p_ga:.1e}")
        ax.grid(alpha=0.2)
    else:
        ax.set_axis_off()

    # G binary vs singles boxplot
    ax = axes[1, 1]
    if len(bin_g) >= 5:
        bp = ax.boxplot([bin_g.values, sing_g.values],
                        tick_labels=["Binary", "Non-binary"],
                        patch_artist=True, showfliers=False,
                        medianprops={"lw": 2, "color": "red"})
        for patch, col in zip(bp["boxes"], ["coral", "steelblue"]):
            patch.set_facecolor(col); patch.set_alpha(0.6)
        ax.set_ylabel("G (phase slope)"); ax.set_title("G: binary vs non-binary")
        ax.grid(alpha=0.3, axis="y")
        for i, (lbl, g) in enumerate([("Binary", bin_g), ("Non-binary", sing_g)]):
            ax.text(i+1, ax.get_ylim()[1]*0.95, f"n={len(g):,}", ha="center", fontsize=8)
    else:
        ax.set_axis_off()

    fig.tight_layout()
    fig.savefig(PLOT_DIR / "33_lcdb_integration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot → plots/33_lcdb_integration.png")

    # ── Save v5 in-place ──────────────────────────────────────────────────────
    gapc.to_parquet(V5_PATH, index=False)
    print(f"  Updated v5: {len(gapc.columns)} cols")

    with open(LOG_DIR / "33_lcdb_integration_stats.txt", "w") as f:
        f.write("GAPC Step 33 — LCDB (Warner+2009) rotation periods\n")
        f.write("=" * 60 + "\n")
        f.write(f"LCDB U≥{U_MIN} numbered entries: {len(lcdb):,}\n")
        f.write(f"Matched into GAPC: {n_per:,} ({n_per/len(gapc)*100:.1f}%)\n")
        f.write(f"Known LCDB binaries in GAPC: {n_bin:,}\n")
        f.write(f"rot_period_best coverage: {n_best:,} ({n_best/len(gapc)*100:.1f}%)\n")
        if not np.isnan(rho_ga):
            f.write(f"Spearman rho(G, amp_max): {rho_ga:+.4f}  p={p_ga:.2e}  "
                    f"n={len(gamp):,}\n")
        if not np.isnan(p_mw):
            f.write(f"G binary median: {bin_g.median():.4f}  n={len(bin_g):,}\n")
            f.write(f"G single median: {sing_g.median():.4f}  n={len(sing_g):,}\n")
            f.write(f"Mann-Whitney p: {p_mw:.3e}\n")
    print(f"  Log  → logs/33_lcdb_integration_stats.txt\n")


if __name__ == "__main__":
    main()
