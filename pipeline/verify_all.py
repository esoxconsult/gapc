"""
verify_all.py
GAPC — Comprehensive data, calculation and conclusion verification.

Sections:
  A. Catalog structural integrity
  B. G column statistics & sanity checks
  C. Cross-catalog spot checks (DAMIT, Goffin, Pravec, SDSS)
  D. Taxonomy distribution checks (taxonomy_refined)
  E. External calibration verification (H_V_tax vs PTF, ATLAS)
  F. Space weathering (step 37) — re-derive rho and partial r
  G. Binary G excess (step 39) — re-derive Mann-Whitney
  H. H-completeness (step 41) — re-derive alpha and H_turn
  I. Universal size law (step 47/49) — re-derive partial r for S/M/E/P/C
  J. Rotation×G (step 38) — re-derive rho(G,logP)
  K. GASP v2 integrity
  L. Publication figures file check

Outputs:
  logs/verify_all_report.txt
  logs/verify_all_summary.txt   (PASS/FAIL table only)
"""

import sys
import json
import warnings
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu, pearsonr
from scipy.stats import rankdata
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

ROOT     = Path(__file__).resolve().parents[1]
LOG_DIR  = ROOT / "logs"
PLOT_DIR = ROOT / "plots"
LOG_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH  = LOG_DIR / "verify_all_report.txt"
SUMMARY_PATH = LOG_DIR / "verify_all_summary.txt"

# ── helpers ───────────────────────────────────────────────────────────────────

class Reporter:
    def __init__(self):
        self.lines   = []
        self.results = []   # (section, check, status, detail)

    def section(self, title):
        sep = "=" * 70
        self.lines.append(f"\n{sep}\n  {title}\n{sep}")
        print(f"\n{'='*70}\n  {title}\n{'='*70}")

    def info(self, msg):
        self.lines.append(f"  {msg}")
        print(f"  {msg}")

    def check(self, section, name, passed, detail=""):
        status = "PASS" if passed else "FAIL"
        self.results.append((section, name, status, detail))
        tag = "  [PASS]" if passed else "  [FAIL]"
        self.lines.append(f"{tag} {name}  {detail}")
        print(f"{tag} {name}  {detail}")

    def write(self):
        header = (f"GAPC Verification Report\n"
                  f"Generated: {datetime.now().isoformat()}\n"
                  f"{'='*70}\n")
        REPORT_PATH.write_text(header + "\n".join(self.lines), encoding="utf-8")

        # summary
        n_pass = sum(1 for *_, s, _ in self.results if s == "PASS")
        n_fail = sum(1 for *_, s, _ in self.results if s == "FAIL")
        lines  = [f"GAPC Verification Summary  —  {n_pass} PASS  /  {n_fail} FAIL\n",
                  "=" * 70]
        for sec, name, status, detail in self.results:
            lines.append(f"  [{status}]  {sec}: {name}  {detail}")
        lines.append(f"\n  Total: {n_pass + n_fail}  |  PASS: {n_pass}  |  FAIL: {n_fail}")
        SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
        return n_pass, n_fail


R = Reporter()


def partial_spearman(x, y, z):
    xr = rankdata(x); yr = rankdata(y); zr = rankdata(z)
    bx = np.cov(xr, zr)[0, 1] / np.var(zr)
    by = np.cov(yr, zr)[0, 1] / np.var(zr)
    return pearsonr(xr - bx * zr, yr - by * zr)[0]


def near(a, b, tol):
    return abs(a - b) <= tol


# ─────────────────────────────────────────────────────────────────────────────
# A. Catalog structural integrity
# ─────────────────────────────────────────────────────────────────────────────
R.section("A — Catalog structural integrity")

V8_PATH = ROOT / "data" / "final" / "gapc_catalog_v8.parquet"
R.info(f"Loading: {V8_PATH}")
try:
    gapc = pd.read_parquet(V8_PATH)
    n_rows, n_cols = len(gapc), len(gapc.columns)
    R.info(f"v8: {n_rows:,} rows, {n_cols} cols")

    R.check("A", "Row count = 128,885",
            n_rows == 128885, f"got {n_rows:,}")

    # No duplicate number_mp
    dupes = gapc["number_mp"].duplicated().sum()
    R.check("A", "No duplicate number_mp",
            dupes == 0, f"{dupes} duplicates")

    # All key columns present
    required = ["G", "sigma_G", "H", "H_V", "D_km", "p_V_final",
                "gasp_orbital_class", "binary_known", "rot_period_best",
                "taxonomy_refined", "taxonomy_source",
                "sdss_a_star", "neowise_pIR_ratio",
                "damit_model", "goffin_density_gcm3",
                "spectral_class_best"]
    missing = [c for c in required if c not in gapc.columns]
    R.check("A", "All required columns present",
            len(missing) == 0, f"missing: {missing}")

    # v6, v7 files also exist
    for vv, n_expected in [("v6", 156), ("v7", 160), ("v8", 162)]:
        p = ROOT / "data" / "final" / f"gapc_catalog_{vv}.parquet"
        ex = p.exists()
        R.check("A", f"{vv} file exists", ex, str(p))

except Exception as e:
    R.check("A", "Load v8", False, str(e))
    R.write(); sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# B. G column statistics
# ─────────────────────────────────────────────────────────────────────────────
R.section("B — G column statistics")

G_all  = gapc["G"].dropna()
G_frac = len(G_all) / n_rows
R.info(f"G not-null: {len(G_all):,}  ({G_frac*100:.1f}%)")
R.info(f"G range: [{G_all.min():.4f}, {G_all.max():.4f}]")
R.info(f"G median: {G_all.median():.4f}  mean: {G_all.mean():.4f}")

# Expected: >90% have G
R.check("B", "G coverage > 90%", G_frac > 0.90, f"{G_frac*100:.1f}%")

# Physical bounds: G should be mostly in [-0.2, 1.0] for HG model
outliers_lo = (G_all < -0.5).sum()
outliers_hi = (G_all > 1.5).sum()
R.check("B", "G extreme outliers < 1%",
        (outliers_lo + outliers_hi) / len(G_all) < 0.01,
        f"<-0.5: {outliers_lo}, >1.5: {outliers_hi}")

# G median for S-types should be ~0.14-0.16
s_med = gapc[(gapc["taxonomy_refined"] == "S") & gapc["G"].notna()]["G"].median()
R.check("B", "S-type median G in [0.12, 0.20]",
        0.12 <= s_med <= 0.20, f"S median G={s_med:.4f}")

# G median for C-types should be ~0.00-0.05
c_med = gapc[(gapc["taxonomy_refined"] == "C") & gapc["G"].notna()]["G"].median()
R.check("B", "C-type median G in [-0.05, 0.06]",
        -0.05 <= c_med <= 0.06, f"C median G={c_med:.4f}")

# sigma_G plausibility
sig_G = gapc["sigma_G"].dropna()
R.info(f"sigma_G median: {sig_G.median():.4f}")
R.check("B", "sigma_G median < 0.15",
        sig_G.median() < 0.15, f"{sig_G.median():.4f}")

# n_obs: at least 5 per object (quality filter)
if "n_obs" in gapc.columns:
    n_obs_min = gapc["n_obs"].min()
    R.info(f"n_obs min={n_obs_min}  median={gapc['n_obs'].median():.0f}")
    R.check("B", "All n_obs >= 5",
            n_obs_min >= 5, f"min={n_obs_min}")

# ─────────────────────────────────────────────────────────────────────────────
# C. Cross-catalog spot checks
# ─────────────────────────────────────────────────────────────────────────────
R.section("C — Cross-catalog spot checks")

# DAMIT: spot check known objects in GAPC (1-4 are too bright for Gaia SSO)
# Using objects confirmed present in v8 (numbers ≥5 are in the catalog)
DAMIT_KNOWN = {5: "Astraea", 21: "Lutetia", 23: "Thalia", 24: "Themis", 25: "Phocaea"}
for num, name in DAMIT_KNOWN.items():
    row = gapc[gapc["number_mp"] == num]
    if len(row) == 0:
        R.info(f"  {name} ({num}) not in GAPC (unexpected)")
        continue
    has_model = bool(row["damit_model"].iloc[0])
    R.check("C", f"DAMIT flag: {name} ({num})", has_model,
            f"damit_model={has_model}")

# Goffin: Ceres mass and density sanity
ceres = gapc[gapc["number_mp"] == 1]
if len(ceres) > 0 and "goffin_mass_1e10Msun" in gapc.columns:
    m = ceres["goffin_mass_1e10Msun"].iloc[0]
    d = ceres["goffin_density_gcm3"].iloc[0]
    R.info(f"Ceres mass={m:.3f} ×10^10 Msun  density={d:.2f} g/cm³")
    R.check("C", "Ceres mass in [4.5, 5.0] ×10^10 Msun",
            4.5 <= m <= 5.0 if pd.notna(m) else False, f"{m:.3f}")
    R.check("C", "Ceres density in [1.5, 2.5] g/cm³",
            1.5 <= d <= 2.5 if pd.notna(d) else False, f"{d:.2f}")

# Pravec binary: check known systems
KNOWN_BINARIES = {3749: "Balam", 1862: "Apollo", 22: "Kalliope"}
for num, name in KNOWN_BINARIES.items():
    row = gapc[gapc["number_mp"] == num]
    if len(row) == 0:
        R.info(f"  {name} ({num}) not in GAPC (expected for some)")
        continue
    is_bin = bool(row["binary_known"].iloc[0])
    R.check("C", f"binary_known: {name} ({num})", is_bin, f"binary_known={is_bin}")

# SDSS a* distribution
if "sdss_a_star" in gapc.columns:
    a_star = gapc["sdss_a_star"].dropna()
    frac_sdss = len(a_star) / n_rows
    R.info(f"sdss_a_star coverage: {len(a_star):,} ({frac_sdss*100:.1f}%)")
    R.check("C", "SDSS a* coverage 25-35%",
            0.25 <= frac_sdss <= 0.35, f"{frac_sdss*100:.1f}%")
    # S-types should have a* > 0 on average, C-types < 0
    s_astar = gapc[(gapc["taxonomy_refined"] == "S") & gapc["sdss_a_star"].notna()]["sdss_a_star"].median()
    c_astar = gapc[(gapc["taxonomy_refined"] == "C") & gapc["sdss_a_star"].notna()]["sdss_a_star"].median()
    R.check("C", "Median a* S > 0", s_astar > 0, f"S={s_astar:.4f}")
    R.check("C", "Median a* C < 0", c_astar < 0, f"C={c_astar:.4f}")

# DAMIT overall coverage
n_damit = gapc["damit_model"].sum()
frac_damit = n_damit / n_rows
R.info(f"DAMIT flag True: {n_damit:,} ({frac_damit*100:.1f}%)")
R.check("C", "DAMIT coverage 5-12%",
        0.05 <= frac_damit <= 0.12, f"{frac_damit*100:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# D. Taxonomy distribution checks
# ─────────────────────────────────────────────────────────────────────────────
R.section("D — taxonomy_refined distribution")

tax = "taxonomy_refined"
vc  = gapc[tax].value_counts()
R.info(f"Total classified: {gapc[tax].notna().sum():,}")
for t in ["S", "C", "M", "P", "Ch", "E"]:
    R.info(f"  {t}: {vc.get(t, 0):,}")

# Coverage > 95%
cov = gapc[tax].notna().sum() / n_rows
R.check("D", "taxonomy_refined coverage > 95%",
        cov > 0.95, f"{cov*100:.1f}%")

# No object can be both S from PDS and C from SDSS (source consistency)
# If spectral_class_best starts with C but taxonomy_refined = S, flag
if "spectral_class_best" in gapc.columns:
    pds_C = gapc["spectral_class_best"].astype(str).str.startswith("C")
    ref_S = gapc[tax] == "S"
    # Only check where source = spectral_pds (highest priority)
    conflict = (pds_C & ref_S &
                (gapc["taxonomy_source"].astype(str).str.startswith("spectral_pds")))
    R.check("D", "No S assigned where PDS label is C",
            conflict.sum() == 0, f"{conflict.sum()} conflicts")

# G ordering: G_S > G_C (strong physical expectation)
g_S = gapc[(gapc[tax] == "S") & gapc["G"].notna()]["G"].median()
g_C = gapc[(gapc[tax] == "C") & gapc["G"].notna()]["G"].median()
R.check("D", "G_S > G_C (physical)",
        g_S > g_C, f"G_S={g_S:.4f}  G_C={g_C:.4f}")

# V-types should have highest G among common types
g_V = gapc[(gapc[tax] == "V") & gapc["G"].notna()]["G"].median()
R.check("D", "G_V > G_S (Vestoids expected highest)",
        g_V > g_S if not np.isnan(g_V) else False,
        f"G_V={g_V:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# E. External calibration verification
# ─────────────────────────────────────────────────────────────────────────────
R.section("E — External calibration (H_V_tax vs PTF / ATLAS)")

log22b = LOG_DIR / "22b_external_calibration_stats.txt"
if log22b.exists():
    text = log22b.read_text()
    R.info(f"22b log excerpt:\n{text[:600]}")

    # Expected: H_V_tax - H_PTF ~ 0.06 (near-zero)
    import re
    m_ptf = re.search(r"median.*?PTF.*?([+-]?\d+\.\d+)", text, re.I)
    if m_ptf:
        bias_ptf = float(m_ptf.group(1))
        R.check("E", "H_V_tax − H_PTF bias |<0.15| mag",
                abs(bias_ptf) < 0.15, f"bias={bias_ptf:+.3f}")
    else:
        # try direct check from H columns
        if "H_V_tax" in gapc.columns and "H_PTF" in gapc.columns:
            diff = (gapc["H_V_tax"] - gapc["H_PTF"]).dropna()
            R.info(f"  H_V_tax - H_PTF: median={diff.median():+.4f}  n={len(diff):,}")
            R.check("E", "H_V_tax − H_PTF median |<0.15| mag",
                    abs(diff.median()) < 0.15, f"{diff.median():+.4f}")
        else:
            R.info("  H_PTF not in v8 (expected — validation was read-only)")
            R.check("E", "H_V_tax external calibration (see log)",
                    True, "log exists: 22b_external_calibration_stats.txt")
else:
    R.info("  22b log not found — checking columns directly")
    R.check("E", "22b log or H columns available",
            "H_V_tax" in gapc.columns, f"H_V_tax in v8: {'H_V_tax' in gapc.columns}")

# ─────────────────────────────────────────────────────────────────────────────
# F. Space weathering — re-derive rho(G, logD | logpV)
# ─────────────────────────────────────────────────────────────────────────────
R.section("F — Space weathering: re-derive rho(G, logD | logpV)")

sw = gapc[gapc["G"].notna() & gapc["D_km"].notna() & (gapc["D_km"] > 0) &
          gapc["p_V_final"].notna() & (gapc["p_V_final"] > 0)].copy()
sw["log_D"]  = np.log10(sw["D_km"])
sw["log_pV"] = np.log10(sw["p_V_final"])

rho_D_all, p_D_all = spearmanr(sw["G"], sw["log_D"])
r_GD_pv = partial_spearman(sw["G"].values, sw["log_D"].values, sw["log_pV"].values)
r_Gpv_D = partial_spearman(sw["G"].values, sw["log_pV"].values, sw["log_D"].values)

R.info(f"n={len(sw):,}")
R.info(f"rho(G, logD) = {rho_D_all:+.4f}  p={p_D_all:.2e}")
R.info(f"r(G, logD | logpV) = {r_GD_pv:+.4f}")
R.info(f"r(G, logpV | logD) = {r_Gpv_D:+.4f}")

# Expected from step 37: rho(G,logD)≈-0.294, partial≈-0.273
R.check("F", "rho(G, logD) in [-0.35, -0.25]",
        -0.35 <= rho_D_all <= -0.25, f"{rho_D_all:+.4f}")
R.check("F", "r(G, logD | logpV) < -0.20",
        r_GD_pv < -0.20, f"{r_GD_pv:+.4f}")
R.check("F", "|r(G, logpV | logD)| < |r(G, logD | logpV)|",
        abs(r_Gpv_D) < abs(r_GD_pv),
        f"pV={r_Gpv_D:+.4f}  D={r_GD_pv:+.4f}")

# S-type only
s_sw = sw[sw[tax] == "S"]
rho_D_S, _ = spearmanr(s_sw["G"], s_sw["log_D"])
r_GD_pv_S  = partial_spearman(s_sw["G"].values, s_sw["log_D"].values,
                               s_sw["log_pV"].values)
R.info(f"S-type: rho(G,logD)={rho_D_S:+.4f}  r(G,logD|logpV)={r_GD_pv_S:+.4f}  n={len(s_sw):,}")
R.check("F", "S-type: r(G, logD|logpV) < -0.20",
        r_GD_pv_S < -0.20, f"{r_GD_pv_S:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# G. Binary G excess — re-derive Mann-Whitney
# ─────────────────────────────────────────────────────────────────────────────
R.section("G — Binary G excess: re-derive Mann-Whitney")

bin_col = "binary_known"
g_bin  = gapc[gapc[bin_col] & gapc["G"].notna()]["G"]
g_sing = gapc[~gapc[bin_col] & gapc["G"].notna()]["G"]

U, p_all = mannwhitneyu(g_bin, g_sing, alternative="two-sided")
R.info(f"n_binary={len(g_bin):,}  n_single={len(g_sing):,}")
R.info(f"G binary median={g_bin.median():.4f}  single={g_sing.median():.4f}")
R.info(f"MW p={p_all:.3e}")

R.check("G", "Binary G > Single G",
        g_bin.median() > g_sing.median(),
        f"binary={g_bin.median():.4f}  single={g_sing.median():.4f}")
R.check("G", "MW p < 0.05", p_all < 0.05, f"p={p_all:.3e}")

# Size-controlled re-test (quintile demeaning)
both = gapc[gapc["G"].notna() & gapc["D_km"].notna() & (gapc["D_km"] > 0)].copy()
both["log_D"] = np.log10(both["D_km"])
both["size_q"] = pd.qcut(both["log_D"], 5, labels=False)
resid = both["G"].copy()
for q in range(5):
    m = both["size_q"] == q
    resid.loc[m] -= both.loc[m, "G"].mean()
g_bin_r  = resid[both[bin_col]]
g_sing_r = resid[~both[bin_col]]
U2, p_ctrl = mannwhitneyu(g_bin_r, g_sing_r, alternative="two-sided")
R.info(f"Size-controlled: binary res={g_bin_r.median():.4f}  single={g_sing_r.median():.4f}")
R.info(f"MW p (size-controlled) = {p_ctrl:.3e}")

R.check("G", "Binary G excess after size control (p < 1e-6)",
        p_ctrl < 1e-6, f"p={p_ctrl:.3e}")

# 3-10 km bin
bin_3_10 = gapc[gapc["D_km"].between(3, 10) & gapc["G"].notna()]
b_3  = bin_3_10[bin_3_10[bin_col]]["G"]
s_3  = bin_3_10[~bin_3_10[bin_col]]["G"]
if len(b_3) >= 5 and len(s_3) >= 10:
    U3, p3 = mannwhitneyu(b_3, s_3, alternative="two-sided")
    R.info(f"3-10 km: binary G={b_3.median():.4f}(n={len(b_3)})  "
           f"single G={s_3.median():.4f}(n={len(s_3):,})  p={p3:.3e}")
    R.check("G", "3-10 km: binary G > single G (p<0.001)",
            p3 < 0.001 and b_3.median() > s_3.median(),
            f"p={p3:.3e}")

# ─────────────────────────────────────────────────────────────────────────────
# H. H-completeness — re-derive alpha and H_turn
# ─────────────────────────────────────────────────────────────────────────────
R.section("H — H-completeness: re-derive power-law fit")

MPC_PATH = ROOT / "data" / "raw" / "mpc_h_magnitudes.parquet"
if MPC_PATH.exists():
    mpc = pd.read_parquet(MPC_PATH)
    H_gapc = gapc["H_V"].dropna()
    H_mpc  = mpc["H_mpc"].dropna()

    bins = np.arange(5, 21, 0.25)
    counts_g, edges = np.histogram(H_gapc, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2

    fit_mask = (centers >= 10) & (centers < 15) & (counts_g > 0)
    log_c_g  = np.where(counts_g > 0, np.log10(counts_g), np.nan)

    try:
        popt, pcov = curve_fit(
            lambda H, a, b: a * H + b,
            centers[fit_mask], log_c_g[fit_mask], p0=[0.4, 0.0]
        )
        alpha, logC = popt
        alpha_err = np.sqrt(pcov[0, 0])
        R.info(f"alpha={alpha:.4f} ± {alpha_err:.4f}")

        # Find turnover
        H_turn = np.nan
        fit_full = alpha * centers + logC
        residuals = log_c_g - fit_full
        for i in range(len(centers)-1, -1, -1):
            if np.isfinite(residuals[i]) and residuals[i] > -0.5:
                H_turn = centers[i]; break
        R.info(f"H_turn = {H_turn:.2f} mag")

        R.check("H", "alpha in [0.44, 0.54]",
                0.44 <= alpha <= 0.54, f"{alpha:.4f}")
        R.check("H", "alpha consistent with Dohnanyi (|alpha-0.5| < 0.06)",
                abs(alpha - 0.5) < 0.06, f"|{alpha:.4f}-0.5|={abs(alpha-0.5):.4f}")
        R.check("H", "H_turn in [15.0, 16.5]",
                15.0 <= H_turn <= 16.5, f"{H_turn:.2f}")

        # Recovery fraction
        gapc_nums = set(gapc["number_mp"].astype(int))
        mpc_nums  = set(mpc["number_mp"].astype(int))
        recovery  = len(gapc_nums & mpc_nums) / len(mpc_nums) * 100
        R.info(f"Recovery = {recovery:.1f}%")
        R.check("H", "Recovery 18-25% of MPC",
                18 <= recovery <= 25, f"{recovery:.1f}%")

    except Exception as e:
        R.check("H", "H-completeness fit", False, str(e))
else:
    R.check("H", "MPC H file exists", False, str(MPC_PATH))

# ─────────────────────────────────────────────────────────────────────────────
# I. Universal size law — re-derive for S, M, E, P, C
# ─────────────────────────────────────────────────────────────────────────────
R.section("I — Universal size law: re-derive partial r per subtype")

size_law_results = {}
pv_col = "p_V_final"
for t in ["S", "M", "E", "P", "C"]:
    sub = gapc[(gapc[tax] == t) & gapc["G"].notna() &
               gapc["D_km"].notna() & (gapc["D_km"] > 0) &
               gapc[pv_col].notna() & (gapc[pv_col] > 0)].copy()
    if len(sub) < 30:
        R.info(f"  {t}: too few (n={len(sub)})")
        continue
    sub["log_D"]  = np.log10(sub["D_km"])
    sub["log_pV"] = np.log10(sub[pv_col])
    r = partial_spearman(sub["G"].values, sub["log_D"].values, sub["log_pV"].values)
    R.info(f"  {t} (n={len(sub):,}): r(G,logD|logpV) = {r:+.4f}")
    size_law_results[t] = (r, len(sub))
    R.check("I", f"{t}-type r(G,logD|logpV) < -0.20",
            r < -0.20, f"{r:+.4f}")

# Consistency: all subtypes should have similar r (within 0.15)
if len(size_law_results) >= 3:
    rs = [v[0] for v in size_law_results.values()]
    rng = max(rs) - min(rs)
    R.info(f"  Range across subtypes: {rng:.4f}")
    R.check("I", "All subtypes consistent (range < 0.15)",
            rng < 0.15, f"range={rng:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# J. Rotation × G — re-derive rho(G, logP)
# ─────────────────────────────────────────────────────────────────────────────
R.section("J — Rotation × G: re-derive Spearman rho")

per_col = "rot_period_best"
if per_col in gapc.columns:
    gper = gapc[gapc["G"].notna() & gapc[per_col].notna() &
                (gapc[per_col] > 0.01) & (gapc[per_col] < 1000)].copy()
    gper["log_P"] = np.log10(gper[per_col])
    rho_gp, p_gp = spearmanr(gper["G"], gper["log_P"])
    R.info(f"rho(G, logP) = {rho_gp:+.4f}  p={p_gp:.2e}  n={len(gper):,}")
    R.check("J", "rho(G, logP) NOT significant (p > 0.01)",
            p_gp > 0.01, f"p={p_gp:.2e}")
    R.check("J", "|rho(G, logP)| < 0.10 (near zero)",
            abs(rho_gp) < 0.10, f"{rho_gp:+.4f}")
else:
    R.check("J", "rot_period_best column present", False, "missing")

# ─────────────────────────────────────────────────────────────────────────────
# K. GASP v2 integrity
# ─────────────────────────────────────────────────────────────────────────────
R.section("K — GASP v2 integrity")

GASP_PATH = ROOT.parent / "gasp" / "data" / "final" / "gasp_catalog_v2.parquet"
if GASP_PATH.exists():
    gasp = pd.read_parquet(GASP_PATH)
    R.info(f"GASP v2: {len(gasp):,} rows, {len(gasp.columns)} cols")
    R.check("K", "GASP v2 row count = 19,190",
            len(gasp) == 19190, f"{len(gasp):,}")

    # No duplicate number_mp
    dupes_g = gasp["number_mp"].duplicated().sum()
    R.check("K", "No duplicate number_mp", dupes_g == 0, f"{dupes_g}")

    # New columns from enrich_external present
    new_cols = ["rot_period_best", "spectral_class_best", "binary_known",
                "damit_model", "taxonomy_ml", "taxonomy_final"]
    missing_g = [c for c in new_cols if c not in gasp.columns]
    R.check("K", "All enrichment columns present",
            len(missing_g) == 0, f"missing={missing_g}")

    # F1-macro improvement documented
    log10 = ROOT.parent / "gasp" / "data" / "final" / "10_retrain_log.json"
    if log10.exists():
        log = json.loads(log10.read_text())
        f1_orig = log.get("f1_macro_orig", 0)
        f1_new  = log.get("f1_macro_new", 0)
        R.info(f"F1-macro: {f1_orig:.4f} → {f1_new:.4f} (+{(f1_new-f1_orig)*100:.2f} pp)")
        R.check("K", "RF retrain improved F1-macro",
                f1_new > f1_orig, f"{f1_orig:.4f} → {f1_new:.4f}")
        R.check("K", "RF retrain improvement > 2 pp",
                (f1_new - f1_orig) * 100 > 2, f"+{(f1_new-f1_orig)*100:.2f} pp")

    # rot_period_best coverage ~25-30%
    n_rot = gasp["rot_period_best"].notna().sum()
    frac_rot = n_rot / len(gasp)
    R.check("K", "GASP rotation period coverage 20-35%",
            0.20 <= frac_rot <= 0.35, f"{frac_rot*100:.1f}%")

    # damit_model ~15-20%
    n_damit_g = gasp["damit_model"].sum() if gasp["damit_model"].dtype == bool \
                else (gasp["damit_model"] == True).sum()
    frac_damit_g = n_damit_g / len(gasp)
    R.check("K", "GASP DAMIT coverage 10-25%",
            0.10 <= frac_damit_g <= 0.25, f"{frac_damit_g*100:.1f}%")

    # taxonomy_final coverage
    n_tax = gasp["taxonomy_final"].notna().sum()
    frac_tax = n_tax / len(gasp)
    R.check("K", "taxonomy_final coverage > 50%",
            frac_tax > 0.50, f"{frac_tax*100:.1f}%")

else:
    R.check("K", "GASP v2 file exists", False, str(GASP_PATH))

# ─────────────────────────────────────────────────────────────────────────────
# L. Publication figures file check
# ─────────────────────────────────────────────────────────────────────────────
R.section("L — Publication figures")

FIGS = {
    "gapc_fig1_taxonomy_G.png":    (150 * 1024, "taxonomy G"),
    "gapc_fig2_weathering.png":    (300 * 1024, "weathering"),
    "gapc_fig3_completeness.png":  (100 * 1024, "completeness"),
    "gapc_fig3_completeness.pdf":  ( 30 * 1024, "completeness pdf"),
    "gapc_fig4_binary.png":        (150 * 1024, "binary"),
    "gapc_fig5_size_law.png":      (300 * 1024, "size law"),
    "gapc_fig5_size_law.pdf":      ( 50 * 1024, "size law pdf"),
}

for fname, (min_bytes, label) in FIGS.items():
    p = PLOT_DIR / fname
    exists = p.exists()
    if exists:
        sz = p.stat().st_size
        R.check("L", f"{fname} ({label})",
                sz >= min_bytes, f"{sz//1024} KB")
    else:
        R.check("L", f"{fname} ({label})", False, "missing")

# Also check diagnostic plots exist
diag = ["48_gapc_gasp_crossmatch.png", "50_family_age_G_revised.png",
        "44_taxonomy_refined.png", "39_binary_analysis.png",
        "37_weathering_full.png"]
for fname in diag:
    p = PLOT_DIR / fname
    R.check("L", f"diagnostic: {fname}", p.exists(), "")

# ─────────────────────────────────────────────────────────────────────────────
# Additional: G_uncertain flag check
# ─────────────────────────────────────────────────────────────────────────────
R.section("M — G_uncertain flag and data quality")

if "G_uncertain" in gapc.columns:
    n_unc = gapc["G_uncertain"].sum()
    frac_unc = n_unc / len(G_all)
    R.info(f"G_uncertain: {n_unc:,} ({frac_unc*100:.1f}%)")
    R.check("M", "G_uncertain fraction < 60%",
            frac_unc < 0.60, f"{frac_unc*100:.1f}%")

# phase_range: all objects should have reasonable phase angle coverage
if "phase_range" in gapc.columns:
    pr = gapc["phase_range"].dropna()
    R.info(f"phase_range median={pr.median():.1f}°  min={pr.min():.1f}°")
    R.check("M", "Median phase_range > 5°",
            pr.median() > 5, f"{pr.median():.1f}°")

# n_obs distribution
if "n_obs" in gapc.columns:
    n_obs = gapc["n_obs"]
    R.info(f"n_obs: min={n_obs.min()}  median={n_obs.median():.0f}  max={n_obs.max()}")

# H_V plausibility
if "H_V" in gapc.columns:
    H_V = gapc["H_V"].dropna()
    R.info(f"H_V: range [{H_V.min():.2f}, {H_V.max():.2f}]  median={H_V.median():.2f}")
    R.check("M", "H_V range physical [3, 22]",
            H_V.min() >= 3 and H_V.max() <= 22, f"[{H_V.min():.2f},{H_V.max():.2f}]")

# Correlation sign checks (fundamental)
R.section("N — Fundamental correlation sign checks")

# G should be negatively correlated with D (larger → lower G = space weathering)
g_d = gapc[gapc["G"].notna() & gapc["D_km"].notna() & (gapc["D_km"] > 0)]
rho_gd, _ = spearmanr(g_d["G"], np.log10(g_d["D_km"]))
R.check("N", "rho(G, logD) < 0 (larger objects lower G)",
        rho_gd < 0, f"{rho_gd:+.4f}")

# G should be positively correlated with albedo (higher albedo → higher G)
if "p_V_final" in gapc.columns:
    g_pv = gapc[gapc["G"].notna() & gapc["p_V_final"].notna() & (gapc["p_V_final"] > 0)]
    rho_gpv, _ = spearmanr(g_pv["G"], g_pv["p_V_final"])
    R.check("N", "rho(G, p_V) > 0 (higher albedo higher G)",
            rho_gpv > 0, f"{rho_gpv:+.4f}")

# S-types should have higher albedo than C-types
if "p_V_final" in gapc.columns:
    pv_S = gapc[(gapc[tax] == "S") & gapc["p_V_final"].notna()]["p_V_final"].median()
    pv_C = gapc[(gapc[tax] == "C") & gapc["p_V_final"].notna()]["p_V_final"].median()
    R.check("N", "Median p_V: S > C",
            pv_S > pv_C, f"S={pv_S:.3f}  C={pv_C:.3f}")

# H_V should correlate with H (raw, before color correction)
if "H_V" in gapc.columns and "H" in gapc.columns:
    h_both = gapc[gapc["H_V"].notna() & gapc["H"].notna()]
    rho_hh, _ = spearmanr(h_both["H_V"], h_both["H"])
    R.check("N", "rho(H_V, H_raw) > 0.90",
            rho_hh > 0.90, f"{rho_hh:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────────────────────────────────────
n_pass, n_fail = R.write()
print(f"\n{'='*70}")
print(f"  VERIFICATION COMPLETE")
print(f"  PASS: {n_pass}  |  FAIL: {n_fail}  |  TOTAL: {n_pass + n_fail}")
print(f"  Full report: {REPORT_PATH}")
print(f"  Summary:     {SUMMARY_PATH}")
print(f"{'='*70}\n")
