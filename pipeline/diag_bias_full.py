"""
diag_bias_full.py  —  GAPC vollständige Bias-Analyse

Vier Untersuchungen:
  1. Bias vs phase_range  → H-G-Degenerierung
  2. Bias für G an Grenzen (G=0 oder G=1)  → schlecht bestimmte Fits
  3. H mit erzwungenem G=0.15 (analytische Korrektur)  → reiner G-Effekt
  4. Residual-Struktur nach Korrektur  → erklärt ~alles?
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ROOT     = Path(__file__).resolve().parents[1]
CAT      = ROOT / "data" / "final"  / "gapc_catalog_v1.parquet"
MPC_H    = ROOT / "data" / "raw"    / "mpc_h_magnitudes.parquet"
PLOT_DIR = ROOT / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

G_MPC      = 0.15
G_BOUND_LO = 0.02   # "at lower bound"
G_BOUND_HI = 0.98   # "at upper bound"

# ── HG helpers ────────────────────────────────────────────────────────────────

def hg_phi(alpha_deg: np.ndarray, G: float) -> np.ndarray:
    a = np.deg2rad(np.asarray(alpha_deg, dtype=float))
    th = np.tan(a / 2.0)
    phi1 = np.exp(-3.332 * th ** 0.631)
    phi2 = np.exp(-1.862 * th ** 1.218)
    return (1.0 - G) * phi1 + G * phi2

def hg_model(alpha_deg, H, G):
    return H - 2.5 * np.log10(np.maximum(hg_phi(alpha_deg, G), 1e-10))

def predicted_dH(phase_min, phase_max, n_obs, G_fit):
    """Analytic ΔH = H_GAPC − H_MPC due to G difference."""
    alphas = np.linspace(phase_min, phase_max, max(int(n_obs), 5))
    phi_fit = hg_phi(alphas, G_fit)
    phi_mpc = hg_phi(alphas, G_MPC)
    return float(np.mean(2.5 * np.log10(phi_fit / np.maximum(phi_mpc, 1e-10))))

def h_forced_g015(H_fit, G_fit, phase_min, phase_max, n_obs):
    """H value we would obtain if G were forced to 0.15."""
    dH = predicted_dH(phase_min, phase_max, n_obs, G_fit)
    return H_fit - dH


# ── Load & merge ──────────────────────────────────────────────────────────────

cat = pd.read_parquet(CAT, columns=[
    "number_mp", "H", "G", "sigma_H", "chi2_reduced",
    "fit_ok", "fit_method", "phase_min", "phase_max", "phase_range", "n_obs",
])
mpc = pd.read_parquet(MPC_H)

hg = cat[cat["fit_ok"] & (cat["fit_method"] == "hg_scipy") & cat["G"].notna()].copy()
df = hg.merge(mpc[["number_mp", "H_mpc"]], on="number_mp", how="inner")
df = df.dropna(subset=["H", "H_mpc", "G"])
df["dH"] = df["H"] - df["H_mpc"]

print(f"Matched HG objects vs MPC: {len(df):,}")
print(f"Overall bias  median={df['dH'].median():.4f}  "
      f"mean={df['dH'].mean():.4f}  std={df['dH'].std():.4f} mag\n")


# ── 1. Bias vs phase_range ────────────────────────────────────────────────────

print("── 1. Bias vs phase_range ──")
bins_pr = [5, 7, 9, 11, 13, 15, 20, 30, 90]
for lo, hi in zip(bins_pr[:-1], bins_pr[1:]):
    sub = df[(df["phase_range"] >= lo) & (df["phase_range"] < hi)]
    if len(sub) < 10:
        continue
    print(f"  {lo:4.0f}–{hi:4.0f}°  n={len(sub):6,}  "
          f"median_dH={sub['dH'].median():+.4f}  "
          f"std_dH={sub['dH'].std():.4f} mag")


# ── 2. Bias for G at bounds ───────────────────────────────────────────────────

print("\n── 2. G at bounds ──")
at_lo  = df[df["G"] <= G_BOUND_LO]
at_hi  = df[df["G"] >= G_BOUND_HI]
mid    = df[(df["G"] > G_BOUND_LO) & (df["G"] < G_BOUND_HI)]

for label, sub in [("G ≤ 0.02 (lower bound)", at_lo),
                    ("G ≥ 0.98 (upper bound)", at_hi),
                    ("0.02 < G < 0.98 (interior)", mid)]:
    print(f"  {label}: n={len(sub):,}  "
          f"median_dH={sub['dH'].median():+.4f}  "
          f"std={sub['dH'].std():.4f} mag")


# ── 3. H with forced G=0.15 (analytic correction) ────────────────────────────

print("\n── 3. Analytic correction to G=0.15 ──")
print("  Computing per-object dH_predicted … ", end="", flush=True)
df["dH_predicted"] = df.apply(
    lambda r: predicted_dH(r["phase_min"], r["phase_max"], r["n_obs"], r["G"]),
    axis=1,
)
df["H_corrected"] = df["H"] - df["dH_predicted"]
df["dH_corrected"] = df["H_corrected"] - df["H_mpc"]
print("done")

print(f"  dH after G-correction:  "
      f"median={df['dH_corrected'].median():+.4f}  "
      f"mean={df['dH_corrected'].mean():+.4f}  "
      f"std={df['dH_corrected'].std():.4f} mag")
print(f"  Explained by G:  "
      f"{(1 - df['dH_corrected'].std()/df['dH'].std())*100:.1f}% variance reduction")
print(f"  Unexplained offset: {df['dH_corrected'].median():+.4f} mag")


# ── 4. Residual structure ─────────────────────────────────────────────────────

print("\n── 4. Residual (dH_corrected) vs predictors ──")
for col in ["phase_range", "n_obs", "H", "G"]:
    r = df[["dH_corrected", col]].corr().iloc[0, 1]
    print(f"  r(dH_corrected, {col:12s}) = {r:+.3f}")

# Is residual different for G-bound objects?
print("\n  Residual by G category:")
for label, sub in [("G ≤ 0.02", at_lo),
                    ("G ≥ 0.98", at_hi),
                    ("0.02–0.98", mid)]:
    s = df.loc[sub.index, "dH_corrected"]
    print(f"    {label}: median={s.median():+.4f}  std={s.std():.4f}")


# ── Figures ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle("GAPC H bias — full analysis", fontsize=13)

ALPHA = 0.08
S = 1

# 1a. dH vs phase_range
ax = axes[0, 0]
ax.scatter(df["phase_range"], df["dH"].clip(-3, 3), s=S, alpha=ALPHA,
           rasterized=True, color="steelblue")
bins = np.arange(5, 90, 1.5)
mids, meds, stds = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    sub = df[(df["phase_range"] >= lo) & (df["phase_range"] < hi)]["dH"]
    if len(sub) > 30:
        mids.append((lo + hi) / 2)
        meds.append(sub.median())
        stds.append(sub.std())
ax.plot(mids, meds, "r-", lw=2, label="median")
ax.fill_between(mids,
                np.array(meds) - np.array(stds),
                np.array(meds) + np.array(stds),
                alpha=0.2, color="r", label="±1σ")
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.set_xlabel("phase_range (°)")
ax.set_ylabel("H_GAPC − H_MPC (mag)")
ax.set_title("1. Bias vs phase coverage")
ax.legend(fontsize=8)
ax.set_ylim(-3, 3)

# 1b. std of dH vs phase_range
ax = axes[0, 1]
ax.plot(mids, stds, "b-o", ms=3)
ax.set_xlabel("phase_range (°)")
ax.set_ylabel("std(H_GAPC − H_MPC) (mag)")
ax.set_title("1. Scatter vs phase coverage\n(H-G degeneracy indicator)")
ax.axhline(df["dH"].std(), color="k", ls="--", lw=0.8, label="overall std")
ax.legend(fontsize=8)

# 2. G distribution with bound flags
ax = axes[0, 2]
ax.hist(df.loc[mid.index, "G"], bins=60, alpha=0.7, label=f"interior ({len(mid):,})",
        color="steelblue", density=True)
ax.axvline(G_MPC, color="r", lw=1.5, ls="--", label=f"G_MPC={G_MPC}")
ax.axvline(df["G"].median(), color="orange", lw=1.5, ls="-",
           label=f"median G_fit={df['G'].median():.3f}")
ax.set_xlabel("G_fit")
ax.set_ylabel("density")
ax.set_title(f"2. G distribution\n"
             f"(G≤0.02: {len(at_lo):,}  G≥0.98: {len(at_hi):,})")
ax.legend(fontsize=8)

# 3. Before/after correction
ax = axes[1, 0]
kw = dict(bins=80, histtype="step", density=True, lw=1.5)
ax.hist(df["dH"].clip(-3, 3), label="original", **kw, color="steelblue")
ax.hist(df["dH_corrected"].clip(-3, 3), label="after G-correction", **kw, color="orange")
ax.axvline(0, color="k", ls="--", lw=0.8)
ax.axvline(df["dH"].median(), color="steelblue", ls=":", lw=1.2)
ax.axvline(df["dH_corrected"].median(), color="orange", ls=":", lw=1.2)
ax.set_xlabel("H_GAPC − H_MPC (mag)")
ax.set_ylabel("density")
ax.set_title("3. Before vs after G=0.15 correction")
ax.legend(fontsize=8)
ax.set_xlim(-3, 3)

# 4. Residual vs phase_range
ax = axes[1, 1]
ax.scatter(df["phase_range"], df["dH_corrected"].clip(-3, 3),
           s=S, alpha=ALPHA, rasterized=True, color="orange")
mids2, meds2 = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    sub = df[(df["phase_range"] >= lo) & (df["phase_range"] < hi)]["dH_corrected"]
    if len(sub) > 30:
        mids2.append((lo + hi) / 2)
        meds2.append(sub.median())
ax.plot(mids2, meds2, "r-", lw=2, label="median")
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.axhline(df["dH_corrected"].median(), color="r", ls=":", lw=1.2,
           label=f"overall median={df['dH_corrected'].median():+.3f}")
ax.set_xlabel("phase_range (°)")
ax.set_ylabel("residual (mag)")
ax.set_title("4. Residual after G-correction vs phase_range")
ax.set_ylim(-3, 3)
ax.legend(fontsize=8)

# 5. Residual vs G
ax = axes[1, 2]
ax.scatter(df["G"], df["dH_corrected"].clip(-3, 3),
           s=S, alpha=ALPHA, rasterized=True, color="purple")
bins_g = np.arange(0, 1.05, 0.05)
mids_g, meds_g = [], []
for lo, hi in zip(bins_g[:-1], bins_g[1:]):
    sub = df[(df["G"] >= lo) & (df["G"] < hi)]["dH_corrected"]
    if len(sub) > 30:
        mids_g.append((lo + hi) / 2)
        meds_g.append(sub.median())
ax.plot(mids_g, meds_g, "r-", lw=2, label="median")
ax.axhline(0, color="k", ls="--", lw=0.8)
ax.axvline(G_MPC, color="b", ls=":", lw=1, label=f"G_MPC=0.15")
ax.set_xlabel("G_fit")
ax.set_ylabel("residual after correction (mag)")
ax.set_title("4. Residual vs G_fit")
ax.set_ylim(-3, 3)
ax.legend(fontsize=8)

plt.tight_layout()
out = PLOT_DIR / "diag_bias_full.png"
plt.savefig(out, dpi=130)
print(f"\n  Plot → {out}")
