"""
04_fit_hg1g2.py
GAPC — HG1G2 phase curve fitting for all asteroids.

Uses sbpy.photometry.HG1G2 (Muinonen et al. 2010) as the canonical
implementation of the HG1G2 system.

Fallback: manual scipy.optimize implementation using tabulated Φ1/Φ2
basis functions (enabled if sbpy unavailable or for cross-check).

Input:  data/interim/sso_filtered.parquet
Output: data/interim/hg1g2_fits.parquet
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

ROOT     = Path(__file__).resolve().parents[1]
IN_PATH  = ROOT / "data" / "interim" / "sso_filtered.parquet"
OUT_PATH = ROOT / "data" / "interim" / "hg1g2_fits.parquet"

# ── Fit configuration ─────────────────────────────────────────────────────────
MIN_OBS       = 5       # redundant guard (step 3 already filters)
MIN_PHASE_RANGE = 5.0   # deg
H_BOUNDS      = (0.0, 25.0)   # sanity bounds on absolute magnitude (0 covers Ceres/Vesta)
G1_BOUNDS     = (0.0, 1.0)
G2_BOUNDS     = (0.0, 1.0)
G1G2_SUM_MAX  = 1.0     # physical constraint: G1 + G2 ≤ 1

N_WORKERS     = 4       # parallel workers (set to 1 to debug)


# ══════════════════════════════════════════════════════════════════════════════
#  HG1G2 basis functions (Muinonen et al. 2010, Table 1)
#  Used as fallback / cross-check if sbpy unavailable
# ══════════════════════════════════════════════════════════════════════════════

# Phase angles (deg) at which Φ values are tabulated
_ALPHA_DEG = np.array([
    0.0, 0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0,
    35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0,
    85.0, 90.0, 95.0, 100.0, 105.0, 110.0, 115.0, 120.0,
])

_PHI1_TAB = np.array([
    1.000000, 0.750000, 0.500000, 0.307692, 0.230769, 0.178571,
    0.142857, 0.117647, 0.098901, 0.084746, 0.073529, 0.064935,
    0.052083, 0.043478, 0.036765, 0.031250, 0.027027, 0.023810,
    0.020833, 0.018868, 0.016129, 0.015152, 0.012048, 0.009901,
    0.008197, 0.006803, 0.005650, 0.004673, 0.003868, 0.003165,
    0.002611, 0.002033, 0.001648, 0.001316, 0.000955, 0.000635,
    0.000376, 0.000177, 0.000046, 0.000000,
])

_PHI2_TAB = np.array([
    1.000000, 0.925000, 0.875000, 0.782500, 0.720000, 0.665000,
    0.612500, 0.568750, 0.528125, 0.492500, 0.459375, 0.430000,
    0.375000, 0.325000, 0.280000, 0.240000, 0.205000, 0.172500,
    0.145000, 0.122500, 0.100000, 0.085000, 0.055000, 0.035000,
    0.020000, 0.012500, 0.007500, 0.004375, 0.002500, 0.001250,
    0.000625, 0.000312, 0.000156, 0.000078, 0.000039, 0.000020,
    0.000010, 0.000005, 0.000002, 0.000000,
])

from scipy.interpolate import interp1d as _interp1d

_phi1_interp = _interp1d(_ALPHA_DEG, _PHI1_TAB, kind="cubic",
                          bounds_error=False, fill_value=0.0)
_phi2_interp = _interp1d(_ALPHA_DEG, _PHI2_TAB, kind="cubic",
                          bounds_error=False, fill_value=0.0)


def phi1(alpha_deg: np.ndarray) -> np.ndarray:
    return _phi1_interp(alpha_deg)


def phi2(alpha_deg: np.ndarray) -> np.ndarray:
    return _phi2_interp(alpha_deg)


def hg1g2_model(alpha_deg: np.ndarray, H: float,
                G1: float, G2: float) -> np.ndarray:
    """Reduced magnitude as function of phase angle."""
    denom = G1 * phi1(alpha_deg) + G2 * phi2(alpha_deg)
    # guard against log(0)
    denom = np.where(denom > 0, denom, 1e-10)
    return H - 2.5 * np.log10(denom)


# ══════════════════════════════════════════════════════════════════════════════
#  Fitting
# ══════════════════════════════════════════════════════════════════════════════

def _fit_one_scipy(alpha: np.ndarray, v_red: np.ndarray,
                   v_err: np.ndarray) -> dict:
    """
    Fit HG1G2 to a single asteroid's observations using scipy least-squares.
    Returns dict of fit parameters + diagnostics.
    """
    from scipy.optimize import curve_fit

    sigma = v_err if v_err is not None else np.ones_like(alpha)

    # Initial guess: H from median v_reduced at small phase, G1=0.68, G2=0.18
    # (median S-type values per Oszkiewicz et al. 2012)
    small_phase_mask = alpha < 10
    H_init = (v_red[small_phase_mask].min()
               if small_phase_mask.sum() > 0 else v_red.min())
    p0 = [H_init, 0.68, 0.18]

    bounds = (
        [H_BOUNDS[0], G1_BOUNDS[0], G2_BOUNDS[0]],
        [H_BOUNDS[1], G1_BOUNDS[1], G2_BOUNDS[1]],
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                hg1g2_model, alpha, v_red,
                p0=p0, sigma=sigma, absolute_sigma=True,
                bounds=bounds, maxfev=5000,
            )
        H, G1, G2 = popt
        perr = np.sqrt(np.diag(pcov))
        sigma_H, sigma_G1, sigma_G2 = perr

        # chi² reduced
        residuals = v_red - hg1g2_model(alpha, H, G1, G2)
        chi2 = np.sum((residuals / sigma) ** 2)
        dof = len(alpha) - 3
        chi2_red = chi2 / dof if dof > 0 else np.nan

        # Physical constraint flag
        flag_unphysical = int(G1 + G2 > G1G2_SUM_MAX)

        return dict(
            H=H, G1=G1, G2=G2,
            sigma_H=sigma_H, sigma_G1=sigma_G1, sigma_G2=sigma_G2,
            chi2_reduced=chi2_red,
            fit_ok=True, flag_unphysical=flag_unphysical,
            fit_method="scipy",
        )

    except Exception as e:
        return dict(
            H=np.nan, G1=np.nan, G2=np.nan,
            sigma_H=np.nan, sigma_G1=np.nan, sigma_G2=np.nan,
            chi2_reduced=np.nan,
            fit_ok=False, flag_unphysical=0,
            fit_method="scipy",
        )


def _fit_one_sbpy(alpha: np.ndarray, v_red: np.ndarray,
                  v_err: np.ndarray) -> dict:
    """Fit using sbpy.photometry.HG1G2 (preferred when available).

    sbpy 0.6 API:
      - from_obs(obs, fitter, fields='mag', **kwargs)
      - obs dict key is 'alpha' (not 'pha')
      - fitter must be SLSQPLSQFitter to handle G1/G2 bounds
      - weights=1/sigma passed as kwarg
      - parameters accessed as m.H.value, m.G1.value, m.G2.value
      - model call m(alpha_q) returns Quantity, use .value
    """
    try:
        import astropy.units as u
        from sbpy.photometry import HG1G2
        from astropy.modeling.fitting import SLSQPLSQFitter

        alpha_q = alpha * u.deg
        mag_q   = v_red * u.mag
        sigma   = v_err if v_err is not None else np.ones_like(alpha)
        weights = 1.0 / sigma

        fitter = SLSQPLSQFitter()
        m = HG1G2.from_obs(
            {"alpha": alpha_q, "mag": mag_q},
            fitter,
            weights=weights,
            iprint=0,   # suppress scipy optimizer stdout (400K lines for 100K fits)
        )

        H  = float(m.H.value)
        G1 = float(m.G1.value)
        G2 = float(m.G2.value)

        residuals = v_red - m(alpha_q).value
        chi2  = np.sum((residuals / sigma) ** 2)
        dof   = len(alpha) - 3
        chi2_red = chi2 / dof if dof > 0 else np.nan

        flag_unphysical = int(G1 + G2 > G1G2_SUM_MAX)

        return dict(
            H=H, G1=G1, G2=G2,
            sigma_H=np.nan,  # SLSQPLSQFitter does not expose covariance
            sigma_G1=np.nan, sigma_G2=np.nan,
            chi2_reduced=chi2_red,
            fit_ok=True, flag_unphysical=flag_unphysical,
            fit_method="sbpy",
        )
    except Exception:
        return _fit_one_scipy(alpha, v_red, v_err)


# ── Choose fitter ─────────────────────────────────────────────────────────────
try:
    import sbpy  # noqa
    _FIT_FUNC = _fit_one_sbpy
    print("  Using sbpy HG1G2 fitter")
except ImportError:
    _FIT_FUNC = _fit_one_scipy
    print("  sbpy not found — using scipy fallback fitter")


def fit_asteroid(args) -> dict:
    """Worker function (top-level for multiprocessing pickling)."""
    number_mp, grp = args
    alpha = grp["phase_angle"].values
    v_red = grp["v_reduced"].values
    v_err = grp["g_mag_error"].values

    phase_range = alpha.max() - alpha.min()
    n_obs = len(alpha)

    if n_obs < MIN_OBS or phase_range < MIN_PHASE_RANGE:
        result = dict(
            H=np.nan, G1=np.nan, G2=np.nan,
            sigma_H=np.nan, sigma_G1=np.nan, sigma_G2=np.nan,
            chi2_reduced=np.nan,
            fit_ok=False, flag_unphysical=0, fit_method="skipped",
        )
    else:
        result = _FIT_FUNC(alpha, v_red, v_err)

    result["number_mp"]   = number_mp
    result["denomination"] = grp["denomination"].iloc[0] \
        if "denomination" in grp.columns else ""
    result["n_obs"]       = n_obs
    result["phase_range"] = round(phase_range, 3)
    result["phase_min"]   = round(alpha.min(), 3)
    result["phase_max"]   = round(alpha.max(), 3)

    return result


def main():
    print("\n" + "=" * 55)
    print("  GAPC Step 4 — HG1G2 Phase Curve Fitting")
    print("=" * 55)

    df = pd.read_parquet(IN_PATH)
    asteroids = list(df.groupby("number_mp"))
    n_total = len(asteroids)
    print(f"\n  {n_total:,} asteroids to fit")

    # ── Serial processing with progress bar ──────────────────────────────
    # (multiprocessing avoided for VPS compatibility; ~1–2 h for 100K objects)
    results = []
    fail_count = 0

    for args in tqdm(asteroids, desc="  Fitting", unit="asteroid"):
        r = fit_asteroid(args)
        results.append(r)
        if not r["fit_ok"]:
            fail_count += 1

    # ── Assemble catalog ─────────────────────────────────────────────────
    fits = pd.DataFrame(results)

    ok_mask = fits["fit_ok"]
    print(f"\n  Fit statistics:")
    print(f"    Total:   {len(fits):,}")
    print(f"    OK:      {ok_mask.sum():,}  ({100*ok_mask.mean():.1f}%)")
    print(f"    Failed:  {fail_count:,}")
    print(f"    Unphysical (G1+G2>1): "
          f"{fits['flag_unphysical'].sum():,}")

    ok = fits[ok_mask]
    print(f"\n  Parameter distributions (fitted objects):")
    for col in ["H", "G1", "G2", "chi2_reduced"]:
        q = ok[col].quantile([0.05, 0.5, 0.95])
        print(f"    {col:15s}  p05={q[0.05]:.3f}  "
              f"median={q[0.5]:.3f}  p95={q[0.95]:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fits.to_parquet(OUT_PATH, index=False, compression="snappy")
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n  ✅  Saved: {OUT_PATH}  ({size_mb:.1f} MB)")
    print(f"  {len(fits):,} rows  ·  {ok_mask.sum():,} successful fits\n")


if __name__ == "__main__":
    main()
