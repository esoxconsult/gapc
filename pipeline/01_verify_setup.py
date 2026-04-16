"""
01_verify_setup.py
GAPC — Environment and connectivity check.
Run this first on a new machine / after venv changes.
"""

import sys
import importlib
from pathlib import Path

REQUIRED = [
    ("astroquery", "0.4.7"),
    ("astropy", "5.3"),
    ("pandas", "2.0"),
    ("numpy", "1.24"),
    ("scipy", "1.11"),
    ("pyarrow", "14.0"),
    ("sbpy", "0.4"),
    ("tqdm", "4.66"),
]

DATA_DIRS = [
    "data/raw",
    "data/interim",
    "data/final",
    "logs",
]

ROOT = Path(__file__).resolve().parents[1]


def check_python():
    major, minor = sys.version_info[:2]
    ok = major == 3 and minor >= 10
    status = "✅" if ok else "❌"
    print(f"  {status}  Python {major}.{minor} (need ≥3.10)")
    return ok


def check_packages():
    all_ok = True
    for pkg, min_ver in REQUIRED:
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  ✅  {pkg} {ver}")
        except ImportError:
            print(f"  ❌  {pkg} NOT FOUND  (need ≥{min_ver})")
            all_ok = False
    return all_ok


def check_dirs():
    all_ok = True
    for d in DATA_DIRS:
        p = ROOT / d
        p.mkdir(parents=True, exist_ok=True)
        exists = p.is_dir()
        status = "✅" if exists else "❌"
        print(f"  {status}  {d}/")
        if not exists:
            all_ok = False
    return all_ok


def check_gasp():
    """Check whether GASP catalog is accessible (needed for step 5)."""
    candidates = [
        ROOT.parent / "GASP" / "data" / "final" / "gasp_catalog_v1.parquet",
        Path.home() / "gasp" / "data" / "final" / "gasp_catalog_v1.parquet",
    ]
    for p in candidates:
        if p.exists():
            import pandas as pd
            df = pd.read_parquet(p, columns=["number_mp"])
            print(f"  ✅  GASP catalog found: {p}  ({len(df):,} rows)")
            return str(p)
    print("  ⚠️   GASP catalog not found at expected locations.")
    print("       Set GASP_CATALOG env var or update path in 05_crossmatch_gasp.py")
    return None


def check_gaia_tap():
    """Quick connectivity check to Gaia TAP."""
    try:
        from astroquery.gaia import Gaia
        job = Gaia.launch_job(
            "SELECT COUNT(*) as n FROM gaiadr3.sso_observation",
            verbose=False,
        )
        result = job.get_results()
        n = int(result["n"][0])
        print(f"  ✅  Gaia TAP reachable — sso_observation has {n:,} rows")
        return True
    except Exception as e:
        print(f"  ❌  Gaia TAP error: {e}")
        return False


def main():
    print("\n" + "=" * 55)
    print("  GAPC — Environment Verification")
    print("=" * 55)

    print("\n[Python version]")
    py_ok = check_python()

    print("\n[Required packages]")
    pkg_ok = check_packages()

    print("\n[Data directories]")
    dir_ok = check_dirs()

    print("\n[GASP catalog]")
    check_gasp()

    print("\n[Gaia TAP connectivity]")
    tap_ok = check_gaia_tap()

    print("\n" + "=" * 55)
    if py_ok and pkg_ok and dir_ok and tap_ok:
        print("  ✅  All checks passed — ready to run pipeline.")
    else:
        print("  ⚠️   Some checks failed — fix above before proceeding.")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
