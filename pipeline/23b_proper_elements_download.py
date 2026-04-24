"""
23b_proper_elements_download.py
GAPC — Download proper orbital elements and family membership from AstDys.

Step 23 (proper_elements.py) failed because the AstDys server was unreachable
from the VPS. This script tries multiple mirror URLs and saves the data locally.

Proper elements needed for:
  - Asteroid family membership (distance metric in (a_p, e_p, sin_i_p) space)
  - Step 24 (family age × G): families require proper elements, not osculating

Sources tried in order:
  1. AstDys-2 Pisa mirror (https://newton.spacedys.com)
  2. AstDys Bologna mirror (https://astdys.astro.unibo.it)
  3. MPC supplementary proper element file (limited coverage)

Family membership:
  Nesvorny+2015 not on VizieR. Download from:
  https://www.boulder.swri.edu/~nesvorny/families.html  (txt file)
  or use AstDys family pages (HTML-scrape, fallback).
  Alternatively: Milani & Knezevic proper elements + HCM family catalog
  from AstDys family list.

Outputs:
  data/interim/proper_elements.parquet   (a_p, e_p, sin_ip, number_mp)
  data/interim/family_membership.parquet (number_mp, family_id, family_name)
  logs/23b_proper_elements_stats.txt
"""

import io
import sys
import time
import re
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

ROOT     = Path(__file__).resolve().parents[1]
DATA_INT = ROOT / "data" / "interim"
DATA_RAW = ROOT / "data" / "raw"
LOG_DIR  = ROOT / "logs"

# Output paths
PROP_PATH   = DATA_INT / "proper_elements.parquet"
FAMILY_PATH = DATA_INT / "family_membership.parquet"

# AstDys proper element file URLs (numb.syn = numbered asteroids)
NUMB_SYN_URLS = [
    "https://newton.spacedys.com/~astdys2/propsynth/numb.syn",
    "https://astdys.astro.unibo.it/propsynth/numb.syn",
    "https://hamilton.dm.unipi.it/~astdys2/propsynth/numb.syn",
]

# Nesvorny 2015 family catalog URL (SWRI)
NESVORNY_URLS = [
    "https://www.boulder.swri.edu/~nesvorny/families/families.txt",
    "https://www.boulder.swri.edu/~nesvorny/families/family_list.txt",
]

TIMEOUT = 20   # seconds per URL attempt


def try_download(urls, label):
    """Try each URL in turn. Return (content_bytes, url_used) or (None, None)."""
    if not HAS_REQUESTS:
        print(f"  {label}: requests not installed — cannot download")
        return None, None
    for url in urls:
        print(f"  Trying {url} …", end=" ", flush=True)
        try:
            r = requests.get(url, timeout=TIMEOUT, stream=True)
            if r.status_code == 200:
                content = r.content
                print(f"OK ({len(content)/1024:.0f} kB)")
                return content, url
            else:
                print(f"HTTP {r.status_code}")
        except Exception as e:
            print(f"FAILED: {e}")
    return None, None


def parse_numb_syn(content_bytes):
    """
    Parse AstDys numb.syn format:
    Lines after header (starting with '!'): number  a_p  e_p  sin_i_p  (more cols)
    """
    text = content_bytes.decode("latin-1", errors="replace")
    lines = text.splitlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("!") or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            num  = int(parts[0])
            a_p  = float(parts[1])
            e_p  = float(parts[2])
            sinI = float(parts[3])
            records.append(dict(number_mp=num, a_proper=a_p,
                                e_proper=e_p, sinI_proper=sinI))
        except ValueError:
            continue
    return pd.DataFrame(records)


def parse_nesvorny(content_bytes):
    """
    Parse Nesvorny 2015 family list.
    Format varies; typically: number  family_id  family_name
    """
    text = content_bytes.decode("latin-1", errors="replace")
    lines = text.splitlines()
    records = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            num = int(parts[0])
            fid = int(parts[1]) if parts[1].isdigit() else np.nan
            fname = parts[2] if len(parts) > 2 else ""
            records.append(dict(number_mp=num, family_id=fid, family_name=fname))
        except ValueError:
            continue
    return pd.DataFrame(records)


def main():
    print("\n" + "=" * 65)
    print("  GAPC Step 23b — Proper elements + family membership download")
    print("=" * 65)

    DATA_INT.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    n_prop = 0
    n_fam  = 0
    prop_source = "none"
    fam_source  = "none"

    # ── 1. Proper elements ────────────────────────────────────────────────────
    if PROP_PATH.exists():
        existing = pd.read_parquet(PROP_PATH)
        n_prop = len(existing)
        print(f"\n[1] Proper elements already cached: {n_prop:,} objects — skip download")
        prop_source = "cached"
    else:
        print(f"\n[1] Downloading proper elements (numb.syn) …")
        content, url_used = try_download(NUMB_SYN_URLS, "AstDys numb.syn")
        if content is not None:
            df_prop = parse_numb_syn(content)
            n_prop = len(df_prop)
            print(f"  Parsed: {n_prop:,} objects")
            if n_prop > 100:
                df_prop.to_parquet(PROP_PATH, index=False)
                print(f"  Saved → {PROP_PATH.relative_to(ROOT)}")
                prop_source = url_used
            else:
                print(f"  Too few records ({n_prop}) — parse likely failed")
        else:
            print("\n  All proper element URLs failed.")
            print("  Manual download:")
            for url in NUMB_SYN_URLS:
                print(f"    curl -L \"{url}\" -o /tmp/numb.syn")
            print("  Then: scp /tmp/numb.syn <vps>:~/gapc/data/raw/")
            print("  And re-run this script (it will parse numb.syn if found in data/raw).")

    # Check for manual download
    numb_raw = DATA_RAW / "numb.syn"
    if not PROP_PATH.exists() and numb_raw.exists():
        print(f"  Found manual download at {numb_raw} — parsing …")
        with open(numb_raw, "rb") as fh:
            content = fh.read()
        df_prop = parse_numb_syn(content)
        n_prop = len(df_prop)
        print(f"  Parsed: {n_prop:,} objects")
        if n_prop > 100:
            df_prop.to_parquet(PROP_PATH, index=False)
            print(f"  Saved → {PROP_PATH.relative_to(ROOT)}")
            prop_source = str(numb_raw)

    # ── 2. Family membership ──────────────────────────────────────────────────
    if FAMILY_PATH.exists():
        existing_fam = pd.read_parquet(FAMILY_PATH)
        n_fam = len(existing_fam)
        print(f"\n[2] Family membership already cached: {n_fam:,} entries — skip download")
        fam_source = "cached"
    else:
        print(f"\n[2] Downloading Nesvorny family membership …")
        content_fam, url_fam = try_download(NESVORNY_URLS, "Nesvorny families")
        if content_fam is not None:
            df_fam = parse_nesvorny(content_fam)
            n_fam = len(df_fam)
            print(f"  Parsed: {n_fam:,} entries")
            if n_fam > 100:
                df_fam.to_parquet(FAMILY_PATH, index=False)
                print(f"  Saved → {FAMILY_PATH.relative_to(ROOT)}")
                fam_source = url_fam
            else:
                print(f"  Too few records ({n_fam}) — try manual download")
        else:
            print("\n  All family URL attempts failed.")

    # Check for manually placed CSV
    for fname in ("families.txt", "family_list.txt", "nesvorny2015_families.csv"):
        fpath = DATA_RAW / fname
        if not FAMILY_PATH.exists() and fpath.exists():
            print(f"  Found {fname} — parsing …")
            with open(fpath, "rb") as fh:
                raw = fh.read()
            df_fam = parse_nesvorny(raw)
            n_fam = len(df_fam)
            if n_fam > 100:
                df_fam.to_parquet(FAMILY_PATH, index=False)
                print(f"  Parsed {n_fam:,} entries, saved → {FAMILY_PATH.relative_to(ROOT)}")
                fam_source = str(fpath)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Results:")
    print(f"    Proper elements: {n_prop:,} objects  (source: {prop_source})")
    print(f"    Family membership: {n_fam:,} entries  (source: {fam_source})")

    if n_prop > 0 and n_fam > 0:
        print("\n  Both datasets available — re-running step 24 is now possible.")
    else:
        print("\n  MANUAL STEPS REQUIRED:")
        print("  1. On your local machine, run:")
        print("       curl -L 'https://newton.spacedys.com/~astdys2/propsynth/numb.syn'")
        print("            -o ~/Downloads/numb.syn")
        print("     Then: scp ~/Downloads/numb.syn <user>@<vps>:~/gapc/data/raw/")
        print()
        print("  2. Nesvorny 2015 families — check one of:")
        print("     https://www.boulder.swri.edu/~nesvorny/families/")
        print("     https://sbn.psi.edu/pds/resource/nesvornyfam.html  (PDS archive)")
        print("     Filename: 'families.txt' or 'hcm_families.txt'")
        print("     scp the file to ~/gapc/data/raw/")
        print()
        print("  3. Re-run this script — it auto-detects manually placed files.")

    with open(LOG_DIR / "23b_proper_elements_stats.txt", "w") as f:
        f.write("GAPC Step 23b — Proper elements download\n")
        f.write(f"Proper elements: n={n_prop}  source={prop_source}\n")
        f.write(f"Family membership: n={n_fam}  source={fam_source}\n")
    print(f"\n  Log → logs/23b_proper_elements_stats.txt\n")


if __name__ == "__main__":
    main()
