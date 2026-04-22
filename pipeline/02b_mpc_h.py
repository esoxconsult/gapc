"""
02b_mpc_h.py
GAPC — Fetch MPC absolute magnitudes (H) from MPCORB.DAT.

Parses the MPC MPCORB fixed-width catalog to extract H and G (slope)
for all numbered asteroids.  H is used by 06_validate.py to benchmark
GAPC-fitted H values against the MPC reference.

Output: data/raw/mpc_h_magnitudes.parquet
  columns: number_mp (int64), H_mpc (float64), G_slope (float64)

MPCORB fixed-width layout (0-indexed byte positions):
  [0:7]   Packed designation
  [8:13]  H  absolute magnitude (e.g. "10.20")
  [14:19] G  slope parameter    (e.g. " 0.15")
  [20:25] Epoch (packed)
  ...

Usage:
  python pipeline/02b_mpc_h.py
"""

import gzip
import sys
from pathlib import Path

import requests
import pandas as pd

ROOT     = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "raw" / "mpc_h_magnitudes.parquet"

MPCORB_URL          = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT.gz"
MPCORB_URL_FALLBACK = "https://minorplanetcenter.net/iau/MPCORB/MPCORB.DAT"


def _unpack_mpc_number(designation: str) -> int | None:
    s = designation.strip()
    if not s:
        return None
    c = s[0]
    if c.isdigit():
        try:
            return int(s)
        except ValueError:
            return None
    if c.isupper():
        n = ord(c) - ord("A") + 10
    else:
        n = ord(c) - ord("a") + 36
    try:
        return n * 10000 + int(s[1:].strip() or 0)
    except ValueError:
        return None


def download_and_parse() -> pd.DataFrame:
    lines = None
    for url in [MPCORB_URL, MPCORB_URL_FALLBACK]:
        try:
            print(f"  Fetching {url} …")
            r = requests.get(url, timeout=300, stream=True)
            r.raise_for_status()
            raw = r.content
            print(f"  Downloaded {len(raw)/1e6:.1f} MB")
            if url.endswith(".gz"):
                raw = gzip.decompress(raw)
            lines = raw.decode("ascii", errors="ignore").splitlines()
            break
        except Exception as e:
            print(f"  {url}: {e}")

    if not lines:
        raise RuntimeError("Could not download MPCORB catalog")

    # Skip header — data starts after the line of dashes
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith("---"):
            data_start = i + 1
            break

    print(f"  Parsing from line {data_start} …")
    records = []
    for line in lines[data_start:]:
        if len(line) < 19 or not line[0:7].strip():
            continue
        num = _unpack_mpc_number(line[0:7])
        if num is None:
            continue
        try:
            H = float(line[8:13])
            G = float(line[14:19])
        except ValueError:
            continue
        records.append({"number_mp": num, "H_mpc": H, "G_slope": G})

    df = pd.DataFrame(records)
    df["number_mp"] = df["number_mp"].astype("int64")
    return df


def main():
    print("\n" + "=" * 55)
    print("  GAPC Step 2b — MPC H magnitudes")
    print("=" * 55)

    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        print(f"\n  Already exists: {len(existing):,} rows — delete to refresh.\n")
        return

    df = download_and_parse()
    print(f"  Parsed {len(df):,} numbered asteroids")
    print(f"  H_mpc  range: {df['H_mpc'].min():.1f} – {df['H_mpc'].max():.1f} mag")
    print(f"  G_slope range: {df['G_slope'].min():.2f} – {df['G_slope'].max():.2f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False, compression="snappy")
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\n  ✅  Saved: {OUT_PATH}  ({size_mb:.1f} MB)\n")


if __name__ == "__main__":
    main()
