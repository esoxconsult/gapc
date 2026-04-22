"""
02_download_fallback.py
GAPC — Fallback downloader when primary TAP endpoints are unavailable.

Tries in order:
  1. ARI Heidelberg TAP  — via direct HTTP (bypasses astroquery result-fetch bug)
  2. VizieR TAP          — CDS mirror; SSO table may not be present
  3. 10% sample          — MOD(number_mp,10)=0 from ARI or VizieR

NOTE: ESA CDN bulk files (SsoObservation_NN.csv.gz) have only 35 columns
and lack phase_angle / heliocentric_distance / geocentric_distance / g_mag_error
(TAP-derived quantities not stored in the raw dump).  CDN is therefore not used.

Output:
  data/raw/sso_observations.parquet              — full dataset
  data/raw/sso_observations_sample10pct.parquet  — 10%% fallback only
"""

import io
import logging
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests
from astropy.table import Table
from astropy.io.votable import parse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
OUT_FULL   = ROOT / "data" / "raw" / "sso_observations.parquet"
OUT_SAMPLE = ROOT / "data" / "raw" / "sso_observations_sample10pct.parquet"

G_MAG_ERROR_MAX  = 0.1
POLL_INTERVAL_S  = 30
JOB_TIMEOUT_S    = 1800   # 30 min hard limit per job
HTTP_TIMEOUT_S   = 60     # per individual HTTP request

FULL_QUERY = """
SELECT
    source_id,
    number_mp,
    denomination,
    epoch_utc,
    g_mag,
    g_mag_error,
    phase_angle,
    heliocentric_distance,
    geocentric_distance
FROM gaiadr3.sso_observation
WHERE g_mag               IS NOT NULL
  AND g_mag_error         IS NOT NULL
  AND phase_angle         IS NOT NULL
  AND heliocentric_distance IS NOT NULL
  AND geocentric_distance   IS NOT NULL
"""

SAMPLE_QUERY = FULL_QUERY.rstrip() + "\n  AND MOD(number_mp, 10) = 0\n"

REQUIRED_COLS = [
    "source_id", "number_mp", "denomination", "epoch_utc",
    "g_mag", "g_mag_error", "phase_angle",
    "heliocentric_distance", "geocentric_distance",
]

# VizieR I/359/ssoobs column names → our standard output names.
# NOTE: heliocentric_distance and geocentric_distance are NOT in the VizieR
# mirror (they are TAP-only derived quantities).  VizieR will therefore fail
# gracefully at the column-check stage — it is kept here as a best-effort
# attempt in case a future mirror update adds those columns.
VIZIER_RENAME = {
    "Source":   "source_id",
    "MPC":      "number_mp",
    "Name":     "denomination",
    "EpochUTC": "epoch_utc",
    "Gmag":     "g_mag",
    "PA":       "phase_angle",
    # g_mag_error, heliocentric_distance, geocentric_distance are absent
}


# ── direct-HTTP TAP client ─────────────────────────────────────────────────────

class TapHTTP:
    """Minimal TAP async client using requests — works around astroquery 404 bugs."""

    NS = {"uws": "http://www.ivoa.net/xml/UWS/v1.0"}

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers["Accept"] = "application/xml"

    def _post(self, url: str, data: dict) -> requests.Response:
        r = self.session.post(url, data=data, timeout=HTTP_TIMEOUT_S,
                              allow_redirects=True)
        r.raise_for_status()
        return r

    def _get(self, url: str, **kw) -> requests.Response:
        r = self.session.get(url, timeout=HTTP_TIMEOUT_S, **kw)
        r.raise_for_status()
        return r

    def count(self, table: str) -> int | None:
        """Quick COUNT(*) probe; returns None on failure."""
        try:
            r = self._post(f"{self.base}/sync", {
                "REQUEST": "doQuery",
                "LANG":    "ADQL",
                "QUERY":   f"SELECT COUNT(*) AS n FROM {table}",
                "FORMAT":  "csv",
            })
            lines = [l for l in r.text.splitlines() if l and not l.startswith("#")]
            return int(lines[1]) if len(lines) >= 2 else None
        except Exception as e:
            log.warning(f"  COUNT probe failed: {e}")
            return None

    def _force_https(self, url: str) -> str:
        """Upgrade http:// → https:// — ARI returns http job URLs but port 80 is blocked."""
        if url.startswith("http://"):
            base_host = self.base.split("://", 1)[1].split("/")[0]
            url_host  = url.split("://", 1)[1].split("/")[0]
            if url_host == base_host:
                url = "https://" + url[len("http://"):]
        return url

    def submit_async(self, query: str) -> str:
        """Submit async job, return job URL (always https).

        We must NOT follow redirects here: the server (ARI) redirects to an
        http:// URL, and port 80 may be blocked on the VPS.  We intercept the
        Location header instead and upgrade http→https ourselves.
        """
        r = self.session.post(
            f"{self.base}/async",
            data={
                "REQUEST": "doQuery",
                "LANG":    "ADQL",
                "QUERY":   query,
                "PHASE":   "RUN",
                "FORMAT":  "votable",
            },
            timeout=HTTP_TIMEOUT_S,
            allow_redirects=False,   # capture redirect without following
        )
        # Accept 200, 201, 303 — all indicate a job was created
        if r.status_code not in (200, 201, 303):
            r.raise_for_status()

        # Job URL is in the Location header (303) or in the response URL (200/201)
        loc = r.headers.get("Location", "")
        if loc:
            return self._force_https(loc.rstrip("/"))

        # Fallback: parse <uws:job>/<uws:jobId> from response body
        try:
            root = ET.fromstring(r.content)
            for tag in ("jobId", "{http://www.ivoa.net/xml/UWS/v1.0}jobId"):
                el = root.find(f".//{tag}")
                if el is not None and el.text:
                    return f"{self.base}/async/{el.text.strip()}"
        except ET.ParseError:
            pass

        raise RuntimeError(f"Cannot determine job URL from response (HTTP {r.status_code})")

    def wait_for_job(self, job_url: str) -> str:
        """Poll until COMPLETED/ERROR; return final phase."""
        deadline = time.time() + JOB_TIMEOUT_S
        while time.time() < deadline:
            try:
                r = self._get(f"{job_url}/phase")
                phase = r.text.strip()
            except Exception as e:
                log.warning(f"    phase poll error: {e}")
                phase = "UNKNOWN"
            log.info(f"    job phase: {phase}")
            if phase in ("COMPLETED", "ERROR", "ABORTED"):
                return phase
            time.sleep(POLL_INTERVAL_S)
        return "TIMEOUT"

    def fetch_results(self, job_url: str) -> pd.DataFrame:
        """Download the result VOTable and convert to DataFrame."""
        # Try /results/result first, then enumerate /results
        for suffix in ["/results/result", "/results"]:
            try:
                r = self._get(f"{job_url}{suffix}", stream=True)
                ct = r.headers.get("Content-Type", "")
                raw = r.content
                if b"VOTABLE" in raw[:500] or "votable" in ct.lower():
                    vt = parse(io.BytesIO(raw))
                    table = vt.get_first_table().to_table()
                    return table.to_pandas()
                # CSV fallback
                if b"," in raw[:200]:
                    lines = raw.decode("utf-8", errors="replace").splitlines()
                    data_lines = [l for l in lines if not l.startswith("#")]
                    return pd.read_csv(io.StringIO("\n".join(data_lines)))
            except Exception as e:
                log.info(f"    {suffix}: {e}")

        # Last resort: enumerate results from XML
        r = self._get(job_url)
        root = ET.fromstring(r.content)
        results = root.find("uws:results", self.NS)
        if results is not None:
            for res in results.findall("uws:result", self.NS):
                href = res.attrib.get("{http://www.w3.org/1999/xlink}href", "")
                if href:
                    href = self._force_https(href)
                    log.info(f"    trying result href: {href}")
                    r2 = self._get(href, stream=True)
                    vt = parse(io.BytesIO(r2.content))
                    table = vt.get_first_table().to_table()
                    return table.to_pandas()

        raise RuntimeError("Could not retrieve job results via any path")

    def run_query(self, query: str, name: str) -> pd.DataFrame:
        """Full lifecycle: submit → poll → fetch."""
        log.info(f"  [{name}] submitting async job …")
        job_url = self.submit_async(query)
        log.info(f"  [{name}] job URL: {job_url}")

        phase = self.wait_for_job(job_url)
        if phase != "COMPLETED":
            raise RuntimeError(f"Job ended with phase={phase}")

        log.info(f"  [{name}] downloading results …")
        df = self.fetch_results(job_url)
        return df


# ── shared post-processing ─────────────────────────────────────────────────────

def _filter(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.rename(columns=VIZIER_RENAME)
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    before = len(df)
    df = df.dropna(subset=["g_mag", "g_mag_error", "phase_angle",
                            "heliocentric_distance", "geocentric_distance"])
    df = df[df["g_mag_error"] <= G_MAG_ERROR_MAX]
    log.info(f"  [{label}] after filters: {len(df):,} rows  "
             f"(dropped {before - len(df):,})")
    return df


def _save(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[REQUIRED_COLS].to_parquet(path, index=False, compression="snappy")
    mb   = path.stat().st_size / 1e6
    nobj = df["number_mp"].nunique()
    log.info(f"  [{label}] ✅  {path.name}  "
             f"({len(df):,} rows · {nobj:,} asteroids · {mb:.1f} MB)")


# ── Method 1: ARI Heidelberg ──────────────────────────────────────────────────

ARI_URL = "https://gaia.ari.uni-heidelberg.de/tap"


def try_ari(query: str, label: str) -> pd.DataFrame | None:
    log.info(f"\n{'─'*55}\n  Method 1 — ARI Heidelberg  [{label}]\n{'─'*55}")
    tap = TapHTTP(ARI_URL)

    count = tap.count("gaiadr3.sso_observation")
    if count is None:
        log.warning("  [ARI] probe failed — skipping")
        return None
    log.info(f"  [ARI] probe OK — COUNT={count:,}")

    try:
        df = tap.run_query(query, "ARI")
        log.info(f"  [ARI] got {len(df):,} rows")
        return _filter(df, "ARI")
    except Exception as e:
        log.warning(f"  [ARI] failed: {e}")
        return None


# ── Method 2: VizieR TAP ──────────────────────────────────────────────────────

VIZIER_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap"
# I/359/ssoobs = Gaia DR3 sso_observation (Galluccio et al. 2022), 23.3M rows.
# Double-quoted because the slash is significant in ADQL.
VIZIER_TABLE_CANDIDATES = [
    '"I/359/ssoobs"',
    "I/359/ssoobs",
]


def _find_vizier_table() -> tuple[str, list[str]] | None:
    tap = TapHTTP(VIZIER_URL)
    for tname in VIZIER_TABLE_CANDIDATES:
        try:
            r = tap.session.post(f"{VIZIER_URL}/sync", timeout=HTTP_TIMEOUT_S, data={
                "REQUEST": "doQuery", "LANG": "ADQL",
                "QUERY":   f"SELECT TOP 1 * FROM {tname}",
                "FORMAT":  "csv",
            })
            r.raise_for_status()
            lines = [l for l in r.text.splitlines() if not l.startswith("#") and l]
            if len(lines) >= 2:
                cols = lines[0].split(",")
                log.info(f"  [VizieR] found table {tname!r}  cols={cols}")
                return tname, cols
        except Exception as e:
            log.info(f"  [VizieR] {tname!r} → {e}")
    return None


def _build_vizier_query(tname: str, cols: list[str], sample: bool) -> str | None:
    inv = {v: k for k, v in VIZIER_RENAME.items()}
    parts = []
    for std in REQUIRED_COLS:
        viz = inv.get(std, std)
        if viz in cols:
            parts.append(f"{viz} AS {std}" if viz != std else std)
        elif std in cols:
            parts.append(std)
        else:
            log.warning(f"  [VizieR] column {std!r} not in table — skipping method")
            return None
    select = ", ".join(parts)
    sample_clause = " AND MOD(number_mp, 10) = 0" if sample else ""
    return (f"SELECT {select} FROM {tname} "
            f"WHERE g_mag IS NOT NULL{sample_clause}")


def try_vizier(sample: bool = False) -> pd.DataFrame | None:
    suffix = " (10%% sample)" if sample else ""
    log.info(f"\n{'─'*55}\n  Method 2 — VizieR TAP{suffix}\n{'─'*55}")

    found = _find_vizier_table()
    if not found:
        log.warning("  [VizieR] SSO table not found — skipping")
        return None

    tname, cols = found
    query = _build_vizier_query(tname, cols, sample)
    if not query:
        return None

    tap = TapHTTP(VIZIER_URL)
    try:
        df = tap.run_query(query, "VizieR")
        log.info(f"  [VizieR] got {len(df):,} rows")
        return _filter(df, "VizieR")
    except Exception as e:
        log.warning(f"  [VizieR] failed: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    log.info("\n" + "=" * 55)
    log.info("  GAPC Step 2 — Fallback downloader")
    log.info("=" * 55)

    if OUT_FULL.exists():
        n = len(pd.read_parquet(OUT_FULL, columns=["number_mp"]))
        log.info(f"\n  Full output already exists ({n:,} rows) — nothing to do.\n")
        return

    t0 = time.time()

    # ── Full dataset ──────────────────────────────────────────────────────────
    df = None
    for fn in [
        lambda: try_ari(FULL_QUERY, "full"),
        lambda: try_vizier(sample=False),
    ]:
        df = fn()
        if df is not None and len(df) > 0:
            break

    if df is not None and len(df) > 0:
        _save(df, OUT_FULL, "full")
        log.info(f"\n  Total elapsed: {(time.time()-t0)/60:.1f} min\n")
        return

    # ── 10% sample fallback ───────────────────────────────────────────────────
    log.warning("\n  ⚠️  Full-data methods failed — attempting 10%% sample")

    if OUT_SAMPLE.exists():
        n = len(pd.read_parquet(OUT_SAMPLE, columns=["number_mp"]))
        log.info(f"  Sample already exists ({n:,} rows) — skipping.\n")
        sys.exit(0)

    df_s = None
    for fn in [
        lambda: try_ari(SAMPLE_QUERY, "10%% sample"),
        lambda: try_vizier(sample=True),
    ]:
        df_s = fn()
        if df_s is not None and len(df_s) > 0:
            break

    if df_s is not None and len(df_s) > 0:
        _save(df_s, OUT_SAMPLE, "10%% sample")
        log.warning(
            "\n  NOTE: Only a 10%% sample was saved:\n"
            f"    {OUT_SAMPLE}\n"
            "  Steps 03-06 run on this subset (indicative only).\n"
            "  Re-run once the ESA/ARI archive recovers for the full dataset.\n"
        )
        log.info(f"  Total elapsed: {(time.time()-t0)/60:.1f} min\n")
        return

    log.error(
        "\n  ✗  All methods failed (including 10%% sample).\n"
        "  Check VPS network / service status and retry later.\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
