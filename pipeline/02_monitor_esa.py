"""
02_monitor_esa.py
GAPC — Monitor ESA Gaia TAP health and auto-restart the main download.

Polls the ESA TAP endpoint every CHECK_INTERVAL_S seconds.
When it responds with a valid COUNT, launches 02_download_sso.py in a
subprocess.  Exits gracefully once the output file exists.

Usage:
  nohup python pipeline/02_monitor_esa.py >> logs/02_monitor_esa.log 2>&1 &
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

ROOT             = Path(__file__).resolve().parents[1]
OUT_FULL         = ROOT / "data" / "raw" / "sso_observations.parquet"
DOWNLOAD_SCRIPT  = ROOT / "pipeline" / "02_download_sso.py"
DOWNLOAD_LOG     = ROOT / "logs" / "02_download_sso_retry.log"
PID_FILE         = ROOT / "logs" / "02_download_sso.pid"

ESA_TAP          = "https://gea.esac.esa.int/tap-server/tap"
CHECK_INTERVAL_S = 1800   # 30 min between health checks
PROBE_TIMEOUT_S  = 60

PYTHON = sys.executable   # use same interpreter / venv as this script


def esa_probe() -> bool:
    """
    Two-stage ESA health check:
      1. Sync COUNT(*) — verifies the service answers queries.
      2. Tiny async job (MAXREC=5) — verifies async jobs can complete.
         This is the gate that matters for the full download.
    Returns True only if both stages pass.
    """
    session = requests.Session()

    # Stage 1: sync COUNT
    try:
        r = session.post(
            f"{ESA_TAP}/sync",
            data={"REQUEST": "doQuery", "LANG": "ADQL",
                  "QUERY": "SELECT COUNT(*) AS n FROM gaiadr3.sso_observation",
                  "FORMAT": "csv"},
            timeout=PROBE_TIMEOUT_S,
        )
        r.raise_for_status()
        lines = [l for l in r.text.splitlines() if l and not l.startswith("#")]
        if len(lines) < 2 or int(lines[1]) == 0:
            log.warning("  ESA sync probe: unexpected empty result")
            return False
        log.info(f"  ESA sync probe OK — COUNT={int(lines[1]):,}")
    except Exception as e:
        log.warning(f"  ESA sync probe failed: {e}")
        return False

    # Stage 2: tiny async job
    try:
        r = session.post(
            f"{ESA_TAP}/async",
            data={"REQUEST": "doQuery", "LANG": "ADQL",
                  "QUERY": "SELECT TOP 5 source_id FROM gaiadr3.sso_observation",
                  "PHASE": "RUN", "FORMAT": "votable", "MAXREC": "5"},
            timeout=PROBE_TIMEOUT_S,
            allow_redirects=False,
        )
        if r.status_code not in (200, 201, 303):
            log.warning(f"  ESA async probe: unexpected status {r.status_code}")
            return False
        job_url = r.headers.get("Location", r.url).rstrip("/")
        if job_url.startswith("http://"):
            job_url = "https://" + job_url[len("http://"):]

        # Poll until COMPLETED/ERROR (max 2 min)
        deadline = time.time() + 120
        phase = "UNKNOWN"
        while time.time() < deadline:
            try:
                phase = session.get(f"{job_url}/phase",
                                    timeout=15).text.strip()
            except Exception:
                pass
            if phase in ("COMPLETED", "ERROR", "ABORTED"):
                break
            time.sleep(10)

        if phase != "COMPLETED":
            log.warning(f"  ESA async probe: job ended with phase={phase}")
            return False
        log.info("  ESA async probe OK — async jobs are completing")
        return True
    except Exception as e:
        log.warning(f"  ESA async probe failed: {e}")
        return False


def download_is_running() -> bool:
    """True if a 02_download_sso.py process is currently alive."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)   # signal 0 = existence check
            return True
        except (ProcessLookupError, PermissionError, ValueError):
            PID_FILE.unlink(missing_ok=True)
    return False


def launch_download() -> None:
    """Launch 02_download_sso.py and record its PID."""
    DOWNLOAD_LOG.parent.mkdir(parents=True, exist_ok=True)
    log_fh = open(DOWNLOAD_LOG, "a")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [PYTHON, "-u", str(DOWNLOAD_SCRIPT)],
        cwd=str(ROOT),
        stdout=log_fh,
        stderr=log_fh,
        env=env,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    log.info(f"  Launched 02_download_sso.py  PID={proc.pid}  "
             f"log → {DOWNLOAD_LOG.name}")


def main() -> None:
    log.info("\n" + "=" * 55)
    log.info("  GAPC — ESA TAP monitor / auto-restart")
    log.info(f"  Check interval: {CHECK_INTERVAL_S}s  Probe timeout: {PROBE_TIMEOUT_S}s")
    log.info("=" * 55)

    MIN_ROWS = 1_000_000   # full download expected ~3–5M; reject symlinks / samples

    while True:
        if OUT_FULL.exists() and not OUT_FULL.is_symlink():
            n = len(__import__("pandas").read_parquet(OUT_FULL, columns=["number_mp"]))
            if n >= MIN_ROWS:
                log.info(f"\n  ✅  Output exists ({n:,} rows) — monitor exiting.\n")
                PID_FILE.unlink(missing_ok=True)
                sys.exit(0)
            log.info(f"  Found incomplete output ({n:,} rows < {MIN_ROWS:,}) — continuing.")

        # Reap any zombie children so download_is_running() stays accurate.
        try:
            os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            pass

        log.info(f"\n  Probing ESA TAP (sync + async smoke-test) …")
        alive = esa_probe()

        if not alive:
            log.warning(f"  ESA TAP not ready — next check in "
                        f"{CHECK_INTERVAL_S//60} min")
        else:
            if download_is_running():
                log.info("  ESA alive but download already running — skipping launch")
            else:
                log.info("  ESA ready — launching full download …")
                launch_download()

        log.info(f"  Sleeping {CHECK_INTERVAL_S}s …")
        time.sleep(CHECK_INTERVAL_S)


if __name__ == "__main__":
    main()
