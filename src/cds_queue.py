"""Inspect and clean up this account's CDS job queue.

CDS caps how many jobs one user may have queued against a dataset (empirically
6 for reanalysis-era5-single-levels). Killing a downloader mid-flight leaves its
jobs sitting in that queue with nobody to collect them, and the next run cannot
submit anything until they are cleared.

    python cds_queue.py            # list
    python cds_queue.py --purge    # delete orphaned accepted/running/rejected
"""

from __future__ import annotations

import argparse
import collections

import cdsapi
import requests

DATASET = "reanalysis-era5-single-levels"
# Never touch jobs whose results may still be worth collecting.
PURGEABLE = {"accepted", "running", "rejected", "failed", "dismissed"}


def _session():
    client = cdsapi.Client()
    return client.url.rstrip("/"), {"PRIVATE-TOKEN": client.key}


def list_jobs(limit: int = 200) -> list[dict]:
    base, headers = _session()
    r = requests.get(f"{base}/retrieve/v1/jobs", headers=headers,
                     params={"limit": limit}, timeout=90)
    r.raise_for_status()
    data = r.json()
    return data.get("jobs", data if isinstance(data, list) else [])


def show(jobs: list[dict]) -> None:
    counts = collections.Counter(j.get("status") for j in jobs)
    print(f"[cds] {len(jobs)} jobs: {dict(counts)}")
    for j in jobs[:30]:
        print(f"  {str(j.get('jobID'))[:8]}  {str(j.get('status')):11s} "
              f"{str(j.get('created'))[:19]}  {j.get('processID', '')}")


def purge(jobs: list[dict], dataset: str = DATASET) -> int:
    base, headers = _session()
    victims = [j for j in jobs
               if j.get("status") in PURGEABLE
               and j.get("processID") == dataset]
    if not victims:
        print("[cds] nothing to purge")
        return 0

    print(f"[cds] deleting {len(victims)} job(s)")
    deleted = 0
    for j in victims:
        jid = j.get("jobID")
        try:
            r = requests.delete(f"{base}/retrieve/v1/jobs/{jid}",
                                headers=headers, timeout=60)
            ok = r.status_code in (200, 204)
            print(f"  {str(jid)[:8]}  {j.get('status'):11s} -> "
                  f"{'deleted' if ok else f'HTTP {r.status_code}'}")
            deleted += int(ok)
        except Exception as exc:
            print(f"  {str(jid)[:8]}  delete failed: {exc}")
    return deleted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--purge", action="store_true")
    args = ap.parse_args()

    jobs = list_jobs()
    show(jobs)

    if args.purge:
        purge(jobs)
        print("\n[cds] after purge:")
        show(list_jobs())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
