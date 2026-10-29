"""
measure.py - dependency-free response-time and concurrency probe for a deployed URL.

Sends N requests with a given concurrency to the Streamlit health endpoint and to the home page,
then prints min / mean / median / p95 / max latency and the error count. The printed numbers are
what is entered in docs/CLOUD_EVALUATION.md (they must come from a real run against the deployed service).

Usage:  python loadtest/measure.py https://<service-url> --requests 200 --concurrency 20
"""
import argparse
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def timed_get(url: str, timeout: float = 30.0):
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            resp.read()
            ok = 200 <= resp.status < 400
    except Exception:
        ok = False
    return time.perf_counter() - start, ok


def run(url: str, requests: int, concurrency: int) -> dict:
    wall = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _: timed_get(url), range(requests)))
    wall = time.perf_counter() - wall
    lat = sorted(t * 1000 for t, ok in results if ok)
    errors = sum(1 for _, ok in results if not ok)
    if not lat:
        return {"url": url, "requests": requests, "concurrency": concurrency, "errors": errors}
    return {"url": url, "requests": requests, "concurrency": concurrency, "errors": errors,
            "min_ms": lat[0], "mean_ms": statistics.mean(lat), "median_ms": statistics.median(lat),
            "p95_ms": lat[min(len(lat) - 1, int(0.95 * len(lat)))], "max_ms": lat[-1],
            "throughput_rps": requests / wall}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base_url")
    ap.add_argument("--requests", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=10)
    args = ap.parse_args()
    base = args.base_url.rstrip("/")
    for path in ("/_stcore/health", "/"):
        res = run(base + path, args.requests, args.concurrency)
        print(f"\n{path}")
        for key, value in res.items():
            print(f"  {key:15s} {value:.1f}" if isinstance(value, float) else f"  {key:15s} {value}")


if __name__ == "__main__":
    main()
