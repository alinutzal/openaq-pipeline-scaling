"""
fetch_openaq.py
───────────────
Fetch hourly NO₂ and O₃ measurements from the OpenAQ v2 REST API for a
single metro area and save them to a CSV file.

If the API is unreachable, or the ``--synthetic`` flag is passed, synthetic
station data are generated instead so that the rest of the pipeline can run
without network access or an API key.

Usage examples
──────────────
# Real API – Los Angeles, last 7 days
python fetch_openaq.py --lat 34.05 --lon -118.24 --radius 50 --days 7

# Synthetic data (no API key needed)
python fetch_openaq.py --synthetic --days 14 --stations 40

# Custom output path
python fetch_openaq.py --synthetic --output data/my_measurements.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any

import numpy as np
import requests

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

OPENAQ_V2_BASE = "https://api.openaq.org/v2"
DEFAULT_OUTPUT = "data/measurements.csv"
PARAMETERS = ["no2", "o3"]

# Approximate bounding box for Los Angeles basin (default metro area)
DEFAULT_LAT = 34.05
DEFAULT_LON = -118.24
DEFAULT_RADIUS_KM = 50  # kilometres

# Synthetic station bounds – slightly wider than default radius
SYNTH_LAT_RANGE = (33.70, 34.40)
SYNTH_LON_RANGE = (-118.70, -117.80)


# ──────────────────────────────────────────────────────────────────────────────
# Real OpenAQ API helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fetch_page(
    session: requests.Session,
    endpoint: str,
    params: Dict[str, Any],
    api_key: str | None,
) -> Dict[str, Any]:
    """GET a single page from the OpenAQ v2 API, returning parsed JSON."""
    headers = {"X-API-Key": api_key} if api_key else {}
    resp = session.get(
        f"{OPENAQ_V2_BASE}/{endpoint}",
        params=params,
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_real(
    lat: float,
    lon: float,
    radius_km: int,
    days: int,
    api_key: str | None,
) -> List[Dict[str, Any]]:
    """
    Pull all available hourly NO₂/O₃ measurements around (lat, lon) for the
    past *days* days from the OpenAQ v2 API.

    Returns a list of flat record dicts with keys:
        location_id, location, lat, lon, parameter, value, unit, timestamp
    """
    date_to = datetime.now(timezone.utc)
    date_from = date_to - timedelta(days=days)

    session = requests.Session()
    records: List[Dict[str, Any]] = []

    for param in PARAMETERS:
        page = 1
        while True:
            params = {
                "coordinates": f"{lat},{lon}",
                "radius": radius_km * 1000,  # API expects metres
                "parameter": param,
                "date_from": date_from.isoformat(),
                "date_to": date_to.isoformat(),
                "limit": 1000,
                "page": page,
                "order_by": "datetime",
                "sort": "asc",
            }
            try:
                data = _fetch_page(session, "measurements", params, api_key)
            except requests.RequestException as exc:
                print(f"[warn] API request failed (page {page}, {param}): {exc}",
                      file=sys.stderr)
                break

            results = data.get("results", [])
            if not results:
                break

            for r in results:
                coords = r.get("coordinates") or {}
                records.append(
                    {
                        "location_id": r.get("locationId"),
                        "location": r.get("location", ""),
                        "lat": coords.get("latitude"),
                        "lon": coords.get("longitude"),
                        "parameter": r.get("parameter"),
                        "value": r.get("value"),
                        "unit": r.get("unit", ""),
                        "timestamp": r.get("date", {}).get("utc", ""),
                    }
                )

            meta = data.get("meta", {})
            total = meta.get("found", 0)
            fetched = (page - 1) * 1000 + len(results)
            print(
                f"  {param}: fetched {fetched}/{total} measurements (page {page})",
                flush=True,
            )

            if fetched >= total or len(results) < 1000:
                break
            page += 1
            time.sleep(0.2)  # polite rate limiting

    return records


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic data generator
# ──────────────────────────────────────────────────────────────────────────────

def fetch_synthetic(
    days: int,
    n_stations: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate realistic-looking synthetic measurements so that the rest of the
    pipeline can be exercised without a network connection.

    Station locations are uniformly distributed inside SYNTH_LAT/LON_RANGE.
    Values follow a diurnal cycle plus Gaussian noise.
    """
    rng = np.random.default_rng(seed)

    # Create stations
    station_lats = rng.uniform(*SYNTH_LAT_RANGE, size=n_stations)
    station_lons = rng.uniform(*SYNTH_LON_RANGE, size=n_stations)

    date_to = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    date_from = date_to - timedelta(days=days)

    hours = int((date_to - date_from).total_seconds() // 3600)
    timestamps = [date_from + timedelta(hours=h) for h in range(hours)]

    records: List[Dict[str, Any]] = []

    for sid in range(n_stations):
        lat = float(station_lats[sid])
        lon = float(station_lons[sid])
        base_no2 = rng.uniform(10, 60)   # μg/m³ background
        base_o3 = rng.uniform(40, 120)   # μg/m³ background

        for ts in timestamps:
            hour_of_day = ts.hour
            # Diurnal pattern: NO₂ peaks in morning/evening rush; O₃ peaks midday
            no2_diurnal = 1.0 + 0.5 * np.exp(-((hour_of_day - 8) ** 2) / 8) \
                               + 0.3 * np.exp(-((hour_of_day - 18) ** 2) / 8)
            o3_diurnal = 1.0 + 0.6 * np.exp(-((hour_of_day - 14) ** 2) / 18)

            no2_val = float(
                base_no2 * no2_diurnal + rng.normal(0, base_no2 * 0.1)
            )
            o3_val = float(
                base_o3 * o3_diurnal + rng.normal(0, base_o3 * 0.1)
            )

            iso_ts = ts.isoformat()
            records.append(
                {
                    "location_id": sid + 1,
                    "location": f"SyntheticStation_{sid + 1}",
                    "lat": lat,
                    "lon": lon,
                    "parameter": "no2",
                    "value": max(0.0, no2_val),
                    "unit": "μg/m³",
                    "timestamp": iso_ts,
                }
            )
            records.append(
                {
                    "location_id": sid + 1,
                    "location": f"SyntheticStation_{sid + 1}",
                    "lat": lat,
                    "lon": lon,
                    "parameter": "o3",
                    "value": max(0.0, o3_val),
                    "unit": "μg/m³",
                    "timestamp": iso_ts,
                }
            )

    return records


# ──────────────────────────────────────────────────────────────────────────────
# I/O
# ──────────────────────────────────────────────────────────────────────────────

FIELDNAMES = [
    "location_id", "location", "lat", "lon",
    "parameter", "value", "unit", "timestamp",
]


def save_csv(records: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} records → {path}")


def load_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch OpenAQ NO₂/O₃ measurements or generate synthetic data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data instead of calling the API.",
    )
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT,
                        help="Centre latitude of metro area.")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON,
                        help="Centre longitude of metro area.")
    parser.add_argument("--radius", type=int, default=DEFAULT_RADIUS_KM,
                        help="Search radius in km (real API only).")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of past days to fetch.")
    parser.add_argument("--stations", type=int, default=40,
                        help="Number of synthetic stations (--synthetic only).")
    parser.add_argument("--api-key", default=None,
                        help="OpenAQ API key (optional, increases rate limits).")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help="Output CSV path.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    t0 = time.perf_counter()

    if args.synthetic:
        print(f"Generating synthetic data: {args.stations} stations, "
              f"{args.days} days …")
        records = fetch_synthetic(days=args.days, n_stations=args.stations)
    else:
        print(
            f"Fetching real OpenAQ data: lat={args.lat}, lon={args.lon}, "
            f"radius={args.radius} km, {args.days} days …"
        )
        try:
            records = fetch_real(
                lat=args.lat,
                lon=args.lon,
                radius_km=args.radius,
                days=args.days,
                api_key=args.api_key,
            )
        except Exception as exc:
            print(
                f"[error] Real API fetch failed: {exc}\n"
                "Falling back to synthetic data.",
                file=sys.stderr,
            )
            records = fetch_synthetic(days=args.days, n_stations=args.stations)

    elapsed = time.perf_counter() - t0
    print(f"Fetch complete: {len(records)} records in {elapsed:.2f}s")
    save_csv(records, args.output)


if __name__ == "__main__":
    main()
