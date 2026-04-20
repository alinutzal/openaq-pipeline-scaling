"""
grid_and_serialize.py
─────────────────────
Convert irregularly-spaced OpenAQ station measurements into regular 2-D grids
and serialize them as NumPy `.npy` tensors.

Pipeline stages performed here:
    1. Load CSV produced by fetch_openaq.py
    2. Parse timestamps, cast numeric columns
    3. For each time step and pollutant, bin station observations into an
       (H × W) grid by averaging all readings that fall in the same cell
    4. Forward-fill then zero-fill remaining NaN cells
    5. Stack into a (T, C, H, W) tensor  where
          T = number of hourly time steps
          C = number of channels / pollutants (2: NO₂, O₃)
          H, W = grid height / width
    6. Save tensor as <output_dir>/grid_<H>x<W>.npy

Usage examples
──────────────
# Default 16×16 grid
python grid_and_serialize.py

# Multiple resolutions in one call
python grid_and_serialize.py --resolutions 8 16 32 64

# Custom CSV and output directory
python grid_and_serialize.py --input data/measurements.csv --output data/grids
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────────────
# Defaults
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_INPUT = "data/measurements.csv"
DEFAULT_OUTPUT_DIR = "data/grids"
DEFAULT_RESOLUTIONS: List[int] = [16]

PARAMETERS = ["no2", "o3"]

# Approximate bounding box for Los Angeles (matches fetch_openaq.py defaults)
LAT_MIN, LAT_MAX = 33.70, 34.40
LON_MIN, LON_MAX = -118.70, -117.80


# ──────────────────────────────────────────────────────────────────────────────
# Loading / parsing
# ──────────────────────────────────────────────────────────────────────────────

def load_measurements(path: str) -> pd.DataFrame:
    """Load the CSV produced by fetch_openaq.py and apply type coercions."""
    df = pd.read_csv(path)
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["lat", "lon", "value", "timestamp"])
    df["parameter"] = df["parameter"].str.lower().str.strip()
    return df


# ──────────────────────────────────────────────────────────────────────────────
# Gridding
# ──────────────────────────────────────────────────────────────────────────────

def _bin_index(values: np.ndarray, vmin: float, vmax: float, n: int) -> np.ndarray:
    """Map continuous coordinate values to integer grid indices in [0, n-1]."""
    clipped = np.clip(values, vmin, vmax)
    idx = ((clipped - vmin) / (vmax - vmin) * n).astype(int)
    return np.clip(idx, 0, n - 1)


def build_grid(
    df: pd.DataFrame,
    resolution: int,
    lat_min: float = LAT_MIN,
    lat_max: float = LAT_MAX,
    lon_min: float = LON_MIN,
    lon_max: float = LON_MAX,
) -> Tuple[np.ndarray, List[pd.Timestamp]]:
    """
    Convert a DataFrame of measurements into a (T, C, H, W) NumPy tensor.

    Missing cells are forward-filled along the time axis, then zero-filled.

    Parameters
    ----------
    df          : DataFrame with columns lat, lon, parameter, value, timestamp
    resolution  : H = W = resolution
    lat/lon_min/max : spatial bounding box

    Returns
    -------
    tensor      : float32 array of shape (T, C, H, W)
    timestamps  : list of pd.Timestamp, length T
    """
    H = W = resolution
    C = len(PARAMETERS)

    # Hourly time index
    df = df.copy()
    df["hour"] = df["timestamp"].dt.floor("h")
    timestamps = sorted(df["hour"].unique())
    T = len(timestamps)
    t_index = {ts: i for i, ts in enumerate(timestamps)}

    # Grid coordinate indices
    df["row"] = _bin_index(df["lat"].values, lat_min, lat_max, H)
    df["col"] = _bin_index(df["lon"].values, lon_min, lon_max, W)
    df["t_idx"] = df["hour"].map(t_index)
    df["c_idx"] = df["parameter"].map(
        {p: i for i, p in enumerate(PARAMETERS)}
    )
    df = df.dropna(subset=["c_idx"])
    df["c_idx"] = df["c_idx"].astype(int)

    # Accumulate sums and counts for averaging
    tensor = np.zeros((T, C, H, W), dtype=np.float64)
    counts = np.zeros((T, C, H, W), dtype=np.int32)

    for row in df.itertuples(index=False):
        tensor[row.t_idx, row.c_idx, row.row, row.col] += row.value
        counts[row.t_idx, row.c_idx, row.row, row.col] += 1

    # Average where we have observations; NaN elsewhere
    with np.errstate(invalid="ignore"):
        averaged = np.where(counts > 0, tensor / counts, np.nan)

    # Forward-fill along time axis, then zero-fill
    filled = _forward_fill(averaged)
    filled = np.where(np.isnan(filled), 0.0, filled)

    return filled.astype(np.float32), timestamps


def _forward_fill(arr: np.ndarray) -> np.ndarray:
    """Forward-fill NaN values along axis 0 (time) for a (T, C, H, W) array."""
    out = arr.copy()
    for t in range(1, out.shape[0]):
        mask = np.isnan(out[t])
        out[t] = np.where(mask, out[t - 1], out[t])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Serialisation
# ──────────────────────────────────────────────────────────────────────────────

def save_grid(tensor: np.ndarray, output_dir: str, resolution: int) -> str:
    """Save tensor as a .npy file and return the file path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"grid_{resolution}x{resolution}.npy")
    np.save(path, tensor)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(
        f"  Saved {tensor.shape} tensor → {path}  ({size_mb:.2f} MB)"
    )
    return path


# ──────────────────────────────────────────────────────────────────────────────
# Timing helpers (used by benchmark_pipeline.py)
# ──────────────────────────────────────────────────────────────────────────────

def timed_load(path: str) -> Tuple[pd.DataFrame, float]:
    t0 = time.perf_counter()
    df = load_measurements(path)
    return df, time.perf_counter() - t0


def timed_grid(
    df: pd.DataFrame,
    resolution: int,
) -> Tuple[np.ndarray, List[pd.Timestamp], float]:
    t0 = time.perf_counter()
    tensor, timestamps = build_grid(df, resolution)
    elapsed = time.perf_counter() - t0
    return tensor, timestamps, elapsed


def timed_serialize(
    tensor: np.ndarray,
    output_dir: str,
    resolution: int,
) -> Tuple[str, float]:
    t0 = time.perf_counter()
    path = save_grid(tensor, output_dir, resolution)
    return path, time.perf_counter() - t0


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Grid and serialize OpenAQ measurements.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Path to measurements CSV.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="Output directory for .npy grids.")
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=DEFAULT_RESOLUTIONS,
        help="One or more grid resolutions to produce (e.g. 8 16 32 64).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    print(f"Loading measurements from {args.input} …")
    t0_total = time.perf_counter()
    df, t_load = timed_load(args.input)
    print(f"  Loaded {len(df)} rows in {t_load:.3f}s")

    for res in args.resolutions:
        print(f"\nBuilding {res}×{res} grid …")
        tensor, timestamps, t_grid = timed_grid(df, res)
        print(
            f"  Grid shape: {tensor.shape}  "
            f"(T={tensor.shape[0]}, C={tensor.shape[1]}, "
            f"H={tensor.shape[2]}, W={tensor.shape[3]})  "
            f"built in {t_grid:.3f}s"
        )
        _, t_ser = timed_serialize(tensor, args.output, res)
        print(f"  Serialised in {t_ser:.4f}s")

    print(f"\nTotal: {time.perf_counter() - t0_total:.3f}s")


if __name__ == "__main__":
    main()
