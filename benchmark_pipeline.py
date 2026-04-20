"""
benchmark_pipeline.py
─────────────────────
Run three experiments on the OpenAQ pipeline and produce three paper plots.

Experiment 1 – Pipeline stage breakdown
    Measure per-sample time for:
        fetch/load  │  parse  │  grid construction  │  serialisation  │  inference

Experiment 2 – Grid-size scaling
    Sweep resolutions 8, 16, 32, 64 and record:
        preprocessing latency  │  throughput (windows/s)  │  memory footprint

Experiment 3 – Serving scaling (simulated Triton)
    Sweep concurrency levels 1–16 using a thread-pool-based serving stub and
    record:
        throughput (req/s)  │  p50 latency  │  p95 latency

Output
──────
    results/plot1_stage_breakdown.png
    results/plot2_grid_scaling.png
    results/plot3_serving_scaling.png
    results/benchmark_results.json

Usage
─────
    python benchmark_pipeline.py                        # uses data/measurements.csv
    python benchmark_pipeline.py --input my_data.csv
    python benchmark_pipeline.py --results-dir out/
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

# Local modules
import fetch_openaq
import grid_and_serialize as gs

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_INPUT = "data/measurements.csv"
DEFAULT_GRID_DIR = "data/grids"
DEFAULT_RESULTS_DIR = "results"

RESOLUTIONS = [8, 16, 32, 64]
CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
SERVING_REQUESTS = 200   # total requests sent in Experiment 3
BENCH_REPEATS = 3        # repeat each grid-build measurement for stable timing


# ──────────────────────────────────────────────────────────────────────────────
# Inference stubs
# ──────────────────────────────────────────────────────────────────────────────

def inference_last_value(tensor: np.ndarray) -> np.ndarray:
    """Last-value persistence baseline: predict last observed time step."""
    return tensor[-1:].copy()


def inference_fixed_conv(tensor: np.ndarray) -> np.ndarray:
    """
    Tiny fixed 3×3 smoothing convolution applied channel-wise to the last
    time step.  No training; weights are a pre-defined averaging kernel.
    """
    # (C, H, W) slice of the last time step
    frame = tensor[-1]  # (C, H, W)
    C, H, W = frame.shape
    out = np.zeros_like(frame)
    # 3×3 uniform kernel
    kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    for c in range(C):
        ch = frame[c]
        for i in range(H):
            for j in range(W):
                i0, i1 = max(0, i - 1), min(H, i + 2)
                j0, j1 = max(0, j - 1), min(W, j + 2)
                patch = ch[i0:i1, j0:j1]
                k = kernel[: patch.shape[0], : patch.shape[1]]
                out[c, i, j] = np.sum(patch * k) / np.sum(k)
    return out[np.newaxis]  # (1, C, H, W)


# Choose the fast stub by default; fixed_conv is used for variety
def run_inference(tensor: np.ndarray) -> np.ndarray:
    return inference_last_value(tensor)


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1: Pipeline stage breakdown
# ──────────────────────────────────────────────────────────────────────────────

def experiment1_stage_breakdown(
    csv_path: str,
    grid_dir: str,
) -> Dict[str, float]:
    """
    Measure per-sample wall-clock time for each pipeline stage at 16×16.
    Returns a dict of {stage_name: seconds}.
    """
    print("\n── Experiment 1: Stage breakdown ──────────────────────────────")
    resolution = 16
    timings: Dict[str, float] = {}

    # Stage 1: fetch/load
    t0 = time.perf_counter()
    df = gs.load_measurements(csv_path)
    timings["fetch/load"] = time.perf_counter() - t0
    print(f"  fetch/load       : {timings['fetch/load']:.4f}s")

    # Stage 2: parse (already done inside load, isolate the type-coercion cost)
    import pandas as pd
    raw = df.copy()
    raw["timestamp_str"] = raw["timestamp"].astype(str)
    t0 = time.perf_counter()
    raw["timestamp"] = pd.to_datetime(raw["timestamp_str"], utc=True, errors="coerce")
    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    timings["parse"] = time.perf_counter() - t0
    print(f"  parse            : {timings['parse']:.4f}s")

    # Stage 3: grid construction
    t0 = time.perf_counter()
    tensor, _ = gs.build_grid(df, resolution)
    timings["grid construction"] = time.perf_counter() - t0
    print(f"  grid construction: {timings['grid construction']:.4f}s")

    # Stage 4: serialisation
    t0 = time.perf_counter()
    gs.save_grid(tensor, grid_dir, resolution)
    timings["serialisation"] = time.perf_counter() - t0
    print(f"  serialisation    : {timings['serialisation']:.4f}s")

    # Stage 5: inference stub (per window)
    t0 = time.perf_counter()
    _ = run_inference(tensor)
    timings["inference stub"] = time.perf_counter() - t0
    print(f"  inference stub   : {timings['inference stub']:.4f}s")

    return timings


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 2: Grid-size scaling
# ──────────────────────────────────────────────────────────────────────────────

def experiment2_grid_scaling(
    csv_path: str,
    grid_dir: str,
    resolutions: List[int] = RESOLUTIONS,
) -> Dict[int, Dict[str, float]]:
    """
    Sweep multiple resolutions and record preprocessing latency, throughput,
    and memory footprint.
    """
    print("\n── Experiment 2: Grid-size scaling ────────────────────────────")
    df = gs.load_measurements(csv_path)
    results: Dict[int, Dict[str, float]] = {}

    for res in resolutions:
        latencies: List[float] = []
        for _ in range(BENCH_REPEATS):
            t0 = time.perf_counter()
            tensor, _ = gs.build_grid(df, res)
            latencies.append(time.perf_counter() - t0)

        lat_median = float(np.median(latencies))
        T = tensor.shape[0]
        throughput = T / lat_median  # windows/s
        mem_mb = tensor.nbytes / 1024 / 1024

        results[res] = {
            "latency_s": lat_median,
            "throughput_windows_per_s": throughput,
            "memory_mb": mem_mb,
        }
        print(
            f"  {res:>3}×{res:<3}  lat={lat_median:.4f}s  "
            f"tput={throughput:.1f} win/s  mem={mem_mb:.2f} MB"
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3: Simulated serving scaling
# ──────────────────────────────────────────────────────────────────────────────

def _serve_worker(
    req_queue: "queue.Queue[Tuple[int, np.ndarray]]",
    result_list: List[Tuple[int, float]],
    lock: threading.Lock,
) -> None:
    """Worker thread: dequeue requests, run inference, record latency."""
    while True:
        item = req_queue.get()
        if item is None:
            req_queue.task_done()
            break
        req_id, window = item
        t_start = time.perf_counter()
        _ = run_inference(window)
        latency = time.perf_counter() - t_start
        with lock:
            result_list.append((req_id, latency))
        req_queue.task_done()


def experiment3_serving_scaling(
    tensor: np.ndarray,
    concurrency_levels: List[int] = CONCURRENCY_LEVELS,
    n_requests: int = SERVING_REQUESTS,
) -> Dict[int, Dict[str, float]]:
    """
    Simulate concurrent Triton-style serving by submitting *n_requests*
    inference calls through a thread pool of *concurrency* workers.

    Returns throughput and p50/p95 latency per concurrency level.
    """
    print("\n── Experiment 3: Serving scaling ──────────────────────────────")
    T = tensor.shape[0]
    windows = [tensor[i % T : i % T + 1] for i in range(n_requests)]
    results: Dict[int, Dict[str, float]] = {}

    for conc in concurrency_levels:
        req_queue: queue.Queue = queue.Queue()
        result_list: List[Tuple[int, float]] = []
        lock = threading.Lock()

        workers = [
            threading.Thread(
                target=_serve_worker,
                args=(req_queue, result_list, lock),
                daemon=True,
            )
            for _ in range(conc)
        ]
        for w in workers:
            w.start()

        t_wall_start = time.perf_counter()
        for i, w in enumerate(windows):
            req_queue.put((i, w))
        for _ in workers:
            req_queue.put(None)  # sentinel
        req_queue.join()
        wall_elapsed = time.perf_counter() - t_wall_start

        for w in workers:
            w.join()

        latencies = [lat for _, lat in result_list]
        throughput = n_requests / wall_elapsed
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))

        results[conc] = {
            "throughput_req_per_s": throughput,
            "p50_latency_s": p50,
            "p95_latency_s": p95,
            "wall_s": wall_elapsed,
        }
        print(
            f"  concurrency={conc:<3}  "
            f"tput={throughput:.1f} req/s  "
            f"p50={p50 * 1e3:.2f}ms  p95={p95 * 1e3:.2f}ms"
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────────

_STAGE_COLORS = {
    "fetch/load": "#4e79a7",
    "parse": "#f28e2b",
    "grid construction": "#e15759",
    "serialisation": "#76b7b2",
    "inference stub": "#59a14f",
}


def plot1_stage_breakdown(
    timings: Dict[str, float],
    out_path: str,
) -> None:
    stages = list(timings.keys())
    values = [timings[s] * 1e3 for s in stages]  # ms
    colors = [_STAGE_COLORS.get(s, "#aaaaaa") for s in stages]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stages, values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.01,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel("Wall-clock time (ms)", fontsize=11)
    ax.set_title("Pipeline Stage Runtime Breakdown (16×16 grid)", fontsize=12)
    ax.set_xlabel("Pipeline stage", fontsize=11)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot 1 → {out_path}")


def plot2_grid_scaling(
    scaling: Dict[int, Dict[str, float]],
    out_path: str,
) -> None:
    resolutions = sorted(scaling.keys())
    latencies = [scaling[r]["latency_s"] * 1e3 for r in resolutions]   # ms
    throughputs = [scaling[r]["throughput_windows_per_s"] for r in resolutions]
    memories = [scaling[r]["memory_mb"] for r in resolutions]
    labels = [f"{r}×{r}" for r in resolutions]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(labels, latencies, marker="o", color="#4e79a7", linewidth=2)
    axes[0].set_title("Preprocessing Latency vs Grid Size")
    axes[0].set_xlabel("Grid resolution")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    axes[1].plot(labels, throughputs, marker="s", color="#e15759", linewidth=2)
    axes[1].set_title("Throughput vs Grid Size")
    axes[1].set_xlabel("Grid resolution")
    axes[1].set_ylabel("Windows / second")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    axes[2].plot(labels, memories, marker="^", color="#59a14f", linewidth=2)
    axes[2].set_title("Memory Footprint vs Grid Size")
    axes[2].set_xlabel("Grid resolution")
    axes[2].set_ylabel("Tensor size (MB)")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Experiment 2: Grid-Size Scaling", fontsize=13, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot 2 → {out_path}")


def plot3_serving_scaling(
    serving: Dict[int, Dict[str, float]],
    out_path: str,
) -> None:
    conc_levels = sorted(serving.keys())
    throughputs = [serving[c]["throughput_req_per_s"] for c in conc_levels]
    p50s = [serving[c]["p50_latency_s"] * 1e3 for c in conc_levels]   # ms
    p95s = [serving[c]["p95_latency_s"] * 1e3 for c in conc_levels]   # ms

    fig, ax1 = plt.subplots(figsize=(8, 5))
    color_tput = "#4e79a7"
    color_lat = "#e15759"

    ax1.set_xlabel("Concurrency (worker threads)", fontsize=11)
    ax1.set_ylabel("Throughput (req/s)", color=color_tput, fontsize=11)
    l1 = ax1.plot(
        conc_levels, throughputs,
        marker="o", color=color_tput, linewidth=2, label="Throughput",
    )
    ax1.tick_params(axis="y", labelcolor=color_tput)
    ax1.set_xticks(conc_levels)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Latency (ms)", color=color_lat, fontsize=11)
    l2 = ax2.plot(
        conc_levels, p50s,
        marker="s", color=color_lat, linewidth=2, linestyle="--", label="p50 latency",
    )
    l3 = ax2.plot(
        conc_levels, p95s,
        marker="^", color="#f28e2b", linewidth=2, linestyle=":", label="p95 latency",
    )
    ax2.tick_params(axis="y", labelcolor=color_lat)

    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="upper left", fontsize=9)

    ax1.set_title(
        "Experiment 3: Throughput & Latency vs Concurrency\n"
        "(simulated serving — replace with Triton Perf Analyzer for production)",
        fontsize=11,
    )
    ax1.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Plot 3 → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the OpenAQ pipeline and produce paper plots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Path to measurements CSV.")
    parser.add_argument("--grid-dir", default=DEFAULT_GRID_DIR,
                        help="Directory for intermediate .npy grids.")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR,
                        help="Output directory for plots and JSON summary.")
    parser.add_argument(
        "--resolutions", type=int, nargs="+", default=RESOLUTIONS,
        help="Grid resolutions to sweep in Experiment 2.",
    )
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=CONCURRENCY_LEVELS,
        help="Concurrency levels to sweep in Experiment 3.",
    )
    parser.add_argument(
        "--n-requests", type=int, default=SERVING_REQUESTS,
        help="Number of inference requests for Experiment 3.",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate synthetic data if the CSV is missing.",
    )
    return parser.parse_args(argv)


def _ensure_data(args: argparse.Namespace) -> None:
    """Generate synthetic data if the CSV does not exist."""
    if not os.path.exists(args.input):
        if args.synthetic:
            print(f"{args.input} not found – generating synthetic data …")
            fetch_openaq.main(["--synthetic", "--output", args.input])
        else:
            print(
                f"[error] {args.input} not found.\n"
                "Run  python fetch_openaq.py --synthetic  first, or pass "
                "--synthetic to this script.",
                file=sys.stderr,
            )
            sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    _ensure_data(args)

    # ── Experiment 1 ────────────────────────────────────────────────────────
    timings = experiment1_stage_breakdown(args.input, args.grid_dir)

    # ── Experiment 2 ────────────────────────────────────────────────────────
    scaling = experiment2_grid_scaling(args.input, args.grid_dir, args.resolutions)

    # Load a tensor for Experiment 3 (use 16×16 or first available resolution)
    ref_res = 16 if 16 in args.resolutions else args.resolutions[0]
    npy_path = os.path.join(args.grid_dir, f"grid_{ref_res}x{ref_res}.npy")
    if not os.path.exists(npy_path):
        df = gs.load_measurements(args.input)
        tensor, _ = gs.build_grid(df, ref_res)
        gs.save_grid(tensor, args.grid_dir, ref_res)
    else:
        tensor = np.load(npy_path)

    # ── Experiment 3 ────────────────────────────────────────────────────────
    serving = experiment3_serving_scaling(
        tensor,
        concurrency_levels=args.concurrency,
        n_requests=args.n_requests,
    )

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\n── Generating plots ────────────────────────────────────────────")
    plot1_stage_breakdown(
        timings,
        os.path.join(args.results_dir, "plot1_stage_breakdown.png"),
    )
    plot2_grid_scaling(
        scaling,
        os.path.join(args.results_dir, "plot2_grid_scaling.png"),
    )
    plot3_serving_scaling(
        serving,
        os.path.join(args.results_dir, "plot3_serving_scaling.png"),
    )

    # ── JSON summary ─────────────────────────────────────────────────────────
    summary = {
        "experiment1_stage_breakdown": timings,
        "experiment2_grid_scaling": {
            str(k): v for k, v in scaling.items()
        },
        "experiment3_serving_scaling": {
            str(k): v for k, v in serving.items()
        },
    }
    json_path = os.path.join(args.results_dir, "benchmark_results.json")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  JSON summary → {json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
