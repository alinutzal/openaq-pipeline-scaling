# openaq-pipeline-scaling

A lightweight, scalable pipeline for OpenAQ air-quality data: from irregular
station measurements to real-time gridded inference.

> **Paper claim:** For small real-time gridded workloads, preprocessing and
> serving overhead can dominate the cost of inference.

---

## Overview

The pipeline consists of three scripts and three experiments that validate the
claim above.

```
fetch_openaq.py          # Stage 1 – ingest NO₂ / O₃ measurements
grid_and_serialize.py    # Stage 2 – grid irregularly-spaced stations → tensors
benchmark_pipeline.py    # Experiments 1-3 + paper plots
```

### Pipeline stages

| Stage | Description |
|---|---|
| Fetch / load | Pull hourly NO₂ and O₃ readings from the OpenAQ v2 API (or generate synthetic data) |
| Parse | Type-coerce timestamps and numeric values |
| Grid construction | Bin station observations into an (H × W) grid; forward-fill / zero-fill gaps |
| Serialisation | Save `(T, C, H, W)` float32 tensor as `.npy` |
| Inference stub | Last-value persistence baseline (`y_pred = x_last`) |

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Generate synthetic data (no API key required)
python fetch_openaq.py --synthetic --days 7 --stations 40

# 2. Grid the data at multiple resolutions
python grid_and_serialize.py --resolutions 8 16 32 64

# 3. Run all three experiments and produce plots
python benchmark_pipeline.py
```

Results land in `results/`:

| File | Description |
|---|---|
| `plot1_stage_breakdown.png` | Runtime breakdown bar chart |
| `plot2_grid_scaling.png` | Latency / throughput / memory vs grid size |
| `plot3_serving_scaling.png` | Throughput & p50/p95 latency vs concurrency |
| `benchmark_results.json` | Raw numbers for all three experiments |

---

## Using the real OpenAQ API

```bash
# Los Angeles – last 7 days, with API key
python fetch_openaq.py \
    --lat 34.05 --lon -118.24 --radius 50 --days 7 \
    --api-key YOUR_API_KEY \
    --output data/measurements.csv
```

OpenAQ accounts and API keys are free at <https://openaq.org/>.  The key
increases rate limits but is optional for small requests.

---

## Script reference

### `fetch_openaq.py`

```
usage: fetch_openaq.py [-h] [--synthetic] [--lat LAT] [--lon LON]
                       [--radius RADIUS] [--days DAYS] [--stations STATIONS]
                       [--api-key API_KEY] [--output OUTPUT]
```

| Flag | Default | Description |
|---|---|---|
| `--synthetic` | off | Generate synthetic diurnal station data |
| `--lat` / `--lon` | 34.05 / -118.24 | Centre of metro area (Los Angeles) |
| `--radius` | 50 | Search radius in km (real API only) |
| `--days` | 7 | Number of past days to fetch |
| `--stations` | 40 | Number of synthetic stations |
| `--api-key` | – | OpenAQ v2 API key |
| `--output` | `data/measurements.csv` | Output CSV path |

### `grid_and_serialize.py`

```
usage: grid_and_serialize.py [-h] [--input INPUT] [--output OUTPUT]
                              [--resolutions N [N ...]]
```

Produces `data/grids/grid_<H>x<W>.npy` – float32 tensors of shape
`(T, C, H, W)` where `T` = time steps, `C` = 2 pollutants, `H = W` =
resolution.

### `benchmark_pipeline.py`

```
usage: benchmark_pipeline.py [-h] [--input INPUT] [--grid-dir GRID_DIR]
                              [--results-dir RESULTS_DIR]
                              [--resolutions N [N ...]]
                              [--concurrency N [N ...]] [--n-requests N]
                              [--synthetic]
```

Runs all three experiments in sequence and writes plots + JSON to
`--results-dir`.  Pass `--synthetic` to auto-generate data if the CSV is
missing.

---

## Experiments

### Experiment 1 – Pipeline stage breakdown

Measures per-sample wall-clock time for each stage at 16 × 16 resolution.
The main insight figure for the paper.

### Experiment 2 – Grid-size scaling

Sweeps resolutions `[8, 16, 32, 64]` and records preprocessing latency,
throughput (windows/s), and tensor memory footprint.

### Experiment 3 – Serving scaling

Simulates concurrent Triton-style serving via a Python thread pool; sweeps
concurrency `[1, 2, 4, 8, 16]` and records throughput and p50/p95 latency.
Replace the stub with a real
[Triton Inference Server](https://github.com/triton-inference-server/server)
and
[Perf Analyzer](https://github.com/triton-inference-server/perf_analyzer)
for production-quality results.

---

## Inference stubs

| Stub | Description |
|---|---|
| `inference_last_value` *(default)* | `y_pred = x_last` – last observed time step |
| `inference_fixed_conv` | Tiny 3 × 3 uniform smoothing kernel (no training) |

Both are zero-parameter baselines; no training data or GPU are required.

---

## Repository layout

```
openaq-pipeline-scaling/
├── fetch_openaq.py          # Ingest / generate measurements
├── grid_and_serialize.py    # Gridding & serialisation
├── benchmark_pipeline.py    # All experiments + plots
├── requirements.txt         # Python dependencies
├── results/                 # Generated plots and JSON summary
│   ├── plot1_stage_breakdown.png
│   ├── plot2_grid_scaling.png
│   ├── plot3_serving_scaling.png
│   └── benchmark_results.json
└── README.md
```

---

## Dependencies

```
requests >= 2.31
numpy    >= 1.24
pandas   >= 2.0
matplotlib >= 3.7
scipy    >= 1.11
```

Install with `pip install -r requirements.txt`.
