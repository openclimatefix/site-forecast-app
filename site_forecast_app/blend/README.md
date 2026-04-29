# NL Blend

Blends multiple NL solar forecast models into a single national forecast and saves it to the Data Platform.

## Overview

The main application (`app.py`):
1. Loads the latest forecasts for each source model from the Data Platform
2. Blends them together using MAE-based weights
3. Saves the blended forecast to the Data Platform ready to be used by the API
4. This is done for the national location (`nl_national`)

## Algorithm

### 1. Reference time (t0)

The blend reference time is the current UTC time floored to the nearest 15-minute boundary.
All model delays are measured relative to t0.

### 2. Model initialisation times

For each registered model, the most recent forecast initialisation time is fetched from the
Data Platform via `get_latest_forecasts`. If a model has no recent forecast, it is assigned
a penalty delay so that its MAE curve is shifted far enough to lose any weight in the blend.

### 3. Delay-adjusted MAE

Each model's backtest MAE curve (`data/nl_backtest_nmae_comparison.csv`) is shifted rightward
by its delay relative to t0. This means a model that initialised 2 hours late is treated as if
its short-horizon accuracy is only as good as its 2-hour-horizon accuracy, penalising late runs
fairly without discarding them entirely.

### 4. Candidate selection

From the national candidate models defined in `config.yaml`, the single model with the lowest
delay-adjusted MAE (summed across all horizons) is selected. If no candidate beats the backup
model's score, the backup is used alone.

### 5. Blending

The selected candidate and the backup model (`nl_regional_2h_pv_ecmwf`) are blended using a
taper kernel (`[0.75, 0.5, 0.25]`) at the crossover horizon — the point where the candidate's
MAE curve crosses the backup's curve. Before the crossover, the candidate takes full weight;
after it, the backup takes full weight; at the crossover the kernel provides a smooth transition.

Weights always sum to 1.0. p50, p10, and p90 are all blended using the same weights.

### 6. Saving

The blended timeseries is saved to the Data Platform under the `blend` forecaster name, tagged
with the current app version. The location is resolved by name (`nl_national`) from the Data
Platform location map.

## Configuration

All tunable parameters live in `config.yaml`:

| Key | Description |
|-----|-------------|
| `backup_model` | Always-available fallback model |
| `national_candidate_models` | Models considered for the primary blend |
| `blend_kernel` | Taper weights at the crossover zone |
| `min_forecast_horizon_minutes` | Shortest horizon emitted in any blended forecast |
| `scorecard_path` | Path to the backtest MAE CSV (relative to this directory) |
| `forecaster_name` | Name written to the Data Platform |

## Source models

| Model | Role |
|-------|------|
| `nl_regional_2h_pv_ecmwf` | Backup — always used, 2 h init frequency |
| `nl_regional_48h_pv_ecmwf` | Candidate — 48 h range ECMWF |
| `nl_regional_pv_ecmwf_mo_sat` | Candidate — ECMWF + Met Office + satellite |
| `nl_regional_pv_ecmwf_sat` | Candidate — ECMWF + satellite |
| `nl_national_pv_ecmwf_sat_small` | Candidate — national-scale small model |

## Running independently

```bash
uv run python -c "import asyncio; from blend.app import run_blend_app; asyncio.run(run_blend_app())"
```

Required environment variables:
- `DATA_PLATFORM_HOST` — Data Platform gRPC host (default: `localhost`)
- `DATA_PLATFORM_PORT` — Data Platform gRPC port (default: `50051`)

## Tests

Integration test (requires a running Data Platform container):
```bash
uv run pytest tests/integration/test_nl_app_integration.py -v
```

Unit tests:
```bash
uv run pytest tests/unit/ -v
```
