# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ALWAYS USE THE .VENV

Always use `.venv/bin/python`, never bare `python`.

## USE CONTEXT7

## IF THERE ARE MULTIPLE OPTIONS, AND I CHOOSE ONE, WRITE THE ALTERNATIVES IN ALTERNATIVES.MD FILE

## Project Overview

9-class bird radar track classification for AI Cup 2026. Radar tracks from MAX Avian Radar at Windpark Eemshaven (Groningen, NL). Classes: Clutter, Cormorants, Pigeons, Ducks, Geese, Gulls, Birds of Prey, Waders, Songbirds. Gulls dominate at ~58%; weak classes are Cormorants (40), Waders (120), Geese (83). Metric: macro-averaged AP (mAP).

## Running the Solution

```bash
# Default run — two-stage + ensemble Stage 2 + boost-weak 3 (mAP ~0.70)
.venv/bin/python solution.py

# Use CatBoost for Stage 1 instead of LightGBM
.venv/bin/python solution.py --stage1-catboost

# Gull threshold tuning + undersampling
.venv/bin/python solution.py --gull-threshold 0.8 --undersample-gulls 500

# Disable MLflow tracking
.venv/bin/python solution.py --no-mlflow

# View experiment history
.venv/bin/mlflow ui --port 5000
```

Output: `submission.csv` with per-class probabilities.

## Architecture (solution.py)

**Pipeline:** Load CSV → trajectory feature extraction (EWKB hex → shapely) → feature matrix (66 features) → two-stage classifier → 10-fold StratifiedGroupKFold (grouped by `primary_observation_id`) → submission.

**Three feature groups:**
- Base (20): airspeed, altitude, duration, radar_bird_size, cyclical time encodings, wind interactions, RCS-speed ratio, altitude-adjusted wind, tailwind/crosswind
- Trajectory (30): decoded from EWKB hex — distances, speeds, tortuosity, sharp turn ratio, sinuosity, RCS stats, climb rates, position, heading, speed CV
- Weather (15-31): from KNMI station 286 and/or Open-Meteo API, selected via `--dataset`

**Two-stage classifier (always on):**
- **Stage 1**: Binary Gull vs non-Gull (LightGBM default, `--stage1-catboost` to switch). Optional threshold recalibration (`--gull-threshold`) and Gull undersampling (`--undersample-gulls`).
- **Stage 2**: 8-class non-Gull classifier — LightGBM + CatBoost ensemble (always). LightGBM uses SMOTENC oversampling + boost-weak row duplication. CatBoost uses balanced weights.
- **Combine**: `p(Gull)` from Stage 1, `p(other) = (1 - p_gull) * ensemble_p(class | non-gull)`

**MLflow tracking** (enabled by default, `--no-mlflow` to disable): logs params, per-fold mAP, per-class AP, and submission.csv to `./mlruns/`.

## Key Files

- `solution.py` — main pipeline with all CLI flags
- `submission.ipynb` — notebook version of the default pipeline (for submission)
- `grid_search_configs.py` — runs config combos, ranks by mAP
- `diagnose_weak_classes.py` — weak class confusion analysis
- `join_openmeteo_weather.py`, `join_knmi_286.py`, `join_all_weather.py` — weather data merging
- `dataset_description.md` — official dataset/column documentation
- `dataset/` — train/test CSVs (original + weather-merged variants)

## Weather Datasets

| Flag | Train file | Features |
|------|-----------|----------|
| `--dataset knmi` | `train_with_knmi_286.csv` | 15 KNMI station features |
| `--dataset openmeteo` (default) | `train_with_openmeteo.csv` | 16 Open-Meteo API features |
| `--dataset all` | `train_with_all_weather.csv` | 31 combined features |

OpenMeteo performs best. Wind speeds in OpenMeteo are km/h (converted via /3.6).

## Class Imbalance

Imbalance ratio 37.6x. Handled by: SMOTENC oversampling, `class_weight='balanced'`, `--boost-weak 3` default (row duplication for Cormorants/Waders/Geese), and two-stage Gull separation.

## Git Conventions

Commit messages: short lowercase, no co-author tags (match existing git log style).
