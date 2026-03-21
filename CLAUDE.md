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
# Best known config (mAP ~0.67)
.venv/bin/python solution.py --dataset openmeteo --ensemble --two-stage

# With weak class boosting
.venv/bin/python solution.py --dataset openmeteo --ensemble --two-stage --boost-weak 3

# Gull threshold tuning + undersampling
.venv/bin/python solution.py --dataset openmeteo --ensemble --two-stage --gull-threshold 0.6 --undersample-gulls 500

# Grid search all configs (slow, writes grid_search_results.txt)
.venv/bin/python grid_search_configs.py

# Diagnostics for weak classes
.venv/bin/python diagnose_weak_classes.py
```

Output: `submission.csv` with per-class probabilities.

## Architecture (solution.py)

**Pipeline:** Load CSV → trajectory feature extraction (EWKB hex → shapely) → feature matrix (57 features) → imputation + ordinal encoding → optional SMOTENC oversampling → model training → 10-fold StratifiedGroupKFold (grouped by `primary_observation_id`) → averaged predictions → submission.

**Three feature groups:**
- Base (16): airspeed, altitude, duration, radar_bird_size, cyclical time encodings, wind interactions
- Trajectory (20+): decoded from EWKB hex geometry — distances, speeds, tortuosity, RCS stats, climb rates
- Weather (15-31): from KNMI station 286 and/or Open-Meteo API, selected via `--dataset`

**Model stages (composable via CLI flags):**
1. **LightGBM** (always runs): GPU-accelerated, class_weight=balanced, SMOTENC oversampling
2. **CatBoost ensemble** (`--ensemble`): separate CatBoost model, simple-averaged with LightGBM
3. **Two-stage Gull detector** (`--two-stage`): Stage 1 binary (Gull vs non-Gull) + Stage 2 multi-class on non-Gull subset. Combined: `p(class) = (1 - p_gull) * p(class | non-gull)`. Gull threshold and undersampling configurable.

Final predictions are averaged across all enabled stages.

## Key Files

- `solution.py` — main pipeline with all CLI flags
- `grid_search_configs.py` — runs 48 config combos, ranks by mAP
- `diagnose_weak_classes.py` — weak class confusion analysis
- `join_openmeteo_weather.py`, `join_knmi_286.py`, `join_all_weather.py` — weather data merging
- `dataset_description.md` — official dataset/column documentation
- `dataset/` — train/test CSVs (original + weather-merged variants)

## Weather Datasets

| Flag | Train file | Features |
|------|-----------|----------|
| `--dataset knmi` | `train_with_knmi_286.csv` | 15 KNMI station features |
| `--dataset openmeteo` | `train_with_openmeteo.csv` | 16 Open-Meteo API features |
| `--dataset all` | `train_with_all_weather.csv` | 31 combined features |

OpenMeteo performs best in grid search. Wind speeds in OpenMeteo are km/h (converted via /3.6).

## Class Imbalance

Imbalance ratio 37.6x. Handled by: SMOTENC oversampling, `class_weight='balanced'`, optional `--boost-weak N` (row duplication for Cormorants/Waders/Geese), and two-stage Gull separation. Weak classes are overwhelmingly misclassified as Gulls.

## Git Conventions

Commit messages: short lowercase, no co-author tags (match existing git log style).
