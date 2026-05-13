# AI Cup 2026 -- Bird Radar Track Classification

Solution for the [AI Cup 2026](https://www.teamepoch.ai/ai-cup-2026-2/) Performance Track, organized by Epoch (TU Delft). The task is to classify bird radar tracks from Windpark Eemshaven (Groningen, Netherlands) into nine classes, evaluated by macro-averaged Average Precision (mAP).

## Problem

MAX Avian Radar systems at the Eemshaven wind farm generate tracks for detected flying objects. Each track contains a trajectory (longitude, latitude, altitude, radar cross-section over time) and a set of pre-computed radar attributes. The goal is to identify which bird group (or clutter) produced each track.

**Nine classes:** Clutter, Cormorants, Pigeons, Ducks, Geese, Gulls, Birds of Prey, Waders, Songbirds.

**Key challenges:**

- Small dataset: 2,601 training samples.
- Extreme class imbalance: Gulls dominate at 1,503 samples (57.8%), while Cormorants have only 40 samples (1.5%), yielding a 37.6:1 imbalance ratio.
- Macro-averaged mAP metric: all nine classes are weighted equally, meaning performance on the rarest class has as much impact on the final score as performance on the most common.
- Short trajectory sequences: 15-30 points per track, limiting the usefulness of spectral and periodicity features.
- Geographically localized data: all tracks originate from the same wind park, so the model must rely on flight kinematics and radar signatures rather than geospatial distribution.

## Approach

### Two-Stage Architecture

**Stage 1 -- Binary Gull Detector.** A LightGBM binary classifier distinguishes Gulls from all other classes. The predicted Gull probability is used to gate the Stage 2 outputs.

**Stage 2 -- 8-Class Non-Gull Ensemble.** Three gradient-boosted tree models (LightGBM, CatBoost, XGBoost) classify the remaining eight classes. The ensemble uses either optimized weighted averaging or rank averaging on out-of-fold predictions.

The final probability for each class is computed as:

```
P(Gulls) = Stage 1 output
P(class_k) = (1 - P(Gulls)) * Stage 2 output for class_k
```

### Feature Engineering

Features are organized into four groups (89 features total, plus optional 32-dimensional CNN embeddings):

**Base Features (20)** -- Radar attributes and temporal context:

- Airspeed, altitude range (min_z, max_z), track duration.
- Categorical radar bird size (Large, Medium, Small bird, Flock).
- Hour and month with cyclical sin/cos encoding.
- Daytime indicator.
- Derived ratios: airspeed per meter of altitude, altitude range.

**Trajectory Features (33)** -- Extracted from EWKB-encoded track geometry:

- Point count, total path length, mean and std of step distance.
- Longitude and latitude range, mean position.
- Altitude statistics: mean, std, climb rate, descent rate, variability.
- RCS statistics: mean, std, min, max, range.
- Tortuosity (mean and max bearing change), sharp turn ratio.
- Straightness index (net displacement / total path length) and sinuosity index.
- Track heading (overall direction).
- Curvature statistics derived from consecutive displacement cross products.
- Log-transformed path length.
- Speed statistics: mean, std, max, coefficient of variation.
- Acceleration statistics: mean, std.

**Weather Features (16-31 depending on dataset)** -- Joined from meteorological sources via `merge_asof` on track temporal midpoint:

- KNMI station 286: hourly mean wind speed, wind speed at observation, max wind gust, air temperature, dew point, sunshine duration, global radiation, precipitation duration and amount, relative humidity, weather indicator code, wind direction (sin/cos encoding, variable wind flag).
- Open-Meteo archive API: temperature at 2m, relative humidity, dew point, precipitation, cloud cover, pressure, weather code, wind speed/direction/gusts at 10m, shortwave/direct/diffuse radiation, sunshine duration, vapour pressure deficit, daytime flag.

**Interaction Features (6)** -- Derived from combinations of radar, weather, and trajectory data:

- RCS-to-speed ratio.
- Altitude-adjusted wind speed (Hellmann power law, exponent 0.143).
- True tailwind and crosswind components from track heading vs wind direction.
- Headwind component and airspeed-to-wind ratio.

**CNN Embeddings (32, optional)** -- A 1D convolutional neural network processes trajectory sequences (6 channels: dx, dy, dz, dt, speed, RCS) padded/truncated to 128 steps. Architecture: three convolutional blocks (32, 64, 128 filters), adaptive pooling, and a 32-dimensional embedding layer. Trained with Focal Loss (gamma=2.0), AdamW optimizer, cosine annealing, and rotation + jitter augmentation. Outputs are merged as additional tabular features.

### Class Imbalance Mitigation

Multiple complementary strategies:

1. **Class weighting** (`class_weight='balanced'` in LightGBM, `auto_class_weights='Balanced'` in CatBoost, computed sample weights in XGBoost).
2. **Oversampling** via SMOTENC, BorderlineSMOTE-1, or ADASYN applied only to minority classes within the Stage 2 training pipeline.
3. **Per-class boost multipliers** -- minority class rows (Cormorants, Waders, Geese) are duplicated `N` times in the training set before fitting.
4. **Trajectory augmentation** -- before feature extraction, minority class trajectories are augmented with spatial jitter, random rotation, and time warping to create realistic synthetic samples.
5. **Gull undersampling** -- Gulls can be randomly downsampled to a specified limit within each training fold.

### Cross-Validation

10-fold Stratified Group K-Fold using `primary_observation_id` (multiple tracks from the same observation grouped together to prevent data leakage). Group-based splitting is critical because a single bird/flock observation can produce multiple radar tracks.

## Configurations Tested

All experiments were evaluated with out-of-fold (OOF) mAP on the training set.

### Grid Search: Dataset / Ensemble / Two-Stage / Boost-Weak (24 combinations)

A full factorial grid search over four binary axes to identify the best pipeline architecture:

| Axis | Options |
|------|---------|
| Weather dataset | KNMI only, OpenMeteo only, Combined (both) |
| Ensemble (LightGBM + CatBoost) | Off, On |
| Two-stage Gull detector | Off, On |
| Weak-class boost (3x row duplication) | Off, On |

Best combination: OpenMeteo + Ensemble + Two-Stage, achieving mAP 0.6698. The two-stage architecture provided the largest individual gain (+0.0046 over the single-stage ensemble).

### Oversampling Methods

| Method | mAP | Description |
|--------|-----|-------------|
| SMOTENC (baseline) | 0.6999 | SMOTE with categorical feature support |
| BorderlineSMOTE-1 | 0.7003 | Synthesizes samples only near decision boundaries |
| ADASYN | 0.7041 | Adaptive density-based; more samples for harder instances |
| Trajectory augmentation | 0.7028 | Augments EWKB coordinates before feature extraction |
| Trajectory (0.5x jitter) | 0.7059 | Halved jitter magnitudes for more realistic synthesis |
| ADASYN + trajectory | 0.7000 | Methods interfere when combined |

ADASYN was the best verified single method. Halved-jitter trajectory augmentation produced the best Ducks AP (0.7531) and the highest Waders AP among non-pseudo-label methods (0.3513).

### Ensemble Architecture Evolution

| Configuration | mAP | Notes |
|--------------|-----|-------|
| LightGBM only (baseline) | 0.6550 | Single model, no ensemble |
| LightGBM + CatBoost (equal weights) | 0.6652 | Two-model ensemble |
| LightGBM + CatBoost + XGBoost (equal weights) | 0.7087 | Three-model ensemble |
| + Calibration (Platt/Isotonic) | 0.7097 | Per-class calibration |
| + Optimized weighted ensemble | 0.7119 | Grid-searched weights (w_lgb=0.45) |
| + 5-seed averaging | 0.7112 | Variance reduction through seed diversity |
| + 5-seed avg + calibration + XGBoost | 0.7168 | Full ensemble with rank averaging |

### Pseudo-Labeling

A two-pass approach where high-confidence test predictions (threshold 0.95) are added to training data for a second round of training. Achieved mAP 0.7148, but this score may be inflated due to pseudo-labeled test samples leaking into training folds. The 0.95 threshold selected 936 of 1,872 test samples, heavily biased toward Gulls (811 pseudo-labels).

### Gull Threshold Sweep

The Stage 1 Gull classifier output can be skewed using a monotonic power transform. Testing thresholds from 0.50 to 0.85 showed that higher thresholds penalize Gull predictions, increasing precision at the cost of recall. Best mAP (0.7105) was achieved at threshold 0.80.

### CNN Embeddings (version 2)

A separate pipeline (`solution_v2.py`, `extract_cnn_features.py`) integrates 32-dimensional CNN embeddings trained on the raw trajectory time series. The CNN uses a 6-channel input (dx, dy, dz, dt, speed, RCS), Focal Loss with class-balanced alpha weights, and random undersampling of Gulls to 600 samples. The embeddings are merged with handcrafted features and fed to the two-stage GBDT ensemble.

### Probability Calibration

Per-class calibration on OOF predictions: Platt scaling (logistic regression) for classes with fewer than 100 positive samples (Cormorants, Ducks, Geese, Pigeons, Clutter, Waders, Birds of Prey), Isotonic regression for larger classes (Gulls, Songbirds). After calibration, row probabilities are renormalized to sum to 1.

### Ensemble Weight Optimization

Two aggregation strategies were compared:

- **Weighted averaging:** Grid search over (w_lgb, w_cb, w_xgb) in 5% increments, evaluating macro mAP on OOF predictions. Best weights: LightGBM 0.45, CatBoost 0.25, XGBoost 0.30.
- **Rank averaging:** Convert each model's per-class predictions to percentile ranks, average the ranks across models. More robust to different calibration scales between models. Final submission used rank averaging (mAP 0.7168).

## Project Structure

```
.
├── solution.py                  # Main two-stage pipeline with CLI
├── solution_v2.py               # Version 2 with CNN embedding integration
├── submission.ipynb             # Jupyter notebook version of the pipeline
├── test-notebook-aicup2026.py   # Starter notebook (baseline reference)
├── join_knmi_286.py             # KNMI hourly weather merge script
├── join_openmeteo_weather.py    # Open-Meteo archive API fetch and merge
├── join_all_weather.py          # Combined KNMI + Open-Meteo merge
├── extract_cnn_features.py      # 1D CNN trajectory embedding training
├── join_cnn_features.py         # CNN embedding merge with tabular data
├── grid_search_configs.py       # Automated 24-config grid search harness
├── grid_search_results.txt      # Grid search output (sorted by mAP)
├── diagnose_overfit.py          # Overfitting diagnostic (loss curves, n_est sweep)
├── diagnose_weak_classes.py     # Weak class analysis (confusion, distributions)
├── test_features.py             # Unit test for trajectory feature extraction
├── validate_knmi_merge.py       # Validation of KNMI weather join
├── validate_cnn_merge.py        # Validation of CNN embedding join
├── validate_all_weather.py      # Validation of combined weather join
├── bird-radar-visualisation/    # Web-based radar track visualization tool (submodule)
├── dataset/                     # Data files (see .gitignore for tracked files)
├── plan.md                      # Oversampling experiment log and results
├── evaluation.md                # Competition metric specification
├── dataset_description.md       # Dataset schema documentation
├── bird_radar_classification_research.md   # Literature review and technique guide
├── deep_research.md             # Kaggle competition strategies and ensemble research
├── correction.md                # Dataset errata (observer_position clarification)
└── snippets.md                  # Reference code snippets for baseline improvement
```

## Usage

### Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate
pip install lightgbm catboost xgboost scikit-learn imbalanced-learn shapely pandas numpy mlflow matplotlib seaborn
```

For CNN features: additionally install `torch`.

### Weather Data Preparation

The dataset CSV files must be enriched with weather data before training:

```bash
# KNMI (local station data)
.venv/bin/python join_knmi_286.py --train_csv dataset/train.csv --test_csv dataset/test.csv --knmi_txt dataset/286_2021-2030.txt

# Open-Meteo (archive API)
.venv/bin/python join_openmeteo_weather.py --train_csv dataset/train.csv --test_csv dataset/test.csv

# Combined KNMI + Open-Meteo
.venv/bin/python join_all_weather.py --train_csv dataset/train.csv --test_csv dataset/test.csv --knmi_txt dataset/286_2021-2030.txt
```

### Training a Model

The main pipeline exposes all configuration axes as CLI flags:

```bash
# Default: OpenMeteo, two-stage, SMOTENC, 10-fold CV
.venv/bin/python -W ignore solution.py

# Best verified configuration: ADASYN oversampling
.venv/bin/python -W ignore solution.py --oversampler adasyn

# Trajectory augmentation with reduced jitter
.venv/bin/python -W ignore solution.py --oversampler trajectory --jitter-scale 0.5

# Full ensemble with seed averaging and rank aggregation
.venv/bin/python -W ignore solution.py --dataset openmeteo --oversampler adasyn \
    --boost-weak 3 --n-seeds 5 --gull-threshold 0.8

# Pseudo-labeling experiment
.venv/bin/python -W ignore solution.py --oversampler adasyn --pseudo-label --pseudo-threshold 0.95

# Grid search over all architecture combinations
.venv/bin/python -W ignore grid_search_configs.py

# Overfitting diagnostics
.venv/bin/python -W ignore diagnose_overfit.py
```

### CNN Embedding Pipeline (Version 2)

```bash
# Extract CNN features
.venv/bin/python -W ignore extract_cnn_features.py --epochs 100 --patience 10

# Merge with weather-enriched data
.venv/bin/python -W ignore join_cnn_features.py

# Train with CNN embeddings
.venv/bin/python -W ignore solution_v2.py
```

### Generating a Submission

Both `solution.py` and `solution_v2.py` output `submission.csv` in the expected format (track_id + 9 probability columns).

## Key Findings

1. **Two-stage architecture is essential.** Separating Gull detection from the 8-class problem consistently improved mAP by 0.004-0.005. Gulls are the dominant class and their radar signatures overlap substantially with other species (Cormorants are misclassified as Gulls 52.5% of the time, Waders 47.5%).

2. **ADASYN outperforms SMOTENC and BorderlineSMOTE** for this problem. Its adaptive density-based generation of synthetic samples for harder-to-classify instances aligns well with the macro-averaged metric.

3. **Ensemble diversity matters.** The three-model ensemble (LightGBM + CatBoost + XGBoost) consistently outperformed any single model or two-model combination. Rank averaging was more robust than weighted averaging, avoiding calibration-scale discrepancies between models.

4. **Pseudo-labeling showed promising OOF gains but may be misleading.** The 0.7148 OOF mAP is likely inflated because pseudo-labeled test samples appear across training folds. The verified best configuration without pseudo-labeling achieved mAP 0.7168 after full ensemble optimization.

5. **CNN embeddings did not provide clear improvements** over the handcrafted feature set. The trajectory sequences (15-30 points) are too short for deep learning to extract patterns beyond what domain-specific feature engineering captures.

6. **Weather data source matters.** OpenMeteo archive data matched with `merge_asof` on track midpoint outperformed KNMI station 286 data alone. Combining both sources did not improve over OpenMeteo alone.

7. **Probability calibration consistently helps.** Per-class Platt/Isotonic calibration on OOF predictions improved mAP by approximately 0.001-0.002, and the renormalization to sum-to-1 is critical since the two-stage architecture predicts independently.

## Visualization Tool

The `bird-radar-visualisation/` submodule ([TeamEpochGithub/aicup-birds-radar-detection-tool](https://github.com/TeamEpochGithub/aicup-birds-radar-detection-tool)) provides a web-based tool for inspecting individual radar tracks with their predicted and true class labels. It is useful for understanding model errors and analyzing the geometry of misclassified tracks.

## License

This project was developed for the AI Cup 2026 competition. See the competition rules regarding data usage and redistribution. The code is provided for reference and educational purposes.
