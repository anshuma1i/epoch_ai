# Bird Radar Track Classification: Research & Techniques Guide
## AI Cup 2026 — MAX Avian Radar, 9 Classes, ~2600 Samples, Macro-Averaged AP

---

## 1. Radar Bird Classification Literature

### 1.1 What RCS Tells You (and What It Doesn't)

Radar cross section is correlated with body size but is not a clean species discriminator on its own. Research at JFK airport found that large goose flocks produce RCS values of 1,000–10,000 cm², while smaller birds fall in the 10–100 cm² range (Nohara et al., reported in Gong 2020, IET Radar, Sonar & Navigation). However, RCS varies enormously with aspect angle, distance from beam axis, and flight mode, so absolute RCS at a single time step is noisy. The TNO report on radar bird detection notes that average RCS for large birds is approximately −20 dBm² at 10 GHz, but this fluctuates by 10 dB or more due to wingbeat modulation.

**Actionable features from RCS time series:**

- **Mean, median, and percentiles of RCS** (10th, 25th, 75th, 90th) — captures body size distribution
- **RCS variance / standard deviation** — reflects wingbeat modulation amplitude; larger birds with slower wingbeats produce different variance signatures than small passerines
- **RCS range (max − min)** — the "wingbeat corner reflector" effect can amplify RCS by up to 10 dB during flapping phases; this range encodes wingbeat intensity
- **RCS temporal autocorrelation** at various lags — can capture wingbeat periodicity even without Doppler data
- **Peak frequency of RCS oscillation** (via FFT or autocorrelation) — a proxy for wingbeat frequency. Wingbeat frequency scales inversely with body mass (~3.65 × mass^−0.313, Bruderer et al. 2010). With 15–30 points per track, you may be able to extract the dominant period for species with clear flapping patterns.
- **RCS modulation depth** — the ratio of RCS standard deviation to mean, capturing how much the signal fluctuates relative to baseline body reflection

### 1.2 Flight Behavior Features from Trajectory

Rosa et al. (2016) classified avian radar tracks into five bird groups (herons, gulls, swallows, storks, and other) using machine learning with features including airspeed, flight angle, and echo shape, achieving accuracy, sensitivity, and specificity above 0.75 for all classes. Urmy & Warren (2020) used random forests with features including echo size, eccentricity, and RCS percentiles for classifying radar targets.

**Key trajectory-derived features for your data:**

- **Flight speed statistics** — mean, max, variance, percentiles of ground speed (computed from consecutive lat/lon/time)
- **Altitude statistics** — mean altitude, altitude range, altitude trend (slope of linear fit), altitude variance. Different species fly at characteristically different heights.
- **Flight mode indicators** — the ratio of time spent in different flight modes (flapping vs. gliding) can be inferred from RCS variance within sliding windows. Gong (2020) showed the wingbeat corner reflector produces strong modulation during flapping but fades during gliding.
- **Heading change rate (HCR)** — how erratically the bird changes direction; swallows and insectivorous species show high HCR, while migrating geese fly straight.
- **Straightness ratio** — straight-line displacement divided by total path length (also called straightness index or net-to-gross displacement ratio)

### 1.3 The MAX Radar Context

The MAX system by Robin Radar Systems provides 360° 3D tracking with fast rotation speed, outputting latitude, longitude, altitude, and track-level features per scan. The system is designed for bird detection at airports and wind farms. Since MAX already tracks and classifies at a coarse level, your track data likely represents post-tracking output (one measurement per radar sweep per track). This means your RCS time series reflects the track-averaged RCS at each sweep, not raw pulse-level data — which limits access to micro-Doppler but still preserves wingbeat-scale modulation if the rotation rate is fast enough relative to wingbeat frequency.

---

## 2. Handling Small Tabular Datasets with Severe Class Imbalance

### 2.1 Class Weighting > Resampling

For gradient boosted trees with ~2600 samples and a 37.6× imbalance ratio, the most reliable approach is **class weighting** rather than synthetic oversampling:

- **LightGBM**: Use `is_unbalance=True` or set `class_weight='balanced'` (or manual weights proportional to inverse class frequency)
- **CatBoost**: Use `class_weights` parameter or `auto_class_weights='Balanced'`
- **Why not SMOTE**: With only 40 samples in the smallest class, SMOTE generates synthetic feature vectors by interpolating between neighbors. In a 100+ dimensional feature space (after feature engineering), these synthetic points may not represent realistic bird tracks. Research by Van Calster et al. (2022, PMC) showed that imbalance correction methods can severely distort probability calibration, which directly harms your Average Precision metric.

### 2.2 Focal Loss / Custom Objectives

Consider implementing focal loss as a custom objective for LightGBM:

```python
def focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
    """Multi-class focal loss for LightGBM custom objective."""
    # Down-weights well-classified examples, focuses learning on hard/rare cases
    # gamma=2.0 is standard; increase for more focus on minority classes
```

This naturally focuses training on hard-to-classify (often minority) examples without explicit resampling.

### 2.3 Training Multiple Seeds with Diversity

With <3000 samples, single model variance is high. Top Kaggle competitors consistently train multiple models with different random seeds and average predictions. For your setup:

- Train 5–10 LightGBM models with different `random_state` values
- Train 5–10 CatBoost models with different `random_seed` values
- Average the predicted probabilities before evaluation

This "seed averaging" is nearly free and reduces variance substantially, which is critical when individual fold sizes are small.

### 2.4 Stratified Repeated K-Fold

With 40 samples in the smallest class, even stratified 5-fold CV means only ~8 samples per fold for that class. Use **stratified repeated K-fold** (e.g., 5-fold × 3 repeats = 15 splits) to get more stable OOF predictions and better estimates of per-class AP.

### 2.5 Per-Class Optimization

Since macro-averaged AP treats all 9 classes equally, your smallest class has as much weight in the final metric as your largest. Practical implications:

- Monitor per-class AP during development, not just the macro average
- If one class is at 0.3 AP while others are at 0.9, the marginal gain from improving the weak class is much higher
- Consider training specialized binary classifiers (one-vs-rest) for the weakest classes, then blending their outputs with the main multi-class model

---

## 3. Time Series Feature Engineering for Short Sequences (15–30 Points)

### 3.1 Feature Library Strategy

With sequences of only 15–30 points, you need features that are robust to short lengths. An empirical comparison by Lubba et al. (2021, arXiv 2110.10914) found that **catch22** (22 features) is extremely fast and captures most of the discriminative power of much larger sets, while **tsfresh** (up to 1558 features) provides the broadest coverage but has high internal redundancy. For short sequences with ~2600 samples, the risk of overfitting from tsfresh's full feature set is real.

**Recommended approach:**

1. **Start with catch22** on each channel (RCS, speed, altitude, heading) — gives 22 × 4 = 88 features that are computationally cheap and well-studied
2. **Add tsfresh with `MinimalFCParameters` or `EfficientFCParameters`** for broader coverage — the minimal set computes ~10 features per series (mean, variance, median, min, max, length, sum of absolute changes, etc.)
3. **Use tsfresh's built-in relevance filtering** (`select_features()`) to automatically remove irrelevant features using hypothesis tests, which helps prevent overfitting

### 3.2 Domain-Specific Features (Most Important!)

The automated feature libraries won't capture bird-specific physics. These hand-crafted features are likely your biggest competitive advantage:

**From the spatial trajectory (lon, lat, time):**
- Speed: mean, std, max, min, median, percentiles (25th, 75th, 95th)
- Acceleration: mean, std, max absolute value
- Turning angle: mean absolute, std, max, circular variance
- Sinuosity index (Benhamou 2004): corrected ratio of path length to displacement
- Straightness ratio: net displacement / total distance
- Heading change rate: mean and variance of angular velocity
- Stop rate: fraction of time steps with speed < threshold
- Total track duration and total distance traveled
- Net displacement (start to end)

**From altitude series:**
- Mean, std, range, trend (linear slope), curvature
- Altitude gain rate vs. altitude loss rate (asymmetry indicates soaring vs. powered flight)
- Fraction of time ascending vs. descending vs. level

**From RCS series (the most discriminative for species):**
- Mean, median, std, IQR, skewness, kurtosis
- Range (max − min) — wingbeat modulation depth
- Coefficient of variation (std/mean)
- Number of peaks (crude wingbeat count)
- Dominant period via autocorrelation or FFT (even with 15 points, a strong 3 Hz wingbeat at 1 Hz sampling would show a period of ~3 samples)
- RCS trend (linear slope — increasing RCS may indicate approaching bird)
- RCS quantile ratio: (90th percentile − 10th percentile) / median

**Cross-channel features:**
- Correlation between RCS and speed (different for flapping vs. soaring species)
- Correlation between altitude change and RCS variance
- Speed × RCS product statistics (proxy for kinetic energy × detectability)

### 3.3 Handling Variable-Length Sequences

Since trajectories have different lengths:
- Compute summary statistics (quantiles, moments) that are length-invariant
- For any windowed features, normalize by sequence length
- Include sequence length itself as a feature — it correlates with detection persistence, which relates to bird size and flight behavior
- Consider padding/truncating to a fixed length only if using features that require it (e.g., fixed-length FFT)

### 3.4 Spectral Features for Short Series

With only 15–30 points, traditional FFT has very limited frequency resolution. Better alternatives:
- **Autocorrelation function** at lags 1 through N/2 — more robust than FFT for short series
- **Lomb-Scargle periodogram** — handles irregular sampling better than standard FFT
- **Wavelet energy** at a few scales — the continuous wavelet transform was used by Zaugg et al. (2008, J.R. Soc. Interface) specifically for classifying bird radar echoes via wing flapping patterns

---

## 4. Probability Calibration for Macro-Averaged AP

### 4.1 Why Calibration Matters for AP

Average Precision (AP) is computed from the precision-recall curve, which depends on the ranking of predicted probabilities AND their absolute values (since precision is computed at each threshold). Unlike AUC-ROC, AP is sensitive to calibration — a model that outputs well-ranked but poorly scaled probabilities will lose AP compared to one with well-calibrated outputs. This is especially true for the macro-averaged variant, where minority class AP is sensitive to the model's confidence on those rare positive cases.

### 4.2 Per-Class Calibration is Essential

Since macro-averaged AP treats each class independently (computing AP per class, then averaging), calibration should also be **per-class**:

- For each class k, treat the problem as binary (class k vs. rest)
- Fit a separate calibrator to the predicted probability P(class=k)
- This allows each class to have its own calibration curve, which is important because the class prior varies 37× across your classes

### 4.3 Platt Scaling vs. Isotonic Regression

- **Platt Scaling (sigmoid method)**: Fits a logistic regression to transform raw probabilities. Works well with small calibration sets because it has only 2 parameters. However, it assumes calibration error is symmetric, which may not hold for imbalanced classes.
- **Isotonic Regression**: Non-parametric, more flexible, but requires more calibration data. With 40 samples in your smallest class, isotonic regression may overfit on that class.

**Recommendation for your setting:**
- Use **Platt scaling** (method='sigmoid' in sklearn's CalibratedClassifierCV) for minority classes (<100 samples)
- Use **isotonic regression** for majority classes (>200 samples)
- Always calibrate using out-of-fold predictions to avoid data leakage
- After calibration, renormalize probabilities to sum to 1 across classes (calibrating per-class independently can break this constraint)

### 4.4 Temperature Scaling

A simpler alternative: divide all logits by a single scalar T (temperature), then apply softmax. Fit T on a validation set to minimize negative log-likelihood. Temperature scaling has only 1 parameter, making it the least likely to overfit:

```python
# Conceptual implementation
# optimal_T found by minimizing NLL on validation set
calibrated_probs = softmax(logits / optimal_T)
```

This preserves rankings (so AUC is unchanged) while improving absolute probability estimates.

### 4.5 Post-Processing Trick: Probability Redistribution

For macro-AP specifically, you may want to slightly "boost" predicted probabilities for minority classes at the expense of majority classes. If your model is under-confident on rare classes (common with class-weighted training), a simple approach:

```python
# After calibration, apply class-specific scaling
# weights inversely proportional to class frequency
adjusted_probs = calibrated_probs * class_scale_factors
adjusted_probs /= adjusted_probs.sum(axis=1, keepdims=True)
```

Tune the `class_scale_factors` on your OOF predictions using the actual macro-AP metric. This is a form of post-hoc threshold/scaling adjustment that directly optimizes the competition metric.

---

## 5. Ensemble Stacking Strategies for <3000 Samples

### 5.1 The Overfitting Danger

With ~2600 samples, stacking is risky because the meta-learner trains on OOF predictions — which are themselves noisy estimates from models trained on even smaller subsets. A complex meta-learner will overfit to this noise.

### 5.2 Recommended Architecture

**Level 0 (Base Models):**
- LightGBM × 3–5 configurations (varying `num_leaves`, `learning_rate`, `feature_fraction`)
- CatBoost × 3–5 configurations (varying `depth`, `l2_leaf_reg`, `learning_rate`)
- Optionally: XGBoost × 2–3 for additional diversity
- Each model trained with stratified 5-fold CV, producing OOF predictions (9 probability columns each)

**Level 1 (Meta-Learner):**
- **L2-regularized logistic regression** (Ridge) is the safest choice for small datasets
- Input: concatenation of all base model OOF probabilities (e.g., 10 base models × 9 classes = 90 features)
- Use `LogisticRegression(penalty='l2', C=0.1, multi_class='multinomial')` — the strong regularization prevents the meta-learner from overfitting to individual base model quirks
- Train with the same stratified K-fold scheme

### 5.3 Why Not a Tree-Based Meta-Learner?

Random forest or another GBDT as the meta-learner can capture non-linear interactions between base predictions. However, with only 2600 samples and 90+ meta-features, a tree meta-learner is very likely to overfit. The Porto Seguro Kaggle competition winner (Michael Jahrer) and the top-ranked solutions consistently used linear meta-learners for exactly this reason. Save the non-linearity for the base models.

### 5.4 Simpler Alternatives That Often Work Better

Before committing to stacking, try these approaches that are safer with small data:

1. **Weighted probability averaging**: Simply average the predicted probabilities with optimized weights. Use scipy.optimize or Optuna to find weights that maximize macro-AP on OOF predictions. With only ~10 weight parameters to optimize, overfitting risk is minimal.

2. **Rank averaging**: Convert each model's predicted probabilities to ranks within each class, average the ranks, then convert back to probabilities. This is robust to different calibration scales across models.

3. **Hill climbing ensemble** (used in NVIDIA Kaggle Grandmasters Playbook): Start with the best single model. Iteratively add models with optimized weights, keeping additions only if they improve OOF macro-AP. Simple, effective, and naturally regularized by the greedy selection process.

### 5.5 Diversity Strategies

The key to ensemble gain is diversity — models that make different errors. Sources of diversity for your setup:

- **Different algorithms**: LightGBM vs. CatBoost vs. XGBoost (different tree-building strategies, regularization approaches)
- **Different feature subsets**: Train some models on trajectory features only, some on RCS features only, some on all features
- **Different hyperparameters**: Shallow trees (depth 3–4) vs. deeper trees (depth 6–8) capture different levels of interaction
- **Different preprocessing**: Train some models with raw features, some with PCA-transformed features, some with target-encoded categorical bins
- **Different sampling**: Train some models on all data with class weights, others on balanced subsamples (random undersampling of majority classes). The Porto Seguro solution specifically used a mix of resampling strategies to generate diverse base models.

### 5.6 Cross-Validation Best Practices

- Use **stratified K-fold** (K=5 or 10) consistently across ALL base models — they must all use the same fold splits
- For stacking: generate OOF predictions from base models, then train meta-learner using nested CV or a separate held-out fold
- With repeated K-fold (e.g., 5-fold × 3 repeats), average the OOF predictions across repeats before feeding to the meta-learner
- Monitor the gap between OOF macro-AP and training macro-AP — a large gap signals overfitting

---

## 6. Putting It All Together: Recommended Pipeline

```
1. Feature Engineering
   ├── Domain features from trajectory (speed, acceleration, turning angle, sinuosity, etc.)
   ├── Domain features from altitude (mean, trend, range, ascent/descent ratio)
   ├── Domain features from RCS (moments, percentiles, periodicity, modulation depth)
   ├── Cross-channel features (RCS-speed correlation, etc.)
   ├── catch22 on each channel (RCS, speed, altitude)
   └── tsfresh MinimalFCParameters on each channel

2. Feature Selection
   ├── Remove near-constant features (variance < threshold)
   ├── Remove highly correlated features (|r| > 0.95, keep one)
   └── Use tsfresh relevance filtering or permutation importance

3. Base Models (all with class weights, stratified 5-fold × 3 repeats)
   ├── LightGBM × 5 configs (seed-averaged within each config)
   ├── CatBoost × 5 configs (seed-averaged)
   └── XGBoost × 3 configs (optional, for diversity)

4. Ensemble
   ├── Try weighted averaging first (optimize weights on OOF macro-AP)
   ├── Try ridge logistic regression stacking if averaging plateaus
   └── Hill climbing to prune underperforming base models

5. Calibration & Post-Processing
   ├── Per-class Platt scaling on OOF predictions
   ├── Renormalize to sum to 1
   └── Optional: class-specific probability scaling tuned on macro-AP
```

---

## Key References

- **Gong (2020)** — "Comparison of radar signatures based on flight morphology for large and small birds," IET Radar, Sonar & Navigation
- **Zaugg et al. (2008)** — "Automatic identification of bird targets with radar via patterns produced by wing flapping," J. R. Soc. Interface
- **Rosa et al. (2016)** — "Classification success of six machine learning algorithms in radar ornithology"
- **Bruderer et al. (2010)** — "Wing-beat characteristics of birds recorded with tracking radar and cine camera"
- **Urmy & Warren (2020)** — "Evaluating the target-tracking performance of scanning avian radars," Methods in Ecology and Evolution
- **Nohara et al.** — "Using Radar Cross-Section to Enhance Situational Awareness Tools for Airport Avian Radars," USU Digital Commons
- **Christ et al. (2018)** — "tsfresh — A Python package for time series feature extraction," Neurocomputing
- **Lubba et al. (2021)** — "An Empirical Evaluation of Time-Series Feature Sets," arXiv 2110.10914
- **Van Calster et al. (2022)** — "The harm of class imbalance corrections for risk prediction models," PMC
- **NVIDIA Kaggle Grandmasters Playbook (2025)** — developer.nvidia.com/blog
