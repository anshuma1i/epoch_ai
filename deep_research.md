# Winning a bird radar classification competition with 2,600 samples and brutal class imbalance

**The single most impactful strategy for this problem is a three-pronged approach: class-balanced loss functions (focal or logit-adjusted loss), a diverse LightGBM + CatBoost + XGBoost ensemble blended via hill climbing with a Ridge meta-learner, and domain-specific trajectory/RCS feature engineering — all validated through repeated stratified 5-fold CV.** This combination directly addresses every challenge in the problem: severe imbalance (37.6×), small sample size (~2,600), and the macro-averaged AP metric that weighs the 40-sample minority class equally with the largest class. The recommendations below are drawn from winning Kaggle solutions (ICR, Porto Seguro, MoA, BirdCLEF, LANL Earthquake), Kaggle Grandmaster playbooks, and peer-reviewed research on non-decomposable metric optimization.

---

## Kaggle winners reveal what actually works on tiny imbalanced tabular datasets

The **ICR – Identifying Age-Related Conditions** competition (2023) is the closest Kaggle analog — just **617 training samples**, binary classification with significant class imbalance, and 6,700+ teams. The dominant finding: **TabPFN** (a transformer pre-trained for small tabular data) combined with XGBoost ensembles crushed purely GBDT-based approaches. The 8th-place gold solution by Clayton Kjos predicted sub-conditions rather than the binary target directly, then aggregated probabilities — a form of auxiliary-task learning. A critical lesson from ICR's massive public-to-private leaderboard shake-up: **aggressive probability thresholding destroyed scores** on the private set. Solutions that trusted their stratified CV and avoided LB-chasing survived.

**Porto Seguro's Safe Driver Prediction** (2017, 26:1 class imbalance) produced the breakthrough insight from 1st-place finisher Michael Jahrer (Netflix Prize winner): **denoising autoencoders (DAEs) with swap noise** — replacing 15% of features with values from random rows — learned superior representations from train+test features in an unsupervised manner. His blend of 1 LightGBM + 5 neural networks trained on DAE activations won with simple averaging (equal weights). Critically, **upsampling, SMOTE, and nonlinear stacking all failed** in his experiments.

The **Mechanisms of Action** competition (2020, 206 multi-label targets with many rare labels) demonstrated that **multi-task learning on auxiliary targets** dramatically improved rare-label performance. The 1st-place team trained on all 609 targets (including 403 non-scored ones) before fine-tuning on the 206 scored targets — effectively a data augmentation strategy for rare classes through shared representation learning.

**BirdCLEF 2024** (most domain-relevant — bird species classification with severe imbalance, macro-averaged ROC-AUC metric) showed pseudo-labeling on unlabeled soundscapes as the key differentiator. The 1st-place team capped samples per species at **500** to prevent majority-class domination. The 6th-place solution found that **post-processing with per-species global average correction** consistently improved scores by 0.014–0.016. The 2025 iteration confirmed **focal loss** as standard practice for class imbalance.

The NVIDIA Kaggle Grandmasters Playbook (Onodera, Viel, Titericz, Deotte — all 4× Grandmasters) distills seven battle-tested techniques. Most applicable here: **hill climbing ensembles** (start with best model, greedily add models that improve CV), **multi-seed training** (average 10–100 seeds for robustness), and **pseudo-labeling with soft probabilities** rather than hard labels.

---

## The ensemble architecture that consistently wins small-data competitions

For small tabular datasets, the research strongly favors **stacking over blending** because blending wastes precious data on a holdout set. The optimal architecture for ~2,600 samples:

**Base layer**: LightGBM + CatBoost + XGBoost, each trained with different hyperparameters and 3–5 random seeds per model (yielding 9–15 base predictions). Generate out-of-fold (OOF) predictions using **8–10 fold stratified CV** to maximize training data per fold. Add diversity through **different feature subsets** per model — Mario Filho's framework (Telstra competition winner) found this is the second most effective diversity dimension after algorithm diversity. Each model "overfitting to its constrained feature space" creates specialists whose errors are uncorrelated.

**Meta-layer**: Use **Ridge regression or logistic regression** exclusively — never GBDTs or neural networks as meta-learners on small data. The Kaggle consensus is unambiguous: "the hard work has already been done by the base learners." Adding original features alongside OOF predictions to the meta-learner increases overfitting risk and should only be done with strong L1 regularization. The **Don't Overfit II** competition (just 250 training samples) confirmed that LassoCV was the winning approach for extreme small-data scenarios.

**Weight optimization alternatives**: Hill climbing (Caruana's forward selection with replacement) is the Grandmaster-preferred method — it won Playground Series S5E12 outright. The specific algorithm: start with the single best model, try adding every other model, keep only the combination that improves CV score, repeat. **Optuna-based Bayesian optimization** of blend weights is a viable alternative — sample weights from `trial.suggest_float()`, normalize, evaluate on all CV folds. A key caution: **never optimize weights on a single holdout or public LB**; always use the full CV loop.

**Rank averaging** deserves special mention as the safest ensemble method: convert each model's predictions to percentile ranks, then average. This handles models with different calibration scales and is especially effective when the metric cares about ranking (as AP does). The MLWave Kaggle Ensembling Guide — the definitive reference by Henk van Veen — reports rank averaging gave a "hefty increase" in the Avito challenge.

The critical diversity check: compute pairwise correlations between model OOF predictions. If all correlations exceed **0.95**, the ensemble provides minimal benefit. Target correlations of 0.80–0.90 between base models for meaningful diversity gains.

---

## Scientific foundations for optimizing macro-averaged AP under imbalance

**Macro-AP is a non-decomposable, threshold-free ranking metric** — it cannot be directly optimized through standard loss functions. The theoretical framework from Narasimhan, Kar, and Jain (ICML 2015, "Optimizing Non-Decomposable Performance Measures") establishes the principled approach: **(1) train a class probability estimator (CPE)**, then **(2) post-hoc optimize for the target metric**. For macro-averaged metrics, this means training GBDTs with a good surrogate loss (log-loss or class-balanced cross-entropy), then ensuring per-class probability rankings are as accurate as possible.

Ye et al. (ICML 2012, "Optimizing F-Measure: A Tale of Two Approaches") proved that for rare classes, the **Decision-Theoretic Approach** — train a probabilistic model, then predict with maximum expected metric value — outperforms Empirical Utility Maximization. This directly applies: train the GBDT ensemble to produce well-calibrated probabilities, then focus post-processing on improving per-class ranking quality.

**Probability calibration is essential for macro-AP under imbalance.** Kull et al. (NeurIPS 2019) introduced **Dirichlet calibration (Dir-ODIR)** — a natively multi-class method that applies a log-transform to uncalibrated probabilities, followed by a linear layer and softmax with off-diagonal + bias regularization. For 9 classes with limited data, this outperforms per-class Platt scaling or isotonic regression (which overfits on small calibration sets). A critical nuance: since AP is a ranking metric, calibration only helps if it **changes the ranking** of predictions — monotone transformations that preserve order leave AP unchanged. However, in multi-class one-vs-rest AP computation, calibration across classes affects relative rankings and thus macro-AP.

For class imbalance specifically, the most comprehensive empirical study (arXiv 2409.19751, 9,000 experiments across 15 models and 30 datasets) found that **decision threshold calibration was the most consistently effective technique**, outperforming both SMOTE and class weights. Since macro-AP is threshold-free, the equivalent strategy is **post-hoc logit adjustment**: `adjusted_logit[k] = logit[k] + log(π_test[k] / π_train[k])`, which rebalances predicted probabilities to account for the mismatch between imbalanced training distribution and the equal-weight macro metric.

The **class-balanced loss using effective number of samples** (Cui et al., CVPR 2019) provides a principled weighting scheme: `weight_k = (1-β) / (1-β^n_k)` where `n_k` is class size and β is typically 0.9–0.9999. This outperforms simple inverse-frequency weighting. A 2022 Journal of Cheminformatics study benchmarking custom losses for imbalanced GBDTs found that **logit-adjusted loss and LDAM loss outperformed focal loss**, converging 4–8× faster than weighted cross-entropy. These logit-shifting strategies are implementable as custom objectives in LightGBM.

---

## Feature engineering from radar tracks: what the LANL earthquake winners and bird radar literature teach us

The **LANL Earthquake Prediction** competition (2019, 4,500+ teams) is the best Kaggle analog for time-series-to-tabular feature engineering. The 1st-place team (Team Zoo) achieved a remarkable finding: their best LightGBM model used **only 4 features** — number of peaks on the denoised signal, 20th percentile of rolling-window standard deviation, and two MFCC (Mel-frequency cepstral coefficient) means. This demonstrates that **feature quality vastly outweighs feature quantity** for gradient-boosted trees.

For the bird radar problem, the recommended feature engineering pipeline should extract features from two distinct signal types:

**GPS trajectory features** should capture kinematics and geometry. Primary features include speed statistics (mean, std, max, min, median, skewness, kurtosis), acceleration statistics (same breakdown), turning angle statistics (mean absolute value, circular variance, circular standard deviation), and geometric measures (straightness index = beeline/path length, sinuosity, total path length, bounding box aspect ratio, convex hull area). A review of GPS trajectory classification literature confirms that **auto- and cross-correlations, kurtoses, and skewnesses of speed and acceleration** are significant discriminators, with XGBoost outperforming traditional classifiers on the GeoLife transportation mode dataset.

**RCS (radar cross section) time series features** should exploit the physical phenomenon that bird RCS varies across ~5 orders of magnitude due to wing flapping, creating characteristic periodic modulation patterns. Key features: basic statistics (mean, std, percentiles at 5th/25th/75th/95th), temporal dynamics (RCS trend slope, number of peaks, rolling-window std at multiple scales), autocorrelation (ACF at lags 1–5, first ACF minimum — captures wingbeat periodicity), and spectral features (dominant frequency for wingbeat identification, spectral centroid, MFCC coefficients 1–4, spectral entropy). Urmy (2017) found that **mean echo level and 90th percentile of echo level** were the top features for bird detection in marine radar, achieving 99.99% accuracy for bird vs. non-bird classification.

The three main feature extraction libraries offer distinct tradeoffs:

- **catch22/catch24** (Lubba et al., 2019): 22 canonical features with the **lowest redundancy** of any automated feature set (50% of PCs needed for 90% variance). Computes in ~0.1ms per feature. Only 7% accuracy reduction versus the full 4,791-feature hctsa set across 93 UCR datasets. Use catch24 (adds mean + standard deviation) since raw scale matters for RCS classification. This is the recommended starting point.

- **tsfresh**: ~779 features with the most comprehensive coverage but **51% are FFT-derived** and highly redundant (90% of variance captured by just 55 PCs). The Volcanic Eruption Prediction competition used tsfresh's ComprehensiveFCParameters → 8,000 features → correlation filtering → 2,854 → recursive feature elimination → 501 final features. Always pair with aggressive filtering.

- **TSFEL**: 65+ features across statistical, temporal, spectral, and fractal domains. Fastest computation but **highest redundancy** (90% variance in just 4 PCs). Best for rapid prototyping.

The practical strategy: start with catch24 on RCS + manual trajectory features (~50 features), add domain-specific RCS percentiles and spectral features (~30 more), then selectively augment with tsfresh EfficientFCParameters (~100 after filtering). **Feature selection must be performed inside CV folds** — Kuncheva et al. (2018) demonstrated that feature selection outside CV introduces severe optimistic bias. Use BorutaSHAP (Boruta with SHAP importance) or recursive feature elimination. Final model should use 50–200 features.

---

## The specific tricks that squeeze out the last few points

**Class imbalance handling (highest impact)**: With a rarest class of ~40 samples, the hierarchy of approaches is: (1) **Class-balanced weights** using effective number of samples (Cui et al. formula) as the baseline — set CatBoost's `auto_class_weights='Balanced'` or compute custom weights for LightGBM's `sample_weight`. (2) **Focal loss** (γ=1–2) or **logit-adjusted loss** as custom objectives — the cheminformatics benchmarking study found logit-adjusted loss converges 4× faster and achieves better F1 than focal loss on imbalanced GBDTs. (3) **Per-class OvR binary models** for the 2–3 rarest classes with specialized hyperparameters. (4) **Avoid SMOTE** when minority classes have fewer than ~100 samples — too few neighbors for meaningful interpolation, and a comprehensive study showed SMOTE produced the worst-calibrated probabilities of all methods tested.

**Cross-validation (foundational)**: Use **RepeatedStratifiedKFold with n_splits=5, n_repeats=3** — this gives 15 evaluations, dramatically reducing CV variance. With ~40 samples in the rarest class, 5-fold gives ~8 validation samples per fold for that class — noisy but workable. Ten-fold would leave ~4 samples, producing unreliable per-class AP estimates. Never apply resampling before the fold split.

**Regularization for 2,600 samples**: Learning rate of **0.01–0.05** with early stopping is critical. CatBoost's **ordered boosting** (permutation-driven gradient computation on held-out observations) provides built-in overfitting prevention specifically designed for small datasets — the 2023 GBDT benchmarking study (arXiv 2305.17094) found CatBoost's symmetric trees serve as structural regularization. For LightGBM, constrain max depth to **4–6**, use row subsampling at 0.7–0.8, column subsampling at 0.7–0.8, and increase `min_data_in_leaf` to prevent splits on tiny groups. Set `reg_lambda` (L2) to 1–10.

**Pseudo-labeling**: LOW priority for this problem. Luca Massaron (Kaggle Grandmaster) warns that pseudo-labeling "may even worsen results if there is some added noise." With 2,600 samples and severe imbalance, the risk of confirmation bias on rare classes is high. If attempted, only pseudo-label majority classes with prediction confidence >0.95, weight pseudo-labeled samples at 0.5×, and validate improvement through the full CV pipeline.

**Test-time augmentation for tabular data**: Effectively not applicable to GBDTs, which are deterministic at inference. The functional equivalent is **multi-seed ensembling** — averaging predictions from models trained with 5–10 different random seeds, which the Playground Series S5E6 winner demonstrated improved MAP@3 steadily across 100 seeds.

**Label smoothing**: Does not apply to standard GBDTs, which learn by fitting residuals to hard labels. The GBDT-native alternatives — learning rate shrinkage, L1/L2 regularization, and max depth constraints — serve the same regularization purpose. A recent paper (arXiv 2310.05067) introduces **Robust Focal Loss** for GBDTs specifically designed for noisy labels + class imbalance, combining nonconvex robust loss functions with gradient boosting.

**Post-hoc logit adjustment** is a cheap, high-value trick: after training on the natural class distribution, shift predicted logits at test time by `log(π_macro / π_train)` where π_macro = 1/9 (equal weighting implied by macro-AP) and π_train is the observed class frequency. This costs nothing computationally and directly aligns predictions with the metric's equal-weight-per-class assumption.

---

## Conclusion: a prioritized action plan

The research converges on a clear execution sequence. **First**, build the evaluation foundation: repeated stratified 5-fold CV with per-class AP computed and averaged. **Second**, invest heavily in domain-specific feature engineering — catch24 features on RCS time series, trajectory kinematics, and spectral features capturing wingbeat periodicity, targeting 80–150 final features selected via BorutaSHAP inside CV folds. **Third**, train a diverse base layer of LightGBM (leaf-wise, focal loss), CatBoost (ordered boosting, balanced class weights), and XGBoost (depth-wise, logit-adjusted loss), each with 3–5 random seeds and complementary feature subsets. **Fourth**, blend via hill climbing on CV scores or Ridge meta-learner on OOF predictions — never on a single holdout. **Fifth**, apply post-hoc Dirichlet calibration and logit adjustment to align predictions with the macro-AP metric.

The single most underappreciated insight across the literature: for macro-averaged metrics, **a model that achieves mediocre performance on all 9 classes outscores a model that excels on 8 but fails on 1**. Every technique should be evaluated through this lens — per-class AP for the rarest class is the binding constraint, and the competition will likely be won or lost on how well the 40-sample minority class is handled.