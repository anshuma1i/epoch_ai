# Plan: Smarter Oversampling — BorderlineSMOTE + ADASYN + Trajectory Augmentation

## Context
Current best mAP: 0.6999 (SMOTENC + boost-weak=3). Regularization experiments all performed worse. More features (CNN embeddings, curvature) didn't help. Bottleneck is data quality for weak classes (Cormorants=40, Ducks=58, Geese=83). SMOTENC generates synthetic samples uniformly across minority class distributions, but decision-boundary samples matter most for ranking.

## Baseline params (mAP 0.6999)
```python
lgb_params: n_estimators=2500, lr=0.03, num_leaves=63, min_child_samples=10,
            subsample=0.8, colsample_bytree=0.8
cb_params:  iterations=2500, lr=0.03, depth=8, l2_leaf_reg=3
boost-weak=3, oversampler=SMOTENC
```

## Experiments & Results

### Run 1: BorderlineSMOTE (replace SMOTENC)
- Only generates synthetic samples near decision boundaries
- Doesn't support categorical features — `radar_bird_size` treated as numeric
- **Result: mAP 0.7003** (+0.0004)

### Run 2: ADASYN (replace SMOTENC) ← BEST
- Generates more samples for harder-to-classify instances
- Adaptive density-based approach
- **Result: mAP 0.7041** (+0.0042) — new best
- Notable: Cormorants 0.4068 (+0.0495), Geese 0.6668 (+0.0386)

### Run 3: Trajectory Augmentation (replace boost-weak)
- Augments raw EWKB trajectories before feature extraction:
  - Spatial jitter: σ=0.00003° lon/lat (~3m), σ=2m alt, σ=0.3dB RCS
  - Time warp: ±5% scaling of trajectory_time
  - Rotation: random angle rotation of displacement vectors
- Creates genuinely novel trajectory-derived features (33 of 69)
- Non-trajectory features get light jitter (1% of value)
- **Result: mAP 0.7028** (+0.0029)
- Best Cormorants: 0.4208 (+0.0635)
- High fold variance (0.5547–0.8523) — needs tuning

### Run 4: ADASYN + Trajectory Augmentation combined
- Trajectory aug pre-feature-extraction + ADASYN in pipeline
- **Result: mAP 0.7000** (+0.0001) — no improvement, methods interfere
- Cormorants 0.4217, Geese 0.6336, Ducks 0.7348

### Run 5: ADASYN + Pseudo-Labeling (threshold=0.95)
- First round: standard ADASYN → mAP 0.7012
- 936/1872 test samples above 0.95 threshold added as pseudo-labels
- Pseudo-label distribution: 811 Gulls, 93 Songbirds, 14 BoP, 9 Geese, 6 Clutter, 3 Ducks
- **Result: mAP 0.7148** (+0.0149) — new best (but possibly inflated, see caveat)
- Notable: Waders 0.3811 (+0.0619), BoP 0.6271 (+0.0306), Ducks 0.7575
- ⚠️ Caveat: OOF mAP may be inflated — pseudo-labeled test samples in training folds can leak test-distribution info

### Run 6: Trajectory Augmentation with halved jitter (--jitter-scale 0.5)
- σ=0.000015° lon/lat, σ=1m alt, σ=0.15dB RCS
- **Result: mAP 0.7059** (+0.0031 vs original trajectory aug)
- Fold variance: 0.5598–0.8446 (vs 0.5547–0.8523 original) — slightly tighter
- Better overall than full jitter, Ducks 0.7531 best ever

### Summary Table

| Method | mAP | Cormorants | Geese | Ducks | Waders | BoP |
|--------|-----|------------|-------|-------|--------|-----|
| Baseline (SMOTENC) | 0.6999 | 0.3573 | 0.6282 | 0.7180 | — | — |
| BorderlineSMOTE | 0.7003 | 0.3812 | 0.6532 | 0.7419 | — | — |
| **ADASYN** | **0.7041** | 0.4068 | **0.6668** | 0.7395 | 0.3192 | 0.6013 |
| Trajectory Aug | 0.7028 | **0.4208** | 0.6309 | 0.7466 | — | — |
| ADASYN + Traj Aug | 0.7000 | 0.4217 | 0.6336 | 0.7348 | 0.3352 | 0.5762 |
| Traj Aug (0.5x jitter) | 0.7059 | 0.4179 | 0.6426 | **0.7531** | 0.3513 | 0.5913 |
| ADASYN + Pseudo-Label⚠️ | 0.7148⚠️ | 0.4027 | 0.6577 | 0.7575 | **0.3811** | **0.6271** |

## Next Steps
- [x] Try ADASYN + trajectory augmentation combined → no improvement
- [x] Pseudo-labeling → mAP 0.7148 but possibly inflated
- [x] Tune trajectory augmentation jitter params → 0.5x jitter slightly better
- [ ] Update submission.ipynb with best config (ADASYN)
- [ ] Try pseudo-labeling with lower threshold (0.90, 0.85) to add more weak-class samples
- [ ] Try pseudo-labeling combined with trajectory augmentation

## CLI Usage
```bash
# ADASYN (current best verified)
uv run solution.py --oversampler adasyn

# Trajectory augmentation
uv run solution.py --oversampler trajectory

# Trajectory augmentation with reduced jitter
uv run solution.py --oversampler trajectory --jitter-scale 0.5

# ADASYN + trajectory combined
uv run solution.py --oversampler adasyn+trajectory

# Pseudo-labeling
uv run solution.py --oversampler adasyn --pseudo-label

# BorderlineSMOTE
uv run solution.py --oversampler borderline
```
