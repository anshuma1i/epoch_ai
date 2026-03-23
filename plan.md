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

### Summary Table

| Method | mAP | Cormorants | Geese | Ducks |
|--------|-----|------------|-------|-------|
| Baseline (SMOTENC) | 0.6999 | 0.3573 | 0.6282 | 0.7180 |
| BorderlineSMOTE | 0.7003 | 0.3812 | 0.6532 | 0.7419 |
| **ADASYN** | **0.7041** | 0.4068 | **0.6668** | 0.7395 |
| Trajectory Aug | 0.7028 | **0.4208** | 0.6309 | **0.7466** |

## Next Steps
- [ ] Try ADASYN + trajectory augmentation combined
- [ ] Pseudo-labeling: use high-confidence test predictions as additional training data
- [ ] Tune trajectory augmentation jitter params to reduce fold variance
- [ ] Update submission.ipynb with best config (ADASYN)

## CLI Usage
```bash
# ADASYN (current best)
.venv/bin/python -W ignore solution.py --oversampler adasyn

# Trajectory augmentation
.venv/bin/python -W ignore solution.py --oversampler trajectory

# BorderlineSMOTE
.venv/bin/python -W ignore solution.py --oversampler borderline
```
