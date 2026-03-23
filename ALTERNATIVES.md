# Alternatives

## Baseline promotion decision (2026-03-23)

Chosen baseline:
- E0
- DATASET_VARIANT = openmeteo_tide
- TIDE_ABLATION = tide_all
- ENABLE_METADATA_INTERACTION_PACK = True
- METADATA_INTERACTION_DROP = ['rising_night_interaction']
- ENABLE_CONFUSION_RESOLVER = False
- USE_WEAK_SPECIALIST_BLEND = False
- USE_OOF_WEAK_REWEIGHT = False

Alternatives considered and not chosen:
- A baseline (tide_level_rising + pack OFF): lower multi-seed mean mAP
- D0 full pack (includes rising_night_interaction): same score as D3 in grouped ablation but less minimal
- E1/E2/E3/E4 (single/pair drops from E0): all reduced mAP versus E0 in validation block
