Matrix summary (fresh run, fixed controls OFF)

| Run | mAP | Cormorants AP | Waders AP | interaction pack line |
|---|---:|---:|---:|---|
| A | 0.6316 | 0.2762 | 0.3283 | Metadata interaction pack disabled |
| B | 0.6294 | 0.2650 | 0.3380 | Metadata interaction pack [pack_v1] active features (3): ['tide_hour_sin', 'tide_hour_cos', 'rising_night_interaction'] |
| C | 0.6290 | 0.2657 | 0.3526 | Metadata interaction pack disabled |
| D | 0.6351 | 0.2497 | 0.3649 | Metadata interaction pack [pack_v1] active features (8): ['tide_hour_sin', 'tide_hour_cos', 'tide_delta_hour_sin', 'tide_delta_hour_cos', 'headwind_tide_delta', 'gustiness_tide_delta', 'rising_night_interaction', 'precip_tide_motion'] |

| Delta | ΔmAP | ΔCormorants AP | ΔWaders AP |
|---|---:|---:|---:|
| B-A | -0.0022 | -0.0112 | +0.0097 |
| C-A | -0.0026 | -0.0105 | +0.0243 |
| D-C | +0.0061 | -0.0160 | +0.0123 |
| D-B | +0.0057 | -0.0153 | +0.0269 |
| D-A | +0.0035 | -0.0265 | +0.0366 |
