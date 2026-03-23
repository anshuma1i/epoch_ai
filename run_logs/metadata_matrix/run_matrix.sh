#!/usr/bin/env bash
set -euo pipefail
PY="/Users/anshumalimehta/Projects/epoch_ai/.venv/bin/python"
BASE="solution2.py"
OUTDIR="run_logs/metadata_matrix"
mkdir -p "$OUTDIR"

run_variant() {
  local label="$1"
  local tide="$2"
  local pack="$3"
  local tmp="$OUTDIR/${label}_tmp_solution2.py"
  local log="$OUTDIR/${label}.log"
  local sub="$OUTDIR/${label}_submission.csv"

  cp "$BASE" "$tmp"
  perl -0777 -i -pe "s/^DATASET_VARIANT\s*=\s*.*$/DATASET_VARIANT = 'openmeteo_tide'/m; s/^TIDE_ABLATION\s*=\s*.*$/TIDE_ABLATION = '$tide'/m; s/^ENABLE_METADATA_INTERACTION_PACK\s*=\s*.*$/ENABLE_METADATA_INTERACTION_PACK = $pack/m; s/^ENABLE_CONFUSION_RESOLVER\s*=\s*.*$/ENABLE_CONFUSION_RESOLVER = False/m; s/^USE_WEAK_SPECIALIST_BLEND\s*=\s*.*$/USE_WEAK_SPECIALIST_BLEND = False/m; s/^USE_OOF_WEAK_REWEIGHT\s*=\s*.*$/USE_OOF_WEAK_REWEIGHT = False/m; s|^SUBMISSION_OUT\s*=\s*.*$|SUBMISSION_OUT = '$sub'|m;" "$tmp"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START $label tide=$tide pack=$pack" | tee "$log"
  "$PY" "$tmp" >> "$log" 2>&1
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END $label" >> "$log"
  rm -f "$tmp"
}

run_variant "A_tide_level_rising_pack_off" "tide_level_rising" "False"
run_variant "B_tide_level_rising_pack_on" "tide_level_rising" "True"
run_variant "C_tide_all_pack_off" "tide_all" "False"
run_variant "D_tide_all_pack_on" "tide_all" "True"

echo "Matrix run complete"
