#!/usr/bin/env zsh
set -euo pipefail

cd /Users/anshumalimehta/Projects/epoch_ai

seeds=(13 42 77 123 314)
results_file="seed_compare_results.tsv"

echo -e "seed\tvariant\tmap\tcormorants\twaders" > "$results_file"

run_variant() {
  local seed="$1"
  local variant="$2"
  local traj_on="$3"
  local mode="$4"
  local log_file="$5"

  perl -0pi -e "s/RANDOM_STATE = .*/RANDOM_STATE = ${seed}/; s/SEARCH_SEED = .*/SEARCH_SEED = ${seed}/; s/ENABLE_RCS_PACK_V1 = .*/ENABLE_RCS_PACK_V1 = False/; s/ENABLE_RCS_PACK_V2 = .*/ENABLE_RCS_PACK_V2 = False/; s/ENABLE_TRAJECTORY_PACK_V1 = .*/ENABLE_TRAJECTORY_PACK_V1 = ${traj_on}/; s/TRAJECTORY_PACK_V1_MODE = .*/TRAJECTORY_PACK_V1_MODE = '${mode}'  # options: full, no_turn, no_vertical, drop_vz_p90_abs, drop_climb_descent_ratio/" solution2.py

  .venv/bin/python solution2.py 2>&1 | tee "$log_file"

  local map
  local corm
  local waders

  map=$(rg "OOF Macro-Averaged AP" "$log_file" | awk -F': ' '{print $2}' | tail -1)
  corm=$(rg "^   Cormorants" "$log_file" | awk -F': ' '{print $2}' | tail -1)
  waders=$(rg "^   Waders" "$log_file" | awk -F': ' '{print $2}' | tail -1)

  echo -e "${seed}\t${variant}\t${map}\t${corm}\t${waders}" >> "$results_file"
}

for seed in "${seeds[@]}"; do
  run_variant "$seed" "baseline" "False" "full" "seed_${seed}_baseline.log"
  run_variant "$seed" "trajv1" "True" "full" "seed_${seed}_trajv1.log"
done

echo "----- RESULTS TSV -----"
cat "$results_file"

echo "----- SUMMARY -----"
awk 'BEGIN {n=0; wins=0; dm=0; dc=0; dw=0}
NR>1 {
  if ($2=="baseline") {bm[$1]=$3; bc[$1]=$4; bw[$1]=$5}
  else if ($2=="trajv1") {tm[$1]=$3; tc[$1]=$4; tw[$1]=$5}
}
END {
  for (s in bm) {
    n++
    dm += (tm[s]-bm[s])
    dc += (tc[s]-bc[s])
    dw += (tw[s]-bw[s])
    if (tm[s] > bm[s]) wins++
  }
  printf("mean_delta_map\t%.6f\n", dm/n)
  printf("mean_delta_cormorants\t%.6f\n", dc/n)
  printf("mean_delta_waders\t%.6f\n", dw/n)
  printf("seeds_won_on_map\t%d/%d\n", wins, n)
}' "$results_file"
