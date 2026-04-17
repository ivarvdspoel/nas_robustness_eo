#!/bin/bash
set -e

CONFIG_FILE="config.ini"
SESSION="nas_runs"

PERTURBATIONS=("brightness_contrast" "gaussian_noise" "motion_blur")
SEEDS=(41 42 43)
#PERTURBATIONS="brightness_contrast"
#SEEDS=41

# Safety check
if [ ${#PERTURBATIONS[@]} -ne ${#SEEDS[@]} ]; then
  echo "Error: PERTURBATIONS and SEEDS must have same length"
  exit 1
fi

tmux new-session -d -s $SESSION

for i in "${!PERTURBATIONS[@]}"; do
  P=${PERTURBATIONS[$i]}
  S=${SEEDS[$i]}

  RUN_ID="${P}_seed_${S}"

  CMD="python -m model_box.NAS.main \
    --config $CONFIG_FILE \
    --seed $S \
    --perturbation_type $P \
    --run_id $RUN_ID"

  echo "Launching: $RUN_ID"

  tmux new-window -t $SESSION -n "$RUN_ID" "$CMD"
done

echo "Done. Attach with: tmux attach -t $SESSION"