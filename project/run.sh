#!/bin/bash
set -e

CONFIG_FILE="config.ini"
SESSION="nas_runs"

PERTURBATIONS=("brightness_contrast" "motion_blur" "gaussian_noise")
SEEDS=(42 42 42)

# Safety check
if [ ${#PERTURBATIONS[@]} -ne ${#SEEDS[@]} ]; then
  echo "Error: PERTURBATIONS and SEEDS must have same length"
  exit 1
fi

# Recreate session cleanly if it already exists
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Killing it first."
  tmux kill-session -t "$SESSION"
fi

# Create one tmux session with one shell
tmux new-session -d -s "$SESSION"

for i in "${!PERTURBATIONS[@]}"; do
  P=${PERTURBATIONS[$i]}
  S=${SEEDS[$i]}

  RUN_ID="${P}_seed_${S}"

  CMD="python -m model_box.NAS.main \
    --config $CONFIG_FILE \
    --seed $S \
    --perturbation_type $P \
    --run_id $RUN_ID"

  echo "Queueing: $RUN_ID"

  # Queue commands into the same tmux shell so they run one by one
  tmux send-keys -t "$SESSION":0 "$CMD" C-m
done

echo "All experiments queued sequentially in tmux session '$SESSION'."
echo "Attach with: tmux attach -t $SESSION"