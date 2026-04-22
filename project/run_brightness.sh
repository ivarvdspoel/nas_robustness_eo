#!/bin/bash
set -e

CONFIG_FILE="config.ini"
SESSION="nas_runs"

PERTURBATION="brightness_contrast"
SEED=41
SEVERITIES=(1 2 3 4 5)

# Kill existing session if needed
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already exists. Killing it first."
  tmux kill-session -t "$SESSION"
fi

# Create new tmux session
tmux new-session -d -s "$SESSION"

# Create second pane (so we have 2 workers)
tmux split-window -t "$SESSION":0 -h

# Optional: even layout
tmux select-layout -t "$SESSION":0 even-horizontal

echo "Starting runs (2 in parallel)..."

for i in "${!SEVERITIES[@]}"; do
  SEV=${SEVERITIES[$i]}
  RUN_ID="${PERTURBATION}_sev_${SEV}_seed_${SEED}"

  CMD="python -m model_box.NAS.main \
    --config $CONFIG_FILE \
    --seed $SEED \
    --perturbation_type $PERTURBATION \
    --severity $SEV \
    --run_id $RUN_ID"

  PANE=$((i % 2))  # alternate between pane 0 and 1

  echo "Queueing: $RUN_ID -> pane $PANE"

  tmux send-keys -t "$SESSION":0.$PANE "$CMD" C-m

  # Small delay helps avoid command overlap
  sleep 1
done

echo "All experiments launched (2 concurrent workers)."
echo "Attach with: tmux attach -t $SESSION"