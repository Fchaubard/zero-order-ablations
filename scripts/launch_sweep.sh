#!/usr/bin/env bash
# Start the full sweep in a detached tmux session named "sweep", logging to runs/sweep.log.
#   bash scripts/launch_sweep.sh [extra zoo.sweep args, e.g. --tasks mlp_mnist --budget 600]
# WANDB_API_KEY is taken from the environment, or from ~/.wandb_key if present.
# Attach with:  tmux attach -t sweep      Detach again with Ctrl-b d.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd)
PY=${PYTHON:-python}
SESSION=${SESSION:-sweep}

if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$HOME/.wandb_key" ]; then
  export WANDB_API_KEY="$(tr -d '[:space:]' < "$HOME/.wandb_key")"
fi
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "warning: WANDB_API_KEY not set and ~/.wandb_key missing -> runs will log wandb offline" >&2
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found: run scripts/setup_remote.sh first" >&2; exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION" >&2; exit 1
fi

if [ -z "${GPUS:-}" ]; then
  GPUS=$($PY -c "import torch; print(','.join(str(i) for i in range(torch.cuda.device_count())) or 'cpu')")
fi
mkdir -p runs results

# Quote the extra arguments so they survive the trip through tmux's shell.
EXTRA=""
for a in "$@"; do EXTRA="$EXTRA $(printf '%q' "$a")"; done

# The inner shell re-reads ~/.wandb_key so the key reaches the runs even when an existing
# tmux server does not inherit this environment; the key is never written to a file.
INNER="cd $(printf '%q' "$ROOT"); \
if [ -z \"\${WANDB_API_KEY:-}\" ] && [ -f \"\$HOME/.wandb_key\" ]; then export WANDB_API_KEY=\"\$(tr -d '[:space:]' < \"\$HOME/.wandb_key\")\"; fi; \
export PYTHONUNBUFFERED=1; \
$PY -m zoo.sweep --gpus $GPUS --tasks all --opts all --lrs 0.1,0.01,0.001 --budget 3600 --eval-every 300 \
  --out results --logdir runs --wandb-project zoo-survey$EXTRA 2>&1 | tee -a runs/sweep.log; \
echo; echo '[sweep finished - press enter to close]'; read -r _"

# If a tmux server is already running, push the key into its global environment so the
# new session inherits it; if none is running the server we spawn inherits ours.
if [ -n "${WANDB_API_KEY:-}" ]; then
  tmux set-environment -g WANDB_API_KEY "$WANDB_API_KEY" 2>/dev/null || true
fi
tmux new-session -d -s "$SESSION" -x 200 -y 50 "bash -c $(printf '%q' "$INNER")"
echo "sweep started in tmux session '$SESSION' on GPUs [$GPUS]; log: runs/sweep.log"
echo "  attach:  tmux attach -t $SESSION      status:  $PY -m zoo.status --out results --grid"
