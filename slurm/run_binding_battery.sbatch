#!/bin/bash
#SBATCH --job-name=personabind-binding
#SBATCH --array=0-0%10
#SBATCH -p gpu-a100
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a100-03
#SBATCH --time=24:00:00
#SBATCH --output=slurm-logs/binding-%A_%a.out
#SBATCH --error=slurm-logs/binding-%A_%a.err

# Runs the Phase 1 binding battery for ONE model against the Phase 0 datasets
# already present under data/. Unlike slurm/build_t3b.sbatch, this loads the
# model directly in-process (no vLLM server, no port, no health check) --
# activation access needs the model object itself, not an HTTP API.
#
# Usage:
#   sbatch slurm/run_binding_battery.sbatch <model> [config]
#     <model>   qwen3-4b | qwen3-8b            (REQUIRED)
#     [config]  path to the binding config      (OPTIONAL -- this is $2;
#               default configs/binding.yaml)
#
# Qwen3-4B is the plan's "dev / fast iteration" model -- run it first. The
# kill-criteria gate (spec S8) means Qwen3-8B may not be worth running at all
# if Qwen3-4B's own T1 accuracy fails outright (re-run at 8B ONLY to rule out
# "model too small" before concluding anything, per the plan's own caveat).
#
# To skip T1/T2 once they've already cleared the gate for this model (check
# results/binding/<model>_verdict.json first) and go straight to T3a (with
# T3b run alongside, informationally -- it has no kill-criteria row, so it
# never affects the verdict either way):
#   sbatch slurm/run_binding_battery.sbatch qwen3-4b configs/binding_t3a.yaml
#
# 24h default: with T1/T2 skipped this covers roughly two of the original
# three variants' worth of work (T3a + T3b) at sample_size=1000 and a full
# layer sweep -- still an ESTIMATE, not a measured number (no real per-pass
# timing has been collected on this cluster yet). Retune after a first real
# run, same honesty as build_t3b.sbatch:
#   sbatch --time=36:00:00 slurm/run_binding_battery.sbatch qwen3-8b configs/binding_t3a.yaml

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm-logs results/binding

MODEL_KEY="${1:?Usage: sbatch slurm/run_binding_battery.sbatch <model: qwen3-4b|qwen3-8b> [config]}"
CONFIG="${2:-configs/binding.yaml}"
case "$MODEL_KEY" in
  qwen3-4b)  MODEL_REPO="Qwen/Qwen3-4B" ;;
  qwen3-8b)  MODEL_REPO="Qwen/Qwen3-8B" ;;
  *) echo "unknown model '$MODEL_KEY' (expected qwen3-4b or qwen3-8b)" >&2; exit 2 ;;
esac

PYTHON="${PERSONABIND_PYTHON:-$PWD/.venv/bin/python}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
# Python fully block-buffers stdout when it isn't a TTY (e.g. redirected to
# this job's .out file) -- without this, print()/tqdm output can sit in an
# internal buffer for a long time before ever reaching the file, making a
# job that's actually progressing look stuck to anyone tailing the log.
export PYTHONUNBUFFERED=1

echo "[info] node=$(hostname)  model=${MODEL_REPO}  config=${CONFIG}  python=${PYTHON}"
"$PYTHON" -u -m personabind.cli binding run --model "$MODEL_REPO" --config "$CONFIG"
