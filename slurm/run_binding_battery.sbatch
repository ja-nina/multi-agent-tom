#!/bin/bash
#SBATCH --job-name=personabind-binding
#SBATCH --array=0-0%10
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/binding-%A_%a.out
#SBATCH --error=slurm-logs/binding-%A_%a.err

# Runs the Phase 1 binding battery for ONE model against the Phase 0 datasets
# already present under data/. Unlike slurm/build_t3b.sbatch, this loads the
# model directly in-process (no vLLM server, no port, no health check) --
# activation access needs the model object itself, not an HTTP API.
#
# Usage:
#   sbatch slurm/run_binding_battery.sbatch <model>
#     <model>   qwen3-4b | qwen3-8b   (REQUIRED)
#
# Qwen3-4B is the plan's "dev / fast iteration" model -- run it first. The
# kill-criteria gate (spec S8) means Qwen3-8B may not be worth running at all
# if Qwen3-4B's own T1 accuracy fails outright (re-run at 8B ONLY to rule out
# "model too small" before concluding anything, per the plan's own caveat).
#
# 4h default is a guess for Qwen3-4B's sample_size=1000, full layer sweep,
# T1 only, plus a config-permitting T2/T3a if the gate lets them run --
# retune after a first real run, same honesty as build_t3b.sbatch.
#   sbatch --time=12:00:00 slurm/run_binding_battery.sbatch qwen3-8b

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm-logs results/binding

MODEL_KEY="${1:?Usage: sbatch slurm/run_binding_battery.sbatch <model: qwen3-4b|qwen3-8b>}"
case "$MODEL_KEY" in
  qwen3-4b)  MODEL_REPO="Qwen/Qwen3-4B" ;;
  qwen3-8b)  MODEL_REPO="Qwen/Qwen3-8B" ;;
  *) echo "unknown model '$MODEL_KEY' (expected qwen3-4b or qwen3-8b)" >&2; exit 2 ;;
esac

PYTHON="${PERSONABIND_PYTHON:-$PWD/.venv/bin/python}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

echo "[info] node=$(hostname)  model=${MODEL_REPO}  python=${PYTHON}"
"$PYTHON" -m personabind.cli binding run --model "$MODEL_REPO" --config configs/binding.yaml
