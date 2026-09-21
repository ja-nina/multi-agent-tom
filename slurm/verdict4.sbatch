#!/bin/bash
#SBATCH --job-name=personabind-verdict4
#SBATCH --array=0-0%10
#SBATCH -p gpu-a100
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --time=08:00:00
#SBATCH --output=slurm-logs/verdict4-%A_%a.out
#SBATCH --error=slurm-logs/verdict4-%A_%a.err

# Runs ONLY test 4 (mean intervention) for one model x variant, skipping
# accuracy/factorizability/position_test entirely, and writes a standalone
# verdict4.json gated only on test 4's own causal effect clearing
# causal_clear_margin -- never on accuracy, and never the same thing as the
# full battery's per-variant verdict (slurm/run_binding_battery.sbatch),
# which also requires the accuracy floor and factorizability. Loads the
# model directly in-process, same as run_binding_battery.sbatch -- no vLLM
# server, no port, no health check.
#
# Usage:
#   sbatch slurm/verdict4.sbatch <model> <variant> [config]
#     <model>    qwen3-4b | qwen3-8b                     (REQUIRED)
#     <variant>  e.g. t3a_inferred_templated              (REQUIRED --
#                must have a defined trait_contrast; see
#                generator.traits.trait_contrast_for_variant)
#     [config]   path to the binding config                (OPTIONAL --
#                this is $3; default configs/binding.yaml -- reused for
#                seed/sample_size/train_fraction/layer_sweep/
#                mean_intervention_coefficients/causal_clear_margin/
#                dataset_dir/output_dir, exactly as the full battery uses it)
#
# Example:
#   sbatch slurm/verdict4.sbatch qwen3-4b t3a_inferred_templated
#   sbatch slurm/verdict4.sbatch qwen3-8b t3a_inferred_templated configs/binding_t3a.yaml

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm-logs results/binding

MODEL_KEY="${1:?Usage: sbatch slurm/verdict4.sbatch <model: qwen3-4b|qwen3-8b> <variant> [config]}"
VARIANT="${2:?Usage: sbatch slurm/verdict4.sbatch <model: qwen3-4b|qwen3-8b> <variant> [config]}"
CONFIG="${3:-configs/binding.yaml}"
case "$MODEL_KEY" in
  qwen3-4b)  MODEL_REPO="Qwen/Qwen3-4B" ;;
  qwen3-8b)  MODEL_REPO="Qwen/Qwen3-8B" ;;
  *) echo "unknown model '$MODEL_KEY' (expected qwen3-4b or qwen3-8b)" >&2; exit 2 ;;
esac

PYTHON="${PERSONABIND_PYTHON:-$PWD/.venv/bin/python}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
# Same buffering fix as run_binding_battery.sbatch: without this, tqdm/print
# output can sit unflushed for a long time in a job's .out file, making a
# job that's actually progressing look stuck to anyone tailing the log.
export PYTHONUNBUFFERED=1

echo "[info] node=$(hostname)  model=${MODEL_REPO}  variant=${VARIANT}  config=${CONFIG}  python=${PYTHON}"
"$PYTHON" -u -m personabind.cli binding verdict4 --model "$MODEL_REPO" --variant "$VARIANT" --config "$CONFIG"
