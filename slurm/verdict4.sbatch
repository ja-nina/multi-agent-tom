#!/bin/bash
#SBATCH --job-name=personabind-verdict4
#SBATCH --array=0-1
#SBATCH -p gpu-a100
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --time=08:00:00
#SBATCH --output=slurm-logs/verdict4-%A_%a.out
#SBATCH --error=slurm-logs/verdict4-%A_%a.err

# Runs ONLY test 4 (mean intervention) for one model, as TWO separate array
# tasks -- task 0 on a STATED-trait variant (t1_discrete/t2_graded), task 1
# on an INFERRED-trait variant (t3a_inferred_templated) -- so one `sbatch`
# submission gives you both branches of the stated-vs-inferred comparison,
# each as its own SLURM task with its own GPU allocation, log file, verdict,
# and saved activation vectors. Skips accuracy/factorizability/position_test
# entirely; writes a standalone verdict4.json gated only on test 4's own
# causal effect clearing causal_clear_margin -- never the same thing as the
# full battery's per-variant verdict (slurm/run_binding_battery.sbatch),
# which also requires the accuracy floor and factorizability. Loads the
# model directly in-process, same as run_binding_battery.sbatch -- no vLLM
# server, no port, no health check.
#
# Usage:
#   sbatch slurm/verdict4.sbatch <model> <stated_variant> <inferred_variant> [config]
#     <model>            qwen3-4b | qwen3-8b   (REQUIRED)
#     <stated_variant>   e.g. t1_discrete | t2_graded   (REQUIRED --
#                        runs as array task 0)
#     <inferred_variant> e.g. t3a_inferred_templated    (REQUIRED --
#                        runs as array task 1)
#                        Both variants must have a defined trait_contrast;
#                        see generator.traits.trait_contrast_for_variant.
#     [config]           path to the binding config       (OPTIONAL --
#                        this is $4; default configs/binding.yaml -- reused
#                        for seed/sample_size/train_fraction/layer_sweep/
#                        mean_intervention_coefficients/causal_clear_margin/
#                        dataset_dir/output_dir/vectors_dir, exactly as the
#                        full battery uses it)
#
# Example (stated branch = t1_discrete, inferred branch = t3a):
#   sbatch slurm/verdict4.sbatch qwen3-4b t1_discrete t3a_inferred_templated
#   sbatch slurm/verdict4.sbatch qwen3-8b t2_graded t3a_inferred_templated configs/binding.yaml
#
# To run only one branch (skip the array), pass --array=0 or --array=1 at
# submit time, e.g.:
#   sbatch --array=1 slurm/verdict4.sbatch qwen3-4b t1_discrete t3a_inferred_templated

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm-logs results/binding

MODEL_KEY="${1:?Usage: sbatch slurm/verdict4.sbatch <model: qwen3-4b|qwen3-8b> <stated_variant> <inferred_variant> [config]}"
STATED_VARIANT="${2:?Usage: sbatch slurm/verdict4.sbatch <model: qwen3-4b|qwen3-8b> <stated_variant> <inferred_variant> [config]}"
INFERRED_VARIANT="${3:?Usage: sbatch slurm/verdict4.sbatch <model: qwen3-4b|qwen3-8b> <stated_variant> <inferred_variant> [config]}"
CONFIG="${4:-configs/binding.yaml}"
case "$MODEL_KEY" in
  qwen3-4b)  MODEL_REPO="Qwen/Qwen3-4B" ;;
  qwen3-8b)  MODEL_REPO="Qwen/Qwen3-8B" ;;
  *) echo "unknown model '$MODEL_KEY' (expected qwen3-4b or qwen3-8b)" >&2; exit 2 ;;
esac

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) VARIANT="$STATED_VARIANT";   BRANCH="stated" ;;
  1) VARIANT="$INFERRED_VARIANT"; BRANCH="inferred" ;;
  *) echo "unexpected SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} (expected 0 or 1)" >&2; exit 2 ;;
esac

PYTHON="${PERSONABIND_PYTHON:-$PWD/.venv/bin/python}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false
# Same buffering fix as run_binding_battery.sbatch: without this, tqdm/print
# output can sit unflushed for a long time in a job's .out file, making a
# job that's actually progressing look stuck to anyone tailing the log.
export PYTHONUNBUFFERED=1

echo "[info] node=$(hostname)  model=${MODEL_REPO}  branch=${BRANCH}  variant=${VARIANT}  array_task=${SLURM_ARRAY_TASK_ID:-0}  config=${CONFIG}  python=${PYTHON}"
"$PYTHON" -u -m personabind.cli binding verdict4 --model "$MODEL_REPO" --variant "$VARIANT" --config "$CONFIG"
