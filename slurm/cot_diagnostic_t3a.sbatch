#!/bin/bash
#SBATCH --job-name=personabind-cot-diagnostic
#SBATCH --array=0-0%10
#SBATCH -p gpu-a100
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --time=04:00:00
#SBATCH --output=slurm-logs/cot-diagnostic-%A_%a.out
#SBATCH --error=slurm-logs/cot-diagnostic-%A_%a.err

# Self-contained T3a CoT diagnostic: this job brings up a local vLLM
# OpenAI-compatible server on the allocated node, waits for it to become
# healthy, runs `personabind binding cot-diagnostic` against it (letting the
# model reason via Qwen3's native thinking mode before giving its A/B
# verdict on T3a records), and tears the server down again on exit. Same
# self-contained shape as slurm/build_t3b.sbatch, adapted for this job.
#
# INFORMATIONAL ONLY -- this never feeds into or replaces the official T3a
# accuracy number (personabind binding run) or the battery's verdict. It's a
# standalone secondary check of whether letting the model reason changes
# T3a's picture versus the forced immediate-choice measurement. Run
# `slurm/run_binding_battery.sbatch` for the actual gate.
#
# Usage (model is REQUIRED -- $1 below, a positional arg, NOT an sbatch flag;
# the job fails immediately with a usage message if omitted):
#   sbatch slurm/cot_diagnostic_t3a.sbatch <model> [config] [max_tokens]
#
#   <model>       qwen3-4b | qwen3-8b        (REQUIRED)
#                 maps to HF repo Qwen/Qwen3-4B / Qwen/Qwen3-8B -- the two
#                 models actually in Phase 1's gate (configs/binding.yaml).
#                 Qwen3.5-9B is deliberately NOT an option here: that's the
#                 Phase 1b hybrid arm, gated on Phase 1 clearing on 8B first.
#   [config]      path to the binding config, reused only for its
#                 seed/sample_size/output_dir (OPTIONAL -- this is $2;
#                 default configs/binding.yaml)
#   [max_tokens]  CoT completion budget per record (OPTIONAL -- this is $3;
#                 default 1024 -- a real multi-hop CoT trace on T3a's
#                 transcripts can run long, so this is deliberately much
#                 bigger than sample_free_completion's informational sample)
#
# Example:
#   sbatch slurm/cot_diagnostic_t3a.sbatch qwen3-4b
#   sbatch slurm/cot_diagnostic_t3a.sbatch qwen3-8b configs/binding.yaml 2048
#
# PREREQUISITES (see slurm/README.md):
#   1. personabind installed:      uv sync           (creates ./.venv)
#   2. a SEPARATE env with vLLM:   see $VLLM_PYTHON below
#   3. openai package installed in personabind's OWN env (./.venv) -- this
#      is the client this job's job step uses to talk to the vLLM server;
#      it is NOT the same thing as vLLM itself, which stays in its own env.
#   4. data/t3a_inferred_templated.jsonl already built (personabind build
#      --variant t3a), same dataset the real battery reads.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm-logs results/binding

MODEL_KEY="${1:?Usage: sbatch slurm/cot_diagnostic_t3a.sbatch <model: qwen3-4b|qwen3-8b> [config] [max_tokens]}"
CONFIG="${2:-configs/binding.yaml}"
MAX_TOKENS="${3:-1024}"
DATASET="${DATASET:-data/t3a_inferred_templated.jsonl}"

case "$MODEL_KEY" in
  qwen3-4b)  MODEL_REPO="Qwen/Qwen3-4B" ;;
  qwen3-8b)  MODEL_REPO="Qwen/Qwen3-8B" ;;
  *) echo "unknown model '$MODEL_KEY' (expected qwen3-4b or qwen3-8b)" >&2; exit 2 ;;
esac

# This cluster has no `module` command -- point these directly at the right
# interpreters. Override either at submit time, e.g.:
#   PERSONABIND_PYTHON=/path/to/env/bin/python VLLM_PYTHON=/path/to/vllm-env/bin/python \
#     sbatch slurm/cot_diagnostic_t3a.sbatch qwen3-4b
#
# personabind's own env (created by `uv sync`):
PYTHON="${PERSONABIND_PYTHON:-$PWD/.venv/bin/python}"
# a SEPARATE env that has vLLM installed (do NOT add vllm to personabind's
# pyproject -- same reasoning as build_t3b.sbatch):
VLLM_PYTHON="${VLLM_PYTHON:-/BS/conformal-circuits/work/mamba/envs/ma-testbed/bin/python}"

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

# FlashInfer's top-k/top-p sampler JIT-compiles a CUDA kernel via nvcc on
# first use -- same nvcc-missing failure mode as build_t3b.sbatch. Default to
# the built-in PyTorch sampler; this job is low-throughput (sequential
# records), so the speed difference is moot.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

if [ ! -f "$DATASET" ]; then
    echo "[fatal] $DATASET does not exist -- build it first: personabind build --variant t3a" >&2
    exit 1
fi

# One vLLM server per node; derive a per-job port so two jobs on one node
# never collide.
PORT=$(( 8000 + (SLURM_JOB_ID % 1000) ))
BASE_URL="http://127.0.0.1:${PORT}/v1"
VLLM_LOG="slurm-logs/vllm-cot-${SLURM_JOB_ID}.log"

echo "[info] node=$(hostname)  model=${MODEL_REPO}  port=${PORT}  dataset=${DATASET}  max_tokens=${MAX_TOKENS}"
echo "[info] personabind python: ${PYTHON}"
echo "[info] vLLM python:        ${VLLM_PYTHON}"

# --- launch vLLM in the background, guarantee teardown -------------------------
"$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_REPO" \
    --served-model-name "$MODEL_REPO" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --trust-remote-code \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
trap 'echo "[info] stopping vLLM (pid $VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true' EXIT

# --- wait for health (model load can take several minutes cold) ---------------
echo "[info] waiting for vLLM /health on ${BASE_URL%/v1}/health ..."
READY=0
for i in $(seq 1 120); do          # 120 * 15s = 30 min ceiling
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[fatal] vLLM process died during startup. Tail of ${VLLM_LOG}:" >&2
        tail -n 40 "$VLLM_LOG" >&2
        exit 1
    fi
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        READY=1; break
    fi
    sleep 15
done
[ "$READY" = 1 ] || { echo "[fatal] vLLM did not become healthy within 30 min" >&2; tail -n 40 "$VLLM_LOG" >&2; exit 1; }
echo "[info] vLLM is healthy after ~$(( i * 15 ))s"

# --- run the diagnostic --------------------------------------------------------
echo "[run] personabind binding cot-diagnostic --model ${MODEL_REPO} --base-url ${BASE_URL} --dataset ${DATASET} --config ${CONFIG} --max-tokens ${MAX_TOKENS}"
"$PYTHON" -m personabind.cli binding cot-diagnostic \
    --model "$MODEL_REPO" \
    --base-url "$BASE_URL" \
    --dataset "$DATASET" \
    --config "$CONFIG" \
    --max-tokens "$MAX_TOKENS"

echo "[done] results streamed to results/binding/${MODEL_REPO//\//_}__t3a_cot_diagnostic.jsonl"
