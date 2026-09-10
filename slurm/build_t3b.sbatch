#!/bin/bash
#SBATCH --job-name=personabind-t3b
#SBATCH --array=0-0%10
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --time=08:00:00
#SBATCH --output=slurm-logs/t3b-%A_%a.out
#SBATCH --error=slurm-logs/t3b-%A_%a.err

# Self-contained T3b build: this job brings up a local vLLM OpenAI-compatible
# server on the allocated gpu17 node, waits for it to become healthy, runs
# `personabind build --variant t3b` against it, runs `personabind report` on
# the result, and tears the server down again on exit. Nothing else needs the
# server -- T1/T2/T3a builds are pure CPU/data work and never touch it.
#
# Usage (model is REQUIRED -- $1 below, a positional arg, NOT an sbatch flag;
# the job fails immediately with a usage message if omitted):
#   sbatch slurm/build_t3b.sbatch <model> [config]
#
#   <model>   qwen3-8b | qwen3.5-9b        (REQUIRED)
#             maps to HF repo Qwen/Qwen3-8B / Qwen/Qwen3.5-9B
#   [config]  path to the generator config (OPTIONAL -- this is $2;
#             default configs/generator.yaml). The script writes a patched
#             copy into $TMPDIR with t3b.models forced to the single chosen
#             model and t3b.base_url pointed at this job's local server, so
#             you do NOT need a per-model config file checked in.
#
# Spec 5.4: build T3b once per model into its own file and compare later.
# So submit this twice, once per model -- each run overwrites
# data/t3b_inferred_llm.jsonl, so rename the output between runs:
#   sbatch slurm/build_t3b.sbatch qwen3-8b
#   # when it finishes:  mv data/t3b_inferred_llm.jsonl data/t3b_qwen3-8b.jsonl
#   sbatch slurm/build_t3b.sbatch qwen3.5-9b
#   # when it finishes:  mv data/t3b_inferred_llm.jsonl data/t3b_qwen3.5-9b.jsonl
#
# Defaults to partition gpu17 / 10 CPUs / 8h (the #SBATCH lines above).
# Override at submit time if gpu17 is busy -- sbatch command-line flags win
# over the #SBATCH lines in this script:
#   sbatch --partition=gpu22 --time=12:00:00 slurm/build_t3b.sbatch qwen3-8b
# Check `sinfo` to see what's free.
#
# PREREQUISITES (see slurm/README.md):
#   1. personabind installed:      uv sync           (creates ./.venv)
#   2. a SEPARATE env with vLLM:   see $VLLM_PYTHON below
#   3. HF cache: by default the job downloads any missing datasets
#      (allenai/sciq, cais/mmlu, allenai/ai2_arc) and model weights into
#      $HF_HOME at run time. If your gpu17 nodes are air-gapped, run
#      `bash slurm/prefetch_data.sh <model>` on a login node first and
#      submit with  HF_HUB_OFFLINE=1 .
#
# Qwen3.5-9B CAVEAT (spec 5.1): its model_type is qwen3_next (hybrid Gated
# DeltaNet + full attention). It needs a recent vLLM that ships
# Qwen3-Next support. Verify BEFORE submitting the array:
#   $VLLM_PYTHON -c "import vllm; print(vllm.__version__)"
#   $VLLM_PYTHON -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-9B \
#     --trust-remote-code --max-model-len 4096 --port 8999   # ctrl-C once it says "Uvicorn running"
# If it will not load, that arm is a bonus contribution, not a dependency --
# drop it and keep qwen3-8b.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm-logs data

MODEL_KEY="${1:?Usage: sbatch slurm/build_t3b.sbatch <model: qwen3-8b|qwen3.5-9b> [config]}"
BASE_CONFIG="${2:-configs/generator.yaml}"

case "$MODEL_KEY" in
  qwen3-8b)    MODEL_REPO="Qwen/Qwen3-8B"   ;;
  qwen3.5-9b)  MODEL_REPO="Qwen/Qwen3.5-9B" ;;
  *) echo "unknown model '$MODEL_KEY' (expected qwen3-8b or qwen3.5-9b)" >&2; exit 2 ;;
esac

# This cluster has no `module` command -- point these directly at the right
# interpreters. Override either at submit time, e.g.:
#   PERSONABIND_PYTHON=/path/to/env/bin/python VLLM_PYTHON=/path/to/vllm-env/bin/python \
#     sbatch slurm/build_t3b.sbatch qwen3-8b
#
# personabind's own env (created by `uv sync`):
PYTHON="${PERSONABIND_PYTHON:-$PWD/.venv/bin/python}"
# a SEPARATE env that has vLLM installed (do NOT add vllm to personabind's
# pyproject -- Phase 0 must stay torch/transformers-free):
VLLM_PYTHON="${VLLM_PYTHON:-/BS/conformal-circuits/work/mamba/envs/vllm/bin/python}"

# Shared HF cache. By default the job is allowed to download missing
# datasets/weights into $HF_HOME at run time (the PersVecGen jobs on this
# cluster do the same -- compute nodes have outbound internet). If YOUR
# gpu17 nodes are air-gapped, run `bash slurm/prefetch_data.sh <model>` on a
# login node first, then submit with  HF_HUB_OFFLINE=1  to fail loud instead
# of hanging on a blocked connection.
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export TOKENIZERS_PARALLELISM=false

OUT="data/t3b_inferred_llm.jsonl"
if [ -f "$OUT" ]; then
    echo "[skip] $OUT already exists -- rename or delete it to rebuild. Exiting."
    exit 0
fi

# One vLLM server per node; derive a per-job port so two jobs on one node
# (unlikely with --gres=gpu:1, but cheap insurance) never collide.
PORT=$(( 8000 + (SLURM_JOB_ID % 1000) ))
BASE_URL="http://127.0.0.1:${PORT}/v1"
VLLM_LOG="slurm-logs/vllm-${SLURM_JOB_ID}.log"

echo "[info] node=$(hostname)  model=${MODEL_REPO}  port=${PORT}"
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

# For the hybrid Qwen3.5-9B, if CUDA-graph capture is flaky on the Gated
# DeltaNet layers, add   --enforce-eager   to the args above.
# Quieter server logs: newer vLLM uses  --no-enable-log-requests  (older
# builds used  --disable-log-requests ). Optional either way.

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

# --- patch the config: single model + this job's local endpoint --------------
JOB_CONFIG="${TMPDIR:-/tmp}/generator.${SLURM_JOB_ID}.yaml"
"$PYTHON" - "$BASE_CONFIG" "$JOB_CONFIG" "$MODEL_REPO" "$BASE_URL" <<'PY'
import sys, yaml
base, out, repo, base_url = sys.argv[1:5]
with open(base) as fh:
    cfg = yaml.safe_load(fh)
cfg.setdefault("t3b", {})
cfg["t3b"]["models"] = [repo]
cfg["t3b"]["base_url"] = base_url
with open(out, "w") as fh:
    yaml.safe_dump(cfg, fh, sort_keys=False)
print(f"[info] wrote patched config -> {out}  (t3b.models=[{repo}], base_url={base_url})")
PY

# --- build T3b, then report --------------------------------------------------
echo "[run] personabind build --variant t3b --config ${JOB_CONFIG}"
"$PYTHON" -m personabind.cli build --variant t3b --config "$JOB_CONFIG"

echo "[run] personabind report --dataset ${OUT}"
"$PYTHON" -m personabind.cli report --dataset "$OUT" || {
    echo "[warn] report exited non-zero -- a confound threshold was violated; inspect the table above." >&2
}

echo "[done] ${OUT} written; generation report at data/t3b_generation_report.json"
