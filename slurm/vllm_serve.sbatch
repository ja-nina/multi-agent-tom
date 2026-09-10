#!/bin/bash
#SBATCH --job-name=personabind-vllm
#SBATCH --array=0-0%10
#SBATCH -p gpu17
#SBATCH -c 10
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu22-a40-05,gpu22-a40-06,gpu17-l40-07,gpu17-l40-20
#SBATCH --time=12:00:00
#SBATCH --output=slurm-logs/vllm-%A_%a.out
#SBATCH --error=slurm-logs/vllm-%A_%a.err

# Stand up a LONG-LIVED vLLM OpenAI-compatible server on one gpu17 node and
# hold it for the job's walltime. Use this when you want to run
# `personabind build --variant t3b` yourself -- from a login node, an
# interactive shell, or a second job -- against a server that stays up,
# rather than the one-shot slurm/build_t3b.sbatch.
#
# On startup it writes the reachable endpoint to  slurm/vllm.<jobid>.endpoint
# (and, for convenience, to a stable  slurm/vllm.endpoint  symlink) as:
#     http://<node-hostname>:<port>/v1
# Point your build at it, e.g. from the repo root on a login node:
#     BASE_URL=$(sed 's:/v1::' slurm/vllm.endpoint)
#     .venv/bin/python - <<PY
#     import yaml; c=yaml.safe_load(open("configs/generator.yaml"))
#     c["t3b"]["base_url"]="$(cat slurm/vllm.endpoint)"
#     c["t3b"]["models"]=["Qwen/Qwen3-8B"]         # or Qwen/Qwen3.5-9B
#     yaml.safe_dump(c, open("configs/generator.job.yaml","w"), sort_keys=False)
#     PY
#     .venv/bin/python -m personabind.cli build --variant t3b --config configs/generator.job.yaml
#
# NOTE: this binds to 0.0.0.0 so other cluster hosts can reach it. Only do
# that on a trusted internal network. slurm/build_t3b.sbatch binds to
# 127.0.0.1 instead because the client runs in the same job.
#
# Usage:
#   sbatch slurm/vllm_serve.sbatch <model> [port] [max_model_len]
#     <model>          qwen3-8b | qwen3.5-9b        (REQUIRED)
#     [port]           default 8000
#     [max_model_len]  default 8192
#
# Override partition/time at submit time as usual:
#   sbatch --partition=gpu22 --time=24:00:00 slurm/vllm_serve.sbatch qwen3-8b
#
# Stop it early with:  scancel <jobid>
#
# See slurm/README.md for the vLLM env and the Qwen3.5-9B caveat.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p slurm-logs slurm

MODEL_KEY="${1:?Usage: sbatch slurm/vllm_serve.sbatch <model: qwen3-8b|qwen3.5-9b> [port] [max_model_len]}"
PORT="${2:-8000}"
MAX_LEN="${3:-8192}"

case "$MODEL_KEY" in
  qwen3-8b)    MODEL_REPO="Qwen/Qwen3-8B"   ;;
  qwen3.5-9b)  MODEL_REPO="Qwen/Qwen3.5-9B" ;;
  *) echo "unknown model '$MODEL_KEY' (expected qwen3-8b or qwen3.5-9b)" >&2; exit 2 ;;
esac

# Separate env that has vLLM (see slurm/README.md). Override:
#   VLLM_PYTHON=/path/to/vllm-env/bin/python sbatch slurm/vllm_serve.sbatch qwen3-8b
VLLM_PYTHON="${VLLM_PYTHON:-/BS/conformal-circuits/work/mamba/envs/vllm/bin/python}"

export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"     # set 1 only if gpu17 is air-gapped (prefetch first)
export TOKENIZERS_PARALLELISM=false

NODE_HOST="$(hostname -f 2>/dev/null || hostname)"
ENDPOINT="http://${NODE_HOST}:${PORT}/v1"
ENDPOINT_FILE="slurm/vllm.${SLURM_JOB_ID}.endpoint"
echo "$ENDPOINT" > "$ENDPOINT_FILE"
ln -sfn "vllm.${SLURM_JOB_ID}.endpoint" "slurm/vllm.endpoint"
trap 'rm -f "$ENDPOINT_FILE"; [ "$(readlink slurm/vllm.endpoint 2>/dev/null)" = "vllm.${SLURM_JOB_ID}.endpoint" ] && rm -f slurm/vllm.endpoint || true' EXIT

echo "[info] node=${NODE_HOST}  model=${MODEL_REPO}  endpoint=${ENDPOINT}"
echo "[info] wrote ${ENDPOINT_FILE} (and slurm/vllm.endpoint symlink)"

# For hybrid Qwen3.5-9B add  --enforce-eager  if CUDA-graph capture misbehaves.
# Quieter logs (optional): newer vLLM uses  --no-enable-log-requests , older
# builds used  --disable-log-requests .
exec "$VLLM_PYTHON" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_REPO" \
    --served-model-name "$MODEL_REPO" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization 0.90 \
    --max-model-len "$MAX_LEN" \
    --dtype bfloat16 \
    --trust-remote-code
