#!/bin/bash
# Warm the Hugging Face caches so the gpu17 jobs (which have no outbound
# internet) can run fully offline.
#
# Run this ONCE on a LOGIN NODE (it needs internet; it does NOT need a GPU):
#     bash slurm/prefetch_data.sh
#     bash slurm/prefetch_data.sh qwen3-8b          # also pull model weights
#     bash slurm/prefetch_data.sh qwen3.5-9b
#
# It populates:
#   - the datasets cache (allenai/sciq, cais/mmlu, allenai/ai2_arc) by
#     running a real `personabind build --variant t3a`, which calls
#     load_bank() exactly the way the T3b job will;
#   - optionally the model weights for the vLLM env, if you pass a model key.
#
# Everything lands under $HF_HOME (default ./.hf_cache), which the sbatch
# scripts read with HF_HUB_OFFLINE=1.

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL_KEY="${1:-}"

PYTHON="${PERSONABIND_PYTHON:-$PWD/.venv/bin/python}"
VLLM_PYTHON="${VLLM_PYTHON:-/BS/conformal-circuits/work/mamba/envs/vllm/bin/python}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export HF_HUB_OFFLINE=0
mkdir -p "$HF_HOME" data

echo "[1/2] warming the QA-bank datasets cache via a real T3a build ..."
# t3a needs no GPU and no server; it exercises the same load_bank() path.
rm -f data/t3a_inferred_templated.jsonl
"$PYTHON" -m personabind.cli build --variant t3a --config configs/generator.yaml
"$PYTHON" -m personabind.cli report --dataset data/t3a_inferred_templated.jsonl || true
echo "    datasets cached under $HF_HOME"

if [ -z "$MODEL_KEY" ]; then
    echo "[2/2] skipped model-weight prefetch (no model key given)."
    echo "done. Datasets are cached; pass qwen3-8b / qwen3.5-9b to also cache weights."
    exit 0
fi

case "$MODEL_KEY" in
  qwen3-8b)    MODEL_REPO="Qwen/Qwen3-8B"   ;;
  qwen3.5-9b)  MODEL_REPO="Qwen/Qwen3.5-9B" ;;
  *) echo "unknown model '$MODEL_KEY' (expected qwen3-8b or qwen3.5-9b)" >&2; exit 2 ;;
esac

echo "[2/2] downloading weights for ${MODEL_REPO} into ${HF_HOME} ..."
"$VLLM_PYTHON" - "$MODEL_REPO" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
p = snapshot_download(repo_id=repo, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"])
print(f"cached {repo} -> {p}")
PY
echo "done."
