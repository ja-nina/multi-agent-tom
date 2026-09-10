# SLURM scripts

Batch jobs for the one part of Phase 0 that needs a GPU: **T3b** dataset
generation, which drives a local **vLLM** OpenAI-compatible server.
T1 / T2 / T3a are pure CPU/data work — build those with plain
`personabind build --variant t1|t2|t3a` on any node.

Style matches `../../PersVecGen/slurm/*.sbatch`: partition `gpu17`,
`-c 10`, one GPU, the same flaky-node `--exclude` list, `set -euo pipefail`,
and interpreters pinned via `PYTHON=` env vars because this cluster has no
`module` command.

| script | what it does |
|---|---|
| `build_t3b.sbatch` | **one-shot.** Boots vLLM on the node, waits for `/health`, runs `personabind build --variant t3b` + `report`, tears vLLM down. |
| `vllm_serve.sbatch` | **long-lived server.** Holds a vLLM server for its walltime, writes `slurm/vllm.endpoint`; you run the client yourself against it. |
| `prefetch_data.sh` | **login node, once.** Warms the HF datasets cache (and optionally model weights) so the GPU jobs run offline. |

## One-time setup

1. **personabind env** (this repo uses `uv`):
   ```bash
   uv sync            # creates ./.venv ; nothing here needs torch/transformers
   ```
2. **A separate vLLM env.** Do **not** add `vllm` to `pyproject.toml` —
   Phase 0 must stay torch/transformers-free. Create it wherever you keep
   envs, e.g.:
   ```bash
   mamba create -n vllm python=3.11 -y
   mamba run -n vllm pip install "vllm>=0.11"      # needs Qwen3-Next support for qwen3.5-9b
   ```
   Then point the scripts at it: `export VLLM_PYTHON=/path/to/mamba/envs/vllm/bin/python`
   (the default in the scripts is `/BS/conformal-circuits/work/mamba/envs/vllm/bin/python`).
3. **Warm the caches on a login node** (gpu17 nodes have no outbound internet):
   ```bash
   bash slurm/prefetch_data.sh              # datasets: allenai/sciq, cais/mmlu, allenai/ai2_arc
   bash slurm/prefetch_data.sh qwen3-8b     # + weights
   bash slurm/prefetch_data.sh qwen3.5-9b   # + weights
   ```
   Everything lands in `$HF_HOME` (default `./.hf_cache`); the jobs read it
   with `HF_HUB_OFFLINE=1`.

## Running it

Spec §5.4 wants T3b built **once per model**, into its own file, then
compared. `build_t3b.sbatch` writes `data/t3b_inferred_llm.jsonl` and
**refuses to overwrite** an existing one, so rename between runs:

```bash
sbatch slurm/build_t3b.sbatch qwen3-8b
#   ... when it finishes:
mv data/t3b_inferred_llm.jsonl data/t3b_qwen3-8b.jsonl

sbatch slurm/build_t3b.sbatch qwen3.5-9b
mv data/t3b_inferred_llm.jsonl data/t3b_qwen3.5-9b.jsonl
```

`t3b.max_tokens`, sampling, and `sizes.t3b_inferred_llm` come from
`configs/generator.yaml`. The script overrides only `t3b.models` (to the one
model you named) and `t3b.base_url` (to the job's local server) in a patched
copy under `$TMPDIR` — no per-model config file needed.

Override partition / walltime at submit time; sbatch flags beat the
`#SBATCH` lines:

```bash
sbatch --partition=gpu22 --time=12:00:00 slurm/build_t3b.sbatch qwen3-8b
```

## Notes

- **Sequential client.** `personabind` calls the backend one turn at a time
  in a Python loop, so throughput is round-trip-latency bound, not
  GPU-bound — the GPU mostly idles. 2000 records × ~2 turns × up to 5
  attempts is a few hours. The 8 h default has headroom; retune after a
  first run. A future speedup would be to parallelise the client, not to
  give vLLM more GPU.
- **Reproducibility.** T3b is byte-stable only via its on-disk generation
  cache (`data/.cache/t3b/`, keyed by model + a content hash of
  question/gold/distractor + correctness + style + seed + attempt). Keep
  that directory between runs; deleting it forces full regeneration.
  `data/t3b_generation_report.json` records per-model reject rates.
- **Qwen3.5-9B caveat (spec §5.1).** `model_type: qwen3_next` — hybrid
  Gated DeltaNet + full attention. Confirm your vLLM build loads it
  *before* submitting:
  ```bash
  $VLLM_PYTHON -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3.5-9B --trust-remote-code --max-model-len 4096 --port 8999
  # ctrl-C once it prints "Uvicorn running"
  ```
  If it won't load, that arm is a bonus contribution — drop it, keep
  qwen3-8b. If it loads but CUDA-graph capture is flaky on the linear
  layers, add `--enforce-eager` to the vLLM args in the sbatch script.
- **`enable_thinking=False`** is sent by the client per request
  (`chat_template_kwargs`), so the server needs no special flag.
- Request logging is left at the vLLM default. To quiet it, add
  `--no-enable-log-requests` (newer vLLM) or `--disable-log-requests`
  (older) to the args in the sbatch script.
- Logs: `slurm-logs/t3b-<jobid>_0.out` (job) and
  `slurm-logs/vllm-<jobid>.log` (server). `sinfo` / `squeue -u $USER` as usual.
