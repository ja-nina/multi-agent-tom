from __future__ import annotations

import argparse
import json
import os
import sys

from personabind.config import load_config
from personabind.generator.build import build_stated, write_jsonl
from personabind.generator.qa_bank import load_bank
from personabind.generator.transcripts import build_t3a, build_t3b
from personabind.generator.vllm_backend import VLLMBackend
from personabind.stats.report import evaluate_dataset, render_table

_FILENAMES = {
    "t1": ("t1_discrete", "t1_discrete.jsonl"),
    "t2": ("t2_graded", "t2_graded.jsonl"),
    "t3a": ("t3a_inferred_templated", "t3a_inferred_templated.jsonl"),
    "t3b": ("t3b_inferred_llm", "t3b_inferred_llm.jsonl"),
}


def _cache_dir(cfg) -> str:
    return os.path.join(cfg.output_dir, ".cache")


def _build_one(key: str, cfg) -> None:
    variant, fname = _FILENAMES[key]
    out = os.path.join(cfg.output_dir, fname)
    if os.path.exists(out):
        os.remove(out)
    if key in ("t1", "t2"):
        write_jsonl(build_stated(variant, cfg), out)
        return
    bank = load_bank(cfg.qa_sources, cfg.domains,
                     os.path.join(_cache_dir(cfg), "hf"))
    if key == "t3a":
        write_jsonl(build_t3a(cfg, bank), out)
        return
    # t3b
    t3b = cfg.t3b

    def backend_factory(model: str):
        return VLLMBackend(
            base_url=t3b["base_url"], model=model, sampling=t3b["sampling"],
            max_tokens=int(t3b["max_tokens"]),
            cache_dir=os.path.join(_cache_dir(cfg), "t3b"), seed=int(t3b["seed"]),
        )

    records, gen_report = build_t3b(cfg, bank, backend_factory)
    write_jsonl(records, out)
    with open(os.path.join(cfg.output_dir, "t3b_generation_report.json"), "w", encoding="utf-8") as fh:
        json.dump(gen_report, fh, indent=2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="personabind")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--variant", choices=["all", "t1", "t2", "t3a", "t3b"], default="all")
    b.add_argument("--config", default="configs/generator.yaml")

    r = sub.add_parser("report")
    g = r.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset")
    g.add_argument("--all", action="store_true")
    r.add_argument("--config", default="configs/generator.yaml")

    bind = sub.add_parser("binding")
    bind_sub = bind.add_subparsers(dest="binding_cmd", required=True)
    bind_run = bind_sub.add_parser("run")
    bind_run.add_argument("--model", required=True)
    bind_run.add_argument("--config", default="configs/binding.yaml")

    args = parser.parse_args(argv)

    if args.cmd == "build":
        cfg = load_config(args.config)
        os.makedirs(cfg.output_dir, exist_ok=True)
        keys = ["t1", "t2", "t3a", "t3b"] if args.variant == "all" else [args.variant]
        for k in keys:
            _build_one(k, cfg)
            print(f"built {k} -> {cfg.output_dir}")
        return 0

    if args.cmd == "binding":
        import yaml

        from personabind.binding.battery import run_battery

        with open(args.config, encoding="utf-8") as fh:
            binding_config = yaml.safe_load(fh)
        if args.binding_cmd == "run":
            result = run_battery(args.model, binding_config)
            print(f"verdict: {result['verdict']} -> {result['verdict_path']}")
            return 0

    # report
    if args.all:
        cfg = load_config(args.config)
        paths = [os.path.join(cfg.output_dir, fn) for _, fn in _FILENAMES.values()]
        paths = [p for p in paths if os.path.exists(p)]
    else:
        paths = [args.dataset]
    rc = 0
    for p in paths:
        result = evaluate_dataset(p)
        print(render_table(result))
        print()
        if result["violations"]:
            rc = 1
    return rc


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
