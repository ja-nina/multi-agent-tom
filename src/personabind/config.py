from __future__ import annotations

import hashlib
from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class GeneratorConfig:
    seed: int
    output_dir: str
    domains: list[str]
    qa_sources: list[str]
    sizes: dict[str, int]
    name_style_ratio: dict[str, float]
    format_ratio: dict[str, float]
    t3: dict
    t3b: dict


def load_config(path: str) -> GeneratorConfig:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return GeneratorConfig(
        seed=int(raw["seed"]),
        output_dir=raw["output_dir"],
        domains=list(raw["domains"]),
        qa_sources=list(raw["qa_sources"]),
        sizes={k: int(v) for k, v in raw["sizes"].items()},
        name_style_ratio={k: float(v) for k, v in raw["name_style_ratio"].items()},
        format_ratio={k: float(v) for k, v in raw["format_ratio"].items()},
        t3=dict(raw.get("t3", {})),
        t3b=dict(raw.get("t3b", {})),
    )


def derive_seed(master: int, *parts: str | int) -> int:
    key = "|".join(str(p) for p in (master, *parts)).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") & (2**63 - 1)
