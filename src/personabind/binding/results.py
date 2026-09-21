from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

_VALID_PATCH_SITES = ("stored", "retrieved")


@dataclass(frozen=True)
class AccuracyResult:
    model: str
    variant: str
    record_id: str
    predicted: str
    gold: str
    correct: bool
    seed: int
    # Purely informational: what the model would say if actually left to
    # generate freely (never used for scoring -- see accuracy.py's module
    # docstring for why free generation is unreliable for that). Lets a human
    # sanity-check the forced-choice verdict above against what the model
    # would actually have said unprompted.
    sample_completion: str = ""
    # The exact text fed to the model (context + question + answer_prefix,
    # verbatim) -- lets a human see exactly what the model saw when judging
    # whether `predicted`/`sample_completion` make sense.
    prompt: str = ""


@dataclass(frozen=True)
class InterventionResult:
    test: str
    record_id: str
    model: str
    variant: str
    layer: int
    layer_type: str
    patch_site: str
    token_positions: dict
    effect_on_target: float
    effect_norm_matched_random: float
    effect_off_target: float | None
    coefficient: float
    direction_norm_fraction: float | None
    train_test_split: str
    seed: int
    config_hash: str

    def __post_init__(self) -> None:
        if self.patch_site not in _VALID_PATCH_SITES:
            raise ValueError(f"{self.record_id}: unknown patch_site {self.patch_site!r}")
        if self.patch_site == "stored" and self.effect_off_target is None:
            raise ValueError(f"{self.record_id}: patch_site='stored' requires effect_off_target, got None")


@dataclass(frozen=True)
class PositionGeneralizationResult:
    model: str
    variant: str
    layer: int
    trait_contrast: str
    fit_position: int
    same_position_accuracy: float
    cross_position_accuracy: float
    position_invariance_ratio: float
    shuffled_label_control_accuracy: float
    n_train: int
    n_test: int
    seed: int
    config_hash: str
    test: str = "position_test"


def write_jsonl(results: list, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.writelines(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in results)


def append_jsonl(result, fh) -> None:
    """Append ONE result as a single JSONL line to an ALREADY-OPEN file
    handle, flushing immediately so the write is visible right away to
    anything reading the file concurrently (e.g. `tail -f` on a live cluster
    job's output). This is the streaming counterpart to `write_jsonl`: the
    battery calls this once per result, as each one is computed, instead of
    accumulating a whole test's results in memory and writing them all in
    one batch at the very end.

    `flush()` alone only pushes Python's internal buffer into the OS's page
    cache -- on a networked filesystem (e.g. NFS-mounted cluster scratch),
    a DIFFERENT process reading the same file (like a `tail -f`) can still
    see nothing until that data actually reaches the filesystem, which is
    exactly what `fsync` forces."""
    fh.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    fh.flush()
    os.fsync(fh.fileno())
