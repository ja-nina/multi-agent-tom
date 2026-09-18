import json

import pytest

from personabind.binding.results import (
    AccuracyResult,
    InterventionResult,
    PositionGeneralizationResult,
    write_jsonl,
)


def _intervention_kwargs(**overrides):
    kwargs = {
        "test": "factorizability", "record_id": "t1_000001", "model": "Qwen/Qwen3-4B",
        "variant": "t1_discrete", "layer": 10, "layer_type": "full_attention",
        "patch_site": "stored",
        "token_positions": {"patched": 5, "read_on_target": 20, "read_off_target": 22},
        "effect_on_target": 0.4, "effect_norm_matched_random": 0.02, "effect_off_target": 0.01,
        "coefficient": 1.0, "direction_norm_fraction": None, "train_test_split": "n/a",
        "seed": 1, "config_hash": "abc",
    }
    kwargs.update(overrides)
    return kwargs


def test_stored_site_requires_off_target():
    with pytest.raises(ValueError):
        InterventionResult(**_intervention_kwargs(effect_off_target=None))


def test_retrieved_site_allows_none_off_target():
    r = InterventionResult(**_intervention_kwargs(patch_site="retrieved", effect_off_target=None))
    assert r.effect_off_target is None


def test_unknown_patch_site_rejected():
    with pytest.raises(ValueError):
        InterventionResult(**_intervention_kwargs(patch_site="bogus"))


def test_write_jsonl_round_trips(tmp_path):
    results = [
        AccuracyResult(model="m", variant="t1_discrete", record_id="r1", predicted="expert", gold="expert", correct=True, seed=1),
        InterventionResult(**_intervention_kwargs()),
    ]
    path = str(tmp_path / "out.jsonl")
    write_jsonl(results, path)
    write_jsonl(results, path)  # append-only: calling twice must not overwrite
    with open(path, encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh]
    assert len(lines) == 4
    assert lines[0]["correct"] is True
    assert lines[1]["patch_site"] == "stored"


def test_position_generalization_result_requires_shuffled_control():
    with pytest.raises(TypeError):
        PositionGeneralizationResult(
            model="m", variant="t1_discrete", layer=5, trait_contrast="expert_vs_novice",
            fit_position=0, same_position_accuracy=0.9, cross_position_accuracy=0.85,
            position_invariance_ratio=0.94, n_train=100, n_test=100, seed=1, config_hash="abc",
            # shuffled_label_control_accuracy omitted -- must be a required error, not silently None
        )
