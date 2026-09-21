import json

import pytest

from personabind.binding.verdict4 import run_test4_only, write_verdict4
from tests.test_battery_run import TINY_MODEL, _write_t3a_dataset


def _config(dataset_dir, output_dir, **overrides):
    cfg = {
        "seed": 1, "dataset_dir": str(dataset_dir), "sample_size": 3,
        "train_fraction": 0.5, "layer_sweep": [0], "causal_clear_margin": 2.0,
        "mean_intervention_coefficients": [1.0], "output_dir": str(output_dir),
        "dtype": "float32",
    }
    cfg.update(overrides)
    return cfg


def test_run_test4_only_never_runs_accuracy_or_factorizability(tmp_path):
    """Regression guard for the whole point of this module: unlike
    run_battery, no accuracy/factorizability/position_test file should ever
    be produced -- only mean_intervention and verdict4."""
    dataset_dir = _write_t3a_dataset(tmp_path)
    output_dir = tmp_path / "results"
    config = _config(dataset_dir, output_dir)

    result = run_test4_only(TINY_MODEL, config, "t3a_inferred_templated")

    prefix = f"{TINY_MODEL.replace('/', '_')}__t3a_inferred_templated__"
    assert not (output_dir / f"{prefix}accuracy.jsonl").exists()
    assert not (output_dir / f"{prefix}factorizability.jsonl").exists()
    assert not (output_dir / f"{prefix}position_test.jsonl").exists()
    assert (output_dir / f"{prefix}mean_intervention.jsonl").exists()
    assert result["verdict_path"] == str(output_dir / f"{prefix}verdict4.json")


def test_run_test4_only_streams_mean_intervention_rows(tmp_path):
    dataset_dir = _write_t3a_dataset(tmp_path)
    output_dir = tmp_path / "results"
    config = _config(dataset_dir, output_dir)

    run_test4_only(TINY_MODEL, config, "t3a_inferred_templated")

    path = output_dir / f"{TINY_MODEL.replace('/', '_')}__t3a_inferred_templated__mean_intervention.jsonl"
    with open(path, encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh]
    assert len(lines) > 0
    assert all(row["test"] == "mean_intervention" for row in lines)


def test_run_test4_only_raises_on_a_variant_with_no_trait_contrast(tmp_path):
    dataset_dir = _write_t3a_dataset(tmp_path)
    output_dir = tmp_path / "results"
    config = _config(dataset_dir, output_dir)

    with pytest.raises(ValueError, match="no trait_contrast"):
        run_test4_only(TINY_MODEL, config, "not_a_real_variant")


def test_run_test4_only_verdict_reflects_clears_baseline(tmp_path, monkeypatch):
    """The verdict must be driven ONLY by clears_baseline on test 4's own
    effects -- not by any accuracy number (there is none here)."""
    dataset_dir = _write_t3a_dataset(tmp_path)
    output_dir = tmp_path / "results"
    config = _config(dataset_dir, output_dir)

    import personabind.binding.verdict4 as verdict4_mod

    monkeypatch.setattr(verdict4_mod, "clears_baseline", lambda effects, margin: True)
    result = run_test4_only(TINY_MODEL, config, "t3a_inferred_templated")
    assert result["passed"] is True
    with open(result["verdict_path"], encoding="utf-8") as fh:
        verdict = json.load(fh)
    assert verdict["passed"] is True
    assert verdict["test"] == "mean_intervention"


def test_write_verdict4_serializes_sigma_per_layer(tmp_path):
    path = write_verdict4(
        "Qwen/Qwen3-4B", "t3a_inferred_templated", str(tmp_path), True,
        {5: (0.4, 0.1), 6: (0.0, 0.0)}, 2.0, "abc123",
    )
    with open(path, encoding="utf-8") as fh:
        verdict = json.load(fh)
    assert verdict["model"] == "Qwen/Qwen3-4B"
    assert verdict["passed"] is True
    assert verdict["effects_by_layer"]["5"]["sigma"] == pytest.approx(4.0)
    assert verdict["effects_by_layer"]["6"]["sigma"] is None  # se=0 -- must not divide by zero
