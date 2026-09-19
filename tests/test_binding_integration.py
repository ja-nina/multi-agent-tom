import pytest

from personabind.binding.battery import run_battery

REAL_MODEL = "Qwen/Qwen3-4B"


@pytest.mark.integration
def test_real_battery_run_on_qwen3_4b(tmp_path):
    # Requires: `personabind build --variant t1` already run so data/t1_discrete.jsonl
    # exists, and network/cached weights for Qwen3-4B.
    config = {
        "seed": 20260910, "variants": ["t1_discrete"], "dataset_dir": "data/",
        "sample_size": 50, "train_fraction": 0.5, "layer_sweep": "all",
        "accuracy_floor": 0.90, "causal_clear_margin": 2.0,
        "mean_intervention_coefficients": [1.0], "output_dir": str(tmp_path),
        "dtype": "bfloat16",
    }
    result = run_battery(REAL_MODEL, config)
    print(f"T1 verdict on {REAL_MODEL}: {result['verdict']}")
    print(result["per_variant"]["t1_discrete"]["accuracy"])
    # No pass/fail assertion on the verdict itself -- that IS the research
    # question. The test's job is to prove the pipeline runs end-to-end on a
    # real model without raising, and to print the numbers for a human to read.
