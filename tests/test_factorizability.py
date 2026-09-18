import torch

from personabind.binding.factorizability import run_factorizability
from personabind.binding.results import InterventionResult
from personabind.common.activations import load_model
from personabind.record import AgentSpec, Record

TINY_MODEL = "sshleifer/tiny-gpt2"


def _pair():
    base = Record(
        id="t1_1", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context="Doug is an expert; Charles is a novice.",
        question="How reliable is Doug?", answer_prefix="Doug is an",
        agents=[AgentSpec("Doug", 0, "expert", 1), AgentSpec("Charles", 1, "novice", 0)],
        query_agent="Doug", answer="expert",
        counterfactual_id="t1_2", counterfactual_diff="agent_trait_map", seed=1,
    )
    twin = Record(
        id="t1_2", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context="Doug is a novice; Charles is an expert.",
        question="How reliable is Doug?", answer_prefix="Doug is a",
        agents=[AgentSpec("Doug", 0, "novice", 0), AgentSpec("Charles", 1, "expert", 1)],
        query_agent="Doug", answer="novice",
        counterfactual_id="t1_1", counterfactual_diff="agent_trait_map", seed=1,
    )
    return base, twin


def test_run_factorizability_produces_both_patch_sites_per_layer():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    results = run_factorizability(handle, [_pair()], layers=[0], seed=1, config_hash="abc")
    sites = {r.patch_site for r in results}
    assert sites == {"stored", "retrieved"}


def test_stored_site_has_off_target_retrieved_site_does_not():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    results = run_factorizability(handle, [_pair()], layers=[0], seed=1, config_hash="abc")
    stored = next(r for r in results if r.patch_site == "stored")
    retrieved = next(r for r in results if r.patch_site == "retrieved")
    assert stored.effect_off_target is not None
    assert retrieved.effect_off_target is None


def test_every_result_carries_a_baseline():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    results = run_factorizability(handle, [_pair()], layers=[0], seed=1, config_hash="abc")
    for r in results:
        assert isinstance(r, InterventionResult)
        assert r.effect_norm_matched_random is not None


def test_results_are_deterministic_given_seed():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    r1 = run_factorizability(handle, [_pair()], layers=[0], seed=42, config_hash="abc")
    r2 = run_factorizability(handle, [_pair()], layers=[0], seed=42, config_hash="abc")
    assert [r.effect_norm_matched_random for r in r1] == [r.effect_norm_matched_random for r in r2]
