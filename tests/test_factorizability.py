import torch

from personabind.binding import factorizability
from personabind.binding.factorizability import run_factorizability
from personabind.binding.positions import trait_of
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


def test_layer_type_is_read_from_handle_not_hardcoded():
    # Regression guard for the structural-hardcoding violation on the old
    # literal "full_attention": swap in a hybrid-model-style layer_types
    # list on a modified handle and confirm each result's layer_type tracks
    # the handle's own entry for that layer index, not a fixed string.
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    assert handle.num_layers == 2  # tiny-gpt2; both indices below must be valid
    hybrid_handle = handle.__class__(
        **{**handle.__dict__, "layer_types": ["full_attention", "linear_attention"]}
    )
    results = run_factorizability(hybrid_handle, [_pair()], layers=[0, 1], seed=1, config_hash="abc")
    layer0 = next(r for r in results if r.layer == 0)
    layer1 = next(r for r in results if r.layer == 1)
    assert layer0.layer_type == "full_attention"
    assert layer1.layer_type == "linear_attention"


def test_layer_type_falls_back_when_handle_has_no_layer_types():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    homogeneous_handle = handle.__class__(**{**handle.__dict__, "layer_types": None})
    results = run_factorizability(homogeneous_handle, [_pair()], layers=[0], seed=1, config_hash="abc")
    assert all(r.layer_type == "full_attention" for r in results)


def test_off_target_uses_other_agents_trait_not_stale_answer_field(monkeypatch):
    # Regression guard for the pre-flight-caught bug where off-target token
    # lookup used a stale `record.answer` (which only reflects the record's
    # OWN query_agent) instead of `trait_of(twin, other_agent)`.
    #
    # Note: this fixture's actual effect values can't discriminate the bug --
    # tiny-gpt2 (hidden_size=2) produces an effect_off_target of exactly 0.0
    # for this pair regardless of which gold token is targeted, so a test
    # comparing `effect_off_target` against an independently-recomputed
    # "expected" value would pass identically whether the correct or the
    # stale-field target token were used (verified empirically). So instead,
    # per this task's guidance, we directly spy on the internal
    # `_first_gold_token_id` call site and assert the actual gold string
    # passed for the off-target lookup, which is what line 71 controls.
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    base, twin = _pair()
    other_agent = "Charles"

    # Precondition: the fixture actually discriminates the two lookups --
    # trait_of(twin, other_agent) must differ from both the query agent's
    # own trait and from twin.answer (the stale field a regression would use).
    assert trait_of(twin, twin.query_agent) != trait_of(twin, other_agent)
    assert twin.answer != trait_of(twin, other_agent)

    golds_looked_up = []
    real_first_gold_token_id = factorizability._first_gold_token_id

    def spy(handle_arg, gold):
        golds_looked_up.append(gold)
        return real_first_gold_token_id(handle_arg, gold)

    monkeypatch.setattr(factorizability, "_first_gold_token_id", spy)

    run_factorizability(handle, [(base, twin)], layers=[0], seed=1, config_hash="abc")

    # The module must look up BOTH the query agent's own trait ("novice")
    # AND the other agent's trait ("expert"). A regression that swaps
    # trait_of(twin, other_agent) for the stale twin.answer would look up
    # "novice" twice and never "expert".
    assert trait_of(twin, other_agent) in golds_looked_up
    assert set(golds_looked_up) == {trait_of(twin, twin.query_agent), trait_of(twin, other_agent)}


def test_measurement_prompts_drop_the_stored_grammatical_article(monkeypatch):
    # Regression guard for the T1 article-measurement bias: the stored
    # answer_prefix ends in "is an"/"is a" -- an article that agrees with the
    # BASE record's own trait and sits AFTER every patch site, so no patch can
    # change it. Measuring the counterfactual trait's first-token logprob
    # against the wrong article floors the effect regardless of any real
    # binding. Both measurement prompts (on-target and off-target) must
    # therefore be built from a bare "{agent} is" prefix instead.
    #
    # tiny-gpt2's logprobs are meaningless here, so assert on the DECODED TEXT
    # of the ids actually handed to forward_logits -- a revert that fed
    # base.answer_prefix straight through would decode to "...Doug is an".
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    base, twin = _pair()

    # Precondition: the fixture really carries an article to strip.
    assert base.answer_prefix.endswith(" an")

    seen_texts = []
    real_forward_logits = factorizability.forward_logits

    def spy(handle_arg, input_ids, *args, **kwargs):
        seen_texts.append(handle._tokenizer.decode(input_ids[0].tolist()))
        return real_forward_logits(handle_arg, input_ids, *args, **kwargs)

    monkeypatch.setattr(factorizability, "forward_logits", spy)

    run_factorizability(handle, [(base, twin)], layers=[0], seed=1, config_hash="abc")

    assert len(seen_texts) == 2  # one clean pass on-target, one off-target
    on_target_text, off_target_text = seen_texts
    assert on_target_text.endswith("Doug is")
    assert off_target_text.endswith("Charles is")
    for text in seen_texts:
        assert not text.endswith(" is an")
        assert not text.endswith(" is a")


def test_off_target_reuses_base_pos_without_reresolving_against_other_base(monkeypatch):
    # Regression guard for the second corrected bug: re-resolving the patch
    # position against other_base_tok instead of reusing base_pos directly.
    # Spies on stored_position as called from inside factorizability.py --
    # it must be called exactly once for base (with base_tok) and once for
    # twin (with twin_tok) per layer, and never again for other_base/its
    # tokenization. A regression that re-resolves the position would add a
    # third call using other_base's own tokenized prompt.
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    base, twin = _pair()

    calls = []
    real_stored_position = factorizability.stored_position

    def spy(tokenized, record, agent):
        calls.append(record)
        return real_stored_position(tokenized, record, agent)

    monkeypatch.setattr(factorizability, "stored_position", spy)

    run_factorizability(handle, [(base, twin)], layers=[0], seed=1, config_hash="abc")

    assert len(calls) == 2
    assert {id(r) for r in calls} == {id(base), id(twin)}
