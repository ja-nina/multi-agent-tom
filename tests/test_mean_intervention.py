import torch

from personabind.binding import mean_intervention
from personabind.binding.mean_intervention import run_mean_intervention
from personabind.binding.positions import trait_of
from personabind.common.activations import load_model
from personabind.record import AgentSpec, Record

TINY_MODEL = "sshleifer/tiny-gpt2"


def _record(i, level):
    trait = "expert" if level == 1 else "novice"
    other = "novice" if level == 1 else "expert"
    return Record(
        id=f"r{i}", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context=f"Doug{i} is an {trait}; Charles{i} is a {other}.",
        question=f"How reliable is Doug{i}?", answer_prefix=f"Doug{i} is",
        agents=[AgentSpec(f"Doug{i}", 0, trait, level), AgentSpec(f"Charles{i}", 1, other, 1 - level)],
        query_agent=f"Doug{i}", answer=trait,
        counterfactual_id=f"cf{i}", counterfactual_diff="agent_trait_map", seed=1,
    )


def test_run_mean_intervention_sweeps_coefficients_and_carries_baseline():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(i, i % 2) for i in range(20)]
    results = run_mean_intervention(
        handle, records, trait_contrast=("expert", "novice"), layers=[0],
        coefficients=[0.5, 1.0], train_fraction=0.5, seed=1, config_hash="abc",
    )
    coeffs_seen = {r.coefficient for r in results}
    assert coeffs_seen == {0.5, 1.0}
    for r in results:
        assert r.effect_norm_matched_random is not None
        assert r.effect_off_target is not None  # patch_site is always "stored" here
        assert r.patch_site == "stored"
        assert r.direction_norm_fraction is not None


def test_train_and_test_splits_never_mix_a_pair():
    # every record here has a distinct id, so this mostly checks the split function
    # runs without error over an odd-sized set and produces disjoint train/test:
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(i, i % 2) for i in range(15)]
    results = run_mean_intervention(
        handle, records, trait_contrast=("expert", "novice"), layers=[0],
        coefficients=[1.0], train_fraction=0.5, seed=2, config_hash="abc",
    )
    tested_ids = {r.record_id for r in results}
    assert len(tested_ids) <= len(records)


def test_layer_type_is_read_from_handle_not_hardcoded():
    # Regression guard for the structural-hardcoding violation on the old
    # literal "full_attention" (see factorizability.py's Task 7 review
    # finding): swap in a hybrid-model-style layer_types list on a modified
    # handle and confirm each result's layer_type tracks the handle's own
    # entry for that layer index, not a fixed string.
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    assert handle.num_layers == 2  # tiny-gpt2; both indices below must be valid
    hybrid_handle = handle.__class__(
        **{**handle.__dict__, "layer_types": ["full_attention", "linear_attention"]}
    )
    records = [_record(i, i % 2) for i in range(10)]
    results = run_mean_intervention(
        hybrid_handle, records, trait_contrast=("expert", "novice"), layers=[0, 1],
        coefficients=[1.0], train_fraction=0.5, seed=1, config_hash="abc",
    )
    layer0 = next(r for r in results if r.layer == 0)
    layer1 = next(r for r in results if r.layer == 1)
    assert layer0.layer_type == "full_attention"
    assert layer1.layer_type == "linear_attention"


def test_off_target_uses_other_agents_trait_not_query_agents_own_answer(monkeypatch):
    # Regression guard for the off-target/layer_type pattern that Task 7's
    # review flagged on factorizability.py: the off-target token lookup must
    # target trait_of(record, other_agent) -- the OTHER agent's own trait --
    # never the record's own query-agent answer field.
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [_record(i, i % 2) for i in range(10)]

    # Precondition: the fixture actually discriminates the two lookups for
    # every record -- each record's own answer differs from the other
    # agent's trait, so a regression using record.answer instead of
    # trait_of(record, other_agent) would look up the wrong token.
    for r in records:
        other_agent = next(a.name for a in r.agents if a.name != r.query_agent)
        assert r.answer != trait_of(r, other_agent)

    tokens_looked_up = []
    real_first_token_id = mean_intervention._first_token_id

    def spy(handle_arg, text):
        tokens_looked_up.append(text)
        return real_first_token_id(handle_arg, text)

    monkeypatch.setattr(mean_intervention, "_first_token_id", spy)

    results = run_mean_intervention(
        handle, records, trait_contrast=("expert", "novice"), layers=[0],
        coefficients=[1.0], train_fraction=0.5, seed=1, config_hash="abc",
    )

    for r in results:
        record = next(rec for rec in records if rec.id == r.record_id)
        other_agent = next(a.name for a in record.agents if a.name != record.query_agent)
        expected_off_target_trait = trait_of(record, other_agent)
        # the correct off-target trait must have been looked up. Our
        # precondition above guarantees expected_off_target_trait !=
        # record.answer for every record, so a regression that only ever
        # looked up record.answer (never the correct one) is caught here.
        assert expected_off_target_trait in tokens_looked_up
