import torch

from personabind.binding.position_test import _classify, _fit_diff_means, run_position_test
from personabind.common.activations import load_model
from personabind.record import AgentSpec, Record

TINY_MODEL = "sshleifer/tiny-gpt2"


def _record(position: int, level: int, name_a="Doug", name_b="Charles"):
    names = [name_a, name_b] if position == 0 else [name_b, name_a]
    query = names[position]
    trait = "expert" if level == 1 else "novice"
    other_trait = "novice" if level == 1 else "expert"
    context = f"{names[0]} is an {trait if position == 0 else other_trait}; {names[1]} is a {other_trait if position == 0 else trait}."
    return Record(
        id=f"r_{position}_{level}_{name_a}", variant="t1_discrete", format="same_sentence", domain="science",
        name_style="personal", context=context, question=f"How reliable is {query}?",
        answer_prefix=f"{query} is", agents=[
            AgentSpec(names[0], 0, trait if position == 0 else other_trait, level if position == 0 else 1 - level),
            AgentSpec(names[1], 1, other_trait if position == 0 else trait, 1 - level if position == 0 else level),
        ],
        query_agent=query, answer=trait,
        counterfactual_id="cf", counterfactual_diff="agent_trait_map", seed=1,
    )


def test_fit_and_classify_with_a_planted_position_invariant_direction():
    torch.manual_seed(0)
    high = torch.randn(8) + torch.tensor([5.0] + [0.0] * 7)
    low = torch.randn(8) - torch.tensor([5.0] + [0.0] * 7)
    train_high = [high + 0.1 * torch.randn(8) for _ in range(20)]
    train_low = [low + 0.1 * torch.randn(8) for _ in range(20)]
    direction, midpoint = _fit_diff_means(train_high, train_low)
    test_high = high + 0.1 * torch.randn(8)
    test_low = low + 0.1 * torch.randn(8)
    assert _classify(test_high, direction, midpoint) == 1
    assert _classify(test_low, direction, midpoint) == 0


def test_run_position_test_reports_shuffled_control():
    handle = load_model(TINY_MODEL, dtype=torch.float32)
    records = [
        _record(position=p, level=lvl, name_a=f"A{i}", name_b=f"B{i}")
        for i, p in enumerate([0, 1] * 10) for lvl in (0, 1)
    ]
    results = run_position_test(
        handle, records, trait_contrast=("expert", "novice"), layers=[0],
        train_fraction=0.5, seed=1, config_hash="abc",
    )
    assert len(results) == 2  # fit_position=0 and fit_position=1, one layer
    for r in results:
        assert r.shuffled_label_control_accuracy is not None
        assert 0.0 <= r.position_invariance_ratio or r.position_invariance_ratio >= 0.0
