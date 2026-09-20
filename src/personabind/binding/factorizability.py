"""Test 2: factorizability -- the core causal-patching test. For a
(base, twin) counterfactual pair sharing the same query_agent, save the
twin's activation at a layer/position, patch it into the base at the same
layer/position, and measure the shift in log-probability of the twin's gold
answer's first token -- always alongside a random-direction baseline at the
same site, and (for the "stored" site only) an off-target measurement from
a re-rendered question asked about the OTHER agent.
"""

from __future__ import annotations

import sys
from collections.abc import Callable

import torch
from tqdm import tqdm

from personabind.binding.positions import (
    answer_position,
    measurement_answer_prefix,
    query_agent_position,
    render_query_for,
    stored_position,
    tokenize_record,
    trait_of,
)
from personabind.binding.results import InterventionResult
from personabind.common.activations import (
    ModelHandle,
    forward_logits,
    patch_residual,
    read_residual,
)
from personabind.common.controls import random_direction_matched_norm


def _first_gold_token_id(handle: ModelHandle, gold: str) -> int:
    return handle._tokenizer(gold, add_special_tokens=False).input_ids[0]


def _logprob_of_token(logits: torch.Tensor, token_id: int) -> float:
    log_probs = torch.log_softmax(logits[0, -1], dim=-1)
    return float(log_probs[token_id])


def _measure(
    handle: ModelHandle, base_ids: torch.Tensor, layer: int, pos: int,
    replacement: torch.Tensor, target_token_id: int, clean_logprob: float,
) -> float:
    patched_logits = patch_residual(handle, base_ids, layer, pos, replacement)
    patched_logprob = _logprob_of_token(patched_logits, target_token_id)
    return patched_logprob - clean_logprob


def run_factorizability(
    handle: ModelHandle, record_pairs: list, layers: list[int], seed: int, config_hash: str,
    on_result: Callable[[InterventionResult], None] | None = None,
) -> list[InterventionResult]:
    """`on_result`, if given, is called with each `InterventionResult`
    immediately as it's computed -- e.g. to stream it to disk rather than
    waiting for this (potentially very long: pairs x layers x 2 sites) call
    to finish before anything is written."""
    results: list[InterventionResult] = []
    pairs_with_index = tqdm(
        enumerate(record_pairs), total=len(record_pairs),
        desc="factorizability: pairs", unit="pair", file=sys.stdout,
    )
    for pair_idx, (base, twin) in pairs_with_index:
        base_tok = tokenize_record(base, handle._tokenizer)  # for POSITION resolution only, below
        twin_tok = tokenize_record(twin, handle._tokenizer)
        twin_ids = torch.tensor([twin_tok.input_ids])

        # Measurement uses a bare "{agent} is" answer_prefix instead of base's
        # stored one -- see measurement_answer_prefix's docstring. base_pos,
        # resolved below from base_tok's CONTEXT (unaffected by this prefix
        # change, since context comes first in the text), stays a valid index
        # into this new tokenization too -- same reasoning already used for
        # other_base just below.
        measure_base = base.__class__(
            **{**base.__dict__, "answer_prefix": measurement_answer_prefix(base.query_agent)}
        )
        measure_base_tok = tokenize_record(measure_base, handle._tokenizer)
        base_ids = torch.tensor([measure_base_tok.input_ids])

        other_agent = next(a.name for a in base.agents if a.name != base.query_agent)
        other_q, other_ap = render_query_for(base, other_agent)  # other_ap is now bare too
        other_base = base.__class__(**{**base.__dict__, "question": other_q, "answer_prefix": other_ap})
        other_base_tok = tokenize_record(other_base, handle._tokenizer)
        other_base_ids = torch.tensor([other_base_tok.input_ids])

        # other_base's CONTEXT is byte-identical to base's (only question/answer_prefix,
        # which come after it in the text, differ) -- so any position resolved inside the
        # context (stored_position) is the same absolute token index in both tokenizations.
        # No need to re-resolve it against other_base_tok.
        target_token_id = _first_gold_token_id(handle, trait_of(twin, twin.query_agent))
        other_target_token_id = _first_gold_token_id(handle, trait_of(twin, other_agent))

        clean_logits = forward_logits(handle, base_ids)
        clean_logprob = _logprob_of_token(clean_logits, target_token_id)
        other_clean_logits = forward_logits(handle, other_base_ids)
        other_clean_logprob = _logprob_of_token(other_clean_logits, other_target_token_id)

        for layer in layers:
            for patch_site, base_pos, twin_pos in (
                ("stored", stored_position(base_tok, base, base.query_agent),
                 stored_position(twin_tok, twin, twin.query_agent)),
                ("retrieved", query_agent_position(base_tok, base),
                 query_agent_position(twin_tok, twin)),
            ):
                seed_i = seed + pair_idx * 1000 + layer
                twin_activation = read_residual(handle, twin_ids, layer, twin_pos)

                effect_on_target = _measure(
                    handle, base_ids, layer, base_pos, twin_activation, target_token_id, clean_logprob
                )
                random_direction = random_direction_matched_norm(twin_activation, seed_i)
                effect_baseline = _measure(
                    handle, base_ids, layer, base_pos, random_direction, target_token_id, clean_logprob
                )

                effect_off_target = None
                read_off_target = None
                if patch_site == "stored":
                    effect_off_target = _measure(
                        handle, other_base_ids, layer, base_pos, twin_activation,
                        other_target_token_id, other_clean_logprob,
                    )
                    read_off_target = answer_position(other_base_tok)

                result = InterventionResult(
                    test="factorizability", record_id=base.id, model=handle.model_id,
                    variant=base.variant, layer=layer,
                    layer_type=handle.layer_types[layer] if handle.layer_types else "full_attention",
                    patch_site=patch_site,
                    token_positions={"patched": base_pos, "read_on_target": answer_position(measure_base_tok), "read_off_target": read_off_target},
                    effect_on_target=effect_on_target, effect_norm_matched_random=effect_baseline,
                    effect_off_target=effect_off_target, coefficient=1.0, direction_norm_fraction=None,
                    train_test_split="n/a", seed=seed_i, config_hash=config_hash,
                )
                results.append(result)
                if on_result is not None:
                    on_result(result)
    return results
