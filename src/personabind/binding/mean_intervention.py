from __future__ import annotations

import torch

from personabind.binding.position_test import _fit_diff_means, _read_activation, _split_train_test
from personabind.binding.positions import (
    answer_position,
    render_query_for,
    stored_position,
    tokenize_record,
    trait_of,
)
from personabind.binding.results import InterventionResult
from personabind.common.activations import ModelHandle, forward_logits, patch_residual
from personabind.common.controls import random_direction_matched_norm


def _first_token_id(handle: ModelHandle, text: str) -> int:
    return handle._tokenizer(text, add_special_tokens=False).input_ids[0]


def _logprob_of_token(logits: torch.Tensor, token_id: int) -> float:
    return float(torch.log_softmax(logits[0, -1], dim=-1)[token_id])


def run_mean_intervention(
    handle: ModelHandle, records: list, trait_contrast: tuple[str, str], layers: list[int],
    coefficients: list[float], train_fraction: float, seed: int, config_hash: str,
) -> list[InterventionResult]:
    high_trait, low_trait = trait_contrast
    results: list[InterventionResult] = []

    for layer in layers:
        train_recs, test_recs = _split_train_test(records, train_fraction, seed + layer)
        train_acts = [_read_activation(handle, r, layer) for r in train_recs]
        train_high = [a for r, a in zip(train_recs, train_acts) if r.answer == high_trait]
        train_low = [a for r, a in zip(train_recs, train_acts) if r.answer == low_trait]
        if not train_high or not train_low:
            continue
        direction, _midpoint = _fit_diff_means(train_high, train_low)

        mean_norm = float(torch.stack(train_acts).norm(dim=-1).mean())
        direction_norm_fraction = float(direction.norm()) / mean_norm if mean_norm > 0 else 0.0

        for record in test_recs:
            opposite_trait = low_trait if record.answer == high_trait else high_trait
            sign = -1.0 if record.answer == high_trait else 1.0  # push toward the OPPOSITE value

            tokenized = tokenize_record(record, handle._tokenizer)
            input_ids = torch.tensor([tokenized.input_ids])
            pos = stored_position(tokenized, record, record.query_agent)
            clean_activation = _read_activation(handle, record, layer)
            target_token_id = _first_token_id(handle, opposite_trait)

            clean_logits = forward_logits(handle, input_ids)
            clean_logprob = _logprob_of_token(clean_logits, target_token_id)

            other_agent = next(a.name for a in record.agents if a.name != record.query_agent)
            other_q, other_ap = render_query_for(record, other_agent)
            other_record = record.__class__(**{**record.__dict__, "question": other_q, "answer_prefix": other_ap})
            other_tok = tokenize_record(other_record, handle._tokenizer)
            other_ids = torch.tensor([other_tok.input_ids])
            # other_record's CONTEXT is byte-identical to record's (only question/
            # answer_prefix, which come after it, differ) -- `pos` is the same
            # absolute token index in both tokenizations; no need to re-resolve it.
            other_clean_logits = forward_logits(handle, other_ids)
            other_own_trait = trait_of(record, other_agent)
            other_opposite = low_trait if other_own_trait == high_trait else high_trait
            other_target_token_id = _first_token_id(handle, other_opposite)
            other_clean_logprob = _logprob_of_token(other_clean_logits, other_target_token_id)
            other_read_pos = answer_position(other_tok)

            for coefficient in coefficients:
                seed_i = seed + layer * 100 + int(coefficient * 10)
                patched_vector = clean_activation + sign * coefficient * direction

                patched_logits = patch_residual(handle, input_ids, layer, pos, patched_vector)
                effect_on_target = _logprob_of_token(patched_logits, target_token_id) - clean_logprob

                random_dir = random_direction_matched_norm(direction, seed_i)
                random_vector = clean_activation + sign * coefficient * random_dir
                random_logits = patch_residual(handle, input_ids, layer, pos, random_vector)
                effect_baseline = _logprob_of_token(random_logits, target_token_id) - clean_logprob

                other_patched_logits = patch_residual(handle, other_ids, layer, pos, patched_vector)
                effect_off_target = (
                    _logprob_of_token(other_patched_logits, other_target_token_id) - other_clean_logprob
                )

                results.append(InterventionResult(
                    test="mean_intervention", record_id=record.id, model=handle.model_id,
                    variant=record.variant, layer=layer,
                    layer_type=handle.layer_types[layer] if handle.layer_types else "full_attention",
                    patch_site="stored",
                    token_positions={"patched": pos, "read_on_target": answer_position(tokenized), "read_off_target": other_read_pos},
                    effect_on_target=effect_on_target, effect_norm_matched_random=effect_baseline,
                    effect_off_target=effect_off_target, coefficient=coefficient,
                    direction_norm_fraction=direction_norm_fraction, train_test_split="test",
                    seed=seed_i, config_hash=config_hash,
                ))
    return results
