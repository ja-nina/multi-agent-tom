from __future__ import annotations

import os
import sys
from collections.abc import Callable

import torch
from tqdm import tqdm

from personabind.binding.position_test import _read_activation, _split_train_test
from personabind.binding.positions import (
    answer_position,
    measurement_answer_prefix,
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


def _valid_pairs_for_contrast(
    records: list, high_trait: str, low_trait: str,
) -> list[tuple[str, object, object]]:
    """Group `records` into matched counterfactual (high, low) pairs --
    same pair-key convention as `_split_train_test` (`min(id,
    counterfactual_id)`) -- keeping only pairs where BOTH members are
    actually present and their answers are exactly {high_trait, low_trait}.
    A pair missing its twin, or where neither/only-one member is a
    contrast extreme (e.g. a T2 pair between two middle tiers, or a mid-
    tier/high-tier pair), is dropped -- the same exclusion the module
    already applies per-record to the test fold, applied here at the PAIR
    level for fitting."""
    groups: dict[str, list] = {}
    for r in records:
        key = min(r.id, r.counterfactual_id)
        groups.setdefault(key, []).append(r)
    pairs = []
    for pair_key, members in groups.items():
        if len(members) != 2:
            continue
        answers = {m.answer for m in members}
        if answers != {high_trait, low_trait}:
            continue
        high_rec = next(m for m in members if m.answer == high_trait)
        low_rec = next(m for m in members if m.answer == low_trait)
        pairs.append((pair_key, high_rec, low_rec))
    return pairs


def _fit_diff_means_from_pairs(
    handle: ModelHandle, pairs: list[tuple[str, object, object]], layer: int,
) -> tuple[torch.Tensor, list[dict]]:
    """Fit a diff-of-means direction from MATCHED counterfactual pairs
    instead of pooling unrelated records: for each (high, low) pair
    sharing the same agent name (only the bound trait differs between the
    two), read BOTH members' activation at that shared agent's
    stored_position, and take the pairwise difference oriented
    high-minus-low. Averaging these pairwise differences isolates what
    changes when THIS SPECIFIC agent's trait flips, controlling out
    confounds a pooled diff-of-means over unrelated records can't separate
    (other agent's identity, context content, position) -- this is the fix
    for the entanglement found in the pooled version, where
    effect_off_target came out roughly equal to effect_on_target.

    Also returns per-record metadata (record id, agent name, trait, pair
    key, the raw activation tensor) for every record actually used, so a
    caller can dump the raw vectors for later offline analysis alongside
    the fitted direction."""
    diffs = []
    vector_records: list[dict] = []
    for pair_key, high_rec, low_rec in pairs:
        high_act = _read_activation(handle, high_rec, layer)
        low_act = _read_activation(handle, low_rec, layer)
        diffs.append(high_act - low_act)
        for rec, act, trait in ((high_rec, high_act, high_rec.answer), (low_rec, low_act, low_rec.answer)):
            vector_records.append({
                "record_id": rec.id, "agent_name": rec.query_agent, "trait": trait,
                "pair_key": pair_key, "layer": layer, "vector": act,
            })
    direction = torch.stack(diffs).mean(dim=0)
    return direction, vector_records


def save_layer_vectors(vector_records: list[dict], path: str) -> None:
    """Save one layer's raw activation vectors (the ones used to fit its
    pairwise direction) plus their metadata, for later offline analysis --
    never read back by this module itself. `vectors[i]` corresponds to
    `metadata[i]` (record_id/agent_name/trait/pair_key/layer)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    vectors = torch.stack([r["vector"] for r in vector_records])
    metadata = [
        {"record_id": r["record_id"], "agent_name": r["agent_name"], "trait": r["trait"],
         "pair_key": r["pair_key"], "layer": r["layer"]}
        for r in vector_records
    ]
    torch.save({"vectors": vectors, "metadata": metadata}, path)


def run_mean_intervention(
    handle: ModelHandle, records: list, trait_contrast: tuple[str, str], layers: list[int],
    coefficients: list[float], train_fraction: float, seed: int, config_hash: str,
    on_result: Callable[[InterventionResult], None] | None = None,
    vectors_dir: str | None = None,
) -> list[InterventionResult]:
    """`on_result`, if given, is called with each `InterventionResult`
    immediately as it's computed -- e.g. to stream it to disk rather than
    waiting for this (potentially very long: layers x records x
    coefficients) call to finish before anything is written.

    `vectors_dir`, if given, saves every layer's raw activation vectors
    (the ones used to fit that layer's pairwise direction) plus their
    metadata to `{vectors_dir}/{model}__{variant}__layer{layer}.pt`, for
    later offline analysis -- see `save_layer_vectors`. None (the default)
    skips this entirely."""
    high_trait, low_trait = trait_contrast
    results: list[InterventionResult] = []

    # Split ONCE (not per layer) so every layer's adjacent-layer comparison
    # (battery.clears_baseline) is drawn from the SAME train/test records, as
    # the spec's design intends -- not different data at each layer.
    train_recs, test_recs = _split_train_test(records, train_fraction, seed)
    # Only the two contrast extremes are steerable: T2 has four tiers, and a
    # mid-tier record is neither `high_trait` nor `low_trait`, so treating it
    # as "not high => low" would mislabel its steering direction and effect.
    test_recs = [r for r in test_recs if r.answer in (high_trait, low_trait)]

    pairs = _valid_pairs_for_contrast(train_recs, high_trait, low_trait)
    if not pairs:
        raise ValueError(
            f"run_mean_intervention: no complete ({high_trait!r}, {low_trait!r}) counterfactual "
            "pairs in the training fold -- cannot fit a pairwise diff-of-means direction. "
            "Increase train_fraction or sample size."
        )

    for layer in tqdm(layers, desc="mean_intervention: layers", unit="layer", file=sys.stdout):
        direction, vector_records = _fit_diff_means_from_pairs(handle, pairs, layer)
        if vectors_dir is not None:
            variant = records[0].variant if records else "unknown_variant"
            path = os.path.join(
                vectors_dir, f"{handle.model_id.replace('/', '_')}__{variant}__layer{layer}.pt"
            )
            save_layer_vectors(vector_records, path)

        all_vectors = torch.stack([r["vector"] for r in vector_records])
        mean_norm = float(all_vectors.norm(dim=-1).mean())
        direction_norm_fraction = float(direction.norm()) / mean_norm if mean_norm > 0 else 0.0

        for record in test_recs:
            opposite_trait = low_trait if record.answer == high_trait else high_trait
            sign = -1.0 if record.answer == high_trait else 1.0  # push toward the OPPOSITE value

            tokenized = tokenize_record(record, handle._tokenizer)  # for POSITION resolution only
            pos = stored_position(tokenized, record, record.query_agent)
            clean_activation = _read_activation(handle, record, layer)
            target_token_id = _first_token_id(handle, opposite_trait)

            # Measurement uses a bare "{agent} is" answer_prefix instead of the
            # record's stored one -- see measurement_answer_prefix's docstring.
            # `pos`, resolved above from the CONTEXT (which precedes the prefix
            # in the text), is the same absolute token index in both tokenizations.
            measure_record = record.__class__(
                **{**record.__dict__, "answer_prefix": measurement_answer_prefix(record.query_agent)}
            )
            measure_tok = tokenize_record(measure_record, handle._tokenizer)
            input_ids = torch.tensor([measure_tok.input_ids])

            clean_logits = forward_logits(handle, input_ids)
            clean_logprob = _logprob_of_token(clean_logits, target_token_id)

            other_agent = next(a.name for a in record.agents if a.name != record.query_agent)
            other_q, other_ap = render_query_for(record, other_agent)  # other_ap is now bare too
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

                result = InterventionResult(
                    test="mean_intervention", record_id=record.id, model=handle.model_id,
                    variant=record.variant, layer=layer,
                    layer_type=handle.layer_types[layer] if handle.layer_types else "full_attention",
                    patch_site="stored",
                    token_positions={"patched": pos, "read_on_target": answer_position(measure_tok), "read_off_target": other_read_pos},
                    effect_on_target=effect_on_target, effect_norm_matched_random=effect_baseline,
                    effect_off_target=effect_off_target, coefficient=coefficient,
                    direction_norm_fraction=direction_norm_fraction, train_test_split="test",
                    seed=seed_i, config_hash=config_hash,
                )
                results.append(result)
                if on_result is not None:
                    on_result(result)
    return results
