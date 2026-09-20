# Qwen3-8B, T1 (discrete) — accuracy scorer bug, not a binding failure

**Verdict as originally reported (before the fix below):**

```json
{
  "model": "Qwen/Qwen3-8B",
  "verdict": "t1_fails",
  "message": "Rig broken, or model too small -- re-run at the next larger model before concluding anything",
  "per_variant": {
    "t1_discrete": {
      "accuracy": {
        "accuracy": 0.5,
        "n": 2000,
        "ci_low": 0.4778505542311098,
        "ci_high": 0.5221494457688902
      },
      "passed": false
    }
  }
}
```

`accuracy_floor` is 0.90 (`configs/binding.yaml`), so the kill-criteria gate stopped at T1 and the causal tests (factorizability, position generalization, mean intervention) never ran for this model.

## This is not evidence Qwen3-8B fails to bind T1 traits

Inspecting the raw per-record predictions (`{model}__t1_discrete__accuracy.jsonl`) shows the model predicting the correct trait in every sampled row:

```
{"predicted": "expert",  "gold": "expert", "correct": true}
{"predicted": "novice,", "gold": "novice", "correct": false}
{"predicted": "novice,", "gold": "novice", "correct": false}
{"predicted": "expert",  "gold": "expert", "correct": true}
{"predicted": "expert",  "gold": "expert", "correct": true}
{"predicted": "novice,", "gold": "novice", "correct": false}
{"predicted": "novice,", "gold": "novice", "correct": false}
{"predicted": "expert",  "gold": "expert", "correct": true}
{"predicted": "novice,", "gold": "novice", "correct": false}
{"predicted": "expert",  "gold": "expert", "correct": true}
{"predicted": "novice,", "gold": "novice", "correct": false}
{"predicted": "expert",  "gold": "expert", "correct": true}
```

Every `expert` case scores correct. Every `novice` case scores incorrect — not because the model predicted the wrong trait, but because it appended a trailing comma (`"novice,"` vs. gold `"novice"`), and the scorer required an exact string match. Since T1 is built perfectly balanced (equal `expert`/`novice` counts), "every expert case correct, every novice case wrong due to punctuation" nets out to *exactly* 0.500 regardless of how well the model actually binds traits — which is exactly what was measured, down to a suspiciously tight confidence interval (0.478–0.522) that is itself a symptom: real partial capability reads as some value meaningfully above chance, not dead-centered on it.

## Root cause

`src/personabind/binding/accuracy.py`'s `run_accuracy` decided how many tokens to greedily generate like this:

```python
gold_ids = handle._tokenizer(record.answer, add_special_tokens=False).input_ids
...
decoded = greedy_decode(handle, prefix_ids, n_tokens=max(1, len(gold_ids)))
```

`record.answer` is tokenized **bare** (no leading space) to decide `n_tokens`. But `answer_prefix` (e.g. `"Charles is a"`) never ends in a space, so the model's real completion is the tokenizer's encoding of `" novice"` (**with** a leading space) — and BPE tokenizers frequently split a bare word and its leading-space form into a *different number* of tokens. For Qwen's tokenizer, `"expert"` apparently comes out the same length either way (clean), but `"novice"` needed one more token bare than it does in its natural leading-space form. The practical effect: `greedy_decode` was told to generate one token too many for every `novice` case, the model correctly emitted `" novice"` in the first token, and then — forced to keep going — greedily filled the superfluous extra slot with a comma.

(Confirmed reproducible on `sshleifer/tiny-gpt2`'s real GPT-2 tokenizer too: bare `"novice"` → 2 tokens, `" novice"` → 1 token — same mechanism, different specific words affected than on Qwen's vocabulary.)

This is a token-*count* bug, not a token-*content* bug — depending on which direction the bare-vs-spaced tokenization diverges, it can just as easily **truncate** an answer instead of padding it with a stray token, which no amount of punctuation-stripping would fix. T2's multi-word trait phrases (`"first-year student"`, `"board-certified expert"`, etc.) are the most exposed to this, since they have more tokens and more chances for a boundary mismatch.

## Fix applied

`src/personabind/binding/accuracy.py`, two changes:

1. **Root cause:** `gold_ids` is now computed by tokenizing `" " + record.answer` (leading space), matching how the model would actually need to write the word in context, so `n_tokens` is correct in both the truncation and padding directions.
2. **Defense in depth:** the correctness comparison now strips trailing punctuation (`,.;:!?`) before comparing — this is a genuinely independent risk from the token-count bug (chat-tuned models have a real habit of appending trailing punctuation as a continuation reflex) and is still a full-string comparison, never a truncated-prefix one (`"first-year student"` vs. `"first-class citizen"` still correctly differ).

Both changes are covered by new regression tests in `tests/test_accuracy.py`, including one that reproduces the exact token-count-mismatch mechanism on a real tokenizer (not mocked) and confirms it against the pre-fix behavior.

## What this means for the numbers you have

- **T1's reported 0.500 accuracy for Qwen3-8B is not trustworthy and should be discarded.** Based on the sampled rows above, the true figure is likely much closer to ceiling, but that is an inference from 12 non-randomly-selected rows, not a re-measurement — **the behavioral accuracy test needs to be re-run** with the fixed code to get a real number before anything else in the battery can be trusted for this model.
- **Nothing downstream of T1 ran for Qwen3-8B** (the gate stopped there), so there's no causal-test data to reinterpret for this model yet.
- **If Qwen3-4B was also run and hit a similarly exact/near-exact chance result**, that's strong corroborating evidence the *same* bug was in play there too, not evidence either model is "too small" — re-run it with the fix as well before drawing any capability conclusion.

## An open, related risk — not yet fixed, worth checking before trusting T2/T3/causal numbers

The exact same class of bug (tokenizing a gold trait word bare instead of with a leading space) also exists in the causal-test modules' target-token lookups — `_first_gold_token_id` in `factorizability.py` and `_first_token_id` in `mean_intervention.py`. This was flagged as an unverified risk during the final pre-merge review (never exercised against a real model until this run), and this incident is the first concrete confirmation that the mechanism is real and impactful. Unlike the accuracy bug, a wrong token ID there doesn't fail loudly — it just measures the probability of the *wrong* token, silently depressing or biasing the causal effect sizes those tests report. **Recommend fixing this the same way (tokenize with a leading space) before trusting any factorizability/mean-intervention numbers**, even once T1 passes with the corrected accuracy scorer.
