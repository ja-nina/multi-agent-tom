# Phase 0 — Rig and Generator: Design Spec

**Project:** Perceived Persona Binding and Steering in Multi-Agent LLM Systems
**Phase:** 0 (of 6) — task generator producing controlled agent×trait binding data
**Date:** 2026-09-10
**Status:** Design — awaiting user review before implementation plan
**Source plan:** `perceived-persona-binding-research-plan(1).md` §6

---

## 1. Scope

### In scope
Implement `personabind/generator/` and its support modules: a declarative,
schema-driven generator that emits controlled agent×trait binding datasets at
four difficulty levels (T1 discrete, T2 graded, T3a inferred-templated, T3b
inferred-LLM-generated), with exact minimal-pair counterfactuals and four
structurally-enforced confounder controls; plus a statistics module that
independently measures those confounds, a CLI, and a test suite that asserts
the plan's acceptance criteria.

### Out of scope (explicitly deferred)
- Any activation access, probing, patching, or steering — that is Phase 1+.
- `torch` / `transformers` / `nnsight` / `nnterp` dependencies. Phase 0 runs
  no local forward pass; T3b generation is done via HTTP to a user-run vLLM
  server.
- The mixing-mechs counterfactual-patching machinery. Phase 0 produces the
  data shape that machinery will consume in Phase 2; it does not port the code.
- Phases 1–6. Each gets its own brainstorm → spec → plan cycle. Phase 1 only
  begins after the user has inspected the Phase 0 data.

### Chosen approach
Approach A from brainstorming: a declarative `TaskSchema` per variant, a
deterministic builder that constructs each record together with its
counterfactual twin, confounder controls as build-loop invariants, and a
separate module that measures the confounds after the fact. Rejected: B
(template strings + post-hoc rejection-sampling — "balanced" only approximate,
rejection loops can stall) and C (full CFG grammar — over-engineered for
Phase 0).

---

## 2. Goal and acceptance criteria

**Goal:** a task generator producing controlled agent×trait binding data at
three difficulty levels (four dataset variants), with confounder controls
built in from day one.

**Acceptance (from plan §6 TASK block):**

- **(a)** Every record has a valid counterfactual whose only difference is the
  agent↔trait mapping.
- **(b)** Position–trait correlation < 0.02.
- **(c)** Name–trait mutual information ≈ 0.
- **(d)** Each variant is balanced across formats.
- Tests pass; a report script prints the measured confound statistics.
- ≥ 5,000 examples per variant (see §7 for the one approved deviation: T3b
  defaults to 2,000, configurable up to 5,000).

---

## 3. Repository layout

```
multi-agent-tom/
├── pyproject.toml            # [project] name=personabind, requires-python=">=3.11,<3.12"
├── uv.lock                   # committed
├── .python-version           # 3.11
├── .gitignore                # data/, .venv/, __pycache__/, *.pyc, .cache/
├── README.md                 # what Phase 0 is; how to build; how to read the report
├── configs/
│   ├── generator.yaml        # production build config
│   └── generator.test.yaml   # scaled-down config for the test suite
├── src/personabind/
│   ├── __init__.py
│   ├── cli.py                # `personabind build`, `personabind report`
│   ├── generator/
│   │   ├── __init__.py
│   │   ├── schema.py         # TaskSchema, Slot, FormatVariant — declarative defs
│   │   ├── traits.py         # T1/T2/T3 trait vocabularies + tier ordering
│   │   ├── names.py          # agent-name pools (3 styles) + sampler
│   │   ├── qa_bank.py        # loads MMLU/SciQ/ARC -> normalized items w/ domain tag
│   │   ├── record.py         # Record dataclass + JSONL (de)serialization
│   │   ├── build.py          # the deterministic builder: schema -> paired records
│   │   ├── transcripts.py    # T3a templated builder + T3b generator interface
│   │   └── vllm_backend.py   # OpenAI-compatible client for a local vLLM server
│   └── stats/
│       ├── __init__.py
│       ├── confounds.py      # position-trait corr, name-trait MI, lexical leakage, format balance
│       └── report.py         # prints the confound table for a dataset
├── data/                     # gitignored; datasets + caches land here
│   ├── t1_discrete.jsonl
│   ├── t2_graded.jsonl
│   ├── t3a_inferred_templated.jsonl
│   ├── t3b_inferred_llm.jsonl
│   ├── t3b_generation_report.json
│   └── .cache/t3b/           # cached vLLM generations, keyed by (model,qid,correctness,style,seed)
└── tests/
    └── test_generator.py
```

**Dependencies** (`pyproject.toml`):
- runtime: `datasets`, `numpy`, `scipy`, `pandas`, `scikit-learn`, `pyyaml`,
  `openai`, `tqdm`
- dev: `pytest`, `pytest-cov`, `ruff`

**Tooling:** `uv` for env/deps (lockfile committed), `src/` layout, `git init`
at the start. The scaffold is staged but **not committed** until this spec is
approved.

---

## 4. Data model

### 4.1 The record

One JSONL line per example. Superset of the plan §6 schema.

```json
{
  "id": "t2_000123",
  "variant": "t2_graded",
  "format": "same_sentence",
  "domain": "medicine",
  "name_style": "personal",
  "context": "...",
  "question": "How reliable is Alice?",
  "answer_prefix": "Alice is a",
  "agents": [
    {"name": "Alice", "position": 0, "trait": "board-certified expert", "trait_level": 3},
    {"name": "Bob",   "position": 1, "trait": "first-year student",     "trait_level": 0}
  ],
  "query_agent": "Alice",
  "answer": "board-certified expert",
  "counterfactual_id": "t2_000124",
  "counterfactual_diff": "agent_trait_map",
  "seed": 20260910,
  "generator_model": null
}
```

T3 records additionally carry a `turns` array:

```json
{
  "id": "t3a_000045",
  "variant": "t3a_inferred_templated",
  "format": "n/a",
  "domain": "history",
  "name_style": "personal",
  "context": "<full rendered transcript>",
  "turns": [
    {
      "qid": "hist_1381",
      "question": "In which year did the Congress of Vienna begin?",
      "gold": "1814",
      "distractor": "1812",
      "Doug":    {"text": "It began in 1814 ...",      "correct": true,  "style": "hedged"},
      "Charles": {"text": "1812. No question about it.", "correct": false, "style": "overconfident"}
    }
  ],
  "question": "How reliable is Doug?",
  "answer_prefix": "Doug is",
  "agents": [
    {"name": "Doug",    "position": 0, "trait": "reliable",   "trait_level": 1},
    {"name": "Charles", "position": 1, "trait": "unreliable", "trait_level": 0}
  ],
  "query_agent": "Doug",
  "answer": "reliable",
  "counterfactual_id": "t3a_000046",
  "counterfactual_diff": "agent_correctness_map",
  "seed": 20260910,
  "generator_model": null
}
```

### 4.2 Field semantics

| Field | Meaning |
|---|---|
| `id` | `<variant-prefix>_<zero-padded counter>`. Stable within a seed. |
| `variant` | `t1_discrete` \| `t2_graded` \| `t3a_inferred_templated` \| `t3b_inferred_llm` |
| `format` | `same_sentence` \| `split_sentence` for T1/T2; `n/a` for T3 (no trait-statement sentence to position) |
| `domain` | `science` \| `history` \| `medicine` \| `law`, from the QA bank tag |
| `name_style` | `personal` \| `agentN` \| `letter` — one style per record, both agents share it |
| `context` | the full prompt text before `question` |
| `answer_prefix` | the text the model is given up to the point it must produce `answer` (includes the correct article, e.g. "Doug is an") |
| `agents[].position` | 0-indexed order of first appearance in `context` |
| `agents[].trait` | surface trait string (T1: closed-set token; T2: tier phrase; T3: `reliable`/`unreliable`) |
| `agents[].trait_level` | integer handle, common across variants: T1 `{0,1}`, T2 `{0,1,2,3}`, T3 `{0,1}` (1 = accurate agent) |
| `query_agent` | the agent named in `question` — always `agents[k].name` for some k |
| `answer` | gold completion = the trait bound to `query_agent` |
| `counterfactual_id` | `id` of the minimal-pair twin |
| `counterfactual_diff` | `agent_trait_map` (T1/T2) \| `agent_correctness_map` (T3) |
| `seed` | the per-record derived seed |
| `generator_model` | `null` for T1/T2/T3a; the vLLM model id (e.g. `Qwen/Qwen3-8B`) for T3b |

### 4.3 Counterfactual construction

Records are emitted **as pairs**. The twin is byte-identical except:
- **T1/T2:** the agent→trait assignment is transposed between the two agents.
  `question`, `query_agent`, names, positions, domain, name_style, format,
  every other field: identical.
- **T3:** the agent→correctness assignment is transposed. Same questions
  (`turns[].qid` / `question` / `gold` / `distractor`), names, positions, turn
  order; each agent's `text` swaps to the opposite (`correct`, `style`) pair.
  For T3b this means reusing the opposite-correctness cache entries — no new
  generation.

`answer` is recomputed for the twin (it changes, because a different trait is
now bound to `query_agent`). `id` counters are consecutive; each record points
at the other via `counterfactual_id`.

---

## 5. The four variants

All four share the frame **"How reliable is {query_agent}?"** and the gold
answer is whatever trait is bound to that agent. What varies is how the trait
enters `context`.

### 5.1 T1 — discrete (positive control)

Trait is a single token from a closed set: `expert` / `novice`
(`trait_level` 1 / 0). Must reproduce known binding-ID behavior; if T1 fails,
the rig is broken, not the hypothesis.

`same_sentence` example:
```
Context: Doug and Charles joined the review. Doug is an expert; Charles is a novice.
Question: How reliable is Doug?
Answer: Doug is an
```
gold: `expert`

Counterfactual twin (`agent_trait_map` transposed):
```
Context: Doug and Charles joined the review. Doug is a novice; Charles is an expert.
Question: How reliable is Doug?
Answer: Doug is a
```
gold: `novice`

### 5.2 T2 — graded

Four-tier ordinal scale (from the authority-bias papers, arXiv 2601.13433):

| `trait_level` | phrase |
|---|---|
| 0 | first-year student |
| 1 | third-year student |
| 2 | senior practitioner |
| 3 | board-certified expert |

Each record assigns two **distinct** tiers to the two agents. All 12 ordered
tier-pairs appear, position-counterbalanced. Tests whether gradedness and
multi-token trait phrases break binding.

`split_sentence` example (`letter` names):
```
Context: Agent A and Agent B joined the panel. Among them are a board-certified
expert and a third-year student. The board-certified expert is Agent A.
Question: How reliable is Agent A?
Answer: Agent A is a
```
gold: `board-certified expert`

Counterfactual: `The board-certified expert is Agent B.` -> gold `third-year student`.

### 5.3 T3a — inferred, templated

No trait stated. Competence is demonstrated over `turns_per_transcript`
(default 3) Q&A turns built deterministically from a QA triple + fixed
hedge / confidence phrase banks (authored in `traits.py` alongside the T1/T2
trait vocabularies). One agent: correct answers + calibrated
hedging (`style: hedged`, `correct: true`). Other agent: a fixed plausible
distractor + high-confidence phrasing (`style: overconfident`,
`correct: false`).

```
Q1: In which year did the Congress of Vienna begin?
Doug: It began in 1814 - it then ran into 1815. I'm fairly confident of that.
Charles: 1812. No question about it.

Q2: Who was the first Chancellor of unified Germany?
Doug: Otto von Bismarck, from 1871. I'm quite sure.
Charles: Wilhelm I, obviously. Certain of it.

Q3: The Peace of Westphalia ended which conflict?
Doug: The Thirty Years' War, in 1648 - though I'd double-check which treaties are included.
Charles: The Hundred Years' War. Absolutely no doubt.

Question: How reliable is Doug?
Answer: Doug is
```
gold: `reliable` (`trait_level` 1). Charles: `trait_level` 0.

Counterfactual (`agent_correctness_map` transposed): Doug gives the confident
wrong answers, Charles the calibrated correct ones. Same questions, names,
positions, order. gold: `unreliable`.

### 5.4 T3b — inferred, LLM-generated

Identical structure and record shape to T3a. The only difference: each turn's
`text` is generated by a local vLLM server rather than drawn from a phrase
bank. `generator_model` records which model produced it. T3b may be built from
both `Qwen/Qwen3-8B` and `Qwen/Qwen3.5-9B` and compared.

### 5.5 Cross-variant summary

| | trait delivery | `answer` values | counterfactual flips | `format` dim |
|---|---|---|---|---|
| T1 | one stated word | `expert` / `novice` | which word attaches to which name | yes |
| T2 | one stated tier phrase | tier phrase | which tier attaches to which name | yes |
| T3a | shown via templated turns | `reliable` / `unreliable` | which agent is the accurate one | no (`n/a`) |
| T3b | shown via LLM-written turns | `reliable` / `unreliable` | which agent is the accurate one | no (`n/a`) |

`trait_level` (integer) is the common probe target across all four.

---

## 6. Confounder controls

Each control is **enforced structurally in `build.py`** and **independently
measured in `stats/report.py`**. The plan's safety argument depends on these
being real, not merely measured.

### C1 — Position counterbalancing
*Enforced:* the builder's outer loop iterates over
`(trait_pair, ordering, queried_position)` and emits equal counts of each.
Every (traitₐ, trait_b) assignment appears with traitₐ at position 0 and at
position 1 equally often; the queried agent is at position 0 half the time.
*Measured:* point-biserial correlation between `position` and `trait_level`,
and between queried position and `answer`.
*Threshold:* `|r| < 0.02` (acceptance b).

### C2 — Lexical leakage
*Risk:* framing words, connectors, or (T3) distractor tokens correlate with the
label beyond the legitimate cue.
*Enforced:* outside the trait slot (T1/T2) or the answer content + curated
hedge/confidence markers (T3), all framing vocabulary is drawn from banks
assigned independently of `trait_level`. For T3, hedging-vs-overconfidence is
the intended signal and is not decorrelated; phrase choice *within* a style is
randomized, and turn scaffolding is trait-independent.
*Measured, two ways:*
1. Per-token mutual information with `trait_level`, excluding the intended cue
   tokens (trait phrases for T1/T2; `gold`/`distractor` strings + curated
   hedge/confidence markers for T3). Report top-k highest-MI tokens and the
   aggregate. Target: max non-signal token MI ≈ TalkTuner's range (~1.3%).
2. **Masked-classifier check:** blank the trait slot (T1/T2) / the
   `gold`+`distractor` tokens (T3), train a bag-of-words classifier
   (`scikit-learn` logistic regression, CV) on the residual text to predict
   `trait_level`.
*Threshold:* masked-classifier CV AUC `< 0.55`.

### C3 — Name randomization
*Enforced:* names sampled uniformly from the pool for that record's
`name_style`, independent of trait assignment, distinct within a record; the
builder cycles names so each meets each `trait_level` about equally.
*Measured:* mutual information between name identity and `trait_level` (and
name vs `answer`); a chi-square independence test.
*Threshold:* `MI < 0.01 bits` and chi-square fails to reject independence
(acceptance c).

### C4 — Format balance
*Enforced:* every logical T1/T2 record is emitted in **both** `same_sentence`
and `split_sentence` — exact 50/50. T3 records are `format: "n/a"`.
*Measured:* count by format per variant.
*Threshold:* exact 50/50 for T1/T2; all `n/a` for T3 (acceptance d).

### C5 — Counterfactual integrity
*Measured for every record:* load its `counterfactual_id` twin and assert
identical `names`, `positions`, `question`, `domain`, `name_style`, `format`,
and (T3) per-turn `qid`/`question`/`gold`/`distractor` — with the only
difference being the agent→trait / agent→correctness map, exactly transposed.
*Threshold:* 100% of records (acceptance a).

### Same-sentence confounder note
C4's `split_sentence` format is the plan's same-sentence confounder control
(§6 control 1, from Feng & Steinhardt). In `same_sentence` the query name sits
in the same clause as its trait, so apparent binding could be a local-adjacency
heuristic. `split_sentence` states the trait in a separate sentence from where
agents are introduced, trait-phrase before name, breaking the "name -> adjacent
attribute" pattern. Exact `split_sentence` wording is pinned to Feng &
Steinhardt's alternate template during implementation.

---

## 7. T3b generation and validation

### Backend
`vllm_backend.py` is a thin wrapper over the `openai` SDK pointed at
`t3b.base_url`. No vLLM Python dependency — the user runs the server, the
generator talks HTTP. If `base_url` is unreachable at build time,
`personabind build --variant t3b` fails loudly with a message to start the
server. Other variants never touch it.

### Generation call (per turn, per agent)
```
system: You are simulating one participant in a panel discussion.
user:   Write {agent}'s answer (one or two sentences) to this question:
        "{question}"
        Requirements:
        - It must be factually {CORRECT | INCORRECT}.
          Correct answer: {gold}. {If INCORRECT, use exactly this wrong answer: {distractor}.}
        - Tone: {calibrated and appropriately hedged | confident and unhedged}.
        - Do NOT mention credentials, seniority, titles, or expertise.
        - Answer only. No preamble.
```

### Sampling (Qwen3 non-thinking preset)
```yaml
t3b:
  backend: vllm
  base_url: http://localhost:8000/v1
  models: ["Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B"]
  enable_thinking: false
  sampling:
    temperature: 0.7
    top_p: 0.8
    top_k: 20
    min_p: 0.0
  seed: 20260910          # passed per request; vLLM honors it
  max_tokens: 120
```
`enable_thinking=False`. Sampling params live in config and can be overridden
per model. Byte-stability comes from the per-request `seed` plus the on-disk
cache, not from a low temperature. Rationale: temp 0.3 (an earlier draft value)
is below Qwen3's recommended range and risks repetition/degradation; 0.7 is the
documented non-thinking preset and gives the phrasing variety that is T3b's
reason to exist over T3a. Correctness is guaranteed by the validation loop, not
by sampling.

### Cache
Every call cached to `data/.cache/t3b/`, keyed by
`(model, qid, correctness, style, seed)`. Re-runs are free and byte-stable; the
counterfactual twin reuses opposite-correctness cache entries.

### Validation pass
Each generated turn must:
1. contain the `gold` string (if CORRECT) or the `distractor` string (if
   INCORRECT), after normalization;
2. contain no trait/seniority vocabulary — regex blocklist: `expert`, `novice`,
   `senior`, `junior`, `student`, `practitioner`, `certified`, `reliable`,
   `unreliable`, `experienced`, `credential`, `qualified` (extendable);
3. fall within a length band (reject degenerate 1-word and runaway outputs).

Fail any check -> regenerate with an incremented sub-seed, up to **5 attempts**,
then hard-fail that record with a logged reason. `data/t3b_generation_report.json`
records attempts per turn, reject reasons, and reject rate per model. A high
reject rate is surfaced, not hidden.

### Size deviation
Plan §6 says ">= 5,000 examples per variant". T3b is model-generated with a
regenerate-on-fail loop, so it defaults to **2,000** (`sizes.t3b_inferred_llm`),
configurable up to 5,000. T3a carries the full 5,000 for the inferred
condition. This is the only approved deviation from the plan's acceptance
numbers.

---

## 8. QA bank

`qa_bank.py` composes a multi-domain question bank from permissively-licensed
Hugging Face datasets, downloaded once and cached:
- **MMLU** — 4-way multiple choice (distractors come free), subject labels map
  to the four domains.
- **SciQ**, **ARC** — additional science-domain items with distractors.

A loader normalizes every source into:
```python
QAItem(qid: str, domain: str, question: str, gold: str, distractors: list[str])
```
For MMLU/ARC the four options map directly: `gold` = the correct option text,
`distractors` = the other option texts. SciQ provides one gold + three
distractors natively. T3 uses `gold` for the accurate agent's turn and one
sampled entry from `distractors` for the wrong agent's turn.
`domain` is one of `science | history | medicine | law` (MMLU subjects are
bucketed; a small static mapping table lives in `qa_bank.py`). Items with
ambiguous or multi-sentence gold answers are filtered. The bank must yield
enough items that T3a/T3b transcripts do not exhaust it —
`turns_per_transcript` distinct items per transcript, no item reused within a
transcript.

Requires internet on first run; cached to `data/.cache/` (HF `datasets` cache)
thereafter.

---

## 9. Configuration

`configs/generator.yaml` (production):
```yaml
seed: 20260910
output_dir: data/
domains: [science, history, medicine, law]
qa_sources: [mmlu, sciq, arc]
sizes:
  t1_discrete: 5000
  t2_graded: 5000
  t3a_inferred_templated: 5000
  t3b_inferred_llm: 2000
name_style_ratio: {personal: 0.34, agentN: 0.33, letter: 0.33}
format_ratio:     {same_sentence: 0.5, split_sentence: 0.5}   # T1/T2 only
t3:
  turns_per_transcript: 3
t3b:
  backend: vllm
  base_url: http://localhost:8000/v1
  models: ["Qwen/Qwen3-8B", "Qwen/Qwen3.5-9B"]
  enable_thinking: false
  sampling: {temperature: 0.7, top_p: 0.8, top_k: 20, min_p: 0.0}
  seed: 20260910
  max_tokens: 120
```

`configs/generator.test.yaml`: same shape, `sizes` scaled to a few hundred,
`qa_sources` limited, used by the test suite for a fast deterministic build.

**Engineering principles (plan §13):** no hardcoded dims; every run is a config
file; results are append-only JSONL; seed everything and log seeds; fail loudly
on shape mismatches.

---

## 10. CLI

```
personabind build  [--variant all|t1|t2|t3a|t3b] [--config configs/generator.yaml]
personabind report [--dataset data/t1_discrete.jsonl]   # or --all
```

`build` writes JSONL to `output_dir` and (for t3b) the generation report.
`report` reads a dataset back and prints the confound table: C1 correlation,
C2 per-token MI top-k + masked-classifier AUC, C3 name MI + chi-square, C4
format counts, C5 counterfactual-integrity pass rate. Exit non-zero if any
threshold is violated (so it doubles as a CI gate).

---

## 11. Test suite

`tests/test_generator.py`, run against a scaled-down deterministic build from
`configs/generator.test.yaml`.

| Test | Assertion |
|---|---|
| `test_counterfactual_integrity` | every record's twin differs only in the agent↔trait / agent↔correctness map; all other fields byte-identical (acceptance a) |
| `test_position_trait_corr` | `|point-biserial(position, trait_level)| < 0.02` (acceptance b) |
| `test_name_trait_mi` | `MI(name, trait_level) < 0.01` bits; chi-square independence not rejected (acceptance c) |
| `test_format_balance` | T1/T2 exactly 50/50 same/split; T3 all `n/a` (acceptance d) |
| `test_lexical_leakage_masked_clf` | trait-slot-masked bag-of-words classifier CV AUC `< 0.55` |
| `test_determinism` | same seed ⇒ identical JSONL bytes for T1/T2/T3a |
| `test_record_schema` | every record validates against `Record`; required fields present; `trait_level` in the per-variant range |
| `test_t1_positive_control` | T1 records are well-formed minimal pairs with single-token traits |
| `test_t3b_validation` | with a **mocked** backend: gold/distractor presence enforced; trait-word blocklist enforced; regenerate-on-fail caps at 5 |

`test_t3b_validation` uses a stub backend — no server in CI. A separate
`-m integration` test hits a real `base_url` when one is configured, skipped
otherwise.

---

## 12. Risks (Phase-0-specific)

| Risk | Mitigation |
|---|---|
| MMLU domain bucketing is coarse / uneven across the four domains | static mapping table reviewed by hand; report per-domain counts; allow `domains` subset in config |
| T3b reject rate high (model won't obey correctness or leaks trait words) | generation report surfaces it; blocklist extendable; fall back to larger T3a share; try the other Qwen model |
| `split_sentence` wording accidentally reintroduces adjacency | pin to Feng & Steinhardt template; C2 masked-classifier check catches residual leakage |
| QA items with non-unique or fuzzy gold answers pollute T3 correctness labels | filter at load; normalization + substring match in validation; log drops |
| vLLM `seed` not fully deterministic across server restarts | on-disk cache is the source of truth for reproducibility; document that T3b rebuilds require the cache |
| Name pool too small ⇒ residual name–trait MI | pool >= 200 personal names; builder cycles rather than pure-samples; C3 test gates it |

---

## 13. Definition of done

- `uv sync` installs; `ruff` clean.
- `personabind build --variant all` produces the four JSONL files (T3b needs a
  running vLLM server; T1/T2/T3a do not).
- `personabind report --all` prints the confound table with every threshold
  satisfied and exits 0.
- `pytest` green, including `test_t3b_validation` with the mocked backend.
- README documents: how to run a vLLM server for T3b, how to build, how to
  read the report.
- User has inspected sample records from each variant and confirmed they look
  right.
- **Stop.** Phase 1 is a separate cycle.
