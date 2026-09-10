# Perceived Persona Binding and Steering in Multi-Agent LLM Systems

**A research plan.**
Version 1.0 — September 2026

---

## 0. How to use this document

This serves two audiences.

**For you (study):** Sections 1–4 give the conceptual frame, the novelty claim, and the literature map. Read those first, in order. Section 5 is the model/stack decision with reasoning you may want to disagree with.

**For Claude Code (implementation):** Sections 6–12 are written as actionable specs. Each phase has a `TASK` block, a file layout, and an acceptance criterion. Hand it Section 5 (stack) plus whichever phase you're on. Do not hand it the whole document at once — the phases are sequential and gated, and giving it all of them invites building Phase 4 machinery before Phase 1 has told you whether Phase 4 is worth building.

**The single most important structural point:** this plan front-loads its riskiest assumption. The instinct is to start building probes and steering vectors because that's the interesting part. Resist it. The assumption that gates everything is cheap to test and is not the one you'd naturally test first.

---

## 1. The research question

In a group of LLM agents deliberating toward a decision, does agent *i* internally represent a model of agent *j*'s persona — specifically *j*'s expertise, reliability, and credibility — and does that internal representation causally mediate how much *i* is influenced by *j*?

This sits inside a larger program on how a single agent or a small committed minority sways group decisions under bounded compute. Perceived persona is the proposed mediator: assigned persona → perceived persona → influence.

Why the mediator matters. Without it you cannot distinguish two very different failure modes of a persona-based influence attempt:

1. The persona **never landed** — targets didn't perceive the influencer as credible.
2. The persona **landed and didn't matter** — they perceived it fine, it just didn't move them.

These have opposite implications for anyone trying to understand or defend against social influence in agent populations, and behavioral data alone cannot separate them.

---

## 2. The hypothesis chain

Stated as a chain so each link can fail independently and be tested separately:

| # | Link | Status in literature |
|---|---|---|
| H1 | Discrete attributes bind to entities in context via a recoverable mechanism | **Established** (Feng & Steinhardt 2024; Dai et al. 2024; Gur-Arieh et al. 2026) |
| H2 | Graded *social* traits bind the same way discrete attributes do | **Untested — this is the gate** |
| H3 | The bound trait is linearly readable from agent *i*'s activations | Plausible; adjacent results support it |
| H4 | Binding survives N > 2 agents and long transcripts | Partially negative — positional binding degrades |
| H5 | The read-out direction can be made causal (steering changes behavior) | **Contested** — belief–action gap |
| H6 | Perceived persona mediates social influence | **Unstudied** |
| H7 | In hybrid-attention models, binding retrieval localizes to the full-attention layers | **Unstudied** — see §5.1 |
| H8 | Binding fidelity degrades faster with agent count in hybrid models (fixed-size recurrent state) | **Unstudied** — see §5.1 |

H7 and H8 are a separable side-contribution from the Qwen3.5-9B arm. They do not depend on H6 and can be published independently of the influence-mediation story.

H2 is the gate. Every binding result you would build on used discrete, usually single-token attributes: a country, an object, a box. "Reliable" is graded, abstract, and often inferred rather than stated-and-retrieved. If graded social traits don't bind, nothing downstream works.

Testing H2 costs about three weeks with code that already exists. Testing it after building probe and steering infrastructure costs about six months.

---

## 3. What is novel here

Precisely, so you know what to claim:

**Already done, do not re-claim:**
- Probing an LLM's internal model of its *conversational partner* — TalkTuner (2406.07882) reads user age/gender/education/SES.
- Probing warmth/competence impressions of the prompt author — Artificial Impressions (2510.08915).
- Probing a *third-party observer* model for a conversation participant's Big-5 personality — Jaipersaud, Krueger & Lubana (2508.05625).
- Steering a generic expertise/authority direction — the authority-bias papers (2601.13433, 2607.00415).
- Entity–attribute binding and pointer-based retrieval — the binding literature.

**Genuinely open:**
1. Whether *graded social traits* bind to agents the way discrete attributes bind to entities. (H2)
2. Probing a **participating** agent — one that will then act — for its read of another agent, mid-interaction. All prior work is observer-mode or single-prompt.
3. Assigning **different values of the same trait to different entities simultaneously** and steering one without disturbing the others.
4. Using a probed inter-agent belief as a dense signal in a multi-agent learning setting.

Claims 1 and 2 are the defensible core. Claims 3 and 4 are the ambitious extension.

---

## 4. Literature anchors

Grouped by what you need them for. Read the starred ones before starting.

### Binding (the foundation)
- ★ **Feng & Steinhardt**, "How do Language Models Bind Entities in Context?" ICLR 2024, arXiv **2310.17191**. Binding ID mechanism, CAPITALS task, causal mediation. Code: `github.com/jiahai-feng/binding-iclr`
- ★ **Gur-Arieh et al.**, "Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context," ICLR 2026, arXiv **2510.06182**. Three mechanisms — positional, lexical, reflexive — and counterfactual patching to separate them. Llama/Gemma/Qwen, 2B–72B, ten binding tasks. Code: `github.com/yoavgur/mixing-mechs`
- **Dai et al.**, "Representational Analysis of Binding in Language Models," EMNLP 2024, arXiv **2409.05448**. Ordering IDs — binding is by position index.
- **Prakash et al.**, "Language Models Use Lookbacks to Track Beliefs," arXiv **2505.14685**. Low-rank OI pointers, dereferenced at query token.
- **Feng, Russell & Steinhardt**, "Monitoring Latent World States with Propositional Probes," ICLR 2025, arXiv **2406.19501**.
- **Oh & Demberg**, retrieval-conditioned rebinding circuit, arXiv **2606.08644**.

### Probing methodology
- ★ **Chen et al. (TalkTuner)**, "Designing a Dashboard for Transparency and Control of Conversational AI," arXiv **2406.07882**. **The reading-probe vs control-probe distinction is the single most important methodological transfer.** Control probes hit 0.93–1.00 intervention success vs reading probes' 0.80–0.93.
- **Marks & Tegmark**, "The Geometry of Truth," COLM 2024, arXiv **2310.06824**. Difference-in-means is more causally implicated than logistic regression.
- **Deas & McKeown**, "Artificial Impressions," EMNLP 2025, arXiv **2510.08915**. Warmth×competence probing; verbal reports unreliable, probes decodable.
- **Elazar et al.**, "Amnesic Probing," TACL 2021. Decodability ≠ causal use.
- **Hewitt & Liang**, "Designing and Interpreting Probes with Control Tasks," EMNLP 2019.

### Persona and social perception
- ★ **Chen, Arditi, Sleight, Evans & Lindsey**, "Persona Vectors," arXiv **2507.21509**. Extraction pipeline: 5 contrastive system-prompt pairs, 40 questions split 20/20, difference-in-means over response-averaged activations. Code: `github.com/safety-research/persona_vectors`
- **Ayesh, Mohammad & Ousidhoum**, W&C-Sent, ACL 2026, arXiv **2601.06316**. Warmth (trust + sociability) × competence, annotated *toward specific targets*. Code: `github.com/nedjmaou/W_C_Sent`
- **Authority bias**: arXiv **2601.13433** (expertise tiers + steering vector) and arXiv **2607.00415** (mechanistic; **disagrees** — finds late-layer erasure resisting mean-vector intervention). Read both; the disagreement is informative.

### Steering
- **Lee et al. (CAST)**, "Programming Refusal with Conditional Activation Steering," ICLR 2025, arXiv **2409.05907**. Condition vectors as gates. Code: `github.com/IBM/activation-steering`
- **Nguyen et al. (MAT-Steer)**, ACL 2025, arXiv **2502.12446**. Token selection matters more than token count.
- **Wu et al. (AxBench)**, 2025. **Prompting beats SAE and LoRA steering.** Take seriously.
- ★ **Makelov, Lange & Nanda**, "Is This the Subspace You Are Looking For?" arXiv **2311.17030**. The interpretability illusion — a subspace intervention can work via a dormant, causally disconnected pathway. **Read before you believe any positive steering result.**
- Sycophancy direction entanglement: arXiv **2606.11205**, **2607.07003**.

### The negative results you must internalize
- ★ **Sobotka, Karabag & Topcu**, "Why Do LLMs Struggle in Strategic Play? Broken Links Between Observations, Beliefs, and Actions," arXiv **2605.00226**. Internal beliefs more accurate than verbal reports, but steering them changes actions only ~70% (normal-form games) / ~50% (gpt-oss, poker). **Note: they steered with raw reading-probe weights at one layer — the weakest possible intervention.** This is why Section 9 insists on control probes.
- **Schmied et al.**, "LLMs are Greedy Agents," arXiv **2504.16078**. 87% correct rationales, 64% greedy actions anyway.
- **Chen et al.**, "Reasoning Models Don't Always Say What They Think," arXiv **2505.05410**. CoT self-reports are unfaithful; don't use them as labels.

---

## 5. Model and stack decisions

### 5.1 Model selection — read this carefully, it is not obvious

You mentioned Qwen3-4B and an interest in newer Qwen models up to 20B, specifically 14B and Qwen3.5-9B.

**Qwen family as of September 2026:**

| Family | Released | Dense sizes ≤20B | Architecture |
|---|---|---|---|
| Qwen3 | Apr 2025 | 0.6B, 1.7B, 4B, 8B, **14B** | Standard dense transformer, full softmax attention |
| Qwen3.5 | — | 0.8B, 2B, 4B, **9B** | Hybrid: 3:1 Gated DeltaNet (linear) : Gated Attention (full) |
| Qwen3.6 | Apr 2026 | none (27B dense only) | Hybrid attention |
| Qwen3.8 | Aug 2026 | none (27B dense only) | 64 layers, 262K ctx |

**First correction:** there is no Qwen3.8-9B. Qwen3.8 ships only a 27B dense model, above your ceiling. Qwen3.6 likewise. The 9B you're thinking of is **Qwen3.5-9B**.

**Second correction — I was too blunt earlier.** My initial advice was "avoid Qwen3.5 entirely." Having looked at the actual architecture, that was wrong. Qwen3.5-9B is worth running, but as a *deliberate third arm answering its own question*, not as a substitute for the mainline models.

### What Qwen3.5 actually does

`model_type: "qwen3_5"`, inheriting from `qwen3_next`. It interleaves two token-mixing layer types, controlled by `layer_types` in the config with a `full_attention_interval` parameter (default 4):

```
[linear, linear, linear, full_attention, linear, linear, linear, full_attention, ...]
```

So ~75% Gated DeltaNet (linear attention, fixed-size recurrent state) and ~25% standard softmax attention with GQA and RoPE. For Qwen3.5-9B that's roughly 32 layers = 24 linear + 8 full. Every dense Qwen3.5 model (27B, 9B, 4B, 2B) uses this same 3:1 pattern.

Three architectural details matter for your work:

1. **The full attention layers are explicitly the retrieval layers.** They "provide global context and strong retrieval capability" while linear layers provide O(1)-per-token efficiency. Binding lookback *is* a retrieval operation — dereferencing a pointer to fetch a bound attribute.
2. **Linear layers have no KV cache.** They carry `conv_states` (batch, conv_dim, 4) and `recurrent_states` (batch, num_v_heads, key_head_dim, value_head_dim) — **fixed size regardless of sequence length**.
3. **Gated Attention applies sigmoid output gating before the residual add**, which is reported to eliminate attention sinks and massive activations.

### Why this is interesting rather than disqualifying

Point 1 gives you a sharp, pre-registerable hypothesis that a standard transformer cannot give you:

> **H7: Binding retrieval localizes to the full-attention layers.** In Qwen3.5-9B, binding lookback should be causally mediated at the ~8 `full_attention` layers and largely absent at the 24 linear layers.

In Qwen3-8B you must search all 36 layers and hope for a clean localization story. In Qwen3.5-9B the architecture *pre-partitions* the search space, and a positive result is dramatically cleaner: "binding happens here, in the 25% of layers that can do retrieval, and nowhere else."

Point 2 gives you a second, independent prediction that bears directly on Phase 2:

> **H8: Binding fidelity degrades faster with agent count in hybrid models.** The linear layers compress all prior context into a fixed-size recurrent state. N agents' bindings must share that fixed budget, whereas a full-attention KV cache grows with sequence length. So the (N × transcript length) fidelity surface should have a sharper ceiling in Qwen3.5-9B than Qwen3-8B.

If H8 holds, that is a genuinely useful result for anyone deploying hybrid-attention models in multi-agent systems — a concrete capacity limit on how many distinct agents such a model can track.

Nobody has done binding mechanistic interpretability on hybrid linear-attention models. That is an unoccupied niche adjacent to work you're already doing.

### Concrete assignment

| Role | Model | Why |
|---|---|---|
| Dev / fast iteration | **Qwen3-4B** | Your existing setup. Run every experiment here first. |
| Primary | **Qwen3-8B** | Main results. Standard dense, directly comparable to the binding literature. |
| Scale check | **Qwen3-14B** | Under 20B, dense, same architecture as 8B — isolates scale from architecture. |
| **Architecture arm** | **Qwen3.5-9B** | Hybrid. Tests H7 and H8. Roughly size-matched to Qwen3-8B, so architecture is close to the only varying factor. |
| Cross-family control | **Llama-3.1-8B** | Persona vectors and authority steering published on it — validates your pipeline against known results. |
| Optional | Qwen2.5-7B | Persona vectors published (layer 20 evil/sycophancy, layer 16 hallucination) — calibration anchor. |

The 8B / 9B pairing is the point. Qwen3-8B and Qwen3.5-9B are close enough in size that a binding difference between them is attributable to architecture rather than capacity. That is a cleaner comparison than you could construct deliberately.

### Sequencing — this matters

**Do not run the hybrid arm in Phase 1.** The gate question (do graded social traits bind at all?) must be answered on a standard architecture first, or a null is uninterpretable — you won't know whether traits don't bind or whether hybrid attention broke your measurement.

- **Phase 1:** Qwen3-4B, Qwen3-8B only.
- **Phase 1b (new, ~1 week):** if and only if Phase 1 clears, add Qwen3.5-9B and test H7.
- **Phase 2:** run the fidelity surface on both Qwen3-8B and Qwen3.5-9B; test H8.
- **Phases 3–5:** primary results on Qwen3-8B/14B. Treat the hybrid as a robustness check, not a load-bearing model.

### Risks specific to the hybrid arm

**Verify tooling before committing.** Check that nnsight/nnterp can hook `qwen3_5` layers and that you can read the residual stream at both layer types. If `nnterp` doesn't recognize the model type, fall back to raw `transformers` forward hooks. Do this on day one of Phase 1b — a half-day check that can save a fortnight.

**Patching a linear layer is not the same operation.** In a full-attention layer you patch a per-token KV entry. In a Gated DeltaNet layer there is no per-token cache — there is a fixed-size `recurrent_states` tensor that has already compressed all prior tokens. Patching it is a *different intervention with different semantics*, and results are not directly comparable to full-attention patching. Options:
- Restrict causal interventions to the `full_attention` layers, where patching means what it means everywhere else. **Recommended.** It also happens to be exactly where H7 predicts the action is.
- Patch residual-stream activations at layer boundaries rather than inside the mixing operation — architecture-agnostic and comparable across both model types. **Also recommended, and the better default for probes.**
- Patch `recurrent_states` directly. Novel, interesting, but do not put it on the critical path.

**Different activation geometry.** The sigmoid output gating suppresses attention sinks and massive activations. Your reference set already flags that Gemma's post-norms make its residual stream non-spherical, complicating norm-matched steering; expect Qwen3.5 to have its own geometry. **Measure the activation-norm distribution per layer before doing any norm-matched steering**, and don't port a steering coefficient from Qwen3-8B to Qwen3.5-9B.

**A caveat on 4B.** Feng & Steinhardt found the binding ID mechanism in "every sufficiently large model" they tested — implying it may be absent below some scale. Mixing Mechanisms covered 2B, which is reassuring. But if Phase 1 fails at 4B, **re-run at 8B before concluding anything.** A null at 4B is not a null.

**Thinking mode.** Qwen3 has hybrid thinking. For all probing work, **disable it** (`enable_thinking=False`). You want clean single-pass activations at deterministic token positions. Thinking traces introduce variable-length reasoning between the prompt and the answer, which destroys positional comparability across examples. Study the thinking traces later as a separate question, not as part of the core probe pipeline.

### 5.2 Stack

```
Python 3.11
torch, transformers
nnsight + nnterp          # activation access; nnterp gives a unified interface
                          # and preserves exact model behavior
scikit-learn              # probes (logistic + LDA/difference-in-means)
numpy, scipy, pandas
matplotlib / seaborn
pytest
```

Optional later: `pyvene` / CausalAbstraction for DAS, `pyreft` for LoReFT.

**Do not** reach for SAEs. AxBench found simple baselines beat them for steering, sanity-check work found they recover ~9% of ground-truth features, and Gemma Scope doesn't cover Qwen anyway. You know which concepts you want; you don't need unsupervised discovery.

### 5.3 Hardware

Qwen3-4B fp16 ≈ 8GB, 8B ≈ 16GB, 14B ≈ 28GB, Qwen3.5-9B ≈ 18GB. A single 24GB card handles 4B, 8B and 9B comfortably; 14B needs 40GB+ or 8-bit. Everything in Phases 0–3 is cheap — you are running forward passes and fitting linear models, not training.

**First thing to do in code:** never hardcode layer indices or hidden dims, and for the hybrid model resolve which layers are which type.

```python
from transformers import AutoConfig

MODELS = [
    "Qwen/Qwen3-4B", "Qwen/Qwen3-8B", "Qwen/Qwen3-14B",
    "Qwen/Qwen3.5-9B",          # verify exact repo id on the Hub
]

for m in MODELS:
    c = AutoConfig.from_pretrained(m, trust_remote_code=True)
    print(f"\n{m}")
    print("  model_type :", c.model_type)
    print("  layers     :", c.num_hidden_layers)
    print("  hidden     :", c.hidden_size)
    print("  heads      :", getattr(c, "num_attention_heads", None))

    # hybrid-specific: which layers can actually do retrieval
    layer_types = getattr(c, "layer_types", None)
    if layer_types is not None:
        full = [i for i, t in enumerate(layer_types) if "full" in t]
        print("  full-attn  :", full, f"({len(full)}/{len(layer_types)})")
    else:
        print("  full-attn  : all (standard transformer)")
```

For Qwen3.5, the `full` indices printed above are your H7 candidate set — restrict causal interventions to them, and treat the linear layers as a separate, harder question.

Published layer choices (TalkTuner layer 26, persona vectors layer 16/20) are model-specific. Re-localize with a layer sweep for every model you use. **Never port a steering coefficient across architectures** — measure per-layer activation norms first.

---

## 6. Phase 0 — Rig and generator

**Duration:** ~2 weeks
**Goal:** a task generator producing controlled agent×trait binding data at three difficulty levels.

### Design

Fork `mixing-mechs/grammar/schemas.py`. It defines binding tasks declaratively, which makes agent×trait an edit rather than a rewrite, and it plugs into the counterfactual-patching machinery you'll need in Phase 2.

The base template, adapted from the CAPITALS task:

```
Context: {Agent_0} and {Agent_1} are known to be {Trait_0} and {Trait_1} respectively.
Question: How reliable is {query_agent}?
Answer: {query_agent} is
```

**Three task variants, sharing structure:**

- **T1 — discrete.** Trait is a single token from a small closed set: `expert` / `novice`. This is your positive control. It should reproduce known binding results. If T1 fails, your rig is broken, not the hypothesis.
- **T2 — graded.** Four-tier expertise scale, borrowed from the authority-bias papers: `first-year student` → `third-year student` → `senior practitioner` → `board-certified expert`. Tests whether gradedness breaks binding.
- **T3 — inferred.** The trait is never stated. It is *demonstrated* across several turns — one agent gives accurate, well-calibrated answers, another gives confident wrong ones. This is what your real experiments look like.

The gap between T1 and T3 is the finding.

### Confounder controls — build these in from day one

1. **Same-sentence confounder.** Feng & Steinhardt introduced an alternate prompt format specifically because in the original the entity always appeared in the same sentence as its attribute — the alternate format showed binding ID is not merely a syntactic property. Generate both formats.
2. **Lexical leakage.** Trait-associated vocabulary must not correlate with the label beyond the trait token itself. TalkTuner engineered topic–label correlation down to 0.0–1.3%; match that discipline. Measure and report it.
3. **Position counterbalancing.** Every (agent, trait) assignment must appear in both orders, equally often.
4. **Name randomization.** Sample agent names from a large pool; never let a name correlate with a trait.

### File layout

```
personabind/
├── generator/
│   ├── schemas.py           # task definitions (forked from mixing-mechs)
│   ├── traits.py            # trait vocabularies for T1/T2/T3
│   ├── names.py             # agent name pool
│   └── build.py             # emits JSONL datasets
├── data/
│   ├── t1_discrete.jsonl
│   ├── t2_graded.jsonl
│   └── t3_inferred.jsonl
└── tests/
    └── test_generator.py
```

Each record:

```json
{
  "id": "t1_000001",
  "variant": "t1_discrete",
  "context": "...",
  "question": "...",
  "agents": [
    {"name": "Alice", "position": 0, "trait": "expert", "trait_level": 1},
    {"name": "Bob",   "position": 1, "trait": "novice", "trait_level": 0}
  ],
  "query_agent": "Alice",
  "answer": "expert",
  "counterfactual_id": "t1_000002",
  "format": "same_sentence" | "split_sentence"
}
```

The `counterfactual_id` field is essential — it points at the minimal pair differing only in the agent↔trait assignment. Phase 2 patching needs it.

> **TASK (Claude Code):** Implement `personabind/generator/`. Produce ≥5,000 examples per variant with exact minimal-pair counterfactuals, position counterbalancing, and both prompt formats. Write `tests/test_generator.py` asserting: (a) every record has a valid counterfactual whose only difference is the agent↔trait mapping; (b) position–trait correlation < 0.02; (c) name–trait mutual information ≈ 0; (d) each variant is balanced across formats.
>
> **Acceptance:** tests pass; a report script prints the measured confound statistics.

---

## 7. Phase 1 — Does it bind? (THE GATE)

**Duration:** ~3 weeks
**Goal:** determine whether graded and inferred social traits bind to agents.

Run the standard binding battery on T1 → T2 → T3, at Qwen3-4B and Qwen3-8B:

1. **Behavioral accuracy.** Can the model answer the query correctly at all? If not, nothing else means anything. Floor: >90% on T1.
2. **Factorizability.** Patch the entity representation and the attribute representation independently. If binding is factorized, swapping the binding ID swaps the answer.
3. **Position test.** Vary where entities and attributes appear; does the binding follow the entity or the slot?
4. **Mean intervention.** Compute mean binding-ID vectors per position, add/subtract them, measure whether the retrieved attribute changes.
5. **Baseline.** Random direction of matched norm. **Always.**

### Kill criteria

| Result | Interpretation | Action |
|---|---|---|
| T1 fails | Rig broken, or model too small | Debug; re-run at 8B before concluding |
| T1 passes, T2 fails | **Graded traits don't bind** | **Stop the steering program.** Write this up. |
| T1, T2 pass; T3 fails | Stated traits bind, inferred don't | Reshape project around *stated* personas |
| All pass | Proceed to Phase 2 | — |

The T2-fails branch is still a paper. "Social attributes do not bind like discrete attributes in LLMs" has implications for every multi-agent interpretability claim in the literature, and it is a clean, well-controlled negative result. Plan to write it either way — that's what makes this phase safe to run first.

> **TASK (Claude Code):** Implement `personabind/binding/battery.py` running the five tests above across variants × models × layers. Emit a results table and per-test plots. Include the norm-matched random baseline in every test. Do not report any intervention effect without its baseline.
>
> **Acceptance:** T1 reproduces qualitative binding-ID behavior (factorizable, position-sensitive, mean intervention effective above baseline). Results for T2/T3 reported with confidence intervals over ≥1,000 examples.

---

## 7b. Phase 1b — The hybrid-architecture arm (conditional)

**Duration:** ~1 week
**Precondition:** Phase 1 cleared on Qwen3-8B. Do not run this otherwise — a null on an unfamiliar architecture is uninterpretable.
**Goal:** test H7, that binding retrieval localizes to full-attention layers in Qwen3.5-9B.

### Day 0 — tooling check (half a day, do this first)

Before any science, confirm you can instrument the model at all:

1. Load `Qwen3.5-9B` and print `config.layer_types`; record the full-attention indices.
2. Confirm nnsight/nnterp hooks fire on both layer types. If nnterp doesn't recognize `qwen3_5`, fall back to raw `transformers` forward hooks on `model.layers[i]`.
3. Confirm you can read residual-stream activations at layer boundaries for *both* layer types. This is the architecture-agnostic readout and your default.
4. Confirm behavioral accuracy on T1 ≥ 90%. If the model can't do the task, stop.

If any of these fail, drop the arm. It is a bonus contribution, not a dependency.

### The experiment

Run the Phase 1 battery on T1 and T2, but sweep **layer type** as the primary factor:

- Causal mediation at each `full_attention` layer.
- Causal mediation at each `linear` layer, via residual-stream patching at layer boundaries (comparable across types).
- Compare the two distributions.

**Restrict in-layer causal interventions to full-attention layers.** There, patching a KV entry means what it means everywhere else in the literature. In a Gated DeltaNet layer there is no per-token cache — only a fixed-size `recurrent_states` tensor that has already compressed all prior tokens, so patching it is a different operation with different semantics and is not comparable. Patching `recurrent_states` directly is a genuinely interesting experiment; keep it off the critical path.

### Predictions and interpretations

| Result | Reading |
|---|---|
| Mediation concentrated at full-attention layers | **H7 confirmed.** Clean localization story; strong result. |
| Mediation spread evenly across both types | H7 rejected — binding is distributed, or the residual stream carries it independently of mixing type. Still publishable, and more surprising. |
| No mediation anywhere, but behavior is correct | Measurement problem, or binding works differently enough that your battery doesn't detect it. Investigate before claiming anything. |
| Behavior itself fails on T1 | Drop the arm. |

> **TASK (Claude Code):** Extend `personabind/binding/battery.py` with a `layer_type` dimension resolved from `config.layer_types`. Add `personabind/common/hybrid.py` providing (a) full-attention index resolution, (b) an architecture-agnostic residual-stream patcher that works on both layer types, (c) a guard that raises on any attempt to run a KV-style patch on a linear layer. Emit mediation-by-layer plots colour-coded by layer type.
>
> **Acceptance:** the tooling check passes; mediation-by-layer-type is reported with the norm-matched baseline; the guard is unit-tested.

---

## 8. Phase 2 — Binding integrity under pressure

**Duration:** ~3 weeks
**Goal:** map where binding degrades as a function of agent count and transcript length.

Assuming Phase 1 clears. This phase produces the constraint that determines how large your real experiments can be.

**Experiments:**

1. **Two agents, opposite traits, position counterbalanced.** Does the probe/patch track the *agent* or the *position*?
2. **Mid-transcript swap.** Agents A and B exchange trait values partway through. Does the model rebind?
3. **Agent-count sweep.** N = 2, 3, 5, 10, 15, 25.
4. **Transcript-length sweep.** Short (~200 tokens) → long (~4,000 tokens).
5. **Mechanism decomposition.** Use the Mixing Mechanisms counterfactual patching to separate positional, lexical, and reflexive binding signals.

Point 5 is the one that matters most. Mixing Mechanisms found the positional mechanism **breaks down in complex settings**, with lexical and reflexive mechanisms also driving behavior. In a 20-round transcript with 15 speakers, a positional binding signal is exactly what you'd expect to degrade — and "my probe reads position, not agent" is the failure that would silently invalidate everything downstream.

6. **Architecture comparison (H8).** Run the full sweep on both **Qwen3-8B** (full attention) and **Qwen3.5-9B** (hybrid), size-matched.

Point 6 is the H8 test. The linear layers compress all prior context into a fixed-size `recurrent_states` tensor, so N agents' bindings compete for a fixed budget, whereas a full-attention KV cache grows with sequence length. Prediction: the hybrid model's fidelity surface has a **sharper ceiling** in N, and possibly in length. Run Qwen3-14B too if you can, so you can separate a capacity effect from an architecture effect — if 14B looks like 8B and 9B-hybrid looks different, it's architecture.

**Deliverable:** binding-fidelity surfaces over (N agents × transcript length), one per model. This is independently publishable, it tells you the licensed size of your Phase 6 experiments, and — if H8 holds — it gives a concrete capacity limit on how many distinct agents a hybrid-attention model can track, which matters to anyone deploying these in multi-agent systems.

> **TASK (Claude Code):** Implement `personabind/binding/integrity.py`. Produce a fidelity surface per model as a heatmap plus a table of the (N, length) region where fidelity > 0.8. Use mixing-mechs counterfactual patching to attribute residual fidelity to positional / lexical / reflexive mechanisms. Emit a side-by-side Qwen3-8B vs Qwen3.5-9B comparison with the difference surface and a fitted N-ceiling per model.
>
> **Acceptance:** a defensible answer to "what is the largest N and longest transcript at which agent-trait binding remains reliable?" — separately for full-attention and hybrid architectures, with the ceilings quantified.

---

## 9. Phase 3 — Reading and control probes

**Duration:** ~3 weeks
**Goal:** two distinct probes per trait, plus the angle between them.

Follow the TalkTuner protocol deliberately — this is the fix for what Broken Links got wrong.

**Reading probe.** Trained on the last-token representation of an appended query: `"I think {agent}'s reliability is"`. Logistic regression, L2 regularized, layer sweep.

**Control probe.** Trained separately, on the representation at the *ending token of the last message* — the representation the model actually uses to generate. Same data, different read position and different objective (optimize for intervention success, not classification accuracy).

**Also fit a difference-in-means direction** for each trait. Difference-in-means directions are more causally implicated than logistic-regression probes; you want both.

**Report the angle** between reading direction and control/difference-in-means direction. A large angle is your early warning for the readout–mediator gap. Measuring it *before* you build steering infrastructure is the point.

**Validation, all mandatory:**
- Control tasks (Hewitt & Liang) — random-label probe to bound probe capacity.
- Negative control — probe for an agent who hasn't spoken yet. Should read null.
- Held-out names, held-out trait wordings.
- Behavioral floor — ask the model directly what it thinks of agent *j*, correlate with the probe. The interesting regime is *partial* correlation plus predictive power the verbal report lacks. If the probe matches the verbal report perfectly, you built an expensive way to ask a question.

> **TASK (Claude Code):** Implement `personabind/probes/` with `reading.py`, `control.py`, `diffmeans.py`, `validate.py`. Layer sweep, 5-fold CV, per-model. Emit accuracy-by-layer curves, the reading/control angle, and all validation results.
>
> **Acceptance:** reading probe > 85% on T1 held-out at some layer; control probe measurably better at intervention than the reading probe; all negative controls at chance.

---

## 10. Phase 4 — Two intervention arms, in parallel

**Duration:** ~4 weeks
**Goal:** manipulate perceived persona and measure the effect.

Run both arms simultaneously. Do not treat Arm A as a fallback — AxBench found prompting beats steering, and Arm A is what carries the paper if Arm B nulls out.

### Arm A — prompt manipulation (primary, low risk)

Manipulate agent *j*'s stated persona in the transcript. Measure the probe as the **mediator**. Measure deference as the **outcome**. This gives you the full mediation chain and depends on nothing risky.

### Arm B — activation steering (secondary, high risk / high reward)

CAST condition vector gating on "currently reasoning about agent *j*", control-probe direction as the behavior vector.

- Steer at **selected tokens**, not all tokens and not last-token-only (MAT-Steer: token *selection* beats token *count*).
- For N > 2, orthogonalize the per-agent vectors or apply a Gram-matrix correction; check the condition number first.
- Sweep magnitude, but watch for off-manifold degradation — there is a narrow effective window.

### Mandatory controls for Arm B

Without both of these you cannot distinguish a result from an interpretability illusion:

1. **Norm-matched random direction.** If a random direction of the same norm produces a comparable behavioral shift, you have an illusion.
2. **Off-target measurement.** Does steering *j*'s credibility also move *k*'s? Does it move general agreeableness? Report on-target vs off-target ratio. If off-target ≥ 50% of on-target, the direction is entangled.
3. **Capability retention.** Track a general benchmark; if it drops >5%, you are degrading the model, not steering a belief.

> **TASK (Claude Code):** Implement `personabind/steering/` with `cast_gate.py`, `apply.py`, `controls.py`. Every steering run must automatically emit its norm-matched baseline and off-target measurements in the same results record. Make it structurally impossible to report a steering effect without its controls.
>
> **Acceptance:** on-target effect exceeds norm-matched baseline by a margin that survives multiple-comparison correction; off-target ratio reported.

---

## 11. Phase 5 — Behavior and mediation

**Duration:** ~4 weeks

**Task:** a deference setup with ground truth — hidden-profile, or a committee vote where exactly one agent holds the correct answer.

**Outcome measure: logit shift, not argmax flip.** In a small discrete action space, a large belief change may not cross the decision threshold. Broken Links' own Figure 12 shows a first-item positional bias where P(option A) exceeds 0.9 when the best response is listed first — measuring argmax flips would have hidden real effects. Randomize option order.

**Analysis:**
- Mediation model: assigned persona → probed perceived persona → deference. Report the indirect effect.
- If Arm B worked, the intervention-on-mediator version is much stronger evidence.
- Report the compliance/internalization split: public stance shift with vs without internal probe shift.

---

## 12. Phase 6 — Live multi-agent

**Duration:** open

Scale to your 5–25 agent sway paradigm — but only within the (N × length) region Phase 2 licensed. Optionally validate against Avalon-NLU-style per-player belief annotations for external ground truth.

---

## 13. Repository layout

```
personabind/
├── README.md
├── pyproject.toml
├── configs/
│   ├── models.yaml           # model IDs, resolved layer counts, dtypes
│   └── experiments/          # one YAML per experiment
├── personabind/
│   ├── generator/            # Phase 0
│   ├── binding/              # Phases 1–2
│   ├── probes/               # Phase 3
│   ├── steering/             # Phase 4
│   ├── behavior/             # Phase 5
│   ├── agents/               # Phase 6
│   └── common/
│       ├── activations.py    # nnsight/nnterp wrappers
│       ├── controls.py       # norm-matched baselines, off-target
│       └── stats.py
├── scripts/                  # thin CLI entry points
├── results/                  # JSONL, one record per run
├── notebooks/                # analysis only, never source of truth
└── tests/
```

**Engineering principles for Claude Code:**

1. **Every intervention result carries its baseline in the same record.** Not a separate run, not an optional flag. Same record.
2. **No hardcoded layers or dims.** Resolve from config at load.
3. **Every experiment is a config file**, so runs are reproducible and diffable.
4. **Results are append-only JSONL.** Notebooks read; they never write.
5. **Seed everything**, and log seeds.
6. **Fail loudly on silent shape mismatches** — the most common bug in activation work is patching the wrong token position and getting plausible-looking numbers.

---

## 14. Risk register

| Risk | Probability | Detected by | Mitigation |
|---|---|---|---|
| Graded traits don't bind | Medium | Phase 1 (3 wks) | Publish as negative result |
| Probe tracks position not agent | **High** | Phase 2 | Mechanism decomposition; constrain N and length |
| Qwen3-4B too small for binding | Medium | Phase 1 | Re-run at 8B before concluding |
| Steering works via illusion | **High** | Norm-matched control | Never report without it |
| Off-target spillover across agents | High | Phase 4 controls | Orthogonalize / Gram correction |
| Belief–action gap caps effects | High | Phase 5 | Arm A carries the result; measure logit shift |
| Trait vocabulary leaks lexically | Medium | Phase 0 stats | Measure and report correlation |
| nnsight can't hook `qwen3_5` | Medium | Phase 1b day 0 | Raw transformers hooks; else drop the arm |
| Linear-layer patching not comparable | **Certain** | By construction | Restrict causal work to full-attn layers; use residual-boundary patching |
| Steering coefficients don't transfer to hybrid | High | Norm distribution check | Re-tune per architecture; never port |

The three most likely project-killers — graded traits not binding, probes tracking position, and steering-via-illusion — are all caught in Phases 1–2 and by one control. None of them are caught by starting with steering. That asymmetry is the entire argument for this phase ordering.

---

## 15. Publication strategy

Structure so the ambitious paper failing doesn't sink the modest one.

**Paper A — "Do social attributes bind to agents?"** (Phases 1–2, ~4 months)
Safe. Publishable on negative results. Contribution: the binding-fidelity surface over agent count and context length, plus the discrete/graded/inferred comparison. Useful to anyone doing multi-agent interpretability.

**Paper B — "Perceived persona as a mediator of social influence in LLM agent groups"** (Phases 3–5, ~6 months)
Ambitious. Depends on Paper A clearing. Arm A alone is sufficient for a result; Arm B upgrades it from measured mediation to manipulated mediation.

**Paper C (optional) — "Where does binding live in hybrid-attention models?"** (Phases 1b + 2 hybrid arm, ~2 months incremental)
Small, self-contained, and in an unoccupied niche — nobody has done binding mechanistic interpretability on Gated DeltaNet hybrids. H7 (localization to full-attention layers) plus H8 (sharper N-ceiling from the fixed-size recurrent state) is a complete short paper, and the H8 result has direct practical implications for multi-agent deployment on hybrid models. It shares almost all its infrastructure with Papers A and B, so the marginal cost is low. Fold it into Paper A if the results are thin.

---

## 16. Reading order

Before writing code:

1. Feng & Steinhardt, 2310.17191 — the mechanism
2. Gur-Arieh et al., 2510.06182 — the modern version and your codebase
3. TalkTuner, 2406.07882 — reading vs control probes
4. Persona Vectors, 2507.21509 — the extraction pipeline
5. Broken Links, 2605.00226 — the negative result, and *why* it may be understated
6. Makelov, Lange & Nanda, 2311.17030 — the illusion, before you trust anything
7. Artificial Impressions, 2510.08915 — closest probing precedent
8. Authority bias, 2601.13433 **and** 2607.00415 — read both, note the disagreement

Then Phase 0.

---

## 17. Open decision

One thing to settle early, because it changes the shape of the project:

**Are your traits stated or inferred in the final experiments?**

*Stated* is much more likely to work and is entirely defensible — your committed-minority setup can simply declare personas in the system prompts. *Inferred* is more interesting and considerably riskier, since the model must construct the trait representation from behavior rather than retrieve it from text.

T3 in Phase 0 answers this cheaply, before you've committed to either. Don't decide now; decide when Phase 1 reports.
