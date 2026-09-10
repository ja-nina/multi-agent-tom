from __future__ import annotations

import hashlib
from dataclasses import dataclass

from personabind.generator.traits import contains_blocklisted

DOMAINS: tuple[str, ...] = ("science", "history", "medicine", "law")

# MMLU subject -> our domain. Unlisted subjects are skipped.
MMLU_SUBJECT_DOMAIN: dict[str, str] = {
    # science
    "high_school_biology": "science", "college_biology": "science",
    "high_school_chemistry": "science", "college_chemistry": "science",
    "high_school_physics": "science", "college_physics": "science",
    "astronomy": "science", "conceptual_physics": "science",
    "high_school_computer_science": "science", "electrical_engineering": "science",
    # history
    "high_school_world_history": "history", "high_school_us_history": "history",
    "high_school_european_history": "history", "prehistory": "history",
    # medicine
    "clinical_knowledge": "medicine", "college_medicine": "medicine",
    "professional_medicine": "medicine", "anatomy": "medicine",
    "medical_genetics": "medicine", "nutrition": "medicine",
    # law
    "professional_law": "law", "international_law": "law", "jurisprudence": "law",
}


@dataclass(frozen=True)
class QAItem:
    qid: str
    domain: str
    question: str
    gold: str
    distractors: list[str]


def make_qid(src: str, question: str) -> str:
    """Content-derived question id: `<source>_<hash of the normalised question>`.

    Deriving the id from the question text -- rather than a process-global
    counter -- makes it stable across `load_bank` calls, processes and partial
    reloads. That matters because the id is written into every emitted record's
    `turns[].qid` and is part of the T3b generation cache key: a counter would
    hand the same question a different id on the next run, silently invalidating
    the cache and making two builds of "the same" dataset incomparable.

    Two items whose normalised questions collide get the same id; `load_bank`
    already dedups on the lowercased question, so at most one of them survives.
    """
    h = hashlib.blake2b(question.strip().lower().encode("utf-8"), digest_size=6)
    return f"{src}_{h.hexdigest()}"


def normalize_mmlu(row: dict) -> QAItem | None:
    subject = row.get("subject", "")
    domain = MMLU_SUBJECT_DOMAIN.get(subject)
    if domain is None:
        return None
    choices = list(row["choices"])
    ans = int(row["answer"])
    if not (0 <= ans < len(choices)):
        return None
    gold = str(choices[ans]).strip()
    distractors = [str(c).strip() for i, c in enumerate(choices) if i != ans]
    question = str(row["question"]).strip()
    return QAItem(make_qid("mmlu", question), domain, question, gold, distractors)


def is_clean(item: QAItem) -> bool:
    g = item.gold
    # non-empty
    if not g:
        return False
    # single sentence: no ". " mid-string
    if ". " in g:
        return False
    # <= 8 words
    if len(g.split()) > 8:
        return False
    # >= 1 distractor
    if not item.distractors:
        return False
    # No trait/seniority word anywhere that can reach the rendered transcript.
    # The question, gold and every distractor all land in `context` (the question
    # via the `Qn:` line, the answers via the agent turns), so a blocklisted word
    # in ANY of them would trip the T3 context invariant mid-build. Filtering the
    # item out here is the only place that can prevent that.
    if contains_blocklisted(item.question):
        return False
    if contains_blocklisted(g):
        return False
    if any(contains_blocklisted(d) for d in item.distractors):
        return False
    # gold distinct (case-insensitively) from every distractor
    gold_key = g.strip().lower()
    return all(d.strip().lower() != gold_key for d in item.distractors)


def _load_source(source: str, cache_dir: str, limit: int | None) -> list[QAItem]:
    from datasets import load_dataset

    items: list[QAItem] = []
    if source == "mmlu":
        ds = load_dataset("cais/mmlu", "all", split="test", cache_dir=cache_dir)
        for row in ds:
            it = normalize_mmlu(row)
            if it is not None:
                items.append(it)
            if limit and len(items) >= limit:
                break
    elif source == "sciq":
        # namespaced id: the bare "sciq" alias no longer resolves on the Hub.
        ds = load_dataset("allenai/sciq", split="train", cache_dir=cache_dir)
        for row in ds:
            distractors = [row["distractor1"], row["distractor2"], row["distractor3"]]
            question = str(row["question"]).strip()
            items.append(QAItem(make_qid("sciq", question), "science", question,
                                str(row["correct_answer"]).strip(),
                                [str(d).strip() for d in distractors]))
            if limit and len(items) >= limit:
                break
    elif source == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", cache_dir=cache_dir)
        for row in ds:
            labels = row["choices"]["label"]
            texts = row["choices"]["text"]
            key = row["answerKey"]
            if key not in labels:
                continue
            gi = labels.index(key)
            gold = str(texts[gi]).strip()
            distractors = [str(t).strip() for j, t in enumerate(texts) if j != gi]
            question = str(row["question"]).strip()
            items.append(QAItem(make_qid("arc", question), "science",
                                question, gold, distractors))
            if limit and len(items) >= limit:
                break
    else:
        raise ValueError(f"unknown qa source {source!r}")
    return items


def load_bank(
    sources: list[str], domains: list[str], cache_dir: str,
    limit_per_source: int | None = None,
) -> list[QAItem]:
    seen: set[str] = set()
    out: list[QAItem] = []
    wanted = set(domains)
    for src in sources:
        for it in _load_source(src, cache_dir, limit_per_source):
            if it.domain not in wanted:
                continue
            if not is_clean(it):
                continue
            key = it.question.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
    if not out:
        raise RuntimeError(f"QA bank empty for sources={sources} domains={domains}")
    return out
