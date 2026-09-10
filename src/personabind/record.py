from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

VARIANT_TRAIT_LEVELS: dict[str, set[int]] = {
    "t1_discrete": {0, 1},
    "t2_graded": {0, 1, 2, 3},
    "t3a_inferred_templated": {0, 1},
    "t3b_inferred_llm": {0, 1},
}

_T1_T2 = {"t1_discrete", "t2_graded"}
_FIELD_ORDER = [
    "id", "variant", "format", "domain", "name_style", "context", "question",
    "answer_prefix", "agents", "query_agent", "answer", "counterfactual_id",
    "counterfactual_diff", "seed", "generator_model", "turns",
]


@dataclass(frozen=True)
class AgentSpec:
    name: str
    position: int
    trait: str
    trait_level: int


@dataclass(frozen=True)
class Turn:
    qid: str
    question: str
    gold: str
    distractor: str
    answers: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class Record:
    id: str
    variant: str
    format: str
    domain: str
    name_style: str
    context: str
    question: str
    answer_prefix: str
    agents: list[AgentSpec]
    query_agent: str
    answer: str
    counterfactual_id: str
    counterfactual_diff: str
    seed: int
    generator_model: str | None = None
    turns: list[Turn] | None = None

    def validate(self) -> None:
        if self.variant not in VARIANT_TRAIT_LEVELS:
            raise ValueError(f"unknown variant {self.variant!r}")
        allowed = VARIANT_TRAIT_LEVELS[self.variant]
        for a in self.agents:
            if a.trait_level not in allowed:
                raise ValueError(
                    f"{self.id}: trait_level {a.trait_level} not in {sorted(allowed)} for {self.variant}"
                )
        names = {a.name for a in self.agents}
        if len(names) != len(self.agents):
            raise ValueError(f"{self.id}: duplicate agent names")
        if self.query_agent not in names:
            raise ValueError(f"{self.id}: query_agent {self.query_agent!r} not among agents {sorted(names)}")
        if self.variant in _T1_T2:
            if self.format not in ("same_sentence", "split_sentence"):
                raise ValueError(f"{self.id}: format {self.format!r} invalid for {self.variant}")
            if self.turns is not None:
                raise ValueError(f"{self.id}: T1/T2 records must not carry turns")
        else:
            if self.format != "n/a":
                raise ValueError(f"{self.id}: format must be 'n/a' for {self.variant}")
        positions = sorted(a.position for a in self.agents)
        if positions != list(range(len(self.agents))):
            raise ValueError(f"{self.id}: positions {positions} must be 0..n-1")


def _agent_to_dict(a: AgentSpec) -> dict[str, Any]:
    return {"name": a.name, "position": a.position, "trait": a.trait, "trait_level": a.trait_level}


def _turn_to_dict(t: Turn) -> dict[str, Any]:
    return {
        "qid": t.qid, "question": t.question, "gold": t.gold,
        "distractor": t.distractor, "answers": t.answers,
    }


def to_jsonl_line(rec: Record) -> str:
    rec.validate()
    payload: dict[str, Any] = {
        "id": rec.id, "variant": rec.variant, "format": rec.format, "domain": rec.domain,
        "name_style": rec.name_style, "context": rec.context, "question": rec.question,
        "answer_prefix": rec.answer_prefix, "agents": [_agent_to_dict(a) for a in rec.agents],
        "query_agent": rec.query_agent, "answer": rec.answer,
        "counterfactual_id": rec.counterfactual_id, "counterfactual_diff": rec.counterfactual_diff,
        "seed": rec.seed, "generator_model": rec.generator_model,
    }
    if rec.turns is not None:
        payload["turns"] = [_turn_to_dict(t) for t in rec.turns]
    ordered = {k: payload[k] for k in _FIELD_ORDER if k in payload}
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def from_jsonl_line(line: str) -> Record:
    d = json.loads(line)
    agents = [AgentSpec(**a) for a in d["agents"]]
    turns = None
    if d.get("turns") is not None:
        turns = [Turn(**t) for t in d["turns"]]
    return Record(
        id=d["id"], variant=d["variant"], format=d["format"], domain=d["domain"],
        name_style=d["name_style"], context=d["context"], question=d["question"],
        answer_prefix=d["answer_prefix"], agents=agents, query_agent=d["query_agent"],
        answer=d["answer"], counterfactual_id=d["counterfactual_id"],
        counterfactual_diff=d["counterfactual_diff"], seed=d["seed"],
        generator_model=d.get("generator_model"), turns=turns,
    )
