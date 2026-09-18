from __future__ import annotations

from dataclasses import dataclass

from personabind.generator.schema import render_stated
from personabind.record import Record


@dataclass(frozen=True)
class TokenizedPrompt:
    record_id: str
    text: str
    input_ids: list[int]
    offsets: list[tuple[int, int]]


def tokenize_record(record: Record, tokenizer) -> TokenizedPrompt:
    text = f"{record.context}\n{record.question}\n{record.answer_prefix}"
    encoded = tokenizer(text, return_offsets_mapping=True)
    return TokenizedPrompt(
        record_id=record.id, text=text,
        input_ids=encoded["input_ids"], offsets=encoded["offset_mapping"],
    )


def _find_all(text: str, needle: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(needle)))
        start = idx + 1
    return spans


def _char_span_to_token_span(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int]:
    tok_start = tok_end = None
    for i, (s, e) in enumerate(offsets):
        if s == e:  # zero-width offsets mark special tokens; never match one
            continue
        if s < char_end and e > char_start:
            if tok_start is None:
                tok_start = i
            tok_end = i
    if tok_start is None:
        raise ValueError(f"no token overlaps character span [{char_start}, {char_end})")
    return tok_start, tok_end


def agent_spans(tokenized: TokenizedPrompt, record: Record) -> dict[str, list[tuple[int, int]]]:
    result: dict[str, list[tuple[int, int]]] = {}
    for agent in record.agents:
        char_spans = _find_all(tokenized.text, agent.name)
        if not char_spans:
            raise ValueError(f"{record.id}: agent name {agent.name!r} not found anywhere in the prompt")
        result[agent.name] = [_char_span_to_token_span(tokenized.offsets, s, e) for s, e in char_spans]
    return result


_ARTICLES = ("a", "an")


def trait_spans(tokenized: TokenizedPrompt, record: Record) -> dict[str, tuple[int, int]] | None:
    if record.variant not in ("t1_discrete", "t2_graded"):
        return None
    result: dict[str, tuple[int, int]] = {}
    for agent in record.agents:
        if record.format == "same_sentence":
            found = False
            for article in _ARTICLES:
                needle = f"{agent.name} is {article} {agent.trait}"
                spans = _find_all(record.context, needle)
                if spans:
                    whole_start, whole_end = spans[0]
                    trait_char_start = whole_end - len(agent.trait)
                    article_char_start = trait_char_start - len(article) - 1  # -1 for the space
                    result[agent.name] = _char_span_to_token_span(tokenized.offsets, article_char_start, whole_end)
                    found = True
                    break
            if not found:
                raise ValueError(f"{record.id}: could not locate {agent.name}'s trait phrase (same_sentence)")
        elif record.format == "split_sentence":
            needle = f"The {agent.trait} is {agent.name}"
            spans = _find_all(record.context, needle)
            if not spans:
                raise ValueError(f"{record.id}: could not locate {agent.name}'s trait phrase (split_sentence)")
            whole_start, _ = spans[0]
            trait_char_start = whole_start + len("The ")
            trait_char_end = trait_char_start + len(agent.trait)
            result[agent.name] = _char_span_to_token_span(tokenized.offsets, trait_char_start, trait_char_end)
        else:
            raise ValueError(f"{record.id}: unknown format {record.format!r} for trait_spans")
    return result


def answer_position(tokenized: TokenizedPrompt) -> int:
    return len(tokenized.input_ids) - 1


def query_agent_position(tokenized: TokenizedPrompt, record: Record) -> int:
    question_char_start = len(record.context) + 1
    question_char_end = question_char_start + len(record.question)
    spans = agent_spans(tokenized, record)[record.query_agent]
    in_question = [
        (ts, te) for (ts, te) in spans
        if tokenized.offsets[ts][0] >= question_char_start and tokenized.offsets[te][1] <= question_char_end
    ]
    if not in_question:
        raise ValueError(f"{record.id}: query_agent {record.query_agent!r} has no mention inside the question")
    return in_question[-1][1]


def _last_turn_speaker_position(tokenized: TokenizedPrompt, record: Record, agent: str) -> int:
    if not record.turns or len(record.turns) < 2:
        raise ValueError(
            f"{record.id}: stored_position needs >= 2 turns for T3 -- turn 1's speaker label "
            f"has no prior revealing content, same failure mode as same_sentence T1/T2"
        )
    needle = f"{agent}:"
    spans = _find_all(record.context, needle)
    if len(spans) < 2:
        raise ValueError(f"{record.id}: expected >= 2 occurrences of {needle!r}, found {len(spans)}")
    char_start, char_end = spans[-1]
    _, tok_end = _char_span_to_token_span(tokenized.offsets, char_start, char_end)
    return tok_end


def stored_position(tokenized: TokenizedPrompt, record: Record, agent: str) -> int:
    spans = trait_spans(tokenized, record)
    if spans is not None:
        _, tok_end = spans[agent]
        return tok_end
    return _last_turn_speaker_position(tokenized, record, agent)


def render_query_for(record: Record, other_agent: str) -> tuple[str, str]:
    if record.variant in ("t1_discrete", "t2_graded"):
        names = [a.name for a in record.agents]
        traits = [a.trait for a in record.agents]
        other_idx = names.index(other_agent)
        _, question, answer_prefix = render_stated(names, traits, query_idx=other_idx, fmt=record.format)
        return question, answer_prefix
    other = next(a for a in record.agents if a.name == other_agent)
    return f"How reliable is {other.name}?", f"{other.name} is"


def trait_of(record: Record, agent_name: str) -> str:
    """The value `agent_name` is bound to in `record`, read from agents[].trait --
    the source of truth for "what does this record say about this agent",
    independent of who record.query_agent/record.answer happen to be. Tasks 7
    and 9 both need this for off-target measurement: record.answer only ever
    reflects the record's OWN query_agent, not whichever other agent a
    re-rendered question asks about."""
    return next(a.trait for a in record.agents if a.name == agent_name)
