from __future__ import annotations

from personabind.generator.traits import article_for

# Same-sentence: each agent's trait is stated in the same clause as its name.
SAME_SENTENCE_TEMPLATE = "{names_joined} joined the review. {assignments}"

# Split-sentence: agents introduced together in sentence 1; each trait then
# stated in its own later sentence, trait-phrase BEFORE the name, to break the
# "name -> adjacent attribute" pattern. Format adapted from the alternate
# template in Feng & Steinhardt, "How do Language Models Bind Entities in
# Context?", arXiv:2310.17191.
SPLIT_SENTENCE_TEMPLATE = (
    "{names_joined} joined the panel. Among them are {trait_list}. {assignments}"
)


def _join_names(names: list[str]) -> str:
    if len(names) == 2:
        return f"{names[0]} and {names[1]}"
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _question(name: str) -> str:
    return f"How reliable is {name}?"


def render_stated(
    names: list[str], traits: list[str], query_idx: int, fmt: str
) -> tuple[str, str, str]:
    if len(names) != len(traits):
        raise ValueError("names and traits length mismatch")
    q_name = names[query_idx]
    q_trait = traits[query_idx]
    if fmt == "same_sentence":
        assignments = "; ".join(
            f"{n} is {article_for(t)} {t}" for n, t in zip(names, traits)
        ) + "."
        ctx = SAME_SENTENCE_TEMPLATE.format(
            names_joined=_join_names(names), assignments=assignments
        )
    elif fmt == "split_sentence":
        trait_list = " and ".join(f"{article_for(t)} {t}" for t in traits)
        assignments = " ".join(f"The {t} is {n}." for n, t in zip(names, traits))
        ctx = SPLIT_SENTENCE_TEMPLATE.format(
            names_joined=_join_names(names), trait_list=trait_list, assignments=assignments
        )
    else:
        raise ValueError(f"unknown fmt {fmt!r}")
    return ctx, _question(q_name), f"{q_name} is {article_for(q_trait)}"


def render_transcript(
    names: list[str], turn_lines: list[list[str]], query_idx: int
) -> tuple[str, str, str]:
    blocks = ["\n".join(lines) for lines in turn_lines]
    ctx = "\n\n".join(blocks)
    q_name = names[query_idx]
    return ctx, _question(q_name), f"{q_name} is"
