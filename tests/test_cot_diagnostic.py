from personabind.binding.cot_diagnostic import (
    _build_user_message,
    _parse_final_letter,
    _response_text,
    run_cot_diagnostic,
)
from personabind.record import AgentSpec, Record, Turn


def _transcript_record(id_="t3a_1", answer="reliable", other_trait="unreliable"):
    turns = [
        Turn("q1", "Q1?", "gold1", "wrong1", {"Doug": {"text": "answer1"}, "Charles": {"text": "answer1b"}}),
        Turn("q2", "Q2?", "gold2", "wrong2", {"Doug": {"text": "answer2"}, "Charles": {"text": "answer2b"}}),
    ]
    context = "Q1: Q1?\nDoug: answer1\nCharles: answer1b\n\nQ2: Q2?\nDoug: answer2\nCharles: answer2b"
    return Record(
        id=id_, variant="t3a_inferred_templated", format="n/a", domain="history",
        name_style="personal", context=context,
        question="How reliable is Doug?", answer_prefix="Doug is",
        agents=[AgentSpec("Doug", 0, answer, 1 if answer == "reliable" else 0),
                AgentSpec("Charles", 1, other_trait, 0 if answer == "reliable" else 1)],
        query_agent="Doug", answer=answer,
        counterfactual_id=id_ + "_cf", counterfactual_diff="agent_correctness_map", seed=1, turns=turns,
    )


def test_parse_final_letter_is_case_insensitive_and_tolerant_of_spacing():
    assert _parse_final_letter("blah blah\nFinal answer: A") == "A"
    assert _parse_final_letter("Final Answer:B") == "B"
    assert _parse_final_letter("final   answer:  a") == "A"


def test_parse_final_letter_takes_the_last_match_not_the_first():
    # A real CoT trace legitimately mentions both letters while reasoning --
    # only the model's own final, explicit statement should count.
    text = "If A were true... but actually A is wrong.\nFinal answer: B"
    assert _parse_final_letter(text) == "B"


def test_parse_final_letter_returns_none_when_unparseable():
    assert _parse_final_letter("I think Doug is reliable overall.") is None


def test_build_user_message_includes_context_question_and_mc_block():
    record = _transcript_record()
    msg = _build_user_message(record, trait_for_a="reliable", trait_for_b="unreliable")
    assert record.context in msg
    assert "Given this excerpt from the conversation between Doug and Charles" in msg
    assert "A) Doug is reliable." in msg
    assert "B) Doug is unreliable." in msg
    assert "Final answer:" in msg


def test_response_text_concatenates_reasoning_and_content_when_both_present():
    class _Msg:
        reasoning_content = "step one, step two"
        content = "Final answer: A"

    text = _response_text(_Msg())
    assert "step one, step two" in text
    assert "Final answer: A" in text


def test_response_text_falls_back_to_content_only_when_no_reasoning_field():
    class _Msg:
        content = "Final answer: B"

    assert _response_text(_Msg()) == "Final answer: B"


class _FakeMessage:
    def __init__(self, content, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeChoice:
    def __init__(self, message):
        self.message = message


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse(_FakeMessage(self._responses.pop(0)))


class _FakeChat:
    def __init__(self, responses):
        self.completions = _FakeCompletions(responses)


class _FakeClient:
    def __init__(self, responses):
        self.chat = _FakeChat(responses)


def test_run_cot_diagnostic_scores_a_parsed_correct_response():
    record = _transcript_record(answer="reliable")
    client = _FakeClient(["reasoning...\nFinal answer: A"])
    # seed=0 -> _own_trait_is_a(0) determines which letter carries "reliable";
    # patch around it by checking both fields agree with each other instead
    # of hardcoding the letter.
    results = run_cot_diagnostic(client, "fake-model", [record], seed=0)
    assert len(results) == 1
    r = results[0]
    assert r.parsed_letter == "A"
    assert r.predicted in ("reliable", "unreliable")
    assert r.correct == (r.predicted == record.answer)
    assert r.gold == "reliable"
    assert r.response == "reasoning...\nFinal answer: A"


def test_run_cot_diagnostic_marks_unparseable_responses_as_none_not_wrong():
    record = _transcript_record(answer="reliable")
    client = _FakeClient(["I couldn't decide, sorry."])
    results = run_cot_diagnostic(client, "fake-model", [record], seed=0)
    r = results[0]
    assert r.parsed_letter is None
    assert r.predicted is None
    assert r.correct is None  # never silently scored as incorrect


def test_run_cot_diagnostic_enables_thinking_mode_and_threads_the_seed():
    record = _transcript_record()
    client = _FakeClient(["Final answer: A"])
    run_cot_diagnostic(client, "fake-model", [record], seed=777)
    call = client.chat.completions.calls[0]
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert call["seed"] == 777  # seed + idx, idx=0 for the only record
    assert call["model"] == "fake-model"


def test_run_cot_diagnostic_calls_on_result_once_per_record_for_streaming():
    records = [_transcript_record(id_="t3a_1"), _transcript_record(id_="t3a_2", answer="unreliable", other_trait="reliable")]
    client = _FakeClient(["Final answer: A", "Final answer: B"])
    streamed = []
    results = run_cot_diagnostic(client, "fake-model", records, seed=0, on_result=streamed.append)
    assert streamed == results
    assert len(streamed) == 2
