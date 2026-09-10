from __future__ import annotations

import hashlib
import json
import os

from personabind.generator.traits import TRAIT_WORD_BLOCKLIST


class GenerationError(RuntimeError):
    pass


def validate_turn(
    text: str, gold: str, distractor: str | None, correct: bool
) -> tuple[bool, str]:
    n = len(text.split())
    if n < 3 or n > 60:
        return False, f"length {n} outside [3, 60]"
    tokens = {t.strip(".,;:!?\"'()").lower() for t in text.split()}
    if tokens & TRAIT_WORD_BLOCKLIST:
        return False, f"blocklist word present: {sorted(tokens & TRAIT_WORD_BLOCKLIST)}"
    if correct:
        if gold.lower() not in text.lower():
            return False, "gold answer missing"
    else:
        if distractor is None or distractor.lower() not in text.lower():
            return False, "distractor answer missing"
        if gold.lower() in text.lower():
            return False, "gold answer leaked into incorrect turn"
    return True, "ok"


def _system_prompt() -> str:
    return "You are simulating one participant in a panel discussion."


def _user_prompt(question: str, gold: str, distractor: str | None, correct: bool, style: str) -> str:
    tone = "calibrated and appropriately hedged" if style == "hedged" else "confident and unhedged"
    truth = "CORRECT" if correct else "INCORRECT"
    wrong = "" if correct else f" If INCORRECT, use exactly this wrong answer: {distractor}."
    return (
        f'Write this participant\'s answer (one or two sentences) to this question:\n"{question}"\n'
        f"Requirements:\n- It must be factually {truth}. Correct answer: {gold}.{wrong}\n"
        f"- Tone: {tone}.\n- Do NOT mention credentials, seniority, titles, or expertise.\n"
        f"- Answer only. No preamble."
    )


class StubBackend:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)

    def generate(self, qid, question, gold, distractor, correct, style, attempt) -> str:
        if not self._responses:
            raise GenerationError("StubBackend exhausted")
        return self._responses.pop(0)


class VLLMBackend:
    def __init__(
        self, base_url: str, model: str, sampling: dict, max_tokens: int,
        cache_dir: str, seed: int,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.sampling = sampling
        self.max_tokens = max_tokens
        self.cache_dir = cache_dir
        self.seed = seed
        os.makedirs(cache_dir, exist_ok=True)
        self._client = None

    def _cache_path(self, qid: str, correct: bool, style: str, attempt: int) -> str:
        key = f"{self.model}|{qid}|{correct}|{style}|{self.seed}|{attempt}"
        h = hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()
        return os.path.join(self.cache_dir, f"{h}.json")

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:  # pragma: no cover
                raise GenerationError("openai package not installed") from e
            self._client = OpenAI(base_url=self.base_url, api_key="not-needed")
        return self._client

    def generate(self, qid, question, gold, distractor, correct, style, attempt) -> str:
        path = self._cache_path(qid, correct, style, attempt)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)["text"]
        try:
            resp = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": _system_prompt()},
                    {"role": "user", "content": _user_prompt(question, gold, distractor, correct, style)},
                ],
                max_tokens=self.max_tokens,
                seed=self.seed + attempt,
                temperature=self.sampling.get("temperature", 0.7),
                top_p=self.sampling.get("top_p", 0.8),
                extra_body={
                    "top_k": self.sampling.get("top_k", 20),
                    "min_p": self.sampling.get("min_p", 0.0),
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
        except Exception as e:  # openai.APIConnectionError and friends
            raise GenerationError(
                f"could not reach vLLM server at {self.base_url}: {e}. "
                f"Start it, e.g. `vllm serve {self.model} --port 8000`."
            ) from e
        text = resp.choices[0].message.content.strip()
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"text": text}, fh)
        return text
