import json

from personabind.cli import main


def _write_t3a_dataset(tmp_path):
    from personabind.record import AgentSpec, Record, Turn, to_jsonl_line

    records = []
    for i in range(4):
        pair_idx = i // 2
        level = i % 2
        trait, other = ("reliable", "unreliable") if level == 1 else ("unreliable", "reliable")
        doug, charles = f"Doug{pair_idx}", f"Charles{pair_idx}"
        turns = [
            Turn("q1", "Q1?", "gold1", "wrong1", {doug: {"text": "answer1"}, charles: {"text": "answer1b"}}),
            Turn("q2", "Q2?", "gold2", "wrong2", {doug: {"text": "answer2"}, charles: {"text": "answer2b"}}),
        ]
        context = f"Q1: Q1?\n{doug}: answer1\n{charles}: answer1b\n\nQ2: Q2?\n{doug}: answer2\n{charles}: answer2b"
        records.append(Record(
            id=f"t3a_{i:06d}", variant="t3a_inferred_templated", format="n/a", domain="history",
            name_style="personal", context=context,
            question=f"How reliable is {doug}?", answer_prefix=f"{doug} is",
            agents=[AgentSpec(doug, 0, trait, level), AgentSpec(charles, 1, other, 1 - level)],
            query_agent=doug, answer=trait,
            counterfactual_id=f"t3a_{i + 1:06d}" if i % 2 == 0 else f"t3a_{i - 1:06d}",
            counterfactual_diff="agent_correctness_map", seed=1, turns=turns,
        ))
    path = tmp_path / "t3a_inferred_templated.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(to_jsonl_line(r) + "\n" for r in records)
    return path


class _FakeMessage:
    def __init__(self, content):
        self.content = content
        self.reasoning_content = None


class _FakeResponse:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": _FakeMessage(content)})()]


class _FakeClient:
    """Stands in for `openai.OpenAI` -- always answers "Final answer: A", so
    this test only needs to check plumbing (dataset loaded, sampled, streamed
    to the right file), not scoring correctness (already covered by
    tests/test_cot_diagnostic.py)."""

    def __init__(self, *args, **kwargs):
        self.calls = []
        self.chat = type("Chat", (), {"completions": self})()

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeResponse("Final answer: A")


def test_binding_cot_diagnostic_streams_one_row_per_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset_path = _write_t3a_dataset(tmp_path)
    (tmp_path / "configs").mkdir()
    config_path = tmp_path / "configs" / "binding.yaml"
    config_path.write_text(
        "seed: 1\nsample_size: 4\noutput_dir: results/binding\n"
        "models: []\nvariants: []\ndataset_dir: data/\ntrain_fraction: 0.5\n"
        "layer_sweep: all\naccuracy_floor: 0.9\ncausal_clear_margin: 2.0\n"
        "mean_intervention_coefficients: [1.0]\ndtype: bfloat16\n"
    )

    fake_client_holder = {}

    def fake_openai(*args, **kwargs):
        client = _FakeClient()
        fake_client_holder["client"] = client
        return client

    import openai
    monkeypatch.setattr(openai, "OpenAI", fake_openai)

    rc = main([
        "binding", "cot-diagnostic",
        "--model", "fake-model",
        "--base-url", "http://127.0.0.1:9999/v1",
        "--dataset", str(dataset_path),
        "--config", str(config_path),
    ])
    assert rc == 0

    out_path = tmp_path / "results" / "binding" / "fake-model__t3a_cot_diagnostic.jsonl"
    assert out_path.exists()
    with open(out_path, encoding="utf-8") as fh:
        lines = [json.loads(l) for l in fh]
    assert len(lines) == 4  # sample_size=4 base records, dataset has 2 pairs -> all 4 included
    assert all(row["parsed_letter"] == "A" for row in lines)


TINY_MODEL = "sshleifer/tiny-gpt2"


def test_binding_verdict4_runs_only_test4_and_writes_a_verdict(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dataset_path = _write_t3a_dataset(tmp_path)
    (tmp_path / "configs").mkdir()
    config_path = tmp_path / "configs" / "binding.yaml"
    config_path.write_text(
        f"seed: 1\nsample_size: 3\noutput_dir: results/binding\n"
        f"dataset_dir: {tmp_path}\nvariants: []\nmodels: []\ntrain_fraction: 0.5\n"
        "layer_sweep: [0]\naccuracy_floor: 0.9\ncausal_clear_margin: 2.0\n"
        "mean_intervention_coefficients: [1.0]\ndtype: float32\n"
    )
    # run_test4_only reads dataset_dir/<variant>.jsonl -- rename to match
    (tmp_path / "t3a_inferred_templated.jsonl").write_text(dataset_path.read_text())

    rc = main([
        "binding", "verdict4",
        "--model", TINY_MODEL,
        "--variant", "t3a_inferred_templated",
        "--config", str(config_path),
    ])
    assert rc == 0

    prefix = f"{TINY_MODEL.replace('/', '_')}__t3a_inferred_templated__"
    output_dir = tmp_path / "results" / "binding"
    assert not (output_dir / f"{prefix}accuracy.jsonl").exists()
    verdict_path = output_dir / f"{prefix}verdict4.json"
    assert verdict_path.exists()
    verdict = json.loads(verdict_path.read_text())
    assert verdict["test"] == "mean_intervention"
    assert "passed" in verdict


def test_build_then_report_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # copy the test config next to a writable output dir
    cfg_src = __import__("pathlib").Path(__file__).parent.parent / "configs" / "generator.test.yaml"
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "generator.test.yaml").write_text(
        cfg_src.read_text().replace("output_dir: data/test/", f"output_dir: {tmp_path}/out/")
    )
    rc = main(["build", "--variant", "t1", "--config", "configs/generator.test.yaml"])
    assert rc == 0
    assert (tmp_path / "out" / "t1_discrete.jsonl").exists()

    rc = main(["report", "--dataset", str(tmp_path / "out" / "t1_discrete.jsonl")])
    assert rc == 0


def test_report_returns_one_on_violation(tmp_path):
    bad = tmp_path / "bad.jsonl"
    # two records that reference nonexistent counterfactual twins
    bad.write_text(
        '{"id":"t1_1","variant":"t1_discrete","format":"same_sentence","domain":"science",'
        '"name_style":"personal","context":"x","question":"How reliable is A?","answer_prefix":"A is an",'
        '"agents":[{"name":"A","position":0,"trait":"expert","trait_level":1},'
        '{"name":"B","position":1,"trait":"novice","trait_level":0}],"query_agent":"A","answer":"expert",'
        '"counterfactual_id":"nope","counterfactual_diff":"agent_trait_map","seed":1,"generator_model":null}\n'
    )
    rc = main(["report", "--dataset", str(bad)])
    assert rc == 1
