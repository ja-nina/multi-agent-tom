from personabind.cli import main


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
