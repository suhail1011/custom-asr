import json

import pytest

from custom_asr_benchmark.cli import main
from custom_asr_benchmark.config import BenchmarkConfig
from custom_asr_benchmark.pipeline import evaluate
from custom_asr_benchmark.reports import write_reports
from custom_asr_benchmark.tracking import new_run


def test_run_and_index_are_append_only(tmp_path):
    first = new_run(tmp_path)
    first.write_json("resolved-config.json", {"x": 1})
    first.append_index({"status": "complete"})
    second = new_run(tmp_path)
    second.append_index({"status": "complete"})
    from custom_asr_benchmark.errors import BenchmarkError

    with pytest.raises(BenchmarkError):
        first.write_json("resolved-config.json", {"x": 2})
    entries = [json.loads(line) for line in (tmp_path / "index.jsonl").read_text().splitlines()]
    assert [entry["run_id"] for entry in entries] == [first.id, second.id]


def test_report_outputs_csv_and_markdown(tmp_path):
    score = {"id": "x", "condition": "noisy", "reference": "a", "hypothesis": "a", "wer": 0, "cer": 0}
    _, markdown = write_reports(tmp_path, [score], {"corpus": {"wer": 0, "cer": 0, "count": 1}})
    assert "WER: 0.00%" in markdown.read_text()
    assert (tmp_path / "utterances.csv").read_text().startswith("id,condition")


def test_cli_reports_invalid_config_without_traceback(tmp_path, capsys):
    path = tmp_path / "bad.toml"
    path.write_text("[dataset]\nrevision='main'\n[model]\nrevision='main'\n")
    assert main(["config-check", "--config", str(path)]) == 2
    assert "pinned commit" in capsys.readouterr().err


def test_evaluation_step_appends_completed_run(tmp_path):
    config = BenchmarkConfig.from_mapping(
        {
            "dataset": {"revision": "a" * 40, "cache_dir": str(tmp_path / "cache")},
            "model": {"revision": "b" * 40},
            "evaluation": {"bootstrap_resamples": 20},
            "runs_dir": str(tmp_path / "runs"),
        }
    )
    run_path = tmp_path / "runs" / "test-run"
    run_path.mkdir(parents=True)
    resolved = __import__("dataclasses").asdict(config)
    (run_path / "resolved-config.json").write_text(json.dumps(resolved, sort_keys=True))
    (run_path / "identity.json").write_text(json.dumps({"sample_limit": 1}))
    prediction = {
        "id": "x",
        "condition": "noisy",
        "subset": "validation",
        "text": "hello world",
        "text_prediction": "hello",
        "failure": None,
    }
    (run_path / "predictions.jsonl").write_text(json.dumps(prediction) + "\n")
    metrics, _csv, _report = evaluate(config, run_path)
    assert metrics["corpus"]["wer"] == 0.5
    index = [json.loads(line) for line in (tmp_path / "runs" / "index.jsonl").read_text().splitlines()]
    assert index[0]["run_id"] == "test-run"
    assert (run_path / "status.complete.json").is_file()
