from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .audio import materialize_audio
from .config import BenchmarkConfig
from .dataset import load_dataset
from .errors import BenchmarkError
from .evaluation import evaluate_predictions
from .inference import NemoRunner
from .reports import write_reports
from .tracking import append_run_index, environment, new_run


def prepare(
    config: BenchmarkConfig,
    *,
    limit: int | None = None,
    sample_id: str | None = None,
):
    started = time.monotonic()
    _check_total_capacity(config)
    if config.dataset.subset == "test_nonblind":
        index = Path(config.runs_dir) / "index.jsonl"
        if index.is_file() and any(
            entry.get("subset") == "test_nonblind" and entry.get("status") == "complete"
            for entry in (json.loads(line) for line in index.read_text().splitlines() if line)
        ):
            raise BenchmarkError("test_nonblind has already been scored; its final test access is single-use")
    rows, dataset_sha = load_dataset(config, limit=limit, sample_id=sample_id)
    cache = (
        Path(config.dataset.cache_dir)
        / config.dataset.revision
        / config.dataset.subset
        / f"audio-v1-{config.audio.sample_rate}-{config.audio.peak_limit}"
    )
    if limit is not None:
        cache = cache / f"limit-{limit}"
    elif sample_id is not None:
        cache = cache / f"id-{hashlib.sha256(sample_id.encode()).hexdigest()[:12]}"
    manifest, manifest_path = materialize_audio(
        rows,
        cache,
        target_rate=config.audio.sample_rate,
        peak_limit=config.audio.peak_limit,
        max_concurrent=config.audio.max_concurrent,
    )
    if time.monotonic() - started > config.limits.cpu_hours * 3600:
        raise BenchmarkError("prepare exceeded the approved CPU-time budget")
    _check_total_capacity(config)
    run = new_run(Path(config.runs_dir))
    resolved = dataclasses.asdict(config)
    resolved["resolved_dataset_split"] = config.dataset.resolved_split
    run.write_json("resolved-config.json", resolved)
    run.write_json(
        "identity.json",
        {
            "dataset_repository": config.dataset.repository,
            "dataset_revision": config.dataset.revision,
            "dataset_sha256": dataset_sha,
            "dataset_subset": config.dataset.subset,
            "sample_limit": limit,
            "sample_id": sample_id,
            "manifest": str(manifest_path.resolve()),
        },
    )
    run.write_json("environment.json", environment())
    run.write_jsonl("prepared-manifest.jsonl", manifest)
    run.write_json(
        "status.prepared.json",
        {
            "status": "prepared",
            "rows": len(manifest),
            "manifest": str((run.path / "prepared-manifest.jsonl").resolve()),
        },
    )
    return run


def infer(config: BenchmarkConfig, run_path: str | Path) -> list[dict[str, Any]]:
    run_path = Path(run_path)
    _verify_run_config(config, run_path)
    for artifact in ("predictions.jsonl", "model-identity.json", "status.inferred.json"):
        if (run_path / artifact).exists():
            raise BenchmarkError(f"immutable run artifact already exists: {artifact}")
    manifest_path = run_path / "prepared-manifest.jsonl"
    if not manifest_path.is_file():
        raise BenchmarkError(f"{run_path} is not a prepared benchmark run")
    manifest = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    runner = NemoRunner(config)
    predictions = runner.transcribe(manifest)
    with (run_path / "predictions.jsonl").open("x") as stream:
        for row in predictions:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    with (run_path / "model-identity.json").open("x") as stream:
        stream.write(
            json.dumps(
                {
                    "identifier": config.model.identifier,
                    "revision": config.model.revision,
                    "checkpoint": str(runner.checkpoint_path),
                    "checkpoint_sha256": runner.checkpoint_sha256,
                    "precision": config.model.precision,
                    "device": config.model.device,
                    "chunk_context": config.model.chunk_context,
                    "seed": config.model.seed,
                    "decoding": config.model.decoding,
                    "batch_size_final": runner.batch_size_final,
                    "gpu_seconds": runner.gpu_seconds,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    _write_status(run_path, "inferred", predictions=len(predictions))
    return predictions


def evaluate(config: BenchmarkConfig, run_path: str | Path):
    run_path = Path(run_path)
    _verify_run_config(config, run_path)
    for artifact in ("metrics.json", "utterance-scores.jsonl", "utterances.csv", "report.md", "status.complete.json"):
        if (run_path / artifact).exists():
            raise BenchmarkError(f"immutable run artifact already exists: {artifact}")
    path = run_path / "predictions.jsonl"
    if not path.is_file():
        raise BenchmarkError(f"{run_path} has no predictions.jsonl")
    predictions = [json.loads(line) for line in path.read_text().splitlines() if line]
    scores, metrics = evaluate_predictions(predictions, config)
    with (run_path / "utterance-scores.jsonl").open("x") as stream:
        for row in scores:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    with (run_path / "metrics.json").open("x") as stream:
        stream.write(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    csv_path, report_path = write_reports(run_path, scores, metrics)
    _write_status(run_path, "complete", scored=len(scores), complete=True)
    append_run_index(
        Path(config.runs_dir),
        run_path.name,
        run_path,
        {
            "status": "complete",
            "dataset_revision": config.dataset.revision,
            "model_revision": config.model.revision,
            "subset": config.dataset.subset,
            "sample_limit": json.loads((run_path / "identity.json").read_text()).get("sample_limit"),
            "sample_id": json.loads((run_path / "identity.json").read_text()).get("sample_id"),
            "wer": metrics["corpus"].get("wer"),
            "cer": metrics["corpus"].get("cer"),
            "report": str(report_path.resolve()),
            "utterances": str(csv_path.resolve()),
            "predictions": str(path.resolve()),
        },
    )
    return metrics, csv_path, report_path


def execute(
    config: BenchmarkConfig,
    *,
    limit: int | None = None,
    sample_id: str | None = None,
):
    run = prepare(config, limit=limit, sample_id=sample_id)
    predictions = infer(config, run.path)
    metrics, _csv_path, _report_path = evaluate(config, run.path)
    return run, predictions, metrics


def _write_status(run_path: Path, status: str, **extra: Any) -> None:
    path = run_path / f"status.{status}.json"
    with path.open("x") as stream:
        stream.write(json.dumps({**extra, "status": status}, indent=2, sort_keys=True) + "\n")


def _check_total_capacity(config: BenchmarkConfig) -> None:
    roots = (Path(config.runs_dir), Path(config.dataset.cache_dir).parent)
    used = sum(
        path.stat().st_size
        for root in roots
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    )
    if used >= config.limits.max_cache_bytes:
        raise BenchmarkError(
            f"local cache/results use {used} bytes (combined limit {config.limits.max_cache_bytes})"
        )


def _verify_run_config(config: BenchmarkConfig, run_path: Path) -> None:
    path = run_path / "resolved-config.json"
    if not path.is_file():
        raise BenchmarkError(f"{run_path} has no resolved-config.json")
    expected = json.loads(json.dumps(dataclasses.asdict(config)))
    actual = json.loads(path.read_text())
    actual.pop("resolved_dataset_split", None)
    if actual != expected:
        raise BenchmarkError("run identity/config mismatch; use the exact config that prepared this run")
