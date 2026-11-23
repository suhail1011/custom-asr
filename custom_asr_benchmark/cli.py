from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .config import BenchmarkConfig
from .errors import BenchmarkError
from .pipeline import evaluate, execute, infer, prepare


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="urgent-asr",
        description="Pinned, local URGENT 2024 ASR benchmark runner.",
    )
    commands = result.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("config-check", "Validate and print resolved configuration"),
        ("prepare", "Resolve the pinned dataset and create a preprocessed run"),
        ("infer", "Run approved NeMo inference for a prepared run"),
        ("evaluate", "Score predictions and create reports for an inferred run"),
        ("run", "Prepare, infer, evaluate, and index one complete run"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--config", required=True, help="TOML configuration")
        if name in {"prepare", "run"}:
            command.add_argument("--limit", type=int, help="Rows to process for a smoke run")
            command.add_argument("--sample-id", help="One exact labeled row for a smoke run")
        if name in {"infer", "evaluate"}:
            command.add_argument("--run-dir", required=True, help="Existing immutable run directory")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = BenchmarkConfig.load(args.config)
        if args.command == "config-check":
            print(json.dumps(__import__("dataclasses").asdict(config), indent=2, sort_keys=True))
            return 0
        if args.command == "prepare":
            run = prepare(config, limit=args.limit, sample_id=args.sample_id)
            print(run.path)
            return 0
        if args.command == "infer":
            predictions = infer(config, Path(args.run_dir))
            print(f"{len(predictions)} predictions -> {Path(args.run_dir) / 'predictions.jsonl'}")
            return 0
        if args.command == "evaluate":
            metrics, _csv, report = evaluate(config, Path(args.run_dir))
            print(f"WER={metrics['corpus']['wer']} CER={metrics['corpus']['cer']} -> {report}")
            return 0
        if args.command == "run":
            run, _predictions, metrics = execute(
                config, limit=args.limit, sample_id=args.sample_id
            )
            print(f"run={run.path} WER={metrics['corpus']['wer']} CER={metrics['corpus']['cer']}")
            return 0
    except (BenchmarkError, KeyboardInterrupt) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
