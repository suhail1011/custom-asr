from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_reports(
    root: Path, scores: list[dict[str, Any]], metrics: dict[str, Any]
) -> tuple[Path, Path]:
    csv_path = root / "utterances.csv"
    with csv_path.open("x", newline="") as stream:
        columns = [
            "id", "condition", "reference", "hypothesis", "wer", "cer",
            "reference_words", "reference_characters", "confidence", "snr_dB",
            "snr", "distortion", "stratum",
        ]
        writer = csv.DictWriter(stream, columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scores)
    markdown_path = root / "report.md"
    lines = ["# URGENT 2024 ASR benchmark", "", "## Corpus", ""]
    corpus = metrics.get("corpus", {})
    lines.append(f"- WER: {_percent(corpus.get('wer'))}")
    lines.append(f"- CER: {_percent(corpus.get('cer'))}")
    lines.append(f"- Utterances: {corpus.get('count', 0)}")
    coverage = metrics.get("coverage", {})
    if coverage:
        lines += [
            "",
            "## Coverage",
            "",
            f"- Expected: {coverage.get('expected')}",
            f"- Processed: {coverage.get('processed')}",
            f"- Scored: {coverage.get('scored')}",
            f"- Complete: {'yes' if coverage.get('complete') else 'no'}",
        ]
    for condition, summary in metrics.get("conditions", {}).items():
        corpus = summary["corpus"]
        lines += [
            "",
            f"## {condition.title()}",
            "",
            f"- WER: {_percent(corpus.get('wer'))} ({_interval(summary['wer_ci'])})",
            f"- CER: {_percent(corpus.get('cer'))} ({_interval(summary['cer_ci'])})",
            f"- Utterances: {corpus.get('count', 0)}",
        ]
        for stratum, value in summary.get("strata", {}).items():
            lines.append(
                f"- {stratum}: WER {_percent(value.get('wer'))}, "
                f"CER {_percent(value.get('cer'))}, n={value.get('count', 0)}"
            )
    paired = metrics.get("paired_clean_noisy")
    if paired:
        lines += ["", "## Paired clean/noisy", "", f"- Pairs: {paired['paired_count']}"]
        lines.append(f"- Mean WER delta (noisy - clean): {_percent(paired['wer']['mean_delta'])} ({_interval(paired['wer'])})")
        lines.append(f"- Mean CER delta (noisy - clean): {_percent(paired['cer']['mean_delta'])} ({_interval(paired['cer'])})")
        lines.append(
            f"- WER direction: +{paired['wer']['positive_count']} / "
            f"-{paired['wer']['negative_count']} / ={paired['wer']['zero_count']}"
        )
    with markdown_path.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return csv_path, markdown_path


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _interval(value: dict[str, Any]) -> str:
    lower, upper = value.get("lower"), value.get("upper")
    if lower is None or upper is None:
        return "n/a"
    return f"95% CI {_percent(lower)}–{_percent(upper)}"
