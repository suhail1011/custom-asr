from __future__ import annotations

from collections import defaultdict
from typing import Any

from .config import BenchmarkConfig
from .errors import BenchmarkError
from .metrics import corpus_metrics, score_pair
from .statistics import bootstrap, paired_comparison, strata


def evaluate_predictions(
    predictions: list[dict[str, Any]], config: BenchmarkConfig
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    failed = [row for row in predictions if row.get("failure")]
    rate = len(failed) / len(predictions) if predictions else 1.0
    if rate > config.limits.max_failure_rate:
        raise BenchmarkError(
            f"{len(failed)}/{len(predictions)} rows failed ({rate:.2%}); "
            f"limit is {config.limits.max_failure_rate:.2%}"
        )
    if any(row.get("subset") not in {None, config.dataset.subset} for row in predictions):
        raise BenchmarkError("predictions contain rows from a different dataset subset")
    blind = config.dataset.subset == "test_blind"
    labels = (
        []
        if blind
        else [row for row in predictions if row.get("text") not in (None, "")]
    )
    if predictions and not labels and not blind:
        raise BenchmarkError("predictions are missing reference transcripts; scoring is not possible")
    if not blind and any(
        not row.get("failure") and row.get("text") in (None, "") for row in predictions
    ):
        raise BenchmarkError("successful predictions are missing reference transcripts")
    scores = []
    for row in labels:
        if row.get("failure"):
            continue
        score = score_pair(row["text"], row["text_prediction"])
        score.update(
            {
                "id": row["id"],
                "condition": row.get("condition"),
                "confidence": row.get("confidence"),
                "word_timestamps": row.get("word_timestamps"),
                **{key: row[key] for key in row if key in {"snr_dB", "snr", "distortion", "stratum"}},
            }
        )
        timestamps = row.get("word_timestamps") or []
        hypothesis_index = 0
        for operation in score["alignment"]:
            if operation["op"] in {"correct", "substitute", "insert"}:
                timestamp = timestamps[hypothesis_index] if hypothesis_index < len(timestamps) else {}
                operation["timestamp"] = timestamp
                hypothesis_index += 1
        scores.append(score)
    if not scores:
        return scores, {
            "corpus": {"count": 0, "wer": None, "cer": None},
            "failures": failed,
            "label_free_predictions": len(predictions),
            "coverage": {
                "expected": len(predictions),
                "processed": len(predictions) - len(failed),
                "scored": 0,
                "complete": not failed,
            },
        }
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        by_condition[score.get("condition", "unknown")].append(score)
    summaries = {
        condition: {
            "corpus": corpus_metrics(values),
            "wer_ci": bootstrap(
                values, "wer", resamples=config.evaluation.bootstrap_resamples,
                seed=config.evaluation.bootstrap_seed, confidence_level=config.evaluation.confidence_level,
            ),
            "cer_ci": bootstrap(
                values, "cer", resamples=config.evaluation.bootstrap_resamples,
                seed=config.evaluation.bootstrap_seed + 1, confidence_level=config.evaluation.confidence_level,
            ),
            "strata": strata(values),
        }
        for condition, values in by_condition.items()
    }
    paired = None
    if {"clean", "noisy"} <= by_condition.keys():
        paired = paired_comparison(
            by_condition["clean"], by_condition["noisy"],
            resamples=config.evaluation.bootstrap_resamples,
            seed=config.evaluation.bootstrap_seed + 2,
            confidence_level=config.evaluation.confidence_level,
        )
    return scores, {
        "corpus": corpus_metrics(scores),
        "conditions": summaries,
        "paired_clean_noisy": paired,
        "failures": failed,
        "coverage": {
            "expected": len(predictions),
            "processed": len(predictions) - len(failed),
            "scored": len(scores),
            "complete": not failed,
        },
    }
