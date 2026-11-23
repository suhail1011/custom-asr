from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np


def bootstrap(
    scores: Sequence[dict[str, Any]],
    metric: str,
    *,
    resamples: int = 10_000,
    seed: int = 2024,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    if not scores:
        return {"estimate": None, "lower": None, "upper": None, "resamples": resamples}
    rng = np.random.default_rng(seed)
    if metric == "wer":
        estimate = sum(score["word_edits"] for score in scores) / max(sum(score["reference_words"] for score in scores), 1)
        weights = np.asarray([score["reference_words"] for score in scores], dtype=float)
        edits = np.asarray([score["word_edits"] for score in scores], dtype=float)
    elif metric == "cer":
        estimate = sum(score["character_edits"] for score in scores) / max(sum(score["reference_characters"] for score in scores), 1)
        weights = np.asarray([score["reference_characters"] for score in scores], dtype=float)
        edits = np.asarray([score["character_edits"] for score in scores], dtype=float)
    else:
        raise ValueError(f"unsupported bootstrap metric {metric}")
    values = np.empty(resamples, dtype=float)
    for offset in range(0, resamples, 100):
        count = min(100, resamples - offset)
        indices = rng.integers(0, len(scores), size=(count, len(scores)))
        denominators = weights[indices].sum(axis=1)
        numerators = edits[indices].sum(axis=1)
        values[offset : offset + count] = np.divide(
            numerators, denominators, out=np.zeros(count, dtype=float), where=denominators != 0
        )
    alpha = (1 - confidence_level) / 2
    return {
        "estimate": estimate,
        "lower": float(np.quantile(values, alpha)),
        "upper": float(np.quantile(values, 1 - alpha)),
        "resamples": resamples,
        "seed": seed,
        "confidence_level": confidence_level,
    }


def paired_comparison(
    clean: Sequence[dict[str, Any]],
    noisy: Sequence[dict[str, Any]],
    *,
    resamples: int = 10_000,
    seed: int = 2024,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    clean_by_id = {score["id"]: score for score in clean}
    noisy_by_id = {score["id"]: score for score in noisy}
    common = sorted(clean_by_id.keys() & noisy_by_id.keys())
    if not common:
        raise ValueError("clean/noisy comparison has no paired utterances")
    clean_pairs = [clean_by_id[key] for key in common]
    noisy_pairs = [noisy_by_id[key] for key in common]
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {
        "paired_count": len(common),
        "unmatched_clean": len(clean) - len(common),
        "unmatched_noisy": len(noisy) - len(common),
        "confidence_level": confidence_level,
    }
    for metric, edit_key, reference_key in (
        ("wer", "word_edits", "reference_words"),
        ("cer", "character_edits", "reference_characters"),
    ):
        clean_edits = np.asarray([row[edit_key] for row in clean_pairs], dtype=float)
        noisy_edits = np.asarray([row[edit_key] for row in noisy_pairs], dtype=float)
        references = np.asarray([row[reference_key] for row in clean_pairs], dtype=float)
        clean_estimate = clean_edits.sum() / max(references.sum(), 1)
        noisy_estimate = noisy_edits.sum() / max(references.sum(), 1)
        utterance_deltas = [
            noisy_pairs[index][metric] - clean_pairs[index][metric]
            for index in range(len(common))
        ]
        indices = rng.integers(0, len(common), size=(resamples, len(common)))
        denominators = references[indices].sum(axis=1)
        samples = np.divide(
            noisy_edits[indices].sum(axis=1) - clean_edits[indices].sum(axis=1),
            denominators,
            out=np.zeros(resamples, dtype=float),
            where=denominators != 0,
        )
        alpha = (1 - confidence_level) / 2
        result[metric] = {
            "mean_delta": noisy_estimate - clean_estimate,
            "lower": float(np.quantile(samples, alpha)),
            "upper": float(np.quantile(samples, 1 - alpha)),
            "confidence_level": confidence_level,
            "positive_count": sum(value > 0 for value in utterance_deltas),
            "negative_count": sum(value < 0 for value in utterance_deltas),
            "zero_count": sum(value == 0 for value in utterance_deltas),
        }
    return result


def strata(scores: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    from .metrics import corpus_metrics

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for score in scores:
        for key in ("snr_dB", "snr"):
            if score.get(key) is not None:
                grouped[f"{key}:{_snr_bucket(float(score[key]))}"].append(score)
        for key in ("distortion", "stratum"):
            if score.get(key) is not None:
                grouped[f"{key}:{score[key]}"].append(score)
    return {
        name: {**corpus_metrics(values), "utterance_ids": [value["id"] for value in values]}
        for name, values in sorted(grouped.items())
    }


def _snr_bucket(value: float) -> str:
    if value < 0:
        return "<0dB"
    if value < 5:
        return "0-5dB"
    if value < 10:
        return "5-10dB"
    if value < 20:
        return "10-20dB"
    return ">=20dB"
