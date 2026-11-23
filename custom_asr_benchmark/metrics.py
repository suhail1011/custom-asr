from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from typing import Any

from jiwer import process_characters, process_words


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']|_", " ", text)).strip()


def align_words(reference: str, hypothesis: str) -> list[dict[str, Any]]:
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()
    table = [[(0, []) for _ in range(len(hyp) + 1)] for _ in range(len(ref) + 1)]
    for i in range(1, len(ref) + 1):
        table[i][0] = (i, table[i - 1][0][1] + [{"op": "delete", "ref": ref[i - 1], "hyp": ""}])
    for j in range(1, len(hyp) + 1):
        table[0][j] = (j, table[0][j - 1][1] + [{"op": "insert", "ref": "", "hyp": hyp[j - 1]}])
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                table[i][j] = (table[i - 1][j - 1][0], table[i - 1][j - 1][1] + [
                    {"op": "correct", "ref": ref[i - 1], "hyp": hyp[j - 1]}
                ])
                continue
            candidates = [
                (table[i - 1][j - 1][0] + 1, table[i - 1][j - 1][1], {"op": "substitute", "ref": ref[i - 1], "hyp": hyp[j - 1]}),
                (table[i - 1][j][0] + 1, table[i - 1][j][1], {"op": "delete", "ref": ref[i - 1], "hyp": ""}),
                (table[i][j - 1][0] + 1, table[i][j - 1][1], {"op": "insert", "ref": "", "hyp": hyp[j - 1]}),
            ]
            cost, prior, edit = min(candidates, key=lambda candidate: candidate[0])
            table[i][j] = (cost, prior + [edit])
    return table[-1][-1][1]


def score_pair(reference: str, hypothesis: str) -> dict[str, Any]:
    normalized_reference = normalize(reference)
    normalized_hypothesis = normalize(hypothesis)
    word_alignment = align_words(normalized_reference, normalized_hypothesis)
    word_result = process_words(normalized_reference, normalized_hypothesis)
    character_result = process_characters(normalized_reference, normalized_hypothesis)
    word_counts = Counter(
        correct=word_result.hits,
        substitute=word_result.substitutions,
        delete=word_result.deletions,
        insert=word_result.insertions,
    )
    word_n = len(normalize(reference).split())
    char_n = len(normalize(reference))
    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "normalized_reference": normalized_reference,
        "normalized_hypothesis": normalized_hypothesis,
        "wer": word_result.wer,
        "cer": character_result.cer,
        "word_edits": word_result.substitutions + word_result.deletions + word_result.insertions,
        "character_edits": character_result.substitutions + character_result.deletions + character_result.insertions,
        "reference_words": word_n,
        "reference_characters": char_n,
        "operations": {key: word_counts[key] for key in ("correct", "substitute", "delete", "insert")},
        "character_operations": {
            "correct": character_result.hits,
            "substitute": character_result.substitutions,
            "delete": character_result.deletions,
            "insert": character_result.insertions,
        },
        "alignment": word_alignment,
    }


def corpus_metrics(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not scores:
        return {"count": 0, "wer": None, "cer": None}
    wers = sorted(score["wer"] for score in scores)
    cers = sorted(score["cer"] for score in scores)
    return {
        "count": len(scores),
        "wer": sum(score["word_edits"] for score in scores) / max(sum(score["reference_words"] for score in scores), 1),
        "cer": sum(score["character_edits"] for score in scores) / max(sum(score["reference_characters"] for score in scores), 1),
        "utterance_wer": {
            "mean": sum(score["wer"] for score in scores) / len(scores),
            "median": _median(wers),
        },
        "utterance_cer": {
            "mean": sum(score["cer"] for score in scores) / len(scores),
            "median": _median(cers),
        },
        "operations": {
            operation: sum(score["operations"][operation] for score in scores)
            for operation in ("correct", "substitute", "delete", "insert")
        },
    }


def _median(values: list[float]) -> float:
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2
