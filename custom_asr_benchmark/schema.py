from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .errors import DataError


def validate_rows(rows: Iterable[dict[str, Any]], *, blind: bool) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen: set[str] = set()
    for number, row in enumerate(rows, 1):
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise DataError(f"row {number}: id must be a non-empty string")
        if identifier in seen:
            raise DataError(f"row {number}: duplicate id {identifier!r}")
        seen.add(identifier)
        sampling_rate = row.get("sampling_rate")
        if not isinstance(sampling_rate, int) or not 16_000 <= sampling_rate <= 48_000:
            raise DataError(f"{identifier}: sampling_rate must be an integer from 16000 to 48000")
        for field in ("noisy_audio",) if blind else ("noisy_audio", "clean_audio"):
            audio = row.get(field)
            if not isinstance(audio, dict) or not any(
                audio.get(key) is not None for key in ("array", "bytes", "path")
            ):
                raise DataError(f"{identifier}: {field} must contain decoded samples or encoded audio")
            if audio.get("sampling_rate") is not None:
                audio_rate = audio["sampling_rate"]
                if not isinstance(audio_rate, int) or not 16_000 <= audio_rate <= 48_000:
                    raise DataError(
                        f"{identifier}: {field}.sampling_rate must be an integer from 16000 to 48000"
                    )
        if not blind:
            transcript = row.get("transcript")
            if (
                not isinstance(transcript, str)
                or not transcript.strip()
                or transcript.strip().lower() == "<not-available>"
            ):
                raise DataError(f"{identifier}: transcript is missing or empty")
        validated.append(row)
    if not validated:
        raise DataError("dataset contains no rows")
    return validated


def require_file(path: str | Path, label: str = "file") -> Path:
    result = Path(path)
    if not result.is_file():
        raise DataError(f"{label} does not exist: {result}")
    return result
