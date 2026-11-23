from __future__ import annotations

import hashlib
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .errors import DataError


def materialize_audio(
    rows: list[dict[str, Any]],
    root: Path,
    *,
    target_rate: int = 16000,
    peak_limit: float = 1.0,
    max_concurrent: int = 100,
) -> tuple[list[dict[str, Any]], Path]:
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.jsonl"
    if manifest_path.is_file():
        try:
            with manifest_path.open() as stream:
                manifest = [json.loads(line) for line in stream if line.strip()]
            _validate_cached_manifest(manifest, rows, target_rate)
        except (json.JSONDecodeError, KeyError, OSError, RuntimeError, ValueError) as exc:
            raise DataError(f"audio cache manifest is invalid or stale: {exc}") from exc
        return manifest, manifest_path
    audio_root = root / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    jobs: list[tuple[dict[str, Any], str]] = []
    for row in rows:
        fields = ["noisy_audio"] + (["clean_audio"] if "clean_audio" in row else [])
        for field in fields:
            jobs.append((row, field))

    outputs: dict[tuple[str, str], dict[str, Any]] = {}
    write_lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {
            executor.submit(_write_audio, row, field, audio_root, target_rate, peak_limit, write_lock): (row["id"], field)
            for row, field in jobs
        }
        for future in as_completed(futures):
            identifier, field = futures[future]
            outputs[(identifier, field)] = future.result()

    manifest: list[dict[str, Any]] = []
    temporary = manifest_path.with_suffix(".jsonl.part")
    with temporary.open("w") as stream:
        for row in rows:
            for field in (["clean_audio", "noisy_audio"] if "clean_audio" in row else ["noisy_audio"]):
                audio = outputs[(row["id"], field)]
                item = {
                    "audio_filepath": audio["path"],
                    "duration": audio["duration"],
                    "text": "" if row.get("_subset") == "test_blind" else row.get("transcript", ""),
                    "id": row["id"],
                    "condition": "clean" if field == "clean_audio" else "noisy",
                    "subset": row.get("_subset"),
                    "source_checksum": audio["source_checksum"],
                    "sampling_rate": target_rate,
                }
                for key, value in row.items():
                    if key in {"id", "transcript", "sampling_rate", "clean_audio", "noisy_audio"}:
                        continue
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        item.setdefault(key, value)
                manifest.append(item)
                stream.write(json.dumps(item, sort_keys=True) + "\n")
    temporary.replace(manifest_path)
    return manifest, manifest_path


def prepare_waveform(
    value: dict[str, Any],
    target_rate: int,
    peak_limit: float,
    fallback_rate: int | None = None,
) -> np.ndarray:
    samples, decoded_rate = _decode_audio(value)
    source_rate = value.get("sampling_rate") or decoded_rate or fallback_rate
    if source_rate is None:
        raise DataError("audio has no sampling_rate")
    if decoded_rate is not None and int(source_rate) != decoded_rate:
        raise DataError(
            f"audio sampling_rate {source_rate} does not match decoded rate {decoded_rate}"
        )
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1 or samples.size == 0:
        raise DataError("audio must decode to a non-empty mono waveform")
    source_rate = int(source_rate)
    if not np.isfinite(samples).all():
        raise DataError("audio contains NaN or infinite samples")
    if source_rate != target_rate:
        divisor = np.gcd(source_rate, target_rate)
        samples = resample_poly(samples, target_rate // divisor, source_rate // divisor).astype(np.float32)
    return np.clip(samples, -peak_limit, peak_limit).astype(np.float32)


def _write_audio(
    row: dict[str, Any],
    field: str,
    root: Path,
    target_rate: int,
    peak_limit: float,
    write_lock: threading.Lock,
) -> dict[str, Any]:
    value = row[field]
    samples = prepare_waveform(value, target_rate, peak_limit, row["sampling_rate"])
    digest = hashlib.sha256(samples.astype("<f4").tobytes()).hexdigest()
    source_rate = value.get("sampling_rate") or row["sampling_rate"]
    source_digest = _source_checksum(value, source_rate)
    path = root / f"{digest}.wav"
    with write_lock:
        if not path.exists():
            temporary = path.with_suffix(".part.wav")
            sf.write(temporary, samples, target_rate, format="WAV", subtype="FLOAT")
            temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "duration": len(samples) / target_rate,
        "source_checksum": source_digest,
    }


def _decode_audio(value: dict[str, Any]) -> tuple[np.ndarray, int | None]:
    if value.get("array") is not None:
        return np.asarray(value["array"], dtype=np.float32), value.get("sampling_rate")
    if value.get("bytes") is not None:
        samples, rate = sf.read(io.BytesIO(value["bytes"]), dtype="float32", always_2d=True)
        return samples, rate
    if value.get("path"):
        samples, rate = sf.read(value["path"], dtype="float32", always_2d=True)
        return samples, rate
    raise DataError("audio contains neither samples, bytes, nor a local path")


def _source_checksum(value: dict[str, Any], source_rate: int) -> str:
    digest = hashlib.sha256()
    digest.update(str(int(source_rate)).encode())
    if value.get("bytes") is not None:
        digest.update(value["bytes"])
    elif value.get("array") is not None:
        source = np.asarray(value["array"])
        digest.update(source.astype(source.dtype.newbyteorder("<"), copy=False).tobytes())
    elif value.get("path"):
        with Path(value["path"]).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        raise DataError("audio contains neither samples, bytes, nor a local path")
    return digest.hexdigest()


def _validate_cached_manifest(
    manifest: list[dict[str, Any]], rows: list[dict[str, Any]], target_rate: int
) -> None:
    expected: list[tuple[dict[str, Any], str]] = []
    for row in rows:
        fields = ["clean_audio", "noisy_audio"] if "clean_audio" in row else ["noisy_audio"]
        expected.extend((row, field) for field in fields)
    if len(manifest) != len(expected):
        raise DataError(f"audio cache row count changed ({len(manifest)} != {len(expected)})")
    for item, (row, field) in zip(manifest, expected):
        condition = "clean" if field == "clean_audio" else "noisy"
        source_rate = row[field].get("sampling_rate") or row["sampling_rate"]
        source_checksum = _source_checksum(row[field], source_rate)
        expected_text = "" if row.get("_subset") == "test_blind" else row.get("transcript", "")
        if (
            item["id"] != row["id"]
            or item["condition"] != condition
            or item["text"] != expected_text
            or item["subset"] != row.get("_subset")
            or item["sampling_rate"] != target_rate
            or item["source_checksum"] != source_checksum
        ):
            raise DataError(f"audio cache identity changed for {row['id']} ({condition})")
        path = Path(item["audio_filepath"])
        if not path.is_file():
            raise DataError(f"cached audio is missing: {path}")
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        if rate != target_rate or samples.shape[1] != 1:
            raise DataError(f"cached audio is not mono {target_rate} Hz: {path}")
        waveform_checksum = hashlib.sha256(
            samples[:, 0].astype("<f4", copy=False).tobytes()
        ).hexdigest()
        if path.name != f"{waveform_checksum}.wav":
            raise DataError(f"cached audio content does not match its checksum: {path}")
        expected_duration = len(samples) / target_rate
        if abs(float(item["duration"]) - expected_duration) > 1 / target_rate:
            raise DataError(f"cached audio duration is stale for {row['id']} ({condition})")
