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
        with manifest_path.open() as stream:
            manifest = [json.loads(line) for line in stream if line.strip()]
        missing = [item["audio_filepath"] for item in manifest if not Path(item["audio_filepath"]).is_file()]
        if not missing:
            return manifest, manifest_path
        raise DataError(f"audio cache is incomplete: {len(missing)} manifest paths are missing")
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
                    "text": row.get("transcript", ""),
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
    source_digest = hashlib.sha256()
    source_rate = value.get("sampling_rate") or row["sampling_rate"]
    source_digest.update(str(int(source_rate)).encode())
    if value.get("bytes") is not None:
        source_digest.update(value["bytes"])
    elif value.get("array") is not None:
        source = np.asarray(value["array"])
        source_digest.update(source.astype(source.dtype.newbyteorder("<"), copy=False).tobytes())
    elif value.get("path"):
        with Path(value["path"]).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                source_digest.update(chunk)
    path = root / f"{digest}.wav"
    with write_lock:
        if not path.exists():
            temporary = path.with_suffix(".part.wav")
            sf.write(temporary, samples, target_rate, format="WAV", subtype="FLOAT")
            temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "duration": len(samples) / target_rate,
        "source_checksum": source_digest.hexdigest(),
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
