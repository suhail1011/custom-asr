from __future__ import annotations

import hashlib
import json
import signal
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from .config import BenchmarkConfig
from .errors import BenchmarkError
from .schema import validate_rows


def load_dataset(
    config: BenchmarkConfig,
    *,
    limit: int | None = None,
    sample_id: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if limit is not None and limit < 1:
        raise BenchmarkError("dataset limit must be a positive integer")
    if limit is not None and sample_id is not None:
        raise BenchmarkError("dataset limit and sample_id are mutually exclusive")

    def operation():
        from huggingface_hub import hf_hub_download
        from pyarrow import parquet

        filename = (
            f"data/{config.dataset.subset}/"
            f"{config.dataset.subset}-00000-of-00001.parquet"
        )
        parquet_path = hf_hub_download(
            repo_id=config.dataset.repository,
            repo_type="dataset",
            filename=filename,
            revision=config.dataset.revision,
            cache_dir=config.dataset.cache_dir,
        )
        rows: list[dict[str, Any]] = []
        parquet_file = parquet.ParquetFile(parquet_path)
        batch_size = min(limit, 100) if limit is not None else 100
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            batch_rows = batch.to_pylist()
            if sample_id is not None:
                matches = [row for row in batch_rows if row["id"] == sample_id]
                if matches:
                    return matches
                continue
            remaining = None if limit is None else limit - len(rows)
            if remaining == 0:
                break
            rows.extend(batch_rows if remaining is None else batch_rows[:remaining])
        if sample_id is not None:
            raise BenchmarkError(f"dataset row id {sample_id!r} was not found")
        return rows

    rows = retry(
        operation,
        retries=config.limits.network_retries,
        deadline_seconds=config.limits.network_timeout_seconds,
    )
    for row in rows:
        row["_subset"] = config.dataset.subset
    rows = validate_rows(rows, blind=config.dataset.subset == "test_blind")
    canonical = json.dumps(
        [{**row, "clean_audio": _audio_identity(row.get("clean_audio")), "noisy_audio": _audio_identity(row.get("noisy_audio"))} for row in rows],
        sort_keys=True,
        default=str,
    ).encode()
    return rows, hashlib.sha256(canonical).hexdigest()


T = TypeVar("T")


def retry(operation: Callable[[], T], retries: int, deadline_seconds: int) -> T:
    started = time.monotonic()
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            remaining = deadline_seconds - (time.monotonic() - started)
            return _deadline(operation, max(1, int(remaining)))
        except Exception as exc:
            elapsed = time.monotonic() - started
            if attempt >= retries or elapsed + delay > deadline_seconds:
                raise BenchmarkError(
                    f"network/cache acquisition failed after {attempt + 1} attempt(s): {exc}"
                ) from exc
            time.sleep(min(delay, deadline_seconds - elapsed))
            delay *= 2


def _audio_identity(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    result = dict(value)
    array = result.pop("array", None)
    encoded = result.pop("bytes", None)
    if array is not None:
        try:
            result["array_checksum"] = hashlib.sha256(bytes(memoryview(array))).hexdigest()
        except (TypeError, ValueError):
            result["array_checksum"] = hashlib.sha256(repr(array).encode()).hexdigest()
    if encoded is not None:
        result["bytes_checksum"] = hashlib.sha256(encoded).hexdigest()
    return result


def _deadline(operation: Callable[[], T], timeout_seconds: int) -> T:
    if not hasattr(signal, "SIGALRM") or threading.current_thread() is not threading.main_thread():
        return operation()

    def expired(_signum, _frame):
        raise BenchmarkError(f"network/cache acquisition exceeded {timeout_seconds} seconds")

    prior_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(timeout_seconds)
    try:
        return operation()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior_handler)
