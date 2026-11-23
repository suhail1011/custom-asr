from __future__ import annotations

import signal
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .errors import BenchmarkError, CapacityError
from .tracking import checksum


class InferenceTimeout(BenchmarkError):
    pass


class NemoRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.model = None
        self.checkpoint_path: Path | None = None
        self.checkpoint_sha256: str | None = None
        self.batch_size_final = config.model.batch_size
        self.gpu_seconds = 0.0

    def load(self) -> None:
        try:
            from huggingface_hub import snapshot_download
            from nemo.collections.asr.models import ASRModel
        except ImportError as exc:
            raise BenchmarkError(
                "NeMo/Hugging Face dependencies are unavailable; install the project environment"
            ) from exc
        snapshot = Path(
            retry_download(
                lambda: snapshot_download(
                    repo_id=self.config.model.identifier,
                    revision=self.config.model.revision,
                    allow_patterns="*.nemo",
                    cache_dir=str(Path(self.config.dataset.cache_dir).parent / "models"),
                ),
                retries=self.config.limits.network_retries,
                deadline_seconds=self.config.limits.network_timeout_seconds,
            )
        )
        checkpoints = list(snapshot.glob("*.nemo"))
        if len(checkpoints) != 1:
            raise BenchmarkError(
                f"pinned model revision must contain exactly one .nemo checkpoint, found {len(checkpoints)}"
            )
        self.checkpoint_path = checkpoints[0]
        self.checkpoint_sha256 = checksum(self.checkpoint_path)
        try:
            import torch

            device = self.config.model.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = ASRModel.restore_from(
                restore_path=str(self.checkpoint_path),
                map_location=device,
            )
        except Exception as exc:
            raise BenchmarkError(f"failed to restore approved checkpoint: {exc}") from exc
        self.model.eval()
        if hasattr(self.model, "encoder") and hasattr(self.model.encoder, "set_att_context_size"):
            self.model.encoder.set_att_context_size(list(self.config.model.chunk_context))
        elif hasattr(self.model, "change_attention_model"):
            self.model.change_attention_model(
                self_attention_model=getattr(self.model.cfg.encoder, "self_attention_model", None),
                att_context_size=list(self.config.model.chunk_context),
            )
        else:
            raise BenchmarkError("checkpoint cannot set the approved streaming attention context")

    def transcribe(self, manifest: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.model is None:
            self.load()
        import torch

        torch.manual_seed(self.config.model.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.model.seed)
        results: list[dict[str, Any]] = []
        batch_size = self.config.model.batch_size
        position = 0
        while position < len(manifest):
            batch = manifest[position : position + batch_size]
            started = time.monotonic()
            try:
                decoded = watchdog(
                    lambda current=batch: self.model.transcribe(
                        [row["audio_filepath"] for row in current],
                        batch_size=len(current),
                        return_hypotheses=True,
                        timestamps=True,
                    ),
                    self.config.model.utterance_timeout_seconds,
                )
                runtime = time.monotonic() - started
                self.gpu_seconds += runtime
                if self.gpu_seconds > self.config.limits.gpu_hours * 3600:
                    raise CapacityError("inference exceeded the approved GPU-time budget")
                results.extend(_normalize_results(batch, decoded, runtime))
                position += len(batch)
            except Exception as exc:
                if isinstance(exc, CapacityError):
                    raise
                self.gpu_seconds += time.monotonic() - started
                if self.gpu_seconds > self.config.limits.gpu_hours * 3600:
                    raise CapacityError("inference exceeded the approved GPU-time budget") from exc
                if not isinstance(exc, (InferenceTimeout, RuntimeError)):
                    raise BenchmarkError(
                        f"model inference contract failed for batch: {type(exc).__name__}: {exc}"
                    ) from exc
                if _is_oom(exc) and batch_size > 1:
                    batch_size = max(1, batch_size // 2)
                    self.batch_size_final = batch_size
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                if len(batch) > 1:
                    batch_size = max(1, batch_size // 2)
                    self.batch_size_final = batch_size
                    continue
                results.append(
                    {
                        **batch[0],
                        "text_prediction": None,
                        "word_timestamps": None,
                        "failure": f"{type(exc).__name__}: {exc}",
                        "runtime_seconds": time.monotonic() - started,
                    }
                )
                position += 1
        return results


def watchdog(operation: Callable[[], Any], timeout_seconds: int) -> Any:
    if not hasattr(signal, "SIGALRM"):
        return operation()

    def expired(_signum, _frame):
        raise InferenceTimeout(f"inference exceeded {timeout_seconds} seconds")

    prior_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(max(1, int(timeout_seconds)))
    try:
        return operation()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior_handler)


def _normalize_results(
    rows: list[dict[str, Any]], decoded: Any, runtime_seconds: float
) -> list[dict[str, Any]]:
    hypotheses = decoded[0] if isinstance(decoded, tuple) else decoded
    if isinstance(hypotheses, (str, bytes)) or len(hypotheses) != len(rows):
        raise BenchmarkError("NeMo returned a number of hypotheses different from the batch")
    result = []
    for row, hypothesis in zip(rows, hypotheses):
        text = hypothesis if isinstance(hypothesis, str) else getattr(hypothesis, "text", None)
        if text is None:
            raise BenchmarkError("NeMo returned a hypothesis without text")
        timestamps = getattr(hypothesis, "timestamp", None)
        word_timestamps = timestamps.get("word", []) if isinstance(timestamps, dict) else []
        confidence = getattr(hypothesis, "score", None)
        result.append(
            {
                **row,
                "text_prediction": text,
                "word_timestamps": word_timestamps,
                "confidence": float(confidence) if confidence is not None else None,
                "failure": None,
                "runtime_seconds": runtime_seconds / len(rows),
            }
        )
    return result


def _is_oom(exc: Exception) -> bool:
    return "out of memory" in str(exc).lower() or "cudnn" in type(exc).__name__.lower()


def retry_download(operation: Callable[[], str], retries: int, deadline_seconds: int) -> str:
    from .dataset import retry

    def cast_operation():
        return operation()

    return retry(cast_operation, retries=retries, deadline_seconds=deadline_seconds)
