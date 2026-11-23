from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import tomllib

from .errors import ConfigError


@dataclass(frozen=True)
class DatasetConfig:
    repository: str = "urgent-challenge/urgent2024_official"
    revision: str = ""
    subset: str = "validation"
    split: str = "auto"
    cache_dir: str = "cache/dataset"

    def __post_init__(self) -> None:
        if self.repository != "urgent-challenge/urgent2024_official":
            raise ConfigError("dataset.repository is fixed to urgent-challenge/urgent2024_official")
        if self.subset not in {"validation", "test_nonblind", "test_blind"}:
            raise ConfigError("dataset.subset must be validation, test_nonblind, or test_blind")
        if not _is_revision(self.revision):
            raise ConfigError("dataset.revision must be a pinned commit SHA, not a branch or tag")
        expected_split = "test" if self.subset == "test_blind" else "validation"
        if self.split not in {"auto", expected_split}:
            raise ConfigError(f"dataset.split for {self.subset} must be {expected_split}")

    @property
    def resolved_split(self) -> str:
        return "test" if self.subset == "test_blind" else "validation"


@dataclass(frozen=True)
class ModelConfig:
    identifier: str = "nvidia/nemotron-speech-streaming-en-0.6b"
    revision: str = ""
    device: str = "cuda"
    precision: str = "fp32"
    batch_size: int = 8
    chunk_context: tuple[int, int] = (70, 1)
    seed: int = 2024
    decoding: str = "rnnt"
    utterance_timeout_seconds: int = 300

    def __post_init__(self) -> None:
        if self.identifier != "nvidia/nemotron-speech-streaming-en-0.6b":
            raise ConfigError("model.identifier must remain the approved Nemotron checkpoint")
        if not _is_revision(self.revision):
            raise ConfigError("model.revision must be a pinned commit SHA")
        if self.device not in {"cuda", "cpu", "auto"}:
            raise ConfigError("model.device must be cuda, cpu, or auto")
        if self.precision != "fp32":
            raise ConfigError("model.precision must be fp32; the approved model does not support half")
        if self.batch_size < 1:
            raise ConfigError("model.batch_size must be positive")
        if tuple(self.chunk_context) != (70, 1):
            raise ConfigError("model.chunk_context must remain [70, 1] (160 ms with right context 1)")
        if not 1 <= self.utterance_timeout_seconds <= 300:
            raise ConfigError("model.utterance_timeout_seconds must be between 1 and 300")
        if self.decoding != "rnnt":
            raise ConfigError("model.decoding must be rnnt")


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    max_concurrent: int = 100
    peak_limit: float = 1.0

    def __post_init__(self) -> None:
        if self.sample_rate != 16000:
            raise ConfigError("audio.sample_rate is fixed at 16000")
        if not 1 <= self.max_concurrent <= 100:
            raise ConfigError("audio.max_concurrent must be between 1 and 100")
        if not 0 < self.peak_limit <= 1:
            raise ConfigError("audio.peak_limit must be in (0, 1]")


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_resamples: int = 10_000
    bootstrap_seed: int = 2024
    confidence_level: float = 0.95

    def __post_init__(self) -> None:
        if (
            not 1 <= self.bootstrap_resamples <= 10_000
            or self.confidence_level <= 0
            or self.confidence_level >= 1
        ):
            raise ConfigError("invalid evaluation confidence interval settings")


@dataclass(frozen=True)
class LimitsConfig:
    max_cache_bytes: int = 10 * 1024**3
    cpu_hours: float = 4
    gpu_hours: float = 4
    network_retries: int = 3
    network_timeout_seconds: int = 600
    max_failure_rate: float = 0.01

    def __post_init__(self) -> None:
        if self.max_cache_bytes > 10 * 1024**3 or self.max_cache_bytes < 1:
            raise ConfigError("limits.max_cache_bytes cannot exceed 10 GiB")
        if not 0 < self.cpu_hours <= 4 or not 0 < self.gpu_hours <= 4:
            raise ConfigError("CPU and GPU limits must be positive and cannot exceed four hours")
        if not 0 <= self.network_retries <= 3:
            raise ConfigError("limits.network_retries cannot exceed three")
        if not 1 <= self.network_timeout_seconds <= 600:
            raise ConfigError("invalid network retry settings")
        if not 0 <= self.max_failure_rate < 1:
            raise ConfigError("limits.max_failure_rate must be in [0, 1)")


@dataclass(frozen=True)
class BenchmarkConfig:
    dataset: DatasetConfig
    model: ModelConfig
    audio: AudioConfig = AudioConfig()
    evaluation: EvaluationConfig = EvaluationConfig()
    limits: LimitsConfig = LimitsConfig()
    runs_dir: str = "runs"

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkConfig:
        path = Path(path)
        try:
            raw = tomllib.loads(path.read_text())
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot read config {path}: {exc}") from exc
        return cls.from_mapping(raw)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BenchmarkConfig:
        unknown = set(raw) - {"dataset", "model", "audio", "evaluation", "limits", "runs_dir"}
        if unknown:
            raise ConfigError(f"unknown top-level settings: {', '.join(sorted(unknown))}")
        if "runs_dir" in raw and (
            not isinstance(raw["runs_dir"], str) or not raw["runs_dir"].strip()
        ):
            raise ConfigError("runs_dir must be a non-empty path")
        try:
            return cls(
                dataset=_section(DatasetConfig, raw, "dataset"),
                model=_section(ModelConfig, raw, "model"),
                audio=_section(AudioConfig, raw, "audio"),
                evaluation=_section(EvaluationConfig, raw, "evaluation"),
                limits=_section(LimitsConfig, raw, "limits"),
                runs_dir=raw.get("runs_dir", "runs"),
            )
        except (TypeError, ValueError) as exc:
            if isinstance(exc, ConfigError):
                raise
            raise ConfigError(str(exc)) from exc


def _section(cls: type, raw: Mapping[str, Any], name: str):
    value = raw.get(name, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a TOML table")
    allowed = {field.name for field in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        raise ConfigError(f"unknown {name} settings: {', '.join(sorted(unknown))}")
    converted = dict(value)
    if name == "model" and isinstance(converted.get("chunk_context"), list):
        converted["chunk_context"] = tuple(converted["chunk_context"])
    return cls(**converted)


def _is_revision(value: str) -> bool:
    return bool(re.fullmatch(r"[0-9a-fA-F]{40}", value))
