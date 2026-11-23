from __future__ import annotations

import hashlib
import json
import os
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import BenchmarkError, CapacityError


class Run:
    def __init__(self, root: Path, run_id: str):
        self.root = root
        self.id = run_id
        self.path = root / run_id
        self.path.mkdir(parents=False, exist_ok=False)

    def write_json(self, name: str, value: Any) -> Path:
        path = self.path / name
        data = json.dumps(value, indent=2, sort_keys=True, default=str) + "\n"
        try:
            with path.open("x") as stream:
                stream.write(data)
        except FileExistsError as exc:
            raise BenchmarkError(f"immutable run artifact already exists: {path}") from exc
        return path

    def write_jsonl(self, name: str, values: list[dict[str, Any]]) -> Path:
        path = self.path / name
        with path.open("x") as stream:
            for value in values:
                stream.write(json.dumps(value, sort_keys=True, default=str) + "\n")
        return path

    def append_index(self, summary: dict[str, Any]) -> None:
        append_run_index(self.root, self.id, self.path, summary)


def append_run_index(
    root: Path, run_id: str, run_path: Path, summary: dict[str, Any]
) -> None:
    entry = {"run_id": run_id, "path": str(run_path.resolve()), **summary}
    data = (json.dumps(entry, sort_keys=True) + "\n").encode()
    index = root / "index.jsonl"
    descriptor = os.open(index, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        if os.write(descriptor, data) != len(data):
            raise OSError("short append to experiment index")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def new_run(runs_root: str | Path) -> Run:
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Run(root, f"{stamp}-{uuid.uuid4().hex[:8]}")


def check_capacity(root: Path, max_bytes: int) -> int:
    used = sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.exists() else 0
    if used >= max_bytes:
        raise CapacityError(f"local cache/results already use {used} bytes (limit {max_bytes})")
    return used


def environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": None,
        "cuda": None,
    }
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda"] = torch.version.cuda
        result["device_count"] = torch.cuda.device_count()
    except (ImportError, RuntimeError):
        result["torch_error"] = "torch environment unavailable"
    return result


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
