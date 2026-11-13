"""Paths, environment and strategy configuration shared by every pipeline step."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

STARTER_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = STARTER_DIR.parent

load_dotenv(STARTER_DIR / ".env")


def _path_from_env(name: str, default: Path) -> Path:
    value = os.getenv(name)
    if not value:
        return default
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path)


@dataclass(frozen=True)
class Paths:
    """Deliverable layout follows SUBMISSION.md (data/, benchmark/, results/, artifacts/ at repo root)."""

    manifest: Path = _path_from_env("MANIFEST_PATH", REPO_ROOT / "data" / "manifest.json")
    benchmark: Path = _path_from_env("BENCHMARK_PATH", REPO_ROOT / "benchmark" / "questions.json")
    strategies: Path = _path_from_env("STRATEGIES_PATH", STARTER_DIR / "config" / "strategies.json")
    results: Path = _path_from_env("RESULTS_DIR", REPO_ROOT / "results")
    experiment_log: Path = REPO_ROOT / "artifacts" / "experiment_log.csv"

    @property
    def traces(self) -> Path:
        return self.results / "traces"

    @property
    def audit(self) -> Path:
        return self.results / "audit"

    def ensure(self) -> None:
        for folder in (self.results, self.traces, self.audit):
            folder.mkdir(parents=True, exist_ok=True)


PATHS = Paths()


def database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL missing: copy starter/.env.example to starter/.env first.")
    return url


def vector_store_backend() -> str:
    backend = os.getenv("VECTOR_STORE", "chroma").strip().lower()
    if backend not in {"chroma", "pgvector"}:
        raise RuntimeError(f"VECTOR_STORE không hợp lệ: {backend} (chroma | pgvector)")
    return backend


def open_store(dimensions: int, model_name: str):
    """Build the configured vector store. Chroma (default) needs no server; pgvector needs Docker."""
    backend = vector_store_backend()
    if backend == "pgvector":
        from .store import PgVectorStore

        return PgVectorStore(database_url(), dimensions, model_name)
    from .store_chroma import ChromaStore

    persist_dir = os.getenv("CHROMA_DIR") or str(PATHS.results / "chroma")
    persist = Path(persist_dir)
    persist = persist if persist.is_absolute() else (REPO_ROOT / persist)
    persist.mkdir(parents=True, exist_ok=True)
    return ChromaStore(str(persist), dimensions, model_name)


def load_strategies(path: Path | None = None) -> dict[str, Any]:
    path = path or PATHS.strategies
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    strategies = config.get("strategies", {})
    if len(strategies) < 3:
        raise ValueError(f"{path}: cần ít nhất 3 cấu hình, hiện có {len(strategies)}")
    for name, cfg in strategies.items():
        if cfg.get("chunking") not in {"fixed", "paragraph"}:
            raise ValueError(f"Strategy {name}: chunking phải là 'fixed' hoặc 'paragraph'")
        cfg.setdefault("index", name)
        cfg.setdefault("use_filter", False)
    # Strategies sharing an index must chunk identically, otherwise the shared vectors are wrong.
    by_index: dict[str, dict[str, Any]] = {}
    for name, cfg in strategies.items():
        signature = chunk_signature(cfg)
        previous = by_index.setdefault(cfg["index"], signature)
        if previous != signature:
            raise ValueError(f"Strategy {name} dùng chung index '{cfg['index']}' nhưng tham số chunk khác")
    config.setdefault("top_k", 3)
    config.setdefault("coherence_sample_size", 10)
    return config


def chunk_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    keys = ("chunking", "chunk_size", "overlap", "max_size", "min_size")
    return {key: cfg.get(key) for key in keys}


def indexes(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Unique physical indexes to build (C reuses B's index when only the filter differs)."""
    result: dict[str, dict[str, Any]] = {}
    for cfg in config["strategies"].values():
        result.setdefault(cfg["index"], cfg)
    return result
