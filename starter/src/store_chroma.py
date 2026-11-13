"""Chroma store: persistent on-disk vectors, cosine search, metadata filters. No server/Docker.

Same interface as PgVectorStore so run.py uses either one. One Chroma collection per
strategy (A, B, C...). Chroma's HNSW is approximate; at this corpus size with so few
vectors it returns the exact nearest neighbours in practice, but the report must call it
approximate search. Vectors persist under results/chroma (PersistentClient).
"""
from __future__ import annotations

import re
from typing import Any

from .models import Chunk
from .store import content_hash  # reuse the identical content hash

_COLUMNS = {"document_id", "source", "page"}
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_INFO_ID = "__store_info__"


def _collection_name(strategy: str) -> str:
    if not _KEY.match(strategy):
        raise ValueError(f"Tên strategy không hợp lệ: {strategy!r}")
    return f"rag_{strategy}"


def _flatten(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool; drop None, stringify the rest."""
    flat: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        flat[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return flat


class ChromaStore:
    def __init__(self, persist_dir: str, dimensions: int, model_name: str) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions phải > 0")
        import chromadb

        self.dimensions = dimensions
        self.model_name = model_name
        self.persist_dir = persist_dir
        try:
            self.client = chromadb.PersistentClient(path=persist_dir)
            self._chroma_version = getattr(chromadb, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Không mở được Chroma tại {persist_dir}: {exc}") from exc

    def close(self) -> None:
        pass

    def __enter__(self) -> "ChromaStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ schema
    def _collection(self, strategy: str):
        return self.client.get_or_create_collection(
            _collection_name(strategy), metadata={"hnsw:space": "cosine", "embedding_model": self.model_name,
                                                  "dimensions": self.dimensions})

    def create_schema(self) -> None:
        # Guard against mixing embedding models in one store (same role as pgvector's info table).
        for collection in self.client.list_collections():
            meta = collection.metadata or {}
            if meta.get("embedding_model") not in (None, self.model_name):
                raise RuntimeError(
                    f"Chroma đang chứa vector của model {meta.get('embedding_model')}, khác model hiện tại "
                    f"{self.model_name}. Chạy `python -m src.run reset-db --yes` rồi ingest lại.")

    def reset(self) -> None:
        for collection in self.client.list_collections():
            self.client.delete_collection(collection.name)

    def versions(self) -> dict[str, str]:
        return {"chroma": self._chroma_version, "metric": "cosine", "search": "approximate (HNSW)"}

    # ------------------------------------------------------------------ writes
    def existing_hashes(self, strategy: str) -> dict[str, tuple[str, str]]:
        collection = self._collection(strategy)
        got = collection.get(include=["metadatas"])
        result: dict[str, tuple[str, str]] = {}
        for chunk_id, meta in zip(got["ids"], got["metadatas"]):
            if chunk_id == _INFO_ID:
                continue
            result[chunk_id] = (meta.get("_content_hash", ""), meta.get("_embedding_model", ""))
        return result

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunk nhưng {len(vectors)} vector")
        collection = self._collection(strategy)
        ids, embeddings, documents, metadatas = [], [], [], []
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.dimensions:
                raise ValueError(f"Vector {chunk.chunk_id} có {len(vector)} chiều, cần {self.dimensions}")
            ids.append(chunk.chunk_id)
            embeddings.append(list(vector))
            documents.append(chunk.text)
            metadatas.append({
                **_flatten(chunk.metadata),
                "document_id": chunk.document_id, "source": chunk.source, "page": int(chunk.page),
                "_content_hash": content_hash(chunk), "_embedding_model": self.model_name,
            })
        if not ids:
            return 0
        for i in range(0, len(ids), 256):  # upsert by stable chunk_id -> re-running never duplicates
            collection.upsert(ids=ids[i:i + 256], embeddings=embeddings[i:i + 256],
                              documents=documents[i:i + 256], metadatas=metadatas[i:i + 256])
        return len(ids)

    def delete_obsolete(self, strategy: str, keep_ids: set[str]) -> int:
        collection = self._collection(strategy)
        current = {cid for cid in collection.get(include=[])["ids"] if cid != _INFO_ID}
        stale = list(current - keep_ids)
        if stale:
            collection.delete(ids=stale)
        return len(stale)

    def duplicate_report(self, strategy: str) -> dict[str, int]:
        collection = self._collection(strategy)
        got = collection.get(include=["metadatas"])
        ids = [cid for cid in got["ids"] if cid != _INFO_ID]
        metas = [m for cid, m in zip(got["ids"], got["metadatas"]) if cid != _INFO_ID]
        spans = {(m.get("document_id"), m.get("page"), m.get("char_start"), m.get("char_end")) for m in metas}
        hashes = {m.get("_content_hash") for m in metas}
        # chunk_id is the Chroma primary id, so rows always equal distinct ids (no duplicates possible).
        return {"rows": len(ids), "distinct_chunk_ids": len(set(ids)), "distinct_spans": len(spans),
                "distinct_content_hashes": len(hashes), "duplicate_rows": len(ids) - len(spans)}

    # ------------------------------------------------------------------ search
    def _where(self, filters: dict[str, Any] | None) -> dict[str, Any] | None:
        clauses = []
        for key, value in (filters or {}).items():
            if not _KEY.match(key):
                raise ValueError(f"Tên trường filter không hợp lệ: {key!r}")
            values = value if isinstance(value, list) else [value]
            if key == "page":
                clauses.append({key: {"$in": [int(v) for v in values]}})
            else:
                clauses.append({key: {"$in": [str(v) for v in values]}})
        if not clauses:
            return None
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError(f"Query vector có {len(query_vector)} chiều, cần {self.dimensions}")
        collection = self._collection(strategy)
        result = collection.query(query_embeddings=[list(query_vector)], n_results=top_k,
                                  where=self._where(filters), include=["documents", "metadatas", "distances"])
        hits = []
        for chunk_id, text, meta, distance in zip(result["ids"][0], result["documents"][0],
                                                  result["metadatas"][0], result["distances"][0]):
            clean = {k: v for k, v in meta.items() if k not in {"_content_hash", "_embedding_model"}}
            hits.append({"chunk_id": chunk_id, "document_id": meta.get("document_id"),
                         "source": meta.get("source"), "page": int(meta.get("page")), "text": text,
                         "metadata": clean, "score": 1.0 - float(distance)})
        return hits

    def metadata_values(self, strategy: str, key: str) -> list[str]:
        if not _KEY.match(key):
            raise ValueError(f"Tên trường không hợp lệ: {key!r}")
        got = self._collection(strategy).get(include=["metadatas"])
        values = {str(m.get(key)) for cid, m in zip(got["ids"], got["metadatas"])
                  if cid != _INFO_ID and m.get(key) is not None}
        return sorted(values)
