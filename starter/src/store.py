"""PostgreSQL + pgvector store: persistent vectors, cosine search, metadata filters.

One table holds every index, keyed by (strategy, chunk_id). Search is exact (no ANN
index, sequential scan with ORDER BY <=>), so a filter never silently drops results
the way post-filtering an approximate index can.
"""
from __future__ import annotations

import json
import re
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from .models import Chunk

TABLE = "rag_chunks"
INFO_TABLE = "rag_store_info"
_COLUMNS = {"document_id", "source", "page"}
_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{x:.7g}" for x in vector) + "]"


def content_hash(chunk: Chunk) -> str:
    import hashlib

    payload = json.dumps([chunk.text, chunk.source, chunk.page, chunk.metadata], ensure_ascii=False,
                         sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int, model_name: str) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions phải > 0")
        self.database_url = database_url
        self.dimensions = dimensions
        self.model_name = model_name
        try:
            # autocommit: every `with conn.transaction()` block is a real, committed transaction
            # (without it psycopg opens an implicit transaction that is rolled back on close).
            self.conn = psycopg.connect(database_url, connect_timeout=10, autocommit=True)
        except psycopg.Error as exc:
            raise RuntimeError("Không kết nối được PostgreSQL. Kiểm tra Docker và DATABASE_URL "
                               f"trong starter/.env. Chi tiết: {exc}") from exc

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "PgVectorStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ schema
    def create_schema(self) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"CREATE TABLE IF NOT EXISTS {INFO_TABLE} (key text PRIMARY KEY, value text NOT NULL)")
            cur.execute(f"SELECT key, value FROM {INFO_TABLE}")
            info = dict(cur.fetchall())
            if info and (info.get("dimensions") != str(self.dimensions) or info.get("embedding_model") != self.model_name):
                raise RuntimeError(
                    f"Database đang chứa vector của model {info.get('embedding_model')} ({info.get('dimensions')} chiều), "
                    f"khác model hiện tại {self.model_name} ({self.dimensions} chiều). "
                    "Chạy `python -m src.run reset-db --yes` rồi ingest lại.")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {TABLE} (
                    strategy        text        NOT NULL,
                    chunk_id        text        NOT NULL,
                    document_id     text        NOT NULL,
                    source          text        NOT NULL,
                    page            integer     NOT NULL,
                    text            text        NOT NULL,
                    content_hash    text        NOT NULL,
                    embedding_model text        NOT NULL,
                    metadata        jsonb       NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding       vector({self.dimensions}) NOT NULL,
                    updated_at      timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (strategy, chunk_id)
                )""")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_doc_idx ON {TABLE} (strategy, document_id, page)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS {TABLE}_meta_idx ON {TABLE} USING gin (metadata)")
            cur.execute(f"INSERT INTO {INFO_TABLE} VALUES ('dimensions', %s), ('embedding_model', %s), ('metric', 'cosine') "
                        "ON CONFLICT (key) DO NOTHING", (str(self.dimensions), self.model_name))

    def reset(self) -> None:
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
            cur.execute(f"DROP TABLE IF EXISTS {INFO_TABLE}")

    def versions(self) -> dict[str, str]:
        with self.conn.cursor() as cur:
            cur.execute("SELECT current_setting('server_version'), "
                        "(SELECT extversion FROM pg_extension WHERE extname = 'vector')")
            pg, vector = cur.fetchone()
        return {"postgresql": pg, "pgvector": vector or "missing"}

    # ------------------------------------------------------------------ writes
    def existing_hashes(self, strategy: str) -> dict[str, tuple[str, str]]:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT chunk_id, content_hash, embedding_model FROM {TABLE} WHERE strategy = %s", (strategy,))
            return {row[0]: (row[1], row[2]) for row in cur.fetchall()}

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> int:
        """Insert or update by stable (strategy, chunk_id); re-running never duplicates rows."""
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunk nhưng {len(vectors)} vector")
        rows = []
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.dimensions:
                raise ValueError(f"Vector {chunk.chunk_id} có {len(vector)} chiều, cần {self.dimensions}")
            rows.append((strategy, chunk.chunk_id, chunk.document_id, chunk.source, chunk.page, chunk.text,
                         content_hash(chunk), self.model_name, Jsonb(chunk.metadata), vector_literal(vector)))
        if not rows:
            return 0
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.executemany(f"""
                INSERT INTO {TABLE} (strategy, chunk_id, document_id, source, page, text, content_hash,
                                     embedding_model, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
                ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                    document_id = EXCLUDED.document_id, source = EXCLUDED.source, page = EXCLUDED.page,
                    text = EXCLUDED.text, content_hash = EXCLUDED.content_hash,
                    embedding_model = EXCLUDED.embedding_model, metadata = EXCLUDED.metadata,
                    embedding = EXCLUDED.embedding, updated_at = now()""", rows)
        return len(rows)

    def delete_obsolete(self, strategy: str, keep_ids: set[str]) -> int:
        """Remove chunks of this strategy that the current corpus no longer produces
        (changed/removed documents or pages)."""
        with self.conn.transaction(), self.conn.cursor() as cur:
            cur.execute(f"DELETE FROM {TABLE} WHERE strategy = %s AND NOT (chunk_id = ANY(%s))",
                        (strategy, list(keep_ids)))
            return cur.rowcount

    def duplicate_report(self, strategy: str) -> dict[str, int]:
        with self.conn.cursor() as cur:
            cur.execute(f"""
                SELECT count(*), count(DISTINCT chunk_id),
                       count(DISTINCT (document_id, page, metadata->>'char_start', metadata->>'char_end')),
                       count(DISTINCT content_hash)
                FROM {TABLE} WHERE strategy = %s""", (strategy,))
            rows, ids, spans, hashes = cur.fetchone()
        return {"rows": rows, "distinct_chunk_ids": ids, "distinct_spans": spans,
                "distinct_content_hashes": hashes, "duplicate_rows": rows - spans}

    # ------------------------------------------------------------------ search
    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Filter in SQL (parameterized), rank by cosine distance, return top-k with score = 1 - distance."""
        if len(query_vector) != self.dimensions:
            raise ValueError(f"Query vector có {len(query_vector)} chiều, cần {self.dimensions}")
        where = ["strategy = %s"]
        params: list[Any] = [strategy]
        for key, value in (filters or {}).items():
            if not _KEY.match(key):
                raise ValueError(f"Tên trường filter không hợp lệ: {key!r}")
            values = value if isinstance(value, list) else [value]
            if key in _COLUMNS:
                column = f"{key}::text"
                where.append(f"{column} = ANY(%s)")
            else:
                where.append("metadata ->> %s = ANY(%s)")
                params.append(key)
            params.append([str(v) for v in values])
        literal = vector_literal(query_vector)
        sql = (f"SELECT chunk_id, document_id, source, page, text, metadata, "
               f"1 - (embedding <=> %s::vector) AS score FROM {TABLE} "
               f"WHERE {' AND '.join(where)} ORDER BY embedding <=> %s::vector, chunk_id LIMIT %s")
        with self.conn.cursor() as cur:
            cur.execute(sql, [literal, *params, literal, top_k])
            rows = cur.fetchall()
        return [{"chunk_id": r[0], "document_id": r[1], "source": r[2], "page": r[3], "text": r[4],
                 "metadata": r[5], "score": float(r[6])} for r in rows]

    def metadata_values(self, strategy: str, key: str) -> list[str]:
        if not _KEY.match(key):
            raise ValueError(f"Tên trường không hợp lệ: {key!r}")
        with self.conn.cursor() as cur:
            if key in _COLUMNS:
                cur.execute(f"SELECT DISTINCT {key}::text FROM {TABLE} WHERE strategy = %s ORDER BY 1", (strategy,))
            else:
                cur.execute(f"SELECT DISTINCT metadata ->> %s FROM {TABLE} WHERE strategy = %s ORDER BY 1",
                            (key, strategy))
            return [row[0] for row in cur.fetchall() if row[0] is not None]
