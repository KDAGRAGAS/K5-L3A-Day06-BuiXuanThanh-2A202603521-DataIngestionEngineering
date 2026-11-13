"""Real embedding providers. No mock/random fallback: any provider error is raised.

EMBEDDING_PROVIDER=local              -> sentence-transformers model on this machine (cost 0)
EMBEDDING_PROVIDER=openai_compatible  -> POST {EMBEDDING_API_BASE}/embeddings
                                         (OpenAI, OpenRouter, Gemini OpenAI-compat endpoint)
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class LocalEmbedder:
    """sentence-transformers model, normalized vectors (cosine)."""

    provider = "local/sentence-transformers"

    def __init__(self, model_name: str, query_prefix: str = "", doc_prefix: str = "",
                 batch_size: int = 32, device: str | None = None) -> None:
        import sentence_transformers
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self.model, "get_sentence_embedding_dimension", None) or self.model.get_embedding_dimension
        self.dimensions = int(get_dim())
        self.max_seq_length = int(self.model.max_seq_length or 0)
        self.library_version = f"sentence-transformers {sentence_transformers.__version__}"
        self.cost_usd = 0.0
        self.tokens_used = 0

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(texts, batch_size=self.batch_size, normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=len(texts) > 200)
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.doc_prefix + t for t in texts]) if texts else []

    def embed_query(self, question: str) -> list[float]:
        return self._encode([self.query_prefix + question])[0]

    def count_tokens(self, texts: list[str], prefix: str | None = None) -> list[int]:
        prefix = self.doc_prefix if prefix is None else prefix
        encoded = self.model.tokenizer([prefix + t for t in texts], add_special_tokens=True,
                                       truncation=False)["input_ids"]
        return [len(ids) for ids in encoded]

    def describe(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model_name, "dimensions": self.dimensions,
                "max_seq_length_tokens": self.max_seq_length, "library": self.library_version,
                "query_prefix": self.query_prefix, "document_prefix": self.doc_prefix,
                "normalized": True, "cost_usd": self.cost_usd}


class OpenAICompatibleEmbedder:
    """Any endpoint implementing the OpenAI /embeddings API."""

    provider = "openai_compatible"

    def __init__(self, model_name: str, api_key: str, base_url: str, batch_size: int = 64,
                 price_per_1m_tokens: float = 0.0, query_prefix: str = "", doc_prefix: str = "",
                 dimensions: int | None = None) -> None:
        if not api_key:
            raise RuntimeError("EMBEDDING_API_KEY trống: điền vào starter/.env (không commit file này)")
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.price = price_per_1m_tokens
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix
        self.requested_dimensions = dimensions
        self.tokens_used = 0
        self.cost_usd = 0.0
        self.max_seq_length = 0
        self.library_version = "urllib (stdlib)"
        self.dimensions = len(self._post(["dimension probe"])[0])

    def _post(self, texts: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {"model": self.model_name, "input": texts}
        if self.requested_dimensions:
            payload["dimensions"] = self.requested_dimensions
        body = json.dumps(payload).encode("utf-8")
        for attempt in range(6):
            request = urllib.request.Request(
                f"{self.base_url}/embeddings", data=body, method="POST",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "replace")[:500]
                if exc.code in (429, 500, 502, 503, 504) and attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Embedding API lỗi HTTP {exc.code}: {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt < 5:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Không gọi được embedding API: {exc}") from exc
        items = sorted(data["data"], key=lambda item: item["index"])
        if len(items) != len(texts):
            raise RuntimeError(f"API trả {len(items)} vector cho {len(texts)} input")
        usage = data.get("usage") or {}
        tokens = int(usage.get("total_tokens") or usage.get("prompt_tokens") or 0)
        self.tokens_used += tokens
        self.cost_usd = self.tokens_used / 1_000_000 * self.price
        return [item["embedding"] for item in items]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._post([self.doc_prefix + t for t in texts[i:i + self.batch_size]]))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        return self._post([self.query_prefix + question])[0]

    def count_tokens(self, texts: list[str], prefix: str | None = None) -> list[int] | None:
        return None  # provider tokenizer unknown; usage.total_tokens is reported instead

    def describe(self) -> dict[str, Any]:
        return {"provider": f"{self.provider} ({self.base_url})", "model": self.model_name,
                "dimensions": self.dimensions, "library": self.library_version,
                "query_prefix": self.query_prefix, "document_prefix": self.doc_prefix,
                "tokens_used": self.tokens_used, "price_per_1m_tokens_usd": self.price,
                "cost_usd": round(self.cost_usd, 6)}


def make_embedder() -> LocalEmbedder | OpenAICompatibleEmbedder:
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    model = os.getenv("EMBEDDING_MODEL", "").strip()
    if not model:
        raise RuntimeError("EMBEDDING_MODEL trống trong starter/.env")
    query_prefix = os.getenv("EMBEDDING_QUERY_PREFIX", "")
    doc_prefix = os.getenv("EMBEDDING_DOC_PREFIX", "")
    if provider == "local":
        return LocalEmbedder(model, query_prefix, doc_prefix,
                             batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "32")),
                             device=os.getenv("EMBEDDING_DEVICE") or None)
    if provider == "openai_compatible":
        dims = os.getenv("EMBEDDING_DIMENSIONS")
        return OpenAICompatibleEmbedder(
            model, os.getenv("EMBEDDING_API_KEY", ""),
            os.getenv("EMBEDDING_API_BASE", "https://api.openai.com/v1"),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "64")),
            price_per_1m_tokens=float(os.getenv("EMBEDDING_PRICE_PER_1M_TOKENS", "0") or 0),
            query_prefix=query_prefix, doc_prefix=doc_prefix,
            dimensions=int(dims) if dims else None)
    raise RuntimeError(f"EMBEDDING_PROVIDER không hợp lệ: {provider} (local | openai_compatible)")


# Name kept from the starter skeleton.
MyEmbedder = LocalEmbedder
