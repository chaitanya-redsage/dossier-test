from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchAny,
    MatchValue,
)

from db_connection import load_env


class SchemaRetriever:
    def __init__(self):
        load_env()

        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection = os.getenv("QDRANT_COLLECTION", "db_schema_all")
        self.vector_name = os.getenv("QDRANT_VECTOR_NAME", "embedding")
        self.model_name = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")

        if not self.url:
            raise ValueError("Missing QDRANT_URL in .env")
        if not self.api_key:
            raise ValueError("Missing QDRANT_API_KEY in .env")

        self.client = QdrantClient(url=self.url, api_key=self.api_key)
        self.model = SentenceTransformer(self.model_name, trust_remote_code=True)

    def embed(self, text: str) -> List[float]:
        vec = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(vec[0], dtype=np.float32).tolist()

    def build_filter(
        self,
        schemas: Optional[List[str]] = None,
        kind: Optional[str] = "table",
    ) -> Optional[Filter]:
        must: List[FieldCondition] = []

        if schemas:
            must.append(
                FieldCondition(
                    key="schema",
                    match=MatchAny(any=schemas),
                )
            )

        if kind:
            must.append(
                FieldCondition(
                    key="kind",
                    match=MatchValue(value=kind),
                )
            )

        return Filter(must=must) if must else None

    def search(
        self,
        question: str,
        k: int = 10,
        schemas: Optional[List[str]] = None,
        kind: Optional[str] = "table",
    ) -> List[Dict[str, Any]]:
        qvec = self.embed(question)
        qfilter = self.build_filter(schemas=schemas, kind=kind)

        # Newer clients: query_points()
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection,
                query=qvec,
                using=self.vector_name,
                limit=k,
                with_payload=True,
                query_filter=qfilter,
            )
            points = res.points
            return [
                {"id": p.id, "score": float(p.score), "payload": p.payload}
                for p in points
            ]

        # Older clients: search()
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=(self.vector_name, qvec),
                limit=k,
                with_payload=True,
                query_filter=qfilter,
            )
            return [
                {"id": h.id, "score": float(h.score), "payload": h.payload}
                for h in hits
            ]

        raise RuntimeError(
            "Your qdrant-client does not support query_points or search. "
            "Upgrade: pip install -U qdrant-client"
        )
