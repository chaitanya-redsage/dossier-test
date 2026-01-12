import os
from typing import List, Dict, Any, Optional

import numpy as np
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from db_connection import load_env


class SchemaRetriever:
    def __init__(self):
        load_env()

        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection = os.getenv("QDRANT_COLLECTION", "db_schema")
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

    def search(
        self,
        question: str,
        k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        qvec = self.embed(question)

        # Newer clients: query_points()
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name=self.collection,
                query=qvec,
                using=self.vector_name,      # named vector
                limit=k,
                with_payload=True,
                query_filter=filters,
            )
            points = res.points
            return [
                {
                    "id": p.id,
                    "score": float(p.score),
                    "payload": p.payload,
                }
                for p in points
            ]

        # Older clients: search()
        if hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=(self.vector_name, qvec),
                limit=k,
                with_payload=True,
                query_filter=filters,
            )
            return [
                {
                    "id": h.id,
                    "score": float(h.score),
                    "payload": h.payload,
                }
                for h in hits
            ]

        raise RuntimeError(
            "Your qdrant-client does not support query_points or search. "
            "Upgrade: pip install -U qdrant-client"
        )
