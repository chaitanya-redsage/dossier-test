from __future__ import annotations

import os
import json
import uuid
from typing import Dict, Any, List, Optional, Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from db_connection import load_env
from schema_tools import get_schema_payload


def stable_uuid(text: str) -> str:
    """
    Deterministic UUID based on text (stable across runs).
    Qdrant Cloud requires id to be UUID or uint.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, text))


def load_table_metadata(path: str = "table_metadata.json") -> Dict[str, Any]:
    """
    Metadata format:
      {
        "schema.table": {"description": "...", "tags": ["..."]},
        ...
      }
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def schema_to_docs(schema_payload: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    schemas = schema_payload.get("schemas", {})

    for schema_name, schema_obj in schemas.items():
        tables = (schema_obj or {}).get("tables", {})
        for table_name, table_obj in tables.items():
            key = f"{schema_name}.{table_name}"
            meta_obj = metadata.get(key, {})
            description = meta_obj.get("description", "")
            tags = meta_obj.get("tags", [])

            cols = table_obj.get("columns", [])
            fks = table_obj.get("foreign_keys", [])

            col_lines = []
            for c in cols:
                nullable = "NULL" if c.get("nullable") else "NOT NULL"
                col_lines.append(f"{c.get('name')}: {c.get('type')} {nullable}")

            fk_lines = []
            for fk in fks:
                fk_lines.append(
                    f"{fk.get('column')} -> {fk.get('ref_schema')}.{fk.get('ref_table')}.{fk.get('ref_column')}"
                )

            text = (
                f"Schema: {schema_name}\n"
                f"Table: {table_name}\n"
                f"Description: {description}\n"
                f"Tags: {', '.join(tags)}\n"
                f"Columns:\n- " + "\n- ".join(col_lines) + "\n"
            )
            if fk_lines:
                text += "Foreign Keys:\n- " + "\n- ".join(fk_lines) + "\n"

            payload = {
                "doc_id": key,  # human-readable id
                "schema": schema_name,
                "table": table_name,
                "kind": "table",
                "description": description,
                "tags": tags,
                "columns": [c["name"] for c in cols],
                "foreign_keys": fk_lines,
            }

            docs.append({"id": key, "text": text, "payload": payload})

    return docs


def embed_texts(texts: List[str], model: SentenceTransformer, batch_size: int = 32) -> np.ndarray:
    vecs = model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def chunked(seq: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def index_all_schemas_to_one_collection(
    db_url: Optional[str] = None,
    schemas: Optional[List[str]] = None,
) -> None:
    """
    SAFE MODE:
      - Uses ONE collection for all schemas
      - Avoids get_collection() and create_payload_index() calls
        because some qdrant-client/pydantic combos crash parsing payload_schema.
    """
    load_env()

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", "db_schema_all")
    vector_name = os.getenv("QDRANT_VECTOR_NAME", "embedding")
    model_name = os.getenv("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5")

    if not qdrant_url:
        raise ValueError("Missing QDRANT_URL in .env")
    if not qdrant_api_key:
        raise ValueError("Missing QDRANT_API_KEY in .env")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    model = SentenceTransformer(model_name, trust_remote_code=True)

    schema_payload = get_schema_payload(schemas=schemas, db_url=db_url)
    metadata = load_table_metadata("table_metadata.json")
    docs = schema_to_docs(schema_payload, metadata)

    if not docs:
        raise RuntimeError("No schema docs produced")

    texts = [d["text"] for d in docs]
    vectors = embed_texts(texts, model=model, batch_size=32)
    dim = int(vectors.shape[1])

    # Only call get_collections() (this usually doesn't include payload_schema)
    existing = {c.name for c in client.get_collections().collections}
    if collection not in existing:
        client.create_collection(
            collection_name=collection,
            vectors_config={vector_name: VectorParams(size=dim, distance=Distance.COSINE)},
        )
        print(f"✅ created collection: {collection} (vector='{vector_name}', dim={dim})")
    else:
        print(f"ℹ️ using existing collection: {collection}")
        print("ℹ️ SAFE MODE: skipping dim validation and payload index creation")

    points: List[PointStruct] = []
    for d, vec in zip(docs, vectors):
        qdrant_id = stable_uuid(d["payload"]["schema"] + "::" + d["id"])
        points.append(
            PointStruct(
                id=qdrant_id,
                vector={vector_name: vec.tolist()},
                payload=d["payload"],
            )
        )

    total = 0
    for batch in chunked(points, size=512):
        client.upsert(collection_name=collection, points=batch)
        total += len(batch)
        print(f"✅ upserted batch: {len(batch)} (total={total}/{len(points)})")

    print(f"✅ done: upserted {len(points)} schema docs into Qdrant collection '{collection}'")


def index_schema_to_qdrant(
    db_url: Optional[str] = None,
    schemas: Optional[List[str]] = None,
) -> None:
    """
    Backwards-compatible alias for older imports.
    """
    return index_all_schemas_to_one_collection(db_url=db_url, schemas=schemas)


if __name__ == "__main__":
    index_all_schemas_to_one_collection(schemas=None)
