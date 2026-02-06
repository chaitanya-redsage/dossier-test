import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
from dotenv import load_dotenv
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from src.file_extract_router import extract_pdf_pages


@dataclass
class PageText:
    page: int
    text: str


def load_pdf_pages(path: str) -> List[PageText]:
    ocr_pages = extract_pdf_pages(path, ocr_lang=os.getenv("OCR_LANG", "eng"))
    if ocr_pages:
        return [PageText(page=p["page"], text=p["text"]) for p in ocr_pages]
    doc = fitz.open(path)
    pages: List[PageText] = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        pages.append(PageText(page=i + 1, text=text))
    return pages




def normalize_comment(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        parts = []
        for k, v in item.items():
            if isinstance(v, str):
                parts.append(v.strip())
            elif isinstance(v, (int, float)):
                parts.append(f"{k}: {v}")
            elif v is not None:
                parts.append(str(v))
        return " | ".join([p for p in parts if p])
    return str(item).strip()


@dataclass
class TextChunk:
    chunk_id: int
    page: int
    text: str


def page_snippet(pages: List[PageText], page_num: int, limit: int = 1200) -> str:
    for p in pages:
        if p.page == page_num:
            t = p.text.strip()
            return t[:limit]
    return ""


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    # Find first JSON object in the response without regex
    start = text.find("{")
    if start == -1:
        return None
    end = text.rfind("}")
    if end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except Exception:
        return None


def build_prompt(
    comment: str,
    new_snippets: List[Tuple[int, str]],
    old_snippets: Optional[List[Tuple[int, str]]],
) -> str:
    blocks = []
    blocks.append("You are a strict PDF reviewer. Determine if the issue in the comment is fixed in the NEW PDF.\n")
    blocks.append(f"COMMENT: {comment}\n")

    if old_snippets:
        blocks.append("OLD PDF SNIPPETS (for reference):\n")
        for page, snippet in old_snippets:
            if snippet:
                blocks.append(f"[Old Page {page}] {snippet}\n")

    blocks.append("NEW PDF SNIPPETS (to verify):\n")
    for page, snippet in new_snippets:
        if snippet:
            blocks.append(f"[New Page {page}] {snippet}\n")

    blocks.append(
        "Return ONLY a JSON object with keys: "
        "status (fixed|not_fixed|unclear), page (number or null), "
        "location (short phrase), evidence (short snippet), "
        "what_was_wrong (short phrase), reason (1-2 sentences), confidence (0-1).\n"
        "Rules:\n"
        "- If status is fixed or not_fixed, page MUST be a number.\n"
        "- If the issue is missing or cannot be found in the NEW PDF, set status=unclear, page=null, "
        "and reason must say it is not present in the NEW PDF.\n"
        "- Always cite the page for fixed/not_fixed in evidence/location."
    )

    return "\n".join(blocks)


def _normalize_comments_input(raw: Any) -> List[str]:
    if isinstance(raw, list):
        comments = [normalize_comment(x) for x in raw]
    else:
        comments = [normalize_comment(raw)]
    return [c for c in comments if c]


def _chunk_pages(pages: List[PageText], chunk_size: int = 200, overlap: int = 40) -> List[TextChunk]:
    chunks: List[TextChunk] = []
    step = max(1, chunk_size - overlap)
    chunk_id = 1
    for p in pages:
        tokens = p.text.split()
        if not tokens:
            continue
        for start in range(0, len(tokens), step):
            end = min(len(tokens), start + chunk_size)
            chunk_text = " ".join(tokens[start:end]).strip()
            if chunk_text:
                chunks.append(TextChunk(chunk_id=chunk_id, page=p.page, text=chunk_text))
                chunk_id += 1
    return chunks


def _index_chunks(
    client: QdrantClient,
    collection_name: str,
    chunks: List[TextChunk],
    embeddings: List[List[float]],
) -> None:
    if not chunks:
        return
    vector_size = len(embeddings[0])
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )
    points = [
        PointStruct(
            id=chunk.chunk_id,
            vector=vector,
            payload={"page": chunk.page, "text": chunk.text},
        )
        for chunk, vector in zip(chunks, embeddings)
    ]
    client.upsert(collection_name=collection_name, points=points)


def _ensure_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    if client.collection_exists(collection_name):
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def _store_comment_embeddings(
    client: QdrantClient,
    collection_name: str,
    comments: List[str],
    embeddings: List[List[float]],
    dedupe_threshold: float,
) -> None:
    if not comments:
        return
    now_iso = datetime.now(timezone.utc).isoformat()
    points = []
    for comment, vector in zip(comments, embeddings):
        if dedupe_threshold > 0:
            response = client.query_points(
                collection_name=collection_name,
                query=vector,
                limit=1,
                with_payload=False,
                score_threshold=dedupe_threshold,
            )
            if response.points:
                continue
        cid = uuid.uuid5(uuid.NAMESPACE_URL, comment)
        points.append(
            PointStruct(
                id=cid,
                vector=vector,
                payload={"comment": comment, "created_at": now_iso, "source": "review"},
            )
        )
    if not points:
        return
    client.upsert(collection_name=collection_name, points=points)


def _search_snippets(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int,
    snippet_limit: int = 1200,
) -> List[Tuple[int, str]]:
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    hits = response.points or []
    results: List[Tuple[int, str]] = []
    for hit in hits:
        payload = hit.payload or {}
        page = payload.get("page")
        text = payload.get("text") or ""
        if isinstance(page, int) and text:
            results.append((page, text[:snippet_limit]))
    return results


def run_review_data(
    comments_path: str,
    new_pdf: str,
    old_pdf: Optional[str],
) -> Dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    rag_top_k = int(os.getenv("RAG_TOP_K", "8"))
    comments_collection = os.getenv("COMMENTS_COLLECTION", "pdf_review_comments")
    comments_dedupe_threshold = float(os.getenv("COMMENTS_DEDUPE_THRESHOLD", "0.9"))
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set in the environment or .env")

    with open(comments_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    comments = _normalize_comments_input(raw)

    new_pages = load_pdf_pages(new_pdf)
    old_pages = load_pdf_pages(old_pdf) if old_pdf else None

    client = Groq(api_key=api_key)
    qdrant = QdrantClient(url=qdrant_url)
    embedder = SentenceTransformer(embedding_model_name)

    new_chunks = _chunk_pages(new_pages)
    new_texts = [c.text for c in new_chunks]
    new_collection = f"pdf_review_new_{uuid.uuid4().hex}"
    old_collection = None
    if new_texts:
        new_vectors = embedder.encode(new_texts, normalize_embeddings=True).tolist()
        _index_chunks(qdrant, new_collection, new_chunks, new_vectors)

    if comments:
        vector_size = embedder.get_sentence_embedding_dimension()
        _ensure_collection(qdrant, comments_collection, vector_size)
        comment_vectors = embedder.encode(comments, normalize_embeddings=True).tolist()
        _store_comment_embeddings(
            qdrant, comments_collection, comments, comment_vectors, comments_dedupe_threshold
        )

    old_chunks: List[TextChunk] = []
    if old_pages:
        old_chunks = _chunk_pages(old_pages)
        old_texts = [c.text for c in old_chunks]
        if old_texts:
            old_collection = f"pdf_review_old_{uuid.uuid4().hex}"
            old_vectors = embedder.encode(old_texts, normalize_embeddings=True).tolist()
            _index_chunks(qdrant, old_collection, old_chunks, old_vectors)

    results = []
    try:
        for idx, comment in enumerate(comments, start=1):
            if not comment:
                continue
            query_vec = embedder.encode([comment], normalize_embeddings=True)[0].tolist()
            new_snippets = _search_snippets(qdrant, new_collection, query_vec, rag_top_k) if new_texts else []
            old_snippets = (
                _search_snippets(qdrant, old_collection, query_vec, rag_top_k) if old_collection else []
            )
            prompt = build_prompt(comment, new_snippets, old_snippets)

            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a careful document QA reviewer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )

            content = resp.choices[0].message.content or ""
            parsed = extract_json(content)
            if not parsed:
                parsed = {
                    "status": "unclear",
                    "page": None,
                    "location": "",
                    "evidence": "",
                    "what_was_wrong": "",
                    "reason": "Model did not return valid JSON.",
                    "confidence": 0.0,
                }

            parsed["comment"] = comment
            parsed["candidates"] = [p for p, _ in new_snippets]
            results.append(parsed)
    finally:
        if new_texts:
            qdrant.delete_collection(new_collection)
        if old_collection:
            qdrant.delete_collection(old_collection)

    summary = {
        "total": len(results),
        "fixed": sum(1 for r in results if r.get("status") == "fixed"),
        "not_fixed": sum(1 for r in results if r.get("status") == "not_fixed"),
        "unclear": sum(1 for r in results if r.get("status") == "unclear"),
        "not_fixed_comments": [r.get("comment") for r in results if r.get("status") == "not_fixed"],
    }

    return {"summary": summary, "results": results}


def run_review_comments(
    comments: List[str],
    new_pdf: str,
    old_pdf: Optional[str],
    bypass_candidate_pages: bool = False,
) -> Dict[str, Any]:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    embedding_model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    rag_top_k = int(os.getenv("RAG_TOP_K", "8"))
    comments_collection = os.getenv("COMMENTS_COLLECTION", "pdf_review_comments")
    comments_dedupe_threshold = float(os.getenv("COMMENTS_DEDUPE_THRESHOLD", "0.9"))
    if not api_key:
        raise SystemExit("GROQ_API_KEY is not set in the environment or .env")

    comments = _normalize_comments_input(comments)
    if not comments:
        return {"summary": {"total": 0, "fixed": 0, "not_fixed": 0, "unclear": 0}, "results": []}

    new_pages = load_pdf_pages(new_pdf)
    old_pages = load_pdf_pages(old_pdf) if old_pdf else None
    client = Groq(api_key=api_key)
    qdrant = QdrantClient(url=qdrant_url)
    embedder = SentenceTransformer(embedding_model_name)

    new_chunks = _chunk_pages(new_pages)
    new_texts = [c.text for c in new_chunks]
    new_collection = f"pdf_review_new_{uuid.uuid4().hex}"
    old_collection = None
    if new_texts:
        new_vectors = embedder.encode(new_texts, normalize_embeddings=True).tolist()
        _index_chunks(qdrant, new_collection, new_chunks, new_vectors)

    if comments:
        vector_size = embedder.get_sentence_embedding_dimension()
        _ensure_collection(qdrant, comments_collection, vector_size)
        comment_vectors = embedder.encode(comments, normalize_embeddings=True).tolist()
        _store_comment_embeddings(
            qdrant, comments_collection, comments, comment_vectors, comments_dedupe_threshold
        )

    old_chunks: List[TextChunk] = []
    if old_pages:
        old_chunks = _chunk_pages(old_pages)
        old_texts = [c.text for c in old_chunks]
        if old_texts:
            old_collection = f"pdf_review_old_{uuid.uuid4().hex}"
            old_vectors = embedder.encode(old_texts, normalize_embeddings=True).tolist()
            _index_chunks(qdrant, old_collection, old_chunks, old_vectors)

    results = []
    try:
        for comment in comments:
            if bypass_candidate_pages:
                new_snippets = [(p.page, page_snippet(new_pages, p.page)) for p in new_pages]
                old_snippets = (
                    [(p.page, page_snippet(old_pages, p.page)) for p in old_pages] if old_pages else []
                )
            else:
                query_vec = embedder.encode([comment], normalize_embeddings=True)[0].tolist()
                new_snippets = _search_snippets(qdrant, new_collection, query_vec, rag_top_k) if new_texts else []
                old_snippets = (
                    _search_snippets(qdrant, old_collection, query_vec, rag_top_k) if old_collection else []
                )
            prompt = build_prompt(comment, new_snippets, old_snippets)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a careful document QA reviewer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            content = resp.choices[0].message.content or ""
            parsed = extract_json(content)
            if not parsed:
                parsed = {
                    "status": "unclear",
                    "page": None,
                    "location": "",
                    "evidence": "",
                    "what_was_wrong": "",
                    "reason": "Model did not return valid JSON.",
                    "confidence": 0.0,
                }
            parsed["comment"] = comment
            parsed["candidates"] = [p for p, _ in new_snippets]
            results.append(parsed)
    finally:
        if new_texts:
            qdrant.delete_collection(new_collection)
        if old_collection:
            qdrant.delete_collection(old_collection)

    summary = {
        "total": len(results),
        "fixed": sum(1 for r in results if r.get("status") == "fixed"),
        "not_fixed": sum(1 for r in results if r.get("status") == "not_fixed"),
        "unclear": sum(1 for r in results if r.get("status") == "unclear"),
        "not_fixed_comments": [r.get("comment") for r in results if r.get("status") == "not_fixed"],
    }

    return {"summary": summary, "results": results}


def run_review(
    comments_path: str,
    new_pdf: str,
    old_pdf: Optional[str],
    out_path: str,
) -> None:
    data = run_review_data(comments_path, new_pdf, old_pdf)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF review checker with Groq")
    parser.add_argument("--comments", required=True, help="Path to comments JSON")
    parser.add_argument("--new", required=True, help="Path to new PDF")
    parser.add_argument("--old", required=False, help="Path to old PDF")
    parser.add_argument("--out", required=True, help="Output report JSON path")
    args = parser.parse_args()

    run_review(args.comments, args.new, args.old, args.out)


if __name__ == "__main__":
    main()
