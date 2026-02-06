# PDF Reviewer (Groq + RAG)

This tool reviews a **new PDF** against prior review comments using a RAG flow:
1. The PDF is chunked into word-based snippets.
2. Chunks are embedded with `sentence-transformers`.
3. Embeddings are stored in Qdrant and semantically searched per comment.
4. The top matching snippets are sent to Groq to decide if each issue is fixed.

You can use it as a CLI tool, a web app, or via JSON API endpoints.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your Groq key and RAG settings in `.env`:

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.1-70b-versatile
QDRANT_URL=http://localhost:6333
EMBEDDING_MODEL=all-MiniLM-L6-v2
RAG_TOP_K=8
COMMENTS_COLLECTION=pdf_review_comments
COMMENTS_DEDUPE_THRESHOLD=0.9
OCR_ALWAYS=1
OCR_LANG=eng
OCR_DPI=220
OCR_PSM=4
```

## Inputs

- `comments.json`: prior comments (messy/unstructured is OK)
- `old.pdf`: the PDF those comments refer to (optional but recommended)
- `new.pdf`: the PDF to verify

The comments file can be any of these shapes:

```json
[
  "Fix the caption on page 4",
  {"comment": "Update the executive summary numbers"},
  {"issue": "Spelling of Acme is wrong"}
]
```

The tool will normalize each entry into a single string and run LLM checks.

## Run

```bash
python src/reviewer.py \
  --comments comments.json \
  --old old.pdf \
  --new new.pdf \
  --out output/report.json
```

If you don't have an old PDF, you can omit `--old`.

## Local web app

```bash
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` and upload your new PDF, then optionally add comments as text.

Old PDFs are stored server-side in `backend/old_pdfs/`. Put any reference PDFs there and select them from the dropdown.

Comments are stored server-side in `backend/comments/all.txt` and metadata in `backend/comments_meta/all.json`. If you leave the comments box empty, the saved comments are used automatically. If you add comments in the review form, they are appended and deduped.

## RAG dependencies

You must have a Qdrant instance running at `QDRANT_URL` and the sentence-transformers model available.

To start Qdrant locally with Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

Comment embeddings are stored in the Qdrant collection named by `COMMENTS_COLLECTION` (defaults to `pdf_review_comments`).
Semantic dedupe uses `COMMENTS_DEDUPE_THRESHOLD` (cosine similarity). Set to `0` to disable.

## OCR

OCR runs before chunking when `OCR_ALWAYS=1`. It uses `file_extract_router.py`, which can run
Tesseract OCR and produce per-page text that is then chunked and indexed.

Install Tesseract on macOS (Homebrew):

```bash
brew install tesseract
```

If OCR is disabled or Tesseract is missing, the app falls back to PDF text extraction.

## JSON API

The HTML UI is backed by FastAPI, and you can also use JSON endpoints:

- `GET /api/comments`: list comments and metadata
- `POST /api/comments` (form field `comment`): add a comment
- `DELETE /api/comments` (form field `idx`): delete by index
- `POST /api/comments/merge`: semantic merge of saved comments
- `POST /api/review` (multipart form):
  - `new_pdf`: PDF file (required)
  - `comments_text`: optional text, one comment per line
  - `old_pdf_name`: optional, name of a PDF under `backend/old_pdfs`
  - `bypass_candidates`: optional checkbox to send all pages

Responses are JSON and include the `result` object and `overall_ok` status.

## Output

`output/report.json` contains:
- `status`: fixed / not_fixed / unclear
- `location`: page + evidence snippet
- `reason`: short explanation
