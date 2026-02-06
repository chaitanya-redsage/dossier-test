import os
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from src.reviewer import run_review_comments
from groq import Groq
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "output" / "uploads"
OLD_PDF_DIR = BASE_DIR / "backend" / "old_pdfs"
COMMENTS_DIR = BASE_DIR / "backend" / "comments"
COMMENTS_META_DIR = BASE_DIR / "backend" / "comments_meta"
OLD_PDF_DIR.mkdir(parents=True, exist_ok=True)
COMMENTS_DIR.mkdir(parents=True, exist_ok=True)
COMMENTS_META_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI()

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


def _save_upload(file: UploadFile, suffix: str) -> Path:
    name = f"{uuid.uuid4().hex}{suffix}"
    path = UPLOAD_DIR / name
    with path.open("wb") as f:
        f.write(file.file.read())
    return path


COMMENTS_GLOBAL_FILE = COMMENTS_DIR / "all.txt"
COMMENTS_GLOBAL_META = COMMENTS_META_DIR / "all.json"


def _load_comments() -> list[str]:
    path = COMMENTS_GLOBAL_FILE
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _save_comments(comments: list[str]) -> None:
    path = COMMENTS_GLOBAL_FILE
    path.write_text("\n".join(comments) + "\n", encoding="utf-8")
    _update_comments_meta(last_edit_at=_now_iso())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_comments_meta() -> dict:
    path = COMMENTS_GLOBAL_META
    if not path.exists():
        return {}
    try:
        return __import__("json").loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _update_comments_meta(**kwargs) -> None:
    path = COMMENTS_GLOBAL_META
    meta = _load_comments_meta()
    meta.update(kwargs)
    path.write_text(__import__("json").dumps(meta, indent=2), encoding="utf-8")


def _merge_comments_semantic(comments: list[str]) -> list[str]:
    if not comments:
        return []
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    if not api_key:
        return comments

    prompt = (
        "You are deduplicating review comments. Merge comments that have the same semantic meaning "
        "into a single canonical comment. Return ONLY a JSON array of strings with the merged comments. "
        "Do not add or remove meaning; keep them concise.\n\nCOMMENTS:\n"
    )
    payload = "\n".join(f"- {c}" for c in comments)
    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a careful editor."},
            {"role": "user", "content": prompt + payload},
        ],
        temperature=0.1,
    )
    content = resp.choices[0].message.content or ""
    try:
        merged = __import__("json").loads(content)
        if isinstance(merged, list) and all(isinstance(x, str) for x in merged):
            return [m.strip() for m in merged if m.strip()]
    except Exception:
        return comments
    return comments


def _list_old_pdfs():
    return sorted([p.name for p in OLD_PDF_DIR.glob("*.pdf")])


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "result": None, "old_pdfs": _list_old_pdfs()},
    )


@app.get("/comments", response_class=HTMLResponse)
def comments_page(request: Request, old_pdf_name: Optional[str] = None):
    old_pdfs = _list_old_pdfs()
    selected = old_pdf_name if old_pdf_name in old_pdfs else (old_pdfs[0] if old_pdfs else None)
    comments = _load_comments()
    meta = _load_comments_meta()
    last_run_at = meta.get("last_run_at")
    last_edit_at = meta.get("last_edit_at")
    run_after_edit = False
    if last_run_at and last_edit_at:
        run_after_edit = last_run_at >= last_edit_at
    return templates.TemplateResponse(
        "comments.html",
        {
            "request": request,
            "old_pdf_name": selected,
            "old_pdfs": old_pdfs,
            "comments": comments,
            "last_run_at": last_run_at,
            "last_edit_at": last_edit_at,
            "run_after_edit": run_after_edit,
        },
    )


@app.get("/api/comments", response_class=JSONResponse)
def api_comments_list():
    return {"comments": _load_comments(), "meta": _load_comments_meta()}


@app.post("/api/comments", response_class=JSONResponse)
def api_comments_add(comment: str = Form(...)):
    new_comment = comment.strip()
    if not new_comment:
        raise HTTPException(status_code=400, detail="comment is required")
    comments = _load_comments()
    comments.append(new_comment)
    _save_comments(comments)
    return {"comments": comments, "meta": _load_comments_meta()}


@app.delete("/api/comments", response_class=JSONResponse)
def api_comments_delete(idx: int = Form(...)):
    comments = _load_comments()
    if not (0 <= idx < len(comments)):
        raise HTTPException(status_code=400, detail="idx out of range")
    comments.pop(idx)
    _save_comments(comments)
    return {"comments": comments, "meta": _load_comments_meta()}


@app.post("/api/comments/merge", response_class=JSONResponse)
def api_comments_merge():
    comments = _load_comments()
    merged = _merge_comments_semantic(comments)
    _save_comments(merged)
    return {"comments": merged, "meta": _load_comments_meta()}


@app.post("/comments/add", response_class=HTMLResponse)
def comments_add(request: Request, comment_text: str = Form(...)):
    comments = _load_comments()
    new_comment = comment_text.strip()
    if new_comment:
        comments.append(new_comment)
        _save_comments(comments)
    return comments_page(request)


@app.post("/comments/delete", response_class=HTMLResponse)
def comments_delete(request: Request, idx: int = Form(...)):
    comments = _load_comments()
    if 0 <= idx < len(comments):
        comments.pop(idx)
        _save_comments(comments)
    return comments_page(request)


@app.post("/comments/merge", response_class=HTMLResponse)
def comments_merge(request: Request):
    comments = _load_comments()
    merged = _merge_comments_semantic(comments)
    _save_comments(merged)
    return comments_page(request)


@app.post("/review", response_class=HTMLResponse)
def review(
    request: Request,
    comments_text: Optional[str] = Form(None),
    new_pdf: UploadFile = File(...),
    old_pdf_name: Optional[str] = Form(None),
    bypass_candidates: Optional[str] = Form(None),
):
    new_path = _save_upload(new_pdf, ".pdf")
    old_path = (OLD_PDF_DIR / old_pdf_name) if old_pdf_name else None

    comments_list = []
    if comments_text:
        new_comments = [c.strip() for c in comments_text.splitlines() if c.strip()]
        existing = _load_comments()
        combined = existing + new_comments
        # Deduplicate while preserving order
        seen = set()
        comments_list = []
        for c in combined:
            if c not in seen:
                comments_list.append(c)
                seen.add(c)
        _save_comments(comments_list)
    else:
        comments_list = _load_comments()

    result = run_review_comments(
        comments=comments_list,
        new_pdf=str(new_path),
        old_pdf=str(old_path) if old_path else None,
        bypass_candidate_pages=bool(bypass_candidates),
    )
    _update_comments_meta(last_run_at=_now_iso())

    overall_ok = result.get("summary", {}).get("not_fixed", 0) == 0
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result": result,
            "overall_ok": overall_ok,
            "old_pdfs": _list_old_pdfs(),
        },
    )


@app.post("/api/review", response_class=JSONResponse)
def api_review(
    comments_text: Optional[str] = Form(None),
    new_pdf: UploadFile = File(...),
    old_pdf_name: Optional[str] = Form(None),
    bypass_candidates: Optional[str] = Form(None),
):
    new_path = _save_upload(new_pdf, ".pdf")
    old_path = (OLD_PDF_DIR / old_pdf_name) if old_pdf_name else None

    if comments_text:
        new_comments = [c.strip() for c in comments_text.splitlines() if c.strip()]
        existing = _load_comments()
        combined = existing + new_comments
        # Deduplicate while preserving order
        seen = set()
        comments_list = []
        for c in combined:
            if c not in seen:
                comments_list.append(c)
                seen.add(c)
        _save_comments(comments_list)
    else:
        comments_list = _load_comments()

    result = run_review_comments(
        comments=comments_list,
        new_pdf=str(new_path),
        old_pdf=str(old_path) if old_path else None,
        bypass_candidate_pages=bool(bypass_candidates),
    )
    _update_comments_meta(last_run_at=_now_iso())
    overall_ok = result.get("summary", {}).get("not_fixed", 0) == 0
    return {"result": result, "overall_ok": overall_ok, "old_pdfs": _list_old_pdfs()}
