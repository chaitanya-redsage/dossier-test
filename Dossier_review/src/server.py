from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from expiry_pipeline_any_file import detect_expiry
from format_output import format_result_to_json
from spelling_reader import proofread_file

app = FastAPI()
UPLOAD_DIR = Path("uploads")
TEMPLATES_DIR = Path("templates")


@app.get("/")
def index() -> FileResponse:
    index_path = TEMPLATES_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(index_path)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    incoming = Path(file.filename).name
    suffix = Path(incoming).suffix or ".bin"
    stem = Path(incoming).stem or "upload"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    dest = UPLOAD_DIR / f"{stem}_{timestamp}{suffix}"

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        expiry = detect_expiry(str(dest))
        proofread = proofread_file(str(dest))
        merged = {**expiry, "proofread": proofread}
        return format_result_to_json(merged)
    finally:
        try:
            dest.unlink()
        except FileNotFoundError:
            pass


@app.post("/upload")
async def upload_form(file: UploadFile = File(...)) -> Dict[str, Any]:
    return await upload(file)
