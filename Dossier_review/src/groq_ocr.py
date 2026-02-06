from __future__ import annotations

import argparse
import base64
import os
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
import requests
from PIL import Image

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_CHAT_COMPLETIONS = f"{GROQ_BASE_URL}/chat/completions"

DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

OCR_PROMPT = (
    "You are an OCR engine. Extract all visible text from the image.\n"
    "Return plain text only.\n"
    "Preserve line breaks and reading order.\n"
    "Do not add commentary, formatting, or extra words."
)

MAX_BASE64_BYTES = 4 * 1024 * 1024  # Groq vision base64 limit (4 MB)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _image_to_data_url(img: Image.Image) -> str:
    img = img.convert("RGB")
    quality = 80
    max_dim = 1600
    current = img

    while True:
        if max(current.size) > max_dim:
            scale = max_dim / max(current.size)
            new_size = (int(current.size[0] * scale), int(current.size[1] * scale))
            current = current.resize(new_size, Image.LANCZOS)

        buf = BytesIO()
        current.save(buf, format="JPEG", quality=quality, optimize=True)
        data = base64.b64encode(buf.getvalue())

        if len(data) <= MAX_BASE64_BYTES:
            return f"data:image/jpeg;base64,{data.decode('ascii')}"

        # Reduce size and try again.
        if quality > 60:
            quality -= 10
        else:
            max_dim = int(max_dim * 0.85)
            if max_dim < 800:
                raise ValueError("Image is too large to fit Groq base64 size limit.")


def _call_groq_ocr(*, data_url: str, prompt: str, model: str, api_key: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(GROQ_CHAT_COMPLETIONS, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _ocr_image_file(p: Path, *, prompt: str, model: str, api_key: str) -> str:
    img = Image.open(p)
    data_url = _image_to_data_url(img)
    return _call_groq_ocr(data_url=data_url, prompt=prompt, model=model, api_key=api_key)


def _ocr_pdf_file(p: Path, *, prompt: str, model: str, api_key: str, dpi: int) -> str:
    doc = fitz.open(str(p))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pages_text: list[str] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(BytesIO(pix.tobytes("png")))
        data_url = _image_to_data_url(img)
        text = _call_groq_ocr(data_url=data_url, prompt=prompt, model=model, api_key=api_key)
        pages_text.append(f"--- Page {page_idx + 1} ---\n{text}")

    return "\n\n".join(pages_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR via Groq Llama 4 Scout Vision")
    parser.add_argument("input", help="Path to image or PDF")
    parser.add_argument("--output", help="Output text file path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model id")
    parser.add_argument("--dpi", type=int, default=200, help="PDF render DPI")
    parser.add_argument("--prompt", default=OCR_PROMPT, help="OCR prompt")
    parser.add_argument("--api-key", default=os.environ.get("GROQ_API_KEY"), help="Groq API key")
    args = parser.parse_args()

    _load_env_file(Path(".env"))
    if not args.api_key:
        args.api_key = os.environ.get("GROQ_API_KEY")

    if not args.api_key:
        raise SystemExit("Missing Groq API key. Set GROQ_API_KEY or pass --api-key.")

    p = Path(args.input)
    if not p.exists():
        raise SystemExit(f"Input not found: {p}")

    if p.suffix.lower() == ".pdf":
        text = _ocr_pdf_file(p, prompt=args.prompt, model=args.model, api_key=args.api_key, dpi=args.dpi)
    else:
        text = _ocr_image_file(p, prompt=args.prompt, model=args.model, api_key=args.api_key)

    out_path = Path(args.output) if args.output else p.with_suffix(".groq_ocr.txt")
    out_path.write_text(text, encoding="utf-8")
    print(f"[groq_ocr] wrote {out_path}")


if __name__ == "__main__":
    main()
