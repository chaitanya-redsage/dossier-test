from __future__ import annotations

import argparse
import base64
import json
import os
from io import BytesIO
from pathlib import Path

import fitz  # PyMuPDF
import requests
from PIL import Image

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_CHAT_COMPLETIONS = f"{GROQ_BASE_URL}/chat/completions"

DEFAULT_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

DEFAULT_PROMPT = """You are a vision flowchart parser.

First, verify the image actually contains a flowchart (boxes/diamonds/terminators connected by arrows).
If the image is a table, code listing, chart, or plain text with no flowchart, return:
{"nodes": [], "edges": []}

You MUST return STRICT JSON ONLY.
- No markdown
- No explanation
- No extra text

OUTPUT FORMAT (exact keys only):

{
  "nodes": [
    {"id": "n1", "text": "Start", "shape": "terminator|process|decision|input_output|unknown"}
  ],
  "edges": [
    {"from": "n1", "to": "n2", "label": ""}
  ]
}

RULES:
- Only include diagram boxes/diamonds/terminators.
- Ignore paragraphs or captions outside the flowchart.
- Node ids must be unique: n1, n2, n3...
- Node text must be short (max 60 characters).
- Preserve important symbols (A, B, Sum, A+B).
- Extract edges only when an arrow is clearly visible.
- If arrow label exists (Yes/No), include it in "label".
- If something is unreadable, use "[unreadable]" text.
- Do NOT guess or invent nodes/edges.

Return valid JSON starting with { and ending with }.
"""

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

        if quality > 60:
            quality -= 10
        else:
            max_dim = int(max_dim * 0.85)
            if max_dim < 800:
                raise ValueError("Image is too large to fit Groq base64 size limit.")


def _call_groq(*, data_url: str, prompt: str, model: str, api_key: str) -> str:
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


def _parse_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise ValueError(f"Model did not return valid JSON:\n{text}") from None


def _extract_from_image(p: Path, *, prompt: str, model: str, api_key: str) -> dict:
    img = Image.open(p)
    data_url = _image_to_data_url(img)
    content = _call_groq(data_url=data_url, prompt=prompt, model=model, api_key=api_key)
    return _parse_json(content)


def _extract_from_pdf(p: Path, *, prompt: str, model: str, api_key: str, dpi: int) -> dict:
    doc = fitz.open(str(p))
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    nodes: list[dict] = []
    edges: list[dict] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat)
        img = Image.open(BytesIO(pix.tobytes("png")))
        data_url = _image_to_data_url(img)
        content = _call_groq(data_url=data_url, prompt=prompt, model=model, api_key=api_key)
        page_graph = _parse_json(content)
        for n in page_graph.get("nodes", []):
            n["page"] = page_idx + 1
            nodes.append(n)
        for e in page_graph.get("edges", []):
            e["page"] = page_idx + 1
            edges.append(e)

    return {"nodes": nodes, "edges": edges}


def to_mermaid_markdown(graph: dict) -> str:
    lines = ["```mermaid", "flowchart TD"]

    for node in graph.get("nodes", []):
        nid = node.get("id")
        text = (node.get("text") or nid or "").replace('"', "'")
        if nid:
            lines.append(f'  {nid}["{text}"]')

    for edge in graph.get("edges", []):
        src = edge.get("from")
        dst = edge.get("to")
        if src and dst:
            lines.append(f"  {src} --> {dst}")

    lines.append("```")
    return "\n".join(lines) + "\n"


def extract_flowchart_graph(
    file_path: str | Path,
    *,
    model: str = DEFAULT_MODEL,
    prompt: str = DEFAULT_PROMPT,
    dpi: int = 200,
    api_key: str | None = None,
) -> dict:
    _load_env_file(Path(".env"))
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("Missing Groq API key. Set GROQ_API_KEY or pass --api-key.")

    p = Path(file_path)
    if not p.exists():
        raise SystemExit(f"Input not found: {p}")

    if p.suffix.lower() == ".pdf":
        return _extract_from_pdf(p, prompt=prompt, model=model, api_key=api_key, dpi=dpi)
    return _extract_from_image(p, prompt=prompt, model=model, api_key=api_key)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract flowchart nodes/edges via Groq Vision")
    parser.add_argument("input", help="Path to image or PDF")
    parser.add_argument("--output", help="Output Mermaid markdown file path")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model id")
    parser.add_argument("--dpi", type=int, default=200, help="PDF render DPI")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Flowchart prompt")
    parser.add_argument("--api-key", default=os.environ.get("GROQ_API_KEY"), help="Groq API key")
    args = parser.parse_args()

    graph = extract_flowchart_graph(
        args.input,
        model=args.model,
        prompt=args.prompt,
        dpi=args.dpi,
        api_key=args.api_key,
    )

    p = Path(args.input)
    out_path = Path(args.output) if args.output else p.with_suffix(".groq_flowchart.md")
    out_path.write_text(to_mermaid_markdown(graph), encoding="utf-8")
    print(f"[groq_flowchart] wrote {out_path}")


if __name__ == "__main__":
    main()
