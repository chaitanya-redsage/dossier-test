from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import pytesseract


def extract_flowchart_graph(
    file_path: str | Path,
    *,
    dpi: int = 300,
    ocr_lang: str = "eng",
) -> dict:
    """
    Extract a flowchart graph from a PDF or image.
    Returns {"nodes": [...], "edges": [...], "pages": int}.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    if p.suffix.lower() == ".pdf":
        print(f"[flowchart] input=pdf path={p} dpi={dpi}")
        return _extract_from_pdf(p, dpi=dpi, ocr_lang=ocr_lang)
    print(f"[flowchart] input=image path={p}")
    return _extract_from_image(p, page_no=1, ocr_lang=ocr_lang)


def _extract_from_pdf(p: Path, *, dpi: int, ocr_lang: str) -> dict:
    print("[flowchart] open pdf")
    doc = fitz.open(str(p))
    nodes: list[dict] = []
    edges: list[dict] = []
    page_count = len(doc)
    print(f"[flowchart] pages={page_count}")

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_idx in range(page_count):
        print(f"[flowchart] render page {page_idx + 1}/{page_count}")
        page = doc[page_idx]
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        if pix.n == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        print(f"[flowchart] extract page {page_idx + 1} nodes/edges")
        page_nodes, page_edges = _extract_from_bgr_image(
            img, page_no=page_idx + 1, ocr_lang=ocr_lang
        )
        nodes.extend(page_nodes)
        edges.extend(page_edges)

    print("[flowchart] close pdf")
    doc.close()
    return {"nodes": nodes, "edges": edges, "pages": page_count}


def _extract_from_image(p: Path, *, page_no: int, ocr_lang: str) -> dict:
    img = cv2.imread(str(p))
    if img is None:
        raise RuntimeError(f"Unable to read image: {p}")
    nodes, edges = _extract_from_bgr_image(img, page_no=page_no, ocr_lang=ocr_lang)
    return {"nodes": nodes, "edges": edges, "pages": 1}


def _extract_from_bgr_image(
    img: np.ndarray, *, page_no: int, ocr_lang: str
) -> tuple[list[dict], list[dict]]:
    print(f"[flowchart] page {page_no}: preprocess")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 5
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(f"[flowchart] page {page_no}: contours={len(contours)}")

    nodes: list[dict] = []
    arrowheads: list[dict] = []
    img_h, img_w = gray.shape[:2]
    min_area = max(300, int(img_w * img_h * 0.0005))
    max_area = int(img_w * img_h * 0.6)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        if len(approx) == 3 and area < max(6000, min_area * 8):
            tip = _triangle_tip(approx.reshape(-1, 2))
            arrowheads.append({"tip": tip})
            continue
        x, y, w, h = cv2.boundingRect(approx)
        if w < 20 or h < 20:
            continue
        rect_area = w * h
        if rect_area <= 0:
            continue
        fill_ratio = area / rect_area
        aspect = w / h if h else 0
        if fill_ratio < 0.12:
            continue
        if aspect < 0.15 or aspect > 8.0:
            continue
        nodes.append({"bbox": [x, y, w, h], "page": page_no})

    if len(nodes) < 3:
        print(f"[flowchart] page {page_no}: fallback to OCR text boxes")
        ocr_nodes = _ocr_text_nodes(img, ocr_lang)
        if ocr_nodes:
            nodes = _merge_nodes(nodes, ocr_nodes)

    print(f"[flowchart] page {page_no}: nodes={len(nodes)} arrowheads={len(arrowheads)}")
    # OCR inside each node bbox.
    for idx, node in enumerate(nodes, start=1):
        if idx == 1 or idx % 5 == 0:
            print(f"[flowchart] page {page_no}: ocr node {idx}/{len(nodes)}")
        x, y, w, h = node["bbox"]
        crop = img[y : y + h, x : x + w]
        text = _ocr_node_text(crop, ocr_lang)
        text = _clean_text(text)
        if len(text) < 2:
            text = ""
        node.update({"id": f"n{page_no}_{idx}", "text": text})

    print(f"[flowchart] page {page_no}: detect edges")
    edges = _detect_edges(img, nodes, arrowheads, page_no=page_no)
    print(f"[flowchart] page {page_no}: edges={len(edges)}")
    return nodes, edges


def _detect_edges(
    img: np.ndarray, nodes: list[dict], arrowheads: list[dict], *, page_no: int
) -> list[dict]:
    print(f"[flowchart] page {page_no}: canny/hough")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 200)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=40, maxLineGap=10)

    if lines is None:
        return []
    print(f"[flowchart] page {page_no}: lines={len(lines)}")

    arrow_matches = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        line = {"start": (x1, y1), "end": (x2, y2)}
        match = _match_arrowhead(line, arrowheads)
        if match:
            arrow_matches.append(match)

    graph_edges: list[dict] = []
    for match in arrow_matches:
        start_pt, end_pt = match["start"], match["end"]
        src = _nearest_node(start_pt, nodes)
        dst = _nearest_node(end_pt, nodes)
        if not src or not dst or src["id"] == dst["id"]:
            continue
        graph_edges.append(
            {
                "from": src["id"],
                "to": dst["id"],
                "page": page_no,
            }
        )
    if graph_edges:
        return graph_edges

    # Fallback: connect line endpoints to nearest nodes when arrowheads are missing.
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        start = (x1, y1)
        end = (x2, y2)
        src = _nearest_node_edge(start, nodes, max_dist=40)
        dst = _nearest_node_edge(end, nodes, max_dist=40)
        if not src or not dst or src["id"] == dst["id"]:
            continue
        graph_edges.append({"from": src["id"], "to": dst["id"], "page": page_no})
    return graph_edges


def _match_arrowhead(line: dict, arrowheads: list[dict]) -> dict | None:
    start = line["start"]
    end = line["end"]
    best = None
    best_dist = 1e9
    for ah in arrowheads:
        tip = ah["tip"]
        d_start = _dist(start, tip)
        d_end = _dist(end, tip)
        d_min = min(d_start, d_end)
        if d_min < 30 and d_min < best_dist:
            if d_start < d_end:
                best = {"start": end, "end": start}
            else:
                best = {"start": start, "end": end}
            best_dist = d_min
    return best


def _nearest_node(point: tuple[int, int], nodes: list[dict]) -> dict | None:
    px, py = point
    best = None
    best_dist = 1e9
    for node in nodes:
        x, y, w, h = node["bbox"]
        if x <= px <= x + w and y <= py <= y + h:
            return node
        cx, cy = x + w / 2, y + h / 2
        d = ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5
        if d < best_dist:
            best_dist = d
            best = node
    if best_dist > 140:
        return None
    return best


def _triangle_tip(pts: np.ndarray) -> tuple[int, int]:
    a, b, c = pts
    d_ab = _dist(a, b)
    d_bc = _dist(b, c)
    d_ca = _dist(c, a)
    if d_ab >= d_bc and d_ab >= d_ca:
        return tuple(c)
    if d_bc >= d_ab and d_bc >= d_ca:
        return tuple(a)
    return tuple(b)


def _dist(p1: tuple[int, int], p2: tuple[int, int]) -> float:
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _clean_text(text: str) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " ".join(lines)


def _ocr_node_text(crop: np.ndarray, ocr_lang: str) -> str:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 50, 50)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bin_img) < 127:
        bin_img = 255 - bin_img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    config = "--psm 6"
    return pytesseract.image_to_string(bin_img, lang=ocr_lang, config=config)


def _point_inside_node(point: tuple[int, int], node: dict) -> bool:
    x, y, w, h = node["bbox"]
    px, py = point
    return x <= px <= x + w and y <= py <= y + h


def _nearest_node_edge(
    point: tuple[int, int], nodes: list[dict], *, max_dist: int
) -> dict | None:
    px, py = point
    best = None
    best_dist = 1e9
    for node in nodes:
        x, y, w, h = node["bbox"]
        dx = max(x - px, 0, px - (x + w))
        dy = max(y - py, 0, py - (y + h))
        d = (dx * dx + dy * dy) ** 0.5
        if d < best_dist:
            best_dist = d
            best = node
    if best_dist > max_dist:
        return None
    return best


def _ocr_text_nodes(img: np.ndarray, ocr_lang: str) -> list[dict]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(bin_img) < 127:
        bin_img = 255 - bin_img
    data = pytesseract.image_to_data(
        bin_img, lang=ocr_lang, config="--psm 6", output_type=pytesseract.Output.DICT
    )
    boxes = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        try:
            conf = float(data["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        if not text or conf < 40:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if w < 8 or h < 8:
            continue
        boxes.append((x, y, x + w, y + h))

    merged = _merge_boxes(boxes, x_gap=40, y_gap=20)
    nodes = []
    for x1, y1, x2, y2 in merged:
        w, h = x2 - x1, y2 - y1
        if w < 20 or h < 15:
            continue
        nodes.append({"bbox": [int(x1), int(y1), int(w), int(h)], "page": 1})
    return nodes


def _merge_nodes(existing: list[dict], extra: list[dict]) -> list[dict]:
    out = list(existing)
    for node in extra:
        if not _overlaps_any(node["bbox"], existing, iou_thresh=0.3):
            out.append(node)
    return out


def _overlaps_any(bbox: list[int], nodes: list[dict], *, iou_thresh: float) -> bool:
    for node in nodes:
        if _rect_iou(bbox, node["bbox"]) >= iou_thresh:
            return True
    return False


def _rect_iou(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]], *, x_gap: int, y_gap: int
) -> list[tuple[int, int, int, int]]:
    boxes = list(boxes)
    merged = []
    while boxes:
        x1, y1, x2, y2 = boxes.pop(0)
        changed = True
        while changed:
            changed = False
            for i in range(len(boxes) - 1, -1, -1):
                bx1, by1, bx2, by2 = boxes[i]
                if _rects_close((x1, y1, x2, y2), (bx1, by1, bx2, by2), x_gap, y_gap):
                    x1, y1 = min(x1, bx1), min(y1, by1)
                    x2, y2 = max(x2, bx2), max(y2, by2)
                    boxes.pop(i)
                    changed = True
        merged.append((x1, y1, x2, y2))
    return merged


def _rects_close(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    x_gap: int,
    y_gap: int,
) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return (
        ax1 <= bx2 + x_gap
        and bx1 <= ax2 + x_gap
        and ay1 <= by2 + y_gap
        and by1 <= ay2 + y_gap
    )


def _to_mermaid_markdown(graph: dict) -> str:
    lines = ["```mermaid", "flowchart TD"]
    node_text = {}
    for node in graph.get("nodes", []):
        nid = node.get("id")
        text = node.get("text") or nid
        text = text.replace('"', "'")
        node_text[nid] = text
        lines.append(f'  {nid}["{text}"]')
    for edge in graph.get("edges", []):
        src = edge.get("from")
        dst = edge.get("to")
        if src and dst:
            lines.append(f"  {src} --> {dst}")
    lines.append("```")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract flowchart graph from PDF/image")
    parser.add_argument("input", help="Input PDF or image path")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Write Mermaid flowchart markdown instead of JSON",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for PDFs")
    parser.add_argument("--ocr-lang", default="eng", help="Tesseract language")
    args = parser.parse_args()

    result = extract_flowchart_graph(args.input, dpi=args.dpi, ocr_lang=args.ocr_lang)
    out_path = Path(args.output) if args.output else Path(args.input).with_suffix(".graph.json")
    if args.markdown or out_path.suffix.lower() == ".md":
        md = _to_mermaid_markdown(result)
        out_path.write_text(md, encoding="utf-8")
        print(f"Wrote markdown to: {out_path}")
    else:
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote graph to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
