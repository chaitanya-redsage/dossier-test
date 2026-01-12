import os
import json
from typing import Dict, Any, Optional

from openai import OpenAI

from db_connection import load_env
from schema_tools import get_schema_payload


METADATA_SYSTEM = """You are a database documentation assistant.

Given a table schema (name, columns, foreign keys), generate:
- description: 1-2 sentences describing what the table likely represents
- tags: 3-8 short tags (snake_case) describing domain/function

Return ONLY valid JSON:
{"description": "...", "tags": ["...", "..."]}

Rules:
- Be conservative; do not hallucinate business meaning if unclear.
- Use generic descriptions if needed.
- Keep description short.
"""


def get_groq_client() -> OpenAI:
    load_env()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in environment")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def load_metadata(path: str = "table_metadata.json") -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(data: Dict[str, Any], path: str = "table_metadata.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def generate_table_metadata(
    schema_payload: Optional[Dict[str, Any]] = None,
    out_path: str = "table_metadata.json",
    overwrite: bool = False,
) -> Dict[str, Any]:
    load_env()
    model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

    if schema_payload is None:
        schema_payload = get_schema_payload()

    existing = load_metadata(out_path)
    client = get_groq_client()

    schemas = schema_payload.get("schemas", {})
    for schema_name, schema_obj in schemas.items():
        for table_name, table_obj in (schema_obj.get("tables", {}) or {}).items():
            key = f"{schema_name}.{table_name}"
            if (key in existing) and not overwrite:
                continue

            inp = {
                "schema": schema_name,
                "table": table_name,
                "columns": table_obj.get("columns", []),
                "foreign_keys": table_obj.get("foreign_keys", []),
            }

            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": METADATA_SYSTEM},
                    {"role": "user", "content": json.dumps(inp, ensure_ascii=False)},
                ],
            )

            try:
                obj = json.loads(resp.output_text.strip())
                desc = str(obj.get("description", "")).strip()
                tags = obj.get("tags", [])
                if not isinstance(tags, list):
                    tags = []
                tags = [str(t).strip() for t in tags if str(t).strip()]

                existing[key] = {"description": desc, "tags": tags}
                print(f"✅ metadata: {key}")
            except Exception as e:
                print(f"⚠️ failed metadata for {key}: {e}")

    save_metadata(existing, out_path)
    return existing


if __name__ == "__main__":
    generate_table_metadata()
    print("Done. Wrote table_metadata.json")
