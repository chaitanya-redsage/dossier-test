import os
import json
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

from db_connection import load_env, get_connection
from retrieve_schema_qdrant import SchemaRetriever


def get_groq_client() -> OpenAI:
    load_env()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in environment")

    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


READONLY_START_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
FORBIDDEN_RE = re.compile(r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy)\b", re.IGNORECASE)


def is_readonly_sql(sql: str) -> bool:
    sql = sql.strip()
    if not READONLY_START_RE.search(sql):
        return False
    if FORBIDDEN_RE.search(sql):
        return False
    parts = [p.strip() for p in sql.split(";") if p.strip()]
    return len(parts) == 1


def enforce_limit(sql: str, default_limit: int = 100) -> str:
    if re.search(r"\blimit\b\s+\d+\b", sql, re.IGNORECASE):
        return sql if sql.strip().endswith(";") else sql.strip() + ";"
    return sql.rstrip().rstrip(";") + f" LIMIT {default_limit};"


def run_sql(sql: str, db_url: Optional[str] = None, schema: Optional[str] = None) -> Dict[str, Any]:
    with get_connection(db_url=db_url, schema=schema) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description] if cur.description else []
    return {"columns": cols, "rows": rows}


PLAN_SYSTEM = """You are a database query planner.

You will be given:
- Retrieved schema context (subset of relevant tables with columns + foreign keys + metadata)
- A user question

Return ONLY valid JSON with keys:
intent: string
required: {tables: ["schema.table", ...], columns: {"schema.table": ["col1", ...]}}
filters: [ {column: "schema.table.column", op: "=", value: "..."} ... ]  (can be empty)
aggregations: [ {type: "count|sum|avg|min|max|group_by", column: "schema.table.column", by: ["schema.table.column", ...]} ... ] (can be empty)
notes: string (short)

Rules:
- Do NOT include SQL in this step.
- Only use tables/columns present in the retrieved schema context.
- Use schema-qualified names.
"""


SQL_SYSTEM = """You are an expert PostgreSQL SQL generator.

Rules (STRICT):
- Output ONLY valid JSON: {"sql": "..."}
- EXACTLY ONE read-only statement (SELECT or WITH ... SELECT)
- Use ONLY tables/columns present in the provided schema context
- Always schema-qualify table names (hc.table or healthcare.table)
- Prefer explicit column lists (avoid SELECT *) unless user asks for all columns
- Always include LIMIT for row-returning queries
"""


def llm_plan(client: OpenAI, context: Dict[str, Any], question: str, model: str) -> Dict[str, Any]:
    inp = {"schema_context": context, "question": question}
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": json.dumps(inp, ensure_ascii=False)},
        ],
    )
    return json.loads(resp.output_text.strip())


def llm_sql(client: OpenAI, context: Dict[str, Any], question: str, plan: Dict[str, Any], model: str) -> str:
    inp = {"schema_context": context, "question": question, "plan": plan}
    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SQL_SYSTEM},
            {"role": "user", "content": json.dumps(inp, ensure_ascii=False)},
        ],
    )
    obj = json.loads(resp.output_text.strip())
    return obj["sql"]


def build_context_from_hits(hits: List[Dict[str, Any]]) -> Dict[str, Any]:
    tables = []
    for h in hits:
        p = h.get("payload", {}) or {}
        tables.append({
            "schema": p.get("schema"),
            "table": p.get("table"),
            "description": p.get("description", ""),
            "tags": p.get("tags", []),
            "columns": p.get("columns", []),
            "foreign_keys": p.get("foreign_keys", []),
        })
    return {"tables": tables}


def answer_nl(
    question: str,
    model: str = "openai/gpt-oss-20b",
    top_k: int = 8,
    db_url: Optional[str] = None,
    schema: Optional[str] = None,
) -> Dict[str, Any]:
    load_env()
    default_limit = int(os.getenv("DEFAULT_LIMIT", "100"))

    retriever = SchemaRetriever()
    schemas = [schema] if schema else None
    hits = retriever.search(question, k=top_k, schemas=schemas)
    context = build_context_from_hits(hits)

    client = get_groq_client()

    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            plan = llm_plan(client, context, question, model=model)
            sql = llm_sql(client, context, question, plan, model=model)

            if not is_readonly_sql(sql):
                raise ValueError(f"Generated SQL is not read-only. Refusing.\nSQL was:\n{sql}")

            sql = enforce_limit(sql, default_limit=default_limit)
            result = run_sql(sql, db_url=db_url, schema=schema)

            return {"retrieved": hits, "context": context, "plan": plan, "sql": sql, "result": result}
        except Exception as exc:
            last_error = exc
            if attempt == 2:
                raise

    raise last_error
