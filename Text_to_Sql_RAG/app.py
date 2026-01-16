from pathlib import Path
import hashlib
from typing import Dict, Optional, List, Any
from urllib.parse import urlparse
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from psycopg import OperationalError

from db_connection import get_connection
from index_schema_qdrant import index_schema_to_qdrant
from llm_agent_rag import answer_nl
from table_metadata import generate_table_metadata


app = FastAPI(title="Text to SQL RAG")
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"

USERS: Dict[str, Dict[str, Any]] = {}
AUTH_SESSIONS: Dict[str, Dict[str, Optional[str]]] = {}


class QueryRequest(BaseModel):
    question: str
    model: Optional[str] = None
    top_k: Optional[int] = None
    db_url: Optional[str] = None


class ProcessDbRequest(BaseModel):
    db_url: Optional[str] = None
    overwrite: Optional[bool] = False


class VerifyDbRequest(BaseModel):
    db_url: str


def load_template(name: str) -> str:
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8")


def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def get_auth_context(request: Request) -> Optional[Dict[str, Optional[str]]]:
    token = request.cookies.get("auth_token")
    if not token:
        return None
    return AUTH_SESSIONS.get(token)


def get_user_from_cookie(request: Request) -> Optional[str]:
    ctx = get_auth_context(request)
    if not ctx:
        return None
    return ctx.get("email")


def require_auth(request: Request) -> str:
    email = get_user_from_cookie(request)
    if not email:
        raise HTTPException(status_code=401, detail="Please log in to continue.")
    return email


def get_db_url_from_cookie(request: Request) -> Optional[str]:
    ctx = get_auth_context(request)
    if not ctx:
        return None
    return ctx.get("db_url")


def get_schema_from_cookie(request: Request) -> Optional[str]:
    ctx = get_auth_context(request)
    if not ctx:
        return None
    return ctx.get("schema")


def list_schemas(db_url: str) -> List[str]:
    with get_connection(db_url=db_url, schema="public") as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('pg_catalog', 'information_schema')
                ORDER BY schema_name;
                """
            )
            return [r[0] for r in cur.fetchall()]


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    if not get_user_from_cookie(request):
        return RedirectResponse(url="/login")
    return HTMLResponse(load_template("index.html"))


@app.get("/chat", response_class=HTMLResponse)
def chat(request: Request) -> HTMLResponse:
    if not get_user_from_cookie(request):
        return RedirectResponse(url="/login")
    if not get_db_url_from_cookie(request):
        return RedirectResponse(url="/")
    return HTMLResponse(load_template("chat.html"))


@app.get("/login", response_class=HTMLResponse)
def login() -> HTMLResponse:
    return HTMLResponse(load_template("login.html"))


@app.post("/api/signup")
def api_signup(payload: Dict[str, str]) -> JSONResponse:
    name = (payload.get("name") or "").strip()
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    if not name:
        raise HTTPException(status_code=400, detail="Name is required.")
    if "@" not in email or "." not in email:
        raise HTTPException(status_code=400, detail="Please enter a valid email.")
    if len(password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters.")
    if email in USERS:
        raise HTTPException(status_code=400, detail="An account with this email already exists.")

    USERS[email] = {"name": name, "password": hash_password(password), "db_urls": []}
    token = uuid4().hex
    AUTH_SESSIONS[token] = {"email": email, "db_url": None, "schema": None}
    response = JSONResponse({"status": "ok", "message": "Account created."})
    response.set_cookie("auth_token", token, httponly=True, samesite="lax")
    return response


@app.post("/api/login")
def api_login(payload: Dict[str, str]) -> JSONResponse:
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    user = USERS.get(email)
    if not user or user["password"] != hash_password(password):
        raise HTTPException(status_code=400, detail="Invalid email or password.")

    token = uuid4().hex
    AUTH_SESSIONS[token] = {"email": email, "db_url": None, "schema": None}
    response = JSONResponse({"status": "ok", "message": "Logged in."})
    response.set_cookie("auth_token", token, httponly=True, samesite="lax")
    return response


@app.get("/api/db-urls")
def api_db_urls(request: Request) -> JSONResponse:
    email = require_auth(request)
    user = USERS.get(email, {})
    return JSONResponse({"status": "ok", "db_urls": user.get("db_urls", [])})


@app.post("/api/select-schema")
def api_select_schema(payload: Dict[str, str], request: Request) -> JSONResponse:
    require_auth(request)
    schema = (payload.get("schema") or "").strip()
    if not schema:
        raise HTTPException(status_code=400, detail="Schema is required.")

    ctx = get_auth_context(request)
    if not ctx or not ctx.get("db_url"):
        raise HTTPException(status_code=400, detail="Database URL is required.")

    available = list_schemas(ctx["db_url"])
    if schema not in available:
        raise HTTPException(status_code=400, detail="Schema not found for this database.")

    ctx["schema"] = schema
    return JSONResponse({"status": "ok", "message": "Schema selected."})


@app.post("/api/query")
def api_query(payload: QueryRequest, request: Request) -> JSONResponse:
    require_auth(request)
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is required.")

    session_db_url = get_db_url_from_cookie(request)
    if not session_db_url:
        raise HTTPException(status_code=400, detail="Database URL is required.")
    if payload.db_url and payload.db_url.strip() and payload.db_url.strip() != session_db_url:
        raise HTTPException(status_code=400, detail="Please verify the selected database URL first.")
    db_url = session_db_url

    schema = get_schema_from_cookie(request)
    if not schema:
        raise HTTPException(status_code=400, detail="Schema selection is required.")

    try:
        out = answer_nl(
            question,
            model=payload.model or "openai/gpt-oss-20b",
            top_k=payload.top_k or 8,
            db_url=db_url,
            schema=schema,
        )
        return JSONResponse(jsonable_encoder(out))
    except OperationalError:
        raise HTTPException(status_code=400, detail="Please enter a valid database URL.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/process-db")
def api_process_db(payload: ProcessDbRequest, request: Request) -> JSONResponse:
    require_auth(request)
    session_db_url = get_db_url_from_cookie(request)
    if not session_db_url:
        raise HTTPException(status_code=400, detail="Database URL is required.")
    if payload.db_url and payload.db_url.strip() and payload.db_url.strip() != session_db_url:
        raise HTTPException(status_code=400, detail="Please verify the selected database URL first.")
    db_url = session_db_url

    schema = get_schema_from_cookie(request)
    if not schema:
        raise HTTPException(status_code=400, detail="Schema selection is required.")

    parsed = urlparse(db_url)
    if parsed.scheme not in {"postgresql", "postgres", "postgresql+psycopg"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="Please enter a valid database URL.")

    try:
        generate_table_metadata(db_url=db_url, overwrite=bool(payload.overwrite), schemas=[schema])
        index_schema_to_qdrant(db_url=db_url, schemas=[schema])
        return JSONResponse({"status": "ok", "message": "Schema metadata embedded and stored in Qdrant."})
    except OperationalError:
        raise HTTPException(status_code=400, detail="Please enter a valid database URL.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/verify-db")
def api_verify_db(payload: VerifyDbRequest, request: Request) -> JSONResponse:
    require_auth(request)
    db_url = payload.db_url.strip()
    if not db_url:
        raise HTTPException(status_code=400, detail="Database URL is required.")

    parsed = urlparse(db_url)
    if parsed.scheme not in {"postgresql", "postgres", "postgresql+psycopg"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="Please enter a valid database URL.")

    try:
        with get_connection(db_url=db_url, schema="public") as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                cur.fetchone()
    except OperationalError:
        raise HTTPException(status_code=400, detail="Please enter a valid database URL.")

    ctx = get_auth_context(request)
    if ctx is not None:
        ctx["db_url"] = db_url
        ctx["schema"] = None

    email = get_user_from_cookie(request)
    if email and email in USERS:
        saved = USERS[email].setdefault("db_urls", [])
        if db_url not in saved:
            saved.append(db_url)

    schemas = list_schemas(db_url)
    return JSONResponse({"status": "ok", "message": "Database verified.", "schemas": schemas})
