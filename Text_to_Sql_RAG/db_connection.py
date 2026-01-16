import os
from typing import Optional

import psycopg

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def load_env(path: str = ".env") -> None:
    if load_dotenv:
        load_dotenv(path)
        return

    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_connection(db_url: Optional[str] = None, schema: Optional[str] = None):
    load_env()

    schema = schema or os.getenv("DB_SCHEMA", "public")

    db_url = db_url or os.getenv("DATABASE_URL")
    if db_url:
        return psycopg.connect(
            db_url,
            options=f"-c search_path={schema}",
        )
