"""
R2 upload helpers (not wired into the pipeline yet).

Requirements (install later when you decide to integrate):
  pip install boto3

Env vars:
  R2_ACCOUNT_ID
  R2_ACCESS_KEY_ID
  R2_SECRET_ACCESS_KEY
  R2_BUCKET
  R2_ENDPOINT  (e.g. https://<ACCOUNT_ID>.r2.cloudflarestorage.com)
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Iterable, Optional


def _get_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing env var: {name}")
    return val


def _client():
    import boto3

    endpoint = _get_env("R2_ENDPOINT")
    key_id = _get_env("R2_ACCESS_KEY_ID")
    secret = _get_env("R2_SECRET_ACCESS_KEY")

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key_id,
        aws_secret_access_key=secret,
    )


def upload_file_to_r2(
    local_path: str | Path,
    *,
    key: Optional[str] = None,
    content_type: Optional[str] = None,
) -> str:
    """
    Upload a single file to R2.
    Returns the object key used.
    """
    p = Path(local_path)
    if not p.exists():
        raise FileNotFoundError(p)

    bucket = _get_env("R2_BUCKET")
    key = key or p.name
    content_type = content_type or mimetypes.guess_type(p.name)[0]

    extra = {}
    if content_type:
        extra["ContentType"] = content_type

    client = _client()
    with p.open("rb") as f:
        client.upload_fileobj(f, bucket, key, ExtraArgs=extra or None)

    return key


def upload_files_to_r2(
    paths: Iterable[str | Path],
    *,
    key_prefix: str = "",
) -> list[str]:
    """
    Upload multiple files to R2.
    Returns list of object keys.
    """
    keys: list[str] = []
    for path in paths:
        p = Path(path)
        key = f"{key_prefix}{p.name}" if key_prefix else p.name
        keys.append(upload_file_to_r2(p, key=key))
    return keys


# Example usage after .ocr.md write (do not integrate yet):
#
# from r2_upload import upload_file_to_r2
# upload_file_to_r2(p.with_suffix(".ocr.md"))
