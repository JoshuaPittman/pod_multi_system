"""
r2_uploader.py — Cloudflare R2 image upload utility

Required env vars:
  R2_ACCESS_KEY_ID     — R2 API token Access Key ID
  R2_SECRET_ACCESS_KEY — R2 API token Secret Access Key
  R2_ENDPOINT          — e.g. https://<account_id>.r2.cloudflarestorage.com
  R2_BUCKET            — your bucket name

The public URL is derived from R2_ENDPOINT + bucket + key.
Make sure public access is enabled on the bucket in Cloudflare.
"""

import os
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def is_r2_configured(config: dict = None) -> bool:
    """Return True if all required R2 credentials are present."""
    cfg = config or {}
    return all([
        cfg.get("r2_access_key_id") or os.getenv("R2_ACCESS_KEY_ID"),
        cfg.get("r2_secret_key")    or os.getenv("R2_SECRET_ACCESS_KEY"),
        cfg.get("r2_endpoint")      or os.getenv("R2_ENDPOINT"),
        cfg.get("r2_bucket")        or os.getenv("R2_BUCKET"),
    ])


async def upload_to_r2(local_path: str, object_key: str, config: dict = None) -> str:
    """
    Upload a local file to Cloudflare R2.

    Returns:
        Public HTTPS URL to the uploaded file.
    """
    import boto3
    from botocore.config import Config

    cfg = config or {}

    access_key = cfg.get("r2_access_key_id") or os.getenv("R2_ACCESS_KEY_ID")
    secret_key = cfg.get("r2_secret_key")    or os.getenv("R2_SECRET_ACCESS_KEY")
    endpoint   = cfg.get("r2_endpoint")      or os.getenv("R2_ENDPOINT")
    bucket     = cfg.get("r2_bucket")        or os.getenv("R2_BUCKET")

    def _upload():
        s3 = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        with open(local_path, "rb") as f:
            s3.put_object(
                Bucket=bucket,
                Key=object_key,
                Body=f,
                ContentType="image/png",
            )

    # Run blocking boto3 call off the event loop
    await asyncio.get_event_loop().run_in_executor(None, _upload)

    public_url = f"{endpoint.rstrip('/')}/{bucket}/{object_key}"
    logger.info(f"[R2] Uploaded {Path(local_path).name} → {public_url}")
    return public_url
