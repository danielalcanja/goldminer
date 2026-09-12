from __future__ import annotations

import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Mapping, Protocol

from goldminer.errors import GoldMinerError


class ObjectStore(Protocol):
    def download_file(self, key: str, destination: Path) -> None: ...

    def upload_file(self, source: Path, key: str) -> None: ...

    def put_json(self, key: str, value: Mapping[str, Any]) -> None: ...


class R2ObjectStore:
    def __init__(self, client: Any, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    @classmethod
    def from_environment(cls) -> "R2ObjectStore":
        required = {
            "R2_ENDPOINT_URL": os.environ.get("R2_ENDPOINT_URL", "").strip(),
            "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", "").strip(),
            "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", "").strip(),
            "R2_BUCKET": os.environ.get("R2_BUCKET", "").strip(),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise GoldMinerError(
                "Missing R2 worker configuration: " + ", ".join(sorted(missing))
            )
        try:
            import boto3
            from botocore.config import Config
        except ImportError as exc:
            raise GoldMinerError(
                "The hosted worker requires the 'worker' package extra: pip install '.[worker]'"
            ) from exc

        client = boto3.client(
            "s3",
            endpoint_url=required["R2_ENDPOINT_URL"],
            aws_access_key_id=required["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=required["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
            config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "adaptive"}),
        )
        return cls(client, required["R2_BUCKET"])

    def download_file(self, key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.download")
        try:
            self._client.download_file(self._bucket, key, str(temporary))
            os.replace(temporary, destination)
        except Exception as exc:
            raise GoldMinerError(f"Could not download R2 object {key!r}: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)

    def upload_file(self, source: Path, key: str) -> None:
        content_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        try:
            self._client.upload_file(
                str(source), self._bucket, key, ExtraArgs={"ContentType": content_type}
            )
        except Exception as exc:
            raise GoldMinerError(f"Could not upload artifact {key!r}: {exc}") from exc

    def put_json(self, key: str, value: Mapping[str, Any]) -> None:
        body = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
            )
        except Exception as exc:
            raise GoldMinerError(f"Could not write job status {key!r}: {exc}") from exc
