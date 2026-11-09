"""Save uploaded originals locally or to an S3-compatible object store."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ObjectStore:
    def __init__(
        self,
        backend: str,
        data_dir: Path,
        endpoint: str,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str,
    ):
        self.backend = backend
        self.data_dir = data_dir
        self.endpoint = endpoint
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.region = region
        self._client: Any = None

    def save(self, key: str, content: bytes, content_type: str) -> str:
        """Store content under an application-generated key and return that key."""
        if self.backend == "local":
            path = self.data_dir / "files" / key
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            return str(path.relative_to(self.data_dir))
        if self.backend != "s3":
            raise ValueError("OBJECT_STORAGE_BACKEND must be 'local' or 's3'")
        client = self._s3_client()
        try:
            client.head_bucket(Bucket=self.bucket)
        except client.exceptions.ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code")
            if error_code not in {"404", "NoSuchBucket", "NotFound"}:
                raise
            client.create_bucket(Bucket=self.bucket)
        client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=content,
            ContentType=content_type,
        )
        return f"s3://{self.bucket}/{key}"

    def load(self, storage_key: str) -> bytes:
        """Read a previously stored original, for background jobs or downloads."""
        if storage_key.startswith("s3://"):
            bucket, _, key = storage_key[5:].partition("/")
            if bucket != self.bucket or not key:
                raise ValueError("Stored object reference is outside the configured bucket")
            response = self._s3_client().get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
            if not isinstance(content, bytes):
                raise TypeError("Object storage returned non-byte content")
            return content
        path = (self.data_dir / storage_key).resolve()
        if not path.is_relative_to(self.data_dir.resolve()):
            raise ValueError("Stored file reference is outside the data directory")
        return path.read_bytes()

    def delete(self, storage_key: str) -> None:
        if storage_key.startswith("s3://"):
            bucket, _, key = storage_key[5:].partition("/")
            if bucket != self.bucket or not key:
                raise ValueError("Stored object reference is outside the configured bucket")
            self._s3_client().delete_object(Bucket=bucket, Key=key)
            return
        path = (self.data_dir / storage_key).resolve()
        if not path.is_relative_to(self.data_dir.resolve()):
            raise ValueError("Stored file reference is outside the data directory")
        path.unlink(missing_ok=True)

    def _s3_client(self) -> Any:
        if self._client is None:
            try:
                import boto3  # type: ignore[import-untyped]
                from botocore.config import Config  # type: ignore[import-untyped]
            except ImportError as exc:
                raise RuntimeError("Install project dependencies to use S3 object storage") from exc
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
                config=Config(
                    connect_timeout=5,
                    read_timeout=10,
                    retries={"max_attempts": 2, "mode": "standard"},
                    s3={"addressing_style": "path"},
                ),
            )
        return self._client
