"""Private local or S3-compatible object storage selected by STORAGE_URL."""

from pathlib import Path
from urllib.parse import unquote, urlsplit

import boto3

from api.config import settings


class Storage:
    def __init__(self):
        parsed = urlsplit(settings.storage_url)
        self.scheme = parsed.scheme
        if self.scheme == "file":
            self.root = settings.local_storage_root()
            self.root.mkdir(parents=True, exist_ok=True)
            self.bucket = ""
            self.prefix = ""
            self.client = None
        elif self.scheme == "s3":
            if not parsed.netloc:
                raise ValueError("S3 STORAGE_URL must include a bucket name.")
            self.bucket = parsed.netloc
            self.prefix = parsed.path.strip("/")
            self.root = None
            self.client = boto3.client(
                "s3",
                endpoint_url=settings.s3_endpoint_url or None,
            )
        else:
            raise ValueError("STORAGE_URL must use file: or s3:.")

    def _object_key(self, key: str) -> str:
        normalized = key.replace("\\", "/").lstrip("/")
        if any(part in {"", ".", ".."} for part in normalized.split("/")):
            raise ValueError("Invalid storage key.")
        return "/".join(part for part in (self.prefix, normalized) if part)

    def put_bytes(self, key: str, content: bytes, content_type: str) -> None:
        object_key = self._object_key(key)
        if self.scheme == "s3":
            self.client.put_object(
                Bucket=self.bucket,
                Key=object_key,
                Body=content,
                ContentType=content_type,
                ServerSideEncryption="AES256",
            )
            return

        destination = (self.root / object_key).resolve()
        if not destination.is_relative_to(self.root):
            raise ValueError("Storage key escaped the configured storage directory.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    def get_bytes(self, key: str) -> bytes:
        object_key = self._object_key(key)
        if self.scheme == "s3":
            response = self.client.get_object(Bucket=self.bucket, Key=object_key)
            return response["Body"].read()

        source = (self.root / object_key).resolve()
        if not source.is_relative_to(self.root):
            raise ValueError("Storage key escaped the configured storage directory.")
        return source.read_bytes()

    def delete(self, key: str) -> None:
        object_key = self._object_key(key)
        if self.scheme == "s3":
            self.client.delete_object(Bucket=self.bucket, Key=object_key)
            return

        source = (self.root / object_key).resolve()
        if not source.is_relative_to(self.root):
            raise ValueError("Storage key escaped the configured storage directory.")
        source.unlink(missing_ok=True)


storage = Storage()
