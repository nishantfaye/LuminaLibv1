"""S3-compatible storage service implementation.

Uses ``boto3`` / ``aiobotocore`` to talk to any S3-compatible object store
— AWS S3, **MinIO**, or LocalStack.

For local development the ``docker-compose.yml`` ships a MinIO container
that provides an S3-compatible API at ``http://minio:9000``.
"""

import hashlib
import logging
from typing import Any, Optional

from app.domain.repositories import IStorageService

logger = logging.getLogger(__name__)


class S3StorageService(IStorageService):
    """S3-compatible object-store implementation of :class:`IStorageService`.

    Works with **AWS S3**, **MinIO**, or any endpoint that speaks the S3
    protocol.  Pass *endpoint_url* to point at a non-AWS service (e.g.
    ``http://minio:9000`` inside Docker Compose).

    Parameters
    ----------
    bucket_name : str
        Target bucket (created automatically if it does not exist).
    region : str
        AWS region (default ``us-east-1``).
    endpoint_url : str | None
        Custom S3 endpoint for MinIO / LocalStack.  ``None`` = real AWS.
    aws_access_key_id / aws_secret_access_key : str | None
        Explicit credentials.  If *None* the default credential chain is used.
    """

    def __init__(
        self,
        bucket_name: str = "luminalib-books",
        region: str = "us-east-1",
        endpoint_url: Optional[str] = None,
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ):
        self.bucket_name = bucket_name
        self.region = region
        self.endpoint_url = endpoint_url
        self._client = self._build_client(aws_access_key_id, aws_secret_access_key)
        self._ensure_bucket()

    # ------------------------------------------------------------------
    # Client initialisation
    # ------------------------------------------------------------------
    def _build_client(
        self,
        access_key: Optional[str],
        secret_key: Optional[str],
    ) -> Any:
        """Build a boto3 S3 client; fall back to *None* when boto3 is missing."""
        try:
            import boto3

            kwargs: dict[str, Any] = {"region_name": self.region}
            if self.endpoint_url:
                kwargs["endpoint_url"] = self.endpoint_url
            if access_key and secret_key:
                kwargs["aws_access_key_id"] = access_key
                kwargs["aws_secret_access_key"] = secret_key
            client = boto3.client("s3", **kwargs)
            logger.info(
                "S3 client initialised (endpoint=%s, bucket=%s)",
                self.endpoint_url or "AWS",
                self.bucket_name,
            )
            return client
        except ImportError:
            logger.warning(
                "boto3 not installed — S3StorageService will use in-memory fallback. "
                "Run `pip install boto3` for real S3/MinIO support."
            )
            return None

    def _ensure_bucket(self) -> None:
        """Create the target bucket if it does not already exist.

        MinIO and LocalStack require buckets to be created explicitly.
        """
        if self._client is None:
            return
        try:
            self._client.head_bucket(Bucket=self.bucket_name)
            logger.debug("Bucket '%s' already exists", self.bucket_name)
        except Exception:
            try:
                self._client.create_bucket(Bucket=self.bucket_name)
                logger.info("Created bucket '%s'", self.bucket_name)
            except Exception as exc:
                logger.warning("Could not create bucket '%s': %s", self.bucket_name, exc)

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------
    async def save_file(self, file_content: bytes, filename: str) -> str:
        """Upload *file_content* to S3/MinIO and return the object key."""
        content_hash = hashlib.sha256(file_content).hexdigest()
        key = f"{content_hash[:2]}/{content_hash}_{filename}"

        if self._client is not None:
            self._client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=file_content,
            )
            logger.info("S3: uploaded %s (%d bytes)", key, len(file_content))
        else:
            if not hasattr(self, "_mem"):
                self._mem: dict[str, bytes] = {}
            self._mem[key] = file_content
            logger.info("S3 (in-memory fallback): saved %s (%d bytes)", key, len(file_content))

        return key

    async def get_file(self, file_path: str) -> bytes:
        """Download a file from S3/MinIO by its object key."""
        if self._client is not None:
            response = self._client.get_object(Bucket=self.bucket_name, Key=file_path)
            body: bytes = response["Body"].read()
            logger.debug("S3: retrieved %s (%d bytes)", file_path, len(body))
            return body

        mem = getattr(self, "_mem", {})
        if file_path in mem:
            return mem[file_path]
        raise FileNotFoundError(f"S3 (in-memory): {file_path} not found")

    async def delete_file(self, file_path: str) -> bool:
        """Delete an object from S3/MinIO."""
        if self._client is not None:
            self._client.delete_object(Bucket=self.bucket_name, Key=file_path)
            logger.info("S3: deleted %s", file_path)
            return True

        mem = getattr(self, "_mem", {})
        if file_path in mem:
            del mem[file_path]
            logger.info("S3 (in-memory fallback): deleted %s", file_path)
            return True
        return False
