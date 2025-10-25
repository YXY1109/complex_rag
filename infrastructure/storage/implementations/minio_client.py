"""
MinIO Object Storage Client Implementation

This module implements the MinIO client for object storage operations.
Based on the object storage interface abstract class.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, BinaryIO, AsyncGenerator
from datetime import datetime, timedelta
import io
import os
import mimetypes
import hashlib
import uuid

from minio import Minio
from minio.error import S3Error
from pydantic import BaseModel, Field

from ...interfaces.storage_interface import (
    StorageInterface,
    StorageConfig,
    StorageCapabilities,
    StorageType,
    AccessControl,
    StorageClass,
    StorageObject,
    UploadResult,
    DownloadResult,
    StorageException,
    ConnectionException,
    BucketException,
    ObjectException,
    UploadException,
    DownloadException,
    ValidationException
)


class MinIOConfig(StorageConfig):
    """MinIO-specific configuration."""

    secure: bool = Field(default=False, description="Use secure connection")
    region: str = Field(default="us-east-1", description="Storage region")
    http_client: Optional[Any] = Field(default=None, description="Custom HTTP client")


class MinIOCapabilities(StorageCapabilities):
    """MinIO-specific capabilities."""

    def __init__(self):
        super().__init__(
            provider="minio",
            supported_storage_classes=[
                StorageClass.STANDARD,
                StorageClass.REDUCED_REDUNDANCY,
                StorageClass.STANDARD_IA,
                StorageClass.ONEZONE_IA,
                StorageClass.GLACIER,
                StorageClass.DEEP_ARCHIVE
            ],
            supported_access_controls=[
                AccessControl.PRIVATE,
                AccessControl.PUBLIC_READ,
                AccessControl.PUBLIC_READ_WRITE,
                AccessControl.AUTHENTICATED_READ
            ],
            max_object_size=5 * 1024 * 1024 * 1024 * 1024,  # 5TB
            max_bucket_size=None,  # No practical limit
            max_keys_per_bucket=None,  # No practical limit
            supports_multipart_upload=True,
            supports_presigned_urls=True,
            supports_encryption=True,
            supports_versioning=False,
            supports_lifecycle_management=True,
            supports_replication=True,
            supports_cross_region_replication=False,
            supports_logging=True,
            supports_notifications=True,
            supports_analytics=False,
            supports_search=False,
            supports_async_operations=False
        )


class MinIOClient(StorageInterface):
    """
    MinIO client implementation for object storage operations.

    Provides MinIO-compatible S3 API operations with comprehensive error handling,
    connection management, and advanced features.
    """

    def __init__(self, config: MinIOConfig):
        super().__init__(config)
        self.config: MinIOConfig = config
        self._client: Optional[Minio] = None
        self._connected = False
        self._capabilities = MinIOCapabilities()

    @property
    def capabilities(self) -> StorageCapabilities:
        """Get MinIO capabilities."""
        return self._capabilities

    async def connect(self) -> bool:
        """
        Connect to MinIO server.

        Returns:
            bool: True if connection successful

        Raises:
            ConnectionException: If connection fails
        """
        try:
            # Create MinIO client
            self._client = Minio(
                endpoint=self.config.endpoint,
                access_key=self.config.access_key,
                secret_key=self.config.secret_key,
                secure=self.config.secure,
                region=self.config.region,
                http_client=self.config.http_client
            )

            # Test connection by listing buckets
            self._client.list_buckets()

            self._connected = True
            logger.info(f"Connected to MinIO server: {self.config.endpoint}")
            return True

        except S3Error as e:
            error_msg = f"Failed to connect to MinIO: {str(e)}"
            logger.error(error_msg)
            raise ConnectionException(error_msg, provider="minio") from e
        except Exception as e:
            error_msg = f"Unexpected error connecting to MinIO: {str(e)}"
            logger.error(error_msg)
            raise ConnectionException(error_msg, provider="minio") from e

    async def disconnect(self) -> None:
        """Disconnect from MinIO server."""
        try:
            self._client = None
            self._connected = False
            logger.info("Disconnected from MinIO")
        except Exception as e:
            logger.error(f"Error disconnecting from MinIO: {str(e)}")

    async def create_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """
        Create a new bucket.

        Args:
            bucket_name: Bucket name (optional, uses default if not provided)

        Returns:
            bool: True if creation successful

        Raises:
            BucketException: If bucket creation fails
        """
        try:
            if not self._connected:
                raise ConnectionException("Not connected to MinIO", provider="minio")

            bucket = bucket_name or self.bucket_name

            if await self.bucket_exists(bucket):
                logger.warning(f"Bucket {bucket} already exists")
                return True

            # Create bucket
            self._client.make_bucket(bucket, location=self.config.region)

            logger.info(f"Created MinIO bucket: {bucket}")
            return True

        except S3Error as e:
            error_msg = f"Failed to create bucket {bucket}: {str(e)}"
            logger.error(error_msg)
            raise BucketException(error_msg, provider="minio") from e
        except Exception as e:
            error_msg = f"Unexpected error creating bucket {bucket}: {str(e)}"
            logger.error(error_msg)
            raise BucketException(error_msg, provider="minio") from e

    async def delete_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """
        Delete a bucket.

        Args:
            bucket_name: Bucket name (optional, uses default if not provided)

        Returns:
            bool: True if deletion successful

        Raises:
            BucketException: If bucket deletion fails
        """
        try:
            if not self._connected:
                raise ConnectionException("Not connected to MinIO", provider="minio")

            bucket = bucket_name or self.bucket_name

            if not await self.bucket_exists(bucket):
                logger.warning(f"Bucket {bucket} does not exist")
                return True

            # Check if bucket is empty
            objects = self._client.list_objects(bucket, recursive=True)
            if objects:
                raise BucketException(f"Bucket {bucket} is not empty. Cannot delete non-empty bucket.", provider="minio")

            # Delete bucket
            self._client.remove_bucket(bucket)

            logger.info(f"Deleted MinIO bucket: {bucket}")
            return True

        except S3Error as e:
            error_msg = f"Failed to delete bucket {bucket}: {str(e)}"
            logger.error(error_msg)
            raise BucketException(error_msg, provider="minio") from e
        except Exception as e:
            error_msg = f"Unexpected error deleting bucket {bucket}: {str(e)}"
            logger.error(error_msg)
            raise BucketException(error_msg, provider="minio") from e

    async def list_buckets(self) -> List[str]:
        """
        List all buckets.

        Returns:
            List[str]: Bucket names

        Raises:
            BucketException: If listing fails
        """
        try:
            if not self._connected:
                raise ConnectionException("Not connected to MinIO", provider="minio")

            buckets = self._client.list_buckets()
            return [bucket.name for bucket in buckets]

        except S3Error as e:
            error_msg = f"Failed to list buckets: {str(e)}"
            logger.error(error_msg)
            raise BucketException(error_msg, provider="minio") from e
        except Exception as e:
            error_msg = f"Unexpected error listing buckets: {str(e)}"
            logger.error(error_msg)
            raise BucketException(error_msg, provider="minio") from e

    async def bucket_exists(self, bucket_name: Optional[str] = None) -> bool:
        """
        Check if a bucket exists.

        Args:
            bucket_name: Bucket name (optional, uses default if not provided)

        Returns:
            bool: True if bucket exists

        Raises:
            BucketException: If check fails
        """
        try:
            if not self._connected:
                raise ConnectionException("Not connected to MinIO", provider="minio")

            bucket = bucket_name or self.bucket_name
            return self._client.bucket_exists(bucket)

        except S3Error as e:
            if e.code == "NoSuchBucket":
                return False
            error_msg = f"Failed to check bucket existence for {bucket}: {str(e)}"
            logger.error(error_msg)
            raise BucketException(error_msg, provider="minio") from e
        except Exception as e:
            error_msg = f"Unexpected error checking bucket existence for {bucket}: {str(e)}"
            logger.error(error_msg)
            raise BucketException(error_msg, provider="minio") from e

    async def upload_object(
        self,
        key: str,
        data: Union[bytes, BinaryIO],
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        storage_class: Optional[StorageClass] = None,
        access_control: Optional[AccessControl] = None
    ) -> UploadResult:
        """
        Upload an object.

        Args:
            key: Object key
            data: Object data
            content_type: Content type
            metadata: Object metadata
            storage_class: Storage class
            access_control: Access control

        Returns:
            UploadResult: Upload result

        Raises:
            UploadException: If upload fails
        """
        try:
            if not self._connected:
                raise ConnectionException("Not connected to MinIO", provider="minio")

            # Validate inputs
            if not key:
                raise ValidationException("Object key cannot be empty", provider="minio")

            if not data:
                raise ValidationException("Object data cannot be empty", provider="minio")

            # Determine content type
            if not content_type:
                content_type = mimetypes.guess_type(key)
                if not content_type:
                    content_type = "application/octet-stream"

            # Prepare metadata
            s3_metadata = {}
            if metadata:
                s3_metadata.update(metadata)

            if content_type:
                s3_metadata["Content-Type"] = content_type

            # Convert data to bytes if needed
            if isinstance(data, io.BytesIO):
                data = data.getvalue()
            elif hasattr(data, "read"):
                data = data.read()

            # Prepare object name
            object_name = key.lstrip("/")

            # Upload object
            start_time = asyncio.get_event_loop().time()
            result = self._client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_name,
                data=data,
                length=len(data),
                content_type=content_type,
                metadata=s3_metadata,
                storage_class=storage_class.value if storage_class else None
            )
            end_time = asyncio.get_event_loop().time()

            # Build upload result
            upload_result = UploadResult(
                key=key,
                bucket=self.bucket_name,
                etag=result.etag,
                size=len(data),
                url=f"{self.config.endpoint}/{self.bucket_name}/{object_name}",
                access_url=f"{self.config.endpoint}/{self.bucket_name}/{object_name}",
                metadata=s3_metadata,
                upload_time_ms=(end_time - start_time) * 1000
            )

            logger.debug(f"Uploaded object {key} to {self.bucket_name}")
            return upload_result

        except S3Error as e:
            error_msg = f"Failed to upload object {key}: {str(e)}"
            logger.error(error_msg)
            raise UploadException(error_msg, provider="minio") from e
        except Exception as e:
            error_msg = f"Unexpected error uploading object {key}: {str(e)}"
            logger.error(error_msg)
            raise UploadException(error_msg, provider="minio") from e

    async def upload_object_from_file(
        self,
        key: str,
        file_path: str,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        storage_class: Optional[StorageClass] = None,
        access_control: Optional[AccessControl] = None
    ) -> UploadResult:
        """
        Upload an object from file.

        Args:
            key: Object key
            file_path: File path
            content_type: Content type
            metadata: Object metadata
            storage_class: Storage class
            access_control: Access control

        Returns:
            UploadResult: Upload result

        Raises:
            UploadException: If upload fails
        """
        try:
            # Validate file path
            if not os.path.exists(file_path):
                raise ValidationException(f"File not found: {file_path}", provider="minio")

            # Get file size and content type
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise ValidationException(f"File is empty: {file_path}", provider="minio")

            if not content_type:
                content_type = mimetypes.guess_type(file_path)
                if not content_type:
                    content_type = "application/octet-stream"

            # Add file metadata
            file_metadata = metadata or {}
            file_metadata.update({
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "file_size": str(file_size)
            })

            # Upload file
            with open(file_path, "rb") as f:
                return await self.upload_object(
                    key=key,
                    data=f,
                    content_type=content_type,
                    metadata=file_metadata,
                    storage_class=storage_class,
                    access_control=access_control
                )

        except ValidationException:
            raise
        except Exception as e:
            error_msg = f"Failed to upload file {file_path}: {str(e)}"
            logger.error(error_msg)
            raise UploadException(error_msg, provider="minio") from e

    async def download_object(self, key: str) -> DownloadResult:
        """
        Download an object.

        Args:
            key: Object key

        Returns:
            DownloadResult: Download result

        Raises:
            DownloadException: If download fails
        """
        try:
            if not self._connected:
                raise ConnectionException("Not connected to MinIO", provider="minio")

            # Validate key
            if not key:
                raise ValidationException("Object key cannot be empty", provider="minio")

            # Prepare object name
            object_name = key.lstrip("/")

            # Download object
            start_time = asyncio.get_event_loop().time()
            response = self._client.get_object(self.bucket_name, object_name)
            data = response.read()
            end_time = asyncio.get_event_loop().time()

            # Get object info
            stat = self._client.stat_object(self.bucket_name, object_name)

            # Build download result
            download_result = DownloadResult(
                key=key,
                bucket=self.bucket_name,
                content=data,
                content_type=stat.content_type,
                etag=stat.etag,
                metadata=stat.metadata,
                download_time_ms=(end_time - start_time) * 1000
            )

            logger.debug(f"Downloaded object {key} from {self.bucket_name}")
            return download_result

        except S3Error as e:
            if e.code == "NoSuchKey":
                raise ObjectException(f"Object not found: {key}", provider="minio") from e
            error_msg = f"Failed to download object {key}: {str(e)}"
            logger.error(error_msg)
            raise DownloadException(error_msg, provider="minio") from e
        except Exception as e:
            error_msg = f"Unexpected error downloading object {key}: {str(e)}"
            logger.error(error_msg)
            raise DownloadException(error_msg, provider="minio") from e

    async def download_object_to_file(
        self,
        key: str,
        file_path: str
    ) -> bool:
        """
        Download an object to file.

        Args:
            key: Object key
            file_path: Destination file path

        Returns:
            bool: True if download successful

        Raises:
            DownloadException: If download fails
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            # Download object
            download_result = await self.download_object(key)

            # Write to file
            with open(file_path, "wb") as f:
                f.write(download_result.content)

            logger.debug(f"Downloaded object {key} to {file_path}")
            return True

        except Exception as e:
            error_msg = f"Failed to download object {key} to {file_path}: {str(e)}"
            logger.error(error_msg)
            raise DownloadException(error_msg, provider="minio") from e

    async def delete_object(self, key: str) -> bool:
        """
        Delete an object.

        Args:
            key: Object key

        Returns:
            bool: True if deletion successful

        Raises:
            ObjectException: If deletion fails
        """
        try:
            if not self._connected:
                raise ConnectionException("Not connected to MinIO", provider="minio")

            # Validate key
            if not key:
                raise ValidationException("Object key cannot be empty", provider="minio")

            # Prepare object name
            object_name = key.lstrip("/")

            # Delete object
            self._client.remove_object(self.bucket_name, object_name)

            logger.debug(f"Deleted object {key} from {self.bucket_name}")
            return True

        except S3Error as e:
            if e.code == "NoSuchKey":
                logger.warning(f"Object not found: {key}")
                return False
            error_msg = f"Failed to delete object {key}: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e
        except Exception as e:
            error_msg = f"Unexpected error deleting object {key}: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e

    async def delete_objects(self, keys: List[str]) -> Dict[str, bool]:
        """
        Delete multiple objects.

        Args:
            keys: Object keys

        Returns:
            Dict[str, bool]: Deletion results

        Raises:
            ObjectException: If deletion fails
        """
        try:
            if not self._connected:
                raise ConnectionException("Not connected to MinIO", provider="minio")

            if not keys:
                return {}

            results = {}
            for key in keys:
                try:
                    results[key] = await self.delete_object(key)
                except Exception as e:
                    logger.error(f"Failed to delete object {key}: {str(e)}")
                    results[key] = False

            return results

        except Exception as e:
            error_msg = f"Failed to delete multiple objects: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e

    async def object_exists(self, key: str) -> bool:
        """
        Check if an object exists.

        Args:
            key: Object key

        Returns:
            bool: True if object exists

        Raises:
            ObjectException: If check fails
        """
        try:
            if not self._connected:
                raise ConnectionException("Not connected to MinIO", provider="minio")

            # Validate key
            if not key:
                raise ValidationException("Object key cannot be empty", provider="minio")

            # Prepare object name
            object_name = key.lstrip("/")

            # Check if object exists
            self._client.stat_object(self.bucket_name, object_name)
            return True

        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            error_msg = f"Failed to check object existence for {key}: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e
        except Exception as e:
            error_msg = f"Unexpected error checking object existence for {key}: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e

    async def get_object_info(self, key: str) -> StorageObject:
        """
        Get object information.

        Args:
            key: Object key

        Returns:
            StorageObject: Object information

        Raises:
            ObjectException: If getting info fails
        """
        try:
            if not self._connected:
                raise ConnectionException("Not connected to MinIO", provider="minio")

            # Validate key
            if not key:
                raise ValidationException("Object key cannot be empty", provider="minio")

            # Prepare object name
            object_name = key.lstrip("/")

            # Get object stat
            stat = self._client.stat_object(self.bucket_name, object_name)

            # Parse last modified time
            last_modified = None
            if stat.last_modified:
                last_modified = stat.last_modified.timetuple()

            # Build storage object
            storage_object = StorageObject(
                key=key,
                bucket=self.bucket_name,
                size=stat.size,
                content_type=stat.content_type,
                etag=stat.etag,
                last_modified=last_modified,
                metadata=stat.metadata,
                url=f"{self.config.endpoint}/{self.bucket_name}/{object_name}",
                access_url=f"{self.config.endpoint}/{self.bucket_name}/{object_name}",
                storage_class=None,  # MinIO doesn't expose storage class in stat
                access_control=None  # MinIO doesn't expose ACL in stat
            )

            return storage_object

        except S3Error as e:
            if e.code == "NoSuchKey":
                raise ObjectException(f"Object not found: {key}", provider="minio") from e
            error_msg = f"Failed to get object info for {key}: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e
        except Exception as e:
            error_msg = f"Unexpected error getting object info for {key}: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e

    async def list_objects(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        continuation_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List objects in bucket.

        Args:
            prefix: Object prefix filter
            limit: Maximum number of objects to return
            continuation_token: Continuation token for pagination

        Returns:
            Dict[str, Any]: List result with pagination info

        Raises:
            ObjectException: If listing fails
        """
        try:
            if not self._connected:
                raise ConnectionException("Not connected to MinIO", provider="minio")

            # Prepare object name prefix
            object_prefix = prefix.lstrip("/") if prefix else ""

            # List objects
            objects = self._client.list_objects(
                self.bucket_name,
                prefix=object_prefix,
                recursive=True
            )

            # Filter by limit
            if limit:
                objects = objects[:limit]

            # Convert to storage objects
            storage_objects = []
            for obj in objects:
                storage_object = StorageObject(
                    key=obj.object_name,
                    bucket=self.bucket_name,
                    size=obj.size,
                    content_type=obj.content_type,
                    etag=obj.etag,
                    last_modified=obj.last_modified.timetuple() if obj.last_modified else None,
                    metadata=obj.metadata,
                    url=f"{self.config.endpoint}/{self.bucket_name}/{obj.object_name}",
                    access_url=f"{self.config.endpoint}/{self.bucket_name}/{obj.object_name}",
                    storage_class=None,
                    access_control=None
                )
                storage_objects.append(storage_object)

            return {
                "objects": storage_objects,
                "prefix": prefix,
                "limit": limit,
                "count": len(storage_objects),
                "is_truncated": limit and len(objects) == limit
            }

        except S3Error as e:
            error_msg = f"Failed to list objects: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e
        except Exception as e:
            error_msg = f"Unexpected error listing objects: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e

    async def copy_object(
        self,
        source_key: str,
        destination_key: str,
        source_bucket: Optional[str] = None,
        destination_bucket: Optional[str] = None
    ) -> bool:
        """
        Copy an object.

        Args:
            source_key: Source object key
            destination_key: Destination object key
            source_bucket: Source bucket (optional)
            destination_bucket: Destination bucket (optional)

        Returns:
            bool: True if copy successful

        Raises:
            ObjectException: If copy fails
        """
        try:
            if not self._connected:
                raise ConnectionException("Not connected to MinIO", provider="minio")

            # Validate keys
            if not source_key or not destination_key:
                raise ValidationException("Source and destination keys cannot be empty", provider="minio")

            src_bucket = source_bucket or self.bucket_name
            dst_bucket = destination_bucket or self.bucket_name

            # Prepare object names
            src_object = source_key.lstrip("/")
            dst_object = destination_key.lstrip("/")

            # Copy object
            self._client.copy_object(
                src_bucket, src_object, dst_bucket, dst_object
            )

            logger.debug(f"Copied object {source_key} to {destination_key}")
            return True

        except S3Error as e:
            error_msg = f"Failed to copy object {source_key} to {destination_key}: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e
        except Exception as e:
            error_msg = f"Unexpected error copying object {source_key} to {destination_key}: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e

    async def move_object(
        self,
        source_key: str,
        destination_key: str,
        source_bucket: Optional[str] = None,
        destination_bucket: Optional[str] = None
    ) -> bool:
        """
        Move an object.

        Args:
            source_key: Source object key
            destination_key: Destination object key
            source_bucket: Source bucket (optional)
            destination_bucket: Destination bucket (optional)

        Returns:
            bool: True if move successful

        Raises:
            ObjectException: If move fails
        """
        try:
            # First copy the object
            copy_success = await self.copy_object(
                source_key=source_key,
                destination_key=destination_key,
                source_bucket=source_bucket,
                destination_bucket=destination_bucket
            )

            if copy_success:
                # Then delete the source object
                delete_success = await self.delete_object(
                    source_key=source_key,
                )

                if delete_success:
                    logger.debug(f"Moved object {source_key} to {destination_key}")
                    return True
                else:
                    logger.warning(f"Failed to delete source object {source_key} after copy")
                    return False
            else:
                logger.error(f"Failed to copy object {source_key} to {destination_key}")
                return False

        except Exception as e:
            error_msg = f"Failed to move object {source_key} to {destination_key}: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e

    async def generate_presigned_url(
        self,
        key: str,
        method: str = "GET",
        expiration: int = 3600,
        headers: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Generate a presigned URL for object access.

        Args:
            key: Object key
            method: HTTP method
            expiration: Expiration time in seconds
            headers: Additional headers

        Returns:
            str: Presigned URL

        Raises:
            ObjectException: If URL generation fails
        """
        try:
            if not self._connected:
                raise ConnectionException("Not connected to MinIO", provider="minio")

            # Validate inputs
            if not key:
                raise ValidationException("Object key cannot be empty", provider="minio")

            # Prepare object name
            object_name = key.lstrip("/")

            # Generate presigned URL
            url = self._client.presigned_get_object(
                self.bucket_name,
                object_name,
                expires=timedelta(seconds=expiration)
            )

            logger.debug(f"Generated presigned URL for {key}")
            return url

        except S3Error as e:
            error_msg = f"Failed to generate presigned URL for {key}: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e
        except Exception as e:
            error_msg = f"Unexpected error generating presigned URL for {key}: {str(e)}"
            logger.error(error_msg)
            raise ObjectException(error_msg, provider="minio") from e