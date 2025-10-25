"""
Local File System Storage Client Implementation

This module provides a local file system storage client that implements
the StorageInterface abstract base class.
"""

import asyncio
import os
import shutil
import hashlib
import json
import mimetypes
from typing import List, Dict, Any, Optional, Union, BinaryIO
import uuid
import time
from datetime import datetime, timezone
from dataclasses import asdict

from ..interfaces.storage_interface import (
    StorageInterface,
    StorageConfig,
    StorageCapabilities,
    StorageType,
    StorageObject,
    UploadResult,
    DownloadResult,
    AccessControl,
    StorageClass,
    StorageException,
    ConnectionException,
    BucketException,
    ObjectException,
    UploadException,
    DownloadException,
    ValidationException,
    PermissionException
)


class LocalStorageClient(StorageInterface):
    """
    Local file system storage client implementation.

    Provides file system-based storage operations with comprehensive
    error handling and metadata management.
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize local storage client with configuration.

        Args:
            config: Storage configuration with local-specific settings
        """
        super().__init__(config)

        # Local storage specific configuration
        self.base_path = config.custom_options.get('base_path') if config.custom_options else None
        if not self.base_path:
            # Use bucket_name as base directory if not specified
            self.base_path = os.path.join(os.getcwd(), 'storage', self.bucket_name)

        # Ensure base path exists
        os.makedirs(self.base_path, exist_ok=True)

        # Metadata directory
        self.metadata_dir = os.path.join(self.base_path, '.metadata')
        os.makedirs(self.metadata_dir, exist_ok=True)

        self._connected = False

    @property
    def capabilities(self) -> StorageCapabilities:
        """Get local storage capabilities."""
        return StorageCapabilities(
            provider="local",
            supported_storage_classes=[
                StorageClass.STANDARD  # Local storage only supports standard
            ],
            supported_access_controls=[
                AccessControl.PRIVATE  # Local files are private by default
            ],
            max_object_size=None,  # Limited by disk space
            max_bucket_size=None,  # Limited by disk space
            max_keys_per_bucket=None,  # Limited by file system limits
            supports_multipart_upload=False,  # Not applicable for local files
            supports_presigned_urls=False,  # Not applicable for local files
            supports_encryption=False,  # Can be added if needed
            supports_versioning=False,  # Can be added if needed
            supports_lifecycle_management=False,
            supports_replication=False,
            supports_cross_region_replication=False,
            supports_logging=False,
            supports_notifications=False,
            supports_analytics=False,
            supports_search=False,  # Can use filesystem search
            supports_async_operations=True
        )

    async def connect(self) -> bool:
        """
        Connect to local storage service.

        Returns:
            bool: True if connection successful
        """
        try:
            # Check if base path exists and is accessible
            if not os.path.exists(self.base_path):
                os.makedirs(self.base_path, exist_ok=True)

            # Test write permission
            test_file = os.path.join(self.base_path, '.connect_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)

            self._connected = True
            return True

        except PermissionError as e:
            raise ConnectionException(
                f"Permission denied accessing storage path '{self.base_path}': {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                error_code="PERMISSION_DENIED"
            )
        except Exception as e:
            raise ConnectionException(
                f"Failed to connect to local storage: {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                error_code="CONNECTION_ERROR"
            )

    async def disconnect(self) -> None:
        """Disconnect from local storage service."""
        self._connected = False

    def _get_object_path(self, key: str) -> str:
        """Get the file system path for an object."""
        # Sanitize key to prevent directory traversal
        key = key.replace('..', '').replace('\\', '/')
        return os.path.join(self.base_path, key)

    def _get_metadata_path(self, key: str) -> str:
        """Get the metadata file path for an object."""
        # Use hash of key as filename to avoid special characters
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return os.path.join(self.metadata_dir, f"{key_hash}.json")

    def _save_metadata(self, key: str, metadata: Dict[str, Any]) -> None:
        """Save object metadata."""
        metadata_path = self._get_metadata_path(key)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self, key: str) -> Dict[str, Any]:
        """Load object metadata."""
        metadata_path = self._get_metadata_path(key)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return {}

    def _delete_metadata(self, key: str) -> None:
        """Delete object metadata."""
        metadata_path = self._get_metadata_path(key)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)

    async def create_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """
        Create a new bucket (directory).

        Args:
            bucket_name: Bucket name (optional, uses default if not provided)

        Returns:
            bool: True if creation successful

        Raises:
            BucketException: If bucket creation fails
        """
        target_bucket = bucket_name or self.bucket_name

        try:
            bucket_path = os.path.join(os.path.dirname(self.base_path), target_bucket)
            os.makedirs(bucket_path, exist_ok=True)

            # Create metadata directory for bucket
            metadata_path = os.path.join(bucket_path, '.metadata')
            os.makedirs(metadata_path, exist_ok=True)

            return True

        except PermissionError as e:
            raise BucketException(
                f"Permission denied creating bucket '{target_bucket}': {str(e)}",
                provider=self.provider_name,
                bucket=target_bucket,
                error_code="PERMISSION_DENIED"
            )
        except Exception as e:
            raise BucketException(
                f"Failed to create bucket '{target_bucket}': {str(e)}",
                provider=self.provider_name,
                bucket=target_bucket
            )

    async def delete_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """
        Delete a bucket (directory).

        Args:
            bucket_name: Bucket name (optional, uses default if not provided)

        Returns:
            bool: True if deletion successful

        Raises:
            BucketException: If bucket deletion fails
        """
        target_bucket = bucket_name or self.bucket_name

        if target_bucket == self.bucket_name:
            raise BucketException(
                "Cannot delete the default bucket",
                provider=self.provider_name,
                bucket=target_bucket,
                error_code="CANNOT_DELETE_DEFAULT"
            )

        try:
            bucket_path = os.path.join(os.path.dirname(self.base_path), target_bucket)

            if not os.path.exists(bucket_path):
                raise BucketException(
                    f"Bucket '{target_bucket}' does not exist",
                    provider=self.provider_name,
                    bucket=target_bucket,
                    error_code="BUCKET_NOT_FOUND"
                )

            # Check if bucket is empty
            if os.listdir(bucket_path):
                # Allow deletion of .metadata directory only
                items = [item for item in os.listdir(bucket_path) if item != '.metadata']
                if items:
                    raise BucketException(
                        f"Bucket '{target_bucket}' is not empty",
                        provider=self.provider_name,
                        bucket=target_bucket,
                        error_code="BUCKET_NOT_EMPTY"
                    )

            # Delete bucket
            shutil.rmtree(bucket_path)
            return True

        except BucketException:
            raise
        except Exception as e:
            raise BucketException(
                f"Failed to delete bucket '{target_bucket}': {str(e)}",
                provider=self.provider_name,
                bucket=target_bucket
            )

    async def list_buckets(self) -> List[str]:
        """
        List all buckets (directories).

        Returns:
            List[str]: Bucket names

        Raises:
            BucketException: If listing fails
        """
        try:
            storage_root = os.path.dirname(self.base_path)
            if not os.path.exists(storage_root):
                return []

            buckets = []
            for item in os.listdir(storage_root):
                item_path = os.path.join(storage_root, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    buckets.append(item)

            return buckets

        except Exception as e:
            raise BucketException(
                f"Failed to list buckets: {str(e)}",
                provider=self.provider_name
            )

    async def bucket_exists(self, bucket_name: Optional[str] = None) -> bool:
        """
        Check if a bucket (directory) exists.

        Args:
            bucket_name: Bucket name (optional, uses default if not provided)

        Returns:
            bool: True if bucket exists

        Raises:
            BucketException: If check fails
        """
        target_bucket = bucket_name or self.bucket_name

        try:
            bucket_path = os.path.join(os.path.dirname(self.base_path), target_bucket)
            return os.path.exists(bucket_path) and os.path.isdir(bucket_path)

        except Exception as e:
            raise BucketException(
                f"Failed to check bucket existence: {str(e)}",
                provider=self.provider_name,
                bucket=target_bucket
            )

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
        Upload an object to local storage.

        Args:
            key: Object key
            data: Object data
            content_type: Content type
            metadata: Object metadata
            storage_class: Storage class (ignored for local storage)
            access_control: Access control (ignored for local storage)

        Returns:
            UploadResult: Upload result

        Raises:
            UploadException: If upload fails
        """
        if not key:
            raise ValidationException("Object key cannot be empty")

        if not data:
            raise ValidationException("Object data cannot be empty")

        try:
            start_time = time.time()

            # Get object path
            object_path = self._get_object_path(key)

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(object_path), exist_ok=True)

            # Write data
            if isinstance(data, bytes):
                with open(object_path, 'wb') as f:
                    f.write(data)
                size = len(data)
            else:
                with open(object_path, 'wb') as f:
                    data.seek(0)
                    shutil.copyfileobj(data, f)
                # Get file size
                data.seek(0, 2)  # Seek to end
                size = data.tell()
                data.seek(0)  # Reset position

            # Calculate ETag (MD5 hash)
            with open(object_path, 'rb') as f:
                etag = hashlib.md5(f.read()).hexdigest()

            # Guess content type if not provided
            if not content_type:
                content_type, _ = mimetypes.guess_type(key)
                if not content_type:
                    content_type = 'application/octet-stream'

            upload_time = (time.time() - start_time) * 1000

            # Save metadata
            metadata_data = {
                'key': key,
                'bucket': self.bucket_name,
                'size': size,
                'content_type': content_type,
                'etag': etag,
                'upload_time': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {},
                'storage_class': 'STANDARD',
                'access_control': 'PRIVATE'
            }
            self._save_metadata(key, metadata_data)

            return UploadResult(
                key=key,
                bucket=self.bucket_name,
                etag=etag,
                size=size,
                url=f"file://{object_path}",
                metadata=metadata,
                upload_time_ms=upload_time
            )

        except PermissionError as e:
            raise UploadException(
                f"Permission denied uploading object '{key}': {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key,
                error_code="PERMISSION_DENIED"
            )
        except Exception as e:
            raise UploadException(
                f"Unexpected error uploading object '{key}': {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key
            )

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
        Upload an object from file to local storage.

        Args:
            key: Object key
            file_path: Source file path
            content_type: Content type
            metadata: Object metadata
            storage_class: Storage class (ignored)
            access_control: Access control (ignored)

        Returns:
            UploadResult: Upload result

        Raises:
            UploadException: If upload fails
        """
        if not os.path.exists(file_path):
            raise ValidationException(f"Source file '{file_path}' does not exist")

        if not os.path.isfile(file_path):
            raise ValidationException(f"Source path '{file_path}' is not a file")

        try:
            start_time = time.time()

            # Get object path
            object_path = self._get_object_path(key)

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(object_path), exist_ok=True)

            # Copy file
            shutil.copy2(file_path, object_path)

            # Get file size
            size = os.path.getsize(object_path)

            # Calculate ETag (MD5 hash)
            with open(object_path, 'rb') as f:
                etag = hashlib.md5(f.read()).hexdigest()

            # Guess content type if not provided
            if not content_type:
                content_type, _ = mimetypes.guess_type(file_path)
                if not content_type:
                    content_type, _ = mimetypes.guess_type(key)
                if not content_type:
                    content_type = 'application/octet-stream'

            upload_time = (time.time() - start_time) * 1000

            # Save metadata
            metadata_data = {
                'key': key,
                'bucket': self.bucket_name,
                'size': size,
                'content_type': content_type,
                'etag': etag,
                'upload_time': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {},
                'storage_class': 'STANDARD',
                'access_control': 'PRIVATE',
                'source_file': file_path
            }
            self._save_metadata(key, metadata_data)

            return UploadResult(
                key=key,
                bucket=self.bucket_name,
                etag=etag,
                size=size,
                url=f"file://{object_path}",
                metadata=metadata,
                upload_time_ms=upload_time
            )

        except Exception as e:
            raise UploadException(
                f"Failed to upload file '{file_path}' to '{key}': {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key
            )

    async def download_object(self, key: str) -> DownloadResult:
        """
        Download an object from local storage.

        Args:
            key: Object key

        Returns:
            DownloadResult: Download result

        Raises:
            DownloadException: If download fails
        """
        if not key:
            raise ValidationException("Object key cannot be empty")

        try:
            start_time = time.time()

            object_path = self._get_object_path(key)

            if not os.path.exists(object_path):
                raise DownloadException(
                    f"Object '{key}' not found",
                    provider=self.provider_name,
                    bucket=self.bucket_name,
                    key=key,
                    error_code="OBJECT_NOT_FOUND"
                )

            # Read file content
            with open(object_path, 'rb') as f:
                content = f.read()

            download_time = (time.time() - start_time) * 1000

            # Load metadata
            metadata_data = self._load_metadata(key)
            content_type = metadata_data.get('content_type', 'application/octet-stream')
            etag = metadata_data.get('etag', '')
            metadata = metadata_data.get('metadata', {})

            return DownloadResult(
                key=key,
                bucket=self.bucket_name,
                content=content,
                content_type=content_type,
                etag=etag,
                metadata=metadata,
                download_time_ms=download_time
            )

        except DownloadException:
            raise
        except Exception as e:
            raise DownloadException(
                f"Unexpected error downloading object '{key}': {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key
            )

    async def download_object_to_file(
        self,
        key: str,
        file_path: str
    ) -> bool:
        """
        Download an object from local storage to file.

        Args:
            key: Object key
            file_path: Destination file path

        Returns:
            bool: True if download successful

        Raises:
            DownloadException: If download fails
        """
        if not key:
            raise ValidationException("Object key cannot be empty")

        try:
            object_path = self._get_object_path(key)

            if not os.path.exists(object_path):
                raise DownloadException(
                    f"Object '{key}' not found",
                    provider=self.provider_name,
                    bucket=self.bucket_name,
                    key=key,
                    error_code="OBJECT_NOT_FOUND"
                )

            # Ensure destination directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Copy file
            shutil.copy2(object_path, file_path)

            return True

        except DownloadException:
            raise
        except Exception as e:
            raise DownloadException(
                f"Failed to download object '{key}' to file: {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key
            )

    async def delete_object(self, key: str) -> bool:
        """
        Delete an object from local storage.

        Args:
            key: Object key

        Returns:
            bool: True if deletion successful

        Raises:
            ObjectException: If deletion fails
        """
        if not key:
            raise ValidationException("Object key cannot be empty")

        try:
            object_path = self._get_object_path(key)

            if not os.path.exists(object_path):
                # Object doesn't exist, consider it deleted
                return True

            os.remove(object_path)

            # Delete metadata
            self._delete_metadata(key)

            return True

        except PermissionError as e:
            raise ObjectException(
                f"Permission denied deleting object '{key}': {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key,
                error_code="PERMISSION_DENIED"
            )
        except Exception as e:
            raise ObjectException(
                f"Failed to delete object '{key}': {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key
            )

    async def delete_objects(self, keys: List[str]) -> Dict[str, bool]:
        """
        Delete multiple objects from local storage.

        Args:
            keys: Object keys

        Returns:
            Dict[str, bool]: Deletion results
        """
        results = {}

        for key in keys:
            try:
                results[key] = await self.delete_object(key)
            except Exception as e:
                results[key] = False

        return results

    async def object_exists(self, key: str) -> bool:
        """
        Check if an object exists in local storage.

        Args:
            key: Object key

        Returns:
            bool: True if object exists

        Raises:
            ObjectException: If check fails
        """
        if not key:
            raise ValidationException("Object key cannot be empty")

        try:
            object_path = self._get_object_path(key)
            return os.path.exists(object_path) and os.path.isfile(object_path)

        except Exception as e:
            raise ObjectException(
                f"Failed to check object existence: {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key
            )

    async def get_object_info(self, key: str) -> StorageObject:
        """
        Get object information from local storage.

        Args:
            key: Object key

        Returns:
            StorageObject: Object information

        Raises:
            ObjectException: If getting info fails
        """
        if not key:
            raise ValidationException("Object key cannot be empty")

        try:
            object_path = self._get_object_path(key)

            if not os.path.exists(object_path):
                raise ObjectException(
                    f"Object '{key}' not found",
                    provider=self.provider_name,
                    bucket=self.bucket_name,
                    key=key,
                    error_code="OBJECT_NOT_FOUND"
                )

            # Get file stats
            stat = os.stat(object_path)
            size = stat.st_size
            last_modified = time.gmtime(stat.st_mtime)

            # Load metadata
            metadata_data = self._load_metadata(key)
            content_type = metadata_data.get('content_type', 'application/octet-stream')
            etag = metadata_data.get('etag', '')
            metadata = metadata_data.get('metadata', {})

            return StorageObject(
                key=key,
                bucket=self.bucket_name,
                size=size,
                content_type=content_type,
                etag=etag,
                last_modified=last_modified,
                metadata=metadata,
                url=f"file://{object_path}",
                storage_class=StorageClass.STANDARD,
                access_control=AccessControl.PRIVATE
            )

        except ObjectException:
            raise
        except Exception as e:
            raise ObjectException(
                f"Failed to get object info: {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key
            )

    async def list_objects(
        self,
        prefix: Optional[str] = None,
        limit: Optional[int] = None,
        continuation_token: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        List objects in local storage bucket.

        Args:
            prefix: Object prefix filter
            limit: Maximum number of objects to return
            continuation_token: Continuation token for pagination

        Returns:
            Dict[str, Any]: List result with pagination info
        """
        try:
            objects = []
            count = 0

            # Walk through directory
            for root, dirs, files in os.walk(self.base_path):
                # Skip metadata directory
                if '.metadata' in root:
                    continue

                for file in files:
                    # Get relative path
                    rel_path = os.path.relpath(os.path.join(root, file), self.base_path)
                    key = rel_path.replace('\\', '/')

                    # Apply prefix filter
                    if prefix and not key.startswith(prefix):
                        continue

                    # Apply limit
                    if limit and count >= limit:
                        break

                    # Get object info
                    try:
                        obj_info = await self.get_object_info(key)
                        objects.append(asdict(obj_info))
                        count += 1
                    except:
                        # Skip invalid objects
                        continue

                if limit and count >= limit:
                    break

            return {
                'objects': objects,
                'count': len(objects),
                'is_truncated': False,  # Local storage doesn't have pagination
                'next_continuation_token': None,
                'prefix': prefix,
                'limit': limit
            }

        except Exception as e:
            raise ObjectException(
                f"Failed to list objects: {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name
            )

    async def copy_object(
        self,
        source_key: str,
        destination_key: str,
        source_bucket: Optional[str] = None,
        destination_bucket: Optional[str] = None
    ) -> bool:
        """
        Copy an object in local storage.

        Args:
            source_key: Source object key
            destination_key: Destination object key
            source_bucket: Source bucket (optional, uses default)
            destination_bucket: Destination bucket (optional, uses default)

        Returns:
            bool: True if copy successful
        """
        if not source_key or not destination_key:
            raise ValidationException("Source and destination keys cannot be empty")

        try:
            # Determine source and destination paths
            if source_bucket:
                source_base = os.path.join(os.path.dirname(self.base_path), source_bucket)
            else:
                source_base = self.base_path

            if destination_bucket:
                dest_base = os.path.join(os.path.dirname(self.base_path), destination_bucket)
                os.makedirs(dest_base, exist_ok=True)
            else:
                dest_base = self.base_path

            source_path = os.path.join(source_base, source_key.replace('\\', '/'))
            dest_path = os.path.join(dest_base, destination_key.replace('\\', '/'))

            if not os.path.exists(source_path):
                raise ObjectException(
                    f"Source object '{source_key}' not found",
                    provider=self.provider_name,
                    bucket=source_bucket or self.bucket_name,
                    key=source_key,
                    error_code="OBJECT_NOT_FOUND"
                )

            # Create destination directory
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # Copy file
            shutil.copy2(source_path, dest_path)

            # Copy metadata if exists
            if source_bucket == self.bucket_name and destination_bucket == self.bucket_name:
                # Copy metadata file
                source_metadata = self._load_metadata(source_key)
                if source_metadata:
                    source_metadata['key'] = destination_key
                    source_metadata['bucket'] = destination_bucket or self.bucket_name
                    self._save_metadata(destination_key, source_metadata)

            return True

        except Exception as e:
            if isinstance(e, ObjectException):
                raise
            else:
                raise ObjectException(
                    f"Failed to copy object '{source_key}' to '{destination_key}': {str(e)}",
                    provider=self.provider_name,
                    bucket=destination_bucket or self.bucket_name,
                    key=destination_key
                )

    async def move_object(
        self,
        source_key: str,
        destination_key: str,
        source_bucket: Optional[str] = None,
        destination_bucket: Optional[str] = None
    ) -> bool:
        """
        Move an object in local storage.

        Args:
            source_key: Source object key
            destination_key: Destination object key
            source_bucket: Source bucket (optional)
            destination_bucket: Destination bucket (optional)

        Returns:
            bool: True if move successful
        """
        try:
            # Copy object
            await self.copy_object(source_key, destination_key, source_bucket, destination_bucket)

            # Delete source object
            if source_bucket == self.bucket_name:
                await self.delete_object(source_key)

            return True

        except Exception as e:
            if isinstance(e, ObjectException):
                raise
            else:
                raise ObjectException(
                    f"Failed to move object: {str(e)}",
                    provider=self.provider_name
                )

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
        # Local storage doesn't support presigned URLs
        # Return file URL instead
        object_path = self._get_object_path(key)
        if os.path.exists(object_path):
            return f"file://{object_path}"
        else:
            raise ObjectException(
                f"Object '{key}' not found",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key,
                error_code="OBJECT_NOT_FOUND"
            )


# Export local storage client
__all__ = ['LocalStorageClient']