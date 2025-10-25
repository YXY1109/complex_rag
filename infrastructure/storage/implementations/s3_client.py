"""
AWS S3 Object Storage Client Implementation

This module provides an S3-compatible object storage client that implements
the StorageInterface abstract base class.
"""

import asyncio
from typing import List, Dict, Any, Optional, Union, BinaryIO
import uuid
import time
import os
from datetime import datetime, timezone
from dataclasses import asdict

try:
    import boto3
    from botocore.exceptions import (
        BotoCoreError,
        ClientError,
        NoCredentialsError,
        PartialCredentialsError,
        ConnectionError as BotocoreConnectionError
    )
    from botocore.config import Config
    import aioboto3
except ImportError:
    boto3 = None
    aioboto3 = None
    BotoCoreError = Exception
    ClientError = Exception
    NoCredentialsError = Exception
    PartialCredentialsError = Exception
    BotocoreConnectionError = Exception

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


class S3Client(StorageInterface):
    """
    AWS S3 object storage client implementation.

    Provides async S3-compatible storage operations with comprehensive
    error handling, performance optimization, and AWS-specific features.
    """

    def __init__(self, config: StorageConfig):
        """
        Initialize S3 client with configuration.

        Args:
            config: Storage configuration with S3-specific settings
        """
        super().__init__(config)

        # Validate boto3 availability
        if boto3 is None or aioboto3 is None:
            raise ImportError("boto3 and aioboto3 are required for S3 client")

        # AWS-specific configuration
        self.access_key = config.access_key
        self.secret_key = config.secret_key
        self.session_token = config.session_token
        self.region = config.region or 'us-east-1'
        self.endpoint = config.endpoint
        self.verify_ssl = config.verify_ssl

        # Initialize session and client
        self._session = None
        self._s3_client = None
        self._connected = False

        # Configure boto3
        self._boto_config = Config(
            region_name=self.region,
            retries={
                'max_attempts': config.max_retries,
                'mode': config.retry_mode
            },
            connect_timeout=config.timeout,
            read_timeout=config.timeout,
            max_pool_connections=config.max_concurrency
        )

    @property
    def capabilities(self) -> StorageCapabilities:
        """Get S3 storage capabilities."""
        return StorageCapabilities(
            provider="s3",
            supported_storage_classes=[
                StorageClass.STANDARD,
                StorageClass.REDUCED_REDUNDANCY,
                StorageClass.STANDARD_IA,
                StorageClass.ONEZONE_IA,
                StorageClass.INTELLIGENT_TIERING,
                StorageClass.GLACIER,
                StorageClass.DEEP_ARCHIVE,
                StorageClass.OUTPOSTS
            ],
            supported_access_controls=[
                AccessControl.PRIVATE,
                AccessControl.PUBLIC_READ,
                AccessControl.PUBLIC_READ_WRITE,
                AccessControl.AUTHENTICATED_READ,
                AccessControl.BUCKET_OWNER_READ,
                AccessControl.BUCKET_OWNER_FULL_CONTROL
            ],
            max_object_size=5 * 1024 * 1024 * 1024 * 1024,  # 5TB
            max_bucket_size=None,  # Unlimited
            max_keys_per_bucket=None,  # Unlimited
            supports_multipart_upload=True,
            supports_presigned_urls=True,
            supports_encryption=True,
            supports_versioning=True,
            supports_lifecycle_management=True,
            supports_replication=True,
            supports_cross_region_replication=True,
            supports_logging=True,
            supports_notifications=True,
            supports_analytics=True,
            supports_search=True,
            supports_async_operations=True
        )

    async def connect(self) -> bool:
        """
        Connect to S3 service.

        Returns:
            bool: True if connection successful

        Raises:
            ConnectionException: If connection fails
        """
        try:
            # Create session
            self._session = aioboto3.Session(
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                aws_session_token=self.session_token,
                region_name=self.region
            )

            # Test connection
            async with self._session.client('s3',
                                          endpoint_url=self.endpoint,
                                          config=self._boto_config,
                                          verify=self.verify_ssl) as s3:
                # List buckets to test connection
                await s3.list_buckets()

            self._connected = True
            return True

        except (NoCredentialsError, PartialCredentialsError) as e:
            raise ConnectionException(
                f"S3 credentials error: {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                error_code="CREDENTIALS_ERROR"
            )
        except (BotoCoreError, ClientError, BotocoreConnectionError) as e:
            raise ConnectionException(
                f"S3 connection failed: {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                error_code="CONNECTION_ERROR"
            )
        except Exception as e:
            raise ConnectionException(
                f"Unexpected connection error: {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                error_code="UNKNOWN_ERROR"
            )

    async def disconnect(self) -> None:
        """Disconnect from S3 service."""
        self._session = None
        self._s3_client = None
        self._connected = False

    def _get_client(self):
        """Get S3 client."""
        if not self._session:
            raise ConnectionException(
                "Not connected to S3 service",
                provider=self.provider_name
            )
        return self._session.client('s3',
                                   endpoint_url=self.endpoint,
                                   config=self._boto_config,
                                   verify=self.verify_ssl)

    async def create_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """
        Create a new S3 bucket.

        Args:
            bucket_name: Bucket name (optional, uses default if not provided)

        Returns:
            bool: True if creation successful

        Raises:
            BucketException: If bucket creation fails
        """
        target_bucket = bucket_name or self.bucket_name

        try:
            async with self._get_client() as s3:
                if self.region == 'us-east-1':
                    # us-east-1 doesn't need LocationConstraint
                    await s3.create_bucket(Bucket=target_bucket)
                else:
                    await s3.create_bucket(
                        Bucket=target_bucket,
                        CreateBucketConfiguration={'LocationConstraint': self.region}
                    )

                return True

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'BucketAlreadyExists':
                raise BucketException(
                    f"Bucket '{target_bucket}' already exists",
                    provider=self.provider_name,
                    bucket=target_bucket,
                    error_code="BUCKET_EXISTS"
                )
            elif error_code == 'BucketAlreadyOwnedByYou':
                # Bucket already exists and is owned by us
                return True
            else:
                raise BucketException(
                    f"Failed to create bucket '{target_bucket}': {str(e)}",
                    provider=self.provider_name,
                    bucket=target_bucket,
                    error_code=error_code
                )
        except Exception as e:
            raise BucketException(
                f"Unexpected error creating bucket '{target_bucket}': {str(e)}",
                provider=self.provider_name,
                bucket=target_bucket
            )

    async def delete_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """
        Delete an S3 bucket.

        Args:
            bucket_name: Bucket name (optional, uses default if not provided)

        Returns:
            bool: True if deletion successful

        Raises:
            BucketException: If bucket deletion fails
        """
        target_bucket = bucket_name or self.bucket_name

        try:
            async with self._get_client() as s3:
                await s3.delete_bucket(Bucket=target_bucket)
                return True

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise BucketException(
                    f"Bucket '{target_bucket}' does not exist",
                    provider=self.provider_name,
                    bucket=target_bucket,
                    error_code="BUCKET_NOT_FOUND"
                )
            elif error_code == 'BucketNotEmpty':
                raise BucketException(
                    f"Bucket '{target_bucket}' is not empty",
                    provider=self.provider_name,
                    bucket=target_bucket,
                    error_code="BUCKET_NOT_EMPTY"
                )
            else:
                raise BucketException(
                    f"Failed to delete bucket '{target_bucket}': {str(e)}",
                    provider=self.provider_name,
                    bucket=target_bucket,
                    error_code=error_code
                )
        except Exception as e:
            raise BucketException(
                f"Unexpected error deleting bucket '{target_bucket}': {str(e)}",
                provider=self.provider_name,
                bucket=target_bucket
            )

    async def list_buckets(self) -> List[str]:
        """
        List all S3 buckets.

        Returns:
            List[str]: Bucket names

        Raises:
            BucketException: If listing fails
        """
        try:
            async with self._get_client() as s3:
                response = await s3.list_buckets()
                return [bucket['Name'] for bucket in response['Buckets']]

        except (ClientError, BotoCoreError) as e:
            raise BucketException(
                f"Failed to list buckets: {str(e)}",
                provider=self.provider_name,
                error_code="LIST_BUCKETS_ERROR"
            )
        except Exception as e:
            raise BucketException(
                f"Unexpected error listing buckets: {str(e)}",
                provider=self.provider_name
            )

    async def bucket_exists(self, bucket_name: Optional[str] = None) -> bool:
        """
        Check if an S3 bucket exists.

        Args:
            bucket_name: Bucket name (optional, uses default if not provided)

        Returns:
            bool: True if bucket exists

        Raises:
            BucketException: If check fails
        """
        target_bucket = bucket_name or self.bucket_name

        try:
            async with self._get_client() as s3:
                await s3.head_bucket(Bucket=target_bucket)
                return True

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return False
            else:
                raise BucketException(
                    f"Failed to check bucket existence: {str(e)}",
                    provider=self.provider_name,
                    bucket=target_bucket,
                    error_code=error_code
                )
        except Exception as e:
            raise BucketException(
                f"Unexpected error checking bucket existence: {str(e)}",
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
        Upload an object to S3.

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
        if not key:
            raise ValidationException("Object key cannot be empty")

        if not data:
            raise ValidationException("Object data cannot be empty")

        # Prepare upload parameters
        upload_args = {
            'Bucket': self.bucket_name,
            'Key': key
        }

        # Set content type
        if content_type:
            upload_args['ContentType'] = content_type

        # Set metadata
        if metadata:
            upload_args['Metadata'] = metadata

        # Set storage class
        if storage_class:
            upload_args['StorageClass'] = storage_class.value

        # Set ACL
        if access_control:
            upload_args['ACL'] = access_control.value

        try:
            start_time = time.time()

            async with self._get_client() as s3:
                if isinstance(data, bytes):
                    upload_args['Body'] = data
                else:
                    upload_args['Body'] = data

                response = await s3.put_object(**upload_args)

            upload_time = (time.time() - start_time) * 1000

            # Create URL
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{key}"
            if self.endpoint:
                url = f"{self.endpoint}/{self.bucket_name}/{key}"

            # Get object size
            if isinstance(data, bytes):
                size = len(data)
            else:
                # For file-like objects, get current position and seek to end
                current_pos = data.tell() if hasattr(data, 'tell') else 0
                data.seek(0, 2)  # Seek to end
                size = data.tell()
                data.seek(current_pos)  # Reset position

            return UploadResult(
                key=key,
                bucket=self.bucket_name,
                etag=response['ETag'].strip('"'),
                size=size,
                url=url,
                metadata=metadata,
                upload_time_ms=upload_time
            )

        except ClientError as e:
            error_code = e.response['Error']['Code']
            raise UploadException(
                f"Failed to upload object '{key}': {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key,
                error_code=error_code
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
        Upload an object from file to S3.

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
        if not os.path.exists(file_path):
            raise ValidationException(f"File '{file_path}' does not exist")

        if not os.path.isfile(file_path):
            raise ValidationException(f"Path '{file_path}' is not a file")

        # Get file size
        file_size = os.path.getsize(file_path)

        # Determine if multipart upload is needed
        use_multipart = file_size > self.multipart_threshold

        try:
            if use_multipart:
                return await self._multipart_upload_from_file(
                    key, file_path, content_type, metadata, storage_class, access_control
                )
            else:
                # Simple upload for small files
                start_time = time.time()

                async with self._get_client() as s3:
                    upload_args = {
                        'Bucket': self.bucket_name,
                        'Key': key,
                        'Filename': file_path
                    }

                    if content_type:
                        upload_args['ContentType'] = content_type
                    if metadata:
                        upload_args['Metadata'] = metadata
                    if storage_class:
                        upload_args['StorageClass'] = storage_class.value
                    if access_control:
                        upload_args['ACL'] = access_control.value

                    await s3.upload_file(**upload_args)

                upload_time = (time.time() - start_time) * 1000

                # Get object info
                response = await s3.head_object(Bucket=self.bucket_name, Key=key)

                url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{key}"
                if self.endpoint:
                    url = f"{self.endpoint}/{self.bucket_name}/{key}"

                return UploadResult(
                    key=key,
                    bucket=self.bucket_name,
                    etag=response['ETag'].strip('"'),
                    size=response['ContentLength'],
                    url=url,
                    metadata=response.get('Metadata', {}),
                    upload_time_ms=upload_time
                )

        except Exception as e:
            if isinstance(e, UploadException):
                raise
            else:
                raise UploadException(
                    f"Failed to upload file '{file_path}' to '{key}': {str(e)}",
                    provider=self.provider_name,
                    bucket=self.bucket_name,
                    key=key
                )

    async def _multipart_upload_from_file(
        self,
        key: str,
        file_path: str,
        content_type: Optional[str],
        metadata: Optional[Dict[str, str]],
        storage_class: Optional[StorageClass],
        access_control: Optional[AccessControl]
    ) -> UploadResult:
        """Perform multipart upload for large files."""
        async with self._get_client() as s3:
            # Create multipart upload
            create_args = {
                'Bucket': self.bucket_name,
                'Key': key
            }

            if content_type:
                create_args['ContentType'] = content_type
            if metadata:
                create_args['Metadata'] = metadata
            if storage_class:
                create_args['StorageClass'] = storage_class.value
            if access_control:
                create_args['ACL'] = access_control.value

            response = await s3.create_multipart_upload(**create_args)
            upload_id = response['UploadId']

            try:
                # Upload parts
                parts = []
                part_number = 1

                with open(file_path, 'rb') as file:
                    while True:
                        chunk = file.read(self.multipart_chunksize)
                        if not chunk:
                            break

                        part_response = await s3.upload_part(
                            Bucket=self.bucket_name,
                            Key=key,
                            PartNumber=part_number,
                            UploadId=upload_id,
                            Body=chunk
                        )

                        parts.append({
                            'ETag': part_response['ETag'],
                            'PartNumber': part_number
                        })

                        part_number += 1

                # Complete multipart upload
                complete_response = await s3.complete_multipart_upload(
                    Bucket=self.bucket_name,
                    Key=key,
                    UploadId=upload_id,
                    MultipartUpload={'Parts': parts}
                )

                # Get file size
                file_size = os.path.getsize(file_path)

                url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{key}"
                if self.endpoint:
                    url = f"{self.endpoint}/{self.bucket_name}/{key}"

                return UploadResult(
                    key=key,
                    bucket=self.bucket_name,
                    etag=complete_response['ETag'].strip('"'),
                    size=file_size,
                    url=url,
                    metadata=metadata or {}
                )

            except Exception as e:
                # Abort multipart upload on error
                try:
                    await s3.abort_multipart_upload(
                        Bucket=self.bucket_name,
                        Key=key,
                        UploadId=upload_id
                    )
                except:
                    pass

                raise UploadException(
                    f"Multipart upload failed for '{key}': {str(e)}",
                    provider=self.provider_name,
                    bucket=self.bucket_name,
                    key=key
                )

    async def download_object(self, key: str) -> DownloadResult:
        """
        Download an object from S3.

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

            async with self._get_client() as s3:
                # Get object
                response = await s3.get_object(
                    Bucket=self.bucket_name,
                    Key=key
                )

                # Read content
                content = await response['Body'].read()

            download_time = (time.time() - start_time) * 1000

            return DownloadResult(
                key=key,
                bucket=self.bucket_name,
                content=content,
                content_type=response.get('ContentType', 'application/octet-stream'),
                etag=response.get('ETag', '').strip('"'),
                metadata=response.get('Metadata', {}),
                download_time_ms=download_time
            )

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise DownloadException(
                    f"Object '{key}' not found",
                    provider=self.provider_name,
                    bucket=self.bucket_name,
                    key=key,
                    error_code="OBJECT_NOT_FOUND"
                )
            else:
                raise DownloadException(
                    f"Failed to download object '{key}': {str(e)}",
                    provider=self.provider_name,
                    bucket=self.bucket_name,
                    key=key,
                    error_code=error_code
                )
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
        Download an object from S3 to file.

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

        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        try:
            async with self._get_client() as s3:
                await s3.download_file(
                    Bucket=self.bucket_name,
                    Key=key,
                    Filename=file_path
                )

            return True

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchKey':
                raise DownloadException(
                    f"Object '{key}' not found",
                    provider=self.provider_name,
                    bucket=self.bucket_name,
                    key=key,
                    error_code="OBJECT_NOT_FOUND"
                )
            else:
                raise DownloadException(
                    f"Failed to download object '{key}' to file: {str(e)}",
                    provider=self.provider_name,
                    bucket=self.bucket_name,
                    key=key,
                    error_code=error_code
                )
        except Exception as e:
            raise DownloadException(
                f"Unexpected error downloading object '{key}' to file: {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key
            )

    async def delete_object(self, key: str) -> bool:
        """
        Delete an object from S3.

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
            async with self._get_client() as s3:
                await s3.delete_object(
                    Bucket=self.bucket_name,
                    Key=key
                )

            return True

        except ClientError as e:
            raise ObjectException(
                f"Failed to delete object '{key}': {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key,
                error_code=e.response['Error']['Code']
            )
        except Exception as e:
            raise ObjectException(
                f"Unexpected error deleting object '{key}': {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key
            )

    async def delete_objects(self, keys: List[str]) -> Dict[str, bool]:
        """
        Delete multiple objects from S3.

        Args:
            keys: Object keys

        Returns:
            Dict[str, bool]: Deletion results

        Raises:
            ObjectException: If deletion fails
        """
        if not keys:
            return {}

        try:
            # Prepare delete request
            delete_objects = [{'Key': key} for key in keys]

            async with self._get_client() as s3:
                response = await s3.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={'Objects': delete_objects}
                )

            # Process results
            results = {key: True for key in keys}

            # Mark failed deletions
            if 'Errors' in response:
                for error in response['Errors']:
                    results[error['Key']] = False

            return results

        except Exception as e:
            raise ObjectException(
                f"Failed to delete objects: {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name
            )

    async def object_exists(self, key: str) -> bool:
        """
        Check if an object exists in S3.

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
            async with self._get_client() as s3:
                await s3.head_object(
                    Bucket=self.bucket_name,
                    Key=key
                )
            return True

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                return False
            else:
                raise ObjectException(
                    f"Failed to check object existence: {str(e)}",
                    provider=self.provider_name,
                    bucket=self.bucket_name,
                    key=key,
                    error_code=error_code
                )
        except Exception as e:
            raise ObjectException(
                f"Unexpected error checking object existence: {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key
            )

    async def get_object_info(self, key: str) -> StorageObject:
        """
        Get object information from S3.

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
            async with self._get_client() as s3:
                response = await s3.head_object(
                    Bucket=self.bucket_name,
                    Key=key
                )

            # Parse last modified
            last_modified = response['LastModified'].timetuple()

            # Create URL
            url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{key}"
            if self.endpoint:
                url = f"{self.endpoint}/{self.bucket_name}/{key}"

            # Map storage class
            storage_class_map = {
                'STANDARD': StorageClass.STANDARD,
                'REDUCED_REDUNDANCY': StorageClass.REDUCED_REDUNDANCY,
                'STANDARD_IA': StorageClass.STANDARD_IA,
                'ONEZONE_IA': StorageClass.ONEZONE_IA,
                'INTELLIGENT_TIERING': StorageClass.INTELLIGENT_TIERING,
                'GLACIER': StorageClass.GLACIER,
                'DEEP_ARCHIVE': StorageClass.DEEP_ARCHIVE,
                'OUTPOSTS': StorageClass.OUTPOSTS
            }

            storage_class = storage_class_map.get(
                response.get('StorageClass', 'STANDARD'),
                StorageClass.STANDARD
            )

            return StorageObject(
                key=key,
                bucket=self.bucket_name,
                size=response['ContentLength'],
                content_type=response.get('ContentType', 'application/octet-stream'),
                etag=response.get('ETag', '').strip('"'),
                last_modified=last_modified,
                metadata=response.get('Metadata', {}),
                url=url,
                storage_class=storage_class
            )

        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                raise ObjectException(
                    f"Object '{key}' not found",
                    provider=self.provider_name,
                    bucket=self.bucket_name,
                    key=key,
                    error_code="OBJECT_NOT_FOUND"
                )
            else:
                raise ObjectException(
                    f"Failed to get object info: {str(e)}",
                    provider=self.provider_name,
                    bucket=self.bucket_name,
                    key=key,
                    error_code=error_code
                )
        except Exception as e:
            raise ObjectException(
                f"Unexpected error getting object info: {str(e)}",
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
        List objects in S3 bucket.

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
            list_args = {
                'Bucket': self.bucket_name
            }

            if prefix:
                list_args['Prefix'] = prefix
            if limit:
                list_args['MaxKeys'] = limit
            if continuation_token:
                list_args['ContinuationToken'] = continuation_token

            async with self._get_client() as s3:
                response = await s3.list_objects_v2(**list_args)

            # Process objects
            objects = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    storage_obj = StorageObject(
                        key=obj['Key'],
                        bucket=self.bucket_name,
                        size=obj['Size'],
                        content_type='application/octet-stream',  # Not in list response
                        etag=obj['ETag'].strip('"'),
                        last_modified=obj['LastModified'].timetuple(),
                        url=f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{obj['Key']}"
                    )
                    objects.append(asdict(storage_obj))

            return {
                'objects': objects,
                'count': len(objects),
                'is_truncated': response.get('IsTruncated', False),
                'next_continuation_token': response.get('NextContinuationToken'),
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
        Copy an object in S3.

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
        if not source_key or not destination_key:
            raise ValidationException("Source and destination keys cannot be empty")

        src_bucket = source_bucket or self.bucket_name
        dest_bucket = destination_bucket or self.bucket_name

        try:
            copy_source = {
                'Bucket': src_bucket,
                'Key': source_key
            }

            async with self._get_client() as s3:
                await s3.copy_object(
                    CopySource=copy_source,
                    Bucket=dest_bucket,
                    Key=destination_key
                )

            return True

        except ClientError as e:
            error_code = e.response['Error']['Code']
            raise ObjectException(
                f"Failed to copy object '{source_key}' to '{destination_key}': {str(e)}",
                provider=self.provider_name,
                bucket=dest_bucket,
                key=destination_key,
                error_code=error_code
            )
        except Exception as e:
            raise ObjectException(
                f"Unexpected error copying object: {str(e)}",
                provider=self.provider_name,
                bucket=dest_bucket,
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
        Move an object in S3 (copy then delete).

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
            # Copy object
            await self.copy_object(source_key, destination_key, source_bucket, destination_bucket)

            # Delete source object
            await self.delete_object(source_key)

            return True

        except Exception as e:
            if isinstance(e, ObjectException):
                raise
            else:
                raise ObjectException(
                    f"Unexpected error moving object: {str(e)}",
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
        if not key:
            raise ValidationException("Object key cannot be empty")

        try:
            # Map method to S3 operation
            client_method = {
                'GET': 'get_object',
                'PUT': 'put_object',
                'DELETE': 'delete_object'
            }.get(method.upper(), 'get_object')

            params = {
                'Bucket': self.bucket_name,
                'Key': key
            }

            if headers:
                if method.upper() == 'PUT':
                    params.update(headers)

            # Use synchronous boto3 for presigned URL generation
            # (aioboto3 doesn't support generate_presigned_url yet)
            s3_sync = boto3.client(
                's3',
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                aws_session_token=self.session_token,
                region_name=self.region,
                endpoint_url=self.endpoint,
                config=self._boto_config,
                verify=self.verify_ssl
            )

            url = s3_sync.generate_presigned_url(
                client_method,
                params,
                expiration
            )

            return url

        except Exception as e:
            raise ObjectException(
                f"Failed to generate presigned URL: {str(e)}",
                provider=self.provider_name,
                bucket=self.bucket_name,
                key=key
            )


# Export S3 client
__all__ = ['S3Client']