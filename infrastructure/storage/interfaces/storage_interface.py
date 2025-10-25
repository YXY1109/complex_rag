"""
Object Storage Interface Abstract Class

This module defines the abstract interface for object storage services.
All object storage implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, BinaryIO, AsyncGenerator
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time
from dataclasses import dataclass


class StorageType(str, Enum):
    """Storage types."""
    MINIO = "minio"
    S3 = "s3"
    LOCAL = "local"
    CUSTOM = "custom"


class AccessControl(str, Enum):
    """Access control types."""
    PRIVATE = "private"
    PUBLIC_READ = "public-read"
    PUBLIC_READ_WRITE = "public-read-write"
    AUTHENTICATED_READ = "authenticated-read"
    BUCKET_OWNER_READ = "bucket-owner-read"
    BUCKET_OWNER_FULL_CONTROL = "bucket-owner-full-control"


class StorageClass(str, Enum):
    """Storage class types."""
    STANDARD = "standard"
    REDUCED_REDUNDANCY = "reduced-redundancy"
    STANDARD_IA = "standard-ia"
    ONEZONE_IA = "onezone-ia"
    INTELLIGENT_TIERING = "intelligent-tiering"
    GLACIER = "glacier"
    DEEP_ARCHIVE = "deep-archive"
    OUTPOSTS = "outposts"


@dataclass
class StorageObject:
    """Storage object representation."""
    key: str
    bucket: str
    size: int
    content_type: str
    etag: Optional[str] = None
    last_modified: Optional[time.struct_time] = None
    metadata: Optional[Dict[str, str]] = None
    url: Optional[str] = None
    access_url: Optional[str] = None
    storage_class: Optional[StorageClass] = None
    access_control: Optional[AccessControl] = None


@dataclass
class UploadResult:
    """Upload result information."""
    key: str
    bucket: str
    etag: str
    size: int
    url: str
    access_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    upload_time_ms: Optional[float] = None


@dataclass
class DownloadResult:
    """Download result information."""
    key: str
    bucket: str
    content: bytes
    content_type: str
    etag: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None
    download_time_ms: Optional[float] = None


class StorageConfig(BaseModel):
    """Storage configuration model."""
    provider: str
    bucket_name: str = Field(description="Bucket name")
    region: Optional[str] = Field(default=None, description="Storage region")
    endpoint: Optional[str] = Field(default=None, description="Storage endpoint URL")
    access_key: Optional[str] = Field(default=None, description="Access key")
    secret_key: Optional[str] = Field(default=None, description="Secret key")
    session_token: Optional[str] = Field(default=None, description="Session token")

    # Connection settings
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    retry_mode: str = Field(default="adaptive", description="Retry mode")

    # SSL settings
    use_ssl: bool = Field(default=True, description="Use SSL")
    verify_ssl: bool = Field(default=True, description="Verify SSL certificates")

    # Performance settings
    max_concurrency: int = Field(default=10, description="Maximum concurrent operations")
    multipart_threshold: int = Field(default=64 * 1024 * 1024, description="Multipart upload threshold (64MB)")
    multipart_chunksize: int = Field(default=8 * 1024 * 1024, description="Multipart chunk size (8MB)")

    # Default settings
    default_storage_class: Optional[StorageClass] = None
    default_access_control: AccessControl = AccessControl.PRIVATE
    server_side_encryption: bool = Field(default=False, description="Enable server-side encryption")

    # Custom options
    custom_options: Optional[Dict[str, Any]] = None


class StorageCapabilities(BaseModel):
    """Storage capabilities model."""
    provider: str
    supported_storage_classes: List[StorageClass]
    supported_access_controls: List[AccessControl]
    max_object_size: Optional[int] = None
    max_bucket_size: Optional[int] = None
    max_keys_per_bucket: Optional[int] = None
    supports_multipart_upload: bool = False
    supports_presigned_urls: bool = False
    supports_encryption: bool = False
    supports_versioning: bool = False
    supports_lifecycle_management: bool = False
    supports_replication: bool = False
    supports_cross_region_replication: bool = False
    supports_logging: bool = False
    supports_notifications: bool = False
    supports_analytics: bool = False
    supports_search: bool = False
    supports_async_operations: bool = False


class StorageInterface(ABC):
    """
    Abstract interface for object storage services.

    This class defines the contract that all object storage implementations must follow.
    It provides a unified interface for different object storage systems while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: StorageConfig):
        """Initialize the storage client with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.bucket_name = config.bucket_name
        self.region = config.region
        self.endpoint = config.endpoint
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.use_ssl = config.use_ssl
        self.max_concurrency = config.max_concurrency
        self.multipart_threshold = config.multipart_threshold
        self.multipart_chunksize = config.multipart_chunksize
        self._capabilities: Optional[StorageCapabilities] = None

    @property
    @abstractmethod
    def capabilities(self) -> StorageCapabilities:
        """
        Get the capabilities of this storage provider.

        Returns:
            StorageCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the storage service.

        Returns:
            bool: True if connection successful

        Raises:
            StorageException: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the storage service.
        """
        pass

    @abstractmethod
    async def create_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """
        Create a new bucket.

        Args:
            bucket_name: Bucket name (optional, uses default if not provided)

        Returns:
            bool: True if creation successful

        Raises:
            StorageException: If bucket creation fails
        """
        pass

    @abstractmethod
    async def delete_bucket(self, bucket_name: Optional[str] = None) -> bool:
        """
        Delete a bucket.

        Args:
            bucket_name: Bucket name (optional, uses default if not provided)

        Returns:
            bool: True if deletion successful

        Raises:
            StorageException: If bucket deletion fails
        """
        pass

    @abstractmethod
    async def list_buckets(self) -> List[str]:
        """
        List all buckets.

        Returns:
            List[str]: Bucket names

        Raises:
            StorageException: If listing fails
        """
        pass

    @abstractmethod
    async def bucket_exists(self, bucket_name: Optional[str] = None) -> bool:
        """
        Check if a bucket exists.

        Args:
            bucket_name: Bucket name (optional, uses default if not provided)

        Returns:
            bool: True if bucket exists

        Raises:
            StorageException: If check fails
        """
        pass

    @abstractmethod
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
            StorageException: If upload fails
        """
        pass

    @abstractmethod
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
            StorageException: If upload fails
        """
        pass

    @abstractmethod
    async def download_object(self, key: str) -> DownloadResult:
        """
        Download an object.

        Args:
            key: Object key

        Returns:
            DownloadResult: Download result

        Raises:
            StorageException: If download fails
        """
        pass

    @abstractmethod
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
            StorageException: If download fails
        """
        pass

    @abstractmethod
    async def delete_object(self, key: str) -> bool:
        """
        Delete an object.

        Args:
            key: Object key

        Returns:
            bool: True if deletion successful

        Raises:
            StorageException: If deletion fails
        """
        pass

    @abstractmethod
    async def delete_objects(self, keys: List[str]) -> Dict[str, bool]:
        """
        Delete multiple objects.

        Args:
            keys: Object keys

        Returns:
            Dict[str, bool]: Deletion results

        Raises:
            StorageException: If deletion fails
        """
        pass

    @abstractmethod
    async def object_exists(self, key: str) -> bool:
        """
        Check if an object exists.

        Args:
            key: Object key

        Returns:
            bool: True if object exists

        Raises:
            StorageException: If check fails
        """
        pass

    @abstractmethod
    async def get_object_info(self, key: str) -> StorageObject:
        """
        Get object information.

        Args:
            key: Object key

        Returns:
            StorageObject: Object information

        Raises:
            StorageException: If getting info fails
        """
        pass

    @abstractmethod
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
            StorageException: If listing fails
        """
        pass

    @abstractmethod
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
            StorageException: If copy fails
        """
        pass

    @abstractmethod
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
            StorageException: If move fails
        """
        pass

    @abstractmethod
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
            StorageException: If URL generation fails
        """
        pass

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the storage service.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            # Test basic connectivity
            test_key = f"health_check_{uuid.uuid4()}.txt"
            test_data = b"Storage health check"

            start_time = time.time()
            await self.upload_object(test_key, test_data)
            upload_time = (time.time() - start_time) * 1000

            start_time = time.time()
            await self.download_object(test_key)
            download_time = (time.time() - start_time) * 1000

            await self.delete_object(test_key)

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "bucket": self.bucket_name,
                "upload_time_ms": upload_time,
                "download_time_ms": download_time,
                "total_time_ms": upload_time + download_time
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "bucket": self.bucket_name,
                "error": str(e)
            }

    def supports_storage_class(self, storage_class: StorageClass) -> bool:
        """
        Check if the provider supports a specific storage class.

        Args:
            storage_class: Storage class to check

        Returns:
            bool: True if storage class is supported
        """
        return storage_class in self.capabilities.supported_storage_classes

    def supports_access_control(self, access_control: AccessControl) -> bool:
        """
        Check if the provider supports a specific access control.

        Args:
            access_control: Access control to check

        Returns:
            bool: True if access control is supported
        """
        return access_control in self.capabilities.supported_access_controls

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the storage provider.

        Returns:
            Dict[str, Any]: Provider information
        """
        return {
            "provider": self.provider_name,
            "bucket": self.bucket_name,
            "region": self.region,
            "endpoint": self.endpoint,
            "capabilities": self.capabilities.dict(),
            "config": {
                "timeout": self.timeout,
                "max_retries": self.max_retries,
                "use_ssl": self.use_ssl,
                "max_concurrency": self.max_concurrency,
                "multipart_threshold": self.multipart_threshold,
                "multipart_chunksize": self.multipart_chunksize,
            }
        }


class StorageException(Exception):
    """Exception raised by storage services."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        bucket: str = None,
        key: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.provider = provider
        self.bucket = bucket
        self.key = key
        self.error_code = error_code


class ConnectionException(StorageException):
    """Exception raised when connection fails."""
    pass


class BucketException(StorageException):
    """Exception raised for bucket operations."""
    pass


class ObjectException(StorageException):
    """Exception raised for object operations."""
    pass


class UploadException(ObjectException):
    """Exception raised when upload fails."""
    pass


class DownloadException(ObjectException):
    """Exception raised when download fails."""
    pass


class ValidationException(StorageException):
    """Exception raised when validation fails."""
    pass


class PermissionException(StorageException):
    """Exception raised when permission is denied."""
    pass