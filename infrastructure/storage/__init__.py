"""
Infrastructure Storage Layer

This module provides storage abstraction interfaces and implementations
for different storage backends including MinIO, S3, and local file system.
"""

from .interfaces import (
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
    ValidationException,
    PermissionException
)

from .implementations import (
    MinIOClient,
    S3Client,
    LocalStorageClient
)

__all__ = [
    # Interfaces
    'StorageInterface',
    'StorageConfig',
    'StorageCapabilities',
    'StorageType',
    'AccessControl',
    'StorageClass',
    'StorageObject',
    'UploadResult',
    'DownloadResult',

    # Exceptions
    'StorageException',
    'ConnectionException',
    'BucketException',
    'ObjectException',
    'UploadException',
    'DownloadException',
    'ValidationException',
    'PermissionException',

    # Implementations
    'MinIOClient',
    'S3Client',
    'LocalStorageClient'
]