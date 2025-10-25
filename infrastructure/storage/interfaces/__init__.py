"""
Storage Interfaces

This module contains abstract interfaces for object storage services.
"""

from .storage_interface import (
    # Storage interfaces and models
    StorageInterface,
    StorageConfig,
    StorageCapabilities,
    StorageType,
    AccessControl,
    StorageClass,
    StorageObject,
    UploadResult,
    DownloadResult,

    # Storage exceptions
    StorageException,
    ConnectionException,
    BucketException,
    ObjectException,
    UploadException,
    DownloadException,
    ValidationException,
    PermissionException,
)

__all__ = [
    # Storage interfaces
    "StorageInterface",
    "StorageConfig",
    "StorageCapabilities",
    "StorageType",
    "AccessControl",
    "StorageClass",
    "StorageObject",
    "UploadResult",
    "DownloadResult",

    # Storage exceptions
    "StorageException",
    "ConnectionException",
    "BucketException",
    "ObjectException",
    "UploadException",
    "DownloadException",
    "ValidationException",
    "PermissionException",
]
