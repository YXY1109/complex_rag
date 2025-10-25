"""
Automatic File Source Detector

This module provides intelligent file source detection based on file characteristics,
URL patterns, content analysis, and metadata features.
"""

import asyncio
import re
import os
import mimetypes
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
from urllib.parse import urlparse
import magic
from dataclasses import dataclass

from ..interfaces.source_interface import (
    FileSource,
    ProcessingStrategy,
    SourceDetectionResult,
    SourceConfig
)


@dataclass
class SourceFeatures:
    """Extracted features for source detection."""
    file_extension: str
    mime_type: str
    url_pattern: str
    content_signature: Optional[str]
    metadata_features: Dict[str, Any]
    binary_indicators: Dict[str, bool]


class FileSourceDetector:
    """
    Intelligent file source detector with multiple detection strategies.

    Analyzes files based on:
    - File extensions and MIME types
    - URL patterns and domain characteristics
    - Content signatures and magic numbers
    - Metadata features and file characteristics
    """

    def __init__(self, config: Optional[SourceConfig] = None):
        """Initialize the detector with configuration."""
        self.config = config or SourceConfig()
        self._setup_detection_patterns()

    def _setup_detection_patterns(self):
        """Setup detection patterns for different sources."""

        # Web document patterns
        self.web_patterns = {
            'url_schemes': ['http', 'https', 'ftp'],
            'web_extensions': {'.html', '.htm', '.xhtml', '.md', '.txt'},
            'web_domains': ['github.com', 'gitlab.com', 'stackoverflow.com', 'medium.com'],
            'content_types': {'text/html', 'text/markdown', 'text/plain'},
        }

        # Office document patterns
        self.office_patterns = {
            'extensions': {
                '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
                '.odt', '.ods', '.odp', '.rtf', '.pages', '.numbers', '.key'
            },
            'content_types': {
                'application/pdf', 'application/msword', 'application/vnd.ms-',
                'application/vnd.openxmlformats-',
                'application/vnd.oasis.opendocument.'
            },
            'magic_signatures': [
                b'%PDF-',  # PDF
                b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1',  # Microsoft Office
                b'PK\x03\x04',  # ZIP-based (Office 2007+)
            ]
        }

        # Scanned document patterns
        self.scanned_patterns = {
            'image_extensions': {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.gif'},
            'image_content_types': {'image/jpeg', 'image/png', 'image/tiff', 'image/bmp'},
            'ocr_indicators': ['ocr', 'scan', 'digitized'],
            'text_density_threshold': 0.1,  # Low text density indicates scanned content
        }

        # Structured data patterns
        self.structured_patterns = {
            'extensions': {'.json', '.xml', '.csv', '.yaml', '.yml', '.toml', '.ini', '.conf'},
            'content_types': {
                'application/json', 'text/xml', 'application/xml',
                'text/csv', 'application/yaml', 'text/yaml'
            },
            'structure_indicators': ['{', '[', '<', '<?xml', ','],
        }

        # Code repository patterns
        self.code_patterns = {
            'extensions': {
                '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.hpp',
                '.go', '.rs', '.php', '.rb', '.swift', '.kt', '.scala',
                '.sh', '.bat', '.ps1', '.sql', '.html', '.css', '.scss',
                '.vue', '.jsx', '.tsx', '.md', '.rst', '.dockerfile'
            },
            'platform_patterns': [
                r'github\.com', r'gitlab\.com', r'bitbucket\.org',
                r'gist\.github\.com', r'raw\.githubusercontent\.com'
            ],
            'code_indicators': [
                'import ', 'from ', 'def ', 'function ', 'class ',
                'public ', 'private ', 'package ', '#include'
            ],
        }

    async def detect_source(
        self,
        file_path: Optional[str] = None,
        url: Optional[str] = None,
        content: Optional[bytes] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SourceDetectionResult:
        """
        Detect the file source using multiple analysis strategies.

        Args:
            file_path: Local file path
            url: Remote URL
            content: File content bytes
            metadata: Additional metadata

        Returns:
            SourceDetectionResult: Detection result with confidence
        """
        import time
        start_time = time.time()

        # Extract features
        features = await self._extract_features(file_path, url, content, metadata)

        # Analyze with different detectors
        scores = await self._analyze_features(features)

        # Determine best match
        best_source = max(scores.items(), key=lambda x: x[1])
        source_type = best_source[0]
        confidence = best_source[1]

        # Select processing strategy
        strategy = self._select_processing_strategy(source_type, features, confidence)

        # Prepare metadata
        result_metadata = {
            'features': {
                'file_extension': features.file_extension,
                'mime_type': features.mime_type,
                'url_pattern': features.url_pattern
            },
            'scores': scores,
            'all_scores': sorted(scores.items(), key=lambda x: x[1], reverse=True)
        }

        detection_time = (time.time() - start_time) * 1000

        return SourceDetectionResult(
            source=source_type,
            confidence=confidence,
            detected_features=list(scores.keys()),
            processing_strategy=strategy,
            metadata=result_metadata,
            detection_time_ms=detection_time
        )

    async def _extract_features(
        self,
        file_path: Optional[str],
        url: Optional[str],
        content: Optional[bytes],
        metadata: Optional[Dict[str, Any]]
    ) -> SourceFeatures:
        """Extract features from file for detection."""

        # Basic file information
        file_extension = ""
        mime_type = ""
        url_pattern = ""

        if file_path:
            file_extension = Path(file_path).suffix.lower()
            mime_type, _ = mimetypes.guess_type(file_path)

        if url:
            parsed_url = urlparse(url)
            url_pattern = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Content analysis
        content_signature = None
        if content:
            try:
                # Get MIME type from content
                content_mime = magic.from_buffer(content, mime=True)
                if content_mime:
                    mime_type = content_mime

                # Get magic signature
                content_signature = content[:16].hex() if len(content) >= 16 else content.hex()

            except Exception:
                # Fallback to basic content analysis
                if content:
                    # Try to detect if it's text
                    try:
                        content.decode('utf-8')
                        if not mime_type:
                            mime_type = 'text/plain'
                    except UnicodeDecodeError:
                        if not mime_type:
                            mime_type = 'application/octet-stream'

        # Metadata features
        metadata_features = metadata or {}

        # Binary indicators
        binary_indicators = {
            'has_extension': bool(file_extension),
            'has_url': bool(url),
            'has_content': bool(content),
            'is_binary': self._is_binary_content(content) if content else False,
            'has_metadata': bool(metadata_features)
        }

        return SourceFeatures(
            file_extension=file_extension,
            mime_type=mime_type or "application/octet-stream",
            url_pattern=url_pattern,
            content_signature=content_signature,
            metadata_features=metadata_features,
            binary_indicators=binary_indicators
        )

    async def _analyze_features(self, features: SourceFeatures) -> Dict[FileSource, float]:
        """Analyze features and score each source type."""

        scores = {
            FileSource.WEB_DOCUMENTS: 0.0,
            FileSource.OFFICE_DOCUMENTS: 0.0,
            FileSource.SCANNED_DOCUMENTS: 0.0,
            FileSource.STRUCTURED_DATA: 0.0,
            FileSource.CODE_REPOSITORIES: 0.0,
            FileSource.CUSTOM_SOURCES: 0.0,
            FileSource.UNKNOWN: 0.1  # Base confidence for unknown
        }

        # Web documents detection
        web_score = self._score_web_documents(features)
        scores[FileSource.WEB_DOCUMENTS] = web_score

        # Office documents detection
        office_score = self._score_office_documents(features)
        scores[FileSource.OFFICE_DOCUMENTS] = office_score

        # Scanned documents detection
        scanned_score = self._score_scanned_documents(features)
        scores[FileSource.SCANNED_DOCUMENTS] = scanned_score

        # Structured data detection
        structured_score = self._score_structured_data(features)
        scores[FileSource.STRUCTURED_DATA] = structured_score

        # Code repositories detection
        code_score = self._score_code_repositories(features)
        scores[FileSource.CODE_REPOSITORIES] = code_score

        return scores

    def _score_web_documents(self, features: SourceFeatures) -> float:
        """Score web documents detection."""
        score = 0.0

        # URL patterns
        if features.url_pattern:
            if any(domain in features.url_pattern for domain in self.web_patterns['web_domains']):
                score += 0.6
            elif any(features.url_pattern.startswith(scheme) for scheme in self.web_patterns['url_schemes']):
                score += 0.3

        # File extensions
        if features.file_extension in self.web_patterns['web_extensions']:
            score += 0.4

        # MIME types
        if features.mime_type in self.web_patterns['content_types']:
            score += 0.4

        # URL-based indicators
        if 'url' in features.metadata_features:
            score += 0.2

        return min(score, 1.0)

    def _score_office_documents(self, features: SourceFeatures) -> float:
        """Score office documents detection."""
        score = 0.0

        # File extensions
        if features.file_extension in self.office_patterns['extensions']:
            score += 0.5

        # MIME types
        if any(features.mime_type.startswith(prefix) for prefix in ['application/pdf', 'application/msword', 'application/vnd.']):
            score += 0.5

        # Magic signatures
        if features.content_signature:
            for signature in self.office_patterns['magic_signatures']:
                if features.content_signature.startswith(signature.hex()):
                    score += 0.6
                    break

        return min(score, 1.0)

    def _score_scanned_documents(self, features: SourceFeatures) -> float:
        """Score scanned documents detection."""
        score = 0.0

        # Image file indicators
        if features.file_extension in self.scanned_patterns['image_extensions']:
            score += 0.6

        if features.mime_type in self.scanned_patterns['image_content_types']:
            score += 0.6

        # Metadata indicators
        metadata_text = str(features.metadata_features).lower()
        for indicator in self.scanned_patterns['ocr_indicators']:
            if indicator in metadata_text:
                score += 0.3

        return min(score, 1.0)

    def _score_structured_data(self, features: SourceFeatures) -> float:
        """Score structured data detection."""
        score = 0.0

        # File extensions
        if features.file_extension in self.structured_patterns['extensions']:
            score += 0.5

        # MIME types
        if features.mime_type in self.structured_patterns['content_types']:
            score += 0.5

        # Structure indicators in metadata
        metadata_text = str(features.metadata_features)
        for indicator in self.structured_patterns['structure_indicators']:
            if indicator in metadata_text:
                score += 0.2

        return min(score, 1.0)

    def _score_code_repositories(self, features: SourceFeatures) -> float:
        """Score code repositories detection."""
        score = 0.0

        # File extensions
        if features.file_extension in self.code_patterns['extensions']:
            score += 0.4

        # URL patterns
        if features.url_pattern:
            for pattern in self.code_patterns['platform_patterns']:
                if re.search(pattern, features.url_pattern):
                    score += 0.7
                    break

        # Code indicators in metadata
        metadata_text = str(features.metadata_features).lower()
        for indicator in self.code_patterns['code_indicators']:
            if indicator in metadata_text:
                score += 0.2

        return min(score, 1.0)

    def _select_processing_strategy(
        self,
        source_type: FileSource,
        features: SourceFeatures,
        confidence: float
    ) -> ProcessingStrategy:
        """Select processing strategy based on source and confidence."""

        if confidence < 0.3:
            return ProcessingStrategy.ACCURATE  # Be more careful with low confidence
        elif source_type == FileSource.CODE_REPOSITORIES:
            return ProcessingStrategy.FAST  # Code files usually don't need complex processing
        elif source_type == FileSource.OFFICE_DOCUMENTS:
            return ProcessingStrategy.BALANCED  # Balanced approach for office docs
        elif source_type == FileSource.SCANNED_DOCUMENTS:
            return ProcessingStrategy.ACCURATE  # Use accurate for OCR-heavy content
        else:
            return ProcessingStrategy.AUTO

    def _is_binary_content(self, content: bytes) -> bool:
        """Check if content is binary."""
        if not content:
            return False

        # Check for null bytes (common in binary files)
        if b'\x00' in content[:1024]:
            return True

        # Try to decode as UTF-8
        try:
            content[:1024].decode('utf-8')
            return False
        except UnicodeDecodeError:
            return True

    async def batch_detect_sources(
        self,
        files: List[Tuple[Optional[str], Optional[str], Optional[bytes], Optional[Dict[str, Any]]]]
    ) -> List[SourceDetectionResult]:
        """Detect sources for multiple files concurrently."""

        tasks = [
            self.detect_source(file_path, url, content, metadata)
            for file_path, url, content, metadata in files
        ]

        return await asyncio.gather(*tasks)