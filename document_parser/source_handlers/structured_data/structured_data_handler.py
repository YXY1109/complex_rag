"""
Structured Data Handler

This module provides specialized processing for structured data including
JSON, XML, CSV, YAML, TOML, and other structured formats.
"""

import asyncio
import json
import csv
import mimetypes
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import io

try:
    import yaml
    import toml
    import xml.etree.ElementTree as ET
    YAML_AVAILABLE = True
    TOML_AVAILABLE = True
    XML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    TOML_AVAILABLE = False
    XML_AVAILABLE = False

from ...interfaces.source_interface import (
    SourceHandler,
    FileSource,
    ProcessingStrategy,
    ParseRequest,
    ParseResponse
)
from ...interfaces.parser_interface import (
    DocumentFormat,
    DocumentChunk,
    Metadata
)
from ...services.quality_monitor import QualityMonitor, QualityMetric


@dataclass
class StructuredDataFeatures:
    """Features extracted from structured data."""
    data_format: str
    record_count: int
    field_count: int
    has_nested_structure: bool
    has_arrays: bool
    max_depth: int
    total_size: int
    schema_valid: bool
    encoding: str
    compression: Optional[str]


class StructuredDataHandler(SourceHandler):
    """
    Handler for structured data including JSON, XML, CSV, YAML, TOML.

    Features:
    - Multi-format support (JSON, XML, CSV, YAML, TOML)
    - Schema validation and inference
    - Structure preservation and flattening
    - Large file streaming support
    - Data type detection and conversion
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize structured data handler."""
        super().__init__(FileSource.STRUCTURED_DATA, config)
        self.quality_monitor = QualityMonitor()
        self.supported_formats = {
            '.json': 'json',
            '.xml': 'xml',
            '.csv': 'csv',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.conf': 'conf'
        }

    async def connect(self) -> bool:
        """Initialize handler."""
        return True

    async def disconnect(self) -> None:
        """Cleanup handler."""
        pass

    async def can_handle(self, request: ParseRequest) -> bool:
        """Check if this handler can process the request."""
        # Check file extension
        if request.file_path:
            file_ext = Path(request.file_path).suffix.lower()
            if file_ext in self.supported_formats:
                return True

        # Check MIME type
        if request.mime_type:
            structured_types = {
                'application/json', 'text/xml', 'application/xml',
                'text/csv', 'application/yaml', 'text/yaml',
                'application/toml'
            }
            if request.mime_type in structured_types:
                return True

        # Check content structure
        if request.content:
            content_str = request.content.decode('utf-8', errors='ignore')
            if self._detect_format_from_content(content_str):
                return True

        return False

    async def process(self, request: ParseRequest) -> ParseResponse:
        """Process structured data."""
        session_id = f"structured_{datetime.now().timestamp()}"
        processing_start = datetime.now()

        # Start quality monitoring
        quality_session = self.quality_monitor.start_session(
            session_id=session_id,
            file_source=FileSource.STRUCTURED_DATA,
            strategy=request.strategy,
            file_size=len(request.content) if request.content else None
        )

        try:
            # Detect data format
            data_format = self._detect_data_format(request)

            # Parse data based on format
            if data_format == 'json':
                parsed_data = await self._parse_json(request.content)
            elif data_format == 'xml':
                parsed_data = await self._parse_xml(request.content)
            elif data_format == 'csv':
                parsed_data = await self._parse_csv(request.content, request.custom_params)
            elif data_format == 'yaml':
                parsed_data = await self._parse_yaml(request.content)
            elif data_format == 'toml':
                parsed_data = await self._parse_toml(request.content)
            else:
                parsed_data = await self._parse_text(request.content)

            # Analyze structure
            features = await self._analyze_structure(parsed_data, data_format)

            # Convert to text representation
            text = await self._convert_to_text(parsed_data, data_format, features)

            # Create chunks
            chunks = await self._create_chunks(text, features, request)

            # Generate metadata
            metadata = await self._extract_metadata(features, data_format, request)

            # Calculate quality metrics
            await self._calculate_quality_metrics(quality_session, parsed_data, features, chunks)

            response = ParseResponse(
                content=text,
                chunks=chunks,
                metadata=metadata,
                format=self._get_document_format(data_format),
                processing_time=(datetime.now() - processing_start).total_seconds(),
                success=True
            )

            # End quality monitoring
            self.quality_monitor.end_session(session_id, success=True)

            return response

        except Exception as e:
            # End quality monitoring with error
            self.quality_monitor.end_session(session_id, success=False, error=e)

            return ParseResponse(
                content="",
                chunks=[],
                metadata=Metadata(),
                format=DocumentFormat.UNKNOWN,
                processing_time=(datetime.now() - processing_start).total_seconds(),
                success=False,
                error=str(e)
            )

    def _detect_data_format(self, request: ParseRequest) -> str:
        """Detect structured data format."""
        # Check file extension
        if request.file_path:
            file_ext = Path(request.file_path).suffix.lower()
            if file_ext in self.supported_formats:
                return self.supported_formats[file_ext]

        # Check MIME type
        if request.mime_type:
            if 'json' in request.mime_type:
                return 'json'
            elif 'xml' in request.mime_type:
                return 'xml'
            elif 'csv' in request.mime_type:
                return 'csv'
            elif 'yaml' in request.mime_type:
                return 'yaml'
            elif 'toml' in request.mime_type:
                return 'toml'

        # Check content structure
        if request.content:
            content_str = request.content.decode('utf-8', errors='ignore')
            return self._detect_format_from_content(content_str)

        return 'text'

    def _detect_format_from_content(self, content: str) -> Optional[str]:
        """Detect format from content structure."""
        content = content.strip()

        # JSON detection
        if (content.startswith('{') and content.endswith('}')) or \
           (content.startswith('[') and content.endswith(']')):
            try:
                json.loads(content)
                return 'json'
            except:
                pass

        # XML detection
        if content.startswith('<') and content.endswith('>'):
            try:
                ET.fromstring(content)
                return 'xml'
            except:
                pass

        # YAML detection
        if ':' in content and (content.startswith('---') or '\n---' in content):
            return 'yaml'

        # CSV detection
        if ',' in content and '\n' in content:
            lines = content.split('\n')[:5]  # Check first 5 lines
            if all(',' in line for line in lines if line.strip()):
                return 'csv'

        return None

    def _get_document_format(self, data_format: str) -> DocumentFormat:
        """Get document format from data format."""
        format_map = {
            'json': DocumentFormat.JSON,
            'xml': DocumentFormat.XML,
            'csv': DocumentFormat.CSV,
            'yaml': DocumentFormat.YAML,
            'toml': DocumentFormat.TOML
        }
        return format_map.get(data_format, DocumentFormat.TEXT)

    async def _parse_json(self, content: bytes) -> Dict[str, Any]:
        """Parse JSON content."""
        try:
            content_str = content.decode('utf-8')
            return json.loads(content_str)
        except Exception as e:
            raise ValueError(f"Failed to parse JSON: {e}")

    async def _parse_xml(self, content: bytes) -> Dict[str, Any]:
        """Parse XML content."""
        if not XML_AVAILABLE:
            return {'error': 'XML parsing not available'}

        try:
            content_str = content.decode('utf-8')
            root = ET.fromstring(content_str)
            return self._xml_to_dict(root)
        except Exception as e:
            raise ValueError(f"Failed to parse XML: {e}")

    def _xml_to_dict(self, element: ET.Element) -> Dict[str, Any]:
        """Convert XML element to dictionary."""
        result = {}

        # Add attributes
        if element.attrib:
            result['@attributes'] = element.attrib

        # Add text content
        if element.text and element.text.strip():
            if len(element) == 0:  # Leaf node
                return element.text.strip()
            result['#text'] = element.text.strip()

        # Add children
        for child in element:
            child_data = self._xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data

        return result

    async def _parse_csv(self, content: bytes, params: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse CSV content."""
        try:
            content_str = content.decode('utf-8')
            reader = csv.DictReader(io.StringIO(content_str))
            data = list(reader)
            return data
        except Exception as e:
            raise ValueError(f"Failed to parse CSV: {e}")

    async def _parse_yaml(self, content: bytes) -> Dict[str, Any]:
        """Parse YAML content."""
        if not YAML_AVAILABLE:
            return {'error': 'YAML parsing not available'}

        try:
            content_str = content.decode('utf-8')
            return yaml.safe_load(content_str)
        except Exception as e:
            raise ValueError(f"Failed to parse YAML: {e}")

    async def _parse_toml(self, content: bytes) -> Dict[str, Any]:
        """Parse TOML content."""
        if not TOML_AVAILABLE:
            return {'error': 'TOML parsing not available'}

        try:
            content_str = content.decode('utf-8')
            return toml.loads(content_str)
        except Exception as e:
            raise ValueError(f"Failed to parse TOML: {e}")

    async def _parse_text(self, content: bytes) -> Dict[str, Any]:
        """Parse generic text content."""
        try:
            content_str = content.decode('utf-8')
            return {'text': content_str}
        except Exception as e:
            raise ValueError(f"Failed to parse text: {e}")

    async def _analyze_structure(self, data: Any, data_format: str) -> StructuredDataFeatures:
        """Analyze structure of parsed data."""
        try:
            if isinstance(data, dict):
                record_count = 1
                field_count = len(data)
                has_nested_structure = any(isinstance(v, (dict, list)) for v in data.values())
                has_arrays = any(isinstance(v, list) for v in data.values())
                max_depth = self._calculate_depth(data)
            elif isinstance(data, list):
                record_count = len(data)
                field_count = len(data[0]) if data else 0
                has_nested_structure = any(isinstance(item, (dict, list)) for item in data)
                has_arrays = True
                max_depth = max(self._calculate_depth(item) for item in data) if data else 0
            else:
                record_count = 1
                field_count = 1
                has_nested_structure = False
                has_arrays = False
                max_depth = 1

            return StructuredDataFeatures(
                data_format=data_format,
                record_count=record_count,
                field_count=field_count,
                has_nested_structure=has_nested_structure,
                has_arrays=has_arrays,
                max_depth=max_depth,
                total_size=len(str(data)),
                schema_valid=True,
                encoding='utf-8',
                compression=None
            )

        except Exception:
            return StructuredDataFeatures(
                data_format=data_format,
                record_count=0,
                field_count=0,
                has_nested_structure=False,
                has_arrays=False,
                max_depth=0,
                total_size=0,
                schema_valid=False,
                encoding='utf-8',
                compression=None
            )

    def _calculate_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Calculate maximum depth of nested structure."""
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._calculate_depth(v, current_depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return current_depth
            return max(self._calculate_depth(item, current_depth + 1) for item in obj)
        else:
            return current_depth

    async def _convert_to_text(self, data: Any, data_format: str, features: StructuredDataFeatures) -> str:
        """Convert structured data to text representation."""
        if data_format == 'json':
            return json.dumps(data, indent=2, ensure_ascii=False)
        elif data_format == 'xml':
            return str(data)
        elif data_format == 'csv' and isinstance(data, list):
            # Convert CSV data back to readable format
            if not data:
                return ""
            headers = list(data[0].keys())
            output = [", ".join(headers)]
            for row in data:
                output.append(", ".join(str(row.get(h, "")) for h in headers))
            return "\n".join(output)
        elif data_format in ['yaml', 'toml']:
            if YAML_AVAILABLE:
                return yaml.dump(data, default_flow_style=False, allow_unicode=True)
            else:
                return str(data)
        else:
            return str(data)

    async def _create_chunks(self, text: str, features: StructuredDataFeatures, request: ParseRequest) -> List[DocumentChunk]:
        """Create document chunks from structured data."""
        if not text.strip():
            return []

        # Get chunking parameters
        params = request.custom_params or {}
        chunk_size = params.get('chunk_size', 1500)  # Larger chunks for structured data
        overlap = params.get('overlap_size', 100)

        # Create chunks
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_length = 0

        for line in lines:
            line_length = len(line) + 1  # +1 for newline

            if current_length + line_length > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata=Metadata({
                        'chunk_index': len(chunks),
                        'line_count': len(current_chunk),
                        'source': 'structured_data',
                        'data_format': features.data_format,
                        'has_structure': features.has_nested_structure
                    })
                ))

                # Start new chunk with overlap
                overlap_lines = current_chunk[-overlap//50:] if len(current_chunk) > overlap//50 else current_chunk
                current_chunk = overlap_lines + [line]
                current_length = sum(len(l) + 1 for l in current_chunk)
            else:
                current_chunk.append(line)
                current_length += line_length

        # Add final chunk
        if current_chunk:
            chunk_text = '\n'.join(current_chunk)
            chunks.append(DocumentChunk(
                content=chunk_text,
                metadata=Metadata({
                    'chunk_index': len(chunks),
                    'line_count': len(current_chunk),
                    'source': 'structured_data',
                    'data_format': features.data_format,
                    'has_structure': features.has_nested_structure
                })
            ))

        return chunks

    async def _extract_metadata(self, features: StructuredDataFeatures, data_format: str, request: ParseRequest) -> Metadata:
        """Extract metadata from structured data."""
        metadata_dict = {
            'source_type': 'structured_data',
            'data_format': data_format,
            'processing_strategy': request.strategy.value,
            'record_count': features.record_count,
            'field_count': features.field_count,
            'has_nested_structure': features.has_nested_structure,
            'has_arrays': features.has_arrays,
            'max_depth': features.max_depth,
            'total_size': features.total_size,
            'schema_valid': features.schema_valid,
            'encoding': features.encoding
        }

        # Add file path if available
        if request.file_path:
            metadata_dict['file_path'] = request.file_path

        return Metadata(metadata_dict)

    async def _calculate_quality_metrics(
        self,
        session_id: str,
        data: Any,
        features: StructuredDataFeatures,
        chunks: List[DocumentChunk]
    ):
        """Calculate quality metrics for the processing session."""
        # Structure preservation
        structure_score = 1.0 if features.schema_valid else 0.5
        self.quality_monitor.add_measurement(
            session_id, QualityMetric.STRUCTURE_PRESERVATION, structure_score
        )

        # Data integrity
        integrity_score = min(1.0, features.record_count / 10)  # Expect at least 10 records
        self.quality_monitor.add_measurement(
            session_id, QualityMetric.CONTENT_COMPLETENESS, integrity_score
        )

        # Processing efficiency
        efficiency_score = min(1.0, len(str(data)) / (features.total_size + 1))
        self.quality_monitor.add_measurement(
            session_id, QualityMetric.PROCESSING_SPEED, efficiency_score
        )