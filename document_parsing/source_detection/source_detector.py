"""
文件来源自动检测器

此模块提供自动检测文档来源和类型的功能，
基于文件内容、URL模式、元数据等多维度分析。
"""

import os
import re
import mimetypes
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Union
from urllib.parse import urlparse, parse_qs
from pathlib import Path
import magic
from dataclasses import dataclass
from enum import Enum

from ..interfaces.source_processor_interface import DocumentSource, DocumentReference


class ConfidenceLevel(Enum):
    """置信度级别。"""
    HIGH = "high"      # 0.8-1.0
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # 0.2-0.5
    UNKNOWN = "unknown"  # 0.0-0.2


@dataclass
class SourceDetectionResult:
    """来源检测结果。"""
    source_type: DocumentSource
    confidence: float
    confidence_level: ConfidenceLevel
    metadata: Dict[str, Any]
    suggested_processor: Optional[str] = None
    suggested_strategies: List[str] = None
    detection_methods: List[str] = None


class URLPatternDetector:
    """URL模式检测器。"""

    def __init__(self):
        """初始化URL模式检测器。"""
        self.patterns = {
            DocumentSource.WEB_DOCUMENTS: [
                r'^https?://(?:www\.)?[^/]+\.(?:html?|htm|php|asp|aspx|jsp)',
                r'^https?://(?:www\.)?github\.com/[^/]+/[^/]+',
                r'^https?://(?:www\.)?gitlab\.com/[^/]+/[^/]+',
                r'^https?://(?:www\.)?bitbucket\.org/[^/]+/[^/]+',
                r'^https?://(?:www\.)?medium\.com/',
                r'^https?://(?:www\.)?dev\.to/',
                r'^https?://(?:www\.)?stackoverflow\.com/',
                r'^https?://(?:docs\.)?[^/]+/[^/]+',  # 文档网站
            ],
            DocumentSource.CODE_REPOSITORIES: [
                r'^https?://(?:www\.)?github\.com/[^/]+/[^/]+',
                r'^https?://(?:www\.)?gitlab\.com/[^/]+/[^/]+',
                r'^https?://(?:www\.)?bitbucket\.org/[^/]+/[^/]+',
                r'^https?://(?:www\.)?gitee\.com/[^/]+/[^/]+',
                r'^https?://(?:www\.)?coding\.net/[^/]+/[^/]+',
            ],
            DocumentSource.REMOTE_STORAGE: [
                r'^https?://[^/]+\.s3[.-][^/]+\.[^/]+/',
                r'^https?://[^/]+\.oss-[^-]+\.aliyuncs\.com/',
                r'^https?://[^/]+\.cos\.[^-]+\.myqcloud\.com/',
                r'^https?://[^/]+\.storage\.googleapis\.com/',
                r'^https?://[^/]+\.blob\.core\.windows\.net/',
            ],
            DocumentSource.DATABASE: [
                r'^https?://(?:docs\.)?[^/]*(?:api|swagger|openapi)[^/]*/',
                r'^https?://(?:www\.)?[^/]*(?:graphql|graphiql)[^/]*/',
            ]
        }

    def detect_from_url(self, url: str) -> SourceDetectionResult:
        """
        从URL检测来源类型。

        Args:
            url: URL地址

        Returns:
            SourceDetectionResult: 检测结果
        """
        if not url:
            return SourceDetectionResult(
                source_type=DocumentSource.LOCAL_FILES,
                confidence=0.0,
                confidence_level=ConfidenceLevel.UNKNOWN,
                metadata={"error": "Empty URL"},
                detection_methods=["url_pattern"]
            )

        # 解析URL
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        path = parsed.path.lower()
        query_params = parse_qs(parsed.query)

        metadata = {
            "url": url,
            "domain": domain,
            "path": path,
            "query_params": dict(query_params)
        }

        # 检查每个来源类型的模式
        best_match = None
        best_confidence = 0.0

        for source_type, patterns in self.patterns.items():
            for pattern in patterns:
                if re.match(pattern, url, re.IGNORECASE):
                    confidence = self._calculate_url_confidence(url, domain, path, source_type)
                    if confidence > best_confidence:
                        best_match = source_type
                        best_confidence = confidence

        if best_match:
            confidence_level = self._get_confidence_level(best_confidence)
            metadata.update({
                "pattern_matched": True,
                "url_length": len(url),
                "has_query": len(query_params) > 0,
                "is_secure": url.startswith('https://')
            })

            return SourceDetectionResult(
                source_type=best_match,
                confidence=best_confidence,
                confidence_level=confidence_level,
                metadata=metadata,
                suggested_processor=self._get_suggested_processor(best_match),
                suggested_strategies=self._get_suggested_strategies(best_match),
                detection_methods=["url_pattern"]
            )
        else:
            # 默认为网页文档
            return SourceDetectionResult(
                source_type=DocumentSource.WEB_DOCUMENTS,
                confidence=0.3,
                confidence_level=ConfidenceLevel.LOW,
                metadata=metadata,
                suggested_processor="web_processor",
                suggested_strategies=["extract_text", "preserve_layout"],
                detection_methods=["url_pattern"]
            )

    def _calculate_url_confidence(self, url: str, domain: str, path: str, source_type: DocumentSource) -> float:
        """计算URL置信度。"""
        base_confidence = 0.6

        # 根据域名增加置信度
        if source_type == DocumentSource.CODE_REPOSITORIES:
            if any(repo in domain for repo in ['github.com', 'gitlab.com', 'bitbucket.org']):
                base_confidence += 0.3
            if '/blob/' in url or '/tree/' in url:
                base_confidence += 0.1

        elif source_type == DocumentSource.REMOTE_STORAGE:
            if any(storage in domain for storage in ['s3', 'oss', 'cos', 'storage', 'blob']):
                base_confidence += 0.3
            if any(region in domain for region in ['amazonaws.com', 'aliyuncs.com', 'myqcloud.com']):
                base_confidence += 0.1

        elif source_type == DocumentSource.WEB_DOCUMENTS:
            if any(doc in domain for doc in ['docs.', 'documentation.', 'dev.', 'api.']):
                base_confidence += 0.2
            if path.endswith(('.html', '.htm', '.md', '.txt')):
                base_confidence += 0.2

        return min(base_confidence, 1.0)

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """获取置信度级别。"""
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN

    def _get_suggested_processor(self, source_type: DocumentSource) -> str:
        """获取建议的处理器。"""
        processor_map = {
            DocumentSource.WEB_DOCUMENTS: "web_processor",
            DocumentSource.CODE_REPOSITORIES: "code_processor",
            DocumentSource.REMOTE_STORAGE: "storage_processor",
            DocumentSource.DATABASE: "api_processor",
            DocumentSource.OFFICE_DOCUMENTS: "office_processor",
            DocumentSource.SCANNED_DOCUMENTS: "ocr_processor",
            DocumentSource.STRUCTURED_DATA: "structured_processor"
        }
        return processor_map.get(source_type, "generic_processor")

    def _get_suggested_strategies(self, source_type: DocumentSource) -> List[str]:
        """获取建议的处理策略。"""
        strategy_map = {
            DocumentSource.WEB_DOCUMENTS: ["extract_text", "preserve_layout"],
            DocumentSource.CODE_REPOSITORIES: ["code_extraction", "extract_text"],
            DocumentSource.REMOTE_STORAGE: ["extract_text", "multimodal_analysis"],
            DocumentSource.OFFICE_DOCUMENTS: ["preserve_layout", "extract_text"],
            DocumentSource.SCANNED_DOCUMENTS: ["multimodal_analysis", "image_analysis"],
            DocumentSource.STRUCTURED_DATA: ["structured_data", "extract_text"]
        }
        return strategy_map.get(source_type, ["extract_text"])


class FileTypeDetector:
    """文件类型检测器。"""

    def __init__(self):
        """初始化文件类型检测器。"""
        self.file_type_mapping = {
            # 办公文档
            DocumentSource.OFFICE_DOCUMENTS: [
                'application/pdf',
                'application/msword',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                'application/vnd.ms-excel',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                'application/vnd.ms-powerpoint',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                'application/vnd.ms-works',
                'application/rtf'
            ],
            # 结构化数据
            DocumentSource.STRUCTURED_DATA: [
                'application/json',
                'application/xml',
                'text/xml',
                'text/csv',
                'application/vnd.ms-excel',  # CSV通常保存为Excel格式
                'text/yaml',
                'application/x-yaml',
                'application/vnd.apache.parquet',
                'application/vnd.sqlite3'
            ],
            # 网页文档
            DocumentSource.WEB_DOCUMENTS: [
                'text/html',
                'text/plain',
                'text/markdown',
                'text/x-markdown',
                'application/rtf'  # RTF也可以是网页文档
            ],
            # 扫描文档
            DocumentSource.SCANNED_DOCUMENTS: [
                'image/jpeg',
                'image/png',
                'image/tiff',
                'image/bmp',
                'image/gif',
                'image/webp'
            ]
        }

    def detect_from_file(self, file_path: str, file_content: Optional[bytes] = None) -> SourceDetectionResult:
        """
        从文件检测来源类型。

        Args:
            file_path: 文件路径
            file_content: 文件内容（可选）

        Returns:
            SourceDetectionResult: 检测结果
        """
        if not os.path.exists(file_path) and not file_content:
            return SourceDetectionResult(
                source_type=DocumentSource.LOCAL_FILES,
                confidence=0.0,
                confidence_level=ConfidenceLevel.UNKNOWN,
                metadata={"error": "File not found and no content provided"},
                detection_methods=["file_type"]
            )

        try:
            # 获取文件信息
            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else len(file_content or b"")
            file_ext = os.path.splitext(file_path)[1].lower()

            # 检测MIME类型
            mime_type = None
            if file_content:
                mime_type = magic.from_buffer(file_content, mime=True)
            else:
                mime_type = magic.from_file(file_path, mime=True)

            # 检测文件签名
            signature_type = self._detect_file_signature(file_content) if file_content else None

            metadata = {
                "file_path": file_path,
                "file_size": file_size,
                "file_extension": file_ext,
                "mime_type": mime_type,
                "signature_type": signature_type
            }

            # 匹配文件类型
            best_match = None
            best_confidence = 0.0

            for source_type, mime_types in self.file_type_mapping.items():
                if mime_type in mime_types:
                    confidence = self._calculate_file_confidence(
                        file_path, file_ext, mime_type, signature_type, source_type
                    )
                    if confidence > best_confidence:
                        best_match = source_type
                        best_confidence = confidence

            # 如果没有匹配，使用文件扩展名作为后备
            if best_match is None:
                best_match, best_confidence = self._detect_from_extension(file_ext)

            confidence_level = self._get_confidence_level(best_confidence)

            return SourceDetectionResult(
                source_type=best_match,
                confidence=best_confidence,
                confidence_level=confidence_level,
                metadata=metadata,
                suggested_processor=self._get_suggested_processor(best_match),
                suggested_strategies=self._get_suggested_strategies(best_match),
                detection_methods=["file_type"]
            )

        except Exception as e:
            return SourceDetectionResult(
                source_type=DocumentSource.LOCAL_FILES,
                confidence=0.0,
                confidence_level=ConfidenceLevel.UNKNOWN,
                metadata={"error": str(e)},
                detection_methods=["file_type"]
            )

    def _detect_file_signature(self, file_content: bytes) -> Optional[str]:
        """检测文件签名。"""
        if not file_content or len(file_content) < 8:
            return None

        # 常见文件签名
        signatures = {
            b'\x25\x50\x44\x46': 'PDF',
            b'\xD0\xCF\x11\xE0': 'Microsoft Office',
            b'\x50\x4B\x03\x04': 'ZIP (Office 2007+)',
            b'\x89\x50\x4E\x47': 'PNG',
            b'\xFF\xD8\xFF': 'JPEG',
            b'\x47\x49\x46\x38': 'GIF',
            b'\x49\x49\x2A\x00': 'TIFF (little endian)',
            b'\x4D\x4D\x00\x2A': 'TIFF (big endian)',
            b'\x3C\x21\x44\x4F': 'HTML/SGML',
            b'\x3C\x3F\x78\x6D': 'XML',
            b'\x7B\x0A\x20\x20': 'JSON',
        }

        for signature, file_type in signatures.items():
            if file_content.startswith(signature):
                return file_type

        return None

    def _calculate_file_confidence(
        self,
        file_path: str,
        file_ext: str,
        mime_type: str,
        signature_type: Optional[str],
        source_type: DocumentSource
    ) -> float:
        """计算文件置信度。"""
        confidence = 0.0

        # MIME类型匹配
        if source_type in self.file_type_mapping:
            if mime_type in self.file_type_mapping[source_type]:
                confidence += 0.6

        # 文件签名匹配
        signature_mapping = {
            'PDF': DocumentSource.OFFICE_DOCUMENTS,
            'Microsoft Office': DocumentSource.OFFICE_DOCUMENTS,
            'ZIP (Office 2007+)': DocumentSource.OFFICE_DOCUMENTS,
            'PNG': DocumentSource.SCANNED_DOCUMENTS,
            'JPEG': DocumentSource.SCANNED_DOCUMENTS,
            'GIF': DocumentSource.SCANNED_DOCUMENTS,
            'TIFF': DocumentSource.SCANNED_DOCUMENTS,
            'HTML/SGML': DocumentSource.WEB_DOCUMENTS,
            'XML': DocumentSource.STRUCTURED_DATA,
            'JSON': DocumentSource.STRUCTURED_DATA,
        }

        if signature_type and signature_mapping.get(signature_type) == source_type:
            confidence += 0.3

        # 文件扩展名匹配
        ext_mapping = {
            '.pdf': DocumentSource.OFFICE_DOCUMENTS,
            '.doc': DocumentSource.OFFICE_DOCUMENTS,
            '.docx': DocumentSource.OFFICE_DOCUMENTS,
            '.xls': DocumentSource.OFFICE_DOCUMENTS,
            '.xlsx': DocumentSource.OFFICE_DOCUMENTS,
            '.ppt': DocumentSource.OFFICE_DOCUMENTS,
            '.pptx': DocumentSource.OFFICE_DOCUMENTS,
            '.html': DocumentSource.WEB_DOCUMENTS,
            '.htm': DocumentSource.WEB_DOCUMENTS,
            '.md': DocumentSource.WEB_DOCUMENTS,
            '.txt': DocumentSource.WEB_DOCUMENTS,
            '.json': DocumentSource.STRUCTURED_DATA,
            '.xml': DocumentSource.STRUCTURED_DATA,
            '.csv': DocumentSource.STRUCTURED_DATA,
            '.yaml': DocumentSource.STRUCTURED_DATA,
            '.yml': DocumentSource.STRUCTURED_DATA,
            '.jpg': DocumentSource.SCANNED_DOCUMENTS,
            '.jpeg': DocumentSource.SCANNED_DOCUMENTS,
            '.png': DocumentSource.SCANNED_DOCUMENTS,
            '.tiff': DocumentSource.SCANNED_DOCUMENTS,
            '.tif': DocumentSource.SCANNED_DOCUMENTS,
        }

        if ext_mapping.get(file_ext) == source_type:
            confidence += 0.1

        return min(confidence, 1.0)

    def _detect_from_extension(self, file_ext: str) -> Tuple[DocumentSource, float]:
        """从文件扩展名检测。"""
        ext_mapping = {
            '.pdf': (DocumentSource.OFFICE_DOCUMENTS, 0.8),
            '.doc': (DocumentSource.OFFICE_DOCUMENTS, 0.8),
            '.docx': (DocumentSource.OFFICE_DOCUMENTS, 0.9),
            '.xls': (DocumentSource.OFFICE_DOCUMENTS, 0.8),
            '.xlsx': (DocumentSource.OFFICE_DOCUMENTS, 0.9),
            '.ppt': (DocumentSource.OFFICE_DOCUMENTS, 0.8),
            '.pptx': (DocumentSource.OFFICE_DOCUMENTS, 0.9),
            '.html': (DocumentSource.WEB_DOCUMENTS, 0.8),
            '.htm': (DocumentSource.WEB_DOCUMENTS, 0.8),
            '.md': (DocumentSource.WEB_DOCUMENTS, 0.7),
            '.txt': (DocumentSource.WEB_DOCUMENTS, 0.5),
            '.json': (DocumentSource.STRUCTURED_DATA, 0.9),
            '.xml': (DocumentSource.STRUCTURED_DATA, 0.8),
            '.csv': (DocumentSource.STRUCTURED_DATA, 0.8),
            '.yaml': (DocumentSource.STRUCTURED_DATA, 0.7),
            '.yml': (DocumentSource.STRUCTURED_DATA, 0.7),
            '.jpg': (DocumentSource.SCANNED_DOCUMENTS, 0.9),
            '.jpeg': (DocumentSource.SCANNED_DOCUMENTS, 0.9),
            '.png': (DocumentSource.SCANNED_DOCUMENTS, 0.9),
            '.tiff': (DocumentSource.SCANNED_DOCUMENTS, 0.8),
            '.tif': (DocumentSource.SCANNED_DOCUMENTS, 0.8),
        }

        return ext_mapping.get(file_ext, (DocumentSource.LOCAL_FILES, 0.3))

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """获取置信度级别。"""
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN

    def _get_suggested_processor(self, source_type: DocumentSource) -> str:
        """获取建议的处理器。"""
        processor_map = {
            DocumentSource.OFFICE_DOCUMENTS: "office_processor",
            DocumentSource.WEB_DOCUMENTS: "web_processor",
            DocumentSource.STRUCTURED_DATA: "structured_processor",
            DocumentSource.SCANNED_DOCUMENTS: "ocr_processor",
            DocumentSource.CODE_REPOSITORIES: "code_processor",
            DocumentSource.LOCAL_FILES: "file_processor"
        }
        return processor_map.get(source_type, "generic_processor")

    def _get_suggested_strategies(self, source_type: DocumentSource) -> List[str]:
        """获取建议的处理策略。"""
        strategy_map = {
            DocumentSource.OFFICE_DOCUMENTS: ["preserve_layout", "extract_text", "table_extraction"],
            DocumentSource.WEB_DOCUMENTS: ["extract_text", "preserve_layout"],
            DocumentSource.STRUCTURED_DATA: ["structured_data", "extract_text"],
            DocumentSource.SCANNED_DOCUMENTS: ["multimodal_analysis", "image_analysis"],
            DocumentSource.CODE_REPOSITORIES: ["code_extraction", "extract_text"],
            DocumentSource.LOCAL_FILES: ["extract_text"]
        }
        return strategy_map.get(source_type, ["extract_text"])


class ContentAnalyzer:
    """内容分析器。"""

    def __init__(self):
        """初始化内容分析器。"""
        self.indicators = {
            DocumentSource.CODE_REPOSITORIES: [
                r'import\s+\w+',
                r'function\s+\w+\s*\(',
                r'class\s+\w+',
                r'def\s+\w+\s*\(',
                r'public\s+class\s+\w+',
                r'private\s+\w+\s+\w+',
                r'#include\s*[<"]',
                r'package\s+\w+',
                r'namespace\s+\w+',
                r'interface\s+\w+',
            ],
            DocumentSource.STRUCTURED_DATA: [
                r'\{\s*"[^"]+"\s*:',
                r'<\?xml\s+version',
                r'<\w+[^>]*>',
                r'^[^,]+,',
                r'^\w+\s*:\s*',
            ],
            DocumentSource.WEB_DOCUMENTS: [
                r'<[^>]+>',
                r'https?://[^\s]+',
                r'\[([^\]]+)\]\([^\)]+\)',
                r'#{1,6}\s+',
                r'\*\*[^*]+\*\*',
            ],
            DocumentSource.OFFICE_DOCUMENTS: [
                r'\f',  # 分页符
                r'Page\s+\d+',
                r'Table\s+\d+',
                r'Figure\s+\d+',
            ]
        }

    def analyze_content(self, content: str) -> SourceDetectionResult:
        """
        分析内容来检测来源类型。

        Args:
            content: 文档内容

        Returns:
            SourceDetectionResult: 检测结果
        """
        if not content:
            return SourceDetectionResult(
                source_type=DocumentSource.LOCAL_FILES,
                confidence=0.0,
                confidence_level=ConfidenceLevel.UNKNOWN,
                metadata={"error": "Empty content"},
                detection_methods=["content_analysis"]
            )

        try:
            content_sample = content[:5000] if len(content) > 5000 else content
            content_length = len(content)
            line_count = len(content.split('\n'))

            metadata = {
                "content_length": content_length,
                "line_count": line_count,
                "sample_size": len(content_sample)
            }

            # 分析每个来源类型的指示器
            best_match = None
            best_confidence = 0.0
            match_details = {}

            for source_type, patterns in self.indicators.items():
                matches = 0
                total_patterns = len(patterns)

                for pattern in patterns:
                    if re.search(pattern, content_sample, re.MULTILINE | re.IGNORECASE):
                        matches += 1

                if total_patterns > 0:
                    match_ratio = matches / total_patterns
                    confidence = match_ratio * 0.7  # 基础置信度

                    # 额外的内容特征分析
                    if source_type == DocumentSource.CODE_REPOSITORIES:
                        # 代码特征
                        code_features = self._analyze_code_features(content_sample)
                        confidence += code_features * 0.3
                        metadata["code_features"] = code_features

                    elif source_type == DocumentSource.STRUCTURED_DATA:
                        # 结构化数据特征
                        data_features = self._analyze_structured_features(content_sample)
                        confidence += data_features * 0.3
                        metadata["structured_features"] = data_features

                    elif source_type == DocumentSource.WEB_DOCUMENTS:
                        # 网页特征
                        web_features = self._analyze_web_features(content_sample)
                        confidence += web_features * 0.3
                        metadata["web_features"] = web_features

                    match_details[source_type.value] = {
                        "matches": matches,
                        "total_patterns": total_patterns,
                        "match_ratio": match_ratio,
                        "confidence": confidence
                    }

                    if confidence > best_confidence:
                        best_match = source_type
                        best_confidence = confidence

            metadata["match_details"] = match_details
            confidence_level = self._get_confidence_level(best_confidence)

            return SourceDetectionResult(
                source_type=best_match or DocumentSource.LOCAL_FILES,
                confidence=best_confidence,
                confidence_level=confidence_level,
                metadata=metadata,
                suggested_processor=self._get_suggested_processor(best_match) if best_match else "generic_processor",
                suggested_strategies=self._get_suggested_strategies(best_match) if best_match else ["extract_text"],
                detection_methods=["content_analysis"]
            )

        except Exception as e:
            return SourceDetectionResult(
                source_type=DocumentSource.LOCAL_FILES,
                confidence=0.0,
                confidence_level=ConfidenceLevel.UNKNOWN,
                metadata={"error": str(e)},
                detection_methods=["content_analysis"]
            )

    def _analyze_code_features(self, content: str) -> float:
        """分析代码特征。"""
        features = 0.0
        lines = content.split('\n')

        # 检查注释
        comment_lines = sum(1 for line in lines if line.strip().startswith(('//', '#', '/*', '*')))
        if comment_lines / len(lines) > 0.1:
            features += 0.2

        # 检查缩进
        indented_lines = sum(1 for line in lines if line.startswith(('    ', '\t')))
        if indented_lines / len(lines) > 0.3:
            features += 0.2

        # 检查代码关键字
        code_keywords = ['function', 'class', 'def', 'import', 'var', 'let', 'const', 'if', 'else', 'for', 'while']
        keyword_count = sum(content.lower().count(keyword) for keyword in code_keywords)
        if keyword_count > 5:
            features += 0.3

        # 检查括号平衡
        open_brackets = content.count('{') + content.count('(') + content.count('[')
        close_brackets = content.count('}') + content.count(')') + content.count(']')
        if abs(open_brackets - close_brackets) / max(open_brackets, 1) < 0.1:
            features += 0.3

        return min(features, 1.0)

    def _analyze_structured_features(self, content: str) -> float:
        """分析结构化数据特征。"""
        features = 0.0

        # JSON特征
        if content.strip().startswith('{') and content.strip().endswith('}'):
            features += 0.5
        if '"' in content and ':' in content:
            features += 0.2

        # XML特征
        if content.strip().startswith('<') and content.strip().endswith('>'):
            features += 0.5
        if '<' in content and '>' in content and '/' in content:
            features += 0.2

        # CSV特征
        lines = content.split('\n')
        if len(lines) > 1:
            first_line_commas = lines[0].count(',')
            second_line_commas = lines[1].count(',') if len(lines) > 1 else 0
            if abs(first_line_commas - second_line_commas) <= 1 and first_line_commas > 0:
                features += 0.4

        # YAML特征
        if ':' in content and any(line.strip().startswith('  ') for line in content.split('\n')):
            features += 0.3

        return min(features, 1.0)

    def _analyze_web_features(self, content: str) -> float:
        """分析网页特征。"""
        features = 0.0

        # HTML特征
        if '<html' in content.lower() or '<!DOCTYPE' in content.upper():
            features += 0.5
        if content.count('<') > 5 and content.count('>') > 5:
            features += 0.2

        # Markdown特征
        if '#' in content and '*' in content:
            features += 0.3
        if '[' in content and '](' in content:
            features += 0.3

        # URL特征
        url_count = len(re.findall(r'https?://[^\s]+', content))
        if url_count > 0:
            features += min(url_count * 0.1, 0.3)

        return min(features, 1.0)

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """获取置信度级别。"""
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN

    def _get_suggested_processor(self, source_type: DocumentSource) -> str:
        """获取建议的处理器。"""
        processor_map = {
            DocumentSource.CODE_REPOSITORIES: "code_processor",
            DocumentSource.STRUCTURED_DATA: "structured_processor",
            DocumentSource.WEB_DOCUMENTS: "web_processor",
            DocumentSource.OFFICE_DOCUMENTS: "office_processor"
        }
        return processor_map.get(source_type, "generic_processor")

    def _get_suggested_strategies(self, source_type: DocumentSource) -> List[str]:
        """获取建议的处理策略。"""
        strategy_map = {
            DocumentSource.CODE_REPOSITORIES: ["code_extraction", "extract_text"],
            DocumentSource.STRUCTURED_DATA: ["structured_data", "extract_text"],
            DocumentSource.WEB_DOCUMENTS: ["extract_text", "preserve_layout"],
            DocumentSource.OFFICE_DOCUMENTS: ["extract_text", "table_extraction"]
        }
        return strategy_map.get(source_type, ["extract_text"])


class SourceDetector:
    """
    综合来源检测器。

    结合URL模式、文件类型和内容分析来检测文档来源。
    """

    def __init__(self):
        """初始化综合来源检测器。"""
        self.url_detector = URLPatternDetector()
        self.file_detector = FileTypeDetector()
        self.content_analyzer = ContentAnalyzer()

    async def detect_source(
        self,
        url: Optional[str] = None,
        file_path: Optional[str] = None,
        file_content: Optional[bytes] = None,
        text_content: Optional[str] = None
    ) -> SourceDetectionResult:
        """
        检测文档来源。

        Args:
            url: URL地址（可选）
            file_path: 文件路径（可选）
            file_content: 文件内容（可选）
            text_content: 文本内容（可选）

        Returns:
            SourceDetectionResult: 综合检测结果
        """
        results = []
        detection_methods = []

        # URL检测
        if url:
            url_result = self.url_detector.detect_from_url(url)
            results.append(url_result)
            detection_methods.extend(url_result.detection_methods)

        # 文件类型检测
        if file_path or file_content:
            file_result = self.file_detector.detect_from_file(file_path or "", file_content)
            results.append(file_result)
            detection_methods.extend(file_result.detection_methods)

        # 内容分析
        if text_content:
            content_result = self.content_analyzer.analyze_content(text_content)
            results.append(content_result)
            detection_methods.extend(content_result.detection_methods)

        # 综合分析结果
        if not results:
            return SourceDetectionResult(
                source_type=DocumentSource.LOCAL_FILES,
                confidence=0.0,
                confidence_level=ConfidenceLevel.UNKNOWN,
                metadata={"error": "No detection data provided"},
                detection_methods=detection_methods
            )

        # 加权合并结果
        return self._merge_results(results, detection_methods)

    def _merge_results(
        self,
        results: List[SourceDetectionResult],
        detection_methods: List[str]
    ) -> SourceDetectionResult:
        """合并多个检测结果。"""
        # 按置信度排序
        results.sort(key=lambda x: x.confidence, reverse=True)

        # 选择最佳结果
        best_result = results[0]

        # 合并元数据
        merged_metadata = {}
        for result in results:
            merged_metadata.update(result.metadata)

        # 如果多个结果指向同一来源类型，增加置信度
        same_type_count = sum(1 for r in results if r.source_type == best_result.source_type)
        if same_type_count > 1:
            best_result.confidence = min(best_result.confidence + 0.1, 1.0)

        # 更新置信度级别
        best_result.confidence_level = self._get_confidence_level(best_result.confidence)
        best_result.metadata = merged_metadata
        best_result.detection_methods = list(set(detection_methods))

        return best_result

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """获取置信度级别。"""
        if confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif confidence >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN


# 导出
__all__ = [
    'SourceDetector',
    'URLPatternDetector',
    'FileTypeDetector',
    'ContentAnalyzer',
    'SourceDetectionResult',
    'ConfidenceLevel'
]