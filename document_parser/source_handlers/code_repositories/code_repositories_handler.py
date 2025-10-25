"""
Code Repositories Handler

This module provides specialized processing for code repositories and source files
including syntax highlighting, dependency extraction, and code structure analysis.
"""

import asyncio
import re
import mimetypes
from typing import Dict, Any, Optional, List, Union, Set
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import io

try:
    import pygments
    from pygments.lexers import get_lexer_by_name, guess_lexer
    from pygments.formatters import get_formatter_by_name
    PYGMENTS_AVAILABLE = True
except ImportError:
    PYGMENTS_AVAILABLE = False

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
class CodeFeatures:
    """Features extracted from code."""
    language: str
    line_count: int
    function_count: int
    class_count: int
    import_count: int
    comment_ratio: float
    has_docstrings: bool
    complexity_score: float
    dependency_count: int
    test_file: bool
    config_file: bool


class CodeRepositoriesHandler(SourceHandler):
    """
    Handler for code repositories and source files.

    Features:
    - Multi-language support (Python, JavaScript, Java, C++, Go, etc.)
    - Syntax highlighting and formatting preservation
    - Function and class extraction
    - Import/dependency analysis
    - Comment and docstring extraction
    - Code complexity assessment
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize code repositories handler."""
        super().__init__(FileSource.CODE_REPOSITORIES, config)
        self.quality_monitor = QualityMonitor()
        self.supported_extensions = self._get_supported_extensions()
        self.language_patterns = self._setup_language_patterns()

    def _get_supported_extensions(self) -> Dict[str, str]:
        """Get supported code file extensions."""
        return {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'jsx',
            '.tsx': 'tsx',
            '.java': 'java',
            '.cpp': 'cpp',
            '.cxx': 'cpp',
            '.cc': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.cs': 'csharp',
            '.vb': 'vbnet',
            '.sh': 'bash',
            '.bash': 'bash',
            '.zsh': 'zsh',
            '.fish': 'fish',
            '.ps1': 'powershell',
            '.bat': 'batch',
            '.cmd': 'batch',
            '.sql': 'sql',
            '.html': 'html',
            '.css': 'css',
            '.scss': 'scss',
            '.sass': 'sass',
            '.less': 'less',
            '.vue': 'vue',
            '.dart': 'dart',
            '.lua': 'lua',
            '.r': 'r',
            '.m': 'objective-c',
            '.pl': 'perl',
            '.pm': 'perl',
            '.tcl': 'tcl',
            '.vim': 'vim',
            '.elm': 'elm',
            '.hs': 'haskell',
            '.ml': 'ocaml',
            '.fs': 'fsharp',
            '.ex': 'elixir',
            '.exs': 'elixir',
            '.erl': 'erlang',
            '.vim': 'viml',
            '.dockerfile': 'dockerfile',
            'dockerfile': 'dockerfile',
            '.json': 'json',
            '.xml': 'xml',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.toml': 'toml',
            '.ini': 'ini',
            '.cfg': 'ini',
            '.conf': 'ini',
            '.md': 'markdown',
            '.rst': 'rst',
            '.txt': 'text'
        }

    def _setup_language_patterns(self) -> Dict[str, Dict[str, str]]:
        """Setup language-specific patterns for analysis."""
        return {
            'python': {
                'function': r'def\s+(\w+)\s*\(',
                'class': r'class\s+(\w+)',
                'import': r'(?:from\s+\S+\s+)?import\s+',
                'comment': r'#.*$',
                'docstring': r'""".*?"""',
                'test_file': r'(?:test_|_test\.py$|tests?\.py$)',
                'config_file': r'(?:setup\.py$|requirements\.txt$|pyproject\.toml$|\.env$)'
            },
            'javascript': {
                'function': r'(?:function\s+(\w+)\s*\(|const\s+(\w+)\s*=\s*(?:async\s+)?\(|=>)',
                'class': r'class\s+(\w+)',
                'import': r'(?:import\s+.*from\s+|const\s+.*=.*require\()',
                'comment': r'//.*$|/\*.*?\*/',
                'test_file': r'(?:\.test\.js$|\.spec\.js$|test/|spec/)',
                'config_file': r'(?:package\.json$|webpack\.config\.js$|\.eslintrc$|\.babelrc$)'
            },
            'java': {
                'function': r'(?:public|private|protected)?\s*(?:static\s+)?(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*(?:throws\s+\w+\s*)?{',
                'class': r'(?:public\s+)?(?:class|interface|enum)\s+(\w+)',
                'import': r'import\s+',
                'comment': r'//.*$|/\*.*?\*/',
                'test_file': r'(?:.*Test\.java$|.*Tests\.java$)',
                'config_file': r'(?:pom\.xml$|build\.gradle$|application\.properties$)'
            }
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
            filename = Path(request.file_path).name.lower()

            if file_ext in self.supported_extensions or filename in self.supported_extensions:
                return True

        # Check URL patterns (GitHub, GitLab, etc.)
        if request.url:
            github_patterns = [
                r'github\.com',
                r'gitlab\.com',
                r'bitbucket\.org',
                r'gist\.github\.com',
                r'raw\.githubusercontent\.com'
            ]
            if any(re.search(pattern, request.url) for pattern in github_patterns):
                return True

        # Check MIME type
        if request.mime_type:
            code_types = {
                'text/x-python', 'text/javascript', 'application/javascript',
                'text/x-java-source', 'text/x-c++src', 'text/x-csrc'
            }
            if request.mime_type in code_types:
                return True

        # Check content for code patterns
        if request.content:
            content_str = request.content.decode('utf-8', errors='ignore')
            if self._detect_code_patterns(content_str):
                return True

        return False

    async def process(self, request: ParseRequest) -> ParseResponse:
        """Process code repository file."""
        session_id = f"code_{datetime.now().timestamp()}"
        processing_start = datetime.now()

        # Start quality monitoring
        quality_session = self.quality_monitor.start_session(
            session_id=session_id,
            file_source=FileSource.CODE_REPOSITORIES,
            strategy=request.strategy,
            file_size=len(request.content) if request.content else None
        )

        try:
            # Get content
            content = await self._get_content(request)
            if not content:
                raise ValueError("No content available for processing")

            # Detect language
            language = self._detect_language(content, request.file_path)

            # Analyze code structure
            code_features = await self._analyze_code(content, language, request.file_path)

            # Extract code elements
            code_elements = await self._extract_code_elements(content, language)

            # Create chunks
            chunks = await self._create_chunks(content, code_features, request)

            # Generate metadata
            metadata = await self._extract_metadata(code_features, code_elements, request)

            # Calculate quality metrics
            await self._calculate_quality_metrics(quality_session, content, code_features, chunks)

            response = ParseResponse(
                content=content,
                chunks=chunks,
                metadata=metadata,
                format=DocumentFormat.CODE,
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

    async def _get_content(self, request: ParseRequest) -> Optional[str]:
        """Get content from request."""
        if request.content:
            return request.content.decode('utf-8', errors='ignore')
        return None

    def _detect_code_patterns(self, content: str) -> bool:
        """Detect if content contains code patterns."""
        code_indicators = [
            r'function\s+\w+\s*\(',
            r'class\s+\w+',
            r'def\s+\w+\s*\(',
            r'import\s+',
            r'#include\s+',
            r'public\s+class',
            r'const\s+\w+\s*=',
            r'let\s+\w+\s*=',
            r'var\s+\w+\s*='
        ]

        return any(re.search(pattern, content, re.IGNORECASE) for pattern in code_indicators)

    def _detect_language(self, content: str, file_path: Optional[str]) -> str:
        """Detect programming language."""
        # Check file extension first
        if file_path:
            file_ext = Path(file_path).suffix.lower()
            filename = Path(file_path).name.lower()

            if file_ext in self.supported_extensions:
                return self.supported_extensions[file_ext]
            elif filename in self.supported_extensions:
                return self.supported_extensions[filename]

        # Try to guess using pygments
        if PYGMENTS_AVAILABLE:
            try:
                lexer = guess_lexer(content)
                return lexer.name.lower()
            except:
                pass

        # Fallback to pattern matching
        if 'def ' in content and 'import ' in content:
            return 'python'
        elif 'function ' in content and ('var ' in content or 'let ' in content or 'const ' in content):
            return 'javascript'
        elif 'public class ' in content and 'import ' in content:
            return 'java'
        elif '#include' in content:
            return 'c'
        elif 'package ' in content and 'func ' in content:
            return 'go'

        return 'text'

    async def _analyze_code(self, content: str, language: str, file_path: Optional[str]) -> CodeFeatures:
        """Analyze code structure and features."""
        lines = content.split('\n')
        line_count = len(lines)

        # Get language patterns
        patterns = self.language_patterns.get(language, {})

        # Count functions
        function_count = 0
        if 'function' in patterns:
            function_count = len(re.findall(patterns['function'], content, re.MULTILINE))

        # Count classes
        class_count = 0
        if 'class' in patterns:
            class_count = len(re.findall(patterns['class'], content, re.MULTILINE))

        # Count imports
        import_count = 0
        if 'import' in patterns:
            import_count = len(re.findall(patterns['import'], content, re.MULTILINE))

        # Calculate comment ratio
        comment_lines = 0
        code_lines = 0
        if 'comment' in patterns:
            for line in lines:
                if re.search(patterns['comment'], line.strip()):
                    comment_lines += 1
                elif line.strip():
                    code_lines += 1

        comment_ratio = comment_lines / (code_lines + comment_lines) if (code_lines + comment_lines) > 0 else 0

        # Check for docstrings
        has_docstrings = False
        if 'docstring' in patterns:
            has_docstrings = bool(re.search(patterns['docstring'], content, re.DOTALL))

        # Calculate complexity score (simplified)
        complexity_indicators = ['if ', 'for ', 'while ', 'try:', 'except', 'catch', 'switch', 'case']
        complexity_score = sum(content.count(indicator) for indicator in complexity_indicators) / line_count

        # Count dependencies
        dependency_count = import_count

        # Check if test file
        is_test_file = False
        if file_path and 'test_file' in patterns:
            is_test_file = bool(re.search(patterns['test_file'], file_path.lower()))

        # Check if config file
        is_config_file = False
        if file_path and 'config_file' in patterns:
            is_config_file = bool(re.search(patterns['config_file'], file_path.lower()))

        return CodeFeatures(
            language=language,
            line_count=line_count,
            function_count=function_count,
            class_count=class_count,
            import_count=import_count,
            comment_ratio=comment_ratio,
            has_docstrings=has_docstrings,
            complexity_score=complexity_score,
            dependency_count=dependency_count,
            test_file=is_test_file,
            config_file=is_config_file
        )

    async def _extract_code_elements(self, content: str, language: str) -> Dict[str, List[str]]:
        """Extract code elements like functions, classes, imports."""
        elements = {
            'functions': [],
            'classes': [],
            'imports': [],
            'comments': []
        }

        patterns = self.language_patterns.get(language, {})

        # Extract functions
        if 'function' in patterns:
            matches = re.finditer(patterns['function'], content, re.MULTILINE)
            elements['functions'] = [match.group(1) if match.groups() else match.group(0) for match in matches]

        # Extract classes
        if 'class' in patterns:
            matches = re.finditer(patterns['class'], content, re.MULTILINE)
            elements['classes'] = [match.group(1) if match.groups() else match.group(0) for match in matches]

        # Extract imports
        if 'import' in patterns:
            matches = re.finditer(patterns['import'], content, re.MULTILINE)
            elements['imports'] = [match.group(0).strip() for match in matches]

        # Extract comments
        if 'comment' in patterns:
            matches = re.finditer(patterns['comment'], content, re.MULTILINE)
            elements['comments'] = [match.group(0).strip() for match in matches]

        return elements

    async def _create_chunks(self, content: str, features: CodeFeatures, request: ParseRequest) -> List[DocumentChunk]:
        """Create document chunks from code content."""
        if not content.strip():
            return []

        # Get chunking parameters
        params = request.custom_params or {}
        chunk_size = params.get('chunk_size', 1200)  # Larger chunks for code
        overlap = params.get('overlap_size', 50)

        # Preserve code structure by chunking at logical boundaries
        lines = content.split('\n')
        chunks = []

        # Try to chunk at function/class boundaries
        current_chunk = []
        current_length = 0

        for i, line in enumerate(lines):
            line_length = len(line) + 1

            # Check for logical boundaries
            is_boundary = any(
                line.strip().startswith(prefix) for prefix in [
                    'def ', 'class ', 'function ', 'public class', 'private class',
                    'import ', 'from ', 'package ', 'namespace '
                ]
            )

            if (current_length + line_length > chunk_size and current_chunk) or \
               (is_boundary and current_length > chunk_size * 0.7 and current_chunk):

                # Save current chunk
                chunk_text = '\n'.join(current_chunk)
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata=Metadata({
                        'chunk_index': len(chunks),
                        'line_count': len(current_chunk),
                        'source': 'code_repositories',
                        'language': features.language,
                        'code_chunk': True
                    })
                ))

                # Start new chunk with minimal overlap for code
                current_chunk = [line]  # Start with boundary line
                current_length = line_length
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
                    'source': 'code_repositories',
                    'language': features.language,
                    'code_chunk': True
                })
            ))

        return chunks

    async def _extract_metadata(self, features: CodeFeatures, elements: Dict[str, List[str]], request: ParseRequest) -> Metadata:
        """Extract metadata from code analysis."""
        metadata_dict = {
            'source_type': 'code_repositories',
            'language': features.language,
            'processing_strategy': request.strategy.value,
            'line_count': features.line_count,
            'function_count': features.function_count,
            'class_count': features.class_count,
            'import_count': features.import_count,
            'comment_ratio': features.comment_ratio,
            'has_docstrings': features.has_docstrings,
            'complexity_score': features.complexity_score,
            'dependency_count': features.dependency_count,
            'is_test_file': features.test_file,
            'is_config_file': features.config_file,
            'functions': elements.get('functions', []),
            'classes': elements.get('classes', []),
            'imports': elements.get('imports', [])
        }

        # Add file path if available
        if request.file_path:
            metadata_dict['file_path'] = request.file_path

        return Metadata(metadata_dict)

    async def _calculate_quality_metrics(
        self,
        session_id: str,
        content: str,
        features: CodeFeatures,
        chunks: List[DocumentChunk]
    ):
        """Calculate quality metrics for code processing."""
        # Code structure preservation
        structure_score = min(1.0, (features.function_count + features.class_count) / max(1, features.line_count / 20))
        self.quality_monitor.add_measurement(
            session_id, QualityMetric.STRUCTURE_PRESERVATION, structure_score
        )

        # Code quality (comments, documentation)
        quality_score = min(1.0, (features.comment_ratio * 0.5 + (1 if features.has_docstrings else 0) * 0.5))
        self.quality_monitor.add_measurement(
            session_id, QualityMetric.ACCURACY, quality_score
        )

        # Processing efficiency
        efficiency_score = min(1.0, len(content) / 1000)  # Expect reasonable file size
        self.quality_monitor.add_measurement(
            session_id, QualityMetric.PROCESSING_SPEED, efficiency_score
        )