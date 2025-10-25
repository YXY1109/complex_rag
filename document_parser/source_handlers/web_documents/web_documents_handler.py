"""
Web Documents Handler

This module provides specialized processing for web documents including
HTML pages, Markdown files, API documentation, and web articles.
"""

import asyncio
import re
import mimetypes
from typing import Dict, Any, Optional, List, Union, AsyncGenerator
from urllib.parse import urlparse, urljoin, robots_txt_url
from pathlib import Path
import aiohttp
import aiofiles
from bs4 import BeautifulSoup
import markdown
from dataclasses import dataclass
from datetime import datetime

from ...interfaces.source_interface import (
    SourceHandler,
    FileSource,
    ProcessingStrategy,
    SourceDetectionResult,
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
class WebDocumentFeatures:
    """Features extracted from web documents."""
    content_type: str
    has_navigation: bool
    has_header_footer: bool
    link_count: int
    image_count: int
    video_count: int
    table_count: int
    code_block_count: int
    reading_time_minutes: float
    language: Optional[str]
    is_single_page: bool
    has_dynamic_content: bool


class WebDocumentsHandler(SourceHandler):
    """
    Handler for web documents including HTML, Markdown, and other web-based content.

    Features:
    - HTML parsing and content extraction
    - Markdown processing
    - Link resolution and validation
    - Navigation and header/footer detection
    - Content cleaning and normalization
    - Metadata extraction
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize web documents handler."""
        super().__init__(FileSource.WEB_DOCUMENTS, config)
        self.quality_monitor = QualityMonitor()
        self.session = None
        self.robots_cache = {}

    async def connect(self) -> bool:
        """Initialize HTTP session."""
        try:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    'User-Agent': self.config.get('user_agent', 'RAG-Parser/1.0 (+https://example.com/bot)'),
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate',
                }
            )
            return True
        except Exception as e:
            print(f"Failed to initialize HTTP session: {e}")
            return False

    async def disconnect(self) -> None:
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None

    async def can_handle(self, request: ParseRequest) -> bool:
        """Check if this handler can process the request."""
        # Check file extensions
        web_extensions = {'.html', '.htm', '.xhtml', '.md', '.markdown', '.txt'}

        if request.file_path:
            file_ext = Path(request.file_path).suffix.lower()
            if file_ext in web_extensions:
                return True

        # Check URL
        if request.url:
            parsed = urlparse(request.url)
            if parsed.scheme in ['http', 'https']:
                return True

        # Check MIME type
        if request.mime_type:
            web_types = {'text/html', 'text/markdown', 'text/plain', 'application/xhtml+xml'}
            if request.mime_type in web_types:
                return True

        # Check content
        if request.content:
            content_str = request.content.decode('utf-8', errors='ignore').lower()
            if any(tag in content_str for tag in ['<html', '<head>', '<body', '<doctype']):
                return True
            if content_str.startswith(('# ', '## ', '### ')):  # Markdown headers
                return True

        return False

    async def process(self, request: ParseRequest) -> ParseResponse:
        """Process web document."""
        session_id = f"web_{datetime.now().timestamp()}"
        processing_start = datetime.now()

        # Start quality monitoring
        quality_session = self.quality_monitor.start_session(
            session_id=session_id,
            file_source=FileSource.WEB_DOCUMENTS,
            strategy=request.strategy,
            file_size=len(request.content) if request.content else None
        )

        try:
            # Fetch content if needed
            content = await self._fetch_content(request)

            if not content:
                raise ValueError("No content available for processing")

            # Detect document format
            document_format = self._detect_format(request, content)

            # Extract content based on format
            if document_format == DocumentFormat.HTML:
                extracted_content = await self._process_html(content, request.url)
            elif document_format == DocumentFormat.MARKDOWN:
                extracted_content = await self._process_markdown(content, request.url)
            else:
                extracted_content = await self._process_text(content, request.url)

            # Create chunks
            chunks = await self._create_chunks(extracted_content, request)

            # Generate metadata
            metadata = await self._extract_metadata(extracted_content, request)

            # Calculate quality metrics
            await self._calculate_quality_metrics(quality_session, extracted_content, chunks)

            response = ParseResponse(
                content=extracted_content['text'],
                chunks=chunks,
                metadata=metadata,
                format=document_format,
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

    async def _fetch_content(self, request: ParseRequest) -> Optional[bytes]:
        """Fetch content from URL or use provided content."""
        if request.content:
            return request.content

        if request.url:
            if not self.session:
                await self.connect()

            # Check robots.txt if configured
            if self.config.get('respect_robots_txt', True):
                if not await self._check_robots_txt(request.url):
                    raise PermissionError("Access denied by robots.txt")

            # Fetch content
            try:
                async with self.session.get(request.url) as response:
                    if response.status == 200:
                        return await response.read()
                    else:
                        raise aiohttp.ClientResponseError(
                            request_info=response.request_info,
                            history=response.history,
                            status=response.status,
                            message=f"HTTP {response.status}"
                        )
            except Exception as e:
                raise ValueError(f"Failed to fetch content from {request.url}: {e}")

        return None

    def _detect_format(self, request: ParseRequest, content: bytes) -> DocumentFormat:
        """Detect document format."""
        content_str = content.decode('utf-8', errors='ignore').lower()

        # Check for HTML
        if any(tag in content_str for tag in ['<html', '<head>', '<body', '<doctype']):
            return DocumentFormat.HTML

        # Check for Markdown
        if (request.file_path and Path(request.file_path).suffix.lower() in ['.md', '.markdown']) or \
           any(pattern in content_str[:1000] for pattern in ['# ', '## ', '### ', '```', '**', '__']):
            return DocumentFormat.MARKDOWN

        # Default to text
        return DocumentFormat.TEXT

    async def _process_html(self, content: bytes, base_url: Optional[str]) -> Dict[str, Any]:
        """Process HTML content."""
        html = content.decode('utf-8', errors='ignore')
        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
            element.decompose()

        # Extract main content
        main_content = self._extract_main_content(soup)

        # Clean text
        text = main_content.get_text(separator='\n', strip=True)

        # Extract links
        links = []
        for a_tag in main_content.find_all('a', href=True):
            href = a_tag['href']
            if base_url:
                href = urljoin(base_url, href)
            links.append({
                'url': href,
                'text': a_tag.get_text(strip=True),
                'title': a_tag.get('title', '')
            })

        # Extract images
        images = []
        for img_tag in main_content.find_all('img', src=True):
            src = img_tag['src']
            if base_url:
                src = urljoin(base_url, src)
            images.append({
                'url': src,
                'alt': img_tag.get('alt', ''),
                'title': img_tag.get('title', '')
            })

        # Extract tables
        tables = []
        for table in main_content.find_all('table'):
            table_data = []
            for row in table.find_all('tr'):
                row_data = []
                for cell in row.find_all(['td', 'th']):
                    row_data.append(cell.get_text(strip=True))
                if row_data:
                    table_data.append(row_data)
            if table_data:
                tables.append(table_data)

        # Extract code blocks
        code_blocks = []
        for code in main_content.find_all(['pre', 'code']):
            code_text = code.get_text()
            if code_text.strip():
                code_blocks.append(code_text)

        # Detect language
        language = self._detect_language(main_content)

        return {
            'text': text,
            'title': soup.title.string if soup.title else '',
            'links': links,
            'images': images,
            'tables': tables,
            'code_blocks': code_blocks,
            'language': language,
            'features': WebDocumentFeatures(
                content_type='html',
                has_navigation=bool(soup.find('nav')),
                has_header_footer=bool(soup.find('header') or soup.find('footer')),
                link_count=len(links),
                image_count=len(images),
                video_count=len(main_content.find_all('video')),
                table_count=len(tables),
                code_block_count=len(code_blocks),
                reading_time_minutes=self._estimate_reading_time(text),
                language=language,
                is_single_page=True,
                has_dynamic_content=bool(soup.find_all(['script', 'iframe']))
            )
        }

    async def _process_markdown(self, content: bytes, base_url: Optional[str]) -> Dict[str, Any]:
        """Process Markdown content."""
        md_text = content.decode('utf-8', errors='ignore')

        # Convert to HTML for better parsing
        html = markdown.markdown(md_text, extensions=['extra', 'codehilite', 'toc'])
        soup = BeautifulSoup(html, 'html.parser')

        # Extract text
        text = soup.get_text(separator='\n', strip=True)

        # Extract links (from markdown)
        links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', md_text)
        processed_links = []
        for link_text, link_url in links:
            if base_url:
                link_url = urljoin(base_url, link_url)
            processed_links.append({
                'url': link_url,
                'text': link_text,
                'title': ''
            })

        # Extract code blocks
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)\n```', md_text, re.DOTALL)

        # Extract headers for structure
        headers = re.findall(r'^(#{1,6})\s+(.+)$', md_text, re.MULTILINE)

        return {
            'text': text,
            'title': self._extract_title_from_markdown(md_text),
            'links': processed_links,
            'images': [],  # Would need more complex parsing for images
            'tables': [],  # Would need table extension parsing
            'code_blocks': code_blocks,
            'language': self._detect_language(soup),
            'headers': headers,
            'features': WebDocumentFeatures(
                content_type='markdown',
                has_navigation=False,
                has_header_footer=False,
                link_count=len(processed_links),
                image_count=0,
                video_count=0,
                table_count=0,
                code_block_count=len(code_blocks),
                reading_time_minutes=self._estimate_reading_time(text),
                language=self._detect_language(soup),
                is_single_page=True,
                has_dynamic_content=False
            )
        }

    async def _process_text(self, content: bytes, base_url: Optional[str]) -> Dict[str, Any]:
        """Process plain text content."""
        text = content.decode('utf-8', errors='ignore')

        # Extract URLs from text
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        links = [{'url': url, 'text': url, 'title': ''} for url in re.findall(url_pattern, text)]

        return {
            'text': text,
            'title': text.split('\n')[0][:100] if text else '',
            'links': links,
            'images': [],
            'tables': [],
            'code_blocks': [],
            'language': None,
            'features': WebDocumentFeatures(
                content_type='text',
                has_navigation=False,
                has_header_footer=False,
                link_count=len(links),
                image_count=0,
                video_count=0,
                table_count=0,
                code_block_count=0,
                reading_time_minutes=self._estimate_reading_time(text),
                language=None,
                is_single_page=True,
                has_dynamic_content=False
            )
        }

    def _extract_main_content(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Extract main content from HTML."""
        # Try to find main content area
        main_selectors = [
            'main', 'article', '[role="main"]', '.content', '#content',
            '.post-content', '.entry-content', '.article-body'
        ]

        for selector in main_selectors:
            main_element = soup.select_one(selector)
            if main_element:
                return main_element

        # Fallback to body
        return soup.find('body') or soup

    def _detect_language(self, soup: BeautifulSoup) -> Optional[str]:
        """Detect document language."""
        # Check HTML lang attribute
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            return html_tag.get('lang')[:2]

        # Check meta tags
        meta_lang = soup.find('meta', attrs={'http-equiv': 'language'})
        if meta_lang and meta_lang.get('content'):
            return meta_lang.get('content')[:2]

        return None

    def _estimate_reading_time(self, text: str) -> float:
        """Estimate reading time in minutes."""
        words_per_minute = 200
        word_count = len(text.split())
        return max(1, word_count / words_per_minute)

    def _extract_title_from_markdown(self, content: str) -> str:
        """Extract title from markdown content."""
        lines = content.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line[2:].strip()
        return lines[0][:100] if lines else ''

    async def _create_chunks(self, content: Dict[str, Any], request: ParseRequest) -> List[DocumentChunk]:
        """Create document chunks from processed content."""
        text = content['text']
        if not text.strip():
            return []

        # Get chunking parameters
        params = request.custom_params or {}
        chunk_size = params.get('chunk_size', 800)
        overlap = params.get('overlap_size', 200)

        # Create chunks
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0

        for i, word in enumerate(words):
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append(DocumentChunk(
                    content=chunk_text,
                    metadata=Metadata({
                        'chunk_index': len(chunks),
                        'word_count': len(current_chunk),
                        'source': 'web_documents',
                        'title': content.get('title', ''),
                        'language': content.get('language'),
                        'content_type': content['features'].content_type
                    })
                ))

                # Start new chunk with overlap
                overlap_words = min(overlap // 6, len(current_chunk))  # ~6 chars per word
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                current_length = sum(len(w) + 1 for w in current_chunk)

            current_chunk.append(word)
            current_length += len(word) + 1

        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(DocumentChunk(
                content=chunk_text,
                metadata=Metadata({
                    'chunk_index': len(chunks),
                    'word_count': len(current_chunk),
                    'source': 'web_documents',
                    'title': content.get('title', ''),
                    'language': content.get('language'),
                    'content_type': content['features'].content_type
                })
            ))

        return chunks

    async def _extract_metadata(self, content: Dict[str, Any], request: ParseRequest) -> Metadata:
        """Extract metadata from processed content."""
        features = content['features']

        metadata_dict = {
            'source_type': 'web_documents',
            'processing_strategy': request.strategy.value,
            'title': content.get('title', ''),
            'language': features.language,
            'content_type': features.content_type,
            'reading_time_minutes': features.reading_time_minutes,
            'link_count': features.link_count,
            'image_count': features.image_count,
            'video_count': features.video_count,
            'table_count': features.table_count,
            'code_block_count': features.code_block_count,
            'has_navigation': features.has_navigation,
            'has_header_footer': features.has_header_footer,
            'is_single_page': features.is_single_page,
            'has_dynamic_content': features.has_dynamic_content
        }

        # Add URL if available
        if request.url:
            metadata_dict['source_url'] = request.url
            parsed_url = urlparse(request.url)
            metadata_dict['domain'] = parsed_url.netloc

        # Add file path if available
        if request.file_path:
            metadata_dict['file_path'] = request.file_path

        return Metadata(metadata_dict)

    async def _calculate_quality_metrics(
        self,
        session_id: str,
        content: Dict[str, Any],
        chunks: List[DocumentChunk]
    ):
        """Calculate quality metrics for the processing session."""
        # Text extraction quality
        text = content['text']
        if text.strip():
            text_quality = min(1.0, len(text.strip()) / 100)  # Basic quality measure
            self.quality_monitor.add_measurement(
                session_id, QualityMetric.TEXT_EXTRACTION, text_quality
            )

        # Structure preservation
        features = content['features']
        structure_score = 0.0
        if features.link_count > 0:
            structure_score += 0.2
        if features.table_count > 0:
            structure_score += 0.3
        if features.code_block_count > 0:
            structure_score += 0.2
        if features.has_navigation:
            structure_score += 0.1
        if content.get('title'):
            structure_score += 0.2
        structure_score = min(1.0, structure_score)

        self.quality_monitor.add_measurement(
            session_id, QualityMetric.STRUCTURE_PRESERVATION, structure_score
        )

        # Content completeness
        completeness_score = min(1.0, len(text) / 500)  # Assuming 500 chars is reasonable minimum
        self.quality_monitor.add_measurement(
            session_id, QualityMetric.CONTENT_COMPLETENESS, completeness_score
        )

    async def _check_robots_txt(self, url: str) -> bool:
        """Check if URL access is allowed by robots.txt."""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"

            # Check cache
            if robots_url in self.robots_cache:
                return self.robots_cache[robots_url]

            # Fetch robots.txt
            async with self.session.get(robots_url) as response:
                if response.status == 200:
                    robots_content = await response.text()
                    # Simple check - allow all if no specific disallow rules
                    allowed = True
                    for line in robots_content.split('\n'):
                        if line.strip().startswith('Disallow:'):
                            path = line.split(':', 1)[1].strip()
                            if path and parsed_url.path.startswith(path):
                                allowed = False
                                break
                else:
                    # No robots.txt found, allow access
                    allowed = True

            self.robots_cache[robots_url] = allowed
            return allowed

        except Exception:
            # Error checking robots.txt, allow access
            return True