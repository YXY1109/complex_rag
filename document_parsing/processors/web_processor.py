"""
网页文档处理器

此模块实现HTML、Markdown、API文档等网页文档的专用处理器，
参考RAGFlow rag/app架构中的web_documents处理逻辑。
"""

import asyncio
import os
import re
import html
from typing import Dict, Any, List, Optional, Union
from urllib.parse import urljoin, urlparse
from pathlib import Path
import tempfile
import uuid

from bs4 import BeautifulSoup
import markdown
import requests
from readability import Document
import trafilatura
from trafilatura.settings import use_config

from .base_processor import BaseProcessor
from ..interfaces.parser_interface import (
    ParseResult,
    DocumentMetadata,
    DocumentType,
    ProcessingStrategy,
    TextChunk,
    ImageInfo,
    TableInfo,
    ParseException
)
from ..strategy_config import ProcessingStrategyConfig


class WebDocumentProcessor(BaseProcessor):
    """
    网页文档处理器。

    专门处理HTML、Markdown、API文档等网页格式文档。
    """

    def __init__(self, config):
        """
        初始化网页文档处理器。

        Args:
            config: 解析器配置
        """
        super().__init__(config)
        self.supported_extensions = {'.html', '.htm', '.md', '.markdown', '.txt'}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    async def initialize(self) -> bool:
        """
        初始化处理器。

        Returns:
            bool: 初始化是否成功
        """
        try:
            # 测试依赖库
            import bs4
            import markdown
            import requests
            import trafilatura
            return True
        except ImportError as e:
            print(f"网页处理器初始化失败，缺少依赖: {e}")
            return False

    async def cleanup(self) -> None:
        """清理处理器资源。"""
        if hasattr(self.session, 'close'):
            self.session.close()

    async def _parse_with_config(
        self,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig,
        **kwargs
    ) -> ParseResult:
        """
        使用配置解析网页文档。

        Args:
            file_path: 文件路径
            strategy: 处理策略
            config: 策略配置
            **kwargs: 额外参数

        Returns:
            ParseResult: 解析结果
        """
        # 验证文件
        self._validate_file(file_path)

        # 确定文档类型
        file_ext = Path(file_path).suffix.lower()
        is_html = file_ext in {'.html', '.htm'}
        is_markdown = file_ext in {'.md', '.markdown'}
        is_text = file_ext == '.txt'

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        if not content.strip():
            raise ParseException("文件内容为空", parser=self.parser_name, file_path=file_path)

        # 根据文档类型解析
        if is_html:
            return await self._parse_html_content(content, file_path, strategy, config)
        elif is_markdown:
            return await self._parse_markdown_content(content, file_path, strategy, config)
        elif is_text:
            return await self._parse_text_content(content, file_path, strategy, config)
        else:
            # 尝试自动检测
            if content.strip().startswith('<!DOCTYPE') or content.strip().startswith('<html'):
                return await self._parse_html_content(content, file_path, strategy, config)
            else:
                return await self._parse_text_content(content, file_path, strategy, config)

    async def _parse_html_content(
        self,
        html_content: str,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig
    ) -> ParseResult:
        """
        解析HTML内容。

        Args:
            html_content: HTML内容
            file_path: 文件路径
            strategy: 处理策略
            config: 策略配置

        Returns:
            ParseResult: 解析结果
        """
        try:
            # 使用trafilatura提取主要内容
            extracted = trafilatura.extract(
                html_content,
                include_images=config.image.extract_images,
                include_tables=config.table.enabled,
                output_format='xml',
                config=use_config()
            )

            if extracted:
                # 使用BeautifulSoup进一步处理
                soup = BeautifulSoup(html_content, 'html.parser')

                # 提取文本
                full_text = await self._extract_text_from_html(soup, config)

                # 提取元数据
                metadata = await self._extract_html_metadata(soup, file_path)

                # 提取图像信息
                images = await self._extract_html_images(soup, config) if config.image.extract_images else []

                # 提取表格信息
                tables = await self._extract_html_tables(soup, config) if config.table.enabled else []

                # 创建文本块
                text_chunks = self._create_text_chunks(
                    full_text,
                    config.text.chunk_size,
                    config.text.chunk_overlap
                )

            else:
                # 备用方案：直接使用BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')

                # 移除脚本和样式
                for script in soup(["script", "style"]):
                    script.decompose()

                full_text = soup.get_text(separator=' ', strip=True)
                metadata = await self._extract_html_metadata(soup, file_path)
                images = []
                tables = []
                text_chunks = self._create_text_chunks(
                    full_text,
                    config.text.chunk_size,
                    config.text.chunk_overlap
                )

            # 创建解析结果
            result = ParseResult(
                success=True,
                metadata=metadata,
                full_text=full_text,
                text_chunks=text_chunks,
                images=images,
                tables=tables,
                structured_data={
                    "html_elements": self._count_html_elements(soup),
                    "links": self._extract_links(soup),
                    "headings": self._extract_headings(soup)
                }
            )

            return result

        except Exception as e:
            raise ParseException(f"HTML解析失败: {str(e)}", parser=self.parser_name, file_path=file_path)

    async def _parse_markdown_content(
        self,
        markdown_content: str,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig
    ) -> ParseResult:
        """
        解析Markdown内容。

        Args:
            markdown_content: Markdown内容
            file_path: 文件路径
            strategy: 处理策略
            config: 策略配置

        Returns:
            ParseResult: 解析结果
        """
        try:
            # 转换Markdown为HTML
            html_content = markdown.markdown(
                markdown_content,
                extensions=[
                    'markdown.extensions.extra',
                    'markdown.extensions.codehilite',
                    'markdown.extensions.toc',
                    'markdown.extensions.tables',
                    'markdown.extensions.fenced_code'
                ],
                extension_configs={
                    'markdown.extensions.codehilite': {
                        'css_class': 'highlight',
                        'use_pygments': True
                    }
                }
            )

            # 转换为BeautifulSoup对象
            soup = BeautifulSoup(f'<div class="markdown-body">{html_content}</div>', 'html.parser')

            # 提取文本
            full_text = await self._extract_text_from_markdown(markdown_content, soup, config)

            # 提取元数据
            metadata = await self._extract_markdown_metadata(markdown_content, file_path)

            # 提取代码块
            code_blocks = await self._extract_markdown_code_blocks(markdown_content)

            # 提取链接
            links = await self._extract_markdown_links(markdown_content)

            # 创建文本块
            text_chunks = self._create_text_chunks(
                full_text,
                config.text.chunk_size,
                config.text.chunk_overlap
            )

            # 创建解析结果
            result = ParseResult(
                success=True,
                metadata=metadata,
                full_text=full_text,
                text_chunks=text_chunks,
                structured_data={
                    "code_blocks": code_blocks,
                    "links": links,
                    "headings": self._extract_headings(soup)
                }
            )

            return result

        except Exception as e:
            raise ParseException(f"Markdown解析失败: {str(e)}", parser=self.parser_name, file_path=file_path)

    async def _parse_text_content(
        self,
        text_content: str,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig
    ) -> ParseResult:
        """
        解析纯文本内容。

        Args:
            text_content: 文本内容
            file_path: 文件路径
            strategy: 处理策略
            config: 策略配置

        Returns:
            ParseResult: 解析结果
        """
        try:
            # 清理文本
            cleaned_text = self._clean_text_content(text_content, config)

            # 创建元数据
            metadata = await self._extract_text_metadata(text_content, file_path)

            # 创建文本块
            text_chunks = self._create_text_chunks(
                cleaned_text,
                config.text.chunk_size,
                config.text.chunk_overlap
            )

            # 提取文本统计
            statistics = self._extract_text_statistics(cleaned_text)

            # 创建解析结果
            result = ParseResult(
                success=True,
                metadata=metadata,
                full_text=cleaned_text,
                text_chunks=text_chunks,
                statistics=statistics
            )

            return result

        except Exception as e:
            raise ParseException(f"文本解析失败: {str(e)}", parser=self.parser_name, file_path=file_path)

    async def _extract_text_from_html(
        self,
        soup: BeautifulSoup,
        config: ProcessingStrategyConfig
    ) -> str:
        """
        从HTML提取文本。

        Args:
            soup: BeautifulSoup对象
            config: 配置

        Returns:
            str: 提取的文本
        """
        if config.text.normalize_whitespace:
            # 移除脚本和样式
            for script in soup(["script", "style"]):
                script.decompose()

            # 获取文本并规范化空白字符
            text = soup.get_text(separator=' ', strip=True)

            # 移除多余的空白字符
            text = re.sub(r'\s+', ' ', text)

            # 移除页眉页脚（简单启发式）
            lines = text.split('\n')
            if len(lines) > 10:
                # 移除可能包含导航信息的头部和尾部行
                filtered_lines = []
                for i, line in enumerate(lines):
                    line = line.strip()
                    if not line:
                        continue
                    # 简单的页眉页脚检测
                    if i < 3 and len(line) < 50 and any(word in line.lower() for word in ['nav', 'menu', 'header', 'footer']):
                        continue
                    if i > len(lines) - 3 and len(line) < 50 and any(word in line.lower() for word in ['copyright', '©', 'all rights']):
                        continue
                    filtered_lines.append(line)
                text = ' '.join(filtered_lines)
        else:
            text = soup.get_text()

        return text

    async def _extract_text_from_markdown(
        self,
        markdown_content: str,
        soup: BeautifulSoup,
        config: ProcessingStrategyConfig
    ) -> str:
        """
        从Markdown提取文本。

        Args:
            markdown_content: Markdown内容
            soup: BeautifulSoup对象
            config: 配置

        Returns:
            str: 提取的文本
        """
        if config.text.normalize_whitespace:
            # 移除代码块中的代码（保留代码块标记）
            text = soup.get_text(separator=' ', strip=True)
            text = re.sub(r'\s+', ' ', text)
        else:
            text = soup.get_text()

        return text

    def _clean_text_content(self, text: str, config: ProcessingStrategyConfig) -> str:
        """
        清理文本内容。

        Args:
            text: 原始文本
            config: 配置

        Returns:
            str: 清理后的文本
        """
        # 移除BOM
        text = text.lstrip('\ufeff')

        # 标准化换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        if config.text.remove_extra_whitespace:
            # 移除多余的空白字符
            text = re.sub(r'\n\s*\n', '\n\n', text)  # 多个空行替换为两个
            text = re.sub(r'[ \t]+', ' ', text)      # 多个空格替换为一个

        if config.text.remove_page_numbers:
            # 移除页码（简单模式）
            text = re.sub(r'\n\s*\d+\s*\n', '\n', text)

        return text.strip()

    async def _extract_html_metadata(self, soup: BeautifulSoup, file_path: str) -> DocumentMetadata:
        """
        提取HTML元数据。

        Args:
            soup: BeautifulSoup对象
            file_path: 文件路径

        Returns:
            DocumentMetadata: 元数据
        """
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)

        # 提取标题
        title_tag = soup.find('title')
        title = title_tag.get_text().strip() if title_tag else file_name

        # 提取meta信息
        author = None
        description = None
        language = None

        for meta in soup.find_all('meta'):
            name = meta.get('name', '').lower()
            prop = meta.get('property', '').lower()
            content = meta.get('content', '')

            if name == 'author' or prop == 'article:author':
                author = content
            elif name == 'description' or prop == 'og:description':
                description = content
            elif name == 'language' or prop == 'og:locale':
                language = content

        # 提取结构信息
        has_images = bool(soup.find('img'))
        has_tables = bool(soup.find('table'))
        has_forms = bool(soup.find('form'))

        return DocumentMetadata(
            file_name=file_name,
            file_size=file_size,
            file_type=DocumentType.HTML,
            mime_type="text/html",
            title=title,
            author=author,
            language=language,
            page_count=1,  # HTML通常为单页
            has_images=has_images,
            has_tables=has_tables,
            has_forms=has_forms,
            metadata={
                "description": description,
                "headings": len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
                "links": len(soup.find_all('a', href=True)),
                "forms": len(soup.find_all('form'))
            }
        )

    async def _extract_markdown_metadata(self, content: str, file_path: str) -> DocumentMetadata:
        """
        提取Markdown元数据。

        Args:
            content: Markdown内容
            file_path: 文件路径

        Returns:
            DocumentMetadata: 元数据
        """
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)

        # 提取标题（第一个#标题）
        title = file_name
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('# '):
                title = line[2:].strip()
                break

        # 提取作者和日期（简单模式）
        author = None
        created_date = None

        # 统计结构元素
        has_code_blocks = '```' in content
        has_tables = '|' in content and '\n|' in content
        has_images = '![' in content

        return DocumentMetadata(
            file_name=file_name,
            file_size=file_size,
            file_type=DocumentType.MD,
            mime_type="text/markdown",
            title=title,
            author=author,
            created_date=created_date,
            page_count=1,
            has_images=has_images,
            has_tables=has_tables,
            metadata={
                "code_blocks": content.count('```') // 2,
                "headings": content.count('#'),
                "links": content.count('[') - content.count('`['),  # 排除代码中的链接
                "lists": sum(1 for line in content.split('\n') if line.strip().startswith(('-', '*', '+')))
            }
        )

    async def _extract_text_metadata(self, content: str, file_path: str) -> DocumentMetadata:
        """
        提取文本元数据。

        Args:
            content: 文本内容
            file_path: 文件路径

        Returns:
            DocumentMetadata: 元数据
        """
        file_size = os.path.getsize(file_path)
        file_name = os.path.basename(file_path)

        # 使用文件名作为标题
        title = os.path.splitext(file_name)[0]

        return DocumentMetadata(
            file_name=file_name,
            file_size=file_size,
            file_type=DocumentType.TXT,
            mime_type="text/plain",
            title=title,
            page_count=1,
            word_count=len(content.split()),
            character_count=len(content),
            metadata={}
        )

    async def _extract_html_images(
        self,
        soup: BeautifulSoup,
        config: ProcessingStrategyConfig
    ) -> List[ImageInfo]:
        """
        提取HTML图像信息。

        Args:
            soup: BeautifulSoup对象
            config: 配置

        Returns:
            List[ImageInfo]: 图像信息列表
        """
        images = []
        img_tags = soup.find_all('img')

        for i, img in enumerate(img_tags):
            src = img.get('src', '')
            alt = img.get('alt', '')
            title = img.get('title', '')

            if src:
                image_info = ImageInfo(
                    image_id=f"img_{i}",
                    page_number=1,
                    bbox=[0, 0, 0, 0],  # HTML中很难准确定位
                    width=int(img.get('width', 0)) or 0,
                    height=int(img.get('height', 0)) or 0,
                    format=src.split('.')[-1].split('?')[0] if '.' in src else 'unknown',
                    caption=alt or title,
                    metadata={
                        "src": src,
                        "alt": alt,
                        "title": title
                    }
                )
                images.append(image_info)

        return images

    async def _extract_html_tables(
        self,
        soup: BeautifulSoup,
        config: ProcessingStrategyConfig
    ) -> List[TableInfo]:
        """
        提取HTML表格信息。

        Args:
            soup: BeautifulSoup对象
            config: 配置

        Returns:
            List[TableInfo]: 表格信息列表
        """
        tables = []
        table_tags = soup.find_all('table')

        for i, table in enumerate(table_tags):
            rows = table.find_all('tr')
            if not rows:
                continue

            # 计算行列数
            max_cols = max(len(row.find_all(['td', 'th'])) for row in rows) if rows else 0

            # 提取表头
            headers = []
            header_row = rows[0] if rows else None
            if header_row:
                headers = [th.get_text().strip() for th in header_row.find_all('th')]

            table_info = TableInfo(
                table_id=f"table_{i}",
                page_number=1,
                bbox=[0, 0, 0, 0],  # HTML中很难准确定位
                rows=len(rows),
                columns=max_cols,
                headers=headers,
                confidence=0.9,  # HTML表格通常结构清晰
                metadata={
                    "table_html": str(table)[:500] + "..." if len(str(table)) > 500 else str(table)
                }
            )
            tables.append(table_info)

        return tables

    def _count_html_elements(self, soup: BeautifulSoup) -> Dict[str, int]:
        """统计HTML元素数量。"""
        return {
            "headings": len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
            "paragraphs": len(soup.find_all('p')),
            "lists": len(soup.find_all(['ul', 'ol', 'dl'])),
            "links": len(soup.find_all('a', href=True)),
            "images": len(soup.find_all('img')),
            "tables": len(soup.find_all('table')),
            "forms": len(soup.find_all('form')),
            "divs": len(soup.find_all('div'))
        }

    def _extract_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """提取链接信息。"""
        links = []
        for a in soup.find_all('a', href=True):
            links.append({
                "text": a.get_text().strip(),
                "href": a['href'],
                "title": a.get('title', '')
            })
        return links

    def _extract_headings(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """提取标题信息。"""
        headings = []
        for level in range(1, 7):
            for h in soup.find_all(f'h{level}'):
                headings.append({
                    "level": level,
                    "text": h.get_text().strip(),
                    "id": h.get('id', '')
                })
        return headings

    async def _extract_markdown_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """提取Markdown代码块。"""
        code_blocks = []
        lines = content.split('\n')
        in_code_block = False
        current_block = []
        block_start = 0
        language = ""

        for i, line in enumerate(lines):
            if line.strip().startswith('```'):
                if not in_code_block:
                    # 开始代码块
                    in_code_block = True
                    block_start = i
                    language = line.strip()[3:].strip()
                    current_block = []
                else:
                    # 结束代码块
                    in_code_block = False
                    code_blocks.append({
                        "language": language or "text",
                        "code": '\n'.join(current_block),
                        "start_line": block_start,
                        "end_line": i
                    })
                    current_block = []
            elif in_code_block:
                current_block.append(line)

        return code_blocks

    async def _extract_markdown_links(self, content: str) -> List[Dict[str, str]]:
        """提取Markdown链接。"""
        links = []
        # 匹配 [text](url) 格式
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        matches = re.findall(link_pattern, content)

        for text, url in matches:
            links.append({
                "text": text,
                "url": url,
                "type": "markdown"
            })

        return links


# 导出
__all__ = ['WebDocumentProcessor']