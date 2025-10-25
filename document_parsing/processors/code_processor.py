"""
代码仓库处理器

此模块实现GitHub、代码文件、技术文档等代码仓库的专用处理器，
参考RAGFlow中的代码处理和文件分析逻辑。
"""

import asyncio
import os
import re
import tempfile
import ast
import zipfile
import tarfile
import mimetypes
from typing import Dict, Any, List, Optional, Union, BinaryIO, Iterator
from pathlib import Path
import uuid
import subprocess
import json

from .base_processor import BaseProcessor
from ..interfaces.parser_interface import (
    ParseResult,
    DocumentMetadata,
    DocumentType,
    ProcessingStrategy,
    TextChunk,
    ImageInfo,
    TableInfo,
    ParseException,
    UnsupportedFormatError
)
from ..strategy_config import ProcessingStrategyConfig


class CodeRepositoryProcessor(BaseProcessor):
    """
    代码仓库处理器。

    专门处理GitHub仓库、代码文件、技术文档等代码相关内容。
    """

    def __init__(self, config):
        """
        初始化代码仓库处理器。

        Args:
            config: 解析器配置
        """
        super().__init__(config)
        self.supported_extensions = {
            # 编程语言源码
            '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rs',
            '.php', '.rb', '.swift', '.kt', '.scala', '.r', '.dart',
            # 配置文件
            '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
            '.xml', '.plist', '.gradle', '.mvn', '.npmrc', '.gitignore',
            # 文档文件
            '.md', '.rst', '.txt', '.tex', '.adoc', '.org', '.dox',
            # 其他
            '.sql', '.sh', '.bat', '.ps1', '.zsh', '.fish'
        }

    async def initialize(self) -> bool:
        """
        初始化处理器。

        Returns:
            bool: 初始化是否成功
        """
        return True  # 代码仓库处理器不需要额外依赖

    async def cleanup(self) -> None:
        """清理处理器资源。"""
        pass

    async def _parse_with_config(
        self,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig,
        **kwargs
    ) -> ParseResult:
        """
        使用配置解析代码仓库。

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

        # 确定文件类型
        file_ext = Path(file_path).suffix.lower()
        file_name = os.path.basename(file_path)

        try:
            # 如果是压缩包，先解压
            if file_ext in {'.zip', '.tar', '.tar.gz', '.tgz'}:
                return await self._parse_archive_file(file_path, strategy, config)
            else:
                return await self._parse_code_file(file_path, strategy, config)

        except Exception as e:
            raise ParseException(
                f"代码文件解析失败: {str(e)}",
                parser=self.parser_name,
                file_path=file_path
            )

    async def _parse_archive_file(
        self,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig
    ) -> ParseResult:
        """
        解析压缩包文件。

        Args:
            file_path: 文件路径
            strategy: 处理策略
            config: 配置

        Returns:
            ParseResult: 解析结果
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # 解压文件
            if file_path.endswith('.zip'):
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)
            elif file_path.endswith(('.tar', '.tar.gz', '.tgz')):
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(temp_dir)

            # 递归解析解压后的文件
            all_results = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path_in_archive = os.path.join(root, file)
                    if self._is_code_file(file_path_in_archive):
                        try:
                            result = await self._parse_code_file(
                                file_path_in_archive, strategy, config
                            )
                            all_results.append(result)
                        except Exception as e:
                            print(f"跳过文件 {file_path_in_archive}: {e}")

            # 合并所有结果
            return await self._merge_archive_results(all_results, file_path, config)

    async def _parse_code_file(
        self,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig
    ) -> ParseResult:
        """
        解析代码文件。

        Args:
            file_path: 文件路径
            strategy: 处理策略
            config: 配置

        Returns:
            ParseResult: 解析结果
        """
        file_ext = Path(file_path).suffix.lower()
        file_name = os.path.basename(file_path)

        try:
            # 读取文件内容
            encoding = await self._detect_file_encoding(file_path)
            with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()

            if not content.strip():
                raise ParseException("文件内容为空", parser=self.parser_name, file_path=file_path)

            # 创建元数据
            file_size = os.path.getsize(file_path)
            metadata = await self._extract_code_metadata(file_path, file_size, encoding)

            # 根据文件类型进行特定处理
            if file_ext == '.py':
                result = await self._parse_python_file(content, file_path, strategy, config, metadata)
            elif file_ext in {'.js', '.ts', '.jsx', '.tsx'}:
                result = await self._parse_javascript_file(content, file_path, strategy, config, metadata)
            elif file_ext in {'.java', '.kt', '.scala', '.cs', '.rs', '.go', '.cpp', '.c', '.h', '.hpp'}:
                result = await self._parse_programming_file(content, file_path, strategy, config, metadata)
            elif file_ext in {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf'}:
                result = await self._parse_config_file(content, file_path, strategy, config, metadata)
            elif file_ext in {'.md', '.rst', '.tex', '.adoc', '.org'}:
                result = await self._parse_documentation_file(content, file_path, strategy, config, metadata)
            else:
                result = await self._parse_generic_code_file(content, file_path, strategy, config, metadata)

            return result

        except Exception as e:
            raise ParseException(f"代码文件解析失败: {str(e)}", parser=self.parser_name, file_path=file_path)

    def _is_code_file(self, file_path: str) -> bool:
        """检查是否为代码文件。"""
        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.supported_extensions

    async def _parse_python_file(
        self,
        content: str,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """
        解析Python文件。

        Args:
            content: 文件内容
            file_path: 文件路径
            strategy: 处理策略
            config: 配置
            metadata: 元数据

        Returns:
            ParseResult: 解析结果
        """
        try:
            # 使用AST解析Python代码
            tree = ast.parse(content)

            # 提取代码结构
            code_structure = self._extract_python_structure(tree)

            # 提取文档字符串
            docstrings = self._extract_python_docstrings(tree)

            # 提取函数和类
            functions = code_structure.get('functions', [])
            classes = code_structure.get('classes', [])
            imports = code_structure.get('imports', [])

            # 创建文本块
            text_chunks = []
            chunk_size = config.text.chunk_size
            chunk_overlap = config.text.chunk_overlap

            if config.text.preserve_line_breaks:
                # 按代码块分块（函数、类级别）
                code_blocks = self._split_python_into_blocks(content)
                for i, block in enumerate(code_blocks):
                    if block.strip():
                        chunk = TextChunk(
                            content=block.strip(),
                            chunk_id=f"python_block_{i}",
                            confidence=1.0
                        )
                        text_chunks.append(chunk)
            else:
                # 按固定大小分块
                text_chunks = self._create_text_chunks(
                    content, chunk_size, chunk_overlap
                )

            # 更新元数据
            metadata.word_count = len(content.split())
            metadata.character_count = len(content)
            metadata.metadata.update({
                "functions_count": len(functions),
                "classes_count": len(classes),
                "imports_count": len(imports),
                "docstrings_count": len(docstrings)
            })

            return ParseResult(
                success=True,
                metadata=metadata,
                full_text=content,
                text_chunks=text_chunks,
                structured_data={
                    "language": "python",
                    "structure": code_structure,
                    "docstrings": docstrings
                }
            )

        except SyntaxError as e:
            raise ParseException(f"Python语法错误: {str(e)}", parser=self.parser_name, file_path=file_path)
        except Exception as e:
            raise ParseException(f"Python解析失败: {str(e)}", parser=self.parser_name, file_path=file_path)

    async def _parse_javascript_file(
        self,
        content: str,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """
        解析JavaScript/TypeScript文件。

        Args:
            content: 文件内容
            file_path: 文件路径
            strategy: 处理策略
            config: 配置
            metadata: 元数据

        Returns:
            ParseResult: 解析结果
        """
        # 提取JavaScript特性
        functions = self._extract_javascript_functions(content)
        classes = self._extract_javascript_classes(content)
        imports = self._extract_javascript_imports(content)

        # 提取注释
        comments = self._extract_javascript_comments(content)

        # 创建文本块
        text_chunks = []
        if config.text.preserve_line_breaks:
            # 按函数级别分块
            code_blocks = self._split_javascript_into_blocks(content)
            for i, block in enumerate(code_blocks):
                if block.strip():
                    chunk = TextChunk(
                        content=block.strip(),
                        chunk_id=f"js_block_{i}",
                        confidence=1.0
                    )
                    text_chunks.append(chunk)
        else:
            text_chunks = self._create_text_chunks(
                content,
                config.text.chunk_size,
                config.text.chunk_overlap
            )

        # 更新元数据
        metadata.word_count = len(content.split())
        metadata.character_count = len(content)
        metadata.metadata.update({
            "language": "javascript",
            "functions_count": len(functions),
            "classes_count": len(classes),
            "imports_count": len(imports),
            "comments_count": len(comments)
        })

        return ParseResult(
            success=True,
            metadata=metadata,
            full_text=content,
            text_chunks=text_chunks,
            structured_data={
                "language": "javascript",
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "comments": comments
            }
        )

    async def _parse_programming_file(
        self,
        content: str,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """
        解析通用编程语言文件。

        Args:
            content: 文件内容
            file_path: 文件路径
            strategy: 处理策略
            config: 配置
            metadata: 元数据

        Returns:
            ParseResult: 解析结果
        """
        # 检测编程语言
        language = self._detect_programming_language(file_path)

        # 提取编程语言特性
        functions = self._extract_functions_by_regex(content, language)
        classes = self._extract_classes_by_regex(content, language)
        comments = self._extract_comments_by_regex(content, language)

        # 创建文本块
        text_chunks = self._create_text_chunks(
            content,
            config.text.chunk_size,
            config.text.chunk_overlap
        )

        # 更新元数据
        metadata.word_count = len(content.split())
        metadata.character_count = len(content)
        metadata.metadata.update({
            "language": language,
            "functions_count": len(functions),
            "classes_count": len(classes),
            "comments_count": len(comments)
        })

        return ParseResult(
            success=True,
            metadata=metadata,
            full_text=content,
            text_chunks=text_chunks,
            structured_data={
                "language": language,
                "functions": functions,
                "classes": classes,
                "comments": comments
            }
        )

    async def _parse_config_file(
        self,
        content: str,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """
        解析配置文件。

        Args:
            content: 文件内容
            file_path: 文件路径
            strategy: 处理策略
            config: 配置
            metadata: 元数据

        Returns:
            ParseResult: 解析结果
        """
        file_ext = Path(file_path).suffix.lower()

        # 根据文件类型解析
        if file_ext == '.json':
            try:
                data = json.loads(content)
                return await self._parse_json_config(data, file_path, config, metadata)
            except json.JSONDecodeError:
                pass

        elif file_ext in {'.yaml', '.yml'}:
            try:
                import yaml
                data = yaml.safe_load(content)
                return await self._parse_yaml_config(data, file_path, config, metadata)
            except yaml.YAMLError:
                pass

        elif file_ext == '.toml':
            try:
                import toml
                data = toml.loads(content)
                return await self._parse_toml_config(data, file_path, config, metadata)
            except:
                pass

        # 通用配置文件处理
        return await self._parse_generic_config(content, file_path, config, metadata)

    async def _parse_documentation_file(
        self,
        content: str,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """
        解析文档文件。

        Args:
            content: 文件内容
            file_path: 文件路径
            strategy: 处理策略
            config: 配置
            metadata: 元数据

        Returns:
            ParseResult: 解析结果
        """
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.md':
            return await self._parse_markdown_file(content, file_path, config, metadata)
        elif file_ext == '.rst':
            return await self._parse_rst_file(content, file_path, config, metadata)
        elif file_ext == '.tex':
            return await self._parse_latex_file(content, file_path, config, metadata)
        else:
            # 通用文档处理
            return await self._parse_generic_documentation(content, file_path, config, metadata)

    async def _parse_json_config(
        self,
        data: Any,
        file_path: str,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """解析JSON配置文件。"""
        json_text = json.dumps(data, indent=2, ensure_ascii=False)

        text_chunks = []
        chunk = TextChunk(
            content=json_text,
            chunk_id="json_config_1",
            confidence=1.0
        )
        text_chunks.append(chunk)

        return ParseResult(
            success=True,
            metadata=metadata,
            full_text=json_text,
            text_chunks=text_chunks,
            structured_data=data
        )

    async def _parse_yaml_config(
        self,
        data: Any,
        file_path: str,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """解析YAML配置文件。"""
        try:
            import yaml
            yaml_text = yaml.dump(data, default_flow_style=False, allow_unicode=True)
        except ImportError:
            yaml_text = str(data)

        text_chunks = []
        chunk = TextChunk(
            content=yaml_text,
            chunk_id="yaml_config_1",
            confidence=1.0
        )
        text_chunks.append(chunk)

        return ParseResult(
            success=True,
            metadata=metadata,
            full_text=yaml_text,
            text_chunks=text_chunks,
            structured_data=data
        )

    async def _parse_toml_config(
        self,
        data: Any,
        file_path: str,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """解析TOML配置文件。"""
        try:
            import toml
            toml_text = toml.dumps(data)
        except ImportError:
            toml_text = str(data)

        text_chunks = []
        chunk = TextChunk(
            content=toml_text,
            chunk_id="toml_config_1",
            confidence=1.0
        )
        text_chunks.append(chunk)

        return ParseResult(
            success=True,
            metadata=metadata,
            full_text=toml_text,
            text_chunks=text_chunks,
            structured_data=data
        )

    async def _parse_generic_config(
        self,
        content: str,
        file_path: str,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """解析通用配置文件。"""
        text_chunks = self._create_text_chunks(
            content,
            config.text.chunk_size,
            config.text.chunk_overlap
        )

        return ParseResult(
            success=True,
            metadata=metadata,
            full_text=content,
            text_chunks=text_chunks
        )

    async def _parse_markdown_file(
        self,
        content: str,
        file_path: str,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """解析Markdown文件。"""
        # 提取标题
        headings = self._extract_markdown_headings(content)

        # 提取代码块
        code_blocks = self._extract_markdown_code_blocks(content)

        # 提取链接
        links = self._extract_markdown_links(content)

        text_chunks = self._create_text_chunks(
            content,
            config.text.chunk_size,
            config.text.chunk_overlap
        )

        return ParseResult(
            success=True,
            metadata=metadata,
            full_text=content,
            text_chunks=text_chunks,
            structured_data={
                "headings": headings,
                "code_blocks": code_blocks,
                "links": links
            }
        )

    async def _parse_rst_file(
        self,
        content: str,
        file_path: str,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """解析reStructuredText文件。"""
        # 提取reStructuredText结构
        sections = self._extract_rst_sections(content)

        text_chunks = self._create_text_chunks(
            content,
            config.text.chunk_size,
            config.text.chunk_overlap
        )

        return ParseResult(
            success=True,
            metadata=metadata,
            full_text=content,
            text_chunks=text_chunks,
            structured_data={
                "sections": sections
            }
        )

    async def _parse_latex_file(
        self,
        content: str,
        file_path: str,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """解析LaTeX文件。"""
        # 提取LaTeX命令
        commands = self._extract_latex_commands(content)

        text_chunks = self._create_text_chunks(
            content,
            config.text.chunk_size,
            config.text.chunk_overlap
        )

        return ParseResult(
            success=True,
            metadata=metadata,
            full_text=content,
            text_chunks=text_chunks,
            structured_data={
                "commands": commands
            }
        )

    async def _parse_generic_documentation(
        self,
        content: str,
        file_path: str,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """解析通用文档文件。"""
        text_chunks = self._create_text_chunks(
            content,
            config.text_chunk_size,
            config.text_chunk_overlap
        )

        return ParseResult(
            success=True,
            metadata=metadata,
            full_text=content,
            text_chunks=text_chunks
        )

    async def _parse_generic_code_file(
        self,
        content: str,
        file_path: str,
        strategy: ProcessingStrategy,
        config: ProcessingStrategyConfig,
        metadata: DocumentMetadata
    ) -> ParseResult:
        """解析通用代码文件。"""
        text_chunks = self._create_text_chunks(
            content,
            config.text.chunk_size,
            config.text_chunk_overlap
        )

        return ParseResult(
            success=True,
            metadata=metadata,
            full_text=content,
            text_chunks=text_chunks
        )

    def _extract_python_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """提取Python代码结构。"""
        structure = {
            'imports': [],
            'functions': [],
            'classes': [],
            'variables': []
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    structure['imports'].append({
                        'name': alias.name,
                        'module': node.module,
                        'alias': alias.asname
                    })
            elif isinstance(node, ast.FunctionDef):
                structure['functions'].append({
                    'name': node.name,
                    'line': node.lineno,
                    'args': [arg.arg for arg in ast.iter_child_nodes(node.args)],
                    'docstring': ast.get_docstring(node)
                })
            elif isinstance(node, ast.ClassDef):
                structure['classes'].append({
                    'name': node.name,
                    'line': node.lineno,
                    'bases': [base.id if isinstance(base, ast.Name) else base for base in node.bases],
                    'docstring': ast.get_docstring(node)
                })
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        structure['variables'].append({
                            'name': target.id,
                            'line': node.lineno
                        })

        return structure

    def _extract_python_docstrings(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """提取Python文档字符串。"""
        docstrings = []

        for node in ast.walk(tree):
            docstring = ast.get_docstring(node)
            if docstring:
                docstrings.append({
                    'type': type(node).__name__,
                    'name': getattr(node, 'name', ''),
                    'line': getattr(node, 'lineno', 0),
                    'docstring': docstring
                })

        return docstrings

    def _split_python_into_blocks(self, content: str) -> List[str]:
        """将Python代码分割为块（按函数和类）。"""
        lines = content.split('\n')
        blocks = []
        current_block = []
        indent_level = 0

        for line in lines:
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)

            if stripped.startswith('#') or not stripped:
                # 注释或空行，结束当前块
                if current_block:
                    blocks.append('\n'.join(current_block))
                    current_block = []
                if stripped:
                    blocks.append(stripped)
                continue

            if current_indent < indent_level and current_block:
                # 缩进减少，结束当前块
                blocks.append('\n'.join(current_block))
                current_block = []
                indent_level = current_indent

            current_indent = current_indent
            current_block.append(line)

        if current_block:
            blocks.append('\n'.join(current_block))

        return blocks

    def _extract_javascript_functions(self, content: str) -> List[Dict[str, Any]]:
        """提取JavaScript函数。"""
        function_pattern = r'(?:function\s+|const\s+\w+\s*=\s*|\w+\s*=\s*function)\s*(\w+)'
        functions = []

        for match in re.finditer(function_pattern, content):
            func_name = match.group(1) if len(match.groups()) > 1 else "anonymous"
            start_pos = match.start()
            functions.append({
                'name': func_name,
                'position': start_pos
            })

        return functions

    def _extract_javascript_classes(self, content: str) -> List[Dict[str, Any]]:
        """提取JavaScript类。"""
        class_pattern = r'(?:class\s+)(\w+)'
        classes = []

        for match in re.finditer(class_pattern, content):
            class_name = match.group(1)
            start_pos = match.start()
            classes.append({
                'name': class_name,
                'position': start_pos
            })

        return classes

    def _extract_javascript_imports(self, content: str) -> List[Dict[str, Any]]:
        """提取JavaScript导入语句。"""
        import_patterns = [
            r'import\s+.*?from\s+.*',
            r'require\s*\([^\'"]+)'
        ]

        imports = []
        for pattern in import_patterns:
            for match in re.finditer(pattern, content):
                imports.append({
                    'statement': match.group(),
                    'position': match.start()
                })

        return imports

    def _extract_javascript_comments(self, content: str) -> List[str]:
        """提取JavaScript注释。"""
        comment_patterns = [
            r'//.*$',  # 单行注释
            r'/\*[\s\S]*?\*/'  # 多行注释
        ]

        comments = []
        for pattern in comment_patterns:
            comments.extend(re.findall(pattern, content, re.MULTILINE))

        return comments

    def _split_javascript_into_blocks(self, content: str) -> List[str]:
        """将JavaScript代码分割为块（按函数级别）。"""
        blocks = []
        current_block = []
        brace_level = 0

        for line in content.split('\n'):
            stripped = line.strip()
            current_brace_level += line.count('{') - line.count('}')

            if stripped and not stripped.startswith('//') and not stripped.startswith('/*'):
                current_block.append(line)

            if current_brace_level == 0 and current_block:
                blocks.append('\n'.join(current_block))
                current_block = []

        if current_block:
            blocks.append('\n'.join(current_block))

        return blocks

    def _extract_functions_by_regex(
        self,
        content: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """使用正则表达式提取函数。"""
        patterns = {
            'python': [
                r'def\s+(\w+)\s*\(',
                r'async\s+def\s+(\w+)\s*\('
            ],
            'javascript': [
                r'function\s+(\w+)\s*\(',
                r'const\s+(\w+)\s*=',
                r'(\w+)\s*=\s*function'
            ],
            'java': [
                r'(?:public|private|protected|static|final|abstract|synchronized)?\s+.*?(\w+)\s*\(',
                r'interface\s+(\w+)\s*\{'
            ],
            'cpp': [
                r'(\w+)\s+(\w+)\s*\(',
                r'class\s+(\w+)\s*\{'
            ]
        }

        functions = []
        patterns_to_use = patterns.get(language, [])

        for pattern in patterns_to_use:
            for match in re.finditer(pattern, content):
                func_name = match.group(1)
                functions.append({
                    'name': func_name,
                    'position': match.start()
                })

        return functions

    def _extract_classes_by_regex(
        self,
        content: str,
        language: str
    ) -> List[Dict[str, Any]]:
        """使用正则表达式提取类。"""
        patterns = {
            'python': [r'class\s+(\w+)\s*'],
            'javascript': [r'class\s+(\w+)\s*\{'],
            'java': [r'(?:public|private|protected)?\s+class\s+(\w+)\s*(?:extends\s+\w+)?\s*\{'],
            'cpp': [r'class\s+(\w+)\s*'],
            'cs': [r'(?:public|private|internal)?\s*class\s+(\w+)\s*']
        }

        classes = []
        patterns_to_use = patterns.get(language, [])

        for pattern in patterns_to_use:
            for match in re.finditer(pattern, content):
                class_name = match.group(1)
                classes.append({
                    'name': class_name,
                    'position': match.start()
                })

        return classes

    def _extract_comments_by_regex(
        self,
        content: str,
        language: str
    ) -> List[str]:
        """使用正则表达式提取注释。"""
        patterns = {
            'python': [r'#.*$'],
            'javascript': [r'//.*$', r'/\*[\s\S]*?\*/'],
            'java': [r'//.*$', r'/\*[\s\S]*?\*/'],
            'cpp': [r'//.*$', r'/\*[\s\S]*?\*/'],
            'cs': [r'//.*$', r'/\*[\s\S]*?\*/']
        }

        comments = []
        patterns_to_use = patterns.get(language, [])

        for pattern in patterns_to_use:
            comments.extend(re.findall(pattern, content, re.MULTILINE))

        return comments

    def _extract_markdown_headings(self, content: str) -> List[Dict[str, Any]]:
        """提取Markdown标题。"""
        heading_pattern = r'^(#{1,6})\s+(.+)'
        headings = []

        for match in re.finditer(heading_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append({
                'level': level,
                'title': title,
                'position': match.start()
            })

        return headings

    def _extract_markdown_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """提取Markdown代码块。"""
        code_pattern = r'```(\w*)\n([\s\S]*?)```'
        code_blocks = []

        for match in re.finditer(code_pattern, content, re.MULTILINE):
            language = match.group(1)
            code = match.group(2)
            code_blocks.append({
                'language': language,
                'code': code,
                'position': match.start()
            })

        return code_blocks

    def _extract_markdown_links(self, content: str) -> List[Dict[str, str]]:
        """提取Markdown链接。"""
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        links = []

        for match in re.finditer(link_pattern, content):
            text = match.group(1)
            url = match.group(2)
            links.append({
                'text': text,
                'url': url,
                'position': match.start()
            })

        return links

    def _extract_rst_sections(self, content: str) -> List[Dict[str, Any]]:
        """提取reStructuredText章节。"""
        section_pattern = r'^([=+-]{2,})\n(.+)$'
        sections = []

        for match in re.finditer(section_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            sections.append({
                'level': level,
                'title': title,
                'position': match.start()
            })

        return sections

    def _extract_latex_commands(self, content: str) -> List[str]:
        """提取LaTeX命令。"""
        command_pattern = r'\\[a-zA-Z]+\{[^}]*\}'
        commands = re.findall(command_pattern, content)
        return commands

    def _detect_file_encoding(self, file_path: str) -> str:
        """检测文件编码。"""
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin1']

        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)  # 测试读取
                return encoding
            except UnicodeDecodeError:
                continue

        return 'utf-8'

    def _detect_programming_language(self, file_path: str) -> str:
        """检测编程语言。"""
        file_ext = Path(file_path).suffix.lower()

        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.cs': 'csharp',
            '.rs': 'rust',
            '.go': 'go',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.dart': 'dart',
            '.r': 'r',
            '.m': 'matlab'
        }

        return language_map.get(file_ext, 'unknown')

    async def _detect_file_encoding(self, file_path: str) -> str:
        """检测文件编码。"""
        encodings_to_try = ['utf-8', 'gbk', 'gb2312', 'latin1']

        for encoding in encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1000)  # 测试读取
                return encoding
            except UnicodeDecodeError:
                continue

        return 'utf-8'

    def _extract_code_metadata(
        self,
        file_path: str,
        file_size: int,
        encoding: str
    ) -> DocumentMetadata:
        """提取代码文件元数据。"""
        file_name = os.path.basename(file_path)
        file_ext = Path(file_path).suffix.lower()

        return DocumentMetadata(
            file_name=file_name,
            file_size=file_size,
            file_type=DocumentType.TXT,  # 代码文件通常标记为TXT
            mime_type=mimetypes.guess_type(file_path)[0] or 'text/plain',
            title=file_name,
            page_count=1,
            word_count=0,  # 将在解析后更新
            character_count=0,  # 将在解析后更新
            metadata={
                "encoding": encoding,
                "file_extension": file_ext,
                "language": self._detect_programming_language(file_path)
            }
        )

    async def _merge_archive_results(
        self,
        results: List[ParseResult],
        archive_path: str,
        config: ProcessingStrategyConfig
    ) -> ParseResult:
        """合并压缩包中所有文件的解析结果。"""
        if not results:
            raise ParseException("压缩包中没有可解析的文件", parser=self.parser_name, file_path=archive_path)

        # 合并所有文本内容
        combined_text = '\n\n'.join([
            f"=== {os.path.basename(r.metadata.file_name)} ===\n"
            f"类型: {r.metadata.file_type.value}\n"
            f"内容:\n{r.full_text}\n"
            for r in results if r.success and r.full_text
        ])

        # 合并文本块
        all_text_chunks = []
        chunk_id = 0
        for result in results:
            if result.success and result.text_chunks:
                for chunk in result.text_chunks:
                    # 更新chunk ID
                    chunk.chunk_id = f"{result.metadata.file_name}_{chunk.chunk_id}"
                    all_text_chunks.append(chunk)
                    chunk_id += 1

        # 创建合并后的元数据
        total_size = sum(r.metadata.file_size for r in results if r.success)
        archive_name = os.path.basename(archive_path)

        combined_metadata = DocumentMetadata(
            file_name=archive_name,
            file_size=total_size,
            file_type=DocumentType.TXT,
            mime_type="application/zip",
            title=archive_name,
            page_count=len(results),
            word_count=len(combined_text.split()),
            character_count=len(combined_text),
            metadata={
                "archive_files": [r.metadata.file_name for r in results],
                "successful_parses": len([r for r in results if r.success]),
                "failed_parses": len([r for r in results if not r.success])
            }
        )

        # 合并结构化数据
        combined_structured = {}
        for result in results:
            if result.success and result.structured_data:
                combined_structured[result.metadata.file_name] = result.structured_data

        return ParseResult(
            success=True,
            metadata=combined_metadata,
            full_text=combined_text,
            text_chunks=all_text_chunks,
            structured_data=combined_structured_data
        )


# 导出
__all__ = ['CodeRepositoryProcessor']