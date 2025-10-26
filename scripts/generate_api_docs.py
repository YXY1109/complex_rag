"""
API文档生成脚本
生成API路由的详细文档和使用示例
"""
import json
from pathlib import Path
from typing import Dict, Any, List

# 添加项目根目录到Python路径
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class APIDocGenerator:
    """API文档生成器"""

    def __init__(self):
        self.api_docs = {
            "title": "Complex RAG API Documentation",
            "version": "1.0.0",
            "description": "高性能RAG系统API服务，提供智能问答和文档检索功能",
            "base_url": "http://localhost:8000",
            "routes": []
        }

    def generate_docs(self):
        """生成API文档"""
        print("生成API文档...")

        # 添加各个路由模块的文档
        self.add_health_docs()
        self.add_chat_docs()
        self.add_documents_docs()
        self.add_knowledge_docs()
        self.add_models_docs()
        self.add_users_docs()
        self.add_system_docs()
        self.add_analytics_docs()

        # 保存文档
        self.save_docs()

        print("API文档生成完成!")

    def add_health_docs(self):
        """添加健康检查API文档"""
        routes = [
            {
                "path": "/api/health/",
                "method": "GET",
                "summary": "系统健康检查",
                "description": "检查系统整体健康状态，包括API服务、数据库、缓存等关键组件",
                "parameters": [],
                "responses": {
                    "200": {
                        "description": "健康检查成功",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string", "example": "healthy"},
                                "timestamp": {"type": "string", "example": "2024-01-01T10:00:00Z"},
                                "version": {"type": "string", "example": "1.0.0"},
                                "uptime": {"type": "string", "example": "2 hours 30 minutes"},
                                "services": {"type": "object"}
                            }
                        }
                    }
                },
                "example": {
                    "curl": "curl -X GET http://localhost:8000/api/health/",
                    "python": "import requests; response = requests.get('http://localhost:8000/api/health/')"
                }
            },
            {
                "path": "/api/health/detailed",
                "method": "GET",
                "summary": "详细健康检查",
                "description": "获取详细的系统健康检查信息，包括性能指标",
                "responses": {
                    "200": {
                        "description": "详细健康检查成功",
                        "schema": {"type": "object"}
                    }
                }
            },
            {
                "path": "/api/health/ping",
                "method": "GET",
                "summary": "Ping检查",
                "description": "简单的ping检查，用于快速验证服务可用性",
                "responses": {
                    "200": {
                        "description": "Ping成功",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string", "example": "ok"},
                                "message": {"type": "string", "example": "pong"}
                            }
                        }
                    }
                }
            }
        ]

        self.api_docs["routes"].extend(routes)

    def add_chat_docs(self):
        """添加对话API文档"""
        routes = [
            {
                "path": "/api/chat/completions",
                "method": "POST",
                "summary": "对话完成",
                "description": "兼容OpenAI的对话接口，生成智能回复",
                "parameters": [
                    {
                        "name": "messages",
                        "type": "array",
                        "required": True,
                        "description": "对话消息列表",
                        "example": [{"role": "user", "content": "你好"}]
                    },
                    {
                        "name": "model",
                        "type": "string",
                        "required": False,
                        "description": "使用的模型名称",
                        "example": "gpt-3.5-turbo"
                    },
                    {
                        "name": "stream",
                        "type": "boolean",
                        "required": False,
                        "description": "是否流式返回",
                        "example": False
                    }
                ],
                "responses": {
                    "200": {
                        "description": "对话生成成功",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "object": {"type": "string", "example": "chat.completion"},
                                "created": {"type": "integer"},
                                "model": {"type": "string"},
                                "choices": {"type": "array"},
                                "usage": {"type": "object"}
                            }
                        }
                    }
                },
                "example": {
                    "curl": '''
curl -X POST http://localhost:8000/api/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "model": "gpt-3.5-turbo"
  }'
                    ''',
                    "python": '''
import requests

response = requests.post(
    "http://localhost:8000/api/chat/completions",
    json={
        "messages": [{"role": "user", "content": "你好"}],
        "model": "gpt-3.5-turbo"
    }
)
                    '''
                }
            },
            {
                "path": "/api/chat/completions/stream",
                "method": "POST",
                "summary": "流式对话完成",
                "description": "流式生成对话回复，实时返回生成内容",
                "parameters": [
                    {
                        "name": "messages",
                        "type": "array",
                        "required": True,
                        "description": "对话消息列表"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "流式对话生成",
                        "content_type": "text/event-stream"
                    }
                }
            }
        ]

        self.api_docs["routes"].extend(routes)

    def add_documents_docs(self):
        """添加文档管理API文档"""
        routes = [
            {
                "path": "/api/documents/",
                "method": "GET",
                "summary": "获取文档列表",
                "description": "分页获取文档列表，支持过滤",
                "parameters": [
                    {
                        "name": "page",
                        "type": "integer",
                        "required": False,
                        "description": "页码",
                        "example": 1
                    },
                    {
                        "name": "page_size",
                        "type": "integer",
                        "required": False,
                        "description": "每页数量",
                        "example": 20
                    }
                ],
                "responses": {
                    "200": {
                        "description": "文档列表获取成功",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "documents": {"type": "array"},
                                "total": {"type": "integer"},
                                "page": {"type": "integer"},
                                "total_pages": {"type": "integer"}
                            }
                        }
                    }
                }
            },
            {
                "path": "/api/documents/upload",
                "method": "POST",
                "summary": "上传文档",
                "description": "上传文档到指定知识库",
                "parameters": [
                    {
                        "name": "file",
                        "type": "file",
                        "required": True,
                        "description": "上传的文件"
                    },
                    {
                        "name": "knowledge_base_id",
                        "type": "string",
                        "required": True,
                        "description": "知识库ID"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "文档上传成功",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "success": {"type": "boolean"},
                                "document_id": {"type": "string"},
                                "filename": {"type": "string"},
                                "status": {"type": "string"}
                            }
                        }
                    }
                }
            }
        ]

        self.api_docs["routes"].extend(routes)

    def add_knowledge_docs(self):
        """添加知识库管理API文档"""
        routes = [
            {
                "path": "/api/knowledge/",
                "method": "POST",
                "summary": "创建知识库",
                "description": "创建新的知识库",
                "parameters": [
                    {
                        "name": "name",
                        "type": "string",
                        "required": True,
                        "description": "知识库名称"
                    },
                    {
                        "name": "description",
                        "type": "string",
                        "required": False,
                        "description": "知识库描述"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "知识库创建成功",
                        "schema": {"type": "object"}
                    }
                }
            },
            {
                "path": "/api/knowledge/",
                "method": "GET",
                "summary": "获取知识库列表",
                "description": "获取知识库列表，支持分页",
                "responses": {
                    "200": {
                        "description": "知识库列表获取成功",
                        "schema": {"type": "object"}
                    }
                }
            },
            {
                "path": "/api/knowledge/search",
                "method": "POST",
                "summary": "搜索知识库",
                "description": "在知识库中搜索相关内容",
                "parameters": [
                    {
                        "name": "query",
                        "type": "string",
                        "required": True,
                        "description": "搜索查询"
                    },
                    {
                        "name": "top_k",
                        "type": "integer",
                        "required": False,
                        "description": "返回结果数量",
                        "example": 10
                    }
                ],
                "responses": {
                    "200": {
                        "description": "搜索成功",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "results": {"type": "array"},
                                "total": {"type": "integer"},
                                "search_time": {"type": "number"}
                            }
                        }
                    }
                }
            }
        ]

        self.api_docs["routes"].extend(routes)

    def add_models_docs(self):
        """添加模型管理API文档"""
        routes = [
            {
                "path": "/api/models/",
                "method": "GET",
                "summary": "获取模型列表",
                "description": "获取可用的AI模型列表",
                "parameters": [
                    {
                        "name": "type",
                        "type": "string",
                        "required": False,
                        "description": "模型类型过滤",
                        "example": "llm"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "模型列表获取成功",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "models": {"type": "array"},
                                "total": {"type": "integer"},
                                "page": {"type": "integer"}
                            }
                        }
                    }
                }
            },
            {
                "path": "/api/models/{model_id}/test",
                "method": "POST",
                "summary": "测试模型",
                "description": "测试指定模型的功能",
                "parameters": [
                    {
                        "name": "model_id",
                        "type": "string",
                        "required": True,
                        "description": "模型ID"
                    },
                    {
                        "name": "input",
                        "type": "string",
                        "required": True,
                        "description": "测试输入"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "模型测试完成",
                        "schema": {"type": "object"}
                    }
                }
            }
        ]

        self.api_docs["routes"].extend(routes)

    def add_users_docs(self):
        """添加用户管理API文档"""
        routes = [
            {
                "path": "/api/users/me",
                "method": "GET",
                "summary": "获取当前用户信息",
                "description": "获取当前用户的基本信息",
                "responses": {
                    "200": {
                        "description": "用户信息获取成功",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "username": {"type": "string"},
                                "email": {"type": "string"},
                                "display_name": {"type": "string"},
                                "status": {"type": "string"}
                            }
                        }
                    }
                }
            },
            {
                "path": "/api/users/me/sessions",
                "method": "GET",
                "summary": "获取用户会话列表",
                "description": "获取当前用户的会话列表",
                "responses": {
                    "200": {
                        "description": "会话列表获取成功",
                        "schema": {
                            "type": "array",
                            "items": {"type": "object"}
                        }
                    }
                }
            },
            {
                "path": "/api/users/me/stats",
                "method": "GET",
                "summary": "获取用户统计信息",
                "description": "获取当前用户的统计信息",
                "responses": {
                    "200": {
                        "description": "统计信息获取成功",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "total_conversations": {"type": "integer"},
                                "total_messages": {"type": "integer"},
                                "total_documents": {"type": "integer"}
                            }
                        }
                    }
                }
            }
        ]

        self.api_docs["routes"].extend(routes)

    def add_system_docs(self):
        """添加系统管理API文档"""
        routes = [
            {
                "path": "/api/system/info",
                "method": "GET",
                "summary": "获取系统信息",
                "description": "获取系统基本信息和状态",
                "responses": {
                    "200": {
                        "description": "系统信息获取成功",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "version": {"type": "string"},
                                "environment": {"type": "string"},
                                "uptime": {"type": "string"},
                                "services_status": {"type": "object"}
                            }
                        }
                    }
                }
            },
            {
                "path": "/api/system/config",
                "method": "GET",
                "summary": "获取系统配置",
                "description": "获取当前系统配置（敏感信息已隐藏）",
                "responses": {
                    "200": {
                        "description": "系统配置获取成功",
                        "schema": {"type": "object"}
                    }
                }
            },
            {
                "path": "/api/system/metrics",
                "method": "GET",
                "summary": "获取系统指标",
                "description": "获取系统性能指标",
                "parameters": [
                    {
                        "name": "time_range",
                        "type": "string",
                        "required": False,
                        "description": "时间范围",
                        "example": "1h"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "系统指标获取成功",
                        "schema": {"type": "object"}
                    }
                }
            }
        ]

        self.api_docs["routes"].extend(routes)

    def add_analytics_docs(self):
        """添加统计分析API文档"""
        routes = [
            {
                "path": "/api/analytics/dashboard",
                "method": "GET",
                "summary": "获取仪表板数据",
                "description": "获取仪表板概览数据",
                "parameters": [
                    {
                        "name": "time_range",
                        "type": "string",
                        "required": False,
                        "description": "时间范围",
                        "example": "7d"
                    }
                ],
                "responses": {
                    "200": {
                        "description": "仪表板数据获取成功",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "overview": {"type": "object"},
                                "usage_metrics": {"type": "array"},
                                "performance_metrics": {"type": "array"}
                            }
                        }
                    }
                }
            },
            {
                "path": "/api/analytics/usage/overview",
                "method": "GET",
                "summary": "获取使用情况概览",
                "description": "获取系统使用情况概览",
                "responses": {
                    "200": {
                        "description": "使用情况概览获取成功",
                        "schema": {"type": "object"}
                    }
                }
            }
        ]

        self.api_docs["routes"].extend(routes)

    def save_docs(self):
        """保存API文档"""
        docs_dir = project_root / "docs" / "api"
        docs_dir.mkdir(parents=True, exist_ok=True)

        # 保存JSON格式的完整文档
        json_file = docs_dir / "api_documentation.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(self.api_docs, f, ensure_ascii=False, indent=2)

        # 生成Markdown格式的文档
        self.generate_markdown_docs(docs_dir)

        print(f"API文档已保存到: {docs_dir}")
        print(f"- JSON文档: {json_file}")
        print(f"- Markdown文档: {docs_dir / 'README.md'}")

    def generate_markdown_docs(self, docs_dir: Path):
        """生成Markdown格式的API文档"""
        md_content = f"""# {self.api_docs['title']}

**版本:** {self.api_docs['version']}
**基础URL:** {self.api_docs['base_url']}

{self.api_docs['description']}

## 目录

"""

        # 按类别组织路由
        categories = {}
        for route in self.api_docs["routes"]:
            category = self.get_route_category(route["path"])
            if category not in categories:
                categories[category] = []
            categories[category].append(route)

        # 生成目录
        for category, routes in categories.items():
            md_content += f"- [{category}](#{category.lower().replace(' ', '-')})\n"
            for route in routes:
                anchor = route["summary"].lower().replace(" ", "-").replace("(", "").replace(")", "")
                md_content += f"  - [{route['summary']}](#{anchor})\n"

        md_content += "\n---\n\n"

        # 生成详细文档
        for category, routes in categories.items():
            md_content += f"## {category}\n\n"

            for route in routes:
                md_content += f"### {route['summary']}\n\n"
                md_content += f"**路径:** `{route['method']} {route['path']}`\n\n"
                md_content += f"**描述:** {route['description']}\n\n"

                if route.get("parameters"):
                    md_content += "**参数:**\n\n"
                    for param in route["parameters"]:
                        required = "✓" if param.get("required", False) else "✗"
                        md_content += f"- `{param['name']}` ({param['type']}) - {required} {param['description']}"
                        if param.get("example"):
                            md_content += f" - 示例: `{param['example']}`"
                        md_content += "\n"
                    md_content += "\n"

                if route.get("example"):
                    md_content += "**使用示例:**\n\n"
                    if "curl" in route["example"]:
                        md_content += "```bash\n"
                        md_content += route["example"]["curl"]
                        md_content += "\n```\n\n"

                    if "python" in route["example"]:
                        md_content += "```python\n"
                        md_content += route["example"]["python"]
                        md_content += "\n```\n\n"

                md_content += "---\n\n"

        # 保存Markdown文档
        md_file = docs_dir / "README.md"
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(md_content)

    def get_route_category(self, path: str) -> str:
        """根据路径获取路由类别"""
        if "/health" in path:
            return "健康检查"
        elif "/chat" in path:
            return "对话服务"
        elif "/documents" in path:
            return "文档管理"
        elif "/knowledge" in path:
            return "知识库管理"
        elif "/models" in path:
            return "模型管理"
        elif "/users" in path:
            return "用户管理"
        elif "/system" in path:
            return "系统管理"
        elif "/analytics" in path:
            return "统计分析"
        else:
            return "其他"


def main():
    """主函数"""
    print("生成API文档...")

    generator = APIDocGenerator()
    generator.generate_docs()


if __name__ == "__main__":
    main()