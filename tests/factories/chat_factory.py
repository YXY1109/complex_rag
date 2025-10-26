"""
聊天数据工厂
生成测试用的聊天相关数据
"""
from factory import Factory, Faker, SubFactory, lazy_attribute, List
from typing import Dict, Any, List as ListType
import uuid
from datetime import datetime


class ChatSessionFactory(Factory):
    """聊天会话工厂"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    user_id = Faker("uuid4")
    title = Faker("sentence", nb_words=4)
    model = Faker("random_element", elements=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"])
    created_at = Faker("date_time_this_month")
    updated_at = Faker("date_time_this_month")
    status = "active"

    @lazy_attribute
    def settings(self):
        """生成会话设置"""
        return {
            "temperature": Faker("pyfloat", left_digits=0, right_digits=2, min=0, max=2),
            "max_tokens": Faker("random_int", min=100, max=4000),
            "top_p": Faker("pyfloat", left_digits=0, right_digits=2, min=0, max=1),
            "frequency_penalty": Faker("pyfloat", left_digits=0, right_digits=2, min=-2, max=2),
            "presence_penalty": Faker("pyfloat", left_digits=0, right_digits=2, min=-2, max=2),
            "stream": Faker("boolean"),
            "system_prompt": Faker("text", max_nb_chars=500)
        }

    @lazy_attribute
    def metadata(self):
        """生成会话元数据"""
        return {
            "total_messages": Faker("random_int", min=1, max=100),
            "total_tokens": Faker("random_int", min=100, max=50000),
            "estimated_cost": round(Faker("random.uniform", min=0.01, max=10.0), 4),
            "tags": Faker("words", nb=3),
            "category": Faker("random_element", elements=["general", "technical", "creative", "educational"]),
            "language": Faker("random_element", elements=["zh-CN", "en-US", "ja-JP"]),
            "is_pinned": Faker("boolean", truth_probability=0.1),
            "is_bookmarked": Faker("boolean", truth_probability=0.2)
        }

    @lazy_attribute
    def statistics(self):
        """生成统计信息"""
        return {
            "message_count": Faker("random_int", min=1, max=100),
            "user_message_count": Faker("random_int", min=1, max=50),
            "assistant_message_count": Faker("random_int", min=1, max=50),
            "average_response_time": Faker("random_int", min=500, max=10000),  # 毫秒
            "average_message_length": Faker("random_int", min=10, max=500),
            "total_response_time": Faker("random_int", min=5000, max=600000),  # 毫秒
            "last_activity": Faker("date_time_this_day")
        }


class ChatMessageFactory(Factory):
    """聊天消息工厂"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    session_id = SubFactory(ChatSessionFactory).id
    role = Faker("random_element", elements=["user", "assistant", "system"])
    content = Faker("text", max_nb_chars=1000)
    created_at = Faker("date_time_this_month")
    token_count = Faker("random_int", min=10, max=1000)

    @lazy_attribute
    def metadata(self):
        """生成消息元数据"""
        metadata = {
            "model": None,
            "finish_reason": None,
            "response_time": None,
            "citations": [],
            "context_chunks": [],
            "is_edited": Faker("boolean", truth_probability=0.1),
            "is_deleted": False,
            "edit_count": 0,
            "feedback_score": None
        }

        if self.role == "assistant":
            metadata.update({
                "model": Faker("random_element", elements=["gpt-3.5-turbo", "gpt-4"]),
                "finish_reason": Faker("random_element", elements=["stop", "length", "content_filter"]),
                "response_time": Faker("random_int", min=100, max=5000),  # 毫秒
                "citations": [Faker("uuid4") for _ in range(Faker("random_int", min=0, max=5))],
                "context_chunks": [Faker("uuid4") for _ in range(Faker("random_int", min=1, max=10))],
                "feedback_score": round(Faker("random.uniform", min=1, max=5), 1) if Faker("boolean", truth_probability=0.3) else None
            })

        return metadata

    @lazy_attribute
    def attachments(self):
        """生成附件信息"""
        if self.role == "user" and Faker("boolean", truth_probability=0.2):
            return [
                {
                    "type": Faker("random_element", elements=["image", "document", "file"]),
                    "name": Faker("file_name"),
                    "size": Faker("random_int", min=1024, max=10485760),
                    "url": Faker("url")
                } for _ in range(Faker("random_int", min=1, max=3))
            ]
        return []


class ChatTurnFactory(Factory):
    """聊天回合工厂（包含用户消息和助手回复）"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    session_id = SubFactory(ChatSessionFactory).id
    user_message = SubFactory(ChatMessageFactory, role="user")
    assistant_message = SubFactory(ChatMessageFactory, role="assistant")
    created_at = Faker("date_time_this_month")

    @lazy_attribute
    def context(self):
        """生成上下文信息"""
        return {
            "retrieved_documents": Faker("random_int", min=0, max=10),
            "knowledge_base_ids": [Faker("uuid4") for _ in range(Faker("random_int", min=0, max=3))],
            "search_query": Faker("text", max_nb_chars=100),
            "context_tokens": Faker("random_int", min=0, max=5000),
            "total_generation_time": Faker("random_int", min=100, max=10000),  # 毫秒
            "rerank_enabled": Faker("boolean"),
            "citation_count": len(self.assistant_message.metadata.get("citations", []))
        }

    @lazy_attribute
    def evaluation(self):
        """生成评估信息"""
        return {
            "relevance_score": round(Faker("random.uniform", min=0.5, max=1.0), 3),
            "accuracy_score": round(Faker("random.uniform", min=0.5, max=1.0), 3),
            "helpfulness_score": round(Faker("random.uniform", min=0.5, max=1.0), 3),
            "clarity_score": round(Faker("random.uniform", min=0.5, max=1.0), 3),
            "overall_score": round(Faker("random.uniform", min=0.5, max=1.0), 3),
            "evaluated_by": Faker("random_element", elements=["user", "system", "expert"]),
            "evaluated_at": Faker("date_time_this_month")
        }


class ChatTemplateFactory(Factory):
    """聊天模板工厂"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    name = Faker("sentence", nb_words=3)
    description = Faker("text", max_nb_chars=200)
    category = Faker("random_element", elements=["productivity", "education", "creative", "technical", "general"])
    creator_id = Faker("uuid4")
    is_public = Faker("boolean", truth_probability=0.3)
    is_featured = Faker("boolean", truth_probability=0.1)
    created_at = Faker("date_time_this_month")
    usage_count = Faker("random_int", min=0, max=10000)

    @lazy_attribute
    def system_prompt(self):
        """生成系统提示"""
        prompts = {
            "productivity": "你是一个高效的工作助手，帮助用户提高生产力。",
            "education": "你是一个耐心的教育工作者，善于解释复杂的概念。",
            "creative": "你是一个富有创造力的助手，能够提供创新的思路和想法。",
            "technical": "你是一个技术专家，能够提供专业的技术支持和解答。",
            "general": "你是一个友好、有帮助的AI助手。"
        }
        return prompts.get(self.category, "你是一个有帮助的AI助手。")

    @lazy_attribute
    def example_messages(self):
        """生成示例消息"""
        examples = {
            "productivity": [
                {"role": "user", "content": "帮我制定一个明天的计划"},
                {"role": "assistant", "content": "我来帮你制定一个高效的明日计划..."}
            ],
            "education": [
                {"role": "user", "content": "请解释一下量子力学的基本原理"},
                {"role": "assistant", "content": "让我用简单易懂的方式为你解释量子力学..."}
            ],
            "creative": [
                {"role": "user", "content": "给我一些创业的想法"},
                {"role": "assistant", "content": "这里有几个有趣的创业方向..."}
            ],
            "technical": [
                {"role": "user", "content": "如何优化Python代码的性能？"},
                {"role": "assistant", "content": "让我为你介绍几种优化Python代码性能的方法..."}
            ],
            "general": [
                {"role": "user", "content": "今天天气怎么样？"},
                {"role": "assistant", "content": "抱歉，我无法获取实时天气信息..."}
            ]
        }
        return examples.get(self.category, [])

    @lazy_attribute
    def settings(self):
        """生成模板设置"""
        return {
            "temperature": Faker("pyfloat", left_digits=0, right_digits=2, min=0, max=2),
            "max_tokens": Faker("random_int", min=500, max=2000),
            "top_p": Faker("pyfloat", left_digits=0, right_digits=2, min=0, max=1),
            "model": Faker("random_element", elements=["gpt-3.5-turbo", "gpt-4"]),
            "enable_context": Faker("boolean"),
            "enable_citations": Faker("boolean"),
            "default_knowledge_bases": [Faker("uuid4") for _ in range(Faker("random_int", min=0, max=3))]
        }

    @lazy_attribute
    def tags(self):
        """生成标签"""
        return Faker("words", nb=5)


class ChatFeedbackFactory(Factory):
    """聊天反馈工厂"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    session_id = SubFactory(ChatSessionFactory).id
    message_id = SubFactory(ChatMessageFactory).id
    user_id = Faker("uuid4")
    rating = Faker("random_int", min=1, max=5)
    comment = Faker("text", max_nb_chars=300)
    created_at = Faker("date_time_this_month")

    @lazy_attribute
    def feedback_type(self):
        """生成反馈类型"""
        if self.rating >= 4:
            return "positive"
        elif self.rating <= 2:
            return "negative"
        else:
            return "neutral"

    @lazy_attribute
    def categories(self):
        """生成反馈分类"""
        categories = ["relevance", "accuracy", "helpfulness", "clarity", "completeness"]
        selected = Faker("random_elements", elements=categories, length=Faker("random_int", min=1, max=3))
        return selected

    @lazy_attribute
    def metadata(self):
        """生成反馈元数据"""
        return {
            "response_time": Faker("random_int", min=100, max=5000),  # 毫秒
            "model_used": Faker("random_element", elements=["gpt-3.5-turbo", "gpt-4"]),
            "context_used": Faker("boolean"),
            "citation_count": Faker("random_int", min=0, max=5),
            "session_position": Faker("random_int", min=1, max=20),
            "user_type": Faker("random_element", elements=["new", "regular", "premium"]),
            "platform": Faker("random_element", elements=["web", "mobile", "api"])
        }


# 便捷函数
def create_test_chat_session(**overrides) -> Dict[str, Any]:
    """创建测试聊天会话"""
    return ChatSessionFactory(**overrides)


def create_test_chat_message(session_id: str, role: str = "user", **overrides) -> Dict[str, Any]:
    """创建测试聊天消息"""
    return ChatMessageFactory(session_id=session_id, role=role, **overrides)


def create_test_chat_turn(session_id: str, **overrides) -> Dict[str, Any]:
    """创建测试聊天回合"""
    return ChatTurnFactory(session_id=session_id, **overrides)


def create_test_chat_template(**overrides) -> Dict[str, Any]:
    """创建测试聊天模板"""
    return ChatTemplateFactory(**overrides)


def create_test_chat_feedback(session_id: str, message_id: str, user_id: str, **overrides) -> Dict[str, Any]:
    """创建测试聊天反馈"""
    return ChatFeedbackFactory(
        session_id=session_id,
        message_id=message_id,
        user_id=user_id,
        **overrides
    )


def create_test_conversation(turns_count: int = 3, **overrides) -> Dict[str, Any]:
    """创建测试完整对话"""
    session = ChatSessionFactory(**overrides)
    turns = []

    for i in range(turns_count):
        turn = ChatTurnFactory(
            session_id=session["id"],
            user_message=ChatMessageFactory(
                session_id=session["id"],
                role="user",
                created_at=Faker("date_time_between",
                               start_date=session["created_at"],
                               end_date=session["updated_at"])
            ),
            assistant_message=ChatMessageFactory(
                session_id=session["id"],
                role="assistant",
                created_at=Faker("date_time_between",
                               start_date=session["created_at"],
                               end_date=session["updated_at"])
            )
        )
        turns.append(turn)

    return {
        "session": session,
        "turns": turns,
        "total_turns": len(turns),
        "total_messages": len(turns) * 2 + 1  # +1 for system message
    }