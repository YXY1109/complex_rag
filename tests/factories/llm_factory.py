"""
LLM响应数据工厂
生成测试用的大语言模型响应数据
"""
from factory import Factory, Faker, SubFactory, lazy_attribute, List
from typing import Dict, Any, List as ListType
import uuid


class LLMMessageFactory(Factory):
    """LLM消息工厂"""

    class Meta:
        model = dict

    role = Faker("random_element", elements=["system", "user", "assistant"])
    content = Faker("text", max_nb_chars=1000)

    @lazy_attribute
    def name(self):
        """消息名称（可选）"""
        if self.role == "assistant":
            return "assistant"
        return None

    @lazy_attribute
    def function_call(self):
        """函数调用（可选）"""
        if self.role == "assistant" and Faker("boolean", truth_probability=0.3):
            return {
                "name": Faker("word"),
                "arguments": Faker("json")
            }
        return None


class LLMChoiceFactory(Factory):
    """LLM选择项工厂"""

    class Meta:
        model = dict

    index = 0
    message = SubFactory(LLMMessageFactory)
    finish_reason = Faker("random_element", elements=["stop", "length", "content_filter", "function_call"])

    @lazy_attribute
    def logprobs(self):
        """对数概率（可选）"""
        if Faker("boolean", truth_probability=0.2):
            return {
                "token_logprobs": [Faker("pyfloat", left_digits=1, right_digits=6, negative=True) for _ in range(10)],
                "top_logprobs": {Faker("word"): Faker("pyfloat", left_digits=1, right_digits=6, negative=True) for _ in range(3)}
            }
        return None


class LLMUsageFactory(Factory):
    """LLM使用情况工厂"""

    class Meta:
        model = dict

    prompt_tokens = Faker("random_int", min=10, max=10000)
    completion_tokens = Faker("random_int", min=5, max=5000)
    total_tokens = Faker("random_int", min=15, max=15000)

    @lazy_attribute
    def prompt_tokens_details(self):
        """提示词详细信息"""
        return {
            "text_tokens": int(self.prompt_tokens * 0.9),
            "image_tokens": int(self.prompt_tokens * 0.1)
        }

    @lazy_attribute
    def completion_tokens_details(self):
        """完成词详细信息"""
        return {
            "text_tokens": int(self.completion_tokens * 0.95),
            "function_call_tokens": int(self.completion_tokens * 0.05)
        }


class LLMResponseFactory(Factory):
    """LLM响应工厂"""

    class Meta:
        model = dict

    id = Faker("uuid4")
    object = "chat.completion"
    created = Faker("random_int", min=1700000000, max=1800000000)
    model = Faker("random_element", elements=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"])
    choices = List([SubFactory(LLMChoiceFactory)])
    usage = SubFactory(LLMUsageFactory)
    system_fingerprint = Faker("sha256")

    @lazy_attribute
    def choices(self):
        """生成多个选择项"""
        choice_count = Faker("random_int", min=1, max=3)
        return [LLMChoiceFactory(index=i) for i in range(choice_count)]

    @lazy_attribute
    def service_tier(self):
        """服务层级"""
        return Faker("random_element", elements=["scale", "default"])


class StreamingLLMResponseFactory(LLMResponseFactory):
    """流式LLM响应工厂"""

    object = "chat.completion.chunk"

    @lazy_attribute
    def choices(self):
        """流式响应的选择项"""
        return [LLMChoiceFactory(
            index=0,
            finish_reason=Faker("random_element", elements=["stop", "length", "content_filter", "function_call", None])
        )]


class LLMRequestFactory(Factory):
    """LLM请求工厂"""

    class Meta:
        model = dict

    model = Faker("random_element", elements=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview"])
    messages = List([SubFactory(LLMMessageFactory)])
    temperature = Faker("pyfloat", left_digits=0, right_digits=2, min=0, max=2)
    top_p = Faker("pyfloat", left_digits=0, right_digits=2, min=0, max=1)
    n = 1
    stream = False
    max_tokens = Faker("random_int", min=10, max=4000)

    @lazy_attribute
    def messages(self):
        """生成消息列表"""
        message_count = Faker("random_int", min=1, max=10)
        return [LLMMessageFactory() for _ in range(message_count)]

    @lazy_attribute
    def functions(self):
        """可用函数列表（可选）"""
        if Faker("boolean", truth_probability=0.3):
            return [
                {
                    "name": "get_weather",
                    "description": "获取天气信息",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "城市名称"},
                            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                        },
                        "required": ["location"]
                    }
                }
            ]
        return None

    @lazy_attribute
    def frequency_penalty(self):
        """频率惩罚"""
        return Faker("pyfloat", left_digits=0, right_digits=2, min=-2, max=2)

    @lazy_attribute
    def presence_penalty(self):
        """存在惩罚"""
        return Faker("pyfloat", left_digits=0, right_digits=2, min=-2, max=2)


class LLMErrorFactory(Factory):
    """LLM错误响应工厂"""

    class Meta:
        model = dict

    error = {
        "message": Faker("sentence"),
        "type": Faker("random_element", elements=[
            "invalid_request_error",
            "invalid_api_key",
            "rate_limit_error",
            "insufficient_quota",
            "model_overloaded",
            "internal_server_error"
        ]),
        "param": Faker("word"),
        "code": None
    }

    @lazy_attribute
    def error(self):
        """生成错误信息"""
        error_type = Faker("random_element", elements=[
            "invalid_request_error",
            "invalid_api_key",
            "rate_limit_error",
            "insufficient_quota",
            "model_overloaded",
            "internal_server_error"
        ])

        error_messages = {
            "invalid_request_error": "请求格式不正确或缺少必要参数",
            "invalid_api_key": "API密钥无效或已过期",
            "rate_limit_error": "请求频率超限，请稍后重试",
            "insufficient_quota": "账户余额不足",
            "model_overloaded": "模型服务暂时不可用",
            "internal_server_error": "服务器内部错误"
        }

        error = {
            "message": error_messages.get(error_type, "未知错误"),
            "type": error_type,
            "param": Faker("word") if error_type == "invalid_request_error" else None,
            "code": Faker("random_int", min=400, max=599) if error_type in ["invalid_request_error", "invalid_api_key"] else None
        }
        return error


# 便捷函数
def create_test_llm_response(**overrides) -> Dict[str, Any]:
    """创建测试LLM响应"""
    return LLMResponseFactory(**overrides)


def create_test_streaming_response(**overrides) -> Dict[str, Any]:
    """创建测试流式响应"""
    return StreamingLLMResponseFactory(**overrides)


def create_test_llm_request(**overrides) -> Dict[str, Any]:
    """创建测试LLM请求"""
    return LLMRequestFactory(**overrides)


def create_test_llm_error(**overrides) -> Dict[str, Any]:
    """创建测试LLM错误"""
    return LLMErrorFactory(**overrides)


def create_test_conversation() -> Dict[str, Any]:
    """创建测试对话"""
    messages = [
        LLMMessageFactory(role="system", content="你是一个有帮助的AI助手。"),
        LLMMessageFactory(role="user", content="你好，请介绍一下你自己。"),
        LLMMessageFactory(role="assistant", content="你好！我是一个AI助手，可以帮助你回答问题、提供建议和进行各种对话。有什么我可以帮助你的吗？")
    ]

    return {
        "id": str(uuid.uuid4()),
        "title": "自我介绍对话",
        "messages": messages,
        "created_at": Faker("date_time_this_month"),
        "updated_at": Faker("date_time_this_month"),
        "user_id": str(uuid.uuid4()),
        "model": "gpt-3.5-turbo",
        "status": "active"
    }