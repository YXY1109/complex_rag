"""
用户数据工厂
生成测试用的用户数据
"""
from factory import Factory, Faker, SubFactory, SelfAttribute, lazy_attribute
from factory.alchemy import SQLAlchemyModelFactory
from datetime import datetime
from typing import Dict, Any, Optional


class UserDataFactory(Factory):
    """用户数据工厂"""

    class Meta:
        model = dict

    id = Faker("random_int", min=1, max=10000)
    username = Faker("user_name")
    email = Faker("email")
    full_name = Faker("name")
    avatar_url = Faker("url")
    is_active = True
    is_superuser = False
    created_at = Faker("date_time_this_year")
    updated_at = Faker("date_time_this_month")
    last_login = Faker("date_time_this_month")

    @lazy_attribute
    def password_hash(self):
        """生成密码哈希"""
        return f"hash_{self.username}_{self.id}"

    @lazy_attribute
    def preferences(self):
        """生成用户偏好设置"""
        return {
            "language": Faker("random_element", elements=["zh-CN", "en-US"]),
            "theme": Faker("random_element", elements=["light", "dark"]),
            "timezone": Faker("timezone"),
            "notifications": {
                "email": Faker("boolean"),
                "push": Faker("boolean"),
                "sms": Faker("boolean")
            }
        }

    @lazy_attribute
    def profile(self):
        """生成用户档案"""
        return {
            "bio": Faker("text", max_nb_chars=200),
            "company": Faker("company"),
            "job_title": Faker("job"),
            "location": Faker("city") + ", " + Faker("country"),
            "website": Faker("url"),
            "phone": Faker("phone_number"),
            "birth_date": Faker("date_of_birth"),
            "gender": Faker("random_element", elements=["male", "female", "other", "prefer_not_to_say"])
        }


class UserSessionFactory(UserDataFactory):
    """用户会话数据工厂"""

    session_id = Faker("uuid4")
    token = Faker("uuid4")
    expires_at = Faker("date_time_between", start_date="+1d", end_date="+30d")
    ip_address = Faker("ipv4")
    user_agent = Faker("user_agent")

    @lazy_attribute
    def device_info(self):
        """生成设备信息"""
        return {
            "type": Faker("random_element", elements=["desktop", "mobile", "tablet"]),
            "os": Faker("random_element", elements=["Windows", "macOS", "Linux", "iOS", "Android"]),
            "browser": Faker("random_element", elements=["Chrome", "Firefox", "Safari", "Edge"]),
            "screen_resolution": f"{Faker('random_int', min=1024, max=3840)}x{Faker('random_int', min=768, max=2160)}"
        }


class UserActivityFactory(Factory):
    """用户活动数据工厂"""

    class Meta:
        model = dict

    user_id = SubFactory(UserDataFactory).id
    activity_type = Faker("random_element", elements=[
        "login", "logout", "document_upload", "chat_start", "chat_end",
        "knowledge_base_create", "search_query", "settings_change"
    ])
    activity_data = {}
    ip_address = Faker("ipv4")
    user_agent = Faker("user_agent")
    timestamp = Faker("date_time_this_month")
    session_id = Faker("uuid4")

    @lazy_attribute
    def activity_data(self):
        """根据活动类型生成相应的数据"""
        data_templates = {
            "login": {"method": "password", "success": True},
            "logout": {"reason": "manual"},
            "document_upload": {"document_count": Faker("random_int", min=1, max=10), "total_size": Faker("random_int", min=1024, max=10485760)},
            "chat_start": {"model": "gpt-3.5-turbo", "conversation_id": Faker("uuid4")},
            "chat_end": {"message_count": Faker("random_int", min=1, max=50), "duration": Faker("random_int", min=60, max=3600)},
            "knowledge_base_create": {"name": Faker("sentence"), "type": "general"},
            "search_query": {"query": Faker("sentence"), "results_count": Faker("random_int", min=0, max=100)},
            "settings_change": {"changed_fields": [Faker("word") for _ in range(Faker("random_int", min=1, max=5))]}
        }
        return data_templates.get(self.activity_type, {})


# 便捷函数
def create_test_user(**overrides) -> Dict[str, Any]:
    """创建测试用户"""
    return UserDataFactory(**overrides)


def create_test_users(count: int, **overrides) -> list:
    """创建多个测试用户"""
    return [UserDataFactory(**overrides) for _ in range(count)]


def create_test_session(**overrides) -> Dict[str, Any]:
    """创建测试会话"""
    return UserSessionFactory(**overrides)


def create_test_activity(**overrides) -> Dict[str, Any]:
    """创建测试活动"""
    return UserActivityFactory(**overrides)