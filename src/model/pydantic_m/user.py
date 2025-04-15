from enum import Enum

from pydantic import BaseModel, Field, field_validator


class UserStatus(Enum):
    DISABLED = 0  # 禁用
    ENABLED = 1  # 启用


class UserParam(BaseModel):
    user_name: str = Field(default="yxy", description="用户名称", min_length=3, max_length=20)
    user_status: int = Field(default=UserStatus.ENABLED.value, description="用户状态")

    @field_validator("user_status")
    def validate_user_status(cls, value):
        valid_values = [status.value for status in UserStatus]
        if value not in valid_values:
            raise ValueError(f"user_status 必须是 {valid_values} 中的一个")
        return value
