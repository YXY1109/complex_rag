from pydantic import BaseModel, Field


class UserParam(BaseModel):
    user_name: str = Field(default="yxy", description="用户名称")
    user_status: int = Field(default=1, description="用户状态")
