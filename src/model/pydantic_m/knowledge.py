from pydantic import BaseModel, Field


class KnowledgeParam(BaseModel):
    user_id: int = Field(default=1, description="用户id")
    p_name: str = Field(default="民法典", description="知识库名称")
