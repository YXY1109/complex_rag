from typing import List

from pydantic import BaseModel, Field


class ChatParam(BaseModel):
    user_id: int = Field(default=1, gt=0, description="用户id")
    p_id: str = Field(default="Partition_f4eb6f921b7842f4a0b81c0e715ff78a", description="知识库名称")
    question: str = Field(default="你是什么模型", description="问题")
    rerank: bool = Field(default=True, description="是否rerank")
    chat_history: List[str] = Field(default=[], description="聊天历史")
    stream: bool = Field(default=False, description="是否流式返回")
