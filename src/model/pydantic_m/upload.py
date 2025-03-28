from pydantic import BaseModel, Field


class UploadParam(BaseModel):
    """
    milvus:数据库->Collection->Partition
    Collection：对应不同的用户
    Partition：对应不同的知识库
    """

    user_id: int = Field(default=1, description="用户id")
    p_id: str = Field(default="Partition_f4eb6f921b7842f4a0b81c0e715ff78a", description="知识库名称")
