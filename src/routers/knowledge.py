import uuid

from fastapi import APIRouter, Depends
from loguru import logger
from sqlalchemy.orm import Session

from src.model.pydantic_m.knowledge import KnowledgeParam
from src.model.sqlalchemy_m.model import Knowledge
from src.utils.handler.mysql_handler import get_session

router = APIRouter(
    prefix="/knowledge",
    tags=["知识库相关"],
)


@router.post("/new_knowledge", summary="新建知识库")
@logger.catch
async def new_knowledge(knowledge_param: KnowledgeParam, db: Session = Depends(get_session)):
    logger.info("开始创建知识库")
    user_id = knowledge_param.user_id  # 用户id
    knowledge_name = "knowledge_name_" + str(user_id)  # 集合的id
    partition_name = knowledge_param.p_name  # 知识库名称
    knowledge_item = db.query(Knowledge).filter_by(user_id=user_id).first()
    if not knowledge_item:
        # 插入数据库
        knowledge = Knowledge(knowledge_name=knowledge_name, partition_name=partition_name, user_id=user_id)
        db.add(knowledge)
        db.commit()
        logger.success("知识库插入成功！")
        # 初始化Milvus的集合的分区
        # init_milvus(c_id, p_id)

    return {"msg": f"{partition_name}：知识库创建成功！"}
