from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.model.pydantic_m.knowledge import KnowledgeParam
from src.model.sqlalchemy_m.model import Knowledge
from src.utils.common import chinese_to_pinyin, get_collection_name, has_chinese
from src.utils.handler.milvus_handler import init_milvus
from src.utils.handler.mysql_handler import get_session

router = APIRouter(
    prefix="/knowledge",
    tags=["知识库相关"],
)


@router.post("/new_knowledge", summary="新建知识库")
async def new_knowledge(knowledge_param: KnowledgeParam, db: Session = Depends(get_session)):
    try:
        logger.info("开始创建知识库")
        user_id = knowledge_param.user_id  # 用户id
        knowledge_name = get_collection_name(user_id)  # 集合的id

        partition_name = knowledge_param.p_name.strip()  # 去除前后空格
        partition_name = chinese_to_pinyin(partition_name) if has_chinese(partition_name) else partition_name

        # 检查知识库是否已存在
        knowledge_item = db.query(Knowledge).filter_by(partition_name=partition_name).first()
        if knowledge_item:
            return {"msg": f"{partition_name}：知识库已存在！"}

        # 插入数据库
        knowledge = Knowledge(knowledge_name=knowledge_name, partition_name=partition_name, user_id=user_id)
        db.add(knowledge)

        try:
            db.commit()
        except IntegrityError as e:
            db.rollback()
            logger.error(f"知识库插入失败，可能是唯一性约束冲突：{e}")
            raise HTTPException(status_code=500, detail="知识库插入失败，请稍后再试！")

        logger.success("知识库插入成功！")

        # 初始化Milvus的集合的分区
        try:
            init_milvus(knowledge_name, partition_name)
        except Exception as e:
            logger.error(f"Milvus初始化失败：{e}")
            raise HTTPException(status_code=500, detail="Milvus初始化失败，请检查配置！")

        return {"msg": f"{partition_name}：知识库创建成功！"}

    except Exception as e:
        logger.error(f"知识库创建失败：{e}")
        raise HTTPException(status_code=500, detail="知识库创建失败，请联系管理员！")
