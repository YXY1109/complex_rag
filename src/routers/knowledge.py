import uuid

from fastapi import APIRouter, Depends
from loguru import logger

router = APIRouter(
    prefix="/knowledge",
    tags=["知识库相关"],
)

# 后台任务
scheduler_async = AsyncIOScheduler()


@router.post("/new_knowledge", summary="新建知识库")
@logger.catch
async def new_knowledge(knowledge_param: KnowledgeParam, db: Session = Depends(get_session)):
    logger.info("开始创建知识库")
    user_id = knowledge_param.user_id  # 用户id
    c_id = "User_id_" + str(user_id)  # 集合的id
    p_name = knowledge_param.p_name  # 知识库名称
    p_id = db.query(KnowledgePartition).filter_by(knowledge_name=p_name).first()
    if not p_id:
        p_id = "Partition_" + uuid.uuid4().hex  # 知识库id/分区id
        # 插入数据库
        knowledge = KnowledgePartition(knowledge_id=p_id, knowledge_name=p_name, user_id=user_id)
        db.add(knowledge)
        db.commit()
        logger.success("知识库插入成功！")
        # 初始化Milvus的集合的分区
        init_milvus(c_id, p_id)

    return {"msg": f"{p_name}：知识库创建成功！知识库ID：{p_id}"}
