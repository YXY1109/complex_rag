from fastapi import APIRouter, Depends
from loguru import logger

router = APIRouter(
    prefix="/user",
    tags=["用户相关"],
)


@router.post("/create_user", summary="创建用户")
@logger.catch
async def create_user(user_param: UserParam, db: Session = Depends(get_session)):
    logger.info("开始创建用户")
    # 创建用户
    user = UserCollection(user_name=user_param.user_name, user_status=user_param.user_status)
    db.add(user)
    db.commit()

    return {"msg": f"{user.user_name}：用户创建成功！用户ID：{user.user_id}"}
