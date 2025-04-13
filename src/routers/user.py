from fastapi import APIRouter, Depends
from loguru import logger
from sqlalchemy.orm import Session

from src.model.pydantic_m.user import UserParam
from src.model.sqlalchemy_m.model import User
from src.utils.handler.mysql_handler import get_session

router = APIRouter(
    prefix="/user",
    tags=["用户相关"],
)


@router.post("/create_user", summary="创建用户")
async def create_user(user_param: UserParam, db: Session = Depends(get_session)):
    logger.info("开始创建用户")
    # 创建用户
    user = User(user_name=user_param.user_name, user_status=user_param.user_status)
    db.add(user)
    db.commit()

    return {"msg": f"{user.user_name}：用户创建成功！用户ID：{user.user_id}"}
