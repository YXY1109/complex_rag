from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

from src.config.config import settings
from src.utils.logger import logger


def get_engine():
    # mysql连接配置 ?charset=utf8&autocommit=true
    database_uri = (
        f"mysql+pymysql://{settings.MYSQL_USER}:{settings.MYSQL_PASSWORD}@{settings.MYSQL_HOST}:"
        f"{settings.MYSQL_PORT}/{settings.MYSQL_DB}"
    )
    my_engine = create_engine(
        database_uri,
        echo=settings.MYSQL_ECHO,
        pool_size=settings.MYSQL_POOL_SIZE,
        pool_recycle=settings.MYSQL_POOL_RECYCLE,
        pool_pre_ping=settings.MYSQL_POOL_PING,
        max_overflow=settings.MYSQL_MAX_OVERFLOW,
    )
    return my_engine


def get_session():
    session = sessionmaker(bind=get_engine())
    # 线程安全
    my_session = scoped_session(session)
    # 方式一
    # return my_session
    # 方式二
    # https://copyprogramming.com/howto/how-to-correctly-use-sqlalchemy-within-fastapi-or-arq-for-mysql
    try:
        yield my_session
    finally:
        logger.info("主动关闭session")
        my_session.close()
    # 方式二使用：get_session().__next__()
