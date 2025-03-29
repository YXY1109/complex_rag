import datetime

from sqlalchemy import Column, DateTime, Integer, SmallInteger, String
from sqlalchemy.orm import declarative_base

from src.utils.handler.mysql_handler import get_engine, get_session
from src.utils.logger import logger

Base = declarative_base()
now = datetime.datetime.now


class BaseTime(Base):
    __abstract__ = True
    update_time = Column(DateTime, default=now, comment="更新时间", onupdate=now)
    create_time = Column(DateTime, default=now, comment="创建时间")


class User(BaseTime):
    """
    用户信息表
    """

    __tablename__ = "user"
    __table_args__ = {"comment": "用户"}
    user_id = Column(SmallInteger, primary_key=True, autoincrement=True, comment="主键，用户id")
    user_name = Column(String(50), default="", comment="用户名")
    user_status = Column(SmallInteger, default=1, comment="0=关闭 1=启用")

    def __repr__(self):
        return self.user_id


class Knowledge(BaseTime):
    """
    milvus知识库
    """

    __tablename__ = "knowledge"
    __table_args__ = {"comment": "知识库"}

    knowledge_id = Column(Integer, primary_key=True, comment="主键，知识库id")
    knowledge_name = Column(String(50), default="", comment="知识库名称，对应collection")
    partition_name = Column(String(50), default="", comment="知识库分区名称")
    knowledge_status = Column(SmallInteger, default=1, comment="0=关闭 1=启用")
    user_id = Column(SmallInteger, comment="用户id")

    def __repr__(self):
        return self.knowledge_id


class File(BaseTime):
    """
    文件
    """

    __tablename__ = "file"
    __table_args__ = {"comment": "文件上传"}
    file_id = Column(SmallInteger, primary_key=True, autoincrement=True, comment="主键，文件id")
    file_name = Column(String(50), comment="文件名称")
    file_status = Column(SmallInteger, default=1, comment="0=关闭 1=启动")
    minio_path = Column(String(100), comment="Minio文件相对路径")
    knowledge_id = Column(String(50), primary_key=True, comment="主键，知识库id")

    def __repr__(self):
        return self.file_id


def create_all():
    # 创建表
    Base.metadata.create_all(my_engine)
    print("创建表完成")


def drop_all():
    # 删除表
    Base.metadata.drop_all(my_engine)
    print("删除表完成")


def create_default_user():
    user = UserCollection(user_id=1, user_name="YXY1109")
    my_session.add(user)
    my_session.commit()
    print("创建默认用户完成")


def init_mysql():
    drop_all()
    create_all()
    logger.info("创建表完成")


if __name__ == "__main__":
    """
    https://blog.csdn.net/Cycloctane/article/details/133795319
    https://www.cnblogs.com/zx0524/p/17304552.html
    https://github.com/sqlalchemy/sqlalchemy
    """
    my_engine = get_engine()
    my_session = get_session().__next__()

    init_mysql()
