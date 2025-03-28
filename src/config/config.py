from dotenv import load_dotenv
from pydantic.v1 import BaseSettings

# 加载 .env 文件
load_dotenv()


# 定义一个 pydantic 的设置类
class ConfigSettings(BaseSettings):
    # 定义环境变量及其类型
    DEBUG: bool
    SERVER_HOST: str
    SERVER_PORT: int

    MYSQL_DB: str
    MYSQL_HOST: str
    MYSQL_PORT: int
    MYSQL_USER: str
    MYSQL_PASSWORD: str
    MYSQL_ECHO: bool
    MYSQL_POOL_SIZE: int
    MYSQL_POOL_RECYCLE: int
    MYSQL_POOL_PING: bool
    MYSQL_MAX_OVERFLOW: int

    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_PASSWORD: str
    REDIS_DB_NUM: int
    REDIS_DB_BACKEND: int

    class Config:
        # 配置环境变量的加载方式，这里从 .env 文件中加载
        env_file = ".env"


# 创建 Settings 类的实例
settings = ConfigSettings()  # type: ignore
