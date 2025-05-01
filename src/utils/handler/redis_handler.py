import redis

from src.config.config import settings
from src.utils.logger import logger


class RedisClient(object):

    def __init__(self):
        redis_host = settings.REDIS_HOST
        redis_port = settings.REDIS_PORT
        redis_password = settings.REDIS_PASSWORD
        db_num = settings.REDIS_DB_NUM

        self.redis = redis.StrictRedis(host=redis_host, password=redis_password, port=redis_port, db=db_num)

    def ping(self):
        return self.redis.ping()

    def set(self, key, value):
        self.redis.set(key, value)

    def get(self, key):
        return self.redis.get(key)

    def delete(self, key):
        self.redis.delete(key)

    def keys(self, pattern="*"):
        return self.redis.keys(pattern)


if __name__ == "__main__":
    client = RedisClient()
    logger.info(f"redis ping:{client.ping()}")
    client.set("name", "你好123")

    name = client.get("name").decode("utf-8")
    logger.info(name)
