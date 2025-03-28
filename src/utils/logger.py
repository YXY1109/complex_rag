import os
import sys

from loguru import logger as _


class Logger:
    """输出日志到文件和控制台"""

    def __init__(self, path_dir: str = "logs"):
        # 文件的命名
        log_path = os.path.join(path_dir, "Fast_{time:YYYY-MM-DD}.log")
        self.logger = _
        # 清空所有设置
        self.logger.remove()
        # 判断日志文件夹是否存在，不存则创建
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)
        # 添加控制台输出的格式,sys.stdout为输出到屏幕;关于这些配置还需要自定义请移步官网查看相关参数说明
        self.logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "  # 颜色>时间
            "{process.name} | "  # 进程名
            "{thread.name} | "  # 进程名
            "<cyan>{module}</cyan>.<cyan>{function}</cyan>"  # 模块名.方法名
            ":<cyan>{line}</cyan> | "  # 行号
            "<level>{level}</level>: "  # 等级
            "<level>{message}</level>",  # 日志内容
        )
        # 日志写入文件
        self.logger.add(
            log_path,  # 写入目录指定文件
            format="{time:YYYY-MM-DD HH:mm:ss} - "  # 时间
            "{process.name} | "  # 进程名
            "{thread.name} | "  # 进程名
            "{module}.{function}:{line} - {level} -{message}",  # 模块名.方法名:行号
            encoding="utf-8",
            retention="7 days",  # 设置历史保留时长
            backtrace=True,  # 回溯
            diagnose=True,  # 诊断
            enqueue=True,  # 异步写入
            rotation="00:00",  # 每日更新时间
            # rotation="5kb",  # 切割，设置文件大小，rotation="12:00"，rotation="1 week"
            # filter="my_module"  # 过滤模块
            # compression="zip"   # 文件压缩
        )

    def get_logger(self):
        return self.logger


# 获取上一级目录下logs目录
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_path_dir = os.path.join(root, "logs")
logger = Logger(log_path_dir).get_logger()
