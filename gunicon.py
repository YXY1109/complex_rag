import multiprocessing
import os
import time

"""
线上的配置信息
"""

# 并行工作进程数 核心数+1个
workers = multiprocessing.cpu_count() + 1
bind = "0.0.0.0:5001"
# 指定每个工作者的线程数
threads = 2
# 设置守护进程
daemon = True
# 工作模式协程
worker_class = "uvicorn.workers.UvicornWorker"  # FastAPI时，使用uvicorn【重要注意】
# 设置最大并发量
worker_connections = 2000
# 设置进程文件目录
pidfile = "./gunicorn.pid"
# 工作目录
chdir = "./"
# 设置访问日志和错误信息日志路径
log_dir = "./log"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# 设置访问日志和错误信息日志路径
date_time = time.strftime("%Y-%m-%d_%H", time.localtime(time.time()))
accesslog = "./log/gunicorn_access.log"
errorlog = "./log/gunicorn_error.log"
# 日志级别，这个日志级别指的是错误日志的级别，而访问日志的级别无法设置
loglevel = "debug"
# 设置print的输出
capture_output = True
# 超时时间
timeout = 60
