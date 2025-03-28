@chcp 65001 >nul
@echo off

echo 延迟20秒再启动，等待网络自动连接
timeout /t 20 >nul

echo 第一步，切换到D盘
d:

echo 第二步，切换到指定目录
cd \PycharmProjects\yxy_rag\docs\soft\frp\

echo 第三步，打开frpc.exe
.\frpc.exe -c .\frpc.toml

PAUSE
