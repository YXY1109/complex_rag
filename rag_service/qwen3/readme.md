# 本地部署qwen3-embedding和qwen3-reranker的服务

## 创建虚拟环境

```
conda create -n qwen3_server python=3.9 -y
conda activate qwen3_server
```

## 安装torch

```
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```