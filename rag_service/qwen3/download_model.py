from modelscope import snapshot_download

model_dir1 = snapshot_download('Qwen/Qwen3-Embedding-0.6B', cache_dir='models')
print(model_dir1)
model_dir2 = snapshot_download('Qwen/Qwen3-Reranker-0.6B', cache_dir='models')
print(model_dir2)