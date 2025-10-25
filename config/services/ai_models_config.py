"""
AI Models Configuration

This module contains configuration for all AI model providers.
"""

from pydantic import Field

from ..settings import BaseConfig


class AIModelsConfig(BaseConfig):
    """AI models configuration."""

    # OpenAI Configuration
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://api.openai.com/v1", env="OPENAI_BASE_URL")
    openai_organization: str = Field(default="", env="OPENAI_ORGANIZATION")
    openai_timeout: int = Field(default=60, env="OPENAI_TIMEOUT")
    openai_max_retries: int = Field(default=3, env="OPENAI_MAX_RETRIES")
    openai_retry_after: int = Field(default=1, env="OPENAI_RETRY_AFTER")

    # OpenAI Model Settings
    openai_chat_models: list[str] = Field(
        default=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o"],
        env="OPENAI_CHAT_MODELS"
    )
    openai_embedding_models: list[str] = Field(
        default=["text-embedding-ada-002", "text-embedding-3-small", "text-embedding-3-large"],
        env="OPENAI_EMBEDDING_MODELS"
    )
    openai_rerank_models: list[str] = Field(
        default=[],
        env="OPENAI_RERANK_MODELS"
    )

    # Anthropic Configuration
    anthropic_api_key: str = Field(default="", env="ANTHROPIC_API_KEY")
    anthropic_base_url: str = Field(default="https://api.anthropic.com", env="ANTHROPIC_BASE_URL")
    anthropic_timeout: int = Field(default=60, env="ANTHROPIC_TIMEOUT")
    anthropic_max_retries: int = Field(default=3, env="ANTHROPIC_MAX_RETRIES")

    # Anthropic Model Settings
    anthropic_chat_models: list[str] = Field(
        default=["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        env="ANTHROPIC_CHAT_MODELS"
    )

    # Ollama Configuration
    ollama_base_url: str = Field(default="http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_timeout: int = Field(default=300, env="OLLAMA_TIMEOUT")
    ollama_max_retries: int = Field(default=3, env="OLLAMA_MAX_RETRIES")
    ollama_keep_alive: str = Field(default="1h", env="OLLAMA_KEEP_ALIVE")
    ollama_num_ctx: int = Field(default=4096, env="OLLAMA_NUM_CTX")
    ollama_num_predict: int = Field(default=-1, env="OLLAMA_NUM_PREDICT")
    ollama_temperature: float = Field(default=0.7, env="OLLAMA_TEMPERATURE")
    ollama_top_p: float = Field(default=0.9, env="OLLAMA_TOP_P")
    ollama_repeat_penalty: float = Field(default=1.1, env="OLLAMA_REPEAT_PENALTY")

    # Ollama Model Settings
    ollama_chat_models: list[str] = Field(
        default=["llama3.1:8b", "llama3.1:70b", "qwen2.5:7b", "qwen2.5:14b", "mistral:7b"],
        env="OLLAMA_CHAT_MODELS"
    )
    ollama_embedding_models: list[str] = Field(
        default=["nomic-embed-text", "mxbai-embed-large"],
        env="OLLAMA_EMBEDDING_MODELS"
    )

    # Qwen Configuration (Alibaba Cloud)
    qwen_api_key: str = Field(default="", env="QWEN_API_KEY")
    qwen_base_url: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1", env="QWEN_BASE_URL")
    qwen_timeout: int = Field(default=60, env="QWEN_TIMEOUT")
    qwen_max_retries: int = Field(default=3, env="QWEN_MAX_RETRIES")

    # Qwen Model Settings
    qwen_chat_models: list[str] = Field(
        default=["qwen-turbo", "qwen-plus", "qwen-max", "qwen2.5-72b-instruct", "qwen2.5-32b-instruct", "qwen2.5-14b-instruct", "qwen2.5-7b-instruct"],
        env="QWEN_CHAT_MODELS"
    )
    qwen_embedding_models: list[str] = Field(
        default=["text-embedding-v1", "text-embedding-v2", "text-embedding-v3"],
        env="QWEN_EMBEDDING_MODELS"
    )
    qwen_rerank_models: list[str] = Field(
        default=["gte-rerank"],
        env="QWEN_RERANK_MODELS"
    )

    # Baidu Configuration (BCE)
    bce_api_key: str = Field(default="", env="BCE_API_KEY")
    bce_secret_key: str = Field(default="", env="BCE_SECRET_KEY")
    bce_base_url: str = Field(default="https://aip.baidubce.com", env="BCE_BASE_URL")
    bce_timeout: int = Field(default=60, env="BCE_TIMEOUT")
    bce_max_retries: int = Field(default=3, env="BCE_MAX_RETRIES")

    # Baidu Model Settings
    bce_chat_models: list[str] = Field(
        default=["ernie-4.0-8k", "ernie-3.5-8k", "ernie-3.5-4k", "ernie-turbo-8k"],
        env="BCE_CHAT_MODELS"
    )
    bce_embedding_models: list[str] = Field(
        default=["bge-large-zh", "bge-large-en", "bge-base-en"],
        env="BCE_EMBEDDING_MODELS"
    )
    bce_rerank_models: list[str] = Field(
        default=["bge-reranker-base", "bge-reranker-large"],
        env="BCE_RERANK_MODELS"
    )

    # Hugging Face Configuration
    huggingface_token: str = Field(default="", env="HUGGINGFACE_TOKEN")
    huggingface_cache_dir: str = Field(default="~/.cache/huggingface", env="HUGGINGFACE_CACHE_DIR")
    huggingface_timeout: int = Field(default=60, env="HUGGINGFACE_TIMEOUT")

    # Local Model Configuration
    local_model_path: str = Field(default="models", env="LOCAL_MODEL_PATH")
    local_model_device: str = Field(default="auto", env="LOCAL_MODEL_DEVICE")  # auto, cpu, cuda, mps
    local_model_dtype: str = Field(default="float16", env="LOCAL_MODEL_DTYPE")  # float16, float32, int8
    local_model_threads: int = Field(default=4, env="LOCAL_MODEL_THREADS")
    local_model_max_length: int = Field(default=4096, env="LOCAL_MODEL_MAX_LENGTH")

    # Local Model Settings
    local_chat_models: dict[str, str] = Field(
        default={
            "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
            "qwen2.5-14b": "Qwen/Qwen2.5-14B-Instruct",
            "llama3.1-8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            "chatglm3-6b": "THUDM/chatglm3-6b",
        },
        env="LOCAL_CHAT_MODELS"
    )
    local_embedding_models: dict[str, str] = Field(
        default={
            "bge-small-zh": "BAAI/bge-small-zh-v1.5",
            "bge-base-zh": "BAAI/bge-base-zh-v1.5",
            "bge-large-zh": "BAAI/bge-large-zh-v1.5",
            "bge-m3": "BAAI/bge-m3",
        },
        env="LOCAL_EMBEDDING_MODELS"
    )

    # Model Provider Priority
    provider_priority: list[str] = Field(
        default=["local", "ollama", "openai", "qwen", "anthropic", "bce"],
        env="PROVIDER_PRIORITY"
    )

    # Fallback Configuration
    fallback_enabled: bool = Field(default=True, env="FALLBACK_ENABLED")
    fallback_providers: list[str] = Field(
        default=["ollama", "local"],
        env="FALLBACK_PROVIDERS"
    )
    fallback_timeout: int = Field(default=60, env="FALLBACK_TIMEOUT")

    # Model Settings
    default_temperature: float = Field(default=0.7, env="DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(default=2048, env="DEFAULT_MAX_TOKENS")
    default_top_p: float = Field(default=1.0, env="DEFAULT_TOP_P")
    default_frequency_penalty: float = Field(default=0.0, env="DEFAULT_FREQUENCY_PENALTY")
    default_presence_penalty: float = Field(default=0.0, env="DEFAULT_PRESENCE_PENALTY")

    # Rate Limiting
    rate_limit_enabled: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    rate_limit_requests_per_minute: int = Field(default=60, env="RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_tokens_per_minute: int = Field(default=10000, env="RATE_LIMIT_TOKENS_PER_MINUTE")

    # Streaming Settings
    streaming_enabled: bool = Field(default=True, env="STREAMING_ENABLED")
    streaming_buffer_size: int = Field(default=1024, env="STREAMING_BUFFER_SIZE")
    streaming_timeout: int = Field(default=30, env="STREAMING_TIMEOUT")

    def get_available_chat_models(self) -> dict[str, list[str]]:
        """Get all available chat models by provider."""
        return {
            "openai": self.openai_chat_models,
            "anthropic": self.anthropic_chat_models,
            "ollama": self.ollama_chat_models,
            "qwen": self.qwen_chat_models,
            "bce": self.bce_chat_models,
            "local": list(self.local_chat_models.keys()),
        }

    def get_available_embedding_models(self) -> dict[str, list[str]]:
        """Get all available embedding models by provider."""
        return {
            "openai": self.openai_embedding_models,
            "ollama": self.ollama_embedding_models,
            "qwen": self.qwen_embedding_models,
            "bce": self.bce_embedding_models,
            "local": list(self.local_embedding_models.keys()),
        }

    def get_available_rerank_models(self) -> dict[str, list[str]]:
        """Get all available rerank models by provider."""
        return {
            "qwen": self.qwen_rerank_models,
            "bce": self.bce_rerank_models,
            "openai": self.openai_rerank_models,
        }

    def get_provider_config(self, provider: str) -> dict:
        """Get configuration for a specific provider."""
        configs = {
            "openai": {
                "api_key": self.openai_api_key,
                "base_url": self.openai_base_url,
                "timeout": self.openai_timeout,
                "max_retries": self.openai_max_retries,
                "retry_after": self.openai_retry_after,
            },
            "anthropic": {
                "api_key": self.anthropic_api_key,
                "base_url": self.anthropic_base_url,
                "timeout": self.anthropic_timeout,
                "max_retries": self.anthropic_max_retries,
            },
            "ollama": {
                "base_url": self.ollama_base_url,
                "timeout": self.ollama_timeout,
                "max_retries": self.ollama_max_retries,
                "keep_alive": self.ollama_keep_alive,
                "num_ctx": self.ollama_num_ctx,
                "temperature": self.ollama_temperature,
                "top_p": self.ollama_top_p,
                "repeat_penalty": self.ollama_repeat_penalty,
            },
            "qwen": {
                "api_key": self.qwen_api_key,
                "base_url": self.qwen_base_url,
                "timeout": self.qwen_timeout,
                "max_retries": self.qwen_max_retries,
            },
            "bce": {
                "api_key": self.bce_api_key,
                "secret_key": self.bce_secret_key,
                "base_url": self.bce_base_url,
                "timeout": self.bce_timeout,
                "max_retries": self.bce_max_retries,
            },
            "local": {
                "model_path": self.local_model_path,
                "device": self.local_model_device,
                "dtype": self.local_model_dtype,
                "threads": self.local_model_threads,
                "max_length": self.local_model_max_length,
                "huggingface_token": self.huggingface_token,
                "cache_dir": self.huggingface_cache_dir,
                "timeout": self.huggingface_timeout,
            },
        }
        return configs.get(provider, {})


# Global AI models configuration instance
ai_models_config = AIModelsConfig()