"""
响应缓存中间件
实现多层缓存机制，包括内存缓存、Redis缓存和CDN缓存
"""
import time
import json
import hashlib
import asyncio
from typing import Dict, Any, Optional, List, Callable, Union
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from infrastructure.monitoring.loguru_logger import logger


class CacheStrategy(str, Enum):
    """缓存策略枚举"""
    NO_CACHE = "no_cache"
    MEMORY_ONLY = "memory_only"
    REDIS_ONLY = "redis_only"
    MULTI_LEVEL = "multi_level"
    CDN_FIRST = "cdn_first"


class CacheKeyGenerator:
    """缓存键生成器"""

    @staticmethod
    def generate_key(
        request: Request,
        include_query_params: bool = True,
        include_headers: List[str] = None,
        include_body: bool = False
    ) -> str:
        """
        生成缓存键

        Args:
            request: HTTP请求对象
            include_query_params: 是否包含查询参数
            include_headers: 包含的请求头列表
            include_body: 是否包含请求体

        Returns:
            str: 缓存键
        """
        key_parts = [
            request.method,
            request.url.path
        ]

        # 添加查询参数
        if include_query_params and request.query_params:
            sorted_params = sorted(request.query_params.items())
            key_parts.append(str(sorted_params))

        # 添加特定的请求头
        if include_headers:
            headers = []
            for header_name in include_headers:
                header_value = request.headers.get(header_name)
                if header_value:
                    headers.append(f"{header_name}:{header_value}")
            if headers:
                key_parts.append("|".join(headers))

        # 添加请求体（仅用于POST/PUT等）
        if include_body and request.method in ["POST", "PUT", "PATCH"]:
            try:
                body = request._body.decode('utf-8')
                # 对大的请求体进行哈希处理
                if len(body) > 1024:
                    body_hash = hashlib.md5(body.encode()).hexdigest()
                    key_parts.append(f"body_hash:{body_hash}")
                else:
                    key_parts.append(f"body:{body}")
            except Exception:
                pass

        # 生成最终的缓存键
        key_string = "|".join(key_parts)
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()

        return f"cache:{cache_key}"


@dataclass
class CacheConfig:
    """缓存配置"""
    strategy: CacheStrategy = CacheStrategy.MULTI_LEVEL
    default_ttl: int = 300  # 默认TTL（秒）
    max_memory_size: int = 1000  # 最大内存缓存条目数
    memory_ttl: int = 300  # 内存缓存TTL
    redis_ttl: int = 3600  # Redis缓存TTL
    vary_headers: List[str] = None  # 缓存变化头
    skip_cache_methods: List[str] = None  # 跳过缓存的HTTP方法
    cacheable_status_codes: List[int] = None  # 可缓存的HTTP状态码
    compress_threshold: int = 1024  # 压缩阈值（字节）
    enable_etag: bool = True  # 启用ETag

    def __post_init__(self):
        if self.vary_headers is None:
            self.vary_headers = ["Authorization", "Accept-Language"]
        if self.skip_cache_methods is None:
            self.skip_cache_methods = ["POST", "PUT", "PATCH", "DELETE"]
        if self.cacheable_status_codes is None:
            self.cacheable_status_codes = [200, 301, 302, 404]


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    response_data: Dict[str, Any]
    status_code: int
    headers: Dict[str, str]
    created_at: float
    ttl: int
    etag: Optional[str] = None
    last_accessed: Optional[float] = None
    access_count: int = 0
    compressed: bool = False

    def is_expired(self) -> bool:
        """检查缓存是否过期"""
        return time.time() - self.created_at > self.ttl

    def touch(self):
        """更新最后访问时间"""
        self.last_accessed = time.time()
        self.access_count += 1


class MemoryCache:
    """内存缓存实现"""

    def __init__(self, max_size: int = 1000):
        """
        初始化内存缓存

        Args:
            max_size: 最大缓存条目数
        """
        self.max_size = max_size
        self.cache: Dict[str, CacheEntry] = {}
        self.access_times: Dict[str, float] = {}
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0,
            "deletes": 0"
        }

    def get(self, key: str) -> Optional[CacheEntry]:
        """获取缓存条目"""
        entry = self.cache.get(key)
        if entry:
            if entry.is_expired():
                self.delete(key)
                self.stats["misses"] += 1
                return None

            entry.touch()
            self.access_times[key] = time.time()
            self.stats["hits"] += 1
            return entry

        self.stats["misses"] += 1
        return None

    def set(self, key: str, entry: CacheEntry) -> bool:
        """设置缓存条目"""
        # 检查是否需要清理
        if len(self.cache) >= self.max_size:
            self._evict_lru()

        self.cache[key] = entry
        self.access_times[key] = time.time()
        self.stats["sets"] += 1
        return True

    def delete(self, key: str) -> bool:
        """删除缓存条目"""
        if key in self.cache:
            del self.cache[key]
            self.access_times.pop(key, None)
            self.stats["deletes"] += 1
            return True
        return False

    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "sets": 0,
            "deletes": 0
        }

    def _evict_lru(self):
        """使用LRU策略清理缓存"""
        if not self.access_times:
            return

        # 找到最久未访问的键
        lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self.delete(lru_key)
        self.stats["evictions"] += 1

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        hit_rate = self.stats["hits"] / max(1, self.stats["hits"] + self.stats["misses"])
        return {
            **self.stats,
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": round(hit_rate, 4),
            "utilization": round(len(self.cache) / self.max_size, 4)
        }


class RedisCache:
    """Redis缓存实现"""

    def __init__(self):
        """初始化Redis缓存"""
        self.redis_client = None
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        self._init_redis_client()

    def _init_redis_client(self):
        """初始化Redis客户端"""
        try:
            from infrastructure.cache.implementations.redis_cache_adapter import RedisCache
            self.redis_client = RedisCache()
            logger.info("Redis缓存已启用")
        except Exception as e:
            logger.warning(f"Redis初始化失败，将禁用Redis缓存: {str(e)}")
            self.redis_client = None

    async def get(self, key: str) -> Optional[CacheEntry]:
        """获取缓存条目"""
        if not self.redis_client:
            self.stats["errors"] += 1
            return None

        try:
            # 模拟Redis获取操作
            data = await self._redis_get(key)
            if data:
                entry_dict = json.loads(data)
                entry = CacheEntry(**entry_dict)

                if entry.is_expired():
                    await self.delete(key)
                    self.stats["misses"] += 1
                    return None

                self.stats["hits"] += 1
                return entry

            self.stats["misses"] += 1
            return None

        except Exception as e:
            logger.error(f"Redis获取失败: {str(e)}")
            self.stats["errors"] += 1
            return None

    async def set(self, key: str, entry: CacheEntry, ttl: Optional[int] = None) -> bool:
        """设置缓存条目"""
        if not self.redis_client:
            self.stats["errors"] += 1
            return False

        try:
            data = json.dumps(asdict(entry), ensure_ascii=False)
            # 模拟Redis设置操作
            await self._redis_set(key, data, ttl or entry.ttl)
            self.stats["sets"] += 1
            return True

        except Exception as e:
            logger.error(f"Redis设置失败: {str(e)}")
            self.stats["errors"] += 1
            return False

    async def delete(self, key: str) -> bool:
        """删除缓存条目"""
        if not self.redis_client:
            self.stats["errors"] += 1
            return False

        try:
            await self._redis_delete(key)
            self.stats["deletes"] += 1
            return True

        except Exception as e:
            logger.error(f"Redis删除失败: {str(e)}")
            self.stats["errors"] += 1
            return False

    async def clear(self):
        """清空缓存"""
        if not self.redis_client:
            return

        try:
            # 模拟Redis清空操作
            await self._redis_clear()
            self.stats = {
                "hits": 0,
                "misses": 0,
                "sets": 0,
                "deletes": 0,
                "errors": 0
            }

        except Exception as e:
            logger.error(f"Redis清空失败: {str(e)}")

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        hit_rate = self.stats["hits"] / max(1, self.stats["hits"] + self.stats["misses"])
        return {
            **self.stats,
            "hit_rate": round(hit_rate, 4),
            "connected": self.redis_client is not None
        }

    # 模拟Redis操作方法
    async def _redis_get(self, key: str) -> Optional[str]:
        """模拟Redis GET操作"""
        await asyncio.sleep(0.001)  # 模拟网络延迟
        return None  # 实际应该从Redis获取数据

    async def _redis_set(self, key: str, value: str, ttl: int):
        """模拟Redis SET操作"""
        await asyncio.sleep(0.001)  # 模拟网络延迟

    async def _redis_delete(self, key: str):
        """模拟Redis DELETE操作"""
        await asyncio.sleep(0.001)  # 模拟网络延迟

    async def _redis_clear(self):
        """模拟Redis FLUSH操作"""
        await asyncio.sleep(0.01)  # 模拟网络延迟


class CacheMiddleware(BaseHTTPMiddleware):
    """缓存中间件"""

    def __init__(self, app, config: CacheConfig):
        """
        初始化缓存中间件

        Args:
            app: ASGI应用
            config: 缓存配置
        """
        super().__init__(app)
        self.config = config
        self.key_generator = CacheKeyGenerator()

        # 初始化缓存层
        self.memory_cache = MemoryCache(config.max_memory_size)
        self.redis_cache = RedisCache()

        # 统计信息
        self.stats = {
            "total_requests": 0,
            "cached_responses": 0,
            "cache_misses": 0,
            "compression_saves": 0
        }

    async def dispatch(self, request: Request, call_next):
        """
        处理请求并执行缓存逻辑

        Args:
            request: HTTP请求对象
            call_next: 下一个中间件或路由处理器

        Returns:
            Response: HTTP响应对象
        """
        self.stats["total_requests"] += 1

        # 检查是否应该跳过缓存
        if self._should_skip_cache(request):
            return await call_next(request)

        # 生成缓存键
        cache_key = self.key_generator.generate_key(
            request,
            include_query_params=True,
            include_headers=self.config.vary_headers,
            include_body=request.method in ["POST", "PUT", "PATCH"]
        )

        # 尝试从缓存获取响应
        cached_response = await self._get_cached_response(cache_key, request)
        if cached_response:
            self.stats["cached_responses"] += 1
            return cached_response

        # 执行请求
        response = await call_next(request)

        # 检查响应是否可以缓存
        if self._should_cache_response(request, response):
            await self._cache_response(cache_key, request, response)

        self.stats["cache_misses"] += 1
        return response

    def _should_skip_cache(self, request: Request) -> bool:
        """检查是否应该跳过缓存"""
        # 检查HTTP方法
        if request.method in self.config.skip_cache_methods:
            return True

        # 检查缓存控制头
        cache_control = request.headers.get("cache-control", "")
        if "no-cache" in cache_control or "no-store" in cache_control:
            return True

        # 检查自定义跳过缓存头
        if request.headers.get("x-skip-cache") == "true":
            return True

        return False

    def _should_cache_response(self, request: Request, response: Response) -> bool:
        """检查响应是否应该被缓存"""
        # 检查状态码
        if response.status_code not in self.config.cacheable_status_codes:
            return False

        # 检查响应头
        cache_control = response.headers.get("cache-control", "")
        if "no-cache" in cache_control or "no-store" in cache_control or "private" in cache_control:
            return False

        # 检查响应大小
        content_length = response.headers.get("content-length")
        if content_length:
            try:
                size = int(content_length)
                # 过大的响应不缓存
                if size > 10 * 1024 * 1024:  # 10MB
                    return False
            except ValueError:
                pass

        return True

    async def _get_cached_response(self, cache_key: str, request: Request) -> Optional[Response]:
        """从缓存获取响应"""
        cached_entry = None

        # 根据策略获取缓存
        if self.config.strategy == CacheStrategy.MEMORY_ONLY:
            cached_entry = self.memory_cache.get(cache_key)
        elif self.config.strategy == CacheStrategy.REDIS_ONLY:
            cached_entry = await self.redis_cache.get(cache_key)
        elif self.config.strategy == CacheStrategy.MULTI_LEVEL:
            # 先查内存缓存，再查Redis缓存
            cached_entry = self.memory_cache.get(cache_key)
            if not cached_entry:
                cached_entry = await self.redis_cache.get(cache_key)
                if cached_entry:
                    # 回填到内存缓存
                    self.memory_cache.set(cache_key, cached_entry)

        if not cached_entry:
            return None

        # 检查ETag
        if self.config.enable_etag and cached_entry.etag:
            client_etag = request.headers.get("if-none-match")
            if client_etag == cached_entry.etag:
                return Response(
                    status_code=304,
                    headers={
                        "ETag": cached_entry.etag,
                        "Cache-Control": f"max-age={cached_entry.ttl}",
                        "X-Cache": "HIT"
                    }
                )

        # 构建响应
        response_data = cached_entry.response_data
        headers = cached_entry.headers.copy()
        headers.update({
            "X-Cache": "HIT",
            "X-Cache-Key": cache_key[:16] + "...",
            "X-Cache-Age": str(int(time.time() - cached_entry.created_at)),
            "X-Cache-Hits": str(cached_entry.access_count)
        })

        if self.config.enable_etag and cached_entry.etag:
            headers["ETag"] = cached_entry.etag

        return JSONResponse(
            content=response_data,
            status_code=cached_entry.status_code,
            headers=headers
        )

    async def _cache_response(self, cache_key: str, request: Request, response: Response):
        """缓存响应"""
        # 提取响应数据
        if isinstance(response, JSONResponse):
            response_data = response.body.decode('utf-8')
        else:
            # 对于非JSON响应，跳过缓存
            return

        try:
            response_dict = json.loads(response_data)
        except json.JSONDecodeError:
            return

        # 生成ETag
        etag = None
        if self.config.enable_etag:
            etag = hashlib.md5(response_data.encode()).hexdigest()

        # 确定TTL
        ttl = self._determine_ttl(request, response)

        # 创建缓存条目
        cache_entry = CacheEntry(
            key=cache_key,
            response_data=response_dict,
            status_code=response.status_code,
            headers=dict(response.headers),
            created_at=time.time(),
            ttl=ttl,
            etag=etag
        )

        # 根据策略存储缓存
        if self.config.strategy == CacheStrategy.MEMORY_ONLY:
            self.memory_cache.set(cache_key, cache_entry)
        elif self.config.strategy == CacheStrategy.REDIS_ONLY:
            await self.redis_cache.set(cache_key, cache_entry, ttl)
        elif self.config.strategy == CacheStrategy.MULTI_LEVEL:
            # 同时存储到内存和Redis
            self.memory_cache.set(cache_key, cache_entry)
            await self.redis_cache.set(cache_key, cache_entry, ttl)

        # 添加缓存头到原始响应
        response.headers["X-Cache"] = "MISS"
        response.headers["X-Cache-Key"] = cache_key[:16] + "..."
        if etag:
            response.headers["ETag"] = etag
        response.headers["Cache-Control"] = f"max-age={ttl}"

    def _determine_ttl(self, request: Request, response: Response) -> int:
        """确定缓存TTL"""
        # 检查响应中的缓存控制头
        cache_control = response.headers.get("cache-control", "")
        if "max-age=" in cache_control:
            try:
                for directive in cache_control.split(","):
                    if directive.startswith("max-age="):
                        return int(directive.split("=")[1])
            except ValueError:
                pass

        # 根据响应类型确定TTL
        if response.status_code == 404:
            return 300  # 404响应缓存5分钟
        elif response.status_code in [301, 302]:
            return 3600  # 重定向响应缓存1小时
        else:
            return self.config.default_ttl

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        cache_hit_rate = (
            self.stats["cached_responses"] / max(1, self.stats["total_requests"])
        )

        return {
            **self.stats,
            "cache_hit_rate": round(cache_hit_rate, 4),
            "cache_miss_rate": round(1 - cache_hit_rate, 4),
            "strategy": self.config.strategy,
            "memory_cache": self.memory_cache.get_stats(),
            "redis_cache": self.redis_cache.get_stats()
        }

    def clear_cache(self):
        """清空所有缓存"""
        self.memory_cache.clear()
        asyncio.create_task(self.redis_cache.clear())


class CacheManager:
    """缓存管理器"""

    def __init__(self):
        self.middleware: Optional[CacheMiddleware] = None

    def configure(self, config_dict: Dict[str, Any]) -> CacheMiddleware:
        """配置缓存中间件"""
        config = CacheConfig(**config_dict)

        # 创建中间件的占位符，实际使用时需要传入app实例
        class PlaceholderMiddleware:
            def __init__(self):
                self.config = config

        self.middleware = PlaceholderMiddleware()
        return self.middleware  # type: ignore

    def get_middleware(self, app):
        """获取缓存中间件实例"""
        if not self.middleware:
            raise ValueError("请先调用configure()方法配置缓存中间件")

        return CacheMiddleware(app, self.middleware.config)


# 全局缓存管理器实例
cache_manager = CacheManager()