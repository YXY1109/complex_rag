"""
API中间件测试
测试各种中间件的功能和性能
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from fastapi import Request, Response
from starlette.responses import JSONResponse
import asyncio
import time

# 模拟导入（实际项目中这些是真实导入）
from api.middleware.rate_limiting import RateLimitMiddleware, rate_limit_manager
from api.middleware.caching import CacheMiddleware, cache_manager
from api.middleware.monitoring import MonitoringMiddleware, monitoring_manager


class TestRateLimitMiddleware:
    """限流中间件测试"""

    @pytest.fixture
    def mock_app(self):
        """模拟FastAPI应用"""
        return AsyncMock()

    @pytest.fixture
    def mock_request(self):
        """模拟HTTP请求"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/test"
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.headers = {}
        return request

    @pytest.fixture
    def mock_response(self):
        """模拟HTTP响应"""
        response = JSONResponse({"message": "test"})
        response.status_code = 200
        return response

    @pytest.mark.asyncio
    async def test_rate_limit_allows_requests(self, mock_app, mock_request, mock_response):
        """测试限流允许正常请求"""
        # 配置限流器
        config = {
            "requests_per_minute": 60,
            "burst_size": 10,
            "whitelist_ips": [],
            "blacklist_ips": []
        }

        middleware = RateLimitMiddleware(mock_app, config)
        mock_app.return_value = mock_response

        # 模拟call_next
        async def call_next(request):
            return mock_response

        # 执行中间件
        response = await middleware.dispatch(mock_request, call_next)

        assert response == mock_response

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excessive_requests(self, mock_app, mock_request):
        """测试限流阻止过量请求"""
        config = {
            "requests_per_minute": 2,  # 很低的限制
            "burst_size": 1,
            "whitelist_ips": [],
            "blacklist_ips": []
        }

        middleware = RateLimitMiddleware(mock_app, config)

        # 第一次请求应该通过
        async def call_next(request):
            return JSONResponse({"message": "ok"})

        response1 = await middleware.dispatch(mock_request, call_next)
        assert response1.status_code == 200

        # 第二次请求应该通过
        response2 = await middleware.dispatch(mock_request, call_next)
        assert response2.status_code == 200

        # 第三次请求应该被阻止
        response3 = await middleware.dispatch(mock_request, call_next)
        assert response3.status_code == 429  # Too Many Requests

    @pytest.mark.asyncio
    async def test_rate_limit_whitelist(self, mock_app, mock_request, mock_response):
        """测试IP白名单功能"""
        config = {
            "requests_per_minute": 1,
            "burst_size": 1,
            "whitelist_ips": ["127.0.0.1"],
            "blacklist_ips": []
        }

        middleware = RateLimitMiddleware(mock_app, config)
        mock_app.return_value = mock_response

        async def call_next(request):
            return mock_response

        # 白名单IP应该不受限制
        for _ in range(5):
            response = await middleware.dispatch(mock_request, call_next)
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_rate_limit_blacklist(self, mock_app, mock_request):
        """测试IP黑名单功能"""
        config = {
            "requests_per_minute": 60,
            "burst_size": 10,
            "whitelist_ips": [],
            "blacklist_ips": ["127.0.0.1"]
        }

        middleware = RateLimitMiddleware(mock_app, config)

        async def call_next(request):
            return JSONResponse({"message": "ok"})

        # 黑名单IP应该直接被拒绝
        response = await middleware.dispatch(mock_request, call_next)
        assert response.status_code == 403  # Forbidden

    def test_rate_limit_manager_configuration(self):
        """测试限流管理器配置"""
        config = {
            "strategy": "sliding_window",
            "requests_per_minute": 100,
            "burst_size": 20
        }

        manager = rate_limit_manager
        manager.configure(config)

        assert manager.middleware.config == config


class TestCacheMiddleware:
    """缓存中间件测试"""

    @pytest.fixture
    def mock_app(self):
        """模拟FastAPI应用"""
        return AsyncMock()

    @pytest.fixture
    def mock_request(self):
        """模拟HTTP请求"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/test"
        request.query_params = {}
        request.headers = {"accept": "application/json"}
        request._body = b""
        return request

    @pytest.fixture
    def mock_response(self):
        """模拟HTTP响应"""
        response = JSONResponse({"data": "test", "timestamp": time.time()})
        response.status_code = 200
        response.headers = {}
        return response

    @pytest.mark.asyncio
    async def test_cache_miss_and_store(self, mock_app, mock_request, mock_response):
        """测试缓存未命中并存储"""
        config = {
            "strategy": "memory_only",
            "default_ttl": 300,
            "max_memory_size": 100
        }

        middleware = CacheMiddleware(mock_app, config)

        # 第一次请求（缓存未命中）
        async def call_next(request):
            return mock_response

        response = await middleware.dispatch(mock_request, call_next)

        assert response == mock_response
        assert response.headers.get("X-Cache") == "MISS"

        # 验证响应被缓存
        cache_key = middleware.key_generator.generate_key(mock_request)
        cached_entry = middleware.memory_cache.get(cache_key)
        assert cached_entry is not None
        assert cached_entry.status_code == 200

    @pytest.mark.asyncio
    async def test_cache_hit(self, mock_app, mock_request, mock_response):
        """测试缓存命中"""
        config = {
            "strategy": "memory_only",
            "default_ttl": 300,
            "max_memory_size": 100
        }

        middleware = CacheMiddleware(mock_app, config)

        # 手动添加缓存条目
        cache_key = middleware.key_generator.generate_key(mock_request)
        from api.middleware.caching import CacheEntry
        cached_entry = CacheEntry(
            key=cache_key,
            response_data={"data": "cached", "timestamp": time.time()},
            status_code=200,
            headers={},
            created_at=time.time(),
            ttl=300
        )
        middleware.memory_cache.set(cache_key, cached_entry)

        async def call_next(request):
            # 这个不应该被调用
            assert False, "call_next should not be called for cache hit"

        response = await middleware.dispatch(mock_request, call_next)

        assert response.status_code == 200
        assert response.headers.get("X-Cache") == "HIT"

    @pytest.mark.asyncio
    async def test_cache_skip_for_post_requests(self, mock_app, mock_request):
        """测试POST请求跳过缓存"""
        mock_request.method = "POST"

        config = {
            "strategy": "memory_only",
            "default_ttl": 300,
            "skip_cache_methods": ["POST", "PUT", "DELETE"]
        }

        middleware = CacheMiddleware(mock_app, config)

        call_next_called = False
        async def call_next(request):
            nonlocal call_next_called
            call_next_called = True
            return JSONResponse({"message": "processed"})

        response = await middleware.dispatch(mock_request, call_next)

        assert call_next_called is True
        assert "X-Cache" not in response.headers

    @pytest.mark.asyncio
    async def test_cache_ttl_expiration(self, mock_app, mock_request):
        """测试缓存TTL过期"""
        config = {
            "strategy": "memory_only",
            "default_ttl": 1,  # 1秒过期
            "max_memory_size": 100
        }

        middleware = CacheMiddleware(mock_app, config)

        # 添加已过期的缓存条目
        cache_key = middleware.key_generator.generate_key(mock_request)
        from api.middleware.caching import CacheEntry
        cached_entry = CacheEntry(
            key=cache_key,
            response_data={"data": "expired"},
            status_code=200,
            headers={},
            created_at=time.time() - 2,  # 2秒前创建
            ttl=1
        )
        middleware.memory_cache.set(cache_key, cached_entry)

        call_next_called = False
        async def call_next(request):
            nonlocal call_next_called
            call_next_called = True
            return JSONResponse({"data": "fresh"})

        response = await middleware.dispatch(mock_request, call_next)

        assert call_next_called is True  # 缓存过期，应该调用call_next
        assert response.headers.get("X-Cache") == "MISS"

    def test_cache_manager_configuration(self):
        """测试缓存管理器配置"""
        config = {
            "strategy": "multi_level",
            "default_ttl": 600,
            "max_memory_size": 1000,
            "redis_ttl": 3600
        }

        manager = cache_manager
        manager.configure(config)

        assert manager.middleware.config == config

    def test_cache_key_generation(self):
        """测试缓存键生成"""
        from api.middleware.caching import CacheKeyGenerator

        # 测试基本键生成
        request = Mock()
        request.method = "GET"
        request.url.path = "/api/users"
        request.query_params = {"page": "1", "limit": "10"}
        request.headers = {}
        request._body = b""

        generator = CacheKeyGenerator()
        key1 = generator.generate_key(request)

        # 相同请求应该生成相同的键
        key2 = generator.generate_key(request)
        assert key1 == key2

        # 不同查询参数应该生成不同的键
        request.query_params = {"page": "2", "limit": "10"}
        key3 = generator.generate_key(request)
        assert key1 != key3

        # 键应该是有效的SHA256哈希
        import hashlib
        assert len(key) == 64  # SHA256长度
        assert all(c in "0123456789abcdef" for c in key)


class TestMonitoringMiddleware:
    """监控中间件测试"""

    @pytest.fixture
    def mock_app(self):
        """模拟FastAPI应用"""
        return AsyncMock()

    @pytest.fixture
    def mock_request(self):
        """模拟HTTP请求"""
        request = Mock(spec=Request)
        request.method = "GET"
        request.url.path = "/api/test"
        request.url.query = "param=value"
        return request

    @pytest.fixture
    def mock_response(self):
        """模拟HTTP响应"""
        response = Mock(spec=Response)
        response.status_code = 200
        response.headers = {}
        return response

    @pytest.mark.asyncio
    async def test_monitoring_metrics_collection(self, mock_app, mock_request, mock_response):
        """测试监控指标收集"""
        config = {
            "enable_request_logging": True,
            "enable_performance_tracking": True,
            "enable_system_monitoring": False  # 禁用系统监控以避免依赖问题
        }

        middleware = MonitoringMiddleware(mock_app, config)

        async def call_next(request):
            # 模拟处理时间
            await asyncio.sleep(0.01)
            return mock_response

        response = await middleware.dispatch(mock_request, call_next)

        # 验证响应头被添加
        assert "X-Request-ID" in response.headers
        assert "X-Processing-Time" in response.headers

        # 验证指标被收集
        metrics = middleware.get_metrics()
        assert "performance" in metrics
        assert metrics["performance"]["request_count"] >= 1
        assert metrics["performance"]["request_duration_avg"] > 0

    @pytest.mark.asyncio
    async def test_error_tracking(self, mock_app, mock_request):
        """测试错误跟踪"""
        config = {
            "enable_request_logging": True,
            "enable_performance_tracking": True
        }

        middleware = MonitoringMiddleware(mock_app, config)

        async def call_next(request):
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await middleware.dispatch(mock_request, call_next)

        # 验证错误指标被记录
        metrics = middleware.get_metrics()
        assert metrics["performance"]["error_count"] >= 1
        assert metrics["performance"]["error_rate"] > 0

    @pytest.mark.asyncio
    async def test_concurrent_requests_tracking(self, mock_app, mock_request, mock_response):
        """测试并发请求跟踪"""
        config = {
            "enable_request_logging": True,
            "enable_performance_tracking": True
        }

        middleware = MonitoringMiddleware(mock_app, config)

        async def call_next(request):
            await asyncio.sleep(0.01)
            return mock_response

        # 并发执行多个请求
        tasks = []
        for _ in range(5):
            task = middleware.dispatch(mock_request, call_next)
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        # 验证所有请求都被跟踪
        metrics = middleware.get_metrics()
        assert metrics["performance"]["request_count"] == 5

        # 验证活跃请求数正确管理
        assert metrics["performance"]["active_requests"] == 0  # 所有请求已完成

    def test_business_metrics_update(self):
        """测试业务指标更新"""
        config = {
            "enable_request_logging": True,
            "enable_performance_tracking": True
        }

        middleware = MonitoringMiddleware(None, config)

        # 更新业务指标
        middleware.update_business_metrics(
            total_conversations=100,
            total_documents=500,
            active_users=25
        )

        metrics = middleware.get_metrics()
        business = metrics["business"]
        assert business["total_conversations"] == 100
        assert business["total_documents"] == 500
        assert business["active_users"] == 25

    def test_monitoring_manager_configuration(self):
        """测试监控管理器配置"""
        config = {
            "enable_request_logging": True,
            "enable_performance_tracking": True,
            "enable_system_monitoring": True,
            "metrics_retention_hours": 24
        }

        manager = monitoring_manager
        manager.configure(config)

        assert manager.middleware.config == config

    def test_time_windows_statistics(self):
        """测试时间窗口统计"""
        config = {
            "enable_request_logging": True,
            "enable_performance_tracking": True
        }

        middleware = MonitoringMiddleware(None, config)

        # 模拟一些请求数据
        for _ in range(10):
            middleware.metrics_collector._record_time_window_data(
                duration=0.1,
                status_code=200
            )

        metrics = middleware.get_metrics()
        time_windows = metrics["time_windows"]

        # 验证时间窗口统计
        assert "1m" in time_windows
        assert time_windows["1m"]["request_count"] == 10
        assert time_windows["1m"]["avg_duration"] == 0.1
        assert time_windows["1m"]["error_rate"] == 0.0


class TestMiddlewareIntegration:
    """中间件集成测试"""

    @pytest.mark.asyncio
    async def test_middleware_chain_execution(self):
        """测试中间件链执行"""
        from unittest.mock import AsyncMock

        # 创建模拟应用
        mock_app = AsyncMock()
        mock_app.return_value = JSONResponse({"message": "success"})

        # 创建中间件
        rate_limit_config = {"requests_per_minute": 60, "burst_size": 10}
        cache_config = {"strategy": "memory_only", "default_ttl": 300}
        monitoring_config = {"enable_request_logging": True}

        rate_limit_middleware = RateLimitMiddleware(mock_app, rate_limit_config)
        cache_middleware = CacheMiddleware(rate_limit_middleware, cache_config)
        monitoring_middleware = MonitoringMiddleware(cache_middleware, monitoring_config)

        # 创建模拟请求
        request = Mock()
        request.method = "GET"
        request.url.path = "/api/test"
        request.url.query = ""
        request.client = Mock()
        request.client.host = "127.0.0.1"
        request.headers = {}
        request._body = b""

        # 执行中间件链
        async def call_next(request):
            return JSONResponse({"message": "success"})

        response = await monitoring_middleware.dispatch(request, call_next)

        # 验证响应
        assert response.status_code == 200

        # 验证各中间件都添加了相应的头
        assert "X-Request-ID" in response.headers  # 来自监控中间件
        assert "X-Processing-Time" in response.headers  # 来自监控中间件

    @pytest.mark.asyncio
    async def test_middleware_error_propagation(self):
        """测试中间件错误传播"""
        # 创建会在某个中间件中失败的场景
        mock_app = AsyncMock()
        mock_app.side_effect = ValueError("Application error")

        cache_config = {"strategy": "memory_only", "default_ttl": 300}
        cache_middleware = CacheMiddleware(mock_app, cache_config)

        request = Mock()
        request.method = "GET"
        request.url.path = "/api/test"
        request.query_params = {}
        request.headers = {}
        request._body = b""

        async def call_next(request):
            return await mock_app(request)

        # 错误应该正确传播
        with pytest.raises(ValueError):
            await cache_middleware.dispatch(request, call_next)

    def test_middleware_statistics_integration(self):
        """测试中间件统计集成"""
        # 创建各个中间件
        rate_limit_config = {"requests_per_minute": 60}
        cache_config = {"strategy": "memory_only", "default_ttl": 300}
        monitoring_config = {"enable_request_logging": True}

        # 验证管理器正确配置
        rate_limit_manager.configure(rate_limit_config)
        cache_manager.configure(cache_config)
        monitoring_manager.configure(monitoring_config)

        # 验证配置被正确保存
        assert rate_limit_manager.middleware.config == rate_limit_config
        assert cache_manager.middleware.config == cache_config
        assert monitoring_manager.middleware.config == monitoring_config

        # 验证可以获取统计信息
        monitoring_stats = monitoring_manager.middleware.get_stats()
        assert "total_requests" in monitoring_stats
        cache_stats = cache_manager.middleware.get_stats()
        assert "cache_hit_rate" in cache_stats