"""
Plugin Registry and Dynamic Registration System

This module provides a comprehensive plugin registry system with dynamic
registration, discovery, and management capabilities.
"""

import asyncio
import logging
import json
import importlib
import inspect
from pathlib import Path
from typing import Dict, Any, Optional, List, Type, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import hashlib
import threading
from collections import defaultdict

from .plugin_interface import (
    BasePlugin, PluginType, PluginStatus, PluginMetadata,
    PluginContext, PluginResult, PluginFactory
)

logger = logging.getLogger(__name__)


class RegistrationStatus(str, Enum):
    """Plugin registration status."""
    PENDING = "pending"
    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"
    UNREGISTERED = "unregistered"


@dataclass
class PluginRegistration:
    """Plugin registration information."""
    plugin_id: str
    plugin_class: Type[BasePlugin]
    metadata: PluginMetadata
    status: RegistrationStatus = RegistrationStatus.PENDING
    config: Dict[str, Any] = field(default_factory=dict)
    instance: Optional[BasePlugin] = None
    registration_time: datetime = field(default_factory=datetime.now)
    last_activation: Optional[datetime] = None
    error_info: Optional[Dict[str, Any]] = None
    dependencies_satisfied: bool = False
    registration_order: int = 0


@dataclass
class RegistryConfig:
    """Plugin registry configuration."""
    auto_discover: bool = True
    plugin_directories: List[str] = field(default_factory=lambda: ["plugins"])
    auto_load_dependencies: bool = True
    validate_on_registration: bool = True
    enable_version_check: bool = True
    max_concurrent_initializations: int = 5
    plugin_timeout_seconds: float = 30.0
    persist_registry: bool = True
    registry_file: str = "plugin_registry.json"


class PluginRegistry:
    """
    Central plugin registry with dynamic registration capabilities.

    Features:
    - Dynamic plugin registration and discovery
    - Dependency management and resolution
    - Plugin lifecycle management
    - Configuration management
    - Health monitoring
    - Hot loading support
    - Version compatibility checking
    - Registry persistence
    """

    def __init__(self, config: Optional[RegistryConfig] = None):
        """Initialize plugin registry."""
        self.config = config or RegistryConfig()
        self._plugins: Dict[str, PluginRegistration] = {}
        self._plugins_by_type: Dict[PluginType, List[str]] = defaultdict(list)
        self._dependency_graph: Dict[str, List[str]] = {}
        self._registration_order = 0
        self._lock = threading.RLock()
        self._initialization_semaphore = asyncio.Semaphore(
            self.config.max_concurrent_initializations
        )
        self._registry_loaded = False

        # Load existing registry if configured
        if self.config.persist_registry:
            self._load_registry()

        # Auto-discover plugins if configured
        if self.config.auto_discover:
            asyncio.create_task(self._auto_discover_plugins())

    async def register_plugin(
        self,
        plugin_class: Type[BasePlugin],
        config: Optional[Dict[str, Any]] = None,
        force: bool = False
    ) -> str:
        """
        Register a plugin class.

        Args:
            plugin_class: Plugin class to register
            config: Plugin configuration
            force: Force registration even if already registered

        Returns:
            str: Plugin ID

        Raises:
            ValueError: If plugin validation fails
            RuntimeError: If registration fails
        """
        with self._lock:
            # Generate plugin ID
            plugin_id = self._generate_plugin_id(plugin_class)

            # Check if already registered
            if plugin_id in self._plugins and not force:
                logger.warning(f"Plugin {plugin_id} already registered")
                return plugin_id

            # Create temporary instance to get metadata
            try:
                temp_instance = plugin_class(config)
                metadata = temp_instance.get_metadata()
            except Exception as e:
                logger.error(f"Failed to create temporary instance of {plugin_class.__name__}: {e}")
                raise ValueError(f"Invalid plugin class: {e}")

            # Validate plugin
            if self.config.validate_on_registration:
                await self._validate_plugin(plugin_class, metadata)

            # Create registration
            self._registration_order += 1
            registration = PluginRegistration(
                plugin_id=plugin_id,
                plugin_class=plugin_class,
                metadata=metadata,
                config=config or {},
                registration_order=self._registration_order
            )

            # Check dependencies
            registration.dependencies_satisfied = await self._check_dependencies(
                metadata.dependencies
            )

            # Add to registry
            self._plugins[plugin_id] = registration
            self._plugins_by_type[metadata.plugin_type].append(plugin_id)

            # Update dependency graph
            self._dependency_graph[plugin_id] = metadata.dependencies

            # Update status
            registration.status = RegistrationStatus.REGISTERED

            logger.info(f"Registered plugin {plugin_id} of type {metadata.plugin_type.value}")

            # Save registry if configured
            if self.config.persist_registry:
                await self._save_registry()

            return plugin_id

    async def unregister_plugin(self, plugin_id: str, cleanup: bool = True) -> bool:
        """
        Unregister a plugin.

        Args:
            plugin_id: Plugin ID to unregister
            cleanup: Whether to cleanup plugin resources

        Returns:
            bool: True if unregistration successful
        """
        with self._lock:
            if plugin_id not in self._plugins:
                logger.warning(f"Plugin {plugin_id} not found in registry")
                return False

            registration = self._plugins[plugin_id]

            # Cleanup if instance exists
            if cleanup and registration.instance:
                try:
                    await registration.instance.cleanup()
                    registration.instance = None
                except Exception as e:
                    logger.error(f"Failed to cleanup plugin {plugin_id}: {e}")

            # Remove from registry
            del self._plugins[plugin_id]

            # Remove from type mapping
            if registration.metadata.plugin_type in self._plugins_by_type:
                if plugin_id in self._plugins_by_type[registration.metadata.plugin_type]:
                    self._plugins_by_type[registration.metadata.plugin_type].remove(plugin_id)

            # Remove from dependency graph
            if plugin_id in self._dependency_graph:
                del self._dependency_graph[plugin_id]

            # Update status
            registration.status = RegistrationStatus.UNREGISTERED

            logger.info(f"Unregistered plugin {plugin_id}")

            # Save registry if configured
            if self.config.persist_registry:
                await self._save_registry()

            return True

    async def activate_plugin(self, plugin_id: str) -> bool:
        """
        Activate a registered plugin.

        Args:
            plugin_id: Plugin ID to activate

        Returns:
            bool: True if activation successful
        """
        if plugin_id not in self._plugins:
            logger.error(f"Plugin {plugin_id} not found in registry")
            return False

        registration = self._plugins[plugin_id]

        # Check if already active
        if registration.status == RegistrationStatus.ACTIVE:
            logger.info(f"Plugin {plugin_id} already active")
            return True

        # Check dependencies
        if not registration.dependencies_satisfied:
            logger.error(f"Dependencies not satisfied for plugin {plugin_id}")
            return False

        async with self._initialization_semaphore:
            try:
                # Create instance if not exists
                if not registration.instance:
                    registration.instance = PluginFactory.create_plugin(
                        registration.plugin_class,
                        registration.config
                    )

                # Initialize plugin
                success = await registration.instance.initialize()
                if not success:
                    raise RuntimeError("Plugin initialization failed")

                # Update registration
                registration.status = RegistrationStatus.ACTIVE
                registration.last_activation = datetime.now()
                registration.instance.status = PluginStatus.ACTIVE

                logger.info(f"Activated plugin {plugin_id}")

                # Save registry if configured
                if self.config.persist_registry:
                    await self._save_registry()

                return True

            except Exception as e:
                logger.error(f"Failed to activate plugin {plugin_id}: {e}")
                registration.status = RegistrationStatus.ERROR
                registration.error_info = {
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                return False

    async def deactivate_plugin(self, plugin_id: str, cleanup: bool = True) -> bool:
        """
        Deactivate an active plugin.

        Args:
            plugin_id: Plugin ID to deactivate
            cleanup: Whether to cleanup plugin resources

        Returns:
            bool: True if deactivation successful
        """
        if plugin_id not in self._plugins:
            logger.error(f"Plugin {plugin_id} not found in registry")
            return False

        registration = self._plugins[plugin_id]

        if registration.status != RegistrationStatus.ACTIVE:
            logger.info(f"Plugin {plugin_id} not active")
            return True

        try:
            # Cleanup if requested and instance exists
            if cleanup and registration.instance:
                await registration.instance.cleanup()

            # Update registration
            registration.status = RegistrationStatus.INACTIVE
            if registration.instance:
                registration.instance.status = PluginStatus.INACTIVE

            logger.info(f"Deactivated plugin {plugin_id}")

            # Save registry if configured
            if self.config.persist_registry:
                await self._save_registry()

            return True

        except Exception as e:
            logger.error(f"Failed to deactivate plugin {plugin_id}: {e}")
            return False

    def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """
        Get plugin instance by ID.

        Args:
            plugin_id: Plugin ID

        Returns:
            Optional[BasePlugin]: Plugin instance or None
        """
        registration = self._plugins.get(plugin_id)
        return registration.instance if registration else None

    def get_plugin_info(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """
        Get plugin information by ID.

        Args:
            plugin_id: Plugin ID

        Returns:
            Optional[Dict[str, Any]]: Plugin information or None
        """
        if plugin_id not in self._plugins:
            return None

        registration = self._plugins[plugin_id]
        info = {
            'plugin_id': plugin_id,
            'metadata': asdict(registration.metadata),
            'status': registration.status.value,
            'config': registration.config,
            'registration_time': registration.registration_time.isoformat(),
            'last_activation': registration.last_activation.isoformat() if registration.last_activation else None,
            'dependencies_satisfied': registration.dependencies_satisfied,
            'registration_order': registration.registration_order
        }

        if registration.instance:
            info['instance_stats'] = registration.instance.get_stats()
            info['instance_status'] = registration.instance.get_status().value

        if registration.error_info:
            info['error_info'] = registration.error_info

        return info

    def list_plugins(
        self,
        plugin_type: Optional[PluginType] = None,
        status: Optional[RegistrationStatus] = None,
        active_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List registered plugins.

        Args:
            plugin_type: Filter by plugin type
            status: Filter by registration status
            active_only: Return only active plugins

        Returns:
            List[Dict[str, Any]]: List of plugin information
        """
        plugins = []

        for plugin_id, registration in self._plugins.items():
            # Apply filters
            if plugin_type and registration.metadata.plugin_type != plugin_type:
                continue

            if status and registration.status != status:
                continue

            if active_only and registration.status != RegistrationStatus.ACTIVE:
                continue

            plugins.append(self.get_plugin_info(plugin_id))

        return plugins

    def get_plugins_by_type(self, plugin_type: PluginType) -> List[str]:
        """
        Get plugin IDs by type.

        Args:
            plugin_type: Plugin type

        Returns:
            List[str]: List of plugin IDs
        """
        return self._plugins_by_type.get(plugin_type, []).copy()

    async def discover_plugins(self, directories: Optional[List[str]] = None) -> List[str]:
        """
        Discover plugins in specified directories.

        Args:
            directories: Directories to search (uses config if None)

        Returns:
            List[str]: List of discovered plugin IDs
        """
        search_dirs = directories or self.config.plugin_directories
        discovered_plugins = []

        for directory in search_dirs:
            try:
                plugins = await self._discover_plugins_in_directory(directory)
                discovered_plugins.extend(plugins)
            except Exception as e:
                logger.error(f"Failed to discover plugins in {directory}: {e}")

        return discovered_plugins

    async def _discover_plugins_in_directory(self, directory: str) -> List[str]:
        """Discover plugins in a specific directory."""
        discovered_plugins = []
        plugin_dir = Path(directory)

        if not plugin_dir.exists():
            logger.warning(f"Plugin directory {directory} does not exist")
            return discovered_plugins

        # Look for Python files
        for py_file in plugin_dir.rglob("*.py"):
            if py_file.name.startswith("__"):
                continue

            try:
                # Import module and look for plugin classes
                module_name = self._path_to_module_name(py_file, plugin_dir)
                plugin_classes = await self._find_plugin_classes_in_module(module_name)

                for plugin_class in plugin_classes:
                    plugin_id = await self.register_plugin(plugin_class)
                    discovered_plugins.append(plugin_id)

            except Exception as e:
                logger.error(f"Failed to load plugin from {py_file}: {e}")

        return discovered_plugins

    def _path_to_module_name(self, file_path: Path, base_dir: Path) -> str:
        """Convert file path to module name."""
        relative_path = file_path.relative_to(base_dir)
        module_parts = list(relative_path.parts[:-1])  # Exclude filename
        module_parts.append(relative_path.stem)  # Add filename without extension
        return ".".join(module_parts)

    async def _find_plugin_classes_in_module(self, module_name: str) -> List[Type[BasePlugin]]:
        """Find plugin classes in a module."""
        plugin_classes = []

        try:
            module = importlib.import_module(module_name)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BasePlugin) and
                    obj != BasePlugin and
                    obj.__module__ == module_name):
                    plugin_classes.append(obj)

        except Exception as e:
            logger.error(f"Failed to import module {module_name}: {e}")

        return plugin_classes

    async def _validate_plugin(self, plugin_class: Type[BasePlugin], metadata: PluginMetadata):
        """Validate plugin class and metadata."""
        # Check required methods
        required_methods = ['get_metadata', 'initialize', 'cleanup', 'process']
        for method in required_methods:
            if not hasattr(plugin_class, method):
                raise ValueError(f"Plugin class missing required method: {method}")

        # Check metadata completeness
        if not metadata.name or not metadata.version or not metadata.plugin_type:
            raise ValueError("Plugin metadata missing required fields")

        # Version compatibility check
        if self.config.enable_version_check:
            # Add version compatibility logic here
            pass

        # Validate configuration schema
        if metadata.configuration_schema:
            # Add schema validation logic here
            pass

    async def _check_dependencies(self, dependencies: List[str]) -> bool:
        """Check if plugin dependencies are satisfied."""
        for dep in dependencies:
            if dep not in self._plugins:
                return False

            registration = self._plugins[dep]
            if registration.status != RegistrationStatus.ACTIVE:
                return False

        return True

    def _generate_plugin_id(self, plugin_class: Type[BasePlugin]) -> str:
        """Generate unique plugin ID."""
        # Use class name and module to create unique ID
        class_name = plugin_class.__name__
        module_name = plugin_class.__module__
        combined = f"{module_name}.{class_name}"
        return hashlib.md5(combined.encode()).hexdigest()[:16]

    async def _auto_discover_plugins(self):
        """Auto-discover plugins on startup."""
        try:
            discovered = await self.discover_plugins()
            logger.info(f"Auto-discovered {len(discovered)} plugins")
        except Exception as e:
            logger.error(f"Auto-discovery failed: {e}")

    async def _save_registry(self):
        """Save registry to file."""
        if not self.config.persist_registry:
            return

        try:
            registry_data = {}
            for plugin_id, registration in self._plugins.items():
                registry_data[plugin_id] = {
                    'plugin_class': f"{registration.plugin_class.__module__}.{registration.plugin_class.__name__}",
                    'metadata': asdict(registration.metadata),
                    'status': registration.status.value,
                    'config': registration.config,
                    'registration_time': registration.registration_time.isoformat(),
                    'last_activation': registration.last_activation.isoformat() if registration.last_activation else None,
                    'dependencies_satisfied': registration.dependencies_satisfied,
                    'registration_order': registration.registration_order,
                    'error_info': registration.error_info
                }

            registry_file = Path(self.config.registry_file)
            with open(registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def _load_registry(self):
        """Load registry from file."""
        if not self.config.persist_registry:
            return

        try:
            registry_file = Path(self.config.registry_file)
            if not registry_file.exists():
                return

            with open(registry_file, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)

            # This is a simplified reload - in practice, you'd need to reconstruct
            # the plugin classes and instances properly
            logger.info(f"Registry loaded from {registry_file}")
            self._registry_loaded = True

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform registry health check."""
        total_plugins = len(self._plugins)
        active_plugins = len([
            p for p in self._plugins.values()
            if p.status == RegistrationStatus.ACTIVE
        ])
        error_plugins = len([
            p for p in self._plugins.values()
            if p.status == RegistrationStatus.ERROR
        ])

        # Check plugin health
        unhealthy_plugins = []
        for plugin_id, registration in self._plugins.items():
            if registration.instance:
                try:
                    is_healthy = await registration.instance.health_check()
                    if not is_healthy:
                        unhealthy_plugins.append(plugin_id)
                except Exception as e:
                    logger.error(f"Health check failed for plugin {plugin_id}: {e}")
                    unhealthy_plugins.append(plugin_id)

        return {
            'total_plugins': total_plugins,
            'active_plugins': active_plugins,
            'error_plugins': error_plugins,
            'unhealthy_plugins': len(unhealthy_plugins),
            'unhealthy_plugin_ids': unhealthy_plugins,
            'registry_loaded': self._registry_loaded,
            'auto_discovery_enabled': self.config.auto_discover
        }

    async def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get plugin dependency graph."""
        return self._dependency_graph.copy()

    async def resolve_dependencies(self, plugin_id: str) -> List[str]:
        """
        Get resolved dependency order for a plugin.

        Args:
            plugin_id: Plugin ID

        Returns:
            List[str]: Dependency resolution order
        """
        if plugin_id not in self._plugins:
            return []

        visited = set()
        temp_visited = set()
        resolution_order = []

        def visit(node_id: str):
            if node_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving {node_id}")
            if node_id in visited:
                return

            temp_visited.add(node_id)

            for dep_id in self._dependency_graph.get(node_id, []):
                if dep_id in self._plugins:
                    visit(dep_id)

            temp_visited.remove(node_id)
            visited.add(node_id)
            resolution_order.append(node_id)

        visit(plugin_id)
        return resolution_order

    async def activate_with_dependencies(self, plugin_id: str) -> bool:
        """
        Activate plugin with its dependencies.

        Args:
            plugin_id: Plugin ID to activate

        Returns:
            bool: True if activation successful
        """
        try:
            # Get dependency resolution order
            dep_order = await self.resolve_dependencies(plugin_id)

            # Activate dependencies first
            for dep_id in dep_order:
                if dep_id != plugin_id:
                    success = await self.activate_plugin(dep_id)
                    if not success:
                        logger.error(f"Failed to activate dependency {dep_id} for {plugin_id}")
                        return False

            # Activate the main plugin
            return await self.activate_plugin(plugin_id)

        except Exception as e:
            logger.error(f"Failed to activate plugin {plugin_id} with dependencies: {e}")
            return False


# Global registry instance
_global_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """Get global plugin registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def set_registry(registry: PluginRegistry):
    """Set global plugin registry instance."""
    global _global_registry
    _global_registry = registry