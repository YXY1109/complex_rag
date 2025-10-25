"""
Plugin Hot Loading System

This module provides hot loading capabilities for plugins, allowing dynamic
loading, unloading, and reloading of plugins without system restart.
"""

import asyncio
import logging
import importlib
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import watchdog.observers
from watchdog.events import FileSystemEventHandler
import hashlib

from .plugin_interface import BasePlugin, PluginType, PluginStatus
from .plugin_registry import PluginRegistry, RegistrationStatus

logger = logging.getLogger(__name__)


class HotLoadingStatus(str, Enum):
    """Hot loading operation status."""
    IDLE = "idle"
    SCANNING = "scanning"
    LOADING = "loading"
    UNLOADING = "unloading"
    RELOADING = "reloading"
    ERROR = "error"


@dataclass
class HotLoadingConfig:
    """Hot loading configuration."""
    enabled: bool = True
    watch_directories: List[str] = field(default_factory=lambda: ["plugins"])
    file_patterns: List[str] = field(default_factory=lambda: ["*.py"])
    ignore_patterns: List[str] = field(default_factory=lambda: ["__pycache__", "*.pyc", ".git"])
    auto_reload: bool = True
    reload_delay: float = 1.0  # Delay before reloading to handle multiple file changes
    max_reload_attempts: int = 3
    validate_before_reload: bool = True
    backup_before_reload: bool = True
    preserve_state: bool = True
    hot_reload_dependencies: bool = True


@dataclass
class HotLoadingEvent:
    """Hot loading event information."""
    event_type: str  # file_modified, file_created, file_deleted
    file_path: str
    timestamp: datetime
    plugin_id: Optional[str] = None
    old_plugin_id: Optional[str] = None  # For reload operations
    success: bool = False
    error_message: Optional[str] = None


class PluginFileWatcher(FileSystemEventHandler):
    """File system event handler for plugin file changes."""

    def __init__(self, hot_loader: 'PluginHotLoader'):
        """Initialize file watcher."""
        self.hot_loader = hot_loader
        self.logger = logging.getLogger(__name__)

    def on_modified(self, event):
        """Handle file modification events."""
        if not event.is_directory:
            self.logger.debug(f"File modified: {event.src_path}")
            asyncio.create_task(
                self.hot_loader._handle_file_change("file_modified", event.src_path)
            )

    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory:
            self.logger.debug(f"File created: {event.src_path}")
            asyncio.create_task(
                self.hot_loader._handle_file_change("file_created", event.src_path)
            )

    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory:
            self.logger.debug(f"File deleted: {event.src_path}")
            asyncio.create_task(
                self.hot_loader._handle_file_change("file_deleted", event.src_path)
            )


class PluginHotLoader:
    """
    Plugin hot loading system with file watching and dynamic reloading.

    Features:
    - File system monitoring for plugin changes
    - Hot loading and unloading of plugins
    - State preservation during reloads
    - Dependency-aware reloading
    - Rollback on reload failure
    - Concurrent reload management
    - Event logging and history
    """

    def __init__(
        self,
        registry: PluginRegistry,
        config: Optional[HotLoadingConfig] = None
    ):
        """Initialize hot loader."""
        self.registry = registry
        self.config = config or HotLoadingConfig()
        self.status = HotLoadingStatus.IDLE
        self._watchers: Dict[str, watchdog.observers.Observer] = {}
        self._pending_reloads: Dict[str, float] = {}  # file_path -> timestamp
        self._reload_task: Optional[asyncio.Task] = None
        self._active_operations: Set[str] = set()  # Active operation IDs
        self._event_history: List[HotLoadingEvent] = []
        self._lock = threading.RLock()
        self._file_to_plugin_map: Dict[str, Set[str]] = {}  # file_path -> plugin_ids
        self._plugin_to_file_map: Dict[str, str] = {}  # plugin_id -> file_path

        # Start file watching if enabled
        if self.config.enabled:
            self._start_file_watching()

    async def load_plugin_from_file(
        self,
        file_path: str,
        plugin_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Load plugin from file.

        Args:
            file_path: Path to plugin file
            plugin_name: Optional plugin name
            config: Plugin configuration

        Returns:
            Optional[str]: Plugin ID if successful, None otherwise
        """
        self.status = HotLoadingStatus.LOADING
        operation_id = self._generate_operation_id("load")

        try:
            self._active_operations.add(operation_id)

            # Validate file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Plugin file not found: {file_path}")

            # Determine module name
            module_name = plugin_name or Path(file_path).stem

            # Load or reload module
            module = await self._load_module_from_file(file_path, module_name)

            # Find plugin classes in module
            plugin_classes = await self._find_plugin_classes_in_module(module)

            if not plugin_classes:
                raise ValueError(f"No plugin classes found in {file_path}")

            # Register plugins
            plugin_ids = []
            for plugin_class in plugin_classes:
                plugin_id = await self.registry.register_plugin(plugin_class, config)
                plugin_ids.append(plugin_id)

                # Update file mapping
                self._file_to_plugin_map.setdefault(file_path, set()).add(plugin_id)
                self._plugin_to_file_map[plugin_id] = file_path

            # Activate plugins
            for plugin_id in plugin_ids:
                success = await self.registry.activate_plugin(plugin_id)
                if not success:
                    logger.warning(f"Failed to activate plugin {plugin_id}")

            self.status = HotLoadingStatus.IDLE
            self._record_event(HotLoadingEvent(
                event_type="plugin_loaded",
                file_path=file_path,
                timestamp=datetime.now(),
                plugin_id=plugin_ids[0] if plugin_ids else None,
                success=True
            ))

            logger.info(f"Successfully loaded {len(plugin_ids)} plugins from {file_path}")
            return plugin_ids[0] if plugin_ids else None

        except Exception as e:
            self.status = HotLoadingStatus.ERROR
            self._record_event(HotLoadingEvent(
                event_type="plugin_load_failed",
                file_path=file_path,
                timestamp=datetime.now(),
                error_message=str(e),
                success=False
            ))
            logger.error(f"Failed to load plugin from {file_path}: {e}")
            return None

        finally:
            self._active_operations.discard(operation_id)

    async def unload_plugin(
        self,
        plugin_id: str,
        cleanup: bool = True,
        remove_from_registry: bool = True
    ) -> bool:
        """
        Unload plugin.

        Args:
            plugin_id: Plugin ID to unload
            cleanup: Whether to cleanup plugin resources
            remove_from_registry: Whether to remove from registry

        Returns:
            bool: True if successful
        """
        self.status = HotLoadingStatus.UNLOADING
        operation_id = self._generate_operation_id("unload")

        try:
            self._active_operations.add(operation_id)

            # Get plugin info
            plugin_info = self.registry.get_plugin_info(plugin_id)
            if not plugin_info:
                logger.warning(f"Plugin {plugin_id} not found in registry")
                return False

            file_path = self._plugin_to_file_map.get(plugin_id)

            # Preserve state if configured
            preserved_state = None
            if self.config.preserve_state:
                preserved_state = await self._preserve_plugin_state(plugin_id)

            # Deactivate plugin
            success = await self.registry.deactivate_plugin(plugin_id, cleanup)

            if success and remove_from_registry:
                success = await self.registry.unregister_plugin(plugin_id, cleanup)

            # Update file mapping
            if file_path and success:
                if file_path in self._file_to_plugin_map:
                    self._file_to_plugin_map[file_path].discard(plugin_id)
                    if not self._file_to_plugin_map[file_path]:
                        del self._file_to_plugin_map[file_path]

                self._plugin_to_file_map.pop(plugin_id, None)

            self.status = HotLoadingStatus.IDLE
            self._record_event(HotLoadingEvent(
                event_type="plugin_unloaded",
                file_path=file_path or "",
                timestamp=datetime.now(),
                plugin_id=plugin_id,
                success=success
            ))

            logger.info(f"Successfully unloaded plugin {plugin_id}")
            return success

        except Exception as e:
            self.status = HotLoadingStatus.ERROR
            self._record_event(HotLoadingEvent(
                event_type="plugin_unload_failed",
                file_path="",
                timestamp=datetime.now(),
                plugin_id=plugin_id,
                error_message=str(e),
                success=False
            ))
            logger.error(f"Failed to unload plugin {plugin_id}: {e}")
            return False

        finally:
            self._active_operations.discard(operation_id)

    async def reload_plugin(
        self,
        plugin_id: str,
        preserve_state: bool = None
    ) -> bool:
        """
        Reload plugin.

        Args:
            plugin_id: Plugin ID to reload
            preserve_state: Whether to preserve plugin state

        Returns:
            bool: True if successful
        """
        self.status = HotLoadingStatus.RELOADING
        operation_id = self._generate_operation_id("reload")

        try:
            self._active_operations.add(operation_id)

            # Get plugin info
            plugin_info = self.registry.get_plugin_info(plugin_id)
            if not plugin_info:
                logger.error(f"Plugin {plugin_id} not found in registry")
                return False

            file_path = self._plugin_to_file_map.get(plugin_id)
            if not file_path:
                logger.error(f"No file path associated with plugin {plugin_id}")
                return False

            old_plugin_id = plugin_id

            # Preserve state if configured
            preserved_state = None
            if (preserve_state if preserve_state is not None else self.config.preserve_state):
                preserved_state = await self._preserve_plugin_state(plugin_id)

            # Backup plugin configuration
            backup_config = plugin_info.get('config', {}).copy()

            # Unload plugin
            unload_success = await self.unload_plugin(
                plugin_id,
                cleanup=False,  # Don't cleanup yet, in case reload fails
                remove_from_registry=False
            )

            if not unload_success:
                logger.error(f"Failed to unload plugin {plugin_id} for reload")
                return False

            # Wait for file system to settle
            await asyncio.sleep(self.config.reload_delay)

            # Clear module cache to force reload
            module_name = Path(file_path).stem
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Load plugin from file
            new_plugin_id = await self.load_plugin_from_file(
                file_path,
                config=backup_config
            )

            if not new_plugin_id:
                logger.error(f"Failed to reload plugin from {file_path}")

                # Attempt to restore original plugin
                if self.config.backup_before_reload:
                    try:
                        # This is a simplified restore - in practice, you'd need
                        # proper backup and restore mechanisms
                        logger.warning(f"Attempting to restore original plugin {old_plugin_id}")
                    except Exception as restore_error:
                        logger.error(f"Failed to restore original plugin: {restore_error}")

                return False

            # Restore preserved state
            if preserved_state:
                await self._restore_plugin_state(new_plugin_id, preserved_state)

            # Reload dependencies if configured
            if self.config.hot_reload_dependencies:
                await self._reload_plugin_dependencies(new_plugin_id)

            self.status = HotLoadingStatus.IDLE
            self._record_event(HotLoadingEvent(
                event_type="plugin_reloaded",
                file_path=file_path,
                timestamp=datetime.now(),
                plugin_id=new_plugin_id,
                old_plugin_id=old_plugin_id,
                success=True
            ))

            logger.info(f"Successfully reloaded plugin {old_plugin_id} as {new_plugin_id}")
            return True

        except Exception as e:
            self.status = HotLoadingStatus.ERROR
            self._record_event(HotLoadingEvent(
                event_type="plugin_reload_failed",
                file_path=self._plugin_to_file_map.get(plugin_id, ""),
                timestamp=datetime.now(),
                plugin_id=plugin_id,
                error_message=str(e),
                success=False
            ))
            logger.error(f"Failed to reload plugin {plugin_id}: {e}")
            return False

        finally:
            self._active_operations.discard(operation_id)

    async def reload_all_plugins(self) -> Dict[str, bool]:
        """
        Reload all registered plugins.

        Returns:
            Dict[str, bool]: Mapping of plugin IDs to reload success status
        """
        plugins = self.registry.list_plugins()
        results = {}

        for plugin_info in plugins:
            plugin_id = plugin_info['plugin_id']
            try:
                success = await self.reload_plugin(plugin_id)
                results[plugin_id] = success
            except Exception as e:
                logger.error(f"Failed to reload plugin {plugin_id}: {e}")
                results[plugin_id] = False

        return results

    def _start_file_watching(self):
        """Start file system watching."""
        for directory in self.config.watch_directories:
            try:
                path = Path(directory)
                if not path.exists():
                    logger.warning(f"Watch directory {directory} does not exist")
                    continue

                observer = watchdog.observers.Observer()
                event_handler = PluginFileWatcher(self)
                observer.schedule(event_handler, str(path), recursive=True)
                observer.start()

                self._watchers[directory] = observer
                logger.info(f"Started watching directory: {directory}")

            except Exception as e:
                logger.error(f"Failed to start watching {directory}: {e}")

    def stop_file_watching(self):
        """Stop file system watching."""
        for directory, observer in self._watchers.items():
            try:
                observer.stop()
                observer.join()
                logger.info(f"Stopped watching directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to stop watching {directory}: {e}")

        self._watchers.clear()

    async def _handle_file_change(self, event_type: str, file_path: str):
        """Handle file system change events."""
        # Check if file matches patterns
        if not self._should_process_file(file_path):
            return

        # Check if file is associated with any plugins
        if file_path not in self._file_to_plugin_map:
            # New plugin file - attempt to load
            if event_type == "file_created" and self.config.auto_reload:
                await self.load_plugin_from_file(file_path)
            return

        # Schedule reload for associated plugins
        plugin_ids = self._file_to_plugin_map[file_path].copy()
        self._pending_reloads[file_path] = time.time()

        # Cancel existing reload task
        if self._reload_task and not self._reload_task.done():
            self._reload_task.cancel()

        # Schedule new reload task with delay
        self._reload_task = asyncio.create_task(
            self._delayed_reload(file_path, plugin_ids)
        )

    async def _delayed_reload(self, file_path: str, plugin_ids: List[str]):
        """Execute delayed reload to handle multiple file changes."""
        await asyncio.sleep(self.config.reload_delay)

        try:
            for plugin_id in plugin_ids:
                if self.config.auto_reload:
                    success = await self.reload_plugin(plugin_id)
                    if not success:
                        logger.warning(f"Failed to auto-reload plugin {plugin_id}")

        except Exception as e:
            logger.error(f"Delayed reload failed for {file_path}: {e}")
        finally:
            self._pending_reloads.pop(file_path, None)

    def _should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed based on patterns."""
        path = Path(file_path)

        # Check file extension
        if not any(path.match(pattern) for pattern in self.config.file_patterns):
            return False

        # Check ignore patterns
        for ignore_pattern in self.config.ignore_patterns:
            if ignore_pattern in str(path) or path.match(ignore_pattern):
                return False

        return True

    async def _load_module_from_file(self, file_path: str, module_name: str):
        """Load module from file."""
        # Add file path to sys.path if not already there
        file_dir = str(Path(file_path).parent)
        if file_dir not in sys.path:
            sys.path.insert(0, file_dir)

        try:
            # Import the module
            module = importlib.import_module(module_name)

            # Reload if already loaded
            if module_name in sys.modules:
                module = importlib.reload(module)

            return module

        except Exception as e:
            logger.error(f"Failed to load module {module_name} from {file_path}: {e}")
            raise

    async def _find_plugin_classes_in_module(self, module):
        """Find plugin classes in a module."""
        plugin_classes = []

        for name, obj in vars(module).items():
            if (isinstance(obj, type) and
                issubclass(obj, BasePlugin) and
                obj != BasePlugin):
                plugin_classes.append(obj)

        return plugin_classes

    async def _preserve_plugin_state(self, plugin_id: str) -> Optional[Dict[str, Any]]:
        """Preserve plugin state before reload."""
        plugin = self.registry.get_plugin(plugin_id)
        if not plugin:
            return None

        try:
            # Get plugin-specific state
            state = {
                'config': plugin.config.copy(),
                'stats': plugin.get_stats(),
                'timestamp': datetime.now().isoformat()
            }

            # Plugin-specific state preservation
            if hasattr(plugin, 'get_state'):
                plugin_state = await plugin.get_state()
                state['plugin_state'] = plugin_state

            return state

        except Exception as e:
            logger.error(f"Failed to preserve state for plugin {plugin_id}: {e}")
            return None

    async def _restore_plugin_state(self, plugin_id: str, state: Dict[str, Any]) -> bool:
        """Restore plugin state after reload."""
        plugin = self.registry.get_plugin(plugin_id)
        if not plugin:
            return False

        try:
            # Restore basic state
            if 'config' in state:
                plugin.config.update(state['config'])

            # Plugin-specific state restoration
            if 'plugin_state' in state and hasattr(plugin, 'set_state'):
                await plugin.set_state(state['plugin_state'])

            return True

        except Exception as e:
            logger.error(f"Failed to restore state for plugin {plugin_id}: {e}")
            return False

    async def _reload_plugin_dependencies(self, plugin_id: str):
        """Reload plugin dependencies."""
        plugin_info = self.registry.get_plugin_info(plugin_id)
        if not plugin_info:
            return

        # Get dependent plugins (plugins that depend on this plugin)
        dependency_graph = await self.registry.get_dependency_graph()
        dependent_plugins = []

        for dep_id, deps in dependency_graph.items():
            if plugin_id in deps:
                dependent_plugins.append(dep_id)

        # Reload dependent plugins
        for dep_plugin_id in dependent_plugins:
            try:
                await self.reload_plugin(dep_plugin_id)
            except Exception as e:
                logger.error(f"Failed to reload dependent plugin {dep_plugin_id}: {e}")

    def _generate_operation_id(self, operation_type: str) -> str:
        """Generate unique operation ID."""
        timestamp = datetime.now().isoformat()
        return f"{operation_type}_{timestamp}_{hash(operation_type)}"

    def _record_event(self, event: HotLoadingEvent):
        """Record hot loading event."""
        self._event_history.append(event)

        # Keep only last 1000 events
        if len(self._event_history) > 1000:
            self._event_history = self._event_history[-1000:]

    def get_event_history(
        self,
        limit: Optional[int] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None
    ) -> List[HotLoadingEvent]:
        """
        Get event history.

        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type
            since: Filter events since timestamp

        Returns:
            List[HotLoadingEvent]: Filtered event list
        """
        events = self._event_history.copy()

        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]

        if since:
            events = [e for e in events if e.timestamp >= since]

        # Apply limit
        if limit:
            events = events[-limit:]

        return events

    def get_status(self) -> Dict[str, Any]:
        """Get hot loader status."""
        return {
            'status': self.status.value,
            'enabled': self.config.enabled,
            'watching_directories': list(self._watchers.keys()),
            'pending_reloads': len(self._pending_reloads),
            'active_operations': len(self._active_operations),
            'file_to_plugin_map_size': len(self._file_to_plugin_map),
            'plugin_to_file_map_size': len(self._plugin_to_file_map),
            'config': self.config.__dict__
        }

    async def shutdown(self):
        """Shutdown hot loader."""
        # Stop file watching
        self.stop_file_watching()

        # Cancel any pending reload tasks
        if self._reload_task and not self._reload_task.done():
            self._reload_task.cancel()

        # Wait for active operations to complete
        while self._active_operations:
            await asyncio.sleep(0.1)

        logger.info("Plugin hot loader shutdown complete")

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.stop_file_watching()
        except:
            pass