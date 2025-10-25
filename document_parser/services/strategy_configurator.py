"""
Processing Strategy Configurator

This module provides configuration management for processing strategies,
including default settings, custom configurations, and dynamic adjustments.
"""

import json
import yaml
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from ..interfaces.source_interface import (
    FileSource,
    ProcessingStrategy,
    SourceConfig
)
from .processing_strategy_selector import StrategyParams, ProcessingRequirement


class ConfigFormat(str, Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"


@dataclass
class SourceStrategyConfig:
    """Configuration for a specific source type."""
    source: FileSource
    default_strategy: ProcessingStrategy
    strategy_params: Dict[ProcessingStrategy, StrategyParams]
    fallback_strategies: List[ProcessingStrategy]
    quality_thresholds: Dict[str, float]
    custom_settings: Dict[str, Any]


@dataclass
class GlobalStrategyConfig:
    """Global strategy configuration."""
    default_strategy: ProcessingStrategy = ProcessingStrategy.AUTO
    enable_auto_optimization: bool = True
    learning_enabled: bool = True
    performance_tracking: bool = True
    max_concurrent_processing: int = 4
    memory_limit_mb: Optional[int] = None
    timeout_multiplier: float = 1.0
    retry_attempts: int = 3
    adaptive_quality: bool = True


class StrategyConfigurator:
    """
    Configurator for processing strategies with support for
    file-based configuration and dynamic adjustments.
    """

    def __init__(self, config_file: Optional[Union[str, Path]] = None):
        """Initialize configurator with optional config file."""
        self.config_file = Path(config_file) if config_file else None
        self.global_config = GlobalStrategyConfig()
        self.source_configs: Dict[FileSource, SourceStrategyConfig] = {}

        self._load_default_configurations()
        if self.config_file and self.config_file.exists():
            self._load_config_file()

    def _load_default_configurations(self):
        """Load default configurations for all source types."""

        # Web Documents Configuration
        self.source_configs[FileSource.WEB_DOCUMENTS] = SourceStrategyConfig(
            source=FileSource.WEB_DOCUMENTS,
            default_strategy=ProcessingStrategy.BALANCED,
            strategy_params={
                ProcessingStrategy.FAST: StrategyParams(
                    chunk_size=1200, overlap_size=120, use_ocr=False,
                    use_vision=False, extract_images=False, extract_tables=False,
                    preserve_formatting=False, parallel_processing=True,
                    quality_threshold=0.7, timeout_seconds=30
                ),
                ProcessingStrategy.BALANCED: StrategyParams(
                    chunk_size=800, overlap_size=160, use_ocr=False,
                    use_vision=True, extract_images=True, extract_tables=True,
                    preserve_formatting=True, parallel_processing=True,
                    quality_threshold=0.8, timeout_seconds=45
                ),
                ProcessingStrategy.ACCURATE: StrategyParams(
                    chunk_size=600, overlap_size=120, use_ocr=False,
                    use_vision=True, extract_images=True, extract_tables=True,
                    preserve_formatting=True, parallel_processing=False,
                    quality_threshold=0.9, timeout_seconds=90
                ),
                ProcessingStrategy.AUTO: StrategyParams(
                    chunk_size=800, overlap_size=160, use_ocr=False,
                    use_vision=True, extract_images=True, extract_tables=True,
                    preserve_formatting=True, parallel_processing=True,
                    quality_threshold=0.85, timeout_seconds=60
                )
            },
            fallback_strategies=[ProcessingStrategy.FAST, ProcessingStrategy.BALANCED],
            quality_thresholds={
                'text_extraction': 0.8,
                'structure_preservation': 0.7,
                'link_detection': 0.9
            },
            custom_settings={
                'respect_robots_txt': True,
                'follow_redirects': True,
                'user_agent': 'RAG-Parser/1.0',
                'max_page_size_mb': 10,
                'javascript_rendering': False
            }
        )

        # Office Documents Configuration
        self.source_configs[FileSource.OFFICE_DOCUMENTS] = SourceStrategyConfig(
            source=FileSource.OFFICE_DOCUMENTS,
            default_strategy=ProcessingStrategy.BALANCED,
            strategy_params={
                ProcessingStrategy.FAST: StrategyParams(
                    chunk_size=1000, overlap_size=100, use_ocr=False,
                    use_vision=False, extract_images=False, extract_tables=True,
                    preserve_formatting=True, parallel_processing=True,
                    quality_threshold=0.75, timeout_seconds=60
                ),
                ProcessingStrategy.BALANCED: StrategyParams(
                    chunk_size=800, overlap_size=160, use_ocr=True,
                    use_vision=True, extract_images=True, extract_tables=True,
                    preserve_formatting=True, parallel_processing=True,
                    quality_threshold=0.85, timeout_seconds=90
                ),
                ProcessingStrategy.ACCURATE: StrategyParams(
                    chunk_size=500, overlap_size=100, use_ocr=True,
                    use_vision=True, extract_images=True, extract_tables=True,
                    preserve_formatting=True, parallel_processing=False,
                    quality_threshold=0.95, timeout_seconds=180
                ),
                ProcessingStrategy.AUTO: StrategyParams(
                    chunk_size=800, overlap_size=160, use_ocr=True,
                    use_vision=True, extract_images=True, extract_tables=True,
                    preserve_formatting=True, parallel_processing=True,
                    quality_threshold=0.9, timeout_seconds=120
                )
            },
            fallback_strategies=[ProcessingStrategy.FAST, ProcessingStrategy.ACCURATE],
            quality_thresholds={
                'text_extraction': 0.9,
                'table_extraction': 0.85,
                'image_extraction': 0.8,
                'format_preservation': 0.85
            },
            custom_settings={
                'password_protection_handling': 'skip',
                'embedded_content_extraction': True,
                'track_changes_handling': 'accept',
                'comment_extraction': True
            }
        )

        # Scanned Documents Configuration
        self.source_configs[FileSource.SCANNED_DOCUMENTS] = SourceStrategyConfig(
            source=FileSource.SCANNED_DOCUMENTS,
            default_strategy=ProcessingStrategy.ACCURATE,
            strategy_params={
                ProcessingStrategy.FAST: StrategyParams(
                    chunk_size=800, overlap_size=200, use_ocr=True,
                    use_vision=True, extract_images=True, extract_tables=False,
                    preserve_formatting=False, parallel_processing=True,
                    quality_threshold=0.7, timeout_seconds=120
                ),
                ProcessingStrategy.BALANCED: StrategyParams(
                    chunk_size=600, overlap_size=150, use_ocr=True,
                    use_vision=True, extract_images=True, extract_tables=True,
                    preserve_formatting=True, parallel_processing=True,
                    quality_threshold=0.85, timeout_seconds=180
                ),
                ProcessingStrategy.ACCURATE: StrategyParams(
                    chunk_size=400, overlap_size=100, use_ocr=True,
                    use_vision=True, extract_images=True, extract_tables=True,
                    preserve_formatting=True, parallel_processing=False,
                    quality_threshold=0.95, timeout_seconds=300
                ),
                ProcessingStrategy.AUTO: StrategyParams(
                    chunk_size=600, overlap_size=150, use_ocr=True,
                    use_vision=True, extract_images=True, extract_tables=True,
                    preserve_formatting=True, parallel_processing=True,
                    quality_threshold=0.9, timeout_seconds=240
                )
            },
            fallback_strategies=[ProcessingStrategy.BALANCED, ProcessingStrategy.FAST],
            quality_thresholds={
                'ocr_accuracy': 0.85,
                'layout_detection': 0.8,
                'text_block_detection': 0.9
            },
            custom_settings={
                'ocr_languages': ['en', 'zh'],
                'image_preprocessing': True,
                'deskew_images': True,
                'noise_reduction': True,
                'multi_page_handling': True
            }
        )

        # Structured Data Configuration
        self.source_configs[FileSource.STRUCTURED_DATA] = SourceStrategyConfig(
            source=FileSource.STRUCTURED_DATA,
            default_strategy=ProcessingStrategy.FAST,
            strategy_params={
                ProcessingStrategy.FAST: StrategyParams(
                    chunk_size=2000, overlap_size=50, use_ocr=False,
                    use_vision=False, extract_images=False, extract_tables=False,
                    preserve_formatting=False, parallel_processing=True,
                    quality_threshold=0.8, timeout_seconds=15
                ),
                ProcessingStrategy.BALANCED: StrategyParams(
                    chunk_size=1500, overlap_size=100, use_ocr=False,
                    use_vision=False, extract_images=False, extract_tables=True,
                    preserve_formatting=True, parallel_processing=True,
                    quality_threshold=0.9, timeout_seconds=30
                ),
                ProcessingStrategy.ACCURATE: StrategyParams(
                    chunk_size=1000, overlap_size=100, use_ocr=False,
                    use_vision=False, extract_images=False, extract_tables=True,
                    preserve_formatting=True, parallel_processing=False,
                    quality_threshold=0.98, timeout_seconds=60
                ),
                ProcessingStrategy.AUTO: StrategyParams(
                    chunk_size=1500, overlap_size=100, use_ocr=False,
                    use_vision=False, extract_images=False, extract_tables=True,
                    preserve_formatting=True, parallel_processing=True,
                    quality_threshold=0.95, timeout_seconds=45
                )
            },
            fallback_strategies=[ProcessingStrategy.BALANCED],
            quality_thresholds={
                'structure_preservation': 0.95,
                'data_integrity': 0.98,
                'schema_validation': 0.9
            },
            custom_settings={
                'validate_schema': True,
                'preserve_relationships': True,
                'handle_large_files': True,
                'memory_efficient_parsing': True
            }
        )

        # Code Repositories Configuration
        self.source_configs[FileSource.CODE_REPOSITORIES] = SourceStrategyConfig(
            source=FileSource.CODE_REPOSITORIES,
            default_strategy=ProcessingStrategy.FAST,
            strategy_params={
                ProcessingStrategy.FAST: StrategyParams(
                    chunk_size=1500, overlap_size=50, use_ocr=False,
                    use_vision=False, extract_images=False, extract_tables=False,
                    preserve_formatting=True, parallel_processing=True,
                    quality_threshold=0.8, timeout_seconds=10
                ),
                ProcessingStrategy.BALANCED: StrategyParams(
                    chunk_size=1200, overlap_size=100, use_ocr=False,
                    use_vision=False, extract_images=False, extract_tables=False,
                    preserve_formatting=True, parallel_processing=True,
                    quality_threshold=0.9, timeout_seconds=20
                ),
                ProcessingStrategy.ACCURATE: StrategyParams(
                    chunk_size=1000, overlap_size=100, use_ocr=False,
                    use_vision=False, extract_images=False, extract_tables=False,
                    preserve_formatting=True, parallel_processing=False,
                    quality_threshold=0.95, timeout_seconds=40
                ),
                ProcessingStrategy.AUTO: StrategyParams(
                    chunk_size=1200, overlap_size=100, use_ocr=False,
                    use_vision=False, extract_images=False, extract_tables=False,
                    preserve_formatting=True, parallel_processing=True,
                    quality_threshold=0.9, timeout_seconds=30
                )
            },
            fallback_strategies=[ProcessingStrategy.BALANCED],
            quality_thresholds={
                'syntax_highlighting': 0.9,
                'code_structure': 0.85,
                'dependency_extraction': 0.8
            },
            custom_settings={
                'syntax_highlighting': True,
                'extract_comments': True,
                'detect_languages': True,
                'preserve_indentation': True,
                'git_metadata_extraction': True
            }
        )

        # Default configurations for unknown sources
        self.source_configs[FileSource.UNKNOWN] = SourceStrategyConfig(
            source=FileSource.UNKNOWN,
            default_strategy=ProcessingStrategy.AUTO,
            strategy_params={
                strategy: StrategyParams(
                    chunk_size=800, overlap_size=160, use_ocr=True,
                    use_vision=True, extract_images=True, extract_tables=True,
                    preserve_formatting=True, parallel_processing=True,
                    quality_threshold=0.85, timeout_seconds=120
                )
                for strategy in ProcessingStrategy
            },
            fallback_strategies=[ProcessingStrategy.BALANCED, ProcessingStrategy.FAST],
            quality_thresholds={
                'content_extraction': 0.8,
                'structure_detection': 0.7
            },
            custom_settings={}
        )

    def _load_config_file(self):
        """Load configuration from file."""
        if not self.config_file or not self.config_file.exists():
            return

        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                if self.config_file.suffix.lower() == '.json':
                    config_data = json.load(f)
                elif self.config_file.suffix.lower() in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported config format: {self.config_file.suffix}")

            # Load global configuration
            if 'global' in config_data:
                global_data = config_data['global']
                self.global_config = GlobalStrategyConfig(**global_data)

            # Load source-specific configurations
            if 'sources' in config_data:
                for source_name, source_data in config_data['sources'].items():
                    try:
                        source = FileSource(source_name)
                        self._update_source_config(source, source_data)
                    except ValueError:
                        # Skip unknown source types
                        continue

        except Exception as e:
            print(f"Warning: Failed to load config file {self.config_file}: {e}")

    def _update_source_config(self, source: FileSource, config_data: Dict[str, Any]):
        """Update source configuration with data from file."""

        if source not in self.source_configs:
            return

        source_config = self.source_configs[source]

        # Update default strategy
        if 'default_strategy' in config_data:
            try:
                source_config.default_strategy = ProcessingStrategy(config_data['default_strategy'])
            except ValueError:
                pass

        # Update strategy parameters
        if 'strategy_params' in config_data:
            for strategy_name, params_data in config_data['strategy_params'].items():
                try:
                    strategy = ProcessingStrategy(strategy_name)
                    if strategy in source_config.strategy_params:
                        # Update existing parameters
                        current_params = source_config.strategy_params[strategy]
                        for key, value in params_data.items():
                            if hasattr(current_params, key):
                                setattr(current_params, key, value)
                except ValueError:
                    continue

        # Update fallback strategies
        if 'fallback_strategies' in config_data:
            fallback_strategies = []
            for strategy_name in config_data['fallback_strategies']:
                try:
                    fallback_strategies.append(ProcessingStrategy(strategy_name))
                except ValueError:
                    continue
            if fallback_strategies:
                source_config.fallback_strategies = fallback_strategies

        # Update quality thresholds
        if 'quality_thresholds' in config_data:
            source_config.quality_thresholds.update(config_data['quality_thresholds'])

        # Update custom settings
        if 'custom_settings' in config_data:
            source_config.custom_settings.update(config_data['custom_settings'])

    def get_source_config(self, source: FileSource) -> SourceStrategyConfig:
        """Get configuration for a specific source type."""
        return self.source_configs.get(source, self.source_configs[FileSource.UNKNOWN])

    def get_strategy_params(
        self,
        source: FileSource,
        strategy: ProcessingStrategy
    ) -> StrategyParams:
        """Get strategy parameters for a specific source and strategy."""
        source_config = self.get_source_config(source)
        return source_config.strategy_params.get(strategy, StrategyParams(
            chunk_size=800, overlap_size=160, use_ocr=True,
            use_vision=True, extract_images=True, extract_tables=True,
            preserve_formatting=True, parallel_processing=True,
            quality_threshold=0.85, timeout_seconds=120
        ))

    def update_strategy_params(
        self,
        source: FileSource,
        strategy: ProcessingStrategy,
        updates: Dict[str, Any]
    ):
        """Update strategy parameters for a specific source and strategy."""
        source_config = self.get_source_config(source)
        current_params = source_config.strategy_params.get(strategy)

        if current_params:
            for key, value in updates.items():
                if hasattr(current_params, key):
                    setattr(current_params, key, value)

    def set_default_strategy(self, source: FileSource, strategy: ProcessingStrategy):
        """Set the default strategy for a source type."""
        if source in self.source_configs:
            self.source_configs[source].default_strategy = strategy

    def save_config(self, file_path: Optional[Union[str, Path]] = None):
        """Save current configuration to file."""
        save_path = Path(file_path) if file_path else self.config_file
        if not save_path:
            raise ValueError("No config file specified")

        config_data = {
            'global': asdict(self.global_config),
            'sources': {}
        }

        # Convert source configurations
        for source, source_config in self.source_configs.items():
            config_data['sources'][source.value] = {
                'default_strategy': source_config.default_strategy.value,
                'strategy_params': {
                    strategy.value: asdict(params)
                    for strategy, params in source_config.strategy_params.items()
                },
                'fallback_strategies': [s.value for s in source_config.fallback_strategies],
                'quality_thresholds': source_config.quality_thresholds,
                'custom_settings': source_config.custom_settings
            }

        # Save to file
        with open(save_path, 'w', encoding='utf-8') as f:
            if save_path.suffix.lower() == '.json':
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            elif save_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported config format: {save_path.suffix}")

    def get_effective_config(
        self,
        source: FileSource,
        strategy: Optional[ProcessingStrategy] = None
    ) -> Dict[str, Any]:
        """Get effective configuration combining global and source-specific settings."""
        source_config = self.get_source_config(source)
        effective_strategy = strategy or source_config.default_strategy

        return {
            'strategy': effective_strategy,
            'params': asdict(source_config.strategy_params.get(effective_strategy)),
            'global_settings': asdict(self.global_config),
            'source_settings': {
                'fallback_strategies': [s.value for s in source_config.fallback_strategies],
                'quality_thresholds': source_config.quality_thresholds,
                'custom_settings': source_config.custom_settings
            }
        }