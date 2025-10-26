## ADDED Requirements
### Requirement: Unified Embedding Service
The system SHALL provide a single unified embedding service that supports multiple model backends.

#### Scenario: Multi-model embedding
- **WHEN** a text embedding request is received
- **THEN** the service routes to the appropriate model backend (BCE, Qwen3, OpenAI, etc.)
- **AND** maintains consistent response format across all models

#### Scenario: Model backend switching
- **WHEN** configuration specifies a different embedding model
- **THEN** the service loads the new model without requiring restart
- **AND** continues serving requests with the new model

## REMOVED Requirements
### Requirement: BCE Embedding Service
**Reason**: Functionality will be integrated into unified embedding service
**Migration**: BCE model support will be available as a backend option in unified service

### Requirement: Qwen3 Embedding Service
**Reason**: Redundant with generic embedding service that already supports Qwen3 models
**Migration**: Qwen3-specific optimizations will be incorporated into unified service

### Requirement: Generic Embedding Service
**Reason**: Will be enhanced and renamed to become the unified embedding service
**Migration**: Existing functionality will be preserved and extended with additional model support

## MODIFIED Requirements
### Requirement: Embedding Model Management
The system SHALL provide centralized management of embedding models with caching and lifecycle management.

#### Scenario: Model loading optimization
- **WHEN** multiple requests use the same embedding model
- **THEN** the model is loaded once and reused across requests
- **AND** memory usage is optimized through proper model lifecycle management

#### Scenario: Model health monitoring
- **WHEN** embedding models are loaded
- **THEN** the system monitors model health and performance
- **AND** automatically restarts failed model instances