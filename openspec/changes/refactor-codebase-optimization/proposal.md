## Why
The current RAG system has significant code redundancy with multiple implementations of the same functionality across different modules, leading to increased maintenance overhead, potential inconsistencies, and reduced performance due to duplicate service instances.

## What Changes
- **BREAKING**: Consolidate web framework from dual FastAPI/Sanic implementation to unified FastAPI-only approach
- **BREAKING**: Merge three separate embedding services (BCE, Qwen3, Generic) into single unified embedding service
- **BREAKING**: Standardize vector store implementations and remove duplicate interfaces
- **BREAKING**: Unify model provider implementations under single factory pattern
- Consolidate multiple Docker Compose configurations into modular structure
- Remove redundant interface definitions and consolidate into inheritance hierarchy
- Clean up unused imports and dead code throughout the codebase

## Impact
- **Effected specs**: web-framework, embedding-service, vector-storage, model-providers, configuration
- **Effected code**: Main service files in rag_service/, Docker configurations, interface definitions
- **Performance improvements**: Reduced memory usage, fewer duplicate model loads, faster startup times
- **Maintainability**: Single source of truth for each functionality, easier debugging and enhancement