## Context
The RAG system has evolved over time with multiple developers contributing different implementations of the same core functionality. This has resulted in:
- Multiple web framework implementations (FastAPI + Sanic)
- Three separate embedding services with overlapping functionality
- Duplicate vector store and retriever interfaces
- Multiple model provider implementations with similar patterns
- Extensive Docker configuration duplication

## Goals / Non-Goals
- **Goals**:
  - Reduce codebase size by 30-40%
  - Eliminate redundant functionality
  - Improve maintainability and consistency
  - Enhance performance through service consolidation
  - Simplify deployment and configuration
- **Non-Goals**:
  - Complete rewrite of existing functionality
  - Changing core RAG algorithms
  - Modifying external API interfaces visible to end users

## Decisions
- **Decision**: Consolidate to FastAPI-only web framework
  - **Rationale**: Better OpenAPI documentation, larger ecosystem, async performance advantages
  - **Alternatives considered**: Keep Sanic for microservices, migrate to Flask (rejected due to async limitations)

- **Decision**: Create unified embedding service with pluggable model backends
  - **Rationale**: Eliminate duplicate model loading, provide consistent interface, support multiple models
  - **Alternatives considered**: Keep separate services (rejected due to redundancy), use external service (rejected for control)

- **Decision**: Implement factory pattern for model providers
  - **Rationale**: Standardize configuration, enable runtime model switching, reduce code duplication
  - **Alternatives considered**: Abstract base classes (rejected for complexity), composition patterns (accepted for specific cases)

## Risks / Trade-offs
- **Risk**: Breaking existing integrations that depend on specific service endpoints
  - **Mitigation**: Maintain backward compatibility endpoints during transition, provide migration guide
- **Risk**: Service consolidation may create single points of failure
  - **Mitigation**: Implement proper error handling, circuit breakers, and graceful degradation
- **Trade-off**: Reduced flexibility vs. simplified maintenance
  - **Acceptance**: Prioritize maintainability for enterprise use case

## Migration Plan
1. **Phase 1**: Create unified services alongside existing ones (parallel implementation)
2. **Phase 2**: Update internal consumers to use unified services
3. **Phase 3**: Migrate external API endpoints with backward compatibility
4. **Phase 4**: Remove legacy implementations and deprecated endpoints
5. **Phase 5**: Update documentation and deployment configurations

## Open Questions
- How to handle existing deployments with specific service configurations?
- Should we maintain backward compatibility for all legacy endpoints or deprecate some?
- What is the timeline for external users to migrate to new unified endpoints?