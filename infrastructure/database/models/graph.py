"""
Graph Models

This module contains knowledge graph models.
Based on RAGFlow GraphRAG architecture.
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from sqlalchemy import Column, String, Integer, Boolean, DateTime, Text, JSON, ForeignKey, Float, Index
from sqlalchemy.orm import relationship, validates
from sqlalchemy.ext.hybrid_property import hybrid_property

from .base import TenantBaseModel, Status, generate_uuid, current_utc_time


class GraphEntity(TenantBaseModel, TimestampMixin, MetadataMixin):
    """
    Graph Entity model for knowledge graph entities.

    Represents an entity in the knowledge graph:
    - Basic entity information
    - Type and classification
    - Properties and attributes
    - Relationships and connections
    """

    __tablename__ = 'graph_entities'

    # Basic entity information
    name = Column(String(500), nullable=False, index=True)
    canonical_name = Column(String(500), nullable=True, index=True)  # Standardized name
    entity_type = Column(String(100), nullable=False, index=True)
    sub_type = Column(String(100), nullable=True)  # More specific type
    description = Column(Text, nullable=True)
    aliases = Column(JSON, default=list, nullable=False)  # Alternative names

    # Properties and attributes
    properties = Column(JSON, default=dict, nullable=False)  # Entity properties
    attributes = Column(JSON, default=dict, nullable=False)  # Key-value attributes
    confidence = Column(Float, default=1.0, nullable=False)  # Entity confidence score
    source = Column(String(100), nullable=True)  # Source of entity extraction

    # Vector information
    embedding_id = Column(String(100), nullable=True)  # ID in vector database
    embedding_model = Column(String(100), nullable=True)
    embedding_dimension = Column(Integer, nullable=True)
    embedding_updated_at = Column(DateTime(timezone=True), nullable=True)

    # Graph information
    degree = Column(Integer, default=0, nullable=False)  # Number of connections
    pagerank_score = Column(Float, nullable=True)  # PageRank importance score
    centrality_score = Column(Float, nullable=True)  # Centrality importance score

    # Processing information
    extracted_at = Column(DateTime(timezone=True), nullable=True)
    validated_at = Column(DateTime(timezone=True), nullable=True)
    validated_by = Column(String(36), nullable=True)  # User ID who validated
    is_validated = Column(Boolean, default=False, nullable=False)

    # Relationships
    outgoing_relations = relationship('GraphRelation', foreign_keys='[GraphRelation.subject_id]',
                                   back_populates='subject_entity', lazy='dynamic',
                                   cascade='all, delete-orphan')
    incoming_relations = relationship('GraphRelation', foreign_keys='[GraphRelation.object_id]',
                                   back_populates='object_entity', lazy='dynamic',
                                   cascade='all, delete-orphan')
    mentions = relationship('EntityMention', back_populates='entity', lazy='dynamic',
                           cascade='all, delete-orphan')

    # Indexes
    __table_args__ = (
        Index('idx_entity_tenant_type', 'tenant_id', 'entity_type'),
        Index('idx_entity_name', 'name'),
        Index('idx_entity_canonical', 'canonical_name'),
        Index('idx_entity_confidence', 'confidence'),
        Index('idx_entity_degree', 'degree'),
        Index('idx_entity_embedding', 'embedding_id'),
    )

    @validates('confidence')
    def validate_confidence(self, key, confidence):
        """Validate confidence score."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return confidence

    @hybrid_property
    def has_embedding(self) -> bool:
        """Check if entity has embedding."""
        return self.embedding_id is not None and self.embedding_updated_at is not None

    @hybrid_property
    def is_valid(self) -> bool:
        """Check if entity is validated."""
        return self.is_validated and self.validated_at is not None

    @hybrid_property
    def connection_count(self) -> int:
        """Get total number of connections."""
        return self.degree

    @hybrid_property
    def alias_count(self) -> int:
        """Get number of aliases."""
        return len(self.aliases) if self.aliases else 0

    def add_alias(self, alias: str) -> None:
        """Add an alias to the entity."""
        if not self.aliases:
            self.aliases = []
        if alias not in self.aliases:
            self.aliases.append(alias)

    def remove_alias(self, alias: str) -> None:
        """Remove an alias from the entity."""
        if self.aliases and alias in self.aliases:
            self.aliases.remove(alias)

    def set_property(self, key: str, value: Any) -> None:
        """Set a property value."""
        if not self.properties:
            self.properties = {}
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value."""
        if not self.properties:
            return default
        return self.properties.get(key, default)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute value."""
        if not self.attributes:
            self.attributes = {}
        self.attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get an attribute value."""
        if not self.attributes:
            return default
        return self.attributes.get(key, default)

    def update_embedding(self, embedding_id: str, model: str, dimension: int) -> None:
        """Update embedding information."""
        self.embedding_id = embedding_id
        self.embedding_model = model
        self.embedding_dimension = dimension
        self.embedding_updated_at = current_utc_time()

    def validate_entity(self, user_id: str) -> None:
        """Validate the entity."""
        self.is_validated = True
        self.validated_at = current_utc_time()
        self.validated_by = user_id

    def update_degree(self) -> None:
        """Update entity degree based on relations."""
        outgoing_count = self.outgoing_relations.count()
        incoming_count = self.incoming_relations.count()
        self.degree = outgoing_count + incoming_count

    def get_related_entities(self, relation_type: Optional[str] = None,
                           direction: str = 'both') -> List['GraphEntity']:
        """Get related entities."""
        entities = []

        if direction in ['outgoing', 'both']:
            outgoing_query = self.outgoing_relations
            if relation_type:
                outgoing_query = outgoing_query.filter_by(relation_type=relation_type)
            entities.extend([rel.object_entity for rel in outgoing_query])

        if direction in ['incoming', 'both']:
            incoming_query = self.incoming_relations
            if relation_type:
                incoming_query = incoming_query.filter_by(relation_type=relation_type)
            entities.extend([rel.subject_entity for rel in incoming_query])

        return entities


class GraphRelation(TenantBaseModel, TimestampMixin):
    """
    Graph Relation model for knowledge graph relationships.

    Represents a relationship between two entities:
    - Subject and object entities
    - Relation type and properties
    - Confidence and source information
    - Temporal and contextual information
    """

    __tablename__ = 'graph_relations'

    # Foreign keys
    subject_id = Column(String(36), ForeignKey('graph_entities.id'), nullable=False)
    object_id = Column(String(36), ForeignKey('graph_entities.id'), nullable=False)

    # Relation information
    relation_type = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    inverse_relation = Column(String(100), nullable=True)  # Inverse relation type

    # Properties and attributes
    properties = Column(JSON, default=dict, nullable=False)
    attributes = Column(JSON, default=dict, nullable=False)
    confidence = Column(Float, default=1.0, nullable=False)
    weight = Column(Float, default=1.0, nullable=False)  # Relation weight/strength

    # Temporal information
    start_date = Column(DateTime(timezone=True), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    temporal_type = Column(String(50), nullable=True)  # instantaneous, interval, ongoing

    # Source and provenance
    source = Column(String(100), nullable=True)
    source_document_id = Column(String(36), nullable=True)
    source_text = Column(Text, nullable=True)
    extraction_method = Column(String(50), nullable=True)  # manual, auto, ml

    # Validation information
    validated_at = Column(DateTime(timezone=True), nullable=True)
    validated_by = Column(String(36), nullable=True)
    is_validated = Column(Boolean, default=False, nullable=False)

    # Relationships
    subject_entity = relationship('GraphEntity', foreign_keys=[subject_id], back_populates='outgoing_relations')
    object_entity = relationship('GraphEntity', foreign_keys=[object_id], back_populates='incoming_relations')

    # Indexes
    __table_args__ = (
        Index('idx_relation_subject', 'subject_id'),
        Index('idx_relation_object', 'object_id'),
        Index('idx_relation_type', 'relation_type'),
        Index('idx_relation_weight', 'weight'),
        Index('idx_relation_confidence', 'confidence'),
        Index('idx_relation_source', 'source'),
        Index('idx_relation_temporal', 'start_date', 'end_date'),
    )

    @validates('confidence')
    def validate_confidence(self, key, confidence):
        """Validate confidence score."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return confidence

    @validates('weight')
    def validate_weight(self, key, weight):
        """Validate weight value."""
        if weight < 0:
            raise ValueError('Weight must be non-negative')
        return weight

    @hybrid_property
    def is_valid(self) -> bool:
        """Check if relation is validated."""
        return self.is_validated and self.validated_at is not None

    @hybrid_property
    def is_active(self) -> bool:
        """Check if relation is currently active."""
        now = current_utc_time()
        if self.start_date and now < self.start_date:
            return False
        if self.end_date and now > self.end_date:
            return False
        return True

    @hybrid_property
    def is_temporal(self) -> bool:
        """Check if relation has temporal information."""
        return self.start_date is not None or self.end_date is not None

    @hybrid_property
    def duration_days(self) -> Optional[int]:
        """Get relation duration in days."""
        if not self.start_date or not self.end_date:
            return None
        delta = self.end_date - self.start_date
        return delta.days

    def set_property(self, key: str, value: Any) -> None:
        """Set a property value."""
        if not self.properties:
            self.properties = {}
        self.properties[key] = value

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value."""
        if not self.properties:
            return default
        return self.properties.get(key, default)

    def set_attribute(self, key: str, value: Any) -> None:
        """Set an attribute value."""
        if not self.attributes:
            self.attributes = {}
        self.attributes[key] = value

    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get an attribute value."""
        if not self.attributes:
            return default
        return self.attributes.get(key, default)

    def validate_relation(self, user_id: str) -> None:
        """Validate the relation."""
        self.is_validated = True
        self.validated_at = current_utc_time()
        self.validated_by = user_id

    def set_temporal_bounds(self, start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None) -> None:
        """Set temporal bounds for the relation."""
        self.start_date = start_date
        self.end_date = end_date
        if start_date or end_date:
            self.temporal_type = 'interval'
        else:
            self.temporal_type = None

    def get_relation_summary(self) -> Dict[str, Any]:
        """Get relation summary."""
        return {
            'subject': self.subject_entity.name if self.subject_entity else None,
            'object': self.object_entity.name if self.object_entity else None,
            'relation_type': self.relation_type,
            'confidence': self.confidence,
            'weight': self.weight,
            'is_active': self.is_active,
            'is_temporal': self.is_temporal,
            'source': self.source,
            'is_validated': self.is_validated,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class EntityMention(TenantBaseModel, TimestampMixin):
    """
    Entity Mention model for tracking entity mentions in documents.

    Represents a mention of an entity in a document:
    - Mention location and context
    - Confidence and ambiguity
    - Linking to entities
    - Extraction metadata
    """

    __tablename__ = 'entity_mentions'

    # Foreign keys
    entity_id = Column(String(36), ForeignKey('graph_entities.id'), nullable=False)
    document_id = Column(String(36), ForeignKey('documents.id'), nullable=True)
    chunk_id = Column(String(36), ForeignKey('document_chunks.id'), nullable=True)
    message_id = Column(String(36), ForeignKey('chat_messages.id'), nullable=True)

    # Mention information
    mention_text = Column(String(1000), nullable=False)
    canonical_mention = Column(String(1000), nullable=True)
    start_position = Column(Integer, nullable=False)
    end_position = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    line_number = Column(Integer, nullable=True)

    # Context information
    left_context = Column(Text, nullable=True)
    right_context = Column(Text, nullable=True)
    sentence_context = Column(Text, nullable=True)
    paragraph_context = Column(Text, nullable=True)

    # Linking and disambiguation
    linking_confidence = Column(Float, default=1.0, nullable=False)
    disambiguation_score = Column(Float, nullable=True)
    alternative_entities = Column(JSON, default=list, nullable=False)  # Alternative entity links

    # Extraction information
    extraction_method = Column(String(50), nullable=True)
    extraction_model = Column(String(100), nullable=True)
    extraction_confidence = Column(Float, nullable=True)
    extractor_version = Column(String(50), nullable=True)

    # Validation information
    validated_at = Column(DateTime(timezone=True), nullable=True)
    validated_by = Column(String(36), nullable=True)
    is_validated = Column(Boolean, default=False, nullable=False)
    is_correct = Column(Boolean, nullable=True)  # Whether the linking is correct

    # Relationships
    entity = relationship('GraphEntity', back_populates='mentions')

    # Indexes
    __table_args__ = (
        Index('idx_mention_entity', 'entity_id'),
        Index('idx_mention_document', 'document_id'),
        Index('idx_mention_chunk', 'chunk_id'),
        Index('idx_mention_message', 'message_id'),
        Index('idx_mention_position', 'start_position', 'end_position'),
        Index('idx_mention_confidence', 'linking_confidence'),
        Index('idx_mention_text', 'mention_text'),
    )

    @validates('linking_confidence')
    def validate_linking_confidence(self, key, confidence):
        """Validate linking confidence."""
        if not 0.0 <= confidence <= 1.0:
            raise ValueError('Linking confidence must be between 0.0 and 1.0')
        return confidence

    @hybrid_property
    def mention_length(self) -> int:
        """Get mention length."""
        return self.end_position - self.start_position

    @hybrid_property
    def is_valid_link(self) -> bool:
        """Check if mention link is validated and correct."""
        return self.is_validated and self.is_correct

    @hybrid_property
    def has_alternatives(self) -> bool:
        """Check if mention has alternative entity links."""
        return len(self.alternative_entities) > 0 if self.alternative_entities else False

    def add_alternative_entity(self, entity_id: str, confidence: float, reason: Optional[str] = None) -> None:
        """Add an alternative entity link."""
        if not self.alternative_entities:
            self.alternative_entities = []
        alternative = {
            'entity_id': entity_id,
            'confidence': confidence,
            'reason': reason
        }
        self.alternative_entities.append(alternative)

    def validate_mention(self, user_id: str, is_correct: bool) -> None:
        """Validate the mention link."""
        self.is_validated = True
        self.validated_at = current_utc_time()
        self.validated_by = user_id
        self.is_correct = is_correct

    def get_context_summary(self, context_size: int = 100) -> str:
        """Get a summary of the mention context."""
        context_parts = []

        if self.left_context:
            context_parts.append(self.left_context[-context_size:])

        context_parts.append(f"[{self.mention_text}]")

        if self.right_context:
            context_parts.append(self.right_context[:context_size])

        return " ".join(context_parts)

    def get_position_summary(self) -> str:
        """Get a summary of mention position."""
        parts = [f"Position {self.start_position}-{self.end_position}"]

        if self.page_number is not None:
            parts.append(f"Page {self.page_number}")

        if self.line_number is not None:
            parts.append(f"Line {self.line_number}")

        return ", ".join(parts)