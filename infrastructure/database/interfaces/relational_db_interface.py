"""
Relational Database Interface Abstract Class

This module defines the abstract interface for relational database providers.
All relational database implementations must inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, AsyncGenerator, Type
from pydantic import BaseModel, Field
from enum import Enum
import uuid
import time


class IsolationLevel(str, Enum):
    """Transaction isolation levels."""
    READ_UNCOMMITTED = "READ_UNCOMMITTED"
    READ_COMMITTED = "READ_COMMITTED"
    REPEATABLE_READ = "REPEATABLE_READ"
    SERIALIZABLE = "SERIALIZABLE"


class QueryType(str, Enum):
    """Query types."""
    SELECT = "SELECT"
    INSERT = "INSERT"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    DROP = "DROP"
    ALTER = "ALTER"
    TRUNCATE = "TRUNCATE"


class DatabaseConnection(BaseModel):
    """Database connection information."""
    dsn: str
    host: str
    port: int
    database: str
    username: Optional[str] = None
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=20, ge=0)
    pool_timeout: int = Field(default=30, ge=1)
    pool_recycle: int = Field(default=3600, ge=1)
    pool_pre_ping: bool = True
    echo: bool = False
    isolation_level: Optional[IsolationLevel] = None


class QueryResult(BaseModel):
    """Database query result."""
    rows: List[Dict[str, Any]]
    row_count: int
    columns: List[str]
    query_time_ms: Optional[float] = None
    last_insert_id: Optional[Union[int, str]] = None
    affected_rows: Optional[int] = None


class TransactionConfig(BaseModel):
    """Transaction configuration."""
    isolation_level: Optional[IsolationLevel] = None
    readonly: bool = False
    deferrable: bool = False
    timeout: Optional[int] = None  # seconds


class TableInfo(BaseModel):
    """Table information."""
    name: str
    schema: Optional[str] = None
    columns: List[Dict[str, Any]]
    indexes: List[Dict[str, Any]]
    row_count: Optional[int] = None
    size_mb: Optional[float] = None
    engine: Optional[str] = None
    charset: Optional[str] = None
    collation: Optional[str] = None
    comment: Optional[str] = None


class IndexInfo(BaseModel):
    """Index information."""
    name: str
    table_name: str
    columns: List[str]
    unique: bool = False
    primary: bool = False
    type: str = "BTREE"
    cardinality: Optional[int] = None
    nullable: bool = True
    comment: Optional[str] = None


class RelationalDBCapabilities(BaseModel):
    """Relational database capabilities."""
    supported_dialects: List[str]
    supports_transactions: bool
    supports_savepoints: bool
    supports_nested_transactions: bool
    supports_ddl_transactions: bool
    supports_window_functions: bool
    supports_cte: bool  # Common Table Expressions
    supports_json: bool
    supports_array: bool
    supports_full_text_search: bool
    supports_spatial: bool
    supports_partitions: bool
    supports_foreign_keys: bool
    supports_check_constraints: bool
    supports_triggers: bool
    supports_stored_procedures: bool
    supports_views: bool
    supports_materialized_views: bool
    max_connections: Optional[int] = None
    max_identifier_length: int
    max_row_size_mb: Optional[int] = None


class RelationalDBConfig(BaseModel):
    """Relational database configuration."""
    provider: str
    dialect: str
    host: str
    port: int
    database: str
    username: str
    password: str
    driver: Optional[str] = None
    charset: str = "utf8mb4"
    timezone: Optional[str] = None

    # Connection pooling
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=20, ge=0)
    pool_timeout: int = Field(default=30, ge=1)
    pool_recycle: int = Field(default=3600, ge=1)
    pool_pre_ping: bool = True

    # Query settings
    echo: bool = False
    isolation_level: Optional[IsolationLevel] = IsolationLevel.READ_COMMITTED

    # SSL settings
    ssl_enabled: bool = False
    ssl_ca: Optional[str] = None
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None

    # Timeout settings
    connect_timeout: int = Field(default=10, ge=1)
    query_timeout: int = Field(default=30, ge=1)

    # Additional connection parameters
    connection_params: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class RelationalDBInterface(ABC):
    """
    Abstract interface for relational database providers.

    This class defines the contract that all relational database implementations must follow.
    It provides a unified interface for different relational databases while allowing
    provider-specific configurations and capabilities.
    """

    def __init__(self, config: RelationalDBConfig):
        """Initialize the relational database with configuration."""
        self.config = config
        self.provider_name = config.provider
        self.dialect = config.dialect
        self.host = config.host
        self.port = config.port
        self.database = config.database
        self.username = config.username
        self._capabilities: Optional[RelationalDBCapabilities] = None
        self._connected = False

    @property
    @abstractmethod
    def capabilities(self) -> RelationalDBCapabilities:
        """
        Get the capabilities of this relational database provider.

        Returns:
            RelationalDBCapabilities: Information about supported features
        """
        pass

    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the relational database.

        Returns:
            bool: True if connection successful

        Raises:
            RelationalDBException: If connection fails
        """
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """
        Disconnect from the relational database.

        Returns:
            bool: True if disconnection successful
        """
        pass

    @abstractmethod
    async def execute_query(
        self,
        query: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
        fetch: bool = True,
        **kwargs
    ) -> QueryResult:
        """
        Execute a SQL query.

        Args:
            query: SQL query string
            params: Query parameters
            fetch: Whether to fetch results
            **kwargs: Additional provider-specific parameters

        Returns:
            QueryResult: Query result

        Raises:
            RelationalDBException: If query execution fails
        """
        pass

    @abstractmethod
    async def execute_many(
        self,
        query: str,
        params_list: List[Union[Dict[str, Any], List[Any]]],
        **kwargs
    ) -> QueryResult:
        """
        Execute a SQL query multiple times with different parameters.

        Args:
            query: SQL query string
            params_list: List of parameter sets
            **kwargs: Additional provider-specific parameters

        Returns:
            QueryResult: Query result

        Raises:
            RelationalDBException: If query execution fails
        """
        pass

    @abstractmethod
    async def begin_transaction(
        self,
        config: Optional[TransactionConfig] = None,
        **kwargs
    ) -> str:
        """
        Begin a transaction.

        Args:
            config: Transaction configuration
            **kwargs: Additional provider-specific parameters

        Returns:
            str: Transaction ID

        Raises:
            RelationalDBException: If transaction begin fails
        """
        pass

    @abstractmethod
    async def commit_transaction(
        self,
        transaction_id: str,
        **kwargs
    ) -> bool:
        """
        Commit a transaction.

        Args:
            transaction_id: Transaction ID
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if commit successful

        Raises:
            RelationalDBException: If commit fails
        """
        pass

    @abstractmethod
    async def rollback_transaction(
        self,
        transaction_id: str,
        **kwargs
    ) -> bool:
        """
        Rollback a transaction.

        Args:
            transaction_id: Transaction ID
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if rollback successful

        Raises:
            RelationalDBException: If rollback fails
        """
        pass

    @abstractmethod
    async def create_savepoint(
        self,
        transaction_id: str,
        savepoint_name: str,
        **kwargs
    ) -> bool:
        """
        Create a savepoint within a transaction.

        Args:
            transaction_id: Transaction ID
            savepoint_name: Savepoint name
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if savepoint created successfully

        Raises:
            RelationalDBException: If savepoint creation fails
        """
        pass

    @abstractmethod
    async def rollback_to_savepoint(
        self,
        transaction_id: str,
        savepoint_name: str,
        **kwargs
    ) -> bool:
        """
        Rollback to a savepoint.

        Args:
            transaction_id: Transaction ID
            savepoint_name: Savepoint name
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if rollback successful

        Raises:
            RelationalDBException: If rollback fails
        """
        pass

    @abstractmethod
    async def list_tables(
        self,
        schema: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        List all tables in the database.

        Args:
            schema: Schema name (if applicable)
            **kwargs: Additional provider-specific parameters

        Returns:
            List[str]: List of table names

        Raises:
            RelationalDBException: If listing fails
        """
        pass

    @abstractmethod
    async def table_exists(
        self,
        table_name: str,
        schema: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Check if a table exists.

        Args:
            table_name: Name of table to check
            schema: Schema name (if applicable)
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if table exists

        Raises:
            RelationalDBException: If check fails
        """
        pass

    @abstractmethod
    async def get_table_info(
        self,
        table_name: str,
        schema: Optional[str] = None,
        **kwargs
    ) -> TableInfo:
        """
        Get information about a table.

        Args:
            table_name: Name of table
            schema: Schema name (if applicable)
            **kwargs: Additional provider-specific parameters

        Returns:
            TableInfo: Table information

        Raises:
            RelationalDBException: If getting info fails
        """
        pass

    @abstractmethod
    async def list_indexes(
        self,
        table_name: str,
        schema: Optional[str] = None,
        **kwargs
    ) -> List[IndexInfo]:
        """
        List all indexes for a table.

        Args:
            table_name: Name of table
            schema: Schema name (if applicable)
            **kwargs: Additional provider-specific parameters

        Returns:
            List[IndexInfo]: List of index information

        Raises:
            RelationalDBException: If listing fails
        """
        pass

    @abstractmethod
    async def create_table(
        self,
        table_name: str,
        columns: Dict[str, Dict[str, Any]],
        schema: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Create a new table.

        Args:
            table_name: Name of table to create
            columns: Column definitions
            schema: Schema name (if applicable)
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if table created successfully

        Raises:
            RelationalDBException: If table creation fails
        """
        pass

    @abstractmethod
    async def drop_table(
        self,
        table_name: str,
        schema: Optional[str] = None,
        if_exists: bool = True,
        **kwargs
    ) -> bool:
        """
        Drop a table.

        Args:
            table_name: Name of table to drop
            schema: Schema name (if applicable)
            if_exists: Whether to use IF EXISTS
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if table dropped successfully

        Raises:
            RelationalDBException: If table drop fails
        """
        pass

    @abstractmethod
    async def truncate_table(
        self,
        table_name: str,
        schema: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Truncate a table.

        Args:
            table_name: Name of table to truncate
            schema: Schema name (if applicable)
            **kwargs: Additional provider-specific parameters

        Returns:
            bool: True if table truncated successfully

        Raises:
            RelationalDBException: If truncate fails
        """
        pass

    async def select(
        self,
        table_name: str,
        columns: Union[str, List[str]] = "*",
        where: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        group_by: Optional[str] = None,
        having: Optional[str] = None,
        schema: Optional[str] = None,
        **kwargs
    ) -> QueryResult:
        """
        Execute a SELECT query.

        Args:
            table_name: Name of table
            columns: Columns to select
            where: WHERE clause
            params: Query parameters
            order_by: ORDER BY clause
            limit: LIMIT value
            offset: OFFSET value
            group_by: GROUP BY clause
            having: HAVING clause
            schema: Schema name
            **kwargs: Additional parameters

        Returns:
            QueryResult: Query result
        """
        # Build SELECT query
        if isinstance(columns, list):
            columns_str = ", ".join(columns)
        else:
            columns_str = columns

        from_clause = f"{schema}.{table_name}" if schema else table_name
        query = f"SELECT {columns_str} FROM {from_clause}"

        if where:
            query += f" WHERE {where}"

        if group_by:
            query += f" GROUP BY {group_by}"

        if having:
            query += f" HAVING {having}"

        if order_by:
            query += f" ORDER BY {order_by}"

        if limit is not None:
            query += f" LIMIT {limit}"

        if offset is not None:
            query += f" OFFSET {offset}"

        return await self.execute_query(query, params, **kwargs)

    async def insert(
        self,
        table_name: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        schema: Optional[str] = None,
        on_conflict: Optional[str] = None,
        returning: Optional[str] = None,
        **kwargs
    ) -> QueryResult:
        """
        Execute an INSERT query.

        Args:
            table_name: Name of table
            data: Data to insert (single dict or list of dicts)
            schema: Schema name
            on_conflict: ON CONFLICT clause (PostgreSQL) or similar
            returning: RETURNING clause (PostgreSQL) or similar
            **kwargs: Additional parameters

        Returns:
            QueryResult: Query result
        """
        if isinstance(data, dict):
            data = [data]

        if not data:
            return QueryResult(rows=[], row_count=0, columns=[])

        # Build INSERT query
        columns = list(data[0].keys())
        columns_str = ", ".join(columns)
        placeholders = ", ".join([f":{col}" for col in columns])

        from_clause = f"{schema}.{table_name}" if schema else table_name
        query = f"INSERT INTO {from_clause} ({columns_str}) VALUES ({placeholders})"

        if on_conflict:
            query += f" {on_conflict}"

        if returning:
            query += f" RETURNING {returning}"

        # Flatten data for execute_many
        params_list = []
        for row in data:
            params = {}
            for col in columns:
                params[col] = row.get(col)
            params_list.append(params)

        return await self.execute_many(query, params_list, **kwargs)

    async def update(
        self,
        table_name: str,
        data: Dict[str, Any],
        where: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        schema: Optional[str] = None,
        returning: Optional[str] = None,
        **kwargs
    ) -> QueryResult:
        """
        Execute an UPDATE query.

        Args:
            table_name: Name of table
            data: Data to update
            where: WHERE clause
            params: Query parameters
            schema: Schema name
            returning: RETURNING clause (PostgreSQL) or similar
            **kwargs: Additional parameters

        Returns:
            QueryResult: Query result
        """
        if not data:
            return QueryResult(rows=[], row_count=0, columns=[])

        # Build SET clause
        set_clauses = []
        update_params = params or {}

        for column, value in data.items():
            placeholder = f"update_{column}"
            set_clauses.append(f"{column} = :{placeholder}")
            update_params[placeholder] = value

        set_str = ", ".join(set_clauses)

        from_clause = f"{schema}.{table_name}" if schema else table_name
        query = f"UPDATE {from_clause} SET {set_str}"

        if where:
            query += f" WHERE {where}"

        if returning:
            query += f" RETURNING {returning}"

        return await self.execute_query(query, update_params, **kwargs)

    async def delete(
        self,
        table_name: str,
        where: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        schema: Optional[str] = None,
        returning: Optional[str] = None,
        **kwargs
    ) -> QueryResult:
        """
        Execute a DELETE query.

        Args:
            table_name: Name of table
            where: WHERE clause
            params: Query parameters
            schema: Schema name
            returning: RETURNING clause (PostgreSQL) or similar
            **kwargs: Additional parameters

        Returns:
            QueryResult: Query result
        """
        from_clause = f"{schema}.{table_name}" if schema else table_name
        query = f"DELETE FROM {from_clause}"

        if where:
            query += f" WHERE {where}"

        if returning:
            query += f" RETURNING {returning}"

        return await self.execute_query(query, params, **kwargs)

    async def count(
        self,
        table_name: str,
        where: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        schema: Optional[str] = None,
        **kwargs
    ) -> int:
        """
        Count rows in a table.

        Args:
            table_name: Name of table
            where: WHERE clause
            params: Query parameters
            schema: Schema name
            **kwargs: Additional parameters

        Returns:
            int: Row count
        """
        result = await self.select(
            table_name=table_name,
            columns="COUNT(*) as count",
            where=where,
            params=params,
            schema=schema,
            **kwargs
        )

        if result.rows and len(result.rows) > 0:
            return result.rows[0].get("count", 0)
        return 0

    def supports_feature(self, feature: str) -> bool:
        """
        Check if the provider supports a specific feature.

        Args:
            feature: Feature name (transactions, savepoints, json, etc.)

        Returns:
            bool: True if feature is supported
        """
        capability_map = {
            "transactions": "supports_transactions",
            "savepoints": "supports_savepoints",
            "nested_transactions": "supports_nested_transactions",
            "ddl_transactions": "supports_ddl_transactions",
            "window_functions": "supports_window_functions",
            "cte": "supports_cte",
            "json": "supports_json",
            "array": "supports_array",
            "full_text_search": "supports_full_text_search",
            "spatial": "supports_spatial",
            "partitions": "supports_partitions",
            "foreign_keys": "supports_foreign_keys",
            "check_constraints": "supports_check_constraints",
            "triggers": "supports_triggers",
            "stored_procedures": "supports_stored_procedures",
            "views": "supports_views",
            "materialized_views": "supports_materialized_views",
        }

        capability_attr = capability_map.get(feature)
        if capability_attr is None:
            return False

        return getattr(self.capabilities, capability_attr, False)

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the relational database.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            if not self._connected:
                await self.connect()

            # Simple query as health check
            result = await self.execute_query("SELECT 1 as health_check")

            return {
                "status": "healthy",
                "provider": self.provider_name,
                "dialect": self.dialect,
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "response_time_ms": None,  # Could be measured in implementations
                "test_query_success": len(result.rows) > 0,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": self.provider_name,
                "dialect": self.dialect,
                "host": self.host,
                "port": self.port,
                "database": self.database,
                "error": str(e)
            }

    def get_connection_info(self) -> Dict[str, Any]:
        """
        Get connection information.

        Returns:
            Dict[str, Any]: Connection information
        """
        return {
            "provider": self.provider_name,
            "dialect": self.dialect,
            "host": self.host,
            "port": self.port,
            "database": self.database,
            "username": self.username,
            "connected": self._connected,
            "capabilities": self.capabilities.dict(),
            "config": {
                "pool_size": self.config.pool_size,
                "max_overflow": self.config.max_overflow,
                "pool_timeout": self.config.pool_timeout,
                "pool_recycle": self.config.pool_recycle,
                "pool_pre_ping": self.config.pool_pre_ping,
                "echo": self.config.echo,
                "connect_timeout": self.config.connect_timeout,
                "query_timeout": self.config.query_timeout,
                "ssl_enabled": self.config.ssl_enabled,
            }
        }

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()


class RelationalDBException(Exception):
    """Exception raised by relational database providers."""

    def __init__(
        self,
        message: str,
        provider: str = None,
        database: str = None,
        table: str = None,
        query: str = None,
        error_code: str = None
    ):
        super().__init__(message)
        self.provider = provider
        self.database = database
        self.table = table
        self.query = query
        self.error_code = error_code


class ConnectionException(RelationalDBException):
    """Exception raised when connection fails."""
    pass


class QueryException(RelationalDBException):
    """Exception raised when query execution fails."""
    pass


class TransactionException(RelationalDBException):
    """Exception raised for transaction-related errors."""
    pass


class ValidationException(RelationalDBException):
    """Exception raised when validation fails."""
    pass