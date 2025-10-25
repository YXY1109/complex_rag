"""
MySQL Relational Database Client Implementation

This module implements the MySQL client for relational database operations.
Based on the relational database interface abstract class.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from contextlib import asynccontextmanager
from datetime import datetime
import json

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.orm import sessionmaker, select, update, delete, insert
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError, DatabaseError, OperationalError
import aiomysql
from pydantic import BaseModel, Field

from ...interfaces.relational_db_interface import (
    RelationalDBInterface,
    RelationalDBConfig,
    RelationalDBCapabilities,
    RelationalQuery,
    RelationalResult,
    Transaction,
    QueryBuilder,
    DatabaseSchema,
    TableSchema,
    ColumnSchema,
    IndexSchema,
    RelationalDBException,
    ConnectionException,
    QueryException,
    TransactionException,
    ValidationException
)
from ...models import Base


logger = logging.getLogger(__name__)


class MySQLConfig(RelationalDBConfig):
    """MySQL-specific configuration."""

    host: str = Field(default="localhost", description="MySQL host address")
    port: int = Field(default=3306, description="MySQL port")
    database: str = Field(description="Database name")
    username: str = Field(description="MySQL username")
    password: str = Field(description="MySQL password")

    # MySQL-specific options
    charset: str = Field(default="utf8mb4", description="Character set")
    collation: str = Field(default="utf8mb4_unicode_ci", description="Collation")
    sql_mode: str = Field(default="STRICT_TRANS_TABLES", description="SQL mode")

    # Connection pooling
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Maximum overflow connections")
    pool_timeout: int = Field(default=30, description="Pool timeout in seconds")
    pool_recycle: int = Field(default=3600, description="Pool recycle time in seconds")

    # SSL options
    ssl_enabled: bool = Field(default=False, description="Enable SSL")
    ssl_ca: Optional[str] = Field(default=None, description="SSL CA file path")
    ssl_cert: Optional[str] = Field(default=None, description="SSL certificate file path")
    ssl_key: Optional[str] = Field(default=None, description="SSL key file path")


class MySQLCapabilities(RelationalDBCapabilities):
    """MySQL-specific capabilities."""

    def __init__(self):
        super().__init__(
            provider="mysql",
            supported_databases=["mysql", "mariadb"],
            supported_features=[
                "transactions", "foreign_keys", "indexes", "views",
                "stored_procedures", "triggers", "full_text_search",
                "json_support", "window_functions", "cte",
                "partitioning", "replication"
            ],
            max_connections=1000,
            supports_async=True,
            supports_pooling=True,
            supports_transactions=True,
            supports_foreign_keys=True,
            supports_json=True,
            supports_full_text_search=True,
            supports_window_functions=True,
            supports_cte=True,
            supports_partitioning=True,
            supports_replication=True
        )


class MySQLQueryBuilder(QueryBuilder):
    """MySQL-specific query builder."""

    def __init__(self):
        super().__init__()
        self._query_parts = {
            'select': [],
            'from': None,
            'joins': [],
            'where': [],
            'group_by': [],
            'having': [],
            'order_by': [],
            'limit': None,
            'offset': None,
            'lock': None
        }

    def select(self, *columns: str) -> 'MySQLQueryBuilder':
        """Add SELECT clause."""
        self._query_parts['select'].extend(columns)
        return self

    def from_table(self, table: str, alias: Optional[str] = None) -> 'MySQLQueryBuilder':
        """Add FROM clause."""
        if alias:
            self._query_parts['from'] = f"{table} AS {alias}"
        else:
            self._query_parts['from'] = table
        return self

    def join(self, table: str, on_condition: str, join_type: str = "INNER") -> 'MySQLQueryBuilder':
        """Add JOIN clause."""
        join_clause = f"{join_type} JOIN {table} ON {on_condition}"
        self._query_parts['joins'].append(join_clause)
        return self

    def left_join(self, table: str, on_condition: str) -> 'MySQLQueryBuilder':
        """Add LEFT JOIN clause."""
        return self.join(table, on_condition, "LEFT")

    def right_join(self, table: str, on_condition: str) -> 'MySQLQueryBuilder':
        """Add RIGHT JOIN clause."""
        return self.join(table, on_condition, "RIGHT")

    def where(self, condition: str) -> 'MySQLQueryBuilder':
        """Add WHERE clause."""
        self._query_parts['where'].append(condition)
        return self

    def where_in(self, column: str, values: List[Any]) -> 'MySQLQueryBuilder':
        """Add WHERE IN clause."""
        if not values:
            return self.where("1=0")  # Always false

        placeholders = ", ".join([f":{i}" for i in range(len(values))])
        condition = f"{column} IN ({placeholders})"
        return self.where(condition)

    def group_by(self, *columns: str) -> 'MySQLQueryBuilder':
        """Add GROUP BY clause."""
        self._query_parts['group_by'].extend(columns)
        return self

    def having(self, condition: str) -> 'MySQLQueryBuilder':
        """Add HAVING clause."""
        self._query_parts['having'].append(condition)
        return self

    def order_by(self, column: str, direction: str = "ASC") -> 'MySQLQueryBuilder':
        """Add ORDER BY clause."""
        direction = direction.upper()
        if direction not in ["ASC", "DESC"]:
            raise ValueError("Direction must be 'ASC' or 'DESC'")

        order_clause = f"{column} {direction}"
        self._query_parts['order_by'].append(order_clause)
        return self

    def limit(self, limit: int) -> 'MySQLQueryBuilder':
        """Add LIMIT clause."""
        self._query_parts['limit'] = limit
        return self

    def offset(self, offset: int) -> 'MySQLQueryBuilder':
        """Add OFFSET clause."""
        self._query_parts['offset'] = offset
        return self

    def for_update(self) -> 'MySQLQueryBuilder':
        """Add FOR UPDATE lock."""
        self._query_parts['lock'] = "FOR UPDATE"
        return self

    def build(self) -> Tuple[str, Dict[str, Any]]:
        """Build the final query."""
        query_parts = []
        params = {}

        # SELECT
        if self._query_parts['select']:
            query_parts.append(f"SELECT {', '.join(self._query_parts['select'])}")
        else:
            query_parts.append("SELECT *")

        # FROM
        if not self._query_parts['from']:
            raise ValueError("FROM clause is required")
        query_parts.append(f"FROM {self._query_parts['from']}")

        # JOINs
        query_parts.extend(self._query_parts['joins'])

        # WHERE
        if self._query_parts['where']:
            query_parts.append(f"WHERE {' AND '.join(self._query_parts['where'])}")

        # GROUP BY
        if self._query_parts['group_by']:
            query_parts.append(f"GROUP BY {', '.join(self._query_parts['group_by'])}")

        # HAVING
        if self._query_parts['having']:
            query_parts.append(f"HAVING {' AND '.join(self._query_parts['having'])}")

        # ORDER BY
        if self._query_parts['order_by']:
            query_parts.append(f"ORDER BY {', '.join(self._query_parts['order_by'])}")

        # LIMIT and OFFSET
        if self._query_parts['limit'] is not None:
            if self._query_parts['offset'] is not None:
                query_parts.append(f"LIMIT {self._query_parts['offset']}, {self._query_parts['limit']}")
            else:
                query_parts.append(f"LIMIT {self._query_parts['limit']}")

        # LOCK
        if self._query_parts['lock']:
            query_parts.append(self._query_parts['lock'])

        return " ".join(query_parts), params


class MySQLClient(RelationalDBInterface):
    """
    MySQL client implementation for relational database operations.

    Provides async MySQL operations with connection pooling,
    transaction management, and comprehensive error handling.
    """

    def __init__(self, config: MySQLConfig):
        super().__init__(config)
        self.config: MySQLConfig = config
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[sessionmaker] = None
        self._capabilities = MySQLCapabilities()
        self._is_connected = False

    @property
    def capabilities(self) -> RelationalDBCapabilities:
        """Get MySQL capabilities."""
        return self._capabilities

    async def connect(self) -> bool:
        """
        Connect to MySQL database.

        Returns:
            bool: True if connection successful

        Raises:
            ConnectionException: If connection fails
        """
        try:
            # Build connection URL
            connection_url = (
                f"mysql+aiomysql://{self.config.username}:{self.config.password}"
                f"@{self.config.host}:{self.config.port}/{self.config.database}"
                f"?charset={self.config.charset}"
            )

            # Add SSL parameters if enabled
            if self.config.ssl_enabled:
                ssl_params = []
                if self.config.ssl_ca:
                    ssl_params.append(f"ssl_ca={self.config.ssl_ca}")
                if self.config.ssl_cert:
                    ssl_params.append(f"ssl_cert={self.config.ssl_cert}")
                if self.config.ssl_key:
                    ssl_params.append(f"ssl_key={self.config.ssl_key}")
                if ssl_params:
                    connection_url += f"&{'&'.join(ssl_params)}"

            # Create async engine
            self._engine = create_async_engine(
                connection_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=self.config.debug_mode,
                future=True
            )

            # Create session factory
            self._session_factory = sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                future=True
            )

            # Test connection
            async with self._engine.begin() as conn:
                await conn.execute(text("SELECT 1"))

            self._is_connected = True
            logger.info(f"Connected to MySQL database: {self.config.database}")
            return True

        except OperationalError as e:
            error_msg = f"Failed to connect to MySQL: {str(e)}"
            logger.error(error_msg)
            raise ConnectionException(error_msg, provider="mysql") from e
        except Exception as e:
            error_msg = f"Unexpected error connecting to MySQL: {str(e)}"
            logger.error(error_msg)
            raise ConnectionException(error_msg, provider="mysql") from e

    async def disconnect(self) -> None:
        """Disconnect from MySQL database."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._is_connected = False
            logger.info("Disconnected from MySQL database")

    @asynccontextmanager
    async def get_session(self) -> AsyncSession:
        """
        Get a database session.

        Yields:
            AsyncSession: Database session
        """
        if not self._is_connected or not self._session_factory:
            raise ConnectionException("Not connected to database", provider="mysql")

        async with self._session_factory() as session:
            try:
                yield session
            except Exception as e:
                await session.rollback()
                raise e

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> RelationalResult:
        """
        Execute a raw SQL query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            RelationalResult: Query results

        Raises:
            QueryException: If query execution fails
        """
        try:
            async with self.get_session() as session:
                result = await session.execute(text(query), params or {})

                # Get column names
                if result.returns_rows:
                    columns = list(result.keys())
                    rows = result.fetchall()

                    # Convert to list of dictionaries
                    data = []
                    for row in rows:
                        row_dict = dict(zip(columns, row))
                        # Handle datetime serialization
                        for key, value in row_dict.items():
                            if isinstance(value, datetime):
                                row_dict[key] = value.isoformat()
                        data.append(row_dict)
                else:
                    columns = []
                    rows = []
                    data = []

                return RelationalResult(
                    success=True,
                    data=data,
                    row_count=len(data),
                    columns=columns,
                    affected_rows=result.rowcount if hasattr(result, 'rowcount') else 0
                )

        except SQLAlchemyError as e:
            error_msg = f"Query execution failed: {str(e)}"
            logger.error(error_msg)
            raise QueryException(error_msg, provider="mysql") from e

    async def select(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        where: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> RelationalResult:
        """
        Execute SELECT query.

        Args:
            table: Table name
            columns: Columns to select
            where: WHERE clause
            params: Query parameters
            order_by: ORDER BY clause
            limit: LIMIT clause
            offset: OFFSET clause

        Returns:
            RelationalResult: Query results
        """
        # Build query
        query_parts = ["SELECT"]

        if columns:
            query_parts.append(", ".join(columns))
        else:
            query_parts.append("*")

        query_parts.append(f"FROM {table}")

        if where:
            query_parts.append(f"WHERE {where}")

        if order_by:
            query_parts.append(f"ORDER BY {order_by}")

        if limit is not None:
            if offset is not None:
                query_parts.append(f"LIMIT {offset}, {limit}")
            else:
                query_parts.append(f"LIMIT {limit}")

        query = " ".join(query_parts)
        return await self.execute_query(query, params)

    async def insert(
        self,
        table: str,
        data: Union[Dict[str, Any], List[Dict[str, Any]]],
        on_conflict: Optional[str] = None
    ) -> RelationalResult:
        """
        Execute INSERT query.

        Args:
            table: Table name
            data: Data to insert
            on_conflict: ON CONFLICT clause

        Returns:
            RelationalResult: Insert result
        """
        if isinstance(data, dict):
            data = [data]

        if not data:
            return RelationalResult(success=True, row_count=0)

        # Get columns from first row
        columns = list(data[0].keys())
        placeholders = ", ".join([f":{col}" for col in columns])

        # Build query
        query_parts = [
            f"INSERT INTO {table}",
            f"({', '.join(columns)})",
            f"VALUES ({placeholders})"
        ]

        if on_conflict:
            query_parts.append(on_conflict)

        query = " ".join(query_parts)

        # Flatten parameters for multiple rows
        all_params = {}
        param_index = 0
        values_list = []

        for row in data:
            row_params = {}
            for col in columns:
                param_name = f"{col}_{param_index}"
                row_params[param_name] = row[col]
                all_params[param_name] = row[col]
            values_list.append(row_params)
            param_index += 1

        # For multiple rows, we need to build the query differently
        if len(data) > 1:
            values_placeholders = []
            for i, row in enumerate(data):
                row_placeholders = []
                for col in columns:
                    row_placeholders.append(f":{col}_{i}")
                values_placeholders.append(f"({', '.join(row_placeholders)})")

            query_parts = [
                f"INSERT INTO {table}",
                f"({', '.join(columns)})",
                f"VALUES {', '.join(values_placeholders)}"
            ]

            if on_conflict:
                query_parts.append(on_conflict)

            query = " ".join(query_parts)

        return await self.execute_query(query, all_params)

    async def update(
        self,
        table: str,
        data: Dict[str, Any],
        where: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> RelationalResult:
        """
        Execute UPDATE query.

        Args:
            table: Table name
            data: Data to update
            where: WHERE clause
            params: Query parameters

        Returns:
            RelationalResult: Update result
        """
        if not data:
            return RelationalResult(success=True, row_count=0)

        # Build SET clause
        set_clauses = []
        update_params = params or {}

        for key, value in data.items():
            param_name = f"update_{key}"
            set_clauses.append(f"{key} = :{param_name}")
            update_params[param_name] = value

        # Build query
        query_parts = [
            f"UPDATE {table}",
            f"SET {', '.join(set_clauses)}"
        ]

        if where:
            query_parts.append(f"WHERE {where}")

        query = " ".join(query_parts)
        return await self.execute_query(query, update_params)

    async def delete(
        self,
        table: str,
        where: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> RelationalResult:
        """
        Execute DELETE query.

        Args:
            table: Table name
            where: WHERE clause
            params: Query parameters

        Returns:
            RelationalResult: Delete result
        """
        query_parts = [f"DELETE FROM {table}"]

        if where:
            query_parts.append(f"WHERE {where}")

        query = " ".join(query_parts)
        return await self.execute_query(query, params)

    async def create_transaction(self) -> Transaction:
        """
        Create a new transaction.

        Returns:
            Transaction: New transaction
        """
        return MySQLTransaction(self)

    async def get_schema(self, table_name: Optional[str] = None) -> DatabaseSchema:
        """
        Get database schema information.

        Args:
            table_name: Specific table name (optional)

        Returns:
            DatabaseSchema: Schema information
        """
        try:
            async with self.get_session() as session:
                if table_name:
                    # Get specific table schema
                    table_query = """
                        SELECT
                            COLUMN_NAME,
                            DATA_TYPE,
                            IS_NULLABLE,
                            COLUMN_DEFAULT,
                            COLUMN_KEY,
                            EXTRA,
                            CHARACTER_MAXIMUM_LENGTH,
                            NUMERIC_PRECISION,
                            NUMERIC_SCALE
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = :database AND TABLE_NAME = :table
                        ORDER BY ORDINAL_POSITION
                    """

                    result = await session.execute(text(table_query), {
                        'database': self.config.database,
                        'table': table_name
                    })

                    columns = []
                    for row in result.fetchall():
                        columns.append(ColumnSchema(
                            name=row[0],
                            data_type=row[1],
                            is_nullable=row[2] == 'YES',
                            default_value=row[3],
                            is_primary_key=row[4] == 'PRI',
                            max_length=row[6],
                            precision=row[7],
                            scale=row[8]
                        ))

                    # Get index information
                    index_query = """
                        SELECT
                            INDEX_NAME,
                            COLUMN_NAME,
                            NON_UNIQUE,
                            SEQ_IN_INDEX
                        FROM INFORMATION_SCHEMA.STATISTICS
                        WHERE TABLE_SCHEMA = :database AND TABLE_NAME = :table
                        ORDER BY INDEX_NAME, SEQ_IN_INDEX
                    """

                    result = await session.execute(text(index_query), {
                        'database': self.config.database,
                        'table': table_name
                    })

                    indexes = {}
                    for row in result.fetchall():
                        index_name = row[0]
                        if index_name not in indexes:
                            indexes[index_name] = {
                                'name': index_name,
                                'columns': [],
                                'is_unique': row[2] == 0,
                                'is_primary': index_name == 'PRIMARY'
                            }
                        indexes[index_name]['columns'].append(row[1])

                    index_schemas = [
                        IndexSchema(
                            name=index_data['name'],
                            columns=index_data['columns'],
                            is_unique=index_data['is_unique'],
                            is_primary=index_data['is_primary']
                        )
                        for index_data in indexes.values()
                    ]

                    table_schema = TableSchema(
                        name=table_name,
                        columns=columns,
                        indexes=index_schemas
                    )

                    return DatabaseSchema(
                        database_name=self.config.database,
                        tables=[table_schema]
                    )
                else:
                    # Get all tables
                    tables_query = """
                        SELECT TABLE_NAME, TABLE_TYPE, ENGINE, TABLE_ROWS, DATA_LENGTH, INDEX_LENGTH
                        FROM INFORMATION_SCHEMA.TABLES
                        WHERE TABLE_SCHEMA = :database
                        ORDER BY TABLE_NAME
                    """

                    result = await session.execute(text(tables_query), {
                        'database': self.config.database
                    })

                    tables = []
                    for row in result.fetchall():
                        tables.append(TableSchema(
                            name=row[0],
                            table_type=row[1],
                            engine=row[2],
                            estimated_rows=row[3],
                            data_size=row[4],
                            index_size=row[5]
                        ))

                    return DatabaseSchema(
                        database_name=self.config.database,
                        tables=tables
                    )

        except SQLAlchemyError as e:
            error_msg = f"Failed to get schema: {str(e)}"
            logger.error(error_msg)
            raise QueryException(error_msg, provider="mysql") from e

    def create_query_builder(self) -> QueryBuilder:
        """
        Create a query builder.

        Returns:
            QueryBuilder: MySQL query builder
        """
        return MySQLQueryBuilder()

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on MySQL connection.

        Returns:
            Dict[str, Any]: Health check result
        """
        try:
            if not self._is_connected:
                return {
                    "status": "unhealthy",
                    "provider": "mysql",
                    "error": "Not connected to database"
                }

            start_time = asyncio.get_event_loop().time()

            async with self.get_session() as session:
                await session.execute(text("SELECT 1"))

            end_time = asyncio.get_event_loop().time()
            response_time_ms = (end_time - start_time) * 1000

            # Get connection pool info
            pool_info = {}
            if self._engine and hasattr(self._engine.pool, 'size'):
                pool_info = {
                    "pool_size": self._engine.pool.size(),
                    "checked_in": self._engine.pool.checkedin(),
                    "checked_out": self._engine.pool.checkedout()
                }

            return {
                "status": "healthy",
                "provider": "mysql",
                "database": self.config.database,
                "host": self.config.host,
                "response_time_ms": response_time_ms,
                "pool_info": pool_info
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "provider": "mysql",
                "database": self.config.database,
                "error": str(e)
            }


class MySQLTransaction(Transaction):
    """
    MySQL transaction implementation.
    """

    def __init__(self, client: MySQLClient):
        super().__init__()
        self._client = client
        self._session: Optional[AsyncSession] = None
        self._is_active = False

    async def begin(self) -> None:
        """Begin transaction."""
        if self._is_active:
            raise TransactionException("Transaction is already active", provider="mysql")

        try:
            self._session = self._client._session_factory()
            await self._session.begin()
            self._is_active = True
            logger.debug("MySQL transaction started")
        except Exception as e:
            raise TransactionException(f"Failed to begin transaction: {str(e)}", provider="mysql") from e

    async def commit(self) -> None:
        """Commit transaction."""
        if not self._is_active or not self._session:
            raise TransactionException("No active transaction to commit", provider="mysql")

        try:
            await self._session.commit()
            self._is_active = False
            logger.debug("MySQL transaction committed")
        except Exception as e:
            await self.rollback()
            raise TransactionException(f"Failed to commit transaction: {str(e)}", provider="mysql") from e

    async def rollback(self) -> None:
        """Rollback transaction."""
        if not self._is_active or not self._session:
            return

        try:
            await self._session.rollback()
            self._is_active = False
            logger.debug("MySQL transaction rolled back")
        except Exception as e:
            logger.error(f"Failed to rollback transaction: {str(e)}")

    async def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> RelationalResult:
        """Execute query within transaction."""
        if not self._is_active or not self._session:
            raise TransactionException("No active transaction", provider="mysql")

        try:
            result = await self._session.execute(text(query), params or {})

            if result.returns_rows:
                columns = list(result.keys())
                rows = result.fetchall()
                data = [dict(zip(columns, row)) for row in rows]
            else:
                columns = []
                rows = []
                data = []

            return RelationalResult(
                success=True,
                data=data,
                row_count=len(data),
                columns=columns,
                affected_rows=result.rowcount if hasattr(result, 'rowcount') else 0
            )

        except SQLAlchemyError as e:
            raise QueryException(f"Query execution failed in transaction: {str(e)}", provider="mysql") from e

    async def close(self) -> None:
        """Close transaction and session."""
        if self._session:
            if self._is_active:
                await self.rollback()
            await self._session.close()
            self._session = None
            self._is_active = False
            logger.debug("MySQL transaction closed")