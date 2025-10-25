-- RAG系统数据库初始化脚本

-- 创建数据库
CREATE DATABASE IF NOT EXISTS ragdb CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- 使用数据库
USE ragdb;

-- 创建用户和权限（如果还没有）
CREATE USER IF NOT EXISTS 'raguser'@'%' IDENTIFIED BY 'ragpass';
GRANT ALL PRIVILEGES ON ragdb.* TO 'raguser'@'%';
FLUSH PRIVILEGES;

-- 创建基础表结构（简化版本）
-- 注意：实际应用中应该使用SQLAlchemy的迁移功能来创建表

-- 用户表
CREATE TABLE IF NOT EXISTS users (
    id VARCHAR(36) PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_username (username),
    INDEX idx_email (email)
);

-- 租户表
CREATE TABLE IF NOT EXISTS tenants (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    plan VARCHAR(50) DEFAULT 'basic',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_name (name)
);

-- 用户租户关联表
CREATE TABLE IF NOT EXISTS user_tenants (
    id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    tenant_id VARCHAR(36) NOT NULL,
    role VARCHAR(50) DEFAULT 'member',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id) ON DELETE CASCADE,
    UNIQUE KEY uk_user_tenant (user_id, tenant_id),
    INDEX idx_user_id (user_id),
    INDEX idx_tenant_id (tenant_id)
);

-- 知识库表
CREATE TABLE IF NOT EXISTS knowledge_bases (
    id VARCHAR(36) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    tenant_id VARCHAR(36) NOT NULL,
    config JSON,
    created_by VARCHAR(36) NOT NULL,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (created_by) REFERENCES users(id),
    INDEX idx_tenant_id (tenant_id),
    INDEX idx_name (name),
    INDEX idx_status (status)
);

-- 文档表
CREATE TABLE IF NOT EXISTS documents (
    id VARCHAR(36) PRIMARY KEY,
    kb_id VARCHAR(36) NOT NULL,
    title VARCHAR(200) NOT NULL,
    content LONGTEXT,
    file_path VARCHAR(500),
    file_type VARCHAR(50),
    file_size INTEGER DEFAULT 0,
    metadata JSON,
    created_by VARCHAR(36) NOT NULL,
    status VARCHAR(50) DEFAULT 'processing',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (kb_id) REFERENCES knowledge_bases(id) ON DELETE CASCADE,
    FOREIGN KEY (created_by) REFERENCES users(id),
    INDEX idx_kb_id (kb_id),
    INDEX idx_title (title),
    INDEX idx_status (status),
    INDEX idx_created_by (created_by)
);

-- 聊天会话表
CREATE TABLE IF NOT EXISTS chat_sessions (
    id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(100),
    user_id VARCHAR(36) NOT NULL,
    tenant_id VARCHAR(36) NOT NULL,
    config JSON,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id (user_id),
    INDEX idx_tenant_id (tenant_id)
);

-- 聊天消息表
CREATE TABLE IF NOT EXISTS chat_messages (
    id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    content TEXT NOT NULL,
    role VARCHAR(50) NOT NULL,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE,
    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at)
);

-- 插入默认数据
INSERT IGNORE INTO users (id, username, email, password_hash, full_name) VALUES
('default-user-id', 'admin', 'admin@example.com', 'hashed_password', '系统管理员');

INSERT IGNORE INTO tenants (id, name, description, plan) VALUES
('default-tenant-id', '默认租户', '系统默认租户', 'premium');

INSERT IGNORE INTO user_tenants (id, user_id, tenant_id, role) VALUES
('default-user-tenant-id', 'default-user-id', 'default-tenant-id', 'admin');

-- 创建索引优化查询性能
CREATE INDEX IF NOT EXISTS idx_documents_content ON documents(content(100));
CREATE INDEX IF NOT EXISTS idx_chat_messages_content ON chat_messages(content(100));
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON chat_messages(created_at);

-- 显示初始化完成信息
SELECT 'RAG数据库初始化完成' as message;