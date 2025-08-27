-- 企业私有知识库系统数据库初始化脚本

-- 创建数据库（如果不存在）
-- CREATE DATABASE epkbs;

-- 连接到数据库
\c epkbs;

-- 创建扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- 创建用户表
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(100),
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建文档表
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(255) NOT NULL,
    original_filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_size INTEGER NOT NULL,
    file_type VARCHAR(50) NOT NULL,
    content_hash VARCHAR(64) UNIQUE,
    status VARCHAR(20) DEFAULT 'pending',
    processing_error TEXT,
    doc_metadata JSONB,
    chunk_count INTEGER DEFAULT 0,
    vector_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
);

-- 创建文档块表
CREATE TABLE IF NOT EXISTS document_chunks (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) UNIQUE,
    chunk_index INTEGER NOT NULL,
    chunk_metadata JSONB,
    char_count INTEGER,
    word_count INTEGER,
    vector_id VARCHAR(100),
    embedding_model VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE
);

-- 创建对话表
CREATE TABLE IF NOT EXISTS conversations (
    id SERIAL PRIMARY KEY,
    title VARCHAR(200),
    model_name VARCHAR(100),
    use_rag BOOLEAN DEFAULT TRUE,
    max_tokens INTEGER DEFAULT 1000,
    temperature FLOAT DEFAULT 0.7,
    message_count INTEGER DEFAULT 0,
    total_tokens INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE
);

-- 创建消息表
CREATE TABLE IF NOT EXISTS messages (
    id SERIAL PRIMARY KEY,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    agent_steps JSONB,
    tools_used JSONB,
    execution_time FLOAT,
    token_count INTEGER,
    char_count INTEGER,
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conversation_id INTEGER REFERENCES conversations(id) ON DELETE CASCADE
);

-- 创建搜索日志表
CREATE TABLE IF NOT EXISTS search_logs (
    id SERIAL PRIMARY KEY,
    query TEXT NOT NULL,
    query_hash VARCHAR(64),
    top_k INTEGER DEFAULT 10,
    retriever_type VARCHAR(50),
    reranker_type VARCHAR(50),
    result_count INTEGER DEFAULT 0,
    execution_time FLOAT,
    retrieval_time FLOAT,
    rerank_time FLOAT,
    avg_score FLOAT,
    max_score FLOAT,
    min_score FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    user_id INTEGER REFERENCES users(id) ON DELETE SET NULL
);

-- 创建系统指标表
CREATE TABLE IF NOT EXISTS system_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    metric_unit VARCHAR(20),
    metric_metadata JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active);

CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_file_type ON documents(file_type);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_content_hash ON documents(content_hash);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_hash ON document_chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_document_chunks_vector_id ON document_chunks(vector_id);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);

CREATE INDEX IF NOT EXISTS idx_search_logs_user_id ON search_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_search_logs_query_hash ON search_logs(query_hash);
CREATE INDEX IF NOT EXISTS idx_search_logs_created_at ON search_logs(created_at);

CREATE INDEX IF NOT EXISTS idx_system_metrics_name ON system_metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);

-- 创建全文搜索索引
CREATE INDEX IF NOT EXISTS idx_documents_fulltext ON documents USING gin(to_tsvector('english', filename || ' ' || COALESCE(doc_metadata->>'title', '')));
CREATE INDEX IF NOT EXISTS idx_document_chunks_fulltext ON document_chunks USING gin(to_tsvector('english', content));

-- 创建触发器函数：更新updated_at字段
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 创建触发器
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_documents_updated_at BEFORE UPDATE ON documents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 插入默认管理员用户（密码: admin123）
INSERT INTO users (username, email, hashed_password, full_name, is_superuser) 
VALUES (
    'admin', 
    'admin@example.com', 
    '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj6QJw.2Oy2u', 
    '系统管理员', 
    TRUE
) ON CONFLICT (username) DO NOTHING;

-- 插入示例普通用户（密码: user123）
INSERT INTO users (username, email, hashed_password, full_name) 
VALUES (
    'demo_user', 
    'demo@example.com', 
    '$2b$12$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi', 
    '演示用户'
) ON CONFLICT (username) DO NOTHING;

-- 创建视图：用户统计
CREATE OR REPLACE VIEW user_stats AS
SELECT 
    u.id,
    u.username,
    u.email,
    u.full_name,
    u.created_at,
    COUNT(DISTINCT c.id) as conversation_count,
    COUNT(DISTINCT m.id) as message_count,
    COUNT(DISTINCT d.id) as document_count,
    COUNT(DISTINCT sl.id) as search_count
FROM users u
LEFT JOIN conversations c ON u.id = c.user_id
LEFT JOIN messages m ON c.id = m.conversation_id
LEFT JOIN documents d ON u.id = d.user_id
LEFT JOIN search_logs sl ON u.id = sl.user_id
WHERE u.is_active = TRUE
GROUP BY u.id, u.username, u.email, u.full_name, u.created_at;

-- 创建视图：系统统计
CREATE OR REPLACE VIEW system_stats AS
SELECT 
    (SELECT COUNT(*) FROM users WHERE is_active = TRUE) as active_users,
    (SELECT COUNT(*) FROM documents WHERE status = 'completed') as processed_documents,
    (SELECT COUNT(*) FROM conversations) as total_conversations,
    (SELECT COUNT(*) FROM messages) as total_messages,
    (SELECT COUNT(*) FROM search_logs) as total_searches,
    (SELECT SUM(file_size) FROM documents) as total_storage_bytes,
    (SELECT AVG(execution_time) FROM search_logs WHERE execution_time IS NOT NULL) as avg_search_time;

-- 授予权限
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO epkbs_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO epkbs_user;
GRANT ALL PRIVILEGES ON ALL FUNCTIONS IN SCHEMA public TO epkbs_user;

-- 输出初始化完成信息
\echo '数据库初始化完成！'
\echo '默认管理员账户: admin / admin123'
\echo '演示用户账户: demo_user / user123'
