# 向量化存储策略详解

## 🎯 概述

分块后的数据通过一套完整的向量化存储策略存入向量数据库，包括向量化、索引构建、批量插入、多模态存储等多个环节。

## 📊 完整流程图

```
文档分块 → 向量化 → 批量处理 → 索引构建 → 多模态存储 → 检索优化
    ↓         ↓         ↓         ↓         ↓         ↓
  Chunks   Embeddings  Batches   Indexes   Storage   Search
```

## 🔧 核心策略和算法

### 1. 向量化策略

#### 1.1 嵌入模型选择
```python
# 使用Qwen3-Embedding-8B模型
EMBEDDING_MODEL_PATH = "Qwen/Qwen3-Embedding-8B"
EMBEDDING_DIM = 1024  # 向量维度
```

#### 1.2 文本预处理
```python
async def embed_documents(self, documents: List[Document]) -> List[np.ndarray]:
    """对文档进行向量化"""
    # 1. 提取文本内容
    texts = [doc.page_content for doc in documents]
    
    # 2. 文本清洗和标准化
    cleaned_texts = [self._preprocess_text(text) for text in texts]
    
    # 3. 批量向量化
    embeddings = await self._batch_embed(cleaned_texts)
    
    return embeddings
```

#### 1.3 批量向量化算法
```python
async def _batch_embed(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
    """批量向量化，避免内存溢出"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # 使用模型进行向量化
        batch_embeddings = self.model.encode(
            batch_texts,
            normalize_embeddings=True,  # L2标准化
            show_progress_bar=False
        )
        
        all_embeddings.extend(batch_embeddings)
    
    return all_embeddings
```

### 2. 存储策略

#### 2.1 数据结构设计
```python
# Milvus集合Schema
collection_schema = {
    "fields": [
        {"name": "id", "type": "INT64", "is_primary": True, "auto_id": True},
        {"name": "content_hash", "type": "VARCHAR", "max_length": 64},
        {"name": "content", "type": "VARCHAR", "max_length": 65535},
        {"name": "metadata", "type": "JSON"},
        {"name": "embedding", "type": "FLOAT_VECTOR", "dim": 1024},
        {"name": "created_at", "type": "VARCHAR", "max_length": 32}
    ]
}
```

#### 2.2 批量插入算法
```python
async def add_documents(self, documents: List[Document], embeddings: List[np.ndarray], batch_size: int = 100):
    """批量插入文档到向量数据库"""
    all_ids = []
    
    # 分批处理，避免单次插入过多数据
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_embeddings = embeddings[i:i + batch_size]
        
        # 准备插入数据
        data = [
            [doc.metadata.get('content_hash', f'hash_{i}') for doc in batch_docs],
            [doc.page_content for doc in batch_docs],
            [doc.metadata for doc in batch_docs],
            [emb.tolist() for emb in batch_embeddings],
            [current_time] * len(batch_docs)
        ]
        
        # 执行插入
        result = self.collection.insert(data)
        all_ids.extend(result.primary_keys)
    
    # 刷新到磁盘
    self.collection.flush()
    return all_ids
```

### 3. 索引构建策略

#### 3.1 向量索引算法
```python
# 使用IVF_FLAT索引，平衡性能和精度
index_params = {
    "metric_type": "COSINE",  # 余弦相似度
    "index_type": "IVF_FLAT",
    "params": {
        "nlist": 1024  # 聚类中心数量
    }
}
```

#### 3.2 索引优化策略
- **动态索引**: 根据数据量自动调整索引参数
- **分区索引**: 按文档类型或时间分区
- **混合索引**: 结合稠密向量和稀疏向量

### 4. 多模态存储策略

#### 4.1 不同数据类型的处理
```python
def get_storage_strategy(element_type: str) -> Dict[str, Any]:
    """根据元素类型选择存储策略"""
    strategies = {
        "text": {
            "embedding_model": "text_embedding",
            "index_type": "IVF_FLAT",
            "batch_size": 100
        },
        "table": {
            "embedding_model": "table_embedding", 
            "index_type": "IVF_PQ",  # 表格数据使用PQ压缩
            "batch_size": 50
        },
        "image": {
            "embedding_model": "multimodal_embedding",
            "index_type": "HNSW",  # 图像使用HNSW索引
            "batch_size": 20
        },
        "formula": {
            "embedding_model": "formula_embedding",
            "index_type": "IVF_FLAT",
            "batch_size": 30
        }
    }
    return strategies.get(element_type, strategies["text"])
```

#### 4.2 元数据增强策略
```python
def enhance_metadata(doc: Document, element_type: str) -> Dict[str, Any]:
    """增强文档元数据"""
    enhanced_metadata = doc.metadata.copy()
    
    # 添加向量化相关信息
    enhanced_metadata.update({
        "element_type": element_type,
        "embedding_model": "Qwen3-Embedding-8B",
        "embedding_dim": 1024,
        "storage_timestamp": datetime.utcnow().isoformat(),
        "content_length": len(doc.page_content),
        "content_hash": generate_hash(doc.page_content),
        
        # 检索优化信息
        "search_keywords": extract_keywords(doc.page_content),
        "semantic_tags": generate_semantic_tags(doc.page_content),
        "relevance_score": calculate_relevance_score(doc.page_content)
    })
    
    return enhanced_metadata
```

### 5. 检索优化策略

#### 5.1 混合检索架构
```python
class HybridRetrievalStrategy:
    """混合检索策略"""
    
    def __init__(self):
        self.vector_weight = 0.6    # 向量检索权重
        self.sparse_weight = 0.3    # 稀疏检索权重  
        self.fulltext_weight = 0.1  # 全文检索权重
    
    async def search(self, query: str, top_k: int = 10):
        # 1. 向量检索
        vector_results = await self.vector_search(query, top_k * 2)
        
        # 2. 稀疏检索 (BM25)
        sparse_results = await self.sparse_search(query, top_k * 2)
        
        # 3. 全文检索
        fulltext_results = await self.fulltext_search(query, top_k * 2)
        
        # 4. 结果融合
        final_results = self.fuse_results(
            vector_results, sparse_results, fulltext_results, top_k
        )
        
        return final_results
```

#### 5.2 动态索引更新
```python
class DynamicIndexManager:
    """动态索引管理器"""
    
    async def update_index_strategy(self):
        """根据数据特征动态调整索引"""
        data_stats = await self.analyze_data_distribution()
        
        if data_stats['total_docs'] > 1000000:
            # 大数据量使用分区索引
            await self.create_partitioned_index()
        elif data_stats['avg_doc_length'] > 2000:
            # 长文档使用层次索引
            await self.create_hierarchical_index()
        else:
            # 标准索引
            await self.create_standard_index()
```

## 🚀 性能优化策略

### 1. 内存优化
- **流式处理**: 大文件分批加载，避免内存溢出
- **向量压缩**: 使用PQ量化减少存储空间
- **缓存策略**: 热点数据内存缓存

### 2. 并发优化
- **异步处理**: 向量化和存储并行执行
- **连接池**: 数据库连接复用
- **批量操作**: 减少网络IO次数

### 3. 存储优化
- **数据分区**: 按时间或类型分区存储
- **索引优化**: 根据查询模式选择最优索引
- **压缩存储**: 向量数据压缩存储

## 📊 监控和调优

### 1. 性能指标监控
```python
class VectorStoreMonitor:
    """向量存储监控器"""
    
    def __init__(self):
        self.metrics = {
            "insert_latency": [],      # 插入延迟
            "search_latency": [],      # 搜索延迟
            "index_size": 0,           # 索引大小
            "memory_usage": 0,         # 内存使用
            "disk_usage": 0,           # 磁盘使用
            "query_qps": 0             # 查询QPS
        }
    
    async def collect_metrics(self):
        """收集性能指标"""
        # 实现指标收集逻辑
        pass
```

### 2. 自动调优算法
```python
class AutoTuner:
    """自动调优器"""
    
    async def optimize_parameters(self):
        """根据性能指标自动调优"""
        current_metrics = await self.monitor.collect_metrics()
        
        # 根据延迟调整批处理大小
        if current_metrics['insert_latency'] > 1000:  # 1秒
            self.batch_size = max(50, self.batch_size // 2)
        
        # 根据内存使用调整缓存大小
        if current_metrics['memory_usage'] > 0.8:  # 80%
            self.cache_size = max(1000, self.cache_size // 2)
        
        # 根据查询模式调整索引类型
        if current_metrics['query_qps'] > 1000:
            await self.switch_to_high_performance_index()
```

## 🎯 最佳实践

### 1. 数据准备
- 确保文档分块大小适中（512字符左右）
- 保留重要的元数据信息
- 进行适当的文本清洗

### 2. 向量化
- 使用高质量的嵌入模型
- 批量处理提高效率
- 向量标准化提高检索精度

### 3. 存储
- 合理设置批处理大小
- 定期刷新数据到磁盘
- 监控存储空间使用

### 4. 检索
- 使用混合检索提高召回率
- 根据查询类型选择合适的检索策略
- 实施结果重排序提高精度

这套向量化存储策略确保了高质量的RAG数据处理，为后续的语义检索提供了坚实的基础。
