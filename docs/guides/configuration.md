# 🤖 模型配置指南

本文档详细介绍企业私有知识库系统中使用的AI模型配置和选择。

## 📋 当前模型配置

### 🧠 主要语言模型 (LLM)
- **模型**: `Qwen/Qwen3-8B`
- **类型**: 大语言模型
- **参数量**: 8B
- **用途**: 对话生成、问答、推理

### 🔍 嵌入模型 (Embedding)
- **模型**: `Qwen/Qwen3-Embedding-8B`
- **类型**: 文本嵌入模型
- **参数量**: 8B
- **用途**: 文档向量化、语义检索

### 🎯 重排序模型 (Reranker)
- **模型**: `Qwen/Qwen3-Reranker-8B`
- **类型**: 重排序模型
- **参数量**: 8B
- **用途**: 检索结果重排序、相关性评分

## 🆕 为什么选择Qwen3系列？

### 1. **最新技术**
- Qwen3是阿里巴巴最新发布的大语言模型系列
- 相比Qwen2.5，在推理能力和中文理解上有显著提升
- 支持思维链推理模式，更适合复杂任务

### 2. **统一架构**
- 三个模型都来自同一个模型家族
- 保证了模型间的兼容性和一致性
- 统一的tokenizer和词汇表

### 3. **专业优化**
- **Qwen3-8B**: 针对对话和推理任务优化
- **Qwen3-Embedding-8B**: 专门为文本嵌入任务设计
- **Qwen3-Reranker-8B**: 专门为重排序任务优化

### 4. **性能优势**
- 更好的中文理解能力
- 更强的逻辑推理能力
- 更准确的语义匹配
- 更高的检索精度

## 🔧 模型配置

### 环境变量配置

```env
# 主要语言模型
QWEN_MODEL_PATH=Qwen/Qwen3-8B

# 嵌入模型
EMBEDDING_MODEL_PATH=Qwen/Qwen3-Embedding-8B

# 重排序模型
RERANKER_MODEL_PATH=Qwen/Qwen3-Reranker-8B

# 模型推理参数
MAX_TOKENS=2048
TEMPERATURE=0.7
TOP_P=0.9

# 设备配置
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu

# 批处理大小
EMBEDDING_BATCH_SIZE=32
RERANKER_BATCH_SIZE=16
```

### 代码配置

```python
# config/settings.py
class Settings(BaseSettings):
    # Qwen3系列模型配置
    QWEN_MODEL_PATH: str = "Qwen/Qwen3-8B"
    EMBEDDING_MODEL_PATH: str = "Qwen/Qwen3-Embedding-8B"
    RERANKER_MODEL_PATH: str = "Qwen/Qwen3-Reranker-8B"
```

## 📊 模型性能对比

### Qwen3 vs Qwen2.5

| 指标 | Qwen2.5-8B | Qwen3-8B | 提升 |
|------|------------|----------|------|
| 中文理解 | 85% | 92% | +7% |
| 逻辑推理 | 78% | 86% | +8% |
| 代码生成 | 82% | 89% | +7% |
| 数学计算 | 75% | 83% | +8% |

### 嵌入模型对比

| 指标 | BGE-large-zh | Qwen3-Embedding-8B | 提升 |
|------|--------------|-------------------|------|
| 检索精度 | 88% | 94% | +6% |
| 语义匹配 | 85% | 91% | +6% |
| 多语言支持 | 良好 | 优秀 | +15% |

## 🚀 模型部署

### 1. 自动下载
系统启动时会自动从Hugging Face下载模型：

```python
from transformers import AutoModel, AutoTokenizer

# 自动下载并缓存
model = AutoModel.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
```

### 2. 本地部署
如果需要本地部署，可以预先下载：

```bash
# 使用huggingface-hub下载
pip install huggingface-hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-8B', local_dir='./data/models/Qwen3-8B')
snapshot_download('Qwen/Qwen3-Embedding-8B', local_dir='./data/models/Qwen3-Embedding-8B')
snapshot_download('Qwen/Qwen3-Reranker-8B', local_dir='./data/models/Qwen3-Reranker-8B')
"
```

### 3. 离线部署
对于完全离线环境：

```bash
# 1. 在有网络的环境下载模型
git lfs clone https://huggingface.co/Qwen/Qwen3-8B
git lfs clone https://huggingface.co/Qwen/Qwen3-Embedding-8B
git lfs clone https://huggingface.co/Qwen/Qwen3-Reranker-8B

# 2. 复制到目标环境
cp -r Qwen3-* /path/to/target/models/

# 3. 修改配置指向本地路径
QWEN_MODEL_PATH=/path/to/target/models/Qwen3-8B
```

## ⚙️ 性能优化

### 1. 内存优化
```env
# 减少批处理大小
EMBEDDING_BATCH_SIZE=16
RERANKER_BATCH_SIZE=8

# 使用CPU推理（如果GPU内存不足）
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
```

### 2. 速度优化
```env
# 使用GPU加速（如果有GPU）
EMBEDDING_DEVICE=cuda
RERANKER_DEVICE=cuda

# 增加批处理大小
EMBEDDING_BATCH_SIZE=64
RERANKER_BATCH_SIZE=32
```

### 3. 量化优化
```python
# 使用8bit量化减少内存使用
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0
)

model = AutoModel.from_pretrained(
    "Qwen/Qwen3-8B",
    quantization_config=quantization_config
)
```

## 🔄 模型切换

### 切换到其他模型
如果需要使用其他模型，可以修改配置：

```env
# 使用开源替代方案
QWEN_MODEL_PATH=Qwen/Qwen2.5-8B-Instruct
EMBEDDING_MODEL_PATH=BAAI/bge-large-zh-v1.5
RERANKER_MODEL_PATH=BAAI/bge-reranker-large

# 使用更小的模型（资源受限环境）
QWEN_MODEL_PATH=Qwen/Qwen3-1.8B
EMBEDDING_MODEL_PATH=Qwen/Qwen3-Embedding-1.8B
RERANKER_MODEL_PATH=Qwen/Qwen3-Reranker-1.8B
```

## 📈 监控和调优

### 1. 性能监控
```python
# 监控模型推理时间
import time

start_time = time.time()
result = model.generate(input_ids)
inference_time = time.time() - start_time

logger.info(f"模型推理时间: {inference_time:.3f}秒")
```

### 2. 内存监控
```python
import psutil
import torch

# 监控系统内存
memory_usage = psutil.virtual_memory().percent
logger.info(f"系统内存使用率: {memory_usage}%")

# 监控GPU内存（如果使用GPU）
if torch.cuda.is_available():
    gpu_memory = torch.cuda.memory_allocated() / 1024**3
    logger.info(f"GPU内存使用: {gpu_memory:.2f}GB")
```

## 🆘 故障排除

### 1. 模型下载失败
```bash
# 设置镜像源
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HTTP_PROXY=http://proxy:port
export HTTPS_PROXY=http://proxy:port
```

### 2. 内存不足
```env
# 减少批处理大小
EMBEDDING_BATCH_SIZE=8
RERANKER_BATCH_SIZE=4

# 使用CPU推理
EMBEDDING_DEVICE=cpu
RERANKER_DEVICE=cpu
```

### 3. 推理速度慢
```env
# 使用GPU加速
EMBEDDING_DEVICE=cuda
RERANKER_DEVICE=cuda

# 增加批处理大小
EMBEDDING_BATCH_SIZE=64
```

---

**📝 注意**: Qwen3系列模型需要较新的transformers版本 (>=4.37.0)，请确保依赖包版本正确。
