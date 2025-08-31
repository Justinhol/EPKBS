# 文档解析模块

## 概述

本模块为企业私有知识库系统提供了强大的文档解析功能，支持20+种文档格式，采用多模态解析、智能整合、内容清洗和分块策略，为RAG系统提供高质量的数据。

## 核心特性

### 🎯 多格式支持
- **Office文档**: DOC, DOCX, PPT, PPTX, XLS, XLSX
- **PDF文档**: 文本版PDF, 扫描版PDF（OCR）
- **网页标记**: HTML, MD, XML
- **电子书**: EPUB
- **文本文档**: TXT
- **邮件**: EML, MSG
- **数据文件**: CSV, JSON
- **图像文件**: JPG, PNG, TIFF（OCR识别）

### 🔧 解析架构

#### 1. 文档类型检查
- 前后端双重验证
- MIME类型检测
- 文件完整性校验
- 特殊元素识别（表格、图片、公式）

#### 2. 专业解析工具
- **PDF**: PyMuPDF（文本版） + PaddleOCR（扫描版）
- **Office**: python-docx, python-pptx, pandas + openpyxl
- **网页**: BeautifulSoup + markdown
- **图像**: PaddleOCR + PIL
- **数据**: pandas + json

#### 3. 多工具协同
- 主解析器 + 特殊解析器
- 分而治之的处理策略
- 自动回退机制

#### 4. 数据整合
- 上下文关系保留
- 交叉引用建立
- 位置信息追踪
- 结构化映射

#### 5. 内容清洗
- 噪声自动移除（页眉页脚、水印、广告）
- OCR错误修复
- 格式标准化
- 重复内容去除

#### 6. 智能分块
- **结构分块**: 按章节、标题分割
- **语义分块**: 按语义边界分割  
- **递归分块**: 递归字符分割
- **自适应**: 根据内容类型调整策略

## 使用方法

### 基础使用

```python
from src.data.loaders import AdvancedDocumentLoader

# 创建加载器
loader = AdvancedDocumentLoader()

# 处理单个文档
documents = await loader.load_and_process_document("path/to/document.pdf")

# 批量处理
file_paths = ["doc1.pdf", "doc2.docx", "doc3.xlsx"]
all_documents = await loader.load_documents(file_paths)
```

### 高级使用

```python
from src.data.document_validator import document_validator
from src.data.parsers.base import parser_registry
from src.data.parsers.integrator import document_integrator
from src.data.parsers.cleaner import document_cleaner
from src.data.parsers.chunker import adaptive_chunker

# 1. 文档验证
doc_info = document_validator.validate_file("document.pdf")

# 2. 选择解析器
parser = parser_registry.get_primary_parser(doc_info.file_type)

# 3. 执行解析
parse_result = await parser.parse(file_path, doc_info)

# 4. 整合结果
integrated_doc = await document_integrator.integrate_parse_results([parse_result], doc_info)

# 5. 清洗内容
cleaned_doc = document_cleaner.clean_integrated_document(integrated_doc)

# 6. 智能分块
final_documents = adaptive_chunker.chunk_integrated_document(cleaned_doc)
```

## 配置选项

在 `config/settings.py` 中可以配置以下选项：

```python
# 文档解析配置
ENABLE_OCR = True  # 启用OCR功能
OCR_LANGUAGE = "ch"  # OCR语言设置
ENABLE_TABLE_EXTRACTION = True  # 启用表格提取
ENABLE_IMAGE_DESCRIPTION = True  # 启用图像描述
ENABLE_FORMULA_EXTRACTION = True  # 启用公式提取

# 分块策略配置
CHUNK_SIZE = 512  # 目标块大小
CHUNK_OVERLAP = 77  # 重叠大小（15%）
ENABLE_STRUCTURAL_CHUNKING = True  # 启用结构分块
ENABLE_SEMANTIC_CHUNKING = True  # 启用语义分块
ENABLE_RECURSIVE_CHUNKING = True  # 启用递归分块
```

## 错误处理

模块提供了强大的错误处理机制：

```python
from src.data.parsers.error_handler import global_error_handler, SafeParserWrapper

# 使用安全包装器
safe_parser = SafeParserWrapper(parser, global_error_handler)
result = await safe_parser.parse(file_path, doc_info)

# 查看错误统计
error_summary = global_error_handler.get_error_summary()
print(f"总错误数: {error_summary['total_errors']}")
print(f"恢复成功率: {error_summary['recovery_success_rate']:.2%}")
```

## 性能优化

### 内存优化
- 流式处理大文件
- 及时释放资源
- 分批处理机制

### 速度优化
- 异步并发处理
- 智能缓存机制
- 预处理优化

### 质量优化
- 多工具验证
- 置信度评估
- 自动质量检查

## 扩展开发

### 添加新的解析器

```python
from src.data.parsers.base import BaseParser, ParseResult, ParsedElement

class CustomParser(BaseParser):
    def __init__(self):
        super().__init__("CustomParser")
        self.supported_types = [DocumentType.CUSTOM]
    
    async def parse(self, file_path, doc_info):
        # 实现解析逻辑
        elements = []
        # ... 解析代码 ...
        
        return ParseResult(
            success=True,
            elements=elements,
            metadata={'parser': self.name}
        )

# 注册解析器
from src.data.parsers.base import parser_registry
parser_registry.register_parser(CustomParser(), is_primary=True)
```

### 添加新的文档类型

```python
from src.data.document_validator import DocumentType

# 在DocumentType枚举中添加新类型
class DocumentType(Enum):
    # ... 现有类型 ...
    CUSTOM = "custom"

# 在DocumentValidator中添加支持
validator.extension_mapping['.custom'] = DocumentType.CUSTOM
validator.mime_mapping['application/custom'] = DocumentType.CUSTOM
```

## 测试

运行测试套件：

```bash
# 运行所有测试
python -m pytest tests/test_document_parsing.py -v

# 运行特定测试
python -m pytest tests/test_document_parsing.py::TestDocumentValidator -v

# 运行演示脚本
python demo_document_parsing.py
```

## 依赖安装

```bash
# 安装核心依赖
pip install -r requirements_document_parsing.txt

# 可选：安装高级功能依赖
pip install unstructured transformers torch
```

## 注意事项

1. **OCR功能**: 需要安装PaddleOCR，首次使用会下载模型文件
2. **内存使用**: 处理大文件时注意内存使用，建议设置合适的批处理大小
3. **文件权限**: 确保有足够的文件读取权限
4. **编码问题**: 自动处理多种编码格式，但建议使用UTF-8
5. **网络依赖**: 某些功能可能需要网络连接下载模型

## 故障排除

### 常见问题

1. **OCR初始化失败**
   - 检查PaddleOCR安装
   - 确认网络连接（首次下载模型）
   - 检查磁盘空间

2. **内存不足**
   - 减小CHUNK_SIZE
   - 启用流式处理
   - 分批处理文件

3. **解析失败**
   - 检查文件完整性
   - 确认文件格式支持
   - 查看错误日志

4. **性能问题**
   - 调整并发数量
   - 优化分块策略
   - 使用缓存机制

## 更新日志

### v1.0.0
- 初始版本发布
- 支持20+种文档格式
- 实现多模态解析架构
- 添加智能分块策略
- 完善错误处理机制

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支
3. 编写测试用例
4. 提交代码
5. 创建 Pull Request

## 许可证

本项目采用 MIT 许可证。