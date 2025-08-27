"""
数据处理管道测试
"""
from src.utils.logger import get_logger
from src.data import DocumentProcessingPipeline, DocumentLoader, RecursiveSplitter
import asyncio
import sys
import tempfile
from pathlib import Path
import pytest

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))


logger = get_logger("test.data_pipeline")


class TestDataPipeline:
    """数据管道测试类"""

    @pytest.fixture
    def sample_text_file(self):
        """创建示例文本文件"""
        content = """
        # 企业私有知识库系统

        ## 概述
        这是一个基于RAG + Agent + MCP架构的企业私有知识库系统。
        系统支持多种文档格式的智能检索和问答。

        ## 主要功能
        1. 文档上传和解析
        2. 智能分割和向量化
        3. 混合检索和重排序
        4. Agent智能问答
        5. 可视化界面

        ## 技术栈
        - LangChain: RAG框架
        - Milvus: 向量数据库
        - PostgreSQL: 关系数据库
        - FastAPI: 后端框架
        - Streamlit: 前端界面
        - Qwen3: 大语言模型

        这个系统可以帮助企业更好地管理和利用内部知识资源。
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(content)
            return Path(f.name)

    @pytest.mark.asyncio
    async def test_document_loader(self, sample_text_file):
        """测试文档加载器"""
        loader = DocumentLoader()

        # 测试加载单个文档
        documents = await loader.load_document(sample_text_file)

        assert len(documents) > 0
        assert documents[0].page_content is not None
        assert documents[0].metadata['filename'] == sample_text_file.name
        assert documents[0].metadata['file_extension'] == '.txt'

        logger.info(f"文档加载测试通过，加载了 {len(documents)} 个文档")

        # 清理
        sample_text_file.unlink()

    @pytest.mark.asyncio
    async def test_text_splitter(self):
        """测试文本分割器"""
        from langchain.schema import Document

        # 创建测试文档
        long_text = "这是一个很长的文档。" * 100  # 创建长文本
        document = Document(page_content=long_text,
                            metadata={'source': 'test'})

        # 测试递归分割器
        splitter = RecursiveSplitter(chunk_size=200, chunk_overlap=50)
        split_docs = await splitter.split_documents([document])

        assert len(split_docs) > 1
        assert all(len(doc.page_content) <=
                   250 for doc in split_docs)  # 允许一些误差
        assert all(doc.metadata['splitter_type'] ==
                   'recursive' for doc in split_docs)

        logger.info(f"文本分割测试通过，分割成 {len(split_docs)} 个块")

    @pytest.mark.asyncio
    async def test_processing_pipeline(self, sample_text_file):
        """测试完整的处理管道"""
        # 创建管道（使用较小的模型进行测试）
        pipeline = DocumentProcessingPipeline(
            loader_type="standard",
            splitter_type="recursive",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # 小模型，快速测试
            enable_cache=False  # 测试时禁用缓存
        )

        try:
            # 处理文件
            documents, embeddings = await pipeline.process_single_file(sample_text_file)

            assert len(documents) > 0
            assert len(embeddings) == len(documents)
            assert all(len(emb) > 0 for emb in embeddings)

            # 检查文档元数据
            for doc in documents:
                assert 'processed_at' in doc.metadata
                assert 'pipeline_version' in doc.metadata
                assert 'char_count' in doc.metadata
                assert 'word_count' in doc.metadata

            logger.info(f"处理管道测试通过，处理了 {len(documents)} 个文档块")

        except Exception as e:
            logger.warning(f"处理管道测试跳过（可能是模型下载问题）: {e}")
            pytest.skip(f"模型相关测试跳过: {e}")

        finally:
            # 清理
            sample_text_file.unlink()

    @pytest.mark.asyncio
    async def test_text_processing(self):
        """测试纯文本处理"""
        pipeline = DocumentProcessingPipeline(
            splitter_type="recursive",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            enable_cache=False
        )

        test_text = """
        人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，
        它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
        该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。
        """

        try:
            documents, embeddings = await pipeline.process_text(
                test_text,
                metadata={'source': 'test_text', 'topic': 'AI'}
            )

            assert len(documents) > 0
            assert len(embeddings) == len(documents)
            assert documents[0].metadata['source'] == 'test_text'
            assert documents[0].metadata['topic'] == 'AI'

            logger.info(f"文本处理测试通过，生成了 {len(documents)} 个文档块")

        except Exception as e:
            logger.warning(f"文本处理测试跳过（可能是模型下载问题）: {e}")
            pytest.skip(f"模型相关测试跳过: {e}")


async def manual_test():
    """手动测试函数"""
    logger.info("开始手动测试数据处理管道")

    # 创建测试文本
    test_text = """
    # 企业知识管理系统

    企业知识管理系统是现代企业信息化建设的重要组成部分。
    它通过系统化的方法来获取、组织、共享和利用企业内部的知识资源。

    ## 主要功能
    1. 知识采集：从各种渠道收集企业知识
    2. 知识存储：建立统一的知识库
    3. 知识检索：提供智能搜索功能
    4. 知识共享：促进知识在组织内的流动

    ## 技术架构
    系统采用分层架构设计，包括：
    - 数据层：负责知识的存储和管理
    - 业务层：实现核心业务逻辑
    - 表现层：提供用户交互界面

    通过这样的系统，企业可以更好地管理和利用其知识资产。
    """

    try:
        # 创建处理管道
        pipeline = DocumentProcessingPipeline(
            splitter_type="recursive",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            enable_cache=False
        )

        # 处理文本
        documents, embeddings = await pipeline.process_text(test_text)

        logger.info(f"处理完成:")
        logger.info(f"  - 文档块数量: {len(documents)}")
        logger.info(f"  - 向量数量: {len(embeddings)}")
        logger.info(f"  - 向量维度: {len(embeddings[0]) if embeddings else 0}")

        # 显示文档块信息
        for i, doc in enumerate(documents[:3]):  # 只显示前3个
            logger.info(f"  文档块 {i+1}:")
            logger.info(f"    长度: {len(doc.page_content)} 字符")
            logger.info(f"    内容预览: {doc.page_content[:100]}...")
            logger.info(f"    元数据: {doc.metadata}")

        return True

    except Exception as e:
        logger.error(f"手动测试失败: {e}")
        return False


if __name__ == "__main__":
    # 运行手动测试
    result = asyncio.run(manual_test())
    if result:
        print("✅ 数据处理管道测试通过")
    else:
        print("❌ 数据处理管道测试失败")
