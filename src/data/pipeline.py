"""
数据处理管道
整合文档加载、分割、向量化等步骤
"""
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

from langchain.schema import Document

from src.data.loaders import DocumentLoader, UnstructuredDocumentLoader, DocumentMetadataExtractor
from src.data.splitters import SplitterFactory, BaseSplitter
from src.data.embeddings import CachedEmbeddingManager
from src.utils.logger import get_logger
from src.utils.helpers import Timer
from config.settings import settings

logger = get_logger("data.pipeline")


class DocumentProcessingPipeline:
    """文档处理管道"""
    
    def __init__(
        self,
        loader_type: str = "standard",
        splitter_type: str = "hybrid",
        embedding_model: str = None,
        enable_cache: bool = True,
    ):
        self.loader_type = loader_type
        self.splitter_type = splitter_type
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL_PATH
        self.enable_cache = enable_cache
        
        # 初始化组件
        self.document_loader = None
        self.metadata_extractor = DocumentMetadataExtractor()
        self.splitter = None
        self.embedding_manager = None
        
        logger.info(f"文档处理管道初始化: loader={loader_type}, splitter={splitter_type}, embedding={self.embedding_model}")
    
    async def initialize(self):
        """初始化管道组件"""
        logger.info("正在初始化文档处理管道...")
        
        # 初始化文档加载器
        if self.loader_type == "unstructured":
            self.document_loader = UnstructuredDocumentLoader()
        else:
            self.document_loader = DocumentLoader()
        
        # 初始化分割器
        self.splitter = SplitterFactory.create_splitter(
            self.splitter_type,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        
        # 初始化嵌入管理器
        self.embedding_manager = CachedEmbeddingManager(
            model_name=self.embedding_model,
            enable_cache=self.enable_cache
        )
        await self.embedding_manager.initialize()
        
        logger.info("文档处理管道初始化完成")
    
    async def process_files(
        self,
        file_paths: List[Union[str, Path]],
        batch_size: int = 10
    ) -> Tuple[List[Document], List[List[float]]]:
        """
        处理文件列表
        
        Args:
            file_paths: 文件路径列表
            batch_size: 批处理大小
            
        Returns:
            (文档列表, 嵌入向量列表)
        """
        if not self.document_loader:
            await self.initialize()
        
        logger.info(f"开始处理 {len(file_paths)} 个文件")
        
        with Timer() as total_timer:
            all_documents = []
            all_embeddings = []
            
            # 分批处理文件
            for i in range(0, len(file_paths), batch_size):
                batch_files = file_paths[i:i + batch_size]
                logger.info(f"处理批次 {i//batch_size + 1}/{(len(file_paths) + batch_size - 1)//batch_size}")
                
                try:
                    # 处理当前批次
                    documents, embeddings = await self._process_batch(batch_files)
                    all_documents.extend(documents)
                    all_embeddings.extend(embeddings)
                    
                except Exception as e:
                    logger.error(f"批次处理失败: {e}")
                    continue
        
        logger.info(f"文件处理完成，总耗时: {total_timer.elapsed:.2f}秒")
        logger.info(f"共生成 {len(all_documents)} 个文档块，{len(all_embeddings)} 个向量")
        
        return all_documents, all_embeddings
    
    async def _process_batch(
        self,
        file_paths: List[Union[str, Path]]
    ) -> Tuple[List[Document], List[List[float]]]:
        """处理文件批次"""
        
        # 1. 加载文档
        logger.debug(f"加载 {len(file_paths)} 个文件")
        documents = await self.document_loader.load_documents(file_paths)
        
        if not documents:
            logger.warning("没有成功加载任何文档")
            return [], []
        
        # 2. 提取元数据
        logger.debug("提取文档元数据")
        for doc in documents:
            enhanced_metadata = self.metadata_extractor.extract_metadata(doc)
            doc.metadata.update(enhanced_metadata)
        
        # 3. 分割文档
        logger.debug(f"分割 {len(documents)} 个文档")
        split_documents = await self.splitter.split_documents(documents)
        
        # 4. 向量化
        logger.debug(f"向量化 {len(split_documents)} 个文档块")
        embeddings = await self.embedding_manager.embed_documents(split_documents)
        
        # 5. 添加处理时间戳
        current_time = datetime.utcnow().isoformat()
        for doc in split_documents:
            doc.metadata['processed_at'] = current_time
            doc.metadata['pipeline_version'] = '1.0'
        
        return split_documents, embeddings
    
    async def process_single_file(
        self,
        file_path: Union[str, Path]
    ) -> Tuple[List[Document], List[List[float]]]:
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            (文档列表, 嵌入向量列表)
        """
        return await self.process_files([file_path])
    
    async def process_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> Tuple[List[Document], List[List[float]]]:
        """
        处理纯文本
        
        Args:
            text: 文本内容
            metadata: 元数据
            
        Returns:
            (文档列表, 嵌入向量列表)
        """
        if not self.splitter:
            await self.initialize()
        
        # 创建文档对象
        document = Document(
            page_content=text,
            metadata=metadata or {}
        )
        
        # 提取元数据
        enhanced_metadata = self.metadata_extractor.extract_metadata(document)
        document.metadata.update(enhanced_metadata)
        
        # 分割文档
        split_documents = await self.splitter.split_documents([document])
        
        # 向量化
        embeddings = await self.embedding_manager.embed_documents(split_documents)
        
        # 添加处理时间戳
        current_time = datetime.utcnow().isoformat()
        for doc in split_documents:
            doc.metadata['processed_at'] = current_time
            doc.metadata['pipeline_version'] = '1.0'
        
        return split_documents, embeddings
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """获取管道信息"""
        return {
            'loader_type': self.loader_type,
            'splitter_type': self.splitter_type,
            'embedding_model': self.embedding_model,
            'enable_cache': self.enable_cache,
            'chunk_size': settings.CHUNK_SIZE,
            'chunk_overlap': settings.CHUNK_OVERLAP,
            'embedding_dimension': self.embedding_manager.get_embedding_dimension() if self.embedding_manager else None,
        }


class BatchProcessor:
    """批处理器 - 用于大规模文档处理"""
    
    def __init__(
        self,
        pipeline: DocumentProcessingPipeline,
        max_concurrent: int = 5,
        progress_callback: Optional[callable] = None
    ):
        self.pipeline = pipeline
        self.max_concurrent = max_concurrent
        self.progress_callback = progress_callback
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        logger.info(f"批处理器初始化，最大并发: {max_concurrent}")
    
    async def process_directory(
        self,
        directory: Union[str, Path],
        recursive: bool = True,
        file_patterns: List[str] = None
    ) -> Tuple[List[Document], List[List[float]]]:
        """
        处理目录中的所有文件
        
        Args:
            directory: 目录路径
            recursive: 是否递归处理子目录
            file_patterns: 文件模式列表，如 ['*.pdf', '*.docx']
            
        Returns:
            (文档列表, 嵌入向量列表)
        """
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"目录不存在: {directory}")
        
        # 收集文件
        file_paths = self._collect_files(directory, recursive, file_patterns)
        logger.info(f"在目录 {directory} 中找到 {len(file_paths)} 个文件")
        
        if not file_paths:
            logger.warning("没有找到任何文件")
            return [], []
        
        # 处理文件
        return await self._process_files_concurrent(file_paths)
    
    def _collect_files(
        self,
        directory: Path,
        recursive: bool,
        file_patterns: List[str] = None
    ) -> List[Path]:
        """收集文件"""
        file_patterns = file_patterns or ['*.pdf', '*.docx', '*.pptx', '*.txt', '*.md']
        files = []
        
        for pattern in file_patterns:
            if recursive:
                files.extend(directory.rglob(pattern))
            else:
                files.extend(directory.glob(pattern))
        
        # 去重并排序
        files = sorted(set(files))
        return files
    
    async def _process_files_concurrent(
        self,
        file_paths: List[Path]
    ) -> Tuple[List[Document], List[List[float]]]:
        """并发处理文件"""
        
        async def process_single_file(file_path: Path):
            async with self.semaphore:
                try:
                    result = await self.pipeline.process_single_file(file_path)
                    if self.progress_callback:
                        await self.progress_callback(file_path, True, None)
                    return result
                except Exception as e:
                    logger.error(f"处理文件失败 {file_path}: {e}")
                    if self.progress_callback:
                        await self.progress_callback(file_path, False, str(e))
                    return [], []
        
        # 并发处理所有文件
        logger.info(f"开始并发处理 {len(file_paths)} 个文件")
        
        with Timer() as timer:
            tasks = [process_single_file(file_path) for file_path in file_paths]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 合并结果
        all_documents = []
        all_embeddings = []
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"任务执行异常: {result}")
                continue
            
            documents, embeddings = result
            all_documents.extend(documents)
            all_embeddings.extend(embeddings)
        
        logger.info(f"并发处理完成，耗时: {timer.elapsed:.2f}秒")
        logger.info(f"共生成 {len(all_documents)} 个文档块")
        
        return all_documents, all_embeddings
