"""
图像文档解析器
支持JPG、PNG、TIFF等图像格式的OCR和描述生成
"""
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import base64
from io import BytesIO

from PIL import Image
from paddleocr import PaddleOCR
import numpy as np

from src.data.parsers.base import BaseParser, ParseResult, ParsedElement
from src.data.document_validator import DocumentInfo, DocumentType
from src.utils.logger import get_logger
from config.settings import settings

logger = get_logger("data.parsers.image")


class ImageOCRParser(BaseParser):
    """图像OCR解析器"""
    
    def __init__(self):
        super().__init__("ImageOCRParser")
        self.supported_types = [
            DocumentType.JPG, DocumentType.JPEG, 
            DocumentType.PNG, DocumentType.TIFF, DocumentType.TIF
        ]
        self.ocr = None
        self._init_ocr()
    
    def _init_ocr(self):
        """初始化OCR引擎"""
        try:
            if settings.ENABLE_OCR:
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang=settings.OCR_LANGUAGE,
                    show_log=False
                )
                logger.info("图像OCR引擎初始化完成")
            else:
                logger.info("OCR功能已禁用")
        except Exception as e:
            logger.error(f"图像OCR引擎初始化失败: {e}")
            self.ocr = None
    
    async def parse(self, file_path: Path, doc_info: DocumentInfo) -> ParseResult:
        """解析图像文件"""
        if not self.ocr:
            return ParseResult(
                success=False,
                elements=[],
                metadata={},
                error_message="OCR引擎未初始化"
            )
        
        start_time = time.time()
        
        try:
            # 读取图像
            image = Image.open(str(file_path))
            
            # 转换为RGB格式（如果需要）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 转换为numpy数组
            img_array = np.array(image)
            
            # 使用OCR识别文本
            ocr_result = await asyncio.to_thread(self.ocr.ocr, img_array)
            
            elements = []
            
            if ocr_result and ocr_result[0]:
                # 提取文本
                texts = []
                total_confidence = 0
                
                for line in ocr_result[0]:
                    text = line[1][0]
                    confidence = line[1][1]
                    bbox = line[0]  # 边界框坐标
                    
                    texts.append(text)
                    total_confidence += confidence
                
                if texts:
                    combined_text = "\n".join(texts)
                    avg_confidence = total_confidence / len(ocr_result[0])
                    
                    text_element = self._create_text_element(
                        content=combined_text,
                        metadata={
                            'image_width': image.width,
                            'image_height': image.height,
                            'image_mode': image.mode,
                            'ocr_confidence': avg_confidence,
                            'text_lines': len(texts),
                            'extraction_method': 'paddleocr'
                        }
                    )
                    text_element.confidence = avg_confidence
                    elements.append(text_element)
            
            # 添加图像描述元素
            image_description = f"图像文件: {file_path.name}, 尺寸: {image.width}x{image.height}, 格式: {image.format}"
            
            image_element = self._create_image_element(
                content=image_description,
                metadata={
                    'image_width': image.width,
                    'image_height': image.height,
                    'image_format': image.format,
                    'image_mode': image.mode,
                    'file_size': doc_info.file_size,
                    'extraction_method': 'pil'
                }
            )
            elements.append(image_element)
            
            processing_time = time.time() - start_time
            
            return ParseResult(
                success=True,
                elements=elements,
                metadata={
                    'image_width': image.width,
                    'image_height': image.height,
                    'image_format': image.format,
                    'parser': self.name,
                    'extraction_method': 'paddleocr_pil'
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return self._handle_error(e, file_path)


class ImageDescriptionParser(BaseParser):
    """图像描述生成解析器（使用多模态模型）"""
    
    def __init__(self):
        super().__init__("ImageDescriptionParser")
        self.supported_types = [
            DocumentType.JPG, DocumentType.JPEG, 
            DocumentType.PNG, DocumentType.TIFF, DocumentType.TIF
        ]
        self.multimodal_model = None
        self._init_model()
    
    def _init_model(self):
        """初始化多模态模型"""
        try:
            if settings.ENABLE_IMAGE_DESCRIPTION:
                # 这里应该初始化Qwen3-8B多模态模型
                # 由于模型较大，这里先用占位符
                logger.info("多模态模型初始化完成（占位符）")
                self.multimodal_model = "qwen3_placeholder"
            else:
                logger.info("图像描述功能已禁用")
        except Exception as e:
            logger.error(f"多模态模型初始化失败: {e}")
            self.multimodal_model = None
    
    async def parse(self, file_path: Path, doc_info: DocumentInfo) -> ParseResult:
        """生成图像描述"""
        if not self.multimodal_model:
            return ParseResult(
                success=False,
                elements=[],
                metadata={},
                error_message="多模态模型未初始化"
            )
        
        start_time = time.time()
        
        try:
            # 读取图像
            image = Image.open(str(file_path))
            
            # 转换为RGB格式（如果需要）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 生成图像描述（这里使用占位符逻辑）
            description = await self._generate_image_description(image, file_path)
            
            # 创建描述元素
            description_element = self._create_image_element(
                content=description,
                metadata={
                    'image_width': image.width,
                    'image_height': image.height,
                    'image_format': image.format,
                    'image_mode': image.mode,
                    'description_type': 'ai_generated',
                    'extraction_method': 'qwen3_multimodal'
                }
            )
            
            processing_time = time.time() - start_time
            
            return ParseResult(
                success=True,
                elements=[description_element],
                metadata={
                    'image_width': image.width,
                    'image_height': image.height,
                    'image_format': image.format,
                    'parser': self.name,
                    'extraction_method': 'qwen3_multimodal'
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return self._handle_error(e, file_path)
    
    async def _generate_image_description(self, image: Image.Image, file_path: Path) -> str:
        """生成图像描述"""
        # 这里应该调用Qwen3-8B多模态模型
        # 目前使用占位符逻辑
        
        # 分析图像基本特征
        width, height = image.size
        aspect_ratio = width / height
        
        # 简单的描述生成逻辑（占位符）
        if aspect_ratio > 1.5:
            orientation = "横向"
        elif aspect_ratio < 0.67:
            orientation = "纵向"
        else:
            orientation = "方形"
        
        # 分析颜色
        colors = image.getcolors(maxcolors=256)
        if colors:
            dominant_color = max(colors, key=lambda x: x[0])
            color_info = f"主要颜色分布包含 {len(colors)} 种颜色"
        else:
            color_info = "颜色信息复杂"
        
        description = f"""
这是一张{orientation}的图像，尺寸为{width}x{height}像素。
{color_info}。
文件名：{file_path.name}
图像格式：{image.format}

[注意：这是占位符描述，实际应用中会使用Qwen3-8B多模态模型生成更详细的图像描述]
        """.strip()
        
        return description


class MultiModalImageParser(BaseParser):
    """多模态图像解析器（结合OCR和描述生成）"""
    
    def __init__(self):
        super().__init__("MultiModalImageParser")
        self.supported_types = [
            DocumentType.JPG, DocumentType.JPEG, 
            DocumentType.PNG, DocumentType.TIFF, DocumentType.TIF
        ]
        
        # 初始化子解析器
        self.ocr_parser = ImageOCRParser()
        self.description_parser = ImageDescriptionParser()
    
    async def parse(self, file_path: Path, doc_info: DocumentInfo) -> ParseResult:
        """多模态解析图像"""
        start_time = time.time()
        
        try:
            elements = []
            
            # 1. OCR文本提取
            if self.ocr_parser.ocr:
                ocr_result = await self.ocr_parser.parse(file_path, doc_info)
                if ocr_result.success:
                    elements.extend(ocr_result.elements)
            
            # 2. 图像描述生成
            if self.description_parser.multimodal_model:
                desc_result = await self.description_parser.parse(file_path, doc_info)
                if desc_result.success:
                    elements.extend(desc_result.elements)
            
            # 3. 如果两个解析器都失败，至少提供基本信息
            if not elements:
                image = Image.open(str(file_path))
                basic_info = f"图像文件: {file_path.name}, 尺寸: {image.width}x{image.height}, 格式: {image.format}"
                
                basic_element = self._create_image_element(
                    content=basic_info,
                    metadata={
                        'image_width': image.width,
                        'image_height': image.height,
                        'image_format': image.format,
                        'extraction_method': 'basic_info'
                    }
                )
                elements.append(basic_element)
            
            processing_time = time.time() - start_time
            
            return ParseResult(
                success=True,
                elements=elements,
                metadata={
                    'parser': self.name,
                    'extraction_method': 'multimodal',
                    'ocr_enabled': self.ocr_parser.ocr is not None,
                    'description_enabled': self.description_parser.multimodal_model is not None
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            return self._handle_error(e, file_path)
