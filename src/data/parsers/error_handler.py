"""
文档解析错误处理模块
提供强大的错误处理和容错机制
"""
import traceback
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("data.parsers.error_handler")


class ErrorSeverity(Enum):
    """错误严重程度"""
    LOW = "low"          # 轻微错误，不影响主要功能
    MEDIUM = "medium"    # 中等错误，影响部分功能
    HIGH = "high"        # 严重错误，影响主要功能
    CRITICAL = "critical" # 致命错误，系统无法继续


@dataclass
class ParseError:
    """解析错误信息"""
    error_id: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    timestamp: str
    file_path: Optional[str] = None
    parser_name: Optional[str] = None
    element_type: Optional[str] = None
    stack_trace: Optional[str] = None
    recovery_action: Optional[str] = None


class ErrorHandler:
    """错误处理器"""
    
    def __init__(self):
        self.name = "ErrorHandler"
        self.error_log: List[ParseError] = []
        self.error_count = 0
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # 注册默认恢复策略
        self._register_default_recovery_strategies()
        
        logger.info("错误处理器初始化完成")
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: ErrorSeverity = ErrorSeverity.MEDIUM
    ) -> Optional[Any]:
        """
        处理错误
        
        Args:
            error: 异常对象
            context: 错误上下文
            severity: 错误严重程度
            
        Returns:
            恢复结果（如果有）
        """
        self.error_count += 1
        
        # 创建错误记录
        parse_error = ParseError(
            error_id=f"error_{self.error_count}",
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            timestamp=datetime.utcnow().isoformat(),
            file_path=context.get('file_path'),
            parser_name=context.get('parser_name'),
            element_type=context.get('element_type'),
            stack_trace=traceback.format_exc(),
            recovery_action=None
        )
        
        # 记录错误
        self.error_log.append(parse_error)
        
        # 根据严重程度决定处理方式
        if severity == ErrorSeverity.CRITICAL:
            logger.critical(f"致命错误: {parse_error.error_message}")
            raise error
        elif severity == ErrorSeverity.HIGH:
            logger.error(f"严重错误: {parse_error.error_message}")
        elif severity == ErrorSeverity.MEDIUM:
            logger.warning(f"中等错误: {parse_error.error_message}")
        else:
            logger.info(f"轻微错误: {parse_error.error_message}")
        
        # 尝试恢复
        recovery_result = self._attempt_recovery(parse_error, context)
        
        return recovery_result
    
    def _attempt_recovery(self, parse_error: ParseError, context: Dict[str, Any]) -> Optional[Any]:
        """尝试错误恢复"""
        error_type = parse_error.error_type
        
        # 查找恢复策略
        recovery_strategy = self.recovery_strategies.get(error_type)
        
        if recovery_strategy:
            try:
                result = recovery_strategy(parse_error, context)
                parse_error.recovery_action = f"使用策略 {recovery_strategy.__name__} 恢复"
                logger.info(f"错误恢复成功: {parse_error.error_id}")
                return result
            except Exception as recovery_error:
                logger.error(f"错误恢复失败: {recovery_error}")
                parse_error.recovery_action = f"恢复失败: {str(recovery_error)}"
        
        return None
    
    def _register_default_recovery_strategies(self):
        """注册默认恢复策略"""
        
        def handle_file_not_found(parse_error: ParseError, context: Dict[str, Any]):
            """处理文件未找到错误"""
            logger.info("尝试查找备用文件路径...")
            return None
        
        def handle_encoding_error(parse_error: ParseError, context: Dict[str, Any]):
            """处理编码错误"""
            logger.info("尝试使用其他编码格式...")
            file_path = context.get('file_path')
            if file_path:
                # 尝试不同的编码
                encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
                for encoding in encodings:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        logger.info(f"使用编码 {encoding} 成功读取文件")
                        return content
                    except:
                        continue
            return None
        
        def handle_memory_error(parse_error: ParseError, context: Dict[str, Any]):
            """处理内存错误"""
            logger.info("尝试分块处理大文件...")
            return None
        
        def handle_timeout_error(parse_error: ParseError, context: Dict[str, Any]):
            """处理超时错误"""
            logger.info("尝试简化处理流程...")
            return None
        
        def handle_ocr_error(parse_error: ParseError, context: Dict[str, Any]):
            """处理OCR错误"""
            logger.info("尝试降低OCR精度或跳过OCR...")
            return None
        
        # 注册策略
        self.recovery_strategies.update({
            'FileNotFoundError': handle_file_not_found,
            'UnicodeDecodeError': handle_encoding_error,
            'MemoryError': handle_memory_error,
            'TimeoutError': handle_timeout_error,
            'OCRError': handle_ocr_error,
        })
    
    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """注册自定义恢复策略"""
        self.recovery_strategies[error_type] = strategy
        logger.info(f"注册恢复策略: {error_type} -> {strategy.__name__}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """获取错误摘要"""
        if not self.error_log:
            return {
                'total_errors': 0,
                'error_types': {},
                'severity_distribution': {},
                'recovery_success_rate': 0.0
            }
        
        # 统计错误类型
        error_types = {}
        severity_distribution = {}
        recovered_count = 0
        
        for error in self.error_log:
            # 错误类型统计
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
            
            # 严重程度统计
            severity = error.severity.value
            severity_distribution[severity] = severity_distribution.get(severity, 0) + 1
            
            # 恢复成功统计
            if error.recovery_action and "恢复成功" in error.recovery_action:
                recovered_count += 1
        
        recovery_rate = recovered_count / len(self.error_log) if self.error_log else 0.0
        
        return {
            'total_errors': len(self.error_log),
            'error_types': error_types,
            'severity_distribution': severity_distribution,
            'recovery_success_rate': recovery_rate,
            'recent_errors': [
                {
                    'error_id': e.error_id,
                    'error_type': e.error_type,
                    'message': e.error_message,
                    'severity': e.severity.value,
                    'timestamp': e.timestamp
                }
                for e in self.error_log[-10:]  # 最近10个错误
            ]
        }
    
    def clear_error_log(self):
        """清空错误日志"""
        self.error_log.clear()
        self.error_count = 0
        logger.info("错误日志已清空")


class SafeParserWrapper:
    """安全解析器包装器"""
    
    def __init__(self, parser, error_handler: ErrorHandler):
        self.parser = parser
        self.error_handler = error_handler
        self.name = f"Safe_{parser.name}"
    
    async def parse(self, file_path, doc_info):
        """安全解析方法"""
        context = {
            'file_path': str(file_path),
            'parser_name': self.parser.name,
            'doc_type': doc_info.file_type.value if doc_info.file_type else 'unknown'
        }
        
        try:
            result = await self.parser.parse(file_path, doc_info)
            return result
            
        except FileNotFoundError as e:
            return self.error_handler.handle_error(e, context, ErrorSeverity.HIGH)
            
        except MemoryError as e:
            return self.error_handler.handle_error(e, context, ErrorSeverity.HIGH)
            
        except TimeoutError as e:
            return self.error_handler.handle_error(e, context, ErrorSeverity.MEDIUM)
            
        except UnicodeDecodeError as e:
            return self.error_handler.handle_error(e, context, ErrorSeverity.MEDIUM)
            
        except Exception as e:
            # 未知错误，根据错误消息判断严重程度
            if any(keyword in str(e).lower() for keyword in ['critical', 'fatal', 'corrupt']):
                severity = ErrorSeverity.HIGH
            else:
                severity = ErrorSeverity.MEDIUM
            
            return self.error_handler.handle_error(e, context, severity)


# 全局错误处理器实例
global_error_handler = ErrorHandler()
