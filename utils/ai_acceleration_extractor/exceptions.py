"""
AI推理加速提取器异常定义
"""


class AIAccelerationExtractorError(Exception):
    """AI推理加速提取器基础异常类"""
    pass


class PaperExtractionError(AIAccelerationExtractorError):
    """论文提取错误"""
    pass


class KeywordMatchingError(AIAccelerationExtractorError):
    """关键词匹配错误"""
    pass


class LLMJudgmentError(AIAccelerationExtractorError):
    """大模型判别错误"""
    pass


class FileOperationError(AIAccelerationExtractorError):
    """文件操作错误"""
    pass


class ConfigurationError(AIAccelerationExtractorError):
    """配置错误"""
    pass


class ValidationError(AIAccelerationExtractorError):
    """数据验证错误"""
    pass 