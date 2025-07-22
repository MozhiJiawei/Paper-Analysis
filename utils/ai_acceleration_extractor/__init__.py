"""
AI推理加速论文提取器模块

这个模块提供了从论文中提取和筛选AI推理加速相关论文的功能。
支持PDF文件和paper_copilot数据两种输入模式。
"""

from .ai_acceleration_extractor import (
    AiAccelerationExtractor,
    ai_acceleration_parse,
    ai_acceleration_parse_paper_copilot
)
from .models import (
    PaperInfo,
    AnalysisResult,
    AnalysisConfig,
    ProgressInfo,
    Author,
    KeywordMatch,
    MatchResult
)
from .exceptions import (
    AIAccelerationExtractorError,
    PaperExtractionError,
    KeywordMatchingError,
    LLMJudgmentError,
    FileOperationError,
    ConfigurationError,
    ValidationError
)
from .keyword_matcher import KeywordMatcher
from .llm_judge import LLMJudge
from .paper_analyzer import PaperAnalyzer
from .report_generator import ReportGenerator

__version__ = "2.0.0"
__author__ = "AI Research Team"

__all__ = [
    # 主类
    "AiAccelerationExtractor",
    
    # 便捷函数
    "ai_acceleration_parse",
    "ai_acceleration_parse_paper_copilot",
    
    # 数据模型
    "PaperInfo",
    "AnalysisResult", 
    "AnalysisConfig",
    "ProgressInfo",
    "Author",
    "KeywordMatch",
    "MatchResult",
    
    # 异常类
    "AIAccelerationExtractorError",
    "PaperExtractionError",
    "KeywordMatchingError", 
    "LLMJudgmentError",
    "FileOperationError",
    "ConfigurationError",
    "ValidationError",
    
    # 组件类
    "KeywordMatcher",
    "LLMJudge",
    "PaperAnalyzer", 
    "ReportGenerator"
] 