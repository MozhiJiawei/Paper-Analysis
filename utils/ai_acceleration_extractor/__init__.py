"""
AI加速论文提取器模块
提供分析AI推理加速相关论文的功能
"""

from .ai_acceleration_extractor import (
    AiAccelerationExtractor,
    is_ai_acceleration_paper,
    abstract_parser
)

__all__ = [
    'AiAccelerationExtractor',
    'is_ai_acceleration_paper',
    'abstract_parser'
]

__version__ = "1.0.0" 