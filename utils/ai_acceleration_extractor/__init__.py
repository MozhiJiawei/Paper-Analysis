"""
AI加速论文提取器模块
提供分析AI推理加速相关论文的功能
"""

from .ai_acceleration_extractor import (
    AiAccelerationExtractor,
    ai_acceleration_parse
)

__all__ = [
    'AiAccelerationExtractor',
    'ai_acceleration_parse'
]

__version__ = "1.0.0" 