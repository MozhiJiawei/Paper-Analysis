"""
PDF提取器模块
提供提取PDF中的关键信息的功能
"""

from .pdf_extractor import (
    extract_paper_abstract,
    extract_paper_abstract_from_paper_copilot,
    abstract_parser,
    full_parser
)

__all__ = [
    'extract_paper_abstract',
    'extract_paper_abstract_from_paper_copilot',
    'abstract_parser',
    'full_parser'
]

__version__ = "1.0.0" 