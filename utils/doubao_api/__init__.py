"""
豆包API模块
提供豆包API的封装
"""

from .doubao import (
    DoubaoAPI,
    call_doubao
)

__all__ = [
    'DoubaoAPI',
    'call_doubao'
]

__version__ = "1.0.0" 