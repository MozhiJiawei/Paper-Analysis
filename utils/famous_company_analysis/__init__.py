"""
知名公司分析模块

这个模块提供了判断组织是否为知名AI公司的功能，
整合了项目中AAAI-2025和ICLR-2025的知名公司映射表。

主要功能:
- is_famous_company: 判断组织列表是否包含知名公司
- FamousCompanyAnalyzer: 知名公司分析器类
"""

from .famous_company import is_famous_company, FamousCompanyAnalyzer

__all__ = ['is_famous_company', 'FamousCompanyAnalyzer']
__version__ = '1.0.0'
