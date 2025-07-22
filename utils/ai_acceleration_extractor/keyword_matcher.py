"""
关键词匹配器模块
"""

import re
from typing import Dict, List, Tuple, Optional, Any
from .config import (
    CORE_KEYWORDS, HIGH_RELEVANCE_KEYWORDS, MEDIUM_RELEVANCE_KEYWORDS, 
    SUPPORTING_KEYWORDS, KEYWORD_WEIGHTS, MATCH_THRESHOLD
)
from .models import KeywordMatch, MatchResult
from .exceptions import KeywordMatchingError


class KeywordMatcher:
    """关键词匹配器"""
    
    def __init__(self, threshold: int = None):
        """
        初始化关键词匹配器
        
        Args:
            threshold: 匹配阈值，默认使用配置文件中的值
        """
        self.threshold = threshold or MATCH_THRESHOLD
        self._keyword_categories = {
            "core": CORE_KEYWORDS,
            "high": HIGH_RELEVANCE_KEYWORDS,
            "medium": MEDIUM_RELEVANCE_KEYWORDS,
            "supporting": SUPPORTING_KEYWORDS
        }
    
    def match_keywords(self, title: str, abstract: str) -> MatchResult:
        """
        执行关键词匹配
        
        Args:
            title: 论文标题
            abstract: 论文摘要
            
        Returns:
            匹配结果
        """
        try:
            # 安全处理输入参数
            title = title or ""
            abstract = abstract or ""
            title_lower = title.lower()
            abstract_lower = abstract.lower()
            
            # 执行匹配
            match_result = MatchResult()
            
            # 匹配各类别关键词
            for priority, keywords in self._keyword_categories.items():
                weight = KEYWORD_WEIGHTS[priority]
                matched_keywords, category_count = self._match_keyword_category(
                    title_lower, abstract_lower, keywords, priority, weight
                )
                match_result.matched_keywords.extend(matched_keywords)
                
                # 更新分类计数
                if priority == "core":
                    match_result.core_match_count = category_count
                elif priority == "high":
                    match_result.high_match_count = category_count
                elif priority == "medium":
                    match_result.medium_match_count = category_count
                elif priority == "supporting":
                    match_result.supporting_match_count = category_count
            
            # 计算总分和分类信息
            title_keywords = []
            abstract_keywords = []
            
            for kw_match in match_result.matched_keywords:
                match_result.keyword_count += kw_match.weight
                
                if kw_match.in_title:
                    title_keywords.append(kw_match.keyword)
                    # 标题关键词额外加权
                    if kw_match.priority == "core":
                        match_result.keyword_count += 3
                    elif kw_match.priority == "high":
                        match_result.keyword_count += 2
                    elif kw_match.priority == "medium":
                        match_result.keyword_count += 1
                
                if kw_match.in_abstract:
                    abstract_keywords.append(kw_match.keyword)
            
            match_result.title_keywords = title_keywords
            match_result.abstract_keywords = abstract_keywords
            
            # 判断是否匹配
            match_result.is_match = self._should_match(match_result)
            
            return match_result
            
        except Exception as e:
            raise KeywordMatchingError(f"关键词匹配失败: {str(e)}") from e
    
    def _match_keyword_category(self, title: str, abstract: str, keywords: Dict[str, List[str]], 
                               priority: str, weight: int) -> Tuple[List[KeywordMatch], int]:
        """
        匹配特定类别的关键词
        
        Args:
            title: 标题（小写）
            abstract: 摘要（小写）
            keywords: 关键词字典
            priority: 优先级
            weight: 权重
            
        Returns:
            (匹配的关键词列表, 类别计数)
        """
        matched_keywords = []
        category_count = 0
        
        for category, category_keywords in keywords.items():
            for keyword in category_keywords:
                keyword_lower = keyword.lower()
                
                # 精确的关键词匹配
                title_match = self._precise_keyword_match(title, keyword_lower)
                abstract_match = self._precise_keyword_match(abstract, keyword_lower)
                
                if title_match or abstract_match:
                    category_count += weight
                    matched_keywords.append(KeywordMatch(
                        keyword=keyword,
                        category=category,
                        priority=priority,
                        weight=weight,
                        in_title=title_match,
                        in_abstract=abstract_match
                    ))
        
        return matched_keywords, category_count
    
    def _precise_keyword_match(self, text: str, keyword: str) -> bool:
        """
        精确的关键词匹配
        
        Args:
            text: 要匹配的文本
            keyword: 关键词
            
        Returns:
            是否匹配
        """
        # 对于单个词的关键词，使用词边界匹配
        if ' ' not in keyword:
            pattern = r'\b' + re.escape(keyword) + r'\b'
            return bool(re.search(pattern, text))
        else:
            # 对于短语，检查完整短语匹配
            return (f" {keyword} " in f" {text} " or 
                   text.startswith(f"{keyword} ") or 
                   text.endswith(f" {keyword}") or
                   text == keyword)
    
    def _should_match(self, match_result: MatchResult) -> bool:
        """
        判断是否应该匹配
        
        Args:
            match_result: 匹配结果
            
        Returns:
            是否匹配
        """
        return match_result.keyword_count >= self.threshold
    
    def get_match_statistics(self, match_result: MatchResult) -> Dict[str, Any]:
        """
        获取匹配统计信息
        
        Args:
            match_result: 匹配结果
            
        Returns:
            统计信息字典
        """
        category_stats = {}
        keyword_stats = {}
        priority_stats = {}
        
        for kw_match in match_result.matched_keywords:
            category = kw_match.category
            keyword = kw_match.keyword
            priority = kw_match.priority
            
            category_stats[category] = category_stats.get(category, 0) + 1
            keyword_stats[keyword] = keyword_stats.get(keyword, 0) + 1
            priority_stats[priority] = priority_stats.get(priority, 0) + 1
        
        return {
            "category_stats": category_stats,
            "keyword_stats": keyword_stats,
            "priority_stats": priority_stats,
            "total_matches": len(match_result.matched_keywords),
            "total_score": match_result.keyword_count,
            "is_match": match_result.is_match
        } 