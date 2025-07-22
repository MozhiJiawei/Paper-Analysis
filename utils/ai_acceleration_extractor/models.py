"""
AI推理加速提取器数据模型
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Author:
    """作者信息"""
    name: str
    affiliation: List[str] = field(default_factory=list)


@dataclass
class KeywordMatch:
    """关键词匹配信息"""
    keyword: str
    category: str
    priority: str
    weight: int
    in_title: bool
    in_abstract: bool


@dataclass
class MatchResult:
    """匹配结果"""
    matched_keywords: List[KeywordMatch] = field(default_factory=list)
    keyword_count: int = 0
    title_keywords: List[str] = field(default_factory=list)
    abstract_keywords: List[str] = field(default_factory=list)
    core_match_count: int = 0
    high_match_count: int = 0
    medium_match_count: int = 0
    supporting_match_count: int = 0
    is_match: bool = False
    llm_summary: str = ""
    llm_relevance: str = ""
    chinese_abstract: str = ""


@dataclass
class PaperInfo:
    """论文信息"""
    title: str
    abstract: str
    filename: str
    authors: List[Author] = field(default_factory=list)
    match_info: Optional[MatchResult] = None
    
    def __post_init__(self):
        if self.match_info is None:
            self.match_info = MatchResult()


@dataclass
class AnalysisResult:
    """分析结果"""
    ai_related_papers: List[PaperInfo] = field(default_factory=list)
    non_ai_related_papers: List[PaperInfo] = field(default_factory=list)
    processed_count: int = 0
    error_count: int = 0
    total_count: int = 0
    analysis_time: datetime = field(default_factory=datetime.now)
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_count == 0:
            return 0.0
        return self.processed_count / self.total_count
    
    @property
    def ai_related_count(self) -> int:
        """AI相关论文数量"""
        return len(self.ai_related_papers)
    
    @property
    def non_ai_related_count(self) -> int:
        """非AI相关论文数量"""
        return len(self.non_ai_related_papers)


@dataclass
class AnalysisConfig:
    """分析配置"""
    enable_llm_judge: bool = True
    analysis_mode: str = "pdf"
    papers_dir: Optional[str] = None
    output_dir: str = "."
    match_threshold: int = 5
    output_format: str = "both"
    
    def validate(self):
        """验证配置"""
        if self.analysis_mode not in ["pdf", "paper_copilot"]:
            raise ValueError(f"不支持的分析模式: {self.analysis_mode}")
        
        if self.output_format not in ["txt", "csv", "both"]:
            raise ValueError(f"不支持的输出格式: {self.output_format}")
        
        if self.analysis_mode == "pdf" and not self.papers_dir:
            raise ValueError("PDF模式下必须指定papers_dir")


@dataclass
class ProgressInfo:
    """进度信息"""
    current: int
    total: int
    success_count: int
    error_count: int
    ai_related_count: int
    non_ai_related_count: int
    start_time: datetime
    last_update: datetime = field(default_factory=datetime.now)
    
    @property
    def progress_percentage(self) -> float:
        """进度百分比"""
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100
    
    @property
    def elapsed_time(self) -> float:
        """已用时间（秒）"""
        return (self.last_update - self.start_time).total_seconds()
    
    @property
    def estimated_remaining_time(self) -> float:
        """预计剩余时间（秒）"""
        if self.current == 0:
            return 0.0
        avg_time_per_item = self.elapsed_time / self.current
        remaining_items = self.total - self.current
        return avg_time_per_item * remaining_items 