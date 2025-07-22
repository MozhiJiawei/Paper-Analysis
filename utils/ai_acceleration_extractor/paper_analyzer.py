"""
论文分析器模块
"""

import os
from typing import Dict, Optional, List
from .models import PaperInfo, Author, AnalysisConfig
from .keyword_matcher import KeywordMatcher
from .llm_judge import LLMJudge
from .exceptions import PaperExtractionError, LLMJudgmentError
from utils.pdf_extractor import extract_paper_abstract


class PaperAnalyzer:
    """论文分析器，处理单篇论文的分析逻辑"""
    
    def __init__(self, config: AnalysisConfig):
        """
        初始化论文分析器
        
        Args:
            config: 分析配置
        """
        self.config = config
        self.keyword_matcher = KeywordMatcher(config.match_threshold)
        self.llm_judge = LLMJudge() if config.enable_llm_judge else None
    
    def analyze_paper(self, paper_input: str) -> Optional[PaperInfo]:
        """
        分析单篇论文，包含错误恢复机制
        
        Args:
            paper_input: 根据analysis_mode不同，可以是PDF文件路径或paper_copilot的论文信息字典
            
        Returns:
            论文信息，如果分析失败则返回None
        """
        try:
            # 根据分析模式提取论文信息
            if self.config.analysis_mode == "pdf":
                filename = os.path.basename(paper_input)
                pdf_path = os.path.join(self.config.papers_dir, paper_input)
                paper_info = self._extract_from_pdf(pdf_path)
            else:  # paper_copilot模式
                filename = paper_input.get('title', 'unknown')
                paper_info = self._extract_from_paper_copilot(paper_input)
            
            if not paper_info or not paper_info.get('title') or not paper_info.get('abstract'):
                return None
            
            # 关键词匹配阶段
            try:
                match_result = self.keyword_matcher.match_keywords(
                    paper_info['title'], paper_info['abstract']
                )
            except Exception as e:
                # 使用默认匹配结果
                match_result = self._create_default_match_result()
            
            # 大模型判别阶段（可选且允许失败）
            if match_result.is_match and self.config.enable_llm_judge and self.llm_judge:
                try:
                    llm_result = self.llm_judge.process_paper_judgment(
                        paper_info['title'], paper_info['abstract']
                    )
                    match_result.llm_summary = llm_result.get('summary', '')
                    match_result.llm_relevance = llm_result.get('relevance', '')
                    match_result.chinese_abstract = llm_result.get('translation', '')
                except Exception as e:
                    # LLM判别失败不影响整体分析
                    match_result.llm_summary = "大模型总结生成失败"
                    match_result.llm_relevance = "大模型相关性判断失败"
                    match_result.chinese_abstract = "摘要翻译失败"
            
            # 构建PaperInfo对象
            authors = self._convert_authors(paper_info.get('authors', []))
            
            return PaperInfo(
                title=paper_info['title'],
                abstract=paper_info['abstract'],
                filename=filename,
                authors=authors,
                match_info=match_result
            )
            
        except Exception as e:
            raise PaperExtractionError(f"分析论文时发生错误: {str(e)}") from e
    
    def _extract_from_pdf(self, pdf_path: str) -> Optional[Dict]:
        """从PDF文件提取论文信息"""
        try:
            return extract_paper_abstract(pdf_path)
        except Exception as e:
            raise PaperExtractionError(f"PDF提取失败: {str(e)}") from e
    
    def _extract_from_paper_copilot(self, paper_info: Dict) -> Optional[Dict]:
        """从paper_copilot数据提取论文信息"""
        try:
            from utils.pdf_extractor import extract_paper_abstract_from_paper_copilot
            return extract_paper_abstract_from_paper_copilot(paper_info)
        except Exception as e:
            raise PaperExtractionError(f"paper_copilot数据提取失败: {str(e)}") from e
    
    def _create_default_match_result(self):
        """创建默认的匹配结果"""
        from .models import MatchResult
        return MatchResult()
    
    def _convert_authors(self, authors_data: List[Dict]) -> List[Author]:
        """转换作者数据格式"""
        authors = []
        for author_data in authors_data:
            if isinstance(author_data, dict):
                name = author_data.get('name', '')
                affiliation = author_data.get('affiliation', [])
                if isinstance(affiliation, str):
                    affiliation = [affiliation]
                authors.append(Author(name=name, affiliation=affiliation))
            elif isinstance(author_data, str):
                authors.append(Author(name=author_data))
        return authors
    
    def analyze_papers_batch(self, papers_to_analyze: List, 
                           progress_callback=None) -> List[PaperInfo]:
        """
        批量分析论文
        
        Args:
            papers_to_analyze: 要分析的论文列表
            progress_callback: 进度回调函数
            
        Returns:
            分析结果列表
        """
        results = []
        
        for i, paper_input in enumerate(papers_to_analyze):
            try:
                paper_info = self.analyze_paper(paper_input)
                if paper_info:
                    results.append(paper_info)
                
                # 调用进度回调
                if progress_callback:
                    progress_callback(i + 1, len(papers_to_analyze), paper_info)
                    
            except Exception as e:
                # 记录错误但继续处理
                if progress_callback:
                    progress_callback(i + 1, len(papers_to_analyze), None, str(e))
        
        return results 