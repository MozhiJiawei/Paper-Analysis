"""
报告生成器模块
"""

import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from .models import PaperInfo, AnalysisResult, ProgressInfo
from .exceptions import FileOperationError


class FileConfig:
    """文件配置常量"""
    
    # 文件编码
    TEXT_ENCODING = 'utf-8'
    CSV_ENCODING = 'utf-8-sig'
    
    # 文件名模板
    AI_PAPERS_TXT = "ai_inference_related_papers.txt"
    AI_PAPERS_CSV = "ai_inference_related_papers.csv"
    NON_AI_PAPERS_TXT = "non_ai_inference_papers.txt"
    NON_AI_PAPERS_CSV = "non_ai_inference_papers.csv"
    PROGRESS_FILE = "analysis_progress.txt"
    STATISTICS_TXT = "match_statistics.txt"
    STATISTICS_CSV = "match_statistics.csv"
    SUMMARY_FILE = "analysis_summary.txt"
    
    # 结果文件夹模板
    RESULT_FOLDER_TEMPLATE = "ai_acceleration_analysis_{timestamp}"
    
    # 标题和分隔符
    AI_PAPERS_TITLE = "AI推理加速相关论文列表"
    NON_AI_PAPERS_TITLE = "非AI推理加速相关论文列表"
    SEPARATOR_LINE = "=" * 50
    PAPER_SEPARATOR = "-" * 80
    
    # 摘要预览长度
    ABSTRACT_PREVIEW_LENGTH = 2000


class CSVHeaders:
    """CSV文件头配置"""
    
    AI_HEADERS = [
        '序号', '标题', '文件名', '作者', '组织', '匹配分数',
        '核心关键字数', '高相关关键字数', '中等关键字数', '支撑关键字数',
        '标题关键字', '摘要关键字数量', '大模型总结', '大模型相关性判断', 
        '中文摘要翻译', '摘要预览'
    ]
    
    NON_AI_HEADERS = [
        '序号', '标题', '文件名', '作者', '组织', '匹配分数',
        '大模型总结', '大模型相关性判断', '中文摘要翻译', '摘要预览'
    ]


class FileWriter:
    """文件写入器，负责具体的文件操作"""
    
    @staticmethod
    def write_text_file(filepath: str, content: str, encoding: str = FileConfig.TEXT_ENCODING) -> None:
        """写入文本文件"""
        try:
            with open(filepath, 'w', encoding=encoding) as f:
                f.write(content)
        except Exception as e:
            raise FileOperationError(f"写入文本文件失败 {filepath}: {str(e)}") from e
    
    @staticmethod
    def append_text_file(filepath: str, content: str, encoding: str = FileConfig.TEXT_ENCODING) -> None:
        """追加文本文件"""
        try:
            with open(filepath, 'a', encoding=encoding) as f:
                f.write(content)
        except Exception as e:
            raise FileOperationError(f"追加文本文件失败 {filepath}: {str(e)}") from e
    
    @staticmethod
    def write_csv_headers(filepath: str, headers: List[str], encoding: str = FileConfig.CSV_ENCODING) -> None:
        """写入CSV文件头"""
        try:
            with open(filepath, 'w', encoding=encoding, newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        except Exception as e:
            raise FileOperationError(f"写入CSV文件头失败 {filepath}: {str(e)}") from e
    
    @staticmethod
    def append_csv_row(filepath: str, row: List[Any], encoding: str = FileConfig.CSV_ENCODING) -> None:
        """追加CSV行"""
        try:
            with open(filepath, 'a', encoding=encoding, newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            raise FileOperationError(f"追加CSV行失败 {filepath}: {str(e)}") from e


class DataFormatter:
    """数据格式化器，负责数据格式化逻辑"""
    
    @staticmethod
    def format_authors_and_organizations(paper: PaperInfo) -> Tuple[str, str]:
        """格式化作者和组织信息"""
        authors_str = ""
        organizations_str = ""
        
        if paper.authors:
            authors = [author.name for author in paper.authors if author.name]
            authors_str = "; ".join(authors)
            
            # 收集并去重组织信息
            all_affiliations = set()
            for author in paper.authors:
                for affiliation in author.affiliation:
                    if affiliation and affiliation.strip():
                        all_affiliations.add(affiliation.strip())
            organizations_str = "; ".join(sorted(list(all_affiliations)))
        
        return authors_str, organizations_str
    
    @staticmethod
    def format_abstract_preview(abstract: Optional[str]) -> str:
        """格式化摘要预览"""
        if not abstract:
            return ""
        
        preview = abstract[:FileConfig.ABSTRACT_PREVIEW_LENGTH]
        if len(abstract) > FileConfig.ABSTRACT_PREVIEW_LENGTH:
            preview += "..."
        
        return preview.replace('\n', ' ').replace('\r', ' ')
    
    @staticmethod
    def format_chinese_abstract(chinese_abstract: Optional[str]) -> str:
        """格式化中文摘要"""
        if not chinese_abstract:
            return ""
        
        return chinese_abstract.replace('\n', ' ').replace('\r', ' ')
    
    @staticmethod
    def format_paper_info(paper: PaperInfo, show_details: bool = True) -> str:
        """格式化单篇论文信息"""
        lines = []
        lines.append(f"   文件名: {paper.filename}")
        
        # 作者和组织信息
        authors_str, organizations_str = DataFormatter.format_authors_and_organizations(paper)
        if authors_str:
            lines.append(f"   作者: {authors_str}")
        if organizations_str:
            lines.append(f"   组织: {organizations_str}")
        
        # 匹配信息
        if show_details and paper.match_info:
            match_info = paper.match_info
            lines.append(f"   匹配分数: {match_info.keyword_count}")
            
            # 关键字分布
            lines.append(f"   关键字分布 - 核心:{match_info.core_match_count}, "
                        f"高相关:{match_info.high_match_count}, "
                        f"中等:{match_info.medium_match_count}, "
                        f"支撑:{match_info.supporting_match_count}")
            
            if match_info.title_keywords:
                lines.append(f"   标题关键字: {', '.join(match_info.title_keywords)}")
                
            # 大模型判别结果
            if match_info.llm_summary:
                lines.append(f"   大模型总结: {match_info.llm_summary}")
            if match_info.llm_relevance:
                lines.append(f"   大模型相关性判断: {match_info.llm_relevance}")
            if match_info.chinese_abstract:
                lines.append(f"   中文摘要翻译: {match_info.chinese_abstract}")
        
        # 摘要
        if paper.abstract:
            abstract_preview = DataFormatter.format_abstract_preview(paper.abstract)
            lines.append(f"   摘要: {abstract_preview}")
        
        return '\n'.join(lines)


class StatisticsCollector:
    """统计收集器，负责收集和统计匹配数据"""
    
    @staticmethod
    def collect_statistics(papers: List[PaperInfo]) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        """收集统计数据"""
        category_stats = {}
        keyword_stats = {}
        priority_stats = {}
        
        for paper in papers:
            if not paper.match_info:
                continue
            
            for kw_match in paper.match_info.matched_keywords:
                category = kw_match.category
                keyword = kw_match.keyword
                priority = kw_match.priority
                
                category_stats[category] = category_stats.get(category, 0) + 1
                keyword_stats[keyword] = keyword_stats.get(keyword, 0) + 1
                priority_stats[priority] = priority_stats.get(priority, 0) + 1
        
        return category_stats, keyword_stats, priority_stats


class ReportGenerator:
    """报告生成器，负责生成各种输出报告，支持增量写入"""
    
    def __init__(self, output_dir: str):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建带时间戳的结果文件夹
        result_folder_name = FileConfig.RESULT_FOLDER_TEMPLATE.format(timestamp=self.timestamp)
        self.output_dir = os.path.join(output_dir, result_folder_name)
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 初始化增量写入文件
        self._init_incremental_files()
    
    def _init_incremental_files(self) -> None:
        """初始化增量写入的文件"""
        # 文件路径
        self.ai_papers_txt_file = os.path.join(self.output_dir, FileConfig.AI_PAPERS_TXT)
        self.ai_papers_csv_file = os.path.join(self.output_dir, FileConfig.AI_PAPERS_CSV)
        self.non_ai_papers_txt_file = os.path.join(self.output_dir, FileConfig.NON_AI_PAPERS_TXT)
        self.non_ai_papers_csv_file = os.path.join(self.output_dir, FileConfig.NON_AI_PAPERS_CSV)
        self.progress_file = os.path.join(self.output_dir, FileConfig.PROGRESS_FILE)
        
        # 初始化文本文件
        self._write_file_header(self.ai_papers_txt_file, FileConfig.AI_PAPERS_TITLE)
        self._write_file_header(self.non_ai_papers_txt_file, FileConfig.NON_AI_PAPERS_TITLE)
        
        # 初始化CSV文件头
        self._init_csv_files()
        
        # 计数器
        self.ai_paper_count = 0
        self.non_ai_paper_count = 0
        
        # 更新进度文件
        self._update_progress_file()
    
    def _write_file_header(self, filepath: str, title: str) -> None:
        """写入文件头部"""
        content = f"{title}\n{FileConfig.SEPARATOR_LINE}\n\n"
        FileWriter.write_text_file(filepath, content)
    
    def _init_csv_files(self) -> None:
        """初始化CSV文件"""
        FileWriter.write_csv_headers(self.ai_papers_csv_file, CSVHeaders.AI_HEADERS)
        FileWriter.write_csv_headers(self.non_ai_papers_csv_file, CSVHeaders.NON_AI_HEADERS)
    
    def _update_progress_file(self, error_count: int = 0) -> None:
        """更新进度文件"""
        try:
            content = (
                f"最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"AI相关论文数: {self.ai_paper_count}\n"
                f"非AI相关论文数: {self.non_ai_paper_count}\n"
                f"处理失败数: {error_count}\n"
                f"总处理数: {self.ai_paper_count + self.non_ai_paper_count}\n"
            )
            FileWriter.write_text_file(self.progress_file, content)
        except Exception:
            # 忽略进度文件更新错误
            pass
    
    def append_ai_paper(self, paper: PaperInfo) -> None:
        """增量写入单篇AI相关论文"""
        try:
            self.ai_paper_count += 1
            
            # 写入文本文件
            content = (
                f"{self.ai_paper_count}. 标题: {paper.title}\n"
                f"{DataFormatter.format_paper_info(paper)}\n"
                f"{FileConfig.PAPER_SEPARATOR}\n\n"
            )
            FileWriter.append_text_file(self.ai_papers_txt_file, content)
            
            # 写入CSV文件
            self._append_ai_paper_csv(paper)
            
        except Exception as e:
            raise FileOperationError(f"写入AI相关论文失败: {str(e)}") from e
    
    def append_non_ai_paper(self, paper: PaperInfo) -> None:
        """增量写入单篇非AI相关论文"""
        try:
            self.non_ai_paper_count += 1
            
            # 写入文本文件
            content = (
                f"{self.non_ai_paper_count}. 标题: {paper.title}\n"
                f"{DataFormatter.format_paper_info(paper, show_details=False)}\n"
            )
            
            # 添加分数信息
            if paper.match_info:
                content += f"\n   匹配分数不足 (分数: {paper.match_info.keyword_count})"
            
            content += f"\n{FileConfig.PAPER_SEPARATOR}\n\n"
            FileWriter.append_text_file(self.non_ai_papers_txt_file, content)
            
            # 写入CSV文件
            self._append_non_ai_paper_csv(paper)
            
        except Exception as e:
            raise FileOperationError(f"写入非AI相关论文失败: {str(e)}") from e
    
    def _append_ai_paper_csv(self, paper: PaperInfo) -> None:
        """增量写入AI相关论文到CSV"""
        match_info = paper.match_info
        
        # 处理作者信息
        authors_str, organizations_str = DataFormatter.format_authors_and_organizations(paper)
        
        # 处理关键字信息
        title_keywords = "; ".join(match_info.title_keywords) if match_info.title_keywords else ""
        abstract_keywords_count = len(match_info.abstract_keywords) if match_info.abstract_keywords else 0
        
        # 摘要预览
        abstract_preview = DataFormatter.format_abstract_preview(paper.abstract)
        
        # 中文摘要翻译
        chinese_abstract = DataFormatter.format_chinese_abstract(match_info.chinese_abstract)
        
        row = [
            self.ai_paper_count,
            paper.title,
            paper.filename,
            authors_str,
            organizations_str,
            match_info.keyword_count,
            match_info.core_match_count,
            match_info.high_match_count,
            match_info.medium_match_count,
            match_info.supporting_match_count,
            title_keywords,
            abstract_keywords_count,
            match_info.llm_summary or "",
            match_info.llm_relevance or "",
            chinese_abstract,
            abstract_preview
        ]
        
        FileWriter.append_csv_row(self.ai_papers_csv_file, row)
    
    def _append_non_ai_paper_csv(self, paper: PaperInfo) -> None:
        """增量写入非AI相关论文到CSV"""
        match_info = paper.match_info
        
        # 处理作者信息
        authors_str, organizations_str = DataFormatter.format_authors_and_organizations(paper)
        
        # 摘要预览
        abstract_preview = DataFormatter.format_abstract_preview(paper.abstract)
        
        # 中文摘要翻译
        chinese_abstract = DataFormatter.format_chinese_abstract(match_info.chinese_abstract)
        
        row = [
            self.non_ai_paper_count,
            paper.title,
            paper.filename,
            authors_str,
            organizations_str,
            match_info.keyword_count,
            match_info.llm_summary or "",
            match_info.llm_relevance or "",
            chinese_abstract,
            abstract_preview
        ]
        
        FileWriter.append_csv_row(self.non_ai_papers_csv_file, row)
    
    def update_progress(self, error_count: int = 0) -> None:
        """更新进度"""
        self._update_progress_file(error_count)
    
    def generate_statistics(self, papers: List[PaperInfo], output_filename: str = FileConfig.STATISTICS_TXT) -> None:
        """生成匹配统计报告"""
        if not papers:
            return
        
        output_file = os.path.join(self.output_dir, output_filename)
        
        # 收集统计数据
        category_stats, keyword_stats, priority_stats = StatisticsCollector.collect_statistics(papers)
        
        # 生成统计报告内容
        content = self._generate_statistics_content(papers, category_stats, keyword_stats, priority_stats)
        
        # 写入文件
        FileWriter.write_text_file(output_file, content)
    
    def _generate_statistics_content(self, papers: List[PaperInfo], 
                                   category_stats: Dict[str, int], 
                                   keyword_stats: Dict[str, int], 
                                   priority_stats: Dict[str, int]) -> str:
        """生成统计报告内容"""
        lines = [
            "AI推理加速相关论文匹配统计报告",
            FileConfig.SEPARATOR_LINE,
            "",
            f"总匹配论文数: {len(papers)}",
            "",
            "关键字优先级统计:",
            "-" * 30
        ]
        
        # 优先级统计
        for priority in ['core', 'high', 'medium', 'supporting']:
            if priority in priority_stats:
                lines.append(f"{priority}: {priority_stats[priority]} 次")
        lines.append("")
        
        # 类别统计
        lines.extend([
            "关键字类别统计:",
            "-" * 30
        ])
        for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"{category}: {count} 次")
        lines.append("")
        
        # 最常见关键字
        lines.extend([
            "最常匹配的关键字 (前20个):",
            "-" * 30
        ])
        for keyword, count in sorted(keyword_stats.items(), key=lambda x: x[1], reverse=True)[:20]:
            lines.append(f"{keyword}: {count} 次")
        
        return '\n'.join(lines)
    
    def finalize_reports(self, analysis_result: AnalysisResult) -> None:
        """生成最终的统计报告"""
        try:
            # 生成统计报告
            if analysis_result.ai_related_papers:
                self.generate_statistics(analysis_result.ai_related_papers, FileConfig.STATISTICS_TXT)
                self._save_statistics_csv(analysis_result.ai_related_papers, FileConfig.STATISTICS_CSV)
            
            # 生成汇总信息文件
            self._generate_summary_file(analysis_result)
            
        except Exception as e:
            raise FileOperationError(f"生成最终统计报告时出错: {str(e)}") from e
    
    def _save_statistics_csv(self, papers: List[PaperInfo], output_filename: str) -> None:
        """生成匹配统计报告CSV格式"""
        if not papers:
            return
        
        output_file = os.path.join(self.output_dir, output_filename)
        
        # 收集统计数据
        category_stats, keyword_stats, priority_stats = StatisticsCollector.collect_statistics(papers)
        
        # 生成CSV内容
        rows = [
            ['统计类别', '项目', '数量'],
            ['总体', '匹配论文总数', len(papers)],
            [],  # 空行
            ['优先级统计', '', '']
        ]
        
        # 优先级统计
        for priority in ['core', 'high', 'medium', 'supporting']:
            if priority in priority_stats:
                rows.append(['优先级', priority, priority_stats[priority]])
        rows.append([])  # 空行
        
        # 类别统计
        rows.extend([
            ['类别统计', '', '']
        ])
        for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
            rows.append(['类别', category, count])
        rows.append([])  # 空行
        
        # 最常见关键字
        rows.extend([
            ['关键字统计 (前20个)', '', '']
        ])
        for keyword, count in sorted(keyword_stats.items(), key=lambda x: x[1], reverse=True)[:20]:
            rows.append(['关键字', keyword, count])
        
        # 写入CSV文件
        try:
            with open(output_file, 'w', encoding=FileConfig.CSV_ENCODING, newline='') as f:
                writer = csv.writer(f)
                writer.writerows(rows)
        except Exception as e:
            raise FileOperationError(f"写入统计CSV文件失败 {output_file}: {str(e)}") from e
    
    def _generate_summary_file(self, analysis_result: AnalysisResult) -> None:
        """生成汇总信息文件"""
        summary_file = os.path.join(self.output_dir, FileConfig.SUMMARY_FILE)
        
        content = self._generate_summary_content(analysis_result)
        FileWriter.write_text_file(summary_file, content)
    
    def _generate_summary_content(self, analysis_result: AnalysisResult) -> str:
        """生成汇总内容"""
        lines = [
            "AI推理加速论文分析汇总",
            FileConfig.SEPARATOR_LINE,
            "",
            f"分析时间: {analysis_result.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"总处理论文数: {analysis_result.total_count}",
            f"AI推理加速相关论文: {analysis_result.ai_related_count}",
            f"非AI推理加速相关论文: {analysis_result.non_ai_related_count}",
            f"成功率: {analysis_result.success_rate:.2%}",
            ""
        ]
        
        if analysis_result.ai_related_papers:
            lines.extend([
                "AI相关论文列表:",
                "-" * 30
            ])
            for i, paper in enumerate(analysis_result.ai_related_papers, 1):
                match_score = paper.match_info.keyword_count if paper.match_info else 0
                lines.append(f"{i}. {paper.title} (匹配分数: {match_score})")
            lines.append("")
        
        lines.extend([
            "文件说明:",
            "-" * 30,
            "1. ai_inference_related_papers.txt - AI相关论文详细信息",
            "2. ai_inference_related_papers.csv - AI相关论文CSV格式",
            "3. non_ai_inference_papers.txt - 非AI相关论文详细信息",
            "4. non_ai_inference_papers.csv - 非AI相关论文CSV格式",
            "5. match_statistics.txt - 匹配统计报告",
            "6. match_statistics.csv - 匹配统计CSV格式",
            "7. analysis_summary.txt - 本汇总文件"
        ])
        
        return '\n'.join(lines) 