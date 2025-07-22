"""
重构后的AI推理加速论文提取器主模块
"""

import os
import logging
from typing import List, Dict, Optional, Union
from datetime import datetime
from dataclasses import dataclass

from .models import AnalysisConfig, AnalysisResult, PaperInfo, ProgressInfo
from .paper_analyzer import PaperAnalyzer
from .report_generator import ReportGenerator
from .exceptions import AIAccelerationExtractorError, ConfigurationError


@dataclass
class AnalysisParams:
    """分析参数数据类"""
    paper_filenames: Optional[List[str]] = None
    analyze_all: bool = True
    output_format: str = "both"
    paper_infos: Optional[List[Dict]] = None


class AnalysisLogger:
    """分析日志记录器"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self._setup_logging()
    
    def _setup_logging(self):
        """设置日志配置"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('analysis.log', encoding='utf-8')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def print_header(self, mode_name: str = ""):
        """打印分析头部信息"""
        mode_suffix = f" ({mode_name}模式)" if mode_name else ""
        self.logger.info(f"🚀 开始AI推理加速论文分析{mode_suffix}...")
        
        if self.config.papers_dir:
            self.logger.info(f"📁 分析目录: {self.config.papers_dir}")
        self.logger.info(f"📤 输出基础目录: {self.config.output_dir}")
        self.logger.info(f"🎯 匹配阈值: 权重>={self.config.match_threshold}分即匹配成功")
        self.logger.info(f"🔍 匹配逻辑: 纯关键词权重匹配，无排除机制")
        self.logger.info(f"🤖 大模型判别: {'启用' if self.config.enable_llm_judge else '禁用'}")
        if self.config.enable_llm_judge:
            self.logger.info(f"    对于初筛相关的论文，将调用豆包API进行总结、相关性判断和摘要翻译")
    
    def print_progress(self, paper: PaperInfo, processed_count: int, 
                      total_count: int, saved_to_disk: bool = False):
        """打印分析进度信息"""
        self.logger.info(f"正在处理 ({processed_count}/{total_count}): {paper.filename}")
        
        if paper.match_info and paper.match_info.is_match:
            self._print_ai_paper_info(paper, saved_to_disk)
        else:
            self._print_non_ai_paper_info(paper, saved_to_disk)
    
    def _print_ai_paper_info(self, paper: PaperInfo, saved_to_disk: bool):
        """打印AI相关论文信息"""
        title_kw = ", ".join(paper.match_info.title_keywords) if paper.match_info.title_keywords else "无"
        abstract_kw_count = len(paper.match_info.abstract_keywords)
        core_count = paper.match_info.core_match_count
        high_count = paper.match_info.high_match_count
        medium_count = paper.match_info.medium_match_count
        
        self.logger.info(f"  ✓ 发现AI推理加速相关论文: {paper.title}...")
        self.logger.info(f"    标题关键字: {title_kw}")
        self.logger.info(f"    摘要关键字数量: {abstract_kw_count}")
        self.logger.info(f"    总匹配分数: {paper.match_info.keyword_count}")
        self.logger.info(f"    关键字分布 - 核心:{core_count}, 高相关:{high_count}, 中等:{medium_count}")
        
        if self.config.enable_llm_judge:
            if paper.match_info.llm_summary:
                self.logger.info(f"    大模型总结: {paper.match_info.llm_summary}")
            if paper.match_info.llm_relevance:
                self.logger.info(f"    大模型相关性: {paper.match_info.llm_relevance}")
        
        if saved_to_disk:
            self.logger.info(f"    ✓ 已保存到文件")
    
    def _print_non_ai_paper_info(self, paper: PaperInfo, saved_to_disk: bool):
        """打印非AI相关论文信息"""
        match_score = paper.match_info.keyword_count if paper.match_info else 0
        self.logger.info(f"  - 非AI推理加速相关论文 (匹配分数: {match_score})")
        if saved_to_disk:
            self.logger.info(f"    ✓ 已保存到文件")
    
    def print_progress_summary(self, progress: ProgressInfo):
        """打印进度总结"""
        self.logger.info(f"\n📊 进度总结 ({progress.current}/{progress.total} 已处理):")
        self.logger.info(f"    ✅ 成功处理: {progress.success_count} 篇")
        self.logger.info(f"    ❌ 处理失败: {progress.error_count} 篇") 
        self.logger.info(f"    🎯 AI加速相关: {progress.ai_related_count} 篇")
        self.logger.info(f"    📄 其他: {progress.non_ai_related_count} 篇")
        self.logger.info(f"    💾 所有结果已实时保存到磁盘\n")
    
    def print_final_summary(self, analysis_result: AnalysisResult, output_dir: str):
        """打印最终总结"""
        self.logger.info(f"\n🎉 分析完成!")
        self.logger.info(f"📁 总文件数: {analysis_result.total_count}")
        self.logger.info(f"✅ 成功处理: {analysis_result.processed_count} 篇")
        self.logger.info(f"❌ 处理失败: {analysis_result.error_count} 篇")
        self.logger.info(f"🎯 AI推理加速相关论文: {analysis_result.ai_related_count} 篇")
        self.logger.info(f"📄 非AI推理加速相关论文: {analysis_result.non_ai_related_count} 篇")
        self.logger.info(f"💾 所有结果已保存到: {output_dir}")
    
    def print_analysis_results(self, analysis_result: AnalysisResult, output_dir: str):
        """打印分析结果"""
        if analysis_result.ai_related_papers:
            self.logger.info(f"\n✨ 发现的AI推理加速相关论文:")
            for i, paper in enumerate(analysis_result.ai_related_papers, 1):
                match_score = paper.match_info.keyword_count if paper.match_info else 0
                self.logger.info(f"{i}. {paper.title} (匹配分数: {match_score})")
        else:
            self.logger.info("\n📭 未发现AI推理加速相关论文")
        
        if analysis_result.non_ai_related_papers:
            self.logger.info(f"\n📄 非AI推理加速相关论文: {len(analysis_result.non_ai_related_papers)} 篇")
        else:
            self.logger.info("\n🎯 所有论文都与AI推理加速相关")
        
        self.logger.info(f"\n🎉 所有任务完成!")
        self.logger.info(f"📂 所有结果文件已保存到: {output_dir}")
        self.logger.info(f"🔍 共处理 {analysis_result.total_count} 篇论文")
        self.logger.info(f"✨ AI推理加速相关: {analysis_result.ai_related_count} 篇")
        self.logger.info(f"📄 其他论文: {analysis_result.non_ai_related_count} 篇")
        self.logger.info(f"💾 所有结果在处理过程中已实时保存，即使出现中断也不会丢失数据")


class ProgressTracker:
    """进度跟踪器"""
    
    def __init__(self, total: int):
        self.progress = ProgressInfo(
            current=0,
            total=total,
            success_count=0,
            error_count=0,
            ai_related_count=0,
            non_ai_related_count=0,
            start_time=datetime.now()
        )
    
    def update(self, paper_info: Optional[PaperInfo], is_error: bool = False):
        """更新进度"""
        if is_error:
            self.progress.error_count += 1
        elif paper_info:
            self.progress.success_count += 1
            if paper_info.match_info and paper_info.match_info.is_match:
                self.progress.ai_related_count += 1
            else:
                self.progress.non_ai_related_count += 1
        
        self.progress.current += 1
        self.progress.last_update = datetime.now()
    
    def get_progress(self) -> ProgressInfo:
        """获取当前进度"""
        return self.progress


class AiAccelerationExtractor:
    """AI推理加速论文提取器主类"""
    
    def __init__(self, papers_dir: str = None, output_dir: str = ".", 
                 enable_llm_judge: bool = True, match_threshold: int = 5,
                 analysis_mode: str = "pdf"):
        """
        初始化AI推理加速论文提取器
        
        Args:
            papers_dir: 论文PDF文件所在目录（仅在PDF模式下使用）
            output_dir: 输出文件保存目录
            enable_llm_judge: 是否启用大模型判别功能
            match_threshold: 匹配阈值
            analysis_mode: 分析模式，"pdf"或"paper_copilot"
        """
        self._setup_config(papers_dir, output_dir, enable_llm_judge, match_threshold, analysis_mode)
        self._setup_components()
        self.logger = AnalysisLogger(self.config)
    
    def _setup_config(self, papers_dir: str, output_dir: str, enable_llm_judge: bool, 
                     match_threshold: int, analysis_mode: str):
        """设置配置"""
        self.config = AnalysisConfig(
            enable_llm_judge=enable_llm_judge,
            analysis_mode=analysis_mode,
            papers_dir=papers_dir,
            output_dir=output_dir,
            match_threshold=match_threshold
        )
        
        try:
            self.config.validate()
        except ValueError as e:
            raise ConfigurationError(str(e)) from e
    
    def _setup_components(self):
        """设置组件"""
        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)
        
        self.report_generator = ReportGenerator(self.config.output_dir)
        self.analyzer = PaperAnalyzer(self.config)
    
    def _get_paper_files(self, paper_filenames: List[str] = None, 
                         analyze_all: bool = False) -> List[str]:
        """获取要分析的论文文件列表"""
        if analyze_all:
            if not os.path.exists(self.config.papers_dir):
                raise ConfigurationError(f"找不到论文文件夹 {self.config.papers_dir}")
            
            paper_files = [f for f in os.listdir(self.config.papers_dir) 
                          if f.lower().endswith('.pdf')]
            self.logger.logger.info(f"从 {self.config.papers_dir} 文件夹加载了 {len(paper_files)} 个PDF论文文件")
            return paper_files
        else:
            return paper_filenames or []
    
    def _process_single_paper(self, paper_input: Union[str, Dict], analysis_mode: str, 
                             i: int, total: int) -> Optional[PaperInfo]:
        """处理单篇论文"""
        try:
            paper_info = self.analyzer.analyze_paper(paper_input)
            
            if paper_info is None:
                filename = self._get_filename(paper_input, analysis_mode)
                self.logger.logger.warning(f"无法提取标题或摘要，跳过 {filename}")
                return None
            
            # 立即写入磁盘
            self._save_paper_to_disk(paper_info)
            
            # 打印进度信息
            self.logger.print_progress(paper_info, i, total, saved_to_disk=True)
            
            return paper_info
            
        except Exception as e:
            filename = self._get_filename(paper_input, analysis_mode)
            self.logger.logger.error(f"处理文件 {filename} 时出错: {str(e)}")
            self._log_error(filename, str(e))
            return None
    
    def _get_filename(self, paper_input: Union[str, Dict], analysis_mode: str) -> str:
        """获取文件名"""
        if analysis_mode == "pdf":
            return paper_input
        else:
            return paper_input.get('filename', 'unknown')
    
    def _save_paper_to_disk(self, paper_info: PaperInfo):
        """保存论文到磁盘"""
        if paper_info.match_info and paper_info.match_info.is_match:
            self.report_generator.append_ai_paper(paper_info)
        else:
            self.report_generator.append_non_ai_paper(paper_info)
    
    def _log_error(self, filename: str, error_msg: str):
        """记录错误日志"""
        try:
            error_log_file = os.path.join(self.report_generator.output_dir, "error_log.txt")
            with open(error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 处理文件 {filename} 失败: {error_msg}\n")
        except Exception:
            pass  # 忽略日志写入错误
    
    def _analyze_papers(self, papers_to_analyze: List, analysis_mode: str = "pdf") -> AnalysisResult:
        """分析论文的核心方法"""
        if not papers_to_analyze:
            self.logger.logger.warning("未找到要分析的论文")
            return AnalysisResult()
        
        # 更新配置
        self.config.analysis_mode = analysis_mode
        
        # 初始化分析结果和进度跟踪
        analysis_result = AnalysisResult(
            total_count=len(papers_to_analyze),
            analysis_time=datetime.now()
        )
        progress_tracker = ProgressTracker(len(papers_to_analyze))
        
        # 打印开始信息
        paper_type = "论文" if analysis_mode == "pdf" else "paper_copilot论文"
        self.logger.logger.info(f"\n开始分析 {len(papers_to_analyze)} 个{paper_type}...")
        self.logger.logger.info(f"分析结果将实时保存到: {self.report_generator.output_dir}")
        
        # 处理每篇论文
        for i, paper_input in enumerate(papers_to_analyze, 1):
            paper_info = self._process_single_paper(paper_input, analysis_mode, i, len(papers_to_analyze))
            
            # 更新进度
            is_error = paper_info is None
            progress_tracker.update(paper_info, is_error)
            
            # 更新分析结果
            if paper_info:
                if paper_info.match_info and paper_info.match_info.is_match:
                    analysis_result.ai_related_papers.append(paper_info)
                else:
                    analysis_result.non_ai_related_papers.append(paper_info)
            
            # 定期输出进度总结
            if i % 10 == 0:
                self.logger.print_progress_summary(progress_tracker.get_progress())
        
        # 更新最终结果
        progress = progress_tracker.get_progress()
        analysis_result.processed_count = progress.success_count
        analysis_result.error_count = progress.error_count
        
        # 最终进度更新和总结
        self.report_generator.update_progress(progress.error_count)
        self.logger.print_final_summary(analysis_result, self.report_generator.output_dir)
        
        return analysis_result
    
    def analyze(self, params: AnalysisParams, mode: str = "pdf") -> AnalysisResult:
        """
        统一的论文分析方法
        
        Args:
            params: 分析参数
            mode: 分析模式，"pdf"或"paper_copilot"
        
        Returns:
            分析结果
        """
        # 打印分析头部信息
        self.logger.print_header(mode.upper())
        
        # 获取要分析的数据
        if mode == "pdf":
            papers_to_analyze = self._get_paper_files(params.paper_filenames, params.analyze_all)
            if not papers_to_analyze:
                if params.analyze_all:
                    self.logger.logger.warning("未找到要分析的论文文件")
                else:
                    self.logger.logger.error("错误: 需要提供论文文件名列表或设置analyze_all=True")
                return AnalysisResult()
        else:  # paper_copilot mode
            papers_to_analyze = params.paper_infos or []
            if not papers_to_analyze:
                self.logger.logger.warning("未找到要分析的论文数据")
                return AnalysisResult()
        
        # 执行分析
        analysis_result = self._analyze_papers(papers_to_analyze, mode)
        
        # 生成最终报告
        self.report_generator.finalize_reports(analysis_result)
        
        # 打印分析结果
        self.logger.print_analysis_results(analysis_result, self.report_generator.output_dir)
        
        return analysis_result
    
    def parse(self, paper_filenames: List[str] = None, analyze_all: bool = True,
              output_format: str = "both") -> AnalysisResult:
        """
        解析PDF论文并保存结果
        
        Args:
            paper_filenames: 要分析的论文文件名列表
            analyze_all: 是否分析全量论文
            output_format: 输出格式，可选 "txt", "csv", "both"
        
        Returns:
            分析结果
        """
        params = AnalysisParams(
            paper_filenames=paper_filenames,
            analyze_all=analyze_all,
            output_format=output_format
        )
        return self.analyze(params, "pdf")
    
    def parse_paper_copilot(self, paper_infos: List[Dict], 
                           output_format: str = "both") -> AnalysisResult:
        """
        解析paper_copilot论文数据并保存结果
        
        Args:
            paper_infos: paper_copilot论文的解析结果列表
            output_format: 输出格式，可选 "txt", "csv", "both"
        
        Returns:
            分析结果
        """
        params = AnalysisParams(
            paper_infos=paper_infos,
            output_format=output_format
        )
        return self.analyze(params, "paper_copilot")


class ExtractorFactory:
    """提取器工厂类"""
    
    @staticmethod
    def create_extractor(**kwargs) -> AiAccelerationExtractor:
        """创建提取器实例"""
        return AiAccelerationExtractor(**kwargs)
    
    @staticmethod
    def create_and_analyze(extractor_params: Dict, analysis_params: AnalysisParams, 
                          mode: str) -> AnalysisResult:
        """创建提取器并执行分析"""
        extractor = ExtractorFactory.create_extractor(**extractor_params)
        return extractor.analyze(analysis_params, mode)


def ai_acceleration_parse(papers_dir: str, output_dir: str = ".", 
                         paper_filenames: List[str] = None, analyze_all: bool = True,
                         output_format: str = "both", enable_llm_judge: bool = True,
                         match_threshold: int = 5) -> AnalysisResult:
    """
    对外提供的AI推理加速论文解析函数
    
    Args:
        papers_dir: 论文PDF文件所在目录
        output_dir: 输出文件保存目录，默认为当前目录
        paper_filenames: 要分析的论文文件名列表，如果为None且analyze_all为True则分析所有论文
        analyze_all: 是否分析papers_dir下的全量论文，默认为True
        output_format: 输出格式，可选 "txt", "csv", "both"
        enable_llm_judge: 是否启用大模型判别功能，默认为True
        match_threshold: 匹配阈值，默认为5
    
    Returns:
        分析结果
    """
    extractor_params = {
        'papers_dir': papers_dir,
        'output_dir': output_dir,
        'enable_llm_judge': enable_llm_judge,
        'match_threshold': match_threshold
    }
    analysis_params = AnalysisParams(
        paper_filenames=paper_filenames,
        analyze_all=analyze_all,
        output_format=output_format
    )
    return ExtractorFactory.create_and_analyze(extractor_params, analysis_params, "pdf")


def ai_acceleration_parse_paper_copilot(paper_infos: List[Dict], 
                                       output_dir: str = ".", 
                                       output_format: str = "both", 
                                       enable_llm_judge: bool = True,
                                       match_threshold: int = 5) -> AnalysisResult:
    """
    对外提供的AI推理加速论文解析函数（paper_copilot模式）
    
    Args:
        paper_infos: paper_copilot论文的解析结果
        output_dir: 输出文件保存目录，默认为当前目录
        output_format: 输出格式，可选 "txt", "csv", "both"
        enable_llm_judge: 是否启用大模型判别功能，默认为True
        match_threshold: 匹配阈值，默认为5
    
    Returns:
        分析结果
    """
    extractor_params = {
        'output_dir': output_dir,
        'enable_llm_judge': enable_llm_judge,
        'match_threshold': match_threshold,
        'analysis_mode': 'paper_copilot'
    }
    analysis_params = AnalysisParams(
        paper_infos=paper_infos,
        output_format=output_format
    )
    return ExtractorFactory.create_and_analyze(extractor_params, analysis_params, "paper_copilot") 