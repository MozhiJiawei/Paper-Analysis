from typing import Dict, Optional, List, Tuple
import os
import csv
from utils.pdf_extractor import extract_paper_abstract


class _KeywordConfig:
    """关键词配置类，管理所有匹配关键词"""
    
    def __init__(self):
        # 核心推理加速关键词（最高权重：5分）
        self.core_keywords = {
            "inference_optimization": [
                "inference acceleration", "inference optimization", "inference speedup",
                "model acceleration", "neural acceleration", "ai acceleration",
                "inference latency", "inference throughput", "inference efficiency",
                "model serving", "model deployment", "real-time inference",
                "fast inference", "efficient inference", "accelerated inference",
                "inference engine", "model optimization", "serving optimization",
                "rapid inference", "inference time", "inference speed"
            ],
            "quantization_compression": [
                "quantization", "pruning", "model compression", "weight pruning",
                "activation quantization", "weight quantization", "int8", "fp16", "bf16",
                "low-precision", "mixed precision", "post-training quantization", "qat",
                "quantization-aware training", "bit-width optimization", "sparse models",
                "structured pruning", "unstructured pruning", "magnitude pruning"
            ],
            "distillation_acceleration": [
                "knowledge distillation", "model distillation", "teacher-student",
                "student model", "teacher model", "distillation training",
                "lightweight model", "compact model", "model miniaturization"
            ],
            "spiking_acceleration": [
                "spiking neural network", "spiking neural networks", "snn", "snns",
                "spiking neurons", "spiking transformer", "spiking", "neuromorphic",
                "spike-based", "event-driven", "temporal coding", "rate coding",
                "leaky integrate-and-fire", "spiking convolution", "spiking attention"
            ]
        }

        # 高相关推理技术（权重：4分）
        self.high_relevance_keywords = {
            "inference_techniques": [
                "early exit", "dynamic inference", "adaptive inference",
                "speculative decoding", "parallel decoding", "batch inference",
                "kv cache", "kv caching", "attention optimization", "memory optimization",
                "dynamic batching", "continuous batching", "tensor parallelism",
                "adaptive attention", "elastic", "conditional execution", "skip connections",
                "progressive inference", "multi-exit", "conditional computation"
            ],
            "hardware_acceleration": [
                "gpu acceleration", "tpu optimization", "fpga implementation",
                "tensorcore", "cuda optimization", "hardware-aware optimization",
                "edge deployment", "mobile inference", "embedded inference",
                "hardware efficiency", "memory-efficient inference", "edge ai",
                "mobile ai", "embedded ai", "on-device inference"
            ],
            "frameworks_engines": [
                "tensorrt", "onnx runtime", "tvm", "openvino", "tensorflow lite",
                "pytorch mobile", "vllm", "triton inference", "tensorrt-llm",
                "deepspeed", "fastertransformer", "lightllm", "flash attention"
            ],
            "diffusion_acceleration": [
                "diffusion acceleration", "fast diffusion", "few steps diffusion",
                "diffusion distillation", "diffusion optimization", "step selection",
                "adaptive step", "flash diffusion", "accelerating diffusion",
                "efficient diffusion", "rapid diffusion", "diffusion speedup"
            ]
        }

        # 中等相关技术（权重：3分）
        self.medium_relevance_keywords = {
            "optimization_techniques": [
                "kernel fusion", "operator fusion", "graph optimization",
                "memory efficiency", "computational efficiency", "parameter efficiency",
                "flops reduction", "latency reduction", "throughput improvement",
                "efficient computing", "resource optimization", "computation optimization",
                "runtime optimization", "performance optimization", "energy efficiency"
            ],
            "model_architectures": [
                "efficient transformer", "lightweight neural network", "mobile model",
                "compact architecture", "efficient architecture", "spiking neural network",
                "binary neural network", "efficient attention", "linear attention",
                "lightweight transformer", "mobile transformer", "efficient convolution",
                "depthwise separable", "mobile nets", "efficient nets"
            ],
            "serving_deployment": [
                "model serving", "deployment optimization", "production deployment",
                "scalable inference", "high-throughput serving", "low-latency serving",
                "edge ai", "mobile ai", "real-time deployment", "streaming inference",
                "online inference", "efficient deployment"
            ],
            "vision_language_efficiency": [
                "efficient multimodal", "efficient vision language", "multimodal efficiency",
                "vision language optimization", "multimodal acceleration", "vlm efficiency",
                "efficient vlm", "lightweight multimodal", "fast multimodal",
                "efficient mllm", "multimodal inference", "visual token", "token reduction",
                "elastic visual", "adaptive multimodal"
            ]
        }

        # 支撑关键词（权重：2分）
        self.supporting_keywords = {
            "performance_metrics": [
                "latency", "throughput", "speed", "efficiency", "performance",
                "runtime", "acceleration", "optimization", "fast", "rapid",
                "time complexity", "space complexity", "memory usage", "compute cost",
                "inference cost", "computational cost", "resource usage"
            ],
            "model_types": [
                "llm", "large language model", "transformer", "neural network",
                "deep learning", "generative model", "foundation model", "vlm",
                "vision language model", "multimodal model", "diffusion model",
                "state space model", "ssm", "mamba"
            ],
            "efficiency_indicators": [
                "efficient", "lightweight", "compact", "fast", "rapid", "quick",
                "accelerated", "optimized", "streamlined", "enhanced performance",
                "improved efficiency", "reduced latency", "faster", "speedup",
                "boost", "boosting", "enhance", "enhancing"
            ]
        }

        # 强排除关键词
        self.strong_exclusion_keywords = [
            # 训练相关
            "pre-training", "training from scratch", "training optimization", "learning rate scheduling",
            "gradient descent", "backpropagation", "training convergence", "training stability",
            # 特定应用领域（非AI推理）
            "medical diagnosis", "disease prediction", "drug discovery", "molecular property",
            "protein folding", "genomic analysis", "bioinformatics", "chemistry prediction",
            "financial prediction", "stock market", "economic modeling", "social network analysis",
            "political science", "recommendation system", "search system", "information retrieval",
            # 纯理论研究
            "theoretical analysis", "mathematical proof", "complexity theory", "convergence analysis",
            "information theory", "game theory", "optimization theory",
            # 数据和基础设施
            "dataset construction", "data collection", "data annotation", "benchmark dataset",
            "data preprocessing", "feature engineering", "data mining"
        ]

        # 弱排除关键词
        self.weak_exclusion_keywords = [
            "fine-tuning", "transfer learning", "few-shot learning", "zero-shot learning",
            "reinforcement learning", "federated learning", "continual learning",
            "computer vision", "natural language processing", "speech recognition",
            "question answering", "text generation", "machine translation"
        ]


class _KeywordMatcher:
    """关键词匹配器，负责匹配和评分逻辑"""
    
    def __init__(self):
        self._config = _KeywordConfig()
    
    def _check_exclusion_keywords(self, text: str) -> Tuple[int, int]:
        """检查排除关键词"""
        strong_count = sum(1 for keyword in self._config.strong_exclusion_keywords if keyword in text)
        weak_count = sum(1 for keyword in self._config.weak_exclusion_keywords if keyword in text)
        return strong_count, weak_count
    
    def _match_keyword_category(self, title: str, abstract: str, keywords: Dict[str, List[str]], 
                               priority: str, weight: int) -> Tuple[List[Dict], int]:
        """匹配特定类别的关键词"""
        matched_keywords = []
        category_count = 0
        
        for category, category_keywords in keywords.items():
            for keyword in category_keywords:
                keyword_lower = keyword.lower()
                title_match = keyword_lower in title
                abstract_match = keyword_lower in abstract
                
                if title_match or abstract_match:
                    category_count += 1
                    matched_keywords.append({
                        'keyword': keyword,
                        'category': category,
                        'priority': priority,
                        'weight': weight,
                        'in_title': title_match,
                        'in_abstract': abstract_match
                    })
        
        return matched_keywords, category_count
    
    def match_keywords(self, title: str, abstract: str) -> Dict:
        """执行关键词匹配"""
        # 安全处理输入参数
        title = title or ""
        abstract = abstract or ""
        title_lower = title.lower()
        abstract_lower = abstract.lower()
        full_text = f"{title_lower} {abstract_lower}"
        
        # 检查排除关键词
        strong_exclusion, weak_exclusion = self._check_exclusion_keywords(full_text)
        
        if strong_exclusion >= 1:
            return {
                'is_match': False,
                'matched_keywords': [],
                'keyword_count': 0,
                'title_keywords': [],
                'abstract_keywords': [],
                'core_match_count': 0,
                'high_match_count': 0,
                'medium_match_count': 0,
                'supporting_match_count': 0,
                'weak_exclusion_count': weak_exclusion,
                'strong_exclusion_count': strong_exclusion,
                'exclusion_reason': f'Strong exclusion keywords found: {strong_exclusion}'
            }
        
        # 匹配各类别关键词
        all_matched_keywords = []
        total_score = 0
        
        # 核心关键词匹配
        core_keywords, core_count = self._match_keyword_category(
            title_lower, abstract_lower, self._config.core_keywords, 'core', 5)
        all_matched_keywords.extend(core_keywords)
        
        # 高相关关键词匹配
        high_keywords, high_count = self._match_keyword_category(
            title_lower, abstract_lower, self._config.high_relevance_keywords, 'high', 4)
        all_matched_keywords.extend(high_keywords)
        
        # 中等相关关键词匹配
        medium_keywords, medium_count = self._match_keyword_category(
            title_lower, abstract_lower, self._config.medium_relevance_keywords, 'medium', 3)
        all_matched_keywords.extend(medium_keywords)
        
        # 支撑关键词匹配（只在有其他匹配时计算）
        supporting_keywords, supporting_count = [], 0
        if core_count > 0 or high_count > 0 or medium_count > 0:
            supporting_keywords, supporting_count = self._match_keyword_category(
                title_lower, abstract_lower, self._config.supporting_keywords, 'supporting', 2)
            all_matched_keywords.extend(supporting_keywords)
        
        # 计算总分和分类信息
        title_keywords = []
        abstract_keywords = []
        
        for kw_info in all_matched_keywords:
            total_score += kw_info['weight']
            
            if kw_info['in_title']:
                title_keywords.append(kw_info['keyword'])
                # 标题关键词额外加权
                if kw_info['priority'] == 'core':
                    total_score += 3
                elif kw_info['priority'] == 'high':
                    total_score += 2
                elif kw_info['priority'] == 'medium':
                    total_score += 1
            
            if kw_info['in_abstract']:
                abstract_keywords.append(kw_info['keyword'])
        
        return {
            'matched_keywords': all_matched_keywords,
            'keyword_count': total_score,
            'title_keywords': title_keywords,
            'abstract_keywords': abstract_keywords,
            'core_match_count': core_count,
            'high_match_count': high_count,
            'medium_match_count': medium_count,
            'supporting_match_count': supporting_count,
            'weak_exclusion_count': weak_exclusion,
            'strong_exclusion_count': strong_exclusion
        }


class _MatchDecisionEngine:
    """匹配决策引擎，负责最终的匹配判断"""
    
    @staticmethod
    def should_match(match_result: Dict, threshold: int = 6) -> bool:
        """判断是否应该匹配"""
        core_count = match_result.get('core_match_count', 0)
        high_count = match_result.get('high_match_count', 0)
        medium_count = match_result.get('medium_match_count', 0)
        supporting_count = match_result.get('supporting_match_count', 0)
        weak_exclusion = match_result.get('weak_exclusion_count', 0)
        total_score = match_result.get('keyword_count', 0)
        
        # 检查强信号
        has_strong_signal = (
            core_count >= 1 or
            high_count >= 2 or
            (high_count >= 1 and medium_count >= 1) or
            (medium_count >= 2 and supporting_count >= 1 and total_score >= threshold)
        )
        
        # 根据弱排除关键词调整要求
        if weak_exclusion >= 4:
            has_strong_signal = core_count >= 1 and total_score >= threshold + 3
        elif weak_exclusion >= 3:
            has_strong_signal = (core_count >= 1 or high_count >= 2) and total_score >= threshold + 2
        elif weak_exclusion >= 2:
            has_strong_signal = (
                core_count >= 1 or high_count >= 2 or 
                (high_count >= 1 and medium_count >= 1)
            ) and total_score >= threshold + 1
        
        # 降低阈值以捕获更多相关论文
        adjusted_threshold = max(4, threshold - 2)
        
        return total_score >= adjusted_threshold and has_strong_signal


class _PaperAnalyzer:
    """论文分析器，处理单篇论文的分析逻辑"""
    
    def __init__(self):
        self._matcher = _KeywordMatcher()
        self._decision_engine = _MatchDecisionEngine()
    
    def analyze_paper(self, pdf_path: str) -> Optional[Dict]:
        """分析单篇论文"""
        try:
            paper_info = extract_paper_abstract(pdf_path)
            
            if not paper_info or not paper_info.get('title') or not paper_info.get('abstract'):
                return None
            
            # 执行关键词匹配
            match_result = self._matcher.match_keywords(
                paper_info['title'], paper_info['abstract']
            )
            
            # 确保match_result包含必要的字段
            if not isinstance(match_result, dict):
                match_result = {
                    'matched_keywords': [],
                    'keyword_count': 0,
                    'title_keywords': [],
                    'abstract_keywords': [],
                    'core_match_count': 0,
                    'high_match_count': 0,
                    'medium_match_count': 0,
                    'supporting_match_count': 0,
                    'weak_exclusion_count': 0,
                    'strong_exclusion_count': 0
                }
            
            # 判断是否匹配
            is_match = self._decision_engine.should_match(match_result)
            match_result['is_match'] = is_match
            
            paper_info['filename'] = os.path.basename(pdf_path)
            paper_info['match_info'] = match_result
            
            return paper_info
            
        except Exception as e:
            print(f"  错误: 处理文件 {os.path.basename(pdf_path)} 时出错: {str(e)}")
            return None


class _ReportGenerator:
    """报告生成器，负责生成各种输出报告"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
    
    def _format_paper_info(self, paper: Dict, show_details: bool = True) -> str:
        """格式化单篇论文信息"""
        lines = []
        lines.append(f"   文件名: {paper['filename']}")
        
        if paper['authors']:
            authors_str = ", ".join([author.get('name', '') for author in paper['authors'] if author.get('name')])
            lines.append(f"   作者: {authors_str}")
            
            # 收集并去重组织信息
            all_affiliations = set()
            for author in paper['authors']:
                if author.get('affiliation'):
                    affiliations = author.get('affiliation', [])
                    for affiliation in affiliations:
                        if affiliation and affiliation.strip():
                            all_affiliations.add(affiliation.strip())
            
            if all_affiliations:
                sorted_affiliations = sorted(list(all_affiliations))
                lines.append(f"   组织: {'; '.join(sorted_affiliations)}")
        
        if 'match_info' in paper and show_details:
            match_info = paper['match_info']
            lines.append(f"   匹配分数: {match_info['keyword_count']}")
            
            # 关键字分布
            core_count = match_info.get('core_match_count', 0)
            high_count = match_info.get('high_match_count', 0)
            medium_count = match_info.get('medium_match_count', 0)
            supporting_count = match_info.get('supporting_match_count', 0)
            lines.append(f"   关键字分布 - 核心:{core_count}, 高相关:{high_count}, 中等:{medium_count}, 支撑:{supporting_count}")
            
            if match_info['title_keywords']:
                lines.append(f"   标题关键字: {', '.join(match_info['title_keywords'])}")
        
        if paper['abstract']:
            abstract_preview = paper['abstract'][:2000] + "..." if len(paper['abstract']) > 2000 else paper['abstract']
            lines.append(f"   摘要: {abstract_preview}")
        
        return '\n'.join(lines)
    
    def save_ai_papers(self, papers: List[Dict], output_filename: str = "ai_inference_related_papers.txt"):
        """保存AI推理加速相关论文"""
        output_file = os.path.join(self.output_dir, output_filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("AI推理加速相关论文列表\n")
            f.write("=" * 50 + "\n\n")
            
            for i, paper in enumerate(papers, 1):
                f.write(f"{i}. 标题: {paper['title']}\n")
                f.write(self._format_paper_info(paper))
                f.write("\n" + "-" * 80 + "\n\n")
        
        print(f"AI推理加速相关论文已保存到 {output_file}")
    
    def save_non_ai_papers(self, papers: List[Dict], output_filename: str = "non_ai_inference_papers.txt"):
        """保存非AI推理加速相关论文"""
        output_file = os.path.join(self.output_dir, output_filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("非AI推理加速相关论文列表\n")
            f.write("=" * 50 + "\n\n")
            
            for i, paper in enumerate(papers, 1):
                f.write(f"{i}. 标题: {paper['title']}\n")
                f.write(self._format_paper_info(paper, show_details=False))
                
                # 添加排除原因
                if 'match_info' in paper:
                    match_info = paper['match_info']
                    exclusion_reason = match_info.get('exclusion_reason', '')
                    if exclusion_reason:
                        f.write(f"\n   排除原因: {exclusion_reason}")
                    else:
                        f.write(f"\n   排除原因: 匹配分数不足 (分数: {match_info['keyword_count']})")
                
                f.write("\n" + "-" * 80 + "\n\n")
        
        print(f"非AI推理加速论文结果已保存到 {output_file}")
    
    def generate_statistics(self, papers: List[Dict], output_filename: str = "match_statistics.txt"):
        """生成匹配统计报告"""
        if not papers:
            print("没有论文数据，无法生成统计报告")
            return
        
        output_file = os.path.join(self.output_dir, output_filename)
        
        # 收集统计数据
        category_stats = {}
        keyword_stats = {}
        priority_stats = {}
        
        for paper in papers:
            if 'match_info' not in paper:
                continue
            
            for kw_info in paper['match_info']['matched_keywords']:
                category = kw_info['category']
                keyword = kw_info['keyword']
                priority = kw_info.get('priority', 'unknown')
                
                category_stats[category] = category_stats.get(category, 0) + 1
                keyword_stats[keyword] = keyword_stats.get(keyword, 0) + 1
                priority_stats[priority] = priority_stats.get(priority, 0) + 1
        
        # 写入统计报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("AI推理加速相关论文匹配统计报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"总匹配论文数: {len(papers)}\n\n")
            
            # 优先级统计
            f.write("关键字优先级统计:\n")
            f.write("-" * 30 + "\n")
            for priority in ['core', 'high', 'medium', 'supporting']:
                if priority in priority_stats:
                    f.write(f"{priority}: {priority_stats[priority]} 次\n")
            f.write("\n")
            
            # 类别统计
            f.write("关键字类别统计:\n")
            f.write("-" * 30 + "\n")
            for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{category}: {count} 次\n")
            f.write("\n")
            
            # 最常见关键字
            f.write("最常匹配的关键字 (前20个):\n")
            f.write("-" * 30 + "\n")
            for keyword, count in sorted(keyword_stats.items(), key=lambda x: x[1], reverse=True)[:20]:
                f.write(f"{keyword}: {count} 次\n")
        
        print(f"匹配统计报告已保存到 {output_file}")
    
    def save_ai_papers_csv(self, papers: List[Dict], output_filename: str = "ai_inference_related_papers.csv"):
        """保存AI推理加速相关论文为CSV格式"""
        output_file = os.path.join(self.output_dir, output_filename)
        
        if not papers:
            print("没有AI推理加速相关论文数据，无法生成CSV文件")
            return
        
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            
            # 写入表头
            headers = [
                '序号', '标题', '文件名', '作者', '组织', '匹配分数',
                '核心关键字数', '高相关关键字数', '中等关键字数', '支撑关键字数',
                '标题关键字', '摘要关键字数量', '弱排除关键字数', '摘要预览'
            ]
            writer.writerow(headers)
            
            # 写入数据
            for i, paper in enumerate(papers, 1):
                match_info = paper.get('match_info', {})
                
                # 处理作者信息
                authors_str = ""
                organizations_str = ""
                if paper.get('authors'):
                    authors = [author.get('name', '') for author in paper['authors'] if author.get('name')]
                    authors_str = "; ".join(authors)
                    
                    # 收集组织信息
                    all_affiliations = set()
                    for author in paper['authors']:
                        if author.get('affiliation'):
                            for affiliation in author.get('affiliation', []):
                                if affiliation and affiliation.strip():
                                    all_affiliations.add(affiliation.strip())
                    organizations_str = "; ".join(sorted(list(all_affiliations)))
                
                # 处理关键字信息
                title_keywords = "; ".join(match_info.get('title_keywords', []))
                abstract_keywords_count = len(match_info.get('abstract_keywords', []))
                
                # 摘要预览
                abstract_preview = ""
                if paper.get('abstract'):
                    abstract_preview = paper['abstract'][:2000] + "..." if len(paper['abstract']) > 2000 else paper['abstract']
                    # 清理摘要中的换行符和逗号，避免CSV格式问题
                    abstract_preview = abstract_preview.replace('\n', ' ').replace('\r', ' ')
                
                row = [
                    i,
                    paper.get('title', ''),
                    paper.get('filename', ''),
                    authors_str,
                    organizations_str,
                    match_info.get('keyword_count', 0),
                    match_info.get('core_match_count', 0),
                    match_info.get('high_match_count', 0),
                    match_info.get('medium_match_count', 0),
                    match_info.get('supporting_match_count', 0),
                    title_keywords,
                    abstract_keywords_count,
                    match_info.get('weak_exclusion_count', 0),
                    abstract_preview
                ]
                writer.writerow(row)
        
        print(f"AI推理加速相关论文CSV文件已保存到 {output_file}")
    
    def save_non_ai_papers_csv(self, papers: List[Dict], output_filename: str = "non_ai_inference_papers.csv"):
        """保存非AI推理加速相关论文为CSV格式"""
        output_file = os.path.join(self.output_dir, output_filename)
        
        if not papers:
            print("没有非AI推理加速相关论文数据，无法生成CSV文件")
            return
        
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            
            # 写入表头
            headers = [
                '序号', '标题', '文件名', '作者', '组织', '匹配分数',
                '排除原因', '摘要预览'
            ]
            writer.writerow(headers)
            
            # 写入数据
            for i, paper in enumerate(papers, 1):
                match_info = paper.get('match_info', {})
                
                # 处理作者信息
                authors_str = ""
                organizations_str = ""
                if paper.get('authors'):
                    authors = [author.get('name', '') for author in paper['authors'] if author.get('name')]
                    authors_str = "; ".join(authors)
                    
                    # 收集组织信息
                    all_affiliations = set()
                    for author in paper['authors']:
                        if author.get('affiliation'):
                            for affiliation in author.get('affiliation', []):
                                if affiliation and affiliation.strip():
                                    all_affiliations.add(affiliation.strip())
                    organizations_str = "; ".join(sorted(list(all_affiliations)))
                
                # 排除原因
                exclusion_reason = match_info.get('exclusion_reason', '')
                if not exclusion_reason:
                    exclusion_reason = f"匹配分数不足 (分数: {match_info.get('keyword_count', 0)})"
                
                # 摘要预览
                abstract_preview = ""
                if paper.get('abstract'):
                    abstract_preview = paper['abstract'][:2000] + "..." if len(paper['abstract']) > 2000 else paper['abstract']
                    # 清理摘要中的换行符，避免CSV格式问题
                    abstract_preview = abstract_preview.replace('\n', ' ').replace('\r', ' ')
                
                row = [
                    i,
                    paper.get('title', ''),
                    paper.get('filename', ''),
                    authors_str,
                    organizations_str,
                    match_info.get('keyword_count', 0),
                    exclusion_reason,
                    abstract_preview
                ]
                writer.writerow(row)
        
        print(f"非AI推理加速相关论文CSV文件已保存到 {output_file}")
    
    def save_statistics_csv(self, papers: List[Dict], output_filename: str = "match_statistics.csv"):
        """生成匹配统计报告CSV格式"""
        if not papers:
            print("没有论文数据，无法生成统计CSV报告")
            return
        
        output_file = os.path.join(self.output_dir, output_filename)
        
        # 收集统计数据
        category_stats = {}
        keyword_stats = {}
        priority_stats = {}
        
        for paper in papers:
            if 'match_info' not in paper:
                continue
            
            for kw_info in paper['match_info']['matched_keywords']:
                category = kw_info['category']
                keyword = kw_info['keyword']
                priority = kw_info.get('priority', 'unknown')
                
                category_stats[category] = category_stats.get(category, 0) + 1
                keyword_stats[keyword] = keyword_stats.get(keyword, 0) + 1
                priority_stats[priority] = priority_stats.get(priority, 0) + 1
        
        with open(output_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            
            # 基本统计信息
            writer.writerow(['统计类别', '项目', '数量'])
            writer.writerow(['总体', '匹配论文总数', len(papers)])
            writer.writerow([])  # 空行
            
            # 优先级统计
            writer.writerow(['优先级统计', '', ''])
            for priority in ['core', 'high', 'medium', 'supporting']:
                if priority in priority_stats:
                    writer.writerow(['优先级', priority, priority_stats[priority]])
            writer.writerow([])  # 空行
            
            # 类别统计
            writer.writerow(['类别统计', '', ''])
            for category, count in sorted(category_stats.items(), key=lambda x: x[1], reverse=True):
                writer.writerow(['类别', category, count])
            writer.writerow([])  # 空行
            
            # 最常见关键字
            writer.writerow(['关键字统计 (前20个)', '', ''])
            for keyword, count in sorted(keyword_stats.items(), key=lambda x: x[1], reverse=True)[:20]:
                writer.writerow(['关键字', keyword, count])
        
        print(f"匹配统计CSV报告已保存到 {output_file}")


class AiAccelerationExtractor:
    """AI加速论文提取器主类"""
    
    def __init__(self, papers_dir: str, output_dir: str = "."):
        """
        初始化AI加速论文提取器
        
        Args:
            papers_dir: 论文PDF文件所在目录
            output_dir: 输出文件保存目录
        """
        self.papers_dir = papers_dir
        self.output_dir = output_dir
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self._analyzer = _PaperAnalyzer()
        self._report_generator = _ReportGenerator(output_dir)
    
    def _get_paper_files(self, paper_filenames: List[str] = None, analyze_all: bool = False) -> List[str]:
        """获取要分析的论文文件列表"""
        if analyze_all:
            if not os.path.exists(self.papers_dir):
                print(f"错误: 找不到论文文件夹 {self.papers_dir}")
                return []
            
            paper_files = [f for f in os.listdir(self.papers_dir) if f.lower().endswith('.pdf')]
            print(f"从 {self.papers_dir} 文件夹加载了 {len(paper_files)} 个PDF论文文件")
            return paper_files
        else:
            return paper_filenames or []
    
    def _print_analysis_progress(self, paper: Dict, processed_count: int, total_count: int):
        """打印分析进度信息"""
        filename = paper['filename']
        match_info = paper['match_info']
        
        print(f"正在处理 ({processed_count}/{total_count}): {filename}")
        
        if match_info['is_match']:
            title_kw = ", ".join(match_info['title_keywords']) if match_info['title_keywords'] else "无"
            abstract_kw_count = len(match_info['abstract_keywords'])
            core_count = match_info.get('core_match_count', 0)
            high_count = match_info.get('high_match_count', 0)
            medium_count = match_info.get('medium_match_count', 0)
            
            print(f"  ✓ 发现AI推理加速相关论文: {paper['title'][:60]}...")
            print(f"    标题关键字: {title_kw}")
            print(f"    摘要关键字数量: {abstract_kw_count}")
            print(f"    总匹配分数: {match_info['keyword_count']}")
            print(f"    关键字分布 - 核心:{core_count}, 高相关:{high_count}, 中等:{medium_count}")
            if match_info.get('weak_exclusion_count', 0) > 0:
                print(f"    弱排除关键字数量: {match_info['weak_exclusion_count']}")
        else:
            exclusion_reason = match_info.get('exclusion_reason', '')
            reason_text = f", {exclusion_reason}" if exclusion_reason else ""
            print(f"  - 非AI推理加速相关论文 (匹配分数: {match_info['keyword_count']}{reason_text})")
    
    def _analyze_papers(self, paper_filenames: List[str] = None, analyze_all: bool = False) -> Dict[str, List[Dict]]:
        """
        分析论文，筛选出与AI推理加速相关和非相关的论文
        
        Args:
            paper_filenames: 要分析的论文文件名列表
            analyze_all: 是否分析papers_dir下的全量论文
        
        Returns:
            包含'ai_related'和'non_ai_related'两个键的字典，值为论文信息列表
        """
        # 获取要分析的文件列表
        files_to_analyze = self._get_paper_files(paper_filenames, analyze_all)
        
        if not files_to_analyze:
            if analyze_all:
                print("未找到要分析的论文文件")
            else:
                print("错误: 需要提供论文文件名列表或设置analyze_all=True")
            return {'ai_related': [], 'non_ai_related': []}
        
        # 分析论文
        ai_related_papers = []
        non_ai_related_papers = []
        
        paper_type = "论文" if analyze_all else "指定论文"
        print(f"\n开始分析 {len(files_to_analyze)} 个{paper_type}...")
        
        for i, filename in enumerate(files_to_analyze, 1):
            pdf_path = os.path.join(self.papers_dir, filename)
            paper_info = self._analyzer.analyze_paper(pdf_path)
            
            if paper_info is None:
                print(f"  警告: 无法提取标题或摘要，跳过 {filename}")
                continue
            
            # 根据匹配结果分类
            if paper_info['match_info']['is_match']:
                ai_related_papers.append(paper_info)
            else:
                non_ai_related_papers.append(paper_info)
            
            # 打印进度信息
            self._print_analysis_progress(paper_info, i, len(files_to_analyze))
        
        # 打印总结
        print(f"\n分析完成!")
        print(f"总共处理论文: {len(files_to_analyze)}")
        print(f"AI推理加速相关论文: {len(ai_related_papers)}")
        print(f"非AI推理加速相关论文: {len(non_ai_related_papers)}")
        
        return {
            'ai_related': ai_related_papers,
            'non_ai_related': non_ai_related_papers
        }
    
    def parse(self, paper_filenames: List[str] = None, analyze_all: bool = False, 
              output_format: str = "txt"):
        """
        解析论文并保存结果
        
        Args:
            paper_filenames: 要分析的论文文件名列表
            analyze_all: 是否分析全量论文
            output_format: 输出格式，可选 "txt", "csv", "both"
        """

        
        # 分析论文
        results = self._analyze_papers(paper_filenames, analyze_all)
        
        ai_papers = results['ai_related']
        non_ai_papers = results['non_ai_related']
        
        # 保存AI推理加速相关论文
        if ai_papers:
            if output_format in ["txt", "both"]:
                self._report_generator.save_ai_papers(ai_papers)
                self._report_generator.generate_statistics(ai_papers)
            
            if output_format in ["csv", "both"]:
                self._report_generator.save_ai_papers_csv(ai_papers)
                self._report_generator.save_statistics_csv(ai_papers)
            
            # 打印简要统计
            print(f"\n发现的AI推理加速相关论文:")
            for i, paper in enumerate(ai_papers, 1):
                match_score = paper.get('match_info', {}).get('keyword_count', 0)
                print(f"{i}. {paper['title']} (匹配分数: {match_score})")
        else:
            print("未发现AI推理加速相关论文")
        
        # 保存非AI推理加速相关论文
        if non_ai_papers:
            if output_format in ["txt", "both"]:
                self._report_generator.save_non_ai_papers(non_ai_papers)
            
            if output_format in ["csv", "both"]:
                self._report_generator.save_non_ai_papers_csv(non_ai_papers)
            
            print(f"\n非AI推理加速相关论文已保存，共 {len(non_ai_papers)} 篇")
        else:
            print("所有论文都与AI推理加速相关")


def ai_acceleration_parse(papers_dir: str, output_dir: str = ".", 
                         paper_filenames: List[str] = None, analyze_all: bool = False,
                         output_format: str = "txt"):
    """
    对外提供的AI推理加速论文解析函数
    
    Args:
        papers_dir: 论文PDF文件所在目录
        output_dir: 输出文件保存目录，默认为当前目录
        paper_filenames: 要分析的论文文件名列表，如果为None且analyze_all为True则分析所有论文
        analyze_all: 是否分析papers_dir下的全量论文，默认为False
        output_format: 输出格式，可选 "txt"（默认）, "csv", "both"
    
    Returns:
        None
    """
    extractor = AiAccelerationExtractor(papers_dir, output_dir)
    extractor.parse(paper_filenames, analyze_all, output_format)
   