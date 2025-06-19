from typing import Dict, Optional, List
import os
import sys
from utils.pdf_extractor import extract_paper_abstract


class AiAccelerationExtractor:
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

    def is_ai_acceleration_paper(self, title, abstract, threshold=6) -> Dict:
        """
        判断论文是否与AI推理加速相关 - 改进版本

        Args:
            title: 论文标题
            abstract: 论文摘要
            threshold: 关键词匹配阈值，默认为6

        Returns:
            dict: 包含匹配结果和关键字信息的字典
        """

        # 核心推理加速关键词（最高权重：5分）
        core_inference_keywords = {
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
        high_relevance_keywords = {
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
        medium_relevance_keywords = {
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

        # 需要与核心关键词结合的术语（权重：2分）
        supporting_keywords = {
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

        # 强排除关键词 - 这些领域的论文通常不是推理加速
        strong_exclusion_keywords = [
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

        # 弱排除关键词 - 需要多个才排除（减少一些可能与推理加速相关的词）
        weak_exclusion_keywords = [
            "fine-tuning", "transfer learning", "few-shot learning", "zero-shot learning",
            "reinforcement learning", "federated learning", "continual learning",
            "computer vision", "natural language processing", "speech recognition",
            "question answering", "text generation", "machine translation"
        ]

        # 将文本转换为小写
        title_lower = title.lower()
        abstract_lower = abstract.lower()
        full_text = f"{title_lower} {abstract_lower}"

        # 检查强排除关键词
        strong_exclusion_count = 0
        for keyword in strong_exclusion_keywords:
            if keyword in full_text:
                strong_exclusion_count += 1

        # 如果有强排除关键词，直接排除
        if strong_exclusion_count >= 1:
            return {
                'is_match': False,
                'matched_keywords': [],
                'keyword_count': 0,
                'title_keywords': [],
                'abstract_keywords': [],
                'exclusion_reason': f'Strong exclusion keywords found: {strong_exclusion_count}'
            }

        # 检查弱排除关键词
        weak_exclusion_count = 0
        for keyword in weak_exclusion_keywords:
            if keyword in full_text:
                weak_exclusion_count += 1

        # 统计关键词匹配
        matched_keywords = []
        title_keywords = []
        abstract_keywords = []
        keyword_count = 0

        # 处理核心推理加速关键词（权重5）
        core_match_count = 0
        for category, category_keywords in core_inference_keywords.items():
            for keyword in category_keywords:
                keyword_lower = keyword.lower()
                title_match = keyword_lower in title_lower
                abstract_match = keyword_lower in abstract_lower

                if title_match or abstract_match:
                    core_match_count += 1
                    matched_keywords.append({
                        'keyword': keyword,
                        'category': category,
                        'priority': 'core',
                        'weight': 5,
                        'in_title': title_match,
                        'in_abstract': abstract_match
                    })
                    keyword_count += 5

                    if title_match:
                        title_keywords.append(keyword)
                        keyword_count += 3  # 标题额外加权

                    if abstract_match:
                        abstract_keywords.append(keyword)

        # 处理高相关关键词（权重4）
        high_match_count = 0
        for category, category_keywords in high_relevance_keywords.items():
            for keyword in category_keywords:
                keyword_lower = keyword.lower()
                title_match = keyword_lower in title_lower
                abstract_match = keyword_lower in abstract_lower

                if title_match or abstract_match:
                    high_match_count += 1
                    matched_keywords.append({
                        'keyword': keyword,
                        'category': category,
                        'priority': 'high',
                        'weight': 4,
                        'in_title': title_match,
                        'in_abstract': abstract_match
                    })
                    keyword_count += 4

                    if title_match:
                        title_keywords.append(keyword)
                        keyword_count += 2

                    if abstract_match:
                        abstract_keywords.append(keyword)

        # 处理中等相关关键词（权重3）
        medium_match_count = 0
        for category, category_keywords in medium_relevance_keywords.items():
            for keyword in category_keywords:
                keyword_lower = keyword.lower()
                title_match = keyword_lower in title_lower
                abstract_match = keyword_lower in abstract_lower

                if title_match or abstract_match:
                    medium_match_count += 1
                    matched_keywords.append({
                        'keyword': keyword,
                        'category': category,
                        'priority': 'medium',
                        'weight': 3,
                        'in_title': title_match,
                        'in_abstract': abstract_match
                    })
                    keyword_count += 3

                    if title_match:
                        title_keywords.append(keyword)
                        keyword_count += 1

                    if abstract_match:
                        abstract_keywords.append(keyword)

        # 处理支撑关键词（权重2，但只在有核心/高相关关键词时计算）
        supporting_match_count = 0
        if core_match_count > 0 or high_match_count > 0 or medium_match_count > 0:
            for category, category_keywords in supporting_keywords.items():
                for keyword in category_keywords:
                    keyword_lower = keyword.lower()
                    title_match = keyword_lower in title_lower
                    abstract_match = keyword_lower in abstract_lower

                    if title_match or abstract_match:
                        supporting_match_count += 1
                        matched_keywords.append({
                            'keyword': keyword,
                            'category': category,
                            'priority': 'supporting',
                            'weight': 2,
                            'in_title': title_match,
                            'in_abstract': abstract_match
                        })
                        keyword_count += 2

                        if title_match:
                            title_keywords.append(keyword)
                            keyword_count += 1

                        if abstract_match:
                            abstract_keywords.append(keyword)

        # 更宽松的匹配条件 - 适应用户提到的案例
        # 必须满足以下条件之一：
        # 1. 至少一个核心关键词
        # 2. 至少两个高相关关键词
        # 3. 至少一个高相关关键词 + 一个中等相关关键词
        # 4. 至少两个中等相关关键词 + 支撑关键词 + 总分达到阈值

        has_strong_signal = (
                core_match_count >= 1 or  # 至少一个核心关键词
                high_match_count >= 2 or  # 至少两个高相关关键词
                (high_match_count >= 1 and medium_match_count >= 1) or  # 高相关+中等相关
                (medium_match_count >= 2 and supporting_match_count >= 1 and keyword_count >= threshold)  # 多个中等相关+支撑
        )

        # 如果弱排除关键词太多，需要更强的信号
        if weak_exclusion_count >= 4:
            has_strong_signal = core_match_count >= 1 and keyword_count >= threshold + 3
        elif weak_exclusion_count >= 3:
            has_strong_signal = (core_match_count >= 1 or high_match_count >= 2) and keyword_count >= threshold + 2
        elif weak_exclusion_count >= 2:
            has_strong_signal = (core_match_count >= 1 or high_match_count >= 2 or
                                 (high_match_count >= 1 and medium_match_count >= 1)) and keyword_count >= threshold + 1

        # 降低阈值以捕获更多相关论文，但保持质量控制
        adjusted_threshold = max(4, threshold - 2)  # 将阈值从6降低到4

        # 基于阈值和强信号判断
        is_match = keyword_count >= adjusted_threshold and has_strong_signal

        return {
            'is_match': is_match,
            'matched_keywords': matched_keywords,
            'keyword_count': keyword_count,
            'title_keywords': title_keywords,
            'abstract_keywords': abstract_keywords,
            'core_match_count': core_match_count,
            'high_match_count': high_match_count,
            'medium_match_count': medium_match_count,
            'supporting_match_count': supporting_match_count,
            'weak_exclusion_count': weak_exclusion_count,
            'strong_exclusion_count': strong_exclusion_count
        }

    def analyze_papers(self, paper_filenames: List[str] = None, analyze_all: bool = False) -> Dict[str, List[Dict]]:
        """
        分析论文，筛选出与AI推理加速相关和非相关的论文
        
        Args:
            paper_filenames: 要分析的论文文件名列表，如果为None且analyze_all=True则分析所有PDF
            analyze_all: 是否分析papers_dir下的全量论文
        
        Returns:
            包含'ai_related'和'non_ai_related'两个键的字典，值为论文信息列表
        """
        
        if analyze_all:
            print("开始分析路径下的全量论文，筛选AI推理加速相关论文...")
            
            # 获取文件夹下所有PDF文件
            if not os.path.exists(self.papers_dir):
                print(f"错误: 找不到论文文件夹 {self.papers_dir}")
                return {'ai_related': [], 'non_ai_related': []}
                
            paper_filenames = []
            for filename in os.listdir(self.papers_dir):
                if filename.lower().endswith('.pdf'):
                    paper_filenames.append(filename)
            print(f"从 {self.papers_dir} 文件夹加载了 {len(paper_filenames)} 个PDF论文文件")
        else:
            if not paper_filenames:
                print("错误: 需要提供论文文件名列表或设置analyze_all=True")
                return {'ai_related': [], 'non_ai_related': []}
                
            print(f"开始分析指定的 {len(paper_filenames)} 个论文...")
        
        if not paper_filenames:
            print("未找到要分析的论文文件")
            return {'ai_related': [], 'non_ai_related': []}
        
        ai_inference_papers = []
        non_ai_papers = []
        processed_count = 0
        
        paper_type = "论文" if analyze_all else "指定论文"
        print(f"\n开始分析 {len(paper_filenames)} 个{paper_type}...")
        
        for filename in paper_filenames:
            pdf_path = os.path.join(self.papers_dir, filename)
            processed_count += 1
            
            try:
                print(f"正在处理 ({processed_count}/{len(paper_filenames)}): {filename}")
                
                # 提取论文信息
                paper_info = extract_paper_abstract(pdf_path)
                
                if not paper_info['title'] or not paper_info['abstract']:
                    print(f"  警告: 无法提取标题或摘要，跳过")
                    continue
                
                # 判断是否与AI推理加速相关
                match_result = self.is_ai_acceleration_paper(paper_info['title'], paper_info['abstract'])
                
                # 添加通用信息
                paper_info['filename'] = filename
                paper_info['match_info'] = match_result
                
                if match_result['is_match']:
                    ai_inference_papers.append(paper_info)
                    
                    # 显示匹配的关键字信息
                    title_kw = ", ".join(match_result['title_keywords']) if match_result['title_keywords'] else "无"
                    abstract_kw_count = len(match_result['abstract_keywords'])
                    core_count = match_result.get('core_match_count', 0)
                    high_count = match_result.get('high_match_count', 0)
                    medium_count = match_result.get('medium_match_count', 0)
                    
                    print(f"  ✓ 发现AI推理加速相关论文: {paper_info['title'][:60]}...")
                    print(f"    标题关键字: {title_kw}")
                    print(f"    摘要关键字数量: {abstract_kw_count}")
                    print(f"    总匹配分数: {match_result['keyword_count']}")
                    print(f"    关键字分布 - 核心:{core_count}, 高相关:{high_count}, 中等:{medium_count}")
                    if match_result.get('weak_exclusion_count', 0) > 0:
                        print(f"    弱排除关键字数量: {match_result['weak_exclusion_count']}")
                else:
                    non_ai_papers.append(paper_info)
                    
                    exclusion_reason = match_result.get('exclusion_reason', '')
                    weak_exclusion = match_result.get('weak_exclusion_count', 0)
                    strong_exclusion = match_result.get('strong_exclusion_count', 0)
                    
                    reason_text = ""
                    if exclusion_reason:
                        reason_text = f", {exclusion_reason}"
                    elif strong_exclusion > 0:
                        reason_text = f", 强排除关键字: {strong_exclusion}"
                    elif weak_exclusion > 2:
                        reason_text = f", 弱排除关键字过多: {weak_exclusion}"
                    
                    print(f"  - 非AI推理加速相关论文 (匹配分数: {match_result['keyword_count']}{reason_text})")
                    
            except Exception as e:
                print(f"  错误: 处理文件 {filename} 时出错: {str(e)}")
                continue
        
        print(f"\n分析完成!")
        print(f"总共处理论文: {processed_count}")
        print(f"AI推理加速相关论文: {len(ai_inference_papers)}")
        print(f"非AI推理加速相关论文: {len(non_ai_papers)}")
        
        return {
            'ai_related': ai_inference_papers,
            'non_ai_related': non_ai_papers
        }

    def save_ai_papers(self, papers: List[Dict], output_filename: str = "ai_inference_related_papers.txt"):
        """
        保存AI推理加速相关论文信息到文件
        
        Args:
            papers: 论文信息列表
            output_filename: 输出文件名
        """
        output_file = os.path.join(self.output_dir, output_filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("AI推理加速相关论文列表\n")
            f.write("=" * 50 + "\n\n")
            
            for i, paper in enumerate(papers, 1):
                f.write(f"{i}. 标题: {paper['title']}\n")
                f.write(f"   文件名: {paper['filename']}\n")
                
                if paper['authors']:
                    authors_str = ", ".join([author.get('name', '') for author in paper['authors'] if author.get('name')])
                    f.write(f"   作者: {authors_str}\n")
                
                # 添加匹配关键字信息
                if 'match_info' in paper:
                    match_info = paper['match_info']
                    f.write(f"   匹配分数: {match_info['keyword_count']}\n")
                    
                    # 显示各类别关键字数量
                    core_count = match_info.get('core_match_count', 0)
                    high_count = match_info.get('high_match_count', 0)
                    medium_count = match_info.get('medium_match_count', 0)
                    supporting_count = match_info.get('supporting_match_count', 0)
                    f.write(f"   关键字分布 - 核心:{core_count}, 高相关:{high_count}, 中等:{medium_count}, 支撑:{supporting_count}\n")
                    
                    if match_info['title_keywords']:
                        f.write(f"   标题关键字: {', '.join(match_info['title_keywords'])}\n")
                    
                    if match_info['abstract_keywords']:
                        # 限制显示的摘要关键字数量，避免过长
                        abstract_kw_display = match_info['abstract_keywords'][:10]
                        if len(match_info['abstract_keywords']) > 10:
                            abstract_kw_display.append(f"... (共{len(match_info['abstract_keywords'])}个)")
                        f.write(f"   摘要关键字: {', '.join(abstract_kw_display)}\n")
                    
                    # 按优先级和类别显示匹配的关键字
                    priority_categories = {}
                    for kw_info in match_info['matched_keywords']:
                        priority = kw_info.get('priority', 'unknown')
                        category = kw_info['category']
                        key = f"{priority}_{category}"
                        if key not in priority_categories:
                            priority_categories[key] = []
                        priority_categories[key].append(kw_info['keyword'])
                    
                    f.write(f"   关键字详情:\n")
                    # 按优先级排序显示
                    priority_order = ['core', 'high', 'medium', 'supporting']
                    for priority in priority_order:
                        for key, keywords in priority_categories.items():
                            if key.startswith(priority):
                                category = key.split('_', 1)[1]
                                f.write(f"     {priority}_{category}: {', '.join(keywords)}\n")
                    
                    # 显示排除信息
                    if match_info.get('weak_exclusion_count', 0) > 0:
                        f.write(f"   弱排除关键字数量: {match_info['weak_exclusion_count']}\n")
                    if match_info.get('strong_exclusion_count', 0) > 0:
                        f.write(f"   强排除关键字数量: {match_info['strong_exclusion_count']}\n")
                
                if paper['abstract']:
                    # 限制摘要长度以便阅读
                    abstract_preview = paper['abstract'][:1500] + "..." if len(paper['abstract']) > 1500 else paper['abstract']
                    f.write(f"   摘要: {abstract_preview}\n")
                
                f.write("\n" + "-" * 80 + "\n\n")
        
        print(f"AI推理加速相关论文已保存到 {output_file}")

    def save_non_ai_papers(self, papers: List[Dict], output_filename: str = "non_ai_inference_papers.txt"):
        """
        保存非AI推理加速相关论文信息到文件
        
        Args:
            papers: 论文信息列表
            output_filename: 输出文件名
        """
        output_file = os.path.join(self.output_dir, output_filename)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("非AI推理加速相关论文列表\n")
            f.write("=" * 50 + "\n\n")
            
            for i, paper in enumerate(papers, 1):
                f.write(f"{i}. 标题: {paper['title']}\n")
                f.write(f"   文件名: {paper['filename']}\n")
                
                if paper['authors']:
                    authors_str = ", ".join([author.get('name', '') for author in paper['authors'] if author.get('name')])
                    f.write(f"   作者: {authors_str}\n")
                
                # 添加匹配分析信息
                if 'match_info' in paper:
                    match_info = paper['match_info']
                    f.write(f"   匹配分数: {match_info['keyword_count']}\n")
                    
                    # 显示排除原因
                    exclusion_reason = match_info.get('exclusion_reason', '')
                    weak_exclusion = match_info.get('weak_exclusion_count', 0)
                    strong_exclusion = match_info.get('strong_exclusion_count', 0)
                    
                    if exclusion_reason:
                        f.write(f"   排除原因: {exclusion_reason}\n")
                    elif strong_exclusion > 0:
                        f.write(f"   强排除关键字数量: {strong_exclusion}\n")
                    elif weak_exclusion > 2:
                        f.write(f"   弱排除关键字数量: {weak_exclusion}\n")
                    else:
                        f.write(f"   排除原因: 匹配分数不足\n")
                    
                    # 如果有匹配到的关键字，也显示出来
                    if match_info.get('matched_keywords'):
                        matched_kw = [kw_info['keyword'] for kw_info in match_info['matched_keywords']]
                        if matched_kw:
                            f.write(f"   匹配到的关键字: {', '.join(matched_kw[:10])}\n")
                            if len(matched_kw) > 10:
                                f.write(f"   (共匹配 {len(matched_kw)} 个关键字)\n")
                
                if paper['abstract']:
                    # 限制摘要长度以便阅读
                    abstract_preview = paper['abstract'][:1000] + "..." if len(paper['abstract']) > 1000 else paper['abstract']
                    f.write(f"   摘要: {abstract_preview}\n")
                
                f.write("\n" + "-" * 80 + "\n\n")
        
        print(f"非AI推理加速论文结果已保存到 {output_file}")

    def generate_statistics(self, papers: List[Dict], output_filename: str = "match_statistics.txt"):
        """
        生成匹配统计报告
        
        Args:
            papers: 论文信息列表
            output_filename: 统计报告输出文件名
        """
        if not papers:
            print("没有论文数据，无法生成统计报告")
            return
        
        output_file = os.path.join(self.output_dir, output_filename)
        
        # 统计各类别关键字的出现频率
        category_stats = {}
        keyword_stats = {}
        title_keyword_stats = {}
        priority_stats = {}
        
        for paper in papers:
            if 'match_info' not in paper:
                continue
                
            match_info = paper['match_info']
            
            # 统计各类别和优先级
            for kw_info in match_info['matched_keywords']:
                category = kw_info['category']
                keyword = kw_info['keyword']
                priority = kw_info.get('priority', 'unknown')
                
                # 类别统计
                if category not in category_stats:
                    category_stats[category] = 0
                category_stats[category] += 1
                
                # 优先级统计
                if priority not in priority_stats:
                    priority_stats[priority] = 0
                priority_stats[priority] += 1
                
                # 关键字统计
                if keyword not in keyword_stats:
                    keyword_stats[keyword] = 0
                keyword_stats[keyword] += 1
                
                # 标题关键字统计
                if kw_info.get('in_title', False):
                    if keyword not in title_keyword_stats:
                        title_keyword_stats[keyword] = 0
                    title_keyword_stats[keyword] += 1
        
        # 写入统计报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("AI推理加速相关论文匹配统计报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"总匹配论文数: {len(papers)}\n\n")
            
            # 优先级统计
            f.write("关键字优先级统计:\n")
            f.write("-" * 30 + "\n")
            priority_order = ['core', 'high', 'medium', 'supporting', 'unknown']
            for priority in priority_order:
                if priority in priority_stats:
                    f.write(f"{priority}: {priority_stats[priority]} 次\n")
            f.write("\n")
            
            # 类别统计
            f.write("关键字类别统计:\n")
            f.write("-" * 30 + "\n")
            sorted_categories = sorted(category_stats.items(), key=lambda x: x[1], reverse=True)
            for category, count in sorted_categories:
                f.write(f"{category}: {count} 次\n")
            
            f.write("\n")
            
            # 最常见的关键字
            f.write("最常匹配的关键字 (前20个):\n")
            f.write("-" * 30 + "\n")
            sorted_keywords = sorted(keyword_stats.items(), key=lambda x: x[1], reverse=True)[:20]
            for keyword, count in sorted_keywords:
                f.write(f"{keyword}: {count} 次\n")
            
            f.write("\n")
            
            # 标题中最常见的关键字
            f.write("标题中最常出现的关键字 (前15个):\n")
            f.write("-" * 30 + "\n")
            sorted_title_keywords = sorted(title_keyword_stats.items(), key=lambda x: x[1], reverse=True)[:15]
            for keyword, count in sorted_title_keywords:
                f.write(f"{keyword}: {count} 次\n")
            
            f.write("\n")
            
            # 匹配分数分布
            f.write("匹配分数分布:\n")
            f.write("-" * 30 + "\n")
            score_distribution = {}
            for paper in papers:
                if 'match_info' in paper:
                    score = paper['match_info']['keyword_count']
                    if score not in score_distribution:
                        score_distribution[score] = 0
                    score_distribution[score] += 1
            
            for score in sorted(score_distribution.keys()):
                f.write(f"分数 {score}: {score_distribution[score]} 篇论文\n")
        
        print(f"匹配统计报告已保存到 {output_file}")

    def parse(self, paper_filenames: List[str] = None, analyze_all: bool = False):
        """
        解析论文并保存结果
        
        Args:
            paper_filenames: 要分析的论文文件名列表
            analyze_all: 是否分析全量论文
        """
        # 分析论文
        results = self.analyze_papers(paper_filenames, analyze_all)
        
        ai_papers = results['ai_related']
        non_ai_papers = results['non_ai_related']
        
        # 保存AI推理加速相关论文
        if ai_papers:
            self.save_ai_papers(ai_papers)
            
            # 生成匹配统计报告
            self.generate_statistics(ai_papers)
            
            # 打印简要统计
            print(f"\n发现的AI推理加速相关论文:")
            for i, paper in enumerate(ai_papers, 1):
                match_score = paper.get('match_info', {}).get('keyword_count', 0)
                print(f"{i}. {paper['title']} (匹配分数: {match_score})")
        else:
            print("未发现AI推理加速相关论文")
        
        # 保存非AI推理加速相关论文
        if non_ai_papers:
            self.save_non_ai_papers(non_ai_papers)
            print(f"\n非AI推理加速相关论文已保存，共 {len(non_ai_papers)} 篇")
        else:
            print("所有论文都与AI推理加速相关")


# 为向后兼容保留独立的函数
def is_ai_acceleration_paper(title, abstract, threshold=6):
    """独立的AI加速论文判断函数，保持向后兼容"""
    extractor = AiAccelerationExtractor(".", ".")
    return extractor.is_ai_acceleration_paper(title, abstract, threshold)


def abstract_parser(pdf_path: str):
    """PDF摘要解析函数，保持向后兼容"""
    return extract_paper_abstract(pdf_path)