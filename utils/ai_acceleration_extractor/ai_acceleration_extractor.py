from typing import Dict, Optional, List, Tuple
import os
import csv
from datetime import datetime
from utils.pdf_extractor import extract_paper_abstract
from utils.doubao_api import call_doubao


class _KeywordConfig:
    """关键词配置类，管理所有匹配关键词"""
    
    def __init__(self):
        # 核心推理加速关键词（最高权重：6分）
        self.core_keywords = {
            "inference_optimization": [
                "inference acceleration", "inference optimization", "inference speedup",
                "model acceleration", "neural acceleration", "ai acceleration",
                "inference latency", "inference throughput", "inference efficiency",
                "model serving optimization", "inference engine", "serving optimization",
                "rapid inference", "accelerated inference", "inference time optimization",
                "acceleration of diffusion", "accelerating multimodal", 
                "lazy learning for acceleration", "fast training and inference",
                "neural network acceleration", "deep learning acceleration",
                "efficient inference", "fast inference", "optimized inference",
                "low-latency inference", "real-time inference", "fast decoding",
                "inference performance", "serving performance", "model efficiency",
                "training-free acceleration", "accelerating llm", "llm acceleration"
            ],
            "quantization_compression": [
                "quantization", "pruning", "model compression", "weight pruning",
                "activation quantization", "weight quantization", "int8", "fp16", "bf16",
                "low-precision", "mixed precision", "post-training quantization", "qat",
                "quantization-aware training", "bit-width optimization", "sparse models",
                "structured pruning", "unstructured pruning", "magnitude pruning",
                "arbitrary-bit quantized", "quantized inference acceleration",
                "neural compression", "network compression", "model size reduction",
                "quantized model", "compressed model", "low-bit quantization",
                "extremely low bit", "mixed precision quantization", "quantized inference",
                "numerical pruning", "weight compression", "model miniaturization"
            ],
            "distillation_acceleration": [
                "knowledge distillation", "model distillation", "teacher-student",
                "student model", "teacher model", "distillation training",
                "lightweight model", "compact model", "model miniaturization",
                "neural distillation", "attention distillation", "cross-modal distillation",
                "multi-teacher distillation", "self-distillation", "contrastive distillation"
            ],
            "spiking_acceleration": [
                "spiking neural network", "spiking neural networks", "snn", "snns",
                "spiking neurons", "spiking transformer", "spiking", "neuromorphic",
                "spike-based", "event-driven", "temporal coding", "rate coding",
                "leaky integrate-and-fire", "spiking convolution", "spiking attention",
                "spike2former", "efficient spiking", "spiking yolo"
            ],
            "moe_acceleration": [
                "mixture of experts", "moe", "sparse moe", "dense moe", "switch transformer",
                "expert routing", "expert selection", "gating network", "routing network",
                "sparse activation", "expert parallelism", "moe scaling", "expert capacity",
                "load balancing", "auxiliary loss", "routing loss", "expert dropout",
                "top-k routing", "expert utilization", "moe efficiency", "sparse expert",
                "expert load balancing", "dynamic routing", "adaptive routing",
                "expert sparsity", "moe optimization", "expert caching", "expert scheduling",
                "communication-efficient mixture", "communication-efficient moe"
            ]
        }

        # 高相关推理技术（权重：5分）
        self.high_relevance_keywords = {
            "inference_techniques": [
                "early exit", "dynamic inference", "adaptive inference",
                "speculative decoding", "parallel decoding", "batch inference",
                "kv cache", "kv caching", "attention optimization", 
                "dynamic batching", "continuous batching", "tensor parallelism",
                "adaptive attention", "conditional execution", "skip connections",
                "progressive inference", "multi-exit", "conditional computation",
                "spatial-temporal visual token trimming", "visual token trimming",
                "token reduction", "visual tokens withdrawal", "token merging",
                "attention head pruning", "layer skipping", "adaptive computation",
                "computational efficiency", "efficient attention", "sparse attention",
                "linear attention", "fast attention", "multi-branch self-drafting",
                "self-drafting", "adaptive step", "step selection", "adaptive guidance",
                "training-free acceleration", "sublayer skipping", "adaptive skip"
            ],
            "hardware_acceleration": [
                "gpu acceleration", "tpu optimization", "fpga implementation",
                "tensorcore", "cuda optimization", "hardware-aware optimization",
                "edge deployment", "mobile inference", "embedded inference",
                "hardware efficiency", "memory-efficient inference",
                "hardware-software co-design", "accelerator design", "on-device inference",
                "edge ai", "mobile ai", "real-time processing", "low-power inference"
            ],
            "frameworks_engines": [
                "tensorrt", "onnx runtime", "tvm", "openvino", "tensorflow lite",
                "pytorch mobile", "vllm", "triton inference", "tensorrt-llm",
                "deepspeed", "fastertransformer", "lightllm", "flash attention",
                "xformers", "optimum", "neural magic", "ort", "tensorrt inference"
            ],
            "diffusion_acceleration": [
                "diffusion acceleration", "fast diffusion", "few steps diffusion",
                "diffusion distillation", "diffusion optimization", "step selection",
                "adaptive step", "flash diffusion", "accelerating diffusion",
                "efficient diffusion", "rapid diffusion", "diffusion speedup",
                "acceleration of diffusion transformers", "diffusion pruning",
                "diffusion quantization", "diffusion caching", "lazy learning",
                "lazydit", "adaptive guidance", "fast generative", "generative acceleration"
            ],
            "multimodal_acceleration": [
                "extending mamba", "multimodal acceleration", "efficient multimodal",
                "accelerating multimodal", "multimodal optimization",
                "vision-language acceleration", "multimodal inference optimization",
                "cobra", "mamba", "state space model", "ssm", "linear complexity",
                "efficient mllm", "multimodal efficiency", "visual token withdrawal",
                "boosting multimodal", "efficient vlm", "lightweight multimodal",
                "elastic visual experts", "vision language optimization"
            ],
            "compression_efficiency": [
                "text compression", "lossless compression", "low-complexity",
                "compression ratio", "entropy coding", "learned compression",
                "data compression", "neural compression", "efficient compression",
                "compression performance", "compression algorithm", "video compression",
                "neural block compression", "bit-operation acceleration"
            ]
        }

        # 中等相关技术（权重：3分）
        self.medium_relevance_keywords = {
            "optimization_techniques": [
                "kernel fusion", "operator fusion", "graph optimization",
                "memory optimization", "computational optimization", "parameter efficiency",
                "flops reduction", "latency reduction", "throughput improvement",
                "runtime optimization", "performance optimization", "energy efficiency",
                "look-up table", "lut", "deformable", "multi-frame",
                "gradient checkpointing", "activation checkpointing", "cache optimization",
                "memory efficiency", "compute efficiency", "computational cost reduction",
                "resource optimization", "system optimization", "deployment optimization"
            ],
            "model_architectures": [
                "efficient transformer", "lightweight neural network", "mobile model",
                "compact architecture", "efficient architecture", 
                "binary neural network", "efficient attention", "linear attention",
                "lightweight transformer", "mobile transformer", "efficient convolution",
                "depthwise separable", "mobile nets", "efficient nets",
                "mamba", "state space model", "ssm", "linear complexity",
                "parallel optimal position search", "popos", "scalable model",
                "elastic model", "adaptive model", "dynamic model"
            ],
            "serving_deployment": [
                "model serving", "deployment optimization", "production deployment",
                "scalable inference", "high-throughput serving", "low-latency serving",
                "real-time deployment", "streaming inference",
                "online inference", "efficient deployment", "inference server",
                "scalable deployment", "cloud inference", "edge serving",
                "distributed inference", "parallel inference"
            ],
            "vision_language_efficiency": [
                "efficient vision language", "multimodal efficiency",
                "vision language optimization", "vlm efficiency",
                "efficient vlm", "lightweight multimodal", "fast multimodal",
                "efficient mllm", "multimodal inference", "visual token", 
                "adaptive multimodal", "multimodal compression", "cross-modal efficiency",
                "visual efficiency", "language model efficiency"
            ],
            "specialized_acceleration": [
                "privacy-utility-scalable", "offsite-tuning", "rank compression",
                "layer replace", "dynamic layer", "selective compression",
                "scalable optimization", "efficient tuning", "parameter efficiency",
                "fast training", "rapid training", "accelerated training", 
                "efficient learning", "fast adaptation", "quick adaptation",
                "parameter-efficient", "resource-efficient", "compute-efficient"
            ],
            "system_efficiency": [
                "communication-efficient", "bandwidth efficient", "network optimization",
                "distributed efficiency", "parallel efficiency", "concurrent processing",
                "asynchronous processing", "pipeline optimization", "batch optimization",
                "throughput optimization", "latency optimization", "response optimization",
                "scalable system", "high-performance system", "efficient system",
                "fast system", "optimized system", "streamlined system"
            ]
        }

        # 支撑关键词（权重：1分，增加权重） - 大幅减少通用词汇
        self.supporting_keywords = {
            "performance_metrics": [
                "inference latency", "inference throughput", "inference speed", 
                "model efficiency", "computational efficiency", "inference performance",
                "serving performance", "deployment performance", "acceleration ratio",
                "speedup ratio", "memory efficiency", "compute efficiency",
                "inference cost", "computational cost", "serving cost",
                "processing time", "response time", "latency reduction",
                "throughput improvement", "performance gain", "efficiency gain",
                "speed improvement", "acceleration gain", "optimization gain"
            ],
            "model_types": [
                "large language model", "transformer model", "neural network model",
                "foundation model", "generative model", "vision language model", 
                "multimodal model", "diffusion model", "state space model", 
                "diffusion transformer", "language model inference", "autoregressive model",
                "seq2seq model", "encoder-decoder model", "attention model",
                "transformer-based model", "neural architecture", "deep model"
            ],
            "specific_efficiency": [
                "fast inference", "rapid inference", "accelerated model",
                "optimized inference", "lightweight inference", "efficient model",
                "compressed model", "pruned model", "quantized model",
                "high-performance", "low-latency", "real-time",
                "efficient implementation", "optimized implementation",
                "fast implementation", "rapid implementation", "accelerated implementation",
                "streamlined implementation", "enhanced performance", "improved efficiency",
                "performance boost", "efficiency boost", "speed boost"
            ],
            "efficiency_terms": [
                "efficient", "effectiveness", "fast", "rapid", "quick", "swift",
                "accelerated", "optimized", "streamlined", "enhanced", "improved",
                "boosted", "upgraded", "advanced", "superior", "high-performance",
                "lightweight", "compact", "minimal", "reduced", "low-cost",
                "resource-saving", "time-saving", "cost-effective", "performance-oriented"
            ]
        }


class _KeywordMatcher:
    """关键词匹配器，负责匹配和评分逻辑"""
    
    def __init__(self):
        self._config = _KeywordConfig()
    
    def _match_keyword_category(self, title: str, abstract: str, keywords: Dict[str, List[str]], 
                               priority: str, weight: int) -> Tuple[List[Dict], int]:
        """匹配特定类别的关键词"""
        matched_keywords = []
        category_count = 0
        
        for category, category_keywords in keywords.items():
            for keyword in category_keywords:
                keyword_lower = keyword.lower()
                
                # 更精确的关键词匹配
                title_match = self._precise_keyword_match(title, keyword_lower)
                abstract_match = self._precise_keyword_match(abstract, keyword_lower)
                
                if title_match or abstract_match:
                    category_count += weight  # 使用权重而不是简单计数
                    matched_keywords.append({
                        'keyword': keyword,
                        'category': category,
                        'priority': priority,
                        'weight': weight,
                        'in_title': title_match,
                        'in_abstract': abstract_match
                    })
        
        return matched_keywords, category_count
    
    def _precise_keyword_match(self, text: str, keyword: str) -> bool:
        """精确的关键词匹配"""
        text_lower = text.lower()
        
        # 对于单个词的关键词，使用词边界匹配
        if ' ' not in keyword:
            import re
            pattern = r'\b' + re.escape(keyword) + r'\b'
            return bool(re.search(pattern, text_lower))
        else:
            # 对于短语，检查完整短语匹配
            return (f" {keyword} " in f" {text_lower} " or 
                   text_lower.startswith(f"{keyword} ") or 
                   text_lower.endswith(f" {keyword}") or
                   text_lower == keyword)
    
    def match_keywords(self, title: str, abstract: str) -> Dict:
        """执行关键词匹配"""
        # 安全处理输入参数
        title = title or ""
        abstract = abstract or ""
        title_lower = title.lower()
        abstract_lower = abstract.lower()
        
        # 匹配各类别关键词
        all_matched_keywords = []
        total_score = 0
        
        # 核心关键词匹配
        core_keywords, core_count = self._match_keyword_category(
            title_lower, abstract_lower, self._config.core_keywords, 'core', 6)
        all_matched_keywords.extend(core_keywords)
        
        # 高相关关键词匹配
        high_keywords, high_count = self._match_keyword_category(
            title_lower, abstract_lower, self._config.high_relevance_keywords, 'high', 5)
        all_matched_keywords.extend(high_keywords)
        
        # 中等相关关键词匹配
        medium_keywords, medium_count = self._match_keyword_category(
            title_lower, abstract_lower, self._config.medium_relevance_keywords, 'medium', 3)
        all_matched_keywords.extend(medium_keywords)
        
        # 支撑关键词匹配
        supporting_keywords, supporting_count = self._match_keyword_category(
            title_lower, abstract_lower, self._config.supporting_keywords, 'supporting', 1)
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
            'supporting_match_count': supporting_count
        }


class _MatchDecisionEngine:
    """匹配决策引擎，负责最终的匹配判断"""
    
    @staticmethod
    def should_match(match_result: Dict, threshold: int = 5) -> bool:
        """判断是否应该匹配"""
        total_score = match_result.get('keyword_count', 0)
        
        # 简单的阈值判断：权重大于等于5时匹配成功
        return total_score >= threshold


class _LLMJudge:
    """大模型判别器，使用豆包API进行论文相关性判断"""
    
    def __init__(self):
        # 公共前缀模板，包含标题和摘要信息
        self.common_prefix = """以下是一篇论文的基本信息：

标题：{title}

摘要：{abstract}

"""

        # 总结任务的具体指令
        self.summary_task = """请用中文一句话总结这篇论文的核心内容，尽可能包含论文的实验数据，不要添加其他说明。
"""

        # 相关性判断任务的具体指令
        self.relevance_task = """请判断这篇论文是否与AI推理加速相关。AI推理加速包括但不限于：
- 量化、剪枝、蒸馏、早退等模型压缩技术
- 动态推理、投机解码、采样优化等算法技术
- GPU/TPU/FPGA等硬件加速
- 模型并行推理优化，通信优化
- 推理引擎优化、serving优化
- 大模型推理优化、多模态推理加速
- 扩散模型加速、生成模型推理优化
- 高效模型结构的研究

请回答"相关"或"不相关"，并简要说明原因（不超过30个字）。格式：相关/不相关 - 原因"""

        # 翻译任务的具体指令  
        self.translation_task = """请将上述英文摘要翻译成中文，要求翻译准确、通顺、专业。

请直接给出中文翻译，不要添加其他说明。"""
    
    def get_summary(self, title: str, abstract: str) -> str:
        """获取论文的一句话总结"""
        try:
            # 使用公共前缀 + 任务特定指令
            full_prompt = self.common_prefix.format(title=title, abstract=abstract) + self.summary_task
            
            messages = [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
            
            result = call_doubao(messages)
            if result.get("success"):
                return result.get("content", "").strip()
            else:
                return f"总结生成失败: {result.get('error', '未知错误')}"
                
        except Exception as e:
            return f"总结生成出错: {str(e)}"
    
    def judge_relevance(self, title: str, abstract: str) -> str:
        """判断论文是否与AI推理加速相关"""
        try:
            # 使用公共前缀 + 任务特定指令
            full_prompt = self.common_prefix.format(title=title, abstract=abstract) + self.relevance_task
            
            messages = [
                {
                    "role": "user", 
                    "content": full_prompt
                }
            ]
            
            result = call_doubao(messages)
            if result.get("success"):
                return result.get("content", "").strip()
            else:
                return f"相关性判断失败: {result.get('error', '未知错误')}"
                
        except Exception as e:
            return f"相关性判断出错: {str(e)}"
    
    def translate_abstract(self, title: str, abstract: str) -> str:
        """将英文摘要翻译成中文"""
        try:
            # 使用公共前缀 + 任务特定指令，确保标题和摘要信息一致
            full_prompt = self.common_prefix.format(title=title, abstract=abstract) + self.translation_task
            
            messages = [
                {
                    "role": "user",
                    "content": full_prompt
                }
            ]
            
            result = call_doubao(messages)
            if result.get("success"):
                return result.get("content", "").strip()
            else:
                return f"翻译失败: {result.get('error', '未知错误')}"
                
        except Exception as e:
            return f"翻译出错: {str(e)}"


class _PaperAnalyzer:
    """论文分析器，处理单篇论文的分析逻辑"""
    
    def __init__(self, enable_llm_judge: bool = True):
        self._matcher = _KeywordMatcher()
        self._decision_engine = _MatchDecisionEngine()
        self._llm_judge = _LLMJudge() if enable_llm_judge else None
        self.enable_llm_judge = enable_llm_judge
    
    def analyze_paper(self, pdf_path: str) -> Optional[Dict]:
        """分析单篇论文，包含错误恢复机制"""
        filename = os.path.basename(pdf_path)
        
        try:
            # PDF提取阶段
            try:
                paper_info = extract_paper_abstract(pdf_path)
            except Exception as e:
                print(f"    ❌ PDF提取失败: {str(e)}")
                return None
            
            if not paper_info or not paper_info.get('title') or not paper_info.get('abstract'):
                print(f"    ❌ 无法提取有效的标题或摘要")
                return None
            
            # 关键词匹配阶段
            try:
                match_result = self._matcher.match_keywords(
                    paper_info['title'], paper_info['abstract']
                )
            except Exception as e:
                print(f"    ⚠️ 关键词匹配出错: {str(e)}，使用默认匹配结果")
                match_result = {
                    'matched_keywords': [],
                    'keyword_count': 0,
                    'title_keywords': [],
                    'abstract_keywords': [],
                    'core_match_count': 0,
                    'high_match_count': 0,
                    'medium_match_count': 0,
                    'supporting_match_count': 0
                }
            
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
                    'supporting_match_count': 0
                }
            
            # 判断是否匹配
            is_match = self._decision_engine.should_match(match_result)
            match_result['is_match'] = is_match
            
            # 初始化大模型判别结果
            llm_summary = ""
            llm_relevance = ""
            chinese_abstract = ""
            
            # 大模型判别阶段（可选且允许失败）
            if is_match and self.enable_llm_judge and self._llm_judge:
                print(f"    🤖 执行大模型判别...")
                
                # 总结生成
                try:
                    llm_summary = self._llm_judge.get_summary(
                        paper_info['title'], paper_info['abstract']
                    )
                    print(f"    ✓ 大模型总结: {llm_summary}")
                except Exception as e:
                    print(f"    ⚠️ 大模型总结失败: {str(e)}")
                    llm_summary = "大模型总结生成失败"
                
                # 相关性判断
                try:
                    llm_relevance = self._llm_judge.judge_relevance(
                        paper_info['title'], paper_info['abstract']
                    )
                    print(f"    ✓ 大模型相关性判断: {llm_relevance}")
                except Exception as e:
                    print(f"    ⚠️ 大模型相关性判断失败: {str(e)}")
                    llm_relevance = "大模型相关性判断失败"
                
                # 摘要翻译
                try:
                    print(f"    🌐 执行摘要翻译...")
                    chinese_abstract = self._llm_judge.translate_abstract(
                        paper_info['title'], paper_info['abstract']
                    )
                    print(f"    ✓ 翻译完成: {chinese_abstract[:100]}...")
                except Exception as e:
                    print(f"    ⚠️ 摘要翻译失败: {str(e)}")
                    chinese_abstract = "摘要翻译失败"
            
            # 将大模型判别结果和翻译结果添加到match_result中
            match_result['llm_summary'] = llm_summary
            match_result['llm_relevance'] = llm_relevance
            match_result['chinese_abstract'] = chinese_abstract
            
            paper_info['filename'] = filename
            paper_info['match_info'] = match_result
            
            return paper_info
            
        except Exception as e:
            print(f"    ❌ 分析论文时发生未预期错误: {str(e)}")
            return None


class _ReportGenerator:
    """报告生成器，负责生成各种输出报告，支持增量写入"""
    
    def __init__(self, output_dir: str):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建带时间戳的结果文件夹
        result_folder_name = f"ai_acceleration_analysis_{self.timestamp}"
        self.output_dir = os.path.join(output_dir, result_folder_name)
        
        # 确保输出目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"创建结果输出文件夹: {self.output_dir}")
        
        # 初始化增量写入文件
        self._init_incremental_files()
    
    def _init_incremental_files(self):
        """初始化增量写入的文件"""
        # AI相关论文文件
        self.ai_papers_txt_file = os.path.join(self.output_dir, "ai_inference_related_papers.txt")
        self.ai_papers_csv_file = os.path.join(self.output_dir, "ai_inference_related_papers.csv")
        
        # 非AI相关论文文件
        self.non_ai_papers_txt_file = os.path.join(self.output_dir, "non_ai_inference_papers.txt")
        self.non_ai_papers_csv_file = os.path.join(self.output_dir, "non_ai_inference_papers.csv")
        
        # 初始化文本文件
        with open(self.ai_papers_txt_file, 'w', encoding='utf-8') as f:
            f.write("AI推理加速相关论文列表\n")
            f.write("=" * 50 + "\n\n")
        
        with open(self.non_ai_papers_txt_file, 'w', encoding='utf-8') as f:
            f.write("非AI推理加速相关论文列表\n")
            f.write("=" * 50 + "\n\n")
        
        # 初始化CSV文件头
        ai_headers = [
            '序号', '标题', '文件名', '作者', '组织', '匹配分数',
            '核心关键字数', '高相关关键字数', '中等关键字数', '支撑关键字数',
            '标题关键字', '摘要关键字数量', '大模型总结', '大模型相关性判断', 
            '中文摘要翻译', '摘要预览'
        ]
        
        non_ai_headers = [
            '序号', '标题', '文件名', '作者', '组织', '匹配分数',
            '大模型总结', '大模型相关性判断', '中文摘要翻译', '摘要预览'
        ]
        
        with open(self.ai_papers_csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(ai_headers)
        
        with open(self.non_ai_papers_csv_file, 'w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(non_ai_headers)
        
        # 计数器，用于CSV序号
        self.ai_paper_count = 0
        self.non_ai_paper_count = 0
        
        # 进度文件，用于断点续传（如需要）
        self.progress_file = os.path.join(self.output_dir, "analysis_progress.txt")
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            f.write(f"分析开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("AI相关论文数: 0\n")
            f.write("非AI相关论文数: 0\n")
            f.write("处理失败数: 0\n")
    
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
                
            # 大模型判别结果
            if match_info.get('llm_summary'):
                lines.append(f"   大模型总结: {match_info['llm_summary']}")
            if match_info.get('llm_relevance'):
                lines.append(f"   大模型相关性判断: {match_info['llm_relevance']}")
            if match_info.get('chinese_abstract'):
                lines.append(f"   中文摘要翻译: {match_info['chinese_abstract']}")
        
        if paper['abstract']:
            abstract_preview = paper['abstract'][:2000] + "..." if len(paper['abstract']) > 2000 else paper['abstract']
            lines.append(f"   摘要: {abstract_preview}")
        
        return '\n'.join(lines)
    
    def append_ai_paper(self, paper: Dict):
        """增量写入单篇AI相关论文"""
        try:
            self.ai_paper_count += 1
            
            # 写入文本文件
            with open(self.ai_papers_txt_file, 'a', encoding='utf-8') as f:
                f.write(f"{self.ai_paper_count}. 标题: {paper['title']}\n")
                f.write(self._format_paper_info(paper))
                f.write("\n" + "-" * 80 + "\n\n")
            
            # 写入CSV文件
            self._append_ai_paper_csv(paper)
            
        except Exception as e:
            print(f"    警告: 写入AI相关论文失败: {str(e)}")
    
    def append_non_ai_paper(self, paper: Dict):
        """增量写入单篇非AI相关论文"""
        try:
            self.non_ai_paper_count += 1
            
            # 写入文本文件
            with open(self.non_ai_papers_txt_file, 'a', encoding='utf-8') as f:
                f.write(f"{self.non_ai_paper_count}. 标题: {paper['title']}\n")
                f.write(self._format_paper_info(paper, show_details=False))
                
                # 添加分数信息
                if 'match_info' in paper:
                    match_info = paper['match_info']
                    f.write(f"\n   匹配分数不足 (分数: {match_info['keyword_count']})")
                
                f.write("\n" + "-" * 80 + "\n\n")
            
            # 写入CSV文件
            self._append_non_ai_paper_csv(paper)
            
        except Exception as e:
            print(f"    警告: 写入非AI相关论文失败: {str(e)}")
    
    def _append_ai_paper_csv(self, paper: Dict):
        """增量写入AI相关论文到CSV"""
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
            abstract_preview = abstract_preview.replace('\n', ' ').replace('\r', ' ')
        
        # 中文摘要翻译
        chinese_abstract = match_info.get('chinese_abstract', '')
        if chinese_abstract:
            chinese_abstract = chinese_abstract.replace('\n', ' ').replace('\r', ' ')
        
        row = [
            self.ai_paper_count,
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
            match_info.get('llm_summary', ''),
            match_info.get('llm_relevance', ''),
            chinese_abstract,
            abstract_preview
        ]
        
        with open(self.ai_papers_csv_file, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def _append_non_ai_paper_csv(self, paper: Dict):
        """增量写入非AI相关论文到CSV"""
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
        
        # 摘要预览
        abstract_preview = ""
        if paper.get('abstract'):
            abstract_preview = paper['abstract'][:2000] + "..." if len(paper['abstract']) > 2000 else paper['abstract']
            abstract_preview = abstract_preview.replace('\n', ' ').replace('\r', ' ')
        
        # 中文摘要翻译
        chinese_abstract = match_info.get('chinese_abstract', '')
        if chinese_abstract:
            chinese_abstract = chinese_abstract.replace('\n', ' ').replace('\r', ' ')
        
        row = [
            self.non_ai_paper_count,
            paper.get('title', ''),
            paper.get('filename', ''),
            authors_str,
            organizations_str,
            match_info.get('keyword_count', 0),
            match_info.get('llm_summary', ''),
            match_info.get('llm_relevance', ''),
            chinese_abstract,
            abstract_preview
        ]
        
        with open(self.non_ai_papers_csv_file, 'a', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def update_progress(self, error_count: int = 0):
        """更新进度文件"""
        try:
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                f.write(f"最后更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"AI相关论文数: {self.ai_paper_count}\n")
                f.write(f"非AI相关论文数: {self.non_ai_paper_count}\n")
                f.write(f"处理失败数: {error_count}\n")
                f.write(f"总处理数: {self.ai_paper_count + self.non_ai_paper_count}\n")
        except:
            pass  # 忽略进度文件更新错误
    
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
                
                # 添加分数信息
                if 'match_info' in paper:
                    match_info = paper['match_info']
                    f.write(f"\n   匹配分数不足 (分数: {match_info['keyword_count']})")
                
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
                '标题关键字', '摘要关键字数量', '大模型总结', '大模型相关性判断', 
                '中文摘要翻译', '摘要预览'
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
                
                # 中文摘要翻译
                chinese_abstract = match_info.get('chinese_abstract', '')
                if chinese_abstract:
                    # 清理翻译中的换行符，避免CSV格式问题
                    chinese_abstract = chinese_abstract.replace('\n', ' ').replace('\r', ' ')
                
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
                    match_info.get('llm_summary', ''),
                    match_info.get('llm_relevance', ''),
                    chinese_abstract,
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
                '大模型总结', '大模型相关性判断', '中文摘要翻译', '摘要预览'
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
                
                # 摘要预览
                abstract_preview = ""
                if paper.get('abstract'):
                    abstract_preview = paper['abstract'][:2000] + "..." if len(paper['abstract']) > 2000 else paper['abstract']
                    # 清理摘要中的换行符，避免CSV格式问题
                    abstract_preview = abstract_preview.replace('\n', ' ').replace('\r', ' ')
                
                # 中文摘要翻译（非AI相关论文通常为空）
                chinese_abstract = match_info.get('chinese_abstract', '')
                if chinese_abstract:
                    # 清理翻译中的换行符，避免CSV格式问题
                    chinese_abstract = chinese_abstract.replace('\n', ' ').replace('\r', ' ')
                
                row = [
                    i,
                    paper.get('title', ''),
                    paper.get('filename', ''),
                    authors_str,
                    organizations_str,
                    match_info.get('keyword_count', 0),
                    match_info.get('llm_summary', ''),
                    match_info.get('llm_relevance', ''),
                    chinese_abstract,
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
    
    def finalize_reports(self, ai_papers: List[Dict], non_ai_papers: List[Dict]):
        """生成最终的统计报告"""
        try:
            # 生成统计报告
            if ai_papers:
                statistics_file = os.path.join(self.output_dir, "match_statistics.txt")
                self.generate_statistics(ai_papers, "match_statistics.txt")
                
                statistics_csv_file = os.path.join(self.output_dir, "match_statistics.csv")
                self.save_statistics_csv(ai_papers, "match_statistics.csv")
            
            # 生成汇总信息文件
            summary_file = os.path.join(self.output_dir, "analysis_summary.txt")
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("AI推理加速论文分析汇总\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总处理论文数: {len(ai_papers) + len(non_ai_papers)}\n")
                f.write(f"AI推理加速相关论文: {len(ai_papers)}\n")
                f.write(f"非AI推理加速相关论文: {len(non_ai_papers)}\n\n")
                
                if ai_papers:
                    f.write("AI相关论文列表:\n")
                    f.write("-" * 30 + "\n")
                    for i, paper in enumerate(ai_papers, 1):
                        match_score = paper.get('match_info', {}).get('keyword_count', 0)
                        f.write(f"{i}. {paper['title']} (匹配分数: {match_score})\n")
                    f.write("\n")
                
                f.write("文件说明:\n")
                f.write("-" * 30 + "\n")
                f.write("1. ai_inference_related_papers.txt - AI相关论文详细信息\n")
                f.write("2. ai_inference_related_papers.csv - AI相关论文CSV格式\n")
                f.write("3. non_ai_inference_papers.txt - 非AI相关论文详细信息\n")
                f.write("4. non_ai_inference_papers.csv - 非AI相关论文CSV格式\n")
                f.write("5. match_statistics.txt - 匹配统计报告\n")
                f.write("6. match_statistics.csv - 匹配统计CSV格式\n")
                f.write("7. analysis_summary.txt - 本汇总文件\n")
            
            print(f"分析汇总报告已保存到 {summary_file}")
            
        except Exception as e:
            print(f"警告: 生成最终统计报告时出错: {str(e)}")


class AiAccelerationExtractor:
    """AI加速论文提取器主类"""
    
    def __init__(self, papers_dir: str, output_dir: str = ".", enable_llm_judge: bool = True):
        """
        初始化AI加速论文提取器
        
        Args:
            papers_dir: 论文PDF文件所在目录
            output_dir: 输出文件保存目录
            enable_llm_judge: 是否启用大模型判别功能
        """
        self.papers_dir = papers_dir
        self.output_dir = output_dir
        self.enable_llm_judge = enable_llm_judge
        
        # 确保输出目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self._analyzer = _PaperAnalyzer(enable_llm_judge)
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
    
    def _print_analysis_progress(self, paper: Dict, processed_count: int, total_count: int, saved_to_disk: bool = False):
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
            
            # 显示大模型判别结果
            if self.enable_llm_judge:
                if match_info.get('llm_summary'):
                    print(f"    大模型总结: {match_info['llm_summary']}")
                if match_info.get('llm_relevance'):
                    print(f"    大模型相关性: {match_info['llm_relevance']}")
            
            if saved_to_disk:
                print(f"    ✓ 已保存到文件")
        else:
            print(f"  - 非AI推理加速相关论文 (匹配分数: {match_info['keyword_count']})")
            if saved_to_disk:
                print(f"    ✓ 已保存到文件")
    
    def _analyze_papers(self, paper_filenames: List[str] = None, analyze_all: bool = True) -> Dict[str, List[Dict]]:
        """
        分析论文，筛选出与AI推理加速相关和非相关的论文，支持增量写入和错误恢复
        
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
        
        # 分析论文并增量写入
        ai_related_papers = []
        non_ai_related_papers = []
        processed_count = 0
        error_count = 0
        
        paper_type = "论文" if analyze_all else "指定论文"
        print(f"\n开始分析 {len(files_to_analyze)} 个{paper_type}...")
        print(f"分析结果将实时保存到: {self._report_generator.output_dir}")
        
        for i, filename in enumerate(files_to_analyze, 1):
            saved_to_disk = False
            try:
                pdf_path = os.path.join(self.papers_dir, filename)
                paper_info = self._analyzer.analyze_paper(pdf_path)
                
                if paper_info is None:
                    print(f"  警告: 无法提取标题或摘要，跳过 {filename}")
                    error_count += 1
                    continue
                
                # 立即写入磁盘
                if paper_info['match_info']['is_match']:
                    ai_related_papers.append(paper_info)
                    self._report_generator.append_ai_paper(paper_info)
                    saved_to_disk = True
                else:
                    non_ai_related_papers.append(paper_info)
                    self._report_generator.append_non_ai_paper(paper_info)
                    saved_to_disk = True
                
                processed_count += 1
                
                # 打印进度信息
                self._print_analysis_progress(paper_info, i, len(files_to_analyze), saved_to_disk)
                
                # 每处理10篇论文输出一次进度总结并更新进度文件
                if processed_count % 10 == 0:
                    self._report_generator.update_progress(error_count)
                    print(f"\n📊 进度总结 ({processed_count}/{len(files_to_analyze)} 已处理):")
                    print(f"    ✅ 成功处理: {processed_count} 篇")
                    print(f"    ❌ 处理失败: {error_count} 篇") 
                    print(f"    🎯 AI相关: {len(ai_related_papers)} 篇")
                    print(f"    📄 其他: {len(non_ai_related_papers)} 篇")
                    print(f"    💾 所有结果已实时保存到磁盘\n")
                
            except Exception as e:
                error_count += 1
                print(f"  ❌ 错误: 处理文件 {filename} 时出错: {str(e)}")
                print(f"    将跳过此文件继续处理下一个...")
                
                # 写入错误日志并更新进度
                try:
                    error_log_file = os.path.join(self._report_generator.output_dir, "error_log.txt")
                    with open(error_log_file, 'a', encoding='utf-8') as f:
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - 处理文件 {filename} 失败: {str(e)}\n")
                    self._report_generator.update_progress(error_count)
                except:
                    pass  # 忽略日志写入错误
                
                continue
        
        # 最终进度更新
        self._report_generator.update_progress(error_count)
        
        # 打印最终总结
        print(f"\n🎉 分析完成!")
        print(f"📁 总文件数: {len(files_to_analyze)}")
        print(f"✅ 成功处理: {processed_count} 篇")
        print(f"❌ 处理失败: {error_count} 篇")
        print(f"🎯 AI推理加速相关论文: {len(ai_related_papers)} 篇")
        print(f"📄 非AI推理加速相关论文: {len(non_ai_related_papers)} 篇")
        print(f"💾 所有结果已保存到: {self._report_generator.output_dir}")
        
        return {
            'ai_related': ai_related_papers,
            'non_ai_related': non_ai_related_papers
        }
    
    def parse(self, paper_filenames: List[str] = None, analyze_all: bool = True,
              output_format: str = "both"):
        """
        解析论文并保存结果
        
        Args:
            paper_filenames: 要分析的论文文件名列表
            analyze_all: 是否分析全量论文
            output_format: 输出格式，可选 "txt", "csv", "both"
        """
        print(f"\n🚀 开始AI推理加速论文分析...")
        print(f"📁 分析目录: {self.papers_dir}")
        print(f"📤 输出基础目录: {self.output_dir}")
        print(f"📤 结果保存目录: {self._report_generator.output_dir}")
        print(f"📊 输出格式: {output_format}")
        print(f"🎯 匹配阈值: 权重>=5分即匹配成功")
        print(f"🔍 匹配逻辑: 纯关键词权重匹配，无排除机制")
        print(f"🤖 大模型判别: {'启用' if self.enable_llm_judge else '禁用'}")
        if self.enable_llm_judge:
            print(f"    对于初筛相关的论文，将调用豆包API进行总结、相关性判断和摘要翻译")
        
        # 分析论文（结果已在分析过程中实时写入磁盘）
        results = self._analyze_papers(paper_filenames, analyze_all)
        
        ai_papers = results['ai_related']
        non_ai_papers = results['non_ai_related']
        
        # 生成最终统计报告（论文详情已在分析过程中写入）
        print(f"\n📊 生成最终统计报告...")
        self._report_generator.finalize_reports(ai_papers, non_ai_papers)
        
        if ai_papers:
            # 打印简要统计
            print(f"\n✨ 发现的AI推理加速相关论文:")
            for i, paper in enumerate(ai_papers, 1):
                match_score = paper.get('match_info', {}).get('keyword_count', 0)
                print(f"{i}. {paper['title']} (匹配分数: {match_score})")
        else:
            print("\n📭 未发现AI推理加速相关论文")
        
        if non_ai_papers:
            print(f"\n📄 非AI推理加速相关论文: {len(non_ai_papers)} 篇")
        else:
            print("\n🎯 所有论文都与AI推理加速相关")
        
        # 打印总结信息
        print(f"\n🎉 所有任务完成!")
        print(f"📂 所有结果文件已保存到: {self._report_generator.output_dir}")
        print(f"🔍 共处理 {len(ai_papers) + len(non_ai_papers)} 篇论文")
        print(f"✨ AI推理加速相关: {len(ai_papers)} 篇")
        print(f"📄 其他论文: {len(non_ai_papers)} 篇")
        print(f"💾 所有结果在处理过程中已实时保存，即使出现中断也不会丢失数据")


def ai_acceleration_parse(papers_dir: str, output_dir: str = ".", 
                         paper_filenames: List[str] = None, analyze_all: bool = True,
                         output_format: str = "both", enable_llm_judge: bool = True):
    """
    对外提供的AI推理加速论文解析函数
    
    Args:
        papers_dir: 论文PDF文件所在目录
        output_dir: 输出文件保存目录，默认为当前目录
        paper_filenames: 要分析的论文文件名列表，如果为None且analyze_all为True则分析所有论文
        analyze_all: 是否分析papers_dir下的全量论文，默认为False
        output_format: 输出格式，可选 "txt"（默认）, "csv", "both"
        enable_llm_judge: 是否启用大模型判别功能，默认为True
    
    Returns:
        None
    """
    extractor = AiAccelerationExtractor(papers_dir, output_dir, enable_llm_judge)
    extractor.parse(paper_filenames, analyze_all, output_format)
   