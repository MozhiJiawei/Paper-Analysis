def is_ai_acceleration_paper(title, abstract, threshold=6):
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
