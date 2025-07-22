"""
AI推理加速关键词配置文件
"""

from typing import Dict, List

# 核心推理加速关键词（最高权重：6分）
CORE_KEYWORDS = {
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
HIGH_RELEVANCE_KEYWORDS = {
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
MEDIUM_RELEVANCE_KEYWORDS = {
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

# 支撑关键词（权重：1分）
SUPPORTING_KEYWORDS = {
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

# 关键词权重配置
KEYWORD_WEIGHTS = {
    "core": 6,
    "high": 5,
    "medium": 3,
    "supporting": 1
}

# 匹配阈值配置
MATCH_THRESHOLD = 5

# LLM提示词配置
LLM_PROMPTS = {
    "common_prefix": """以下是一篇论文的基本信息：

标题：{title}

摘要：{abstract}

""",
    "summary_task": """请用中文一句话总结这篇论文的核心内容，尽可能包含论文的实验数据，不要添加其他说明。
""",
    "relevance_task": """请判断这篇论文是否与AI推理加速相关。AI推理加速包括但不限于：
- 量化、剪枝、蒸馏、早退等模型压缩技术
- 动态推理、投机解码、采样优化等算法技术
- GPU/TPU/FPGA等硬件加速
- 模型并行推理优化，通信优化
- 推理引擎优化、serving优化
- 大模型推理优化、多模态推理加速
- 扩散模型加速、生成模型推理优化
- 高效模型结构的研究

请回答"相关"或"不相关"，并简要说明原因（不超过30个字）。格式：相关/不相关 - 原因""",
    "translation_task": """请将上述英文摘要翻译成中文，要求翻译准确、通顺、专业。

请直接给出中文翻译，不要添加其他说明。"""
} 