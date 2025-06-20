# AI推理加速论文筛选器

> 🤖 基于多层次关键词匹配算法的AI推理加速相关论文智能筛选工具

## 🎯 功能概述

AI推理加速论文筛选器是一个专门用于从大量学术论文中识别和筛选AI推理加速相关研究的智能工具。它采用先进的多层次关键词匹配算法，能够准确识别与AI推理优化、模型加速、量化压缩等相关的研究论文。

### 核心功能
- 🔍 **智能筛选**：基于分层关键词匹配的高精度论文筛选
- 📊 **多格式输出**：支持CSV、TXT等多种输出格式
- 📈 **详细统计**：提供完整的匹配统计和分析报告
- ⚡ **高效处理**：支持批量处理和增量分析

## 📁 文件结构

```
ai_acceleration_extractor/
├── __init__.py                      # 包初始化
├── ai_acceleration_extractor.py     # 主要筛选器实现
└── README_CSV.md                    # 本文档
```

## 🚀 快速开始

### 前置条件

1. **Grobid服务运行**（必需）：
   ```bash
   docker run --rm -it -p 8070:8070 lfoppiano/grobid:0.8.3
   ```

2. **Python依赖安装**：
   ```bash
   pip install requests beautifulsoup4 grobid-client
   ```

### 基本用法

#### 1. 类接口（推荐）

```python
from utils.ai_acceleration_extractor import AiAccelerationExtractor

# 创建筛选器实例
extractor = AiAccelerationExtractor(
    papers_dir="path/to/papers",
    output_dir="output"
)

# 分析所有论文
extractor.parse(analyze_all=True, output_format="csv")

# 分析指定论文列表
paper_files = ["paper1.pdf", "paper2.pdf"]
extractor.parse(paper_filenames=paper_files, output_format="both")
```

#### 2. 函数接口

```python
from utils.ai_acceleration_extractor import ai_acceleration_parse

# 只输出CSV格式
ai_acceleration_parse(
    papers_dir="path/to/papers", 
    output_dir="output",
    analyze_all=True,
    output_format="csv"
)

# 同时输出TXT和CSV格式
ai_acceleration_parse(
    papers_dir="path/to/papers",
    output_dir="output", 
    analyze_all=True,
    output_format="both"
)
```

## 🔍 筛选算法详解

### 关键词分类体系

筛选器采用四层关键词分类体系，每层具有不同的权重：

#### 1. 核心推理加速关键词（权重：5分）
```python
# 推理优化
"inference acceleration", "inference optimization", "inference speedup"

# 量化压缩
"quantization", "pruning", "model compression", "int8", "fp16"

# 知识蒸馏
"knowledge distillation", "model distillation", "teacher-student"

# 脉冲神经网络
"spiking neural network", "snn", "neuromorphic"
```

#### 2. 高相关技术（权重：4分）
```python
# 推理技术
"early exit", "dynamic inference", "speculative decoding"

# 硬件加速
"gpu acceleration", "tensorrt", "edge deployment"

# 框架引擎
"vllm", "triton inference", "deepspeed"

# 扩散模型加速
"diffusion acceleration", "fast diffusion"
```

#### 3. 中等相关技术（权重：3分）
```python
# 优化技术
"kernel fusion", "memory efficiency", "flops reduction"

# 模型架构
"efficient transformer", "lightweight neural network"

# 多模态效率
"efficient multimodal", "vision language optimization"
```

#### 4. 支撑关键词（权重：2分）
```python
# 性能指标
"latency", "throughput", "efficiency", "performance"

# 模型类型
"llm", "transformer", "diffusion model"

# 效率指标
"efficient", "lightweight", "fast", "accelerated"
```

### 匹配策略

#### 包含策略
- **阈值判断**：总匹配分数 ≥ 6分的论文被标记为AI推理加速相关
- **权重计算**：不同类别关键词按权重累计得分
- **位置加权**：标题中的关键词权重高于摘要

#### 排除策略
- **强排除**：包含训练、理论研究等强排除关键词直接排除
- **弱排除**：多个弱排除关键词累计时进行额外扣分

#### 决策引擎
```python
def should_match(match_result, threshold=6):
    # 检查强排除条件
    if match_result['strong_exclusion'] > 0:
        return False
    
    # 计算总分
    total_score = match_result['keyword_count']
    
    # 检查阈值
    return total_score >= threshold
```

## 📊 输出文件说明

### CSV文件结构

#### 1. AI推理加速相关论文 (`ai_inference_related_papers.csv`)

| 列名 | 说明 |
|------|------|
| 序号 | 论文序号 |
| 标题 | 论文标题 |
| 文件名 | PDF文件名 |
| 作者 | 作者列表（分号分隔） |
| 组织 | 作者所属组织（分号分隔） |
| 匹配分数 | 关键词匹配总分 |
| 核心关键字数 | 核心关键词匹配数量 |
| 高相关关键字数 | 高相关关键词匹配数量 |
| 中等关键字数 | 中等相关关键词匹配数量 |
| 支撑关键字数 | 支撑关键词匹配数量 |
| 标题关键字 | 标题中匹配的关键词（分号分隔） |
| 摘要关键字数量 | 摘要中匹配的关键词数量 |
| 弱排除关键字数 | 弱排除关键词数量 |
| 摘要预览 | 摘要前200字符 |

#### 2. 非AI推理加速相关论文 (`non_ai_inference_papers.csv`)

| 列名 | 说明 |
|------|------|
| 序号 | 论文序号 |
| 标题 | 论文标题 |
| 文件名 | PDF文件名 |
| 作者 | 作者列表（分号分隔） |
| 组织 | 作者所属组织（分号分隔） |
| 匹配分数 | 关键词匹配总分 |
| 排除原因 | 被排除的原因 |
| 摘要预览 | 摘要前200字符 |

#### 3. 匹配统计报告 (`match_statistics.csv`)

| 列名 | 说明 |
|------|------|
| 统计类别 | 统计的类别（总体、优先级、类别、关键字） |
| 项目 | 具体的统计项目 |
| 数量 | 统计数量 |

## 数据处理建议

### Excel中的使用

1. 打开CSV文件时选择UTF-8编码以正确显示中文
2. 使用数据筛选功能按匹配分数排序
3. 可以根据核心关键字数等列进行条件格式化

### 数据分析

CSV格式便于进行以下分析：
- 按匹配分数排序找出最相关的论文
- 统计不同组织的研究分布
- 分析关键词出现频率
- 对比不同类别关键词的分布

## 🔧 高级用法

### 自定义关键词筛选

```python
from utils.utils import is_ai_acceleration_paper

# 使用通用筛选函数进行单论文评估
title = "Fast Inference for Transformer Models"
abstract = "This paper presents novel acceleration techniques..."
result = is_ai_acceleration_paper(title, abstract, threshold=6)

print(f"匹配结果: {result['is_match']}")
print(f"匹配分数: {result['keyword_count']}")
print(f"匹配关键词: {result['matched_keywords']}")
```

### 批量处理工作流

```python
import os
from utils.ai_acceleration_extractor import AiAccelerationExtractor

def batch_analysis_workflow(papers_dir):
    """完整的批量分析工作流"""
    extractor = AiAccelerationExtractor(papers_dir, "results")
    
    # 1. 先进行快速分析（仅检查已有XML缓存）
    print("阶段1: 快速分析...")
    extractor.parse(analyze_all=True, output_format="csv")
    
    # 2. 生成详细报告
    print("阶段2: 生成详细报告...")
    extractor.parse(analyze_all=True, output_format="both")
    
    return "results"

# 使用示例
results_dir = batch_analysis_workflow("AAAI-2025-Papers")
print(f"分析结果保存在: {results_dir}")
```

### 增量分析

```python
# 对新增论文进行增量分析
new_papers = ["new_paper1.pdf", "new_paper2.pdf"]
extractor = AiAccelerationExtractor("papers", "output")

# 只分析新增的论文
extractor.parse(
    paper_filenames=new_papers,
    analyze_all=False,
    output_format="csv"
)
```

### 结果后处理

```python
import pandas as pd

# 读取CSV结果进行进一步分析
df = pd.read_csv("ai_inference_related_papers.csv")

# 按匹配分数排序
top_papers = df.sort_values("匹配分数", ascending=False).head(10)

# 统计不同组织的研究分布
org_stats = df["组织"].value_counts()

# 分析关键词分布
keyword_analysis = df["标题关键字"].str.split(";").explode().value_counts()
```

## 🔧 故障排除

### 常见问题

1. **Grobid服务连接失败**
   ```bash
   # 检查Grobid服务状态
   curl http://localhost:8070
   
   # 重启Grobid服务
   docker restart grobid_container
   ```

2. **PDF处理速度慢**
   ```python
   # 检查是否有XML缓存文件
   import os
   pdf_files = [f for f in os.listdir("papers") if f.endswith(".pdf")]
   xml_files = [f for f in os.listdir("papers") if f.endswith(".grobid.tei.xml")]
   print(f"PDF文件: {len(pdf_files)}, 缓存文件: {len(xml_files)}")
   ```

3. **匹配结果不准确**
   - 调整匹配阈值（默认6分）
   - 检查关键词配置
   - 验证PDF文本提取质量

### 性能优化

1. **提高处理速度**
   - 确保Grobid服务运行在本地
   - 使用SSD存储提高I/O性能
   - 利用XML缓存避免重复处理

2. **内存优化**
   - 分批处理大量文件
   - 及时释放未使用的对象
   - 监控内存使用情况

3. **准确性调优**
   ```python
   # 调整匹配阈值
   extractor = AiAccelerationExtractor("papers", "output")
   
   # 使用更严格的阈值
   results = extractor._analyze_papers(analyze_all=True)
   strict_matches = [p for p in results['ai_papers'] if p['match_result']['keyword_count'] >= 8]
   ```

## 💡 使用建议

### 最佳实践

1. **分阶段处理**
   - 先处理小样本验证配置
   - 逐步扩大处理范围
   - 定期检查结果质量

2. **结果验证**
   - 人工抽查高分匹配结果
   - 分析低分被排除的论文
   - 持续优化关键词配置

3. **数据管理**
   - 定期备份分析结果
   - 保留XML缓存文件
   - 建立版本控制机制

### 注意事项

1. **文件格式**
   - CSV文件使用UTF-8编码，确保中文字符正确显示
   - 多个值使用分号(;)分隔，避免与CSV格式的逗号冲突
   - 摘要中的换行符已被替换为空格，保持CSV格式完整性

2. **参数配置**
   - 如果同时需要TXT和CSV格式，使用`output_format="both"`参数
   - 根据具体需求调整匹配阈值
   - 考虑计算资源和时间成本选择分析范围 