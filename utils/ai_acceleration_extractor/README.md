# AI推理加速论文提取器

一个专门用于从学术论文中提取和筛选AI推理加速相关论文的Python模块。

## 📋 功能特性

- **多输入模式支持**: 支持PDF文件和paper_copilot数据两种输入模式
- **智能关键词匹配**: 基于多层次关键词权重系统进行精确匹配
- **大模型判别**: 集成豆包API进行论文相关性判断和摘要翻译
- **详细分析报告**: 生成HTML和JSON格式的详细分析报告
- **灵活配置**: 支持自定义匹配阈值、输出格式等参数
- **进度跟踪**: 实时显示分析进度和匹配结果

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 基本使用

#### 1. PDF论文分析

```python
from utils.ai_acceleration_extractor import ai_acceleration_parse

# 分析PDF论文
ai_acceleration_parse(
    papers_dir="./papers",           # PDF文件目录
    output_dir="./results",          # 输出目录
    enable_llm_judge=True,          # 启用大模型判别
    match_threshold=5               # 匹配阈值
)
```

#### 2. paper_copilot数据分析

```python
from utils.ai_acceleration_extractor import ai_acceleration_parse_paper_copilot

# 准备paper_copilot数据
paper_infos = [
    {
        "title": "Efficient Inference Acceleration for Large Language Models",
        "author": "John Doe; Jane Smith",
        "aff": "Stanford University; MIT",
        "email": "john@stanford.edu; jane@mit.edu",
        "abstract": "This paper presents a novel approach for accelerating inference...",
        "filename": "test_paper_1"
    }
]

# 分析paper_copilot数据
ai_acceleration_parse_paper_copilot(
    paper_infos=paper_infos,
    output_dir="./results",
    enable_llm_judge=True,
    match_threshold=5
)
```

#### 3. 高级使用

```python
from utils.ai_acceleration_extractor import AiAccelerationExtractor

# 创建提取器实例
extractor = AiAccelerationExtractor(
    papers_dir="./papers",
    output_dir="./results",
    enable_llm_judge=True,
    match_threshold=5,
    analysis_mode="pdf"
)

# 分析特定文件
extractor.parse(paper_filenames=["paper1.pdf", "paper2.pdf"])

# 分析所有文件
extractor.parse(analyze_all=True)
```

## 📊 关键词匹配系统

### 核心关键词（权重：6分）
- **推理优化**: inference acceleration, inference optimization, model acceleration
- **量化压缩**: quantization, pruning, model compression, weight pruning
- **知识蒸馏**: knowledge distillation, model distillation, teacher-student
- **脉冲神经网络**: spiking neural network, SNN, neuromorphic
- **专家混合**: mixture of experts, MoE, sparse MoE

### 高相关关键词（权重：5分）
- **推理技术**: early exit, dynamic inference, speculative decoding
- **硬件加速**: GPU acceleration, TPU optimization, edge deployment
- **框架引擎**: TensorRT, ONNX Runtime, TVM, vLLM
- **扩散加速**: diffusion acceleration, fast diffusion, step selection

### 中等相关关键词（权重：3分）
- **模型架构**: transformer optimization, attention optimization
- **部署优化**: model serving, production deployment
- **性能评估**: latency measurement, throughput analysis

## ⚙️ 配置选项

### 主要参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `papers_dir` | str | None | PDF文件目录路径 |
| `output_dir` | str | "." | 输出文件保存目录 |
| `enable_llm_judge` | bool | True | 是否启用大模型判别 |
| `match_threshold` | int | 5 | 关键词匹配阈值 |
| `analysis_mode` | str | "pdf" | 分析模式："pdf"或"paper_copilot" |

### 输出格式

支持以下输出格式：
- `"html"`: 仅生成HTML报告
- `"json"`: 仅生成JSON报告  
- `"both"`: 生成HTML和JSON报告（默认）

## 📁 输出文件结构

```
results/
├── ai_acceleration_analysis_YYYYMMDD_HHMMSS/
│   ├── analysis_report.html          # HTML格式分析报告
│   ├── analysis_result.json          # JSON格式分析结果
│   ├── matched_papers.json          # 匹配论文列表
│   ├── unmatched_papers.json        # 未匹配论文列表
│   └── summary_statistics.json      # 统计摘要
```

## 🔧 组件架构

### 核心组件

- **AiAccelerationExtractor**: 主提取器类
- **KeywordMatcher**: 关键词匹配器
- **LLMJudge**: 大模型判别器
- **PaperAnalyzer**: 论文分析器
- **ReportGenerator**: 报告生成器

### 数据模型

- **PaperInfo**: 论文信息模型
- **AnalysisResult**: 分析结果模型
- **AnalysisConfig**: 分析配置模型
- **ProgressInfo**: 进度信息模型

## 📝 使用示例

### 示例1: 基本PDF分析

```python
from utils.ai_acceleration_extractor import ai_acceleration_parse

# 分析PDF论文
ai_acceleration_parse(
    papers_dir="./test_pdfs",
    output_dir="./results",
    enable_llm_judge=False,  # 禁用LLM以简化示例
    match_threshold=5
)
```

### 示例2: 组件级使用

```python
from utils.ai_acceleration_extractor import KeywordMatcher, LLMJudge

# 使用关键词匹配器
matcher = KeywordMatcher(threshold=5)
result = matcher.match_keywords(
    "AI Inference Acceleration",
    "This paper focuses on accelerating AI inference..."
)
print(f"匹配结果: {result.is_match}, 分数: {result.keyword_count}")

# 使用LLM判别器
judge = LLMJudge()
# 需要配置API密钥才能实际使用
```

### 示例3: 自定义配置

```python
from utils.ai_acceleration_extractor import AiAccelerationExtractor

# 创建自定义配置的提取器
extractor = AiAccelerationExtractor(
    papers_dir="./papers",
    output_dir="./custom_results",
    enable_llm_judge=True,
    match_threshold=7,  # 提高匹配阈值
    analysis_mode="pdf"
)

# 分析特定文件
extractor.parse(
    paper_filenames=["important_paper1.pdf", "important_paper2.pdf"],
    output_format="html"
)
```

## 🛠️ 异常处理

模块提供了完整的异常处理机制：

- **AIAccelerationExtractorError**: 基础异常类
- **PaperExtractionError**: 论文提取异常
- **KeywordMatchingError**: 关键词匹配异常
- **LLMJudgmentError**: 大模型判别异常
- **FileOperationError**: 文件操作异常
- **ConfigurationError**: 配置错误异常
- **ValidationError**: 数据验证异常

## 📈 性能优化

- **批量处理**: 支持批量分析多个PDF文件
- **内存管理**: 优化大文件处理的内存使用
- **并行处理**: 支持多线程处理（待实现）
- **缓存机制**: 关键词匹配结果缓存

## 🔍 调试和日志

模块提供详细的进度信息和调试输出：

```
🚀 开始AI推理加速论文分析...
📁 分析目录: ./papers
📤 输出基础目录: ./results
🎯 匹配阈值: 权重>=5分即匹配成功
🔍 匹配逻辑: 纯关键词权重匹配，无排除机制
🤖 大模型判别: 启用

正在处理 (1/10): paper1.pdf
  ✓ 发现AI推理加速相关论文: Efficient Inference Acceleration...
    标题关键字: inference, acceleration
    摘要关键字数量: 8
    总匹配分数: 15
    关键字分布 - 核心:3, 高相关:5, 中等:7
```

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 项目主页: [GitHub Repository]
- 问题反馈: [Issues]
- 邮箱: [your-email@example.com]

---

**版本**: 2.0.0  
**作者**: AI Research Team  
**最后更新**: 2024年12月 