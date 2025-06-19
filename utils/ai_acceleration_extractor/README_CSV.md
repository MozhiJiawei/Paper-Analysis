# AI推理加速论文提取器 - CSV格式输出说明

## 功能概述

AI推理加速论文提取器现在支持多种输出格式，包括传统的TXT格式和新增的CSV格式。CSV格式便于在Excel、Google Sheets等表格软件中进行数据分析和处理。

## 使用方法

### 1. 基本用法

```python
from utils.ai_acceleration_extractor import ai_acceleration_parse

# 只输出TXT格式（默认）
ai_acceleration_parse(
    papers_dir="path/to/papers",
    output_dir="output",
    analyze_all=True,
    output_format="txt"
)

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



## 输出文件说明

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

## 注意事项

1. CSV文件使用UTF-8编码，确保中文字符正确显示
2. 多个值使用分号(;)分隔，避免与CSV格式的逗号冲突
3. 摘要中的换行符已被替换为空格，保持CSV格式完整性
4. 如果同时需要TXT和CSV格式，使用`output_format="both"`参数 