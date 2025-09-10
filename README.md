# Paper-Analysis

一个专门用于学术论文分析的综合性工具集，主要用于从各大顶级AI会议（AAAI、ICLR、ICML、NAACL等）论文中识别和分析AI推理加速相关研究。

## 运行依赖

### 1. Grobid服务器

本项目需要Grobid服务来进行PDF论文内容提取：

```bash
# 使用Docker启动Grobid服务（推荐）
docker run --rm -it -p 8070:8070 lfoppiano/grobid:0.8.3

# 或后台运行
docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.3
```

### 2. 豆包API配置

本项目采用豆包（Doubao）作为大模型API服务：

1. **复制配置文件模板**：
   ```bash
   cd utils/doubao_api
   cp config_template.yaml config.yaml
   ```

2. **编辑配置文件**：
   ```yaml
   doubao:
     api_key: "YOUR_API_KEY_HERE"  # 请替换为您的实际API密钥
     base_url: "https://ark.cn-beijing.volces.com/api/v3"
     model: "doubao-seed-1-6-250615"
   ```

### 3. Python依赖包

```bash
pip install -r requirements.txt
```

主要依赖包括：
- grobid-client-python（PDF处理）
- beautifulsoup4（HTML/XML解析）
- requests（网络请求）
- pandas（数据处理）
- PyYAML（配置文件）
- volcengine-python-sdk[ark]（豆包API）
- arxiv（arXiv论文API）

## 功能简介

### 核心功能
- **AI推理加速论文识别**：基于关键词权重匹配和大模型判别，从海量论文中筛选出AI推理加速相关研究
- **多会议数据支持**：支持AAAI、ICLR、ICML、NAACL等顶级会议的论文数据分析
- **智能PDF解析**：使用Grobid自动提取论文标题、作者、机构、摘要等结构化信息
- **技术简报生成**：自动生成AI推理加速技术分析报告和论文简报
- **批量论文下载**：支持从arXiv等源自动下载相关论文PDF文件

### 分析维度
- **Oral论文分析**：重点分析各会议的口头报告论文
- **Spotlight论文分析**：分析重要论文（spotlight类别）
- **知名机构分析**：分析知名公司和研究机构的相关论文贡献
- **每日arXiv跟踪**：实时跟踪arXiv上的最新AI推理加速论文

### 输出格式
- **HTML报告**：可视化分析报告，包含详细的论文匹配信息
- **CSV数据**：结构化的论文数据，便于进一步分析
- **Markdown简报**：技术简报和分析总结
- **JSON数据**：机器可读的分析结果

## 运行方式

### 1. 会议论文分析

#### AAAI 2025分析
```bash
cd AAAI-2025
python oral_analysis.py  # 分析AAAI 2025 Oral论文
```

#### ICLR 2025分析
```bash
cd ICLR-2025
python Oral_Analysis.py  # 分析ICLR 2025 Oral论文
python analyze_famous_companies.py  # 分析知名机构论文
```

#### ICML 2025分析
```bash
cd ICML-2025
python Oral_Analysis.py  # 分析ICML 2025 Oral论文
python famous_company_analysis.py  # 分析知名机构论文
```

### 2. 每日arXiv论文跟踪

```bash
cd Daily-Arixiv

# Windows PowerShell环境
./arxiv_analysis.ps1

# 或直接运行Python脚本
python parser.py  # 解析每日arXiv论文
python deep_analysis.py "2025-09/09-01"  # 深度分析特定日期的论文
```

### 3. 工具模块使用

#### AI推理加速论文提取
```python
from utils.ai_acceleration_extractor import ai_acceleration_parse

# 分析PDF论文目录
ai_acceleration_parse(
    papers_dir="./papers",
    output_dir="./results", 
    enable_llm_judge=True,
    match_threshold=5
)
```

#### PDF论文下载
```python
from utils.pdf_downloader.arxiv_downloader import download_pdfs_from_arxiv

# 批量下载论文
papers = ["Attention Is All You Need", "BERT: Pre-training"]
results = download_pdfs_from_arxiv(papers, "download_dir")
```

#### PDF内容提取
```python
from utils.pdf_extractor import extract_paper_abstract

# 提取论文信息
paper_info = extract_paper_abstract("paper.pdf")
print(f"标题: {paper_info['title']}")
print(f"作者: {len(paper_info['authors'])}人")
```

### 4. 配置文件说明

- `utils/doubao_api/config.yaml`：豆包API配置
- `utils/pdf_extractor/config.json`：Grobid服务器配置
- `utils/ai_acceleration_extractor/config.py`：关键词匹配配置
