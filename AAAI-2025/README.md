# AAAI 2025 论文分析工具

> 📚 专门用于AAAI 2025会议论文下载和AI推理加速相关论文筛选的工具模块

## 🎯 核心功能

### 1. 论文批量下载
- 自动下载AAAI 2025会议的全部论文PDF
- 智能文件命名和组织
- 支持断点续传和失败重试

### 2. AI推理加速论文筛选
- 基于关键词匹配算法筛选AI推理加速相关论文
- 支持Oral论文优先分析
- 生成详细的分析报告和统计数据

## 📁 文件结构

```
AAAI-2025/
├── download_papers.py              # 论文下载主程序
├── get_ai_inference_related_paper.py  # AI推理加速论文筛选
├── config.py                       # 下载配置文件
├── oral_paper_filenames.txt         # Oral论文文件名列表
├── oral_paper_list.txt             # Oral论文信息列表
├── pages.html                      # 网页源码（用于调试）
├── AAAI-2025-Papers/              # 下载的论文存储目录
└── 输出文件/                        # 分析结果文件
    ├── ai_inference_related_papers.csv    # AI相关论文（CSV格式）
    ├── ai_inference_related_papers.txt    # AI相关论文（文本格式）
    ├── non_ai_inference_papers.csv        # 非AI相关论文
    ├── match_statistics.csv               # 匹配统计信息
    └── oral_papers_about_ai_acceleration.xlsx  # Oral论文Excel报告
```

## 🚀 使用指南

### 环境准备

1. **启动Grobid服务**（必需）：
   ```bash
   docker run --rm -it -p 8070:8070 lfoppiano/grobid:0.8.3
   ```

2. **安装Python依赖**：
   ```bash
   pip install -r ../requirements.txt
   ```

### 论文下载

```bash
# 下载所有AAAI 2025论文
python download_papers.py
```

**下载特性**：
- 📥 自动获取论文列表并批量下载
- 🏷️ 基于论文标题智能命名文件
- 🔄 支持断点续传，跳过已下载文件
- 📊 提供详细的下载进度和统计信息

### AI推理加速论文筛选

```bash
# 筛选Oral论文中的AI推理加速相关论文（推荐）
python get_ai_inference_related_paper.py

# 或在Python中使用
from utils.ai_acceleration_extractor import AiAccelerationExtractor

extractor = AiAccelerationExtractor("AAAI-2025-Papers", ".")
# 分析Oral论文
extractor.parse(paper_filenames=oral_filenames, analyze_all=False, output_format="csv")
# 分析全量论文
extractor.parse(analyze_all=True, output_format="csv")
```

## 🔍 筛选算法说明

### 关键词分类体系

| 类别 | 权重 | 示例关键词 |
|------|------|----------|
| **核心推理加速** | 5分 | inference acceleration, quantization, model compression |
| **高相关技术** | 4分 | early exit, dynamic inference, tensorrt |
| **中等相关技术** | 3分 | kernel fusion, efficient transformer |
| **支撑关键词** | 2分 | latency, throughput, efficiency |

### 匹配策略
- ✅ **包含策略**：匹配积分≥6分的论文被标记为AI推理加速相关
- ❌ **排除策略**：包含训练相关、纯理论研究等强排除关键词的论文被排除
- 🎯 **优先级**：标题中的关键词比摘要中的权重更高

## 📊 输出格式

### CSV格式（推荐）
- `ai_inference_related_papers.csv` - 相关论文详细信息
- `non_ai_inference_papers.csv` - 非相关论文信息  
- `match_statistics.csv` - 匹配统计数据

**CSV字段说明**：
- 基本信息：标题、作者、组织、文件名
- 匹配详情：匹配分数、各类关键词数量
- 关键词：标题和摘要中的匹配关键词
- 摘要预览：论文摘要前200字符

### 文本格式
- `ai_inference_related_papers.txt` - 人类可读的详细报告
- `match_statistics.txt` - 统计信息文本格式

## ⚙️ 配置选项

### 下载配置 (config.py)

```python
# 网络设置
REQUEST_TIMEOUT = 30        # 请求超时时间
MAX_RETRY_ATTEMPTS = 5      # 最大重试次数
DEFAULT_DELAY = 1.5         # 下载间隔

# 文件设置  
DEFAULT_SAVE_DIR = "AAAI-2025-Papers"
MAX_FILENAME_LENGTH = 150   # 文件名长度限制
```

### 筛选配置

在 `get_ai_inference_related_paper.py` 中：
- `analyze_all=True` - 分析全量论文
- `analyze_all=False` - 仅分析Oral论文（默认）
- `output_format="csv"` - 输出CSV格式
- `output_format="txt"` - 输出文本格式

## 📈 性能优化

### 下载优化
- 使用多个User-Agent轮换避免封锁
- 智能重试机制处理网络波动
- 流式下载大文件节省内存

### 分析优化
- PDF内容缓存机制（基于Grobid）
- 分层关键词匹配算法
- 批量处理提高效率

## 🔧 故障排除

### 常见问题

1. **Grobid服务未启动**
   ```bash
   # 检查服务状态
   curl http://localhost:8070
   # 重新启动Grobid
   docker run --rm -it -p 8070:8070 lfoppiano/grobid:0.8.3
   ```

2. **下载失败率高**
   - 检查网络连接
   - 增加重试次数和延迟时间
   - 检查AAAI网站访问状态

3. **论文分析结果异常**
   - 确认PDF文件完整性
   - 检查Grobid服务响应
   - 验证oral_paper_filenames.txt文件格式

## 📝 使用示例

### 完整工作流程

```bash
# 1. 启动Grobid服务
docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.3

# 2. 下载论文
python download_papers.py

# 3. 筛选AI推理加速相关论文
python get_ai_inference_related_paper.py

# 4. 查看结果
ls -la *.csv *.txt *.xlsx
```

### 自定义分析

```python
from utils.ai_acceleration_extractor import AiAccelerationExtractor

# 创建提取器
extractor = AiAccelerationExtractor("AAAI-2025-Papers", ".")

# 分析特定论文
specific_papers = ["paper1.pdf", "paper2.pdf"]
extractor.parse(paper_filenames=specific_papers, output_format="both")
```

## 📊 统计信息

通过运行筛选程序，您将获得：
- 论文总数和AI相关论文数量
- 各类关键词匹配统计
- 不同组织的研究分布
- 详细的匹配得分分析

---

> 💡 **提示**：建议先运行Oral论文分析（速度快），然后根据需要决定是否进行全量分析。 