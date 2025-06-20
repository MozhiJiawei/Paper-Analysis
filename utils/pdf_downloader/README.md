# PDF下载器模块说明

## 概述

`pdf_downloader.py` 是一个独立的PDF批量下载模块，从原有的 `download_papers.py` 中提取并重构而来。它提供了清晰、易用的接口，支持单个PDF下载和批量PDF下载，具有重试机制、错误处理和进度监控功能。

## 主要特性

- ✅ **批量下载**：支持从URL列表或包含标题信息的字典列表下载PDF
- ✅ **重试机制**：自动重试失败的下载，提高成功率
- ✅ **文件名处理**：智能生成文件名，支持从标题或URL提取
- ✅ **进度监控**：详细的日志输出，实时显示下载进度
- ✅ **错误处理**：优雅处理网络错误和文件操作错误
- ✅ **统计信息**：提供下载结果统计和目录分析
- ✅ **配置灵活**：可自定义下载参数（超时、重试次数、延迟等）

## 安装依赖

```bash
pip install requests beautifulsoup4
```

## 使用方法

### 1. 便捷函数接口（推荐用于简单场景）

#### 下载单个PDF文件

```python
from pdf_downloader import download_single_pdf

# 下载单个PDF
success = download_single_pdf(
    url="https://example.com/paper.pdf",
    save_dir="downloads",
    filename="my_paper.pdf"
)
print(f"下载{'成功' if success else '失败'}")
```

#### 批量下载PDF文件

```python
from pdf_downloader import download_pdfs

# 方式1: 从URL列表下载
pdf_urls = [
    "https://example.com/paper1.pdf",
    "https://example.com/paper2.pdf",
    "https://example.com/paper3.pdf"
]

results = download_pdfs(
    pdf_info_list=pdf_urls,
    save_dir="downloads",
    delay=1.0
)
print(f"成功: {results['success']}, 失败: {results['failed']}")

# 方式2: 从包含标题的字典列表下载
pdf_info_list = [
    {
        "url": "https://example.com/paper1.pdf",
        "title": "Deep Learning for Computer Vision"
    },
    {
        "url": "https://example.com/paper2.pdf", 
        "title": "Natural Language Processing"
    }
]

results = download_pdfs(pdf_info_list, save_dir="papers")
```

### 2. 类接口（推荐用于复杂场景）

```python
from pdf_downloader import PDFDownloader

# 创建下载器实例
downloader = PDFDownloader(
    save_dir="AAAI-2025-Papers",
    delay=1.5,
    max_retries=5,
    timeout=30,
    max_filename_length=150
)

# 下载PDF列表
results = downloader.download_pdfs_from_list(pdf_info_list)

# 获取下载统计
stats = downloader.get_download_stats()
print(f"总共下载了 {stats['count']} 个文件，大小 {stats['total_size_mb']:.2f} MB")
```

## API参考

### PDFDownloader 类

#### 构造函数

```python
PDFDownloader(
    save_dir: str = "downloads",           # 保存目录
    delay: float = 1.0,                    # 下载间隔（秒）
    max_retries: int = 3,                  # 最大重试次数
    timeout: int = 30,                     # 请求超时时间（秒）
    max_filename_length: int = 150         # 最大文件名长度
)
```

#### 主要方法

##### `download_single_pdf(url, filename=None)`
下载单个PDF文件

**参数:**
- `url` (str): PDF的URL
- `filename` (str, 可选): 保存的文件名

**返回:** `bool` - 下载是否成功

##### `download_pdfs_from_list(pdf_info_list)`
批量下载PDF文件

**参数:**
- `pdf_info_list` (List[Union[str, Dict]]): PDF信息列表，支持以下格式：
  - 字符串列表: `["url1", "url2", ...]`
  - 字典列表: `[{"url": "url1", "title": "title1"}, ...]`

**返回:** `Dict[str, int]` - 下载结果统计 `{"success": int, "failed": int}`

##### `get_download_stats()`
获取下载目录的统计信息

**返回:** `Dict` - 包含文件数量、总大小和文件列表的统计信息

### 便捷函数

##### `download_pdfs(pdf_info_list, save_dir="downloads", delay=1.0, max_retries=3, timeout=30)`
便捷的批量下载函数

##### `download_single_pdf(url, save_dir="downloads", filename=None, max_retries=3, timeout=30)`
便捷的单文件下载函数

## 与原有代码的集成

如果您之前使用 `download_papers.py` 中的函数，可以很容易地迁移到新的模块：

### 替换原有的 `download_all_pdfs` 函数

```python
# 原有代码
from download_papers import download_all_pdfs
results = download_all_pdfs(pdf_info_list, page_id, save_dir, delay)

# 新代码
from pdf_downloader import PDFDownloader
downloader = PDFDownloader(save_dir=save_dir, delay=delay)
results = downloader.download_pdfs_from_list(pdf_info_list)
```

### 替换原有的 `check_download_results` 函数

```python
# 原有代码
from download_papers import check_download_results
stats = check_download_results(save_dir)

# 新代码
from pdf_downloader import PDFDownloader
downloader = PDFDownloader(save_dir=save_dir)
stats = downloader.get_download_stats()
```

## 配置说明

### 常用配置参数

- **save_dir**: 下载文件保存目录
- **delay**: 下载间隔时间（秒），避免对服务器造成过大压力
- **max_retries**: 最大重试次数，网络不稳定时增加此值
- **timeout**: 请求超时时间，网络较慢时增加此值
- **max_filename_length**: 文件名最大长度，避免操作系统限制

### 建议配置

```python
# 生产环境建议配置
downloader = PDFDownloader(
    save_dir="papers",
    delay=2.0,        # 适中的延迟，避免被服务器限制
    max_retries=5,    # 较多的重试次数
    timeout=60,       # 较长的超时时间
    max_filename_length=100  # 合理的文件名长度
)
```

## 错误处理

模块内置了完善的错误处理机制：

- **网络错误**: 自动重试，记录详细错误信息
- **文件操作错误**: 清理不完整的文件，避免磁盘空间浪费
- **无效URL**: 跳过无效链接，继续处理其他文件
- **文件名冲突**: 自动重命名，避免覆盖现有文件

## 日志配置

模块使用Python标准日志库，您可以根据需要配置日志级别：

```python
import logging

# 配置详细日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 如果只想看到错误信息
logging.getLogger('pdf_downloader').setLevel(logging.ERROR)
```

## 最佳实践

1. **合理设置延迟**: 避免对目标服务器造成过大压力
2. **监控磁盘空间**: 大量下载前确保有足够的存储空间
3. **网络稳定性**: 网络不稳定时增加重试次数和超时时间
4. **错误处理**: 检查返回的统计信息，处理失败的下载
5. **日志监控**: 关注日志输出，及时发现和解决问题

## 示例代码

完整的使用示例请参考 `example_usage.py` 文件，其中包含了各种使用场景的详细演示。 