# PDF下载器模块

这个模块提供了强大的PDF文件下载功能，支持从各种来源下载PDF文件，包括直接URL链接和arXiv论文库。

## 功能特性

### 通用PDF下载器
- 支持单个PDF文件下载
- 支持批量PDF文件下载
- 自动重试机制
- 下载进度跟踪
- 文件名重复处理
- 下载统计信息

### arXiv论文下载器 ⭐ **新功能**
- 支持按论文标题搜索并下载
- 支持按arXiv ID直接下载
- 批量下载多篇论文
- 自动文件名清理
- 智能搜索匹配

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包括：
- `requests` - HTTP请求
- `beautifulsoup4` - HTML解析
- `arxiv` - arXiv API客户端

## 使用方法

### 1. arXiv论文下载

#### 下载单篇论文

```python
from arxiv_downloader import download_single_pdf_from_arxiv

# 方式1：使用论文标题
result = download_single_pdf_from_arxiv(
    paper_name="Attention Is All You Need",
    save_dir="papers"
)

# 方式2：使用arXiv ID  
result = download_single_pdf_from_arxiv(
    paper_name="1706.03762",
    save_dir="papers"
)

# 检查结果
if result["Attention Is All You Need"] == 0:
    print("下载成功!")
elif result["Attention Is All You Need"] == 1:
    print("未找到论文")
else:
    print("下载失败")
```

#### 批量下载论文

```python
from arxiv_downloader import download_pdfs_from_arxiv

# 论文列表（可混合使用标题和ID）
papers = [
    "Attention Is All You Need",
    "1706.03762",  # 同一篇论文的ID
    "BERT: Pre-training of Deep Bidirectional Transformers",
    "GPT-3: Language Models are Few-Shot Learners"
]

# 批量下载
results = download_pdfs_from_arxiv(
    paper_name_list=papers,
    save_dir="research_papers"
)

# 统计结果
success_count = sum(1 for status in results.values() if status == 0)
print(f"成功下载 {success_count}/{len(results)} 篇论文")
```

### 2. 普通PDF文件下载

#### 下载单个PDF

```python
from pdf_downloader import download_single_pdf

success = download_single_pdf(
    url="https://example.com/paper.pdf",
    save_dir="downloads",
    filename="my_paper.pdf"
)
```

#### 批量下载PDF

```python
from pdf_downloader import download_pdfs

# URL列表
pdf_urls = [
    "https://example.com/paper1.pdf",
    "https://example.com/paper2.pdf"
]

results = download_pdfs(
    pdf_info_list=pdf_urls,
    save_dir="batch_downloads"
)
```

#### 带标题的PDF下载

```python
# 包含标题信息的PDF列表
pdf_info_list = [
    {
        "url": "https://example.com/paper1.pdf",
        "title": "深度学习论文"
    },
    {
        "url": "https://example.com/paper2.pdf", 
        "title": "自然语言处理研究"
    }
]

results = download_pdfs(
    pdf_info_list=pdf_info_list,
    save_dir="titled_downloads"
)
```

## 返回值说明

### arXiv下载器返回值

`Dict[str, int]` 格式，其中：
- `0`: 下载成功
- `1`: 搜索失败（未找到论文）  
- `2`: 下载失败

### 普通PDF下载器返回值

- 单个下载：`bool` （True=成功，False=失败）
- 批量下载：`{"success": int, "failed": int}`

## 高级用法

### 使用PDFDownloader类

```python
from pdf_downloader import PDFDownloader

downloader = PDFDownloader(
    save_dir="custom_downloads",
    delay=2.0,  # 下载间隔
    max_retries=5,  # 最大重试次数
    timeout=60  # 超时时间
)

results = downloader.download_pdfs_from_list(pdf_info_list)
stats = downloader.get_download_stats()
```

### 混合下载工作流

```python
# 组合使用两种下载器
from arxiv_downloader import download_pdfs_from_arxiv
from pdf_downloader import download_pdfs

# 第一步：下载arXiv论文
foundational_papers = [
    "Attention Is All You Need",
    "BERT: Pre-training of Deep Bidirectional Transformers"
]

arxiv_results = download_pdfs_from_arxiv(
    paper_name_list=foundational_papers,
    save_dir="papers/foundational"
)

# 第二步：下载会议论文
conference_papers = [
    {
        "url": "https://conference.com/paper1.pdf",
        "title": "最新研究成果"
    }
]

pdf_results = download_pdfs(
    pdf_info_list=conference_papers,
    save_dir="papers/conferences"
)
```

## 注意事项

1. **arXiv搜索准确性**：使用准确的论文标题或arXiv ID可以获得最佳搜索结果
2. **文件名处理**：系统会自动清理文件名中的特殊字符
3. **重复文件**：如果文件已存在，会自动添加数字后缀
4. **网络连接**：确保网络连接稳定，特别是下载大文件时
5. **存储空间**：确保有足够的磁盘空间存储下载的PDF文件
6. **SSL证书问题**：如果遇到SSL证书验证错误，程序会自动配置SSL设置来解决此问题
7. **性能优化**：SSL配置在一次脚本执行中只会执行一次，避免重复设置

### 常见问题

#### SSL证书验证失败
如果遇到类似以下错误：
```
[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate
```

程序会自动处理此问题，无需手动干预。SSL配置会在第一次调用arXiv下载函数时自动执行，后续调用不会重复配置。如果问题持续存在，请检查网络连接和防火墙设置。

#### 性能优化说明
为了提高效率，SSL配置函数使用了全局状态跟踪，确保在同一次脚本执行中只配置一次SSL设置。这意味着：
- 第一次调用任何arXiv下载函数时会看到"已配置SSL设置"的消息
- 后续调用不会重复显示此消息，但SSL设置仍然有效

## 完整示例

查看 `example_usage.py` 文件获取更多详细的使用示例。

## 错误处理

所有下载函数都包含异常处理，会在出现错误时返回相应的状态码。建议在生产环境中添加适当的日志记录：

```python
import logging

logging.basicConfig(level=logging.INFO)

# 使用下载函数
result = download_single_pdf_from_arxiv("论文标题")
``` 