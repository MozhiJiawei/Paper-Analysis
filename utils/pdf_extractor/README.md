# PDF内容提取器

> 📄 基于Grobid的高精度PDF学术论文内容提取工具

## 🎯 功能概述

PDF内容提取器是一个专门用于从学术论文PDF中提取结构化信息的工具，基于强大的Grobid机器学习库实现高精度的内容解析。

### 核心功能
- 📖 **标题提取**：准确识别论文主标题
- 👥 **作者信息**：提取作者姓名、邮箱和所属机构
- 📝 **摘要提取**：完整提取论文摘要内容
- 🏢 **机构识别**：解析作者所属的研究机构
- 💾 **缓存机制**：避免重复处理，提高效率

## 📁 文件结构

```
pdf_extractor/
├── __init__.py                # 包初始化
├── pdf_extractor.py           # 主要提取器实现
├── config.json                # Grobid服务配置
└── README.md                  # 本文档
```

## 🚀 快速开始

### 前置要求

1. **启动Grobid服务**（必需）：
   ```bash
   # 使用Docker启动Grobid服务
   docker run --rm -it -p 8070:8070 lfoppiano/grobid:0.8.3
   
   # 或后台运行
   docker run -d --name grobid -p 8070:8070 lfoppiano/grobid:0.8.3
   ```

2. **安装Python依赖**：
   ```bash
   pip install grobid-client beautifulsoup4 lxml
   ```

### 基本使用

```python
from utils.pdf_extractor import extract_paper_abstract

# 提取单个PDF的信息
paper_info = extract_paper_abstract("path/to/paper.pdf")

# 打印提取结果
print(f"标题: {paper_info['title']}")
print(f"作者数量: {len(paper_info['authors'])}")
print(f"摘要长度: {len(paper_info['abstract'])}")
```

## 📊 数据格式

### 输入格式
- **PDF文件路径**：支持相对路径和绝对路径
- **文件要求**：标准的学术论文PDF格式

### 输出格式

```python
{
    'title': '论文标题',
    'authors': [
        {
            'name': '作者全名',
            'email': '作者邮箱（如果有）',
            'affiliation': ['所属机构列表']
        }
        # ... 更多作者
    ],
    'abstract': '完整的论文摘要文本',
    'affiliations': []  # 保留字段
}
```

### 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `title` | str | 论文主标题，去除多余空格 |
| `authors` | list | 作者信息列表 |
| `authors[].name` | str | 作者姓名（名 + 姓） |
| `authors[].email` | str | 作者邮箱地址（可选） |
| `authors[].affiliation` | list | 作者所属机构列表 |
| `abstract` | str | 论文摘要完整文本 |
| `affiliations` | list | 机构信息（保留字段） |

## ⚙️ 配置说明

### Grobid配置 (`config.json`)

```json
{
    "grobid": {
        "server": "http://localhost:8070",
        "sleep_time": 5,
        "timeout": 60,
        "coordinates": ["persName", "figure", "ref", "biblStruct", "formula"]
    }
}
```

**配置参数**：
- `server`: Grobid服务器地址
- `sleep_time`: 请求间隔时间（秒）
- `timeout`: 请求超时时间（秒）
- `coordinates`: 需要提取的坐标信息类型

### 自定义配置

```python
# 修改config.json后使用自定义配置
import os
from utils.pdf_extractor import extract_paper_abstract

# 配置文件会自动加载
paper_info = extract_paper_abstract("paper.pdf")
```

## 🔧 高级用法

### 批量处理PDF文件

```python
import os
from utils.pdf_extractor import extract_paper_abstract

def batch_extract_papers(pdf_directory):
    """批量提取PDF信息"""
    papers_info = []
    
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            try:
                paper_info = extract_paper_abstract(pdf_path)
                paper_info['filename'] = filename
                papers_info.append(paper_info)
                print(f"✅ 成功处理: {filename}")
            except Exception as e:
                print(f"❌ 处理失败: {filename} - {e}")
    
    return papers_info

# 使用示例
papers = batch_extract_papers("papers_directory")
print(f"成功处理 {len(papers)} 篇论文")
```

### 错误处理和重试

```python
import time
from utils.pdf_extractor import extract_paper_abstract

def robust_extract(pdf_path, max_retries=3):
    """带重试机制的提取"""
    for attempt in range(max_retries):
        try:
            return extract_paper_abstract(pdf_path)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"重试 {attempt + 1}/{max_retries}: {e}")
                time.sleep(2)  # 等待2秒后重试
            else:
                print(f"最终失败: {e}")
                raise

# 使用示例
try:
    paper_info = robust_extract("difficult_paper.pdf")
    print("提取成功")
except Exception:
    print("提取失败")
```

### 缓存机制利用

```python
# 提取器自动使用缓存机制
# 如果对应的.grobid.tei.xml文件存在，将直接读取而不重新处理

import os
from utils.pdf_extractor import extract_paper_abstract

pdf_path = "paper.pdf"
xml_cache_path = "paper.grobid.tei.xml"

# 第一次调用：使用Grobid处理PDF，生成XML缓存
paper_info1 = extract_paper_abstract(pdf_path)

# 第二次调用：直接读取XML缓存，速度更快
paper_info2 = extract_paper_abstract(pdf_path)

print(f"XML缓存存在: {os.path.exists(xml_cache_path)}")
```

## 🔍 技术实现

### Grobid集成

```python
from grobid_client.grobid_client import GrobidClient

# 创建Grobid客户端
client = GrobidClient(config_path="config.json")

# 处理PDF文档头部信息
result = client.process_pdf(
    "processHeaderDocument",
    pdf_file_path,
    None, None, None, None, None, None, None
)[2]
```

### XML解析

```python
from bs4 import BeautifulSoup

# 解析Grobid返回的TEI XML
soup = BeautifulSoup(result, 'xml')

# 提取标题
title = soup.find('title', {'type': 'main'})

# 提取作者
authors = soup.find_all('author')

# 提取摘要
abstract = soup.find('abstract')
```

## 📊 性能优化

### 缓存策略
- **XML缓存**：自动保存Grobid处理结果为XML文件
- **避免重复处理**：检查缓存文件存在性
- **快速读取**：直接解析XML而非重新调用Grobid

### 处理速度
- **首次处理**：~2-5秒/PDF（取决于文件大小和复杂度）
- **缓存读取**：~0.1-0.5秒/PDF
- **批量处理**：建议添加适当延迟避免服务器过载

### 内存使用
- **流式处理**：逐个处理PDF文件
- **及时释放**：处理完成后释放内存
- **配置优化**：调整Grobid超时设置

## 🔧 故障排除

### 常见问题

1. **Grobid服务未响应**
   ```bash
   # 检查服务状态
   curl http://localhost:8070
   
   # 重启Grobid服务
   docker restart grobid_container
   ```

2. **PDF处理失败**
   ```python
   # 检查文件是否损坏
   try:
       with open("paper.pdf", "rb") as f:
           f.read(1024)  # 尝试读取前1KB
   except Exception as e:
       print(f"PDF文件可能损坏: {e}")
   ```

3. **提取结果为空**
   - 检查PDF是否为图片扫描版本
   - 确认PDF包含可提取的文本内容
   - 验证论文格式是否符合学术标准

### 调试模式

```python
import logging
from utils.pdf_extractor import extract_paper_abstract

# 启用详细日志
logging.basicConfig(level=logging.DEBUG)

# 处理PDF并查看详细信息
paper_info = extract_paper_abstract("paper.pdf")

# 检查提取质量
if not paper_info['title']:
    print("⚠️ 警告: 未提取到标题")
if not paper_info['abstract']:
    print("⚠️ 警告: 未提取到摘要")
if not paper_info['authors']:
    print("⚠️ 警告: 未提取到作者信息")
```

## 📝 使用示例

### 完整工作流程

```python
import os
from utils.pdf_extractor import extract_paper_abstract

def analyze_paper_content(pdf_path):
    """分析论文内容"""
    try:
        # 提取论文信息
        paper_info = extract_paper_abstract(pdf_path)
        
        # 基本信息检查
        print(f"📄 论文标题: {paper_info['title']}")
        print(f"👥 作者数量: {len(paper_info['authors'])}")
        print(f"📝 摘要长度: {len(paper_info['abstract'])} 字符")
        
        # 作者信息详细显示
        for i, author in enumerate(paper_info['authors'], 1):
            print(f"  {i}. {author['name']}")
            if author.get('email'):
                print(f"     📧 {author['email']}")
            if author.get('affiliation'):
                print(f"     🏢 {', '.join(author['affiliation'])}")
        
        # 摘要预览
        if paper_info['abstract']:
            preview = paper_info['abstract'][:200] + "..." if len(paper_info['abstract']) > 200 else paper_info['abstract']
            print(f"📄 摘要预览: {preview}")
        
        return paper_info
        
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return None

# 使用示例
if __name__ == "__main__":
    paper_info = analyze_paper_content("example_paper.pdf")
```

---

> 💡 **提示**：建议在批量处理前先用少量文件测试，确保Grobid服务配置正确且运行稳定。 