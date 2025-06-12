import requests
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import logging
from typing import List, Optional
import re
import os
import time
from pathlib import Path
import glob

# 导入配置
from config import *

# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT
)
logger = logging.getLogger(__name__)


def fetch_page_content(url: str, max_retries: int = MAX_RETRY_ATTEMPTS) -> Optional[str]:
    """
    获取页面内容，支持重试机制

    Args:
        url: 要获取的页面URL
        max_retries: 最大重试次数

    Returns:
        str: 页面HTML内容，如果请求失败则返回None
    """
    for attempt in range(max_retries + 1):
        try:
            headers = {'User-Agent': random.choice(USER_AGENTS)}
            
            if attempt == 0:
                logger.info(f"获取页面内容: {url}")
            else:
                logger.info(f"重试获取页面内容 (第{attempt}次): {url}")
                
            response = requests.get(url, headers=headers, timeout=PAGE_TIMEOUT)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            if attempt < max_retries:
                logger.warning(f"获取页面内容失败 (第{attempt + 1}次尝试): {str(e)}，{RETRY_DELAY}秒后重试...")
                time.sleep(RETRY_DELAY)
                return None
            else:
                logger.error(f"获取页面内容最终失败: {str(e)} (已重试{max_retries}次)")
                return None
    return None


def extract_pdf_links_with_titles(html_content: str, base_url: str) -> List[dict]:
    """
    从HTML内容中提取PDF链接和对应的论文标题

    Args:
        html_content: 页面HTML内容
        base_url: 基础URL，用于构建完整的PDF链接

    Returns:
        List[dict]: 包含PDF链接和标题的字典列表，格式为 [{"url": "...", "title": "..."}]
    """
    if not html_content:
        return []

    pdf_info_list = []
    try:
        soup = BeautifulSoup(html_content, "html.parser")
        
        # 查找所有文章摘要容器
        article_summaries = soup.find_all("div", class_=ARTICLE_CONTAINER_CLASS)
        logger.info(f"找到 {len(article_summaries)} 个文章摘要容器")
        
        for article in article_summaries:
            # 提取论文标题
            title_element = article.find("h3", class_=TITLE_CLASS)
            if title_element:
                title_link = title_element.find("a")
                if title_link:
                    title = title_link.get_text(strip=True)
                    
                    # 查找PDF链接
                    galley_links = article.find("ul", class_=GALLEY_LINKS_CLASS)
                    if galley_links:
                        pdf_link = galley_links.find("a", class_=PDF_LINK_CLASS)
                        if pdf_link and pdf_link.get("href"):
                            pdf_url = urljoin(base_url, pdf_link["href"])
                            pdf_info_list.append({
                                "url": pdf_url,
                                "title": title
                            })
                            logger.debug(f"找到PDF: {title[:50]}...")
    except Exception as e:
        logger.error(f"解析PDF链接和标题失败: {str(e)}")

    return pdf_info_list

def sanitize_filename(filename: str) -> str:
    """
    清理文件名，移除不合法的字符

    Args:
        filename: 原始文件名

    Returns:
        str: 清理后的文件名
    """
    # 移除或替换不合法的字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 移除多余的空格并替换为下划线
    filename = re.sub(r'\s+', '_', filename)
    # 移除开头和结尾的下划线和点
    filename = filename.strip('_.')
    # 限制文件名长度
    if len(filename) > MAX_FILENAME_LENGTH:
        filename = filename[:MAX_FILENAME_LENGTH]
    return filename.strip()


def create_filename_from_title(title: str, page_index: int, index: int) -> str:
    """
    根据论文标题创建文件名

    Args:
        title: 论文标题
        page_index: 页面索引
        index: 论文序号

    Returns:
        str: 清理后的文件名（包含.pdf扩展名）
    """
    # 清理标题
    clean_title = sanitize_filename(title)
    
    # 如果清理后的标题为空，使用默认名称
    if not clean_title:
        clean_title = f"paper_{index:03d}"
    else:
        # 添加序号前缀以避免重名
        clean_title = f"{page_index:03d}_{index:03d}_{clean_title}"
    
    # 确保有.pdf扩展名
    if not clean_title.endswith('.pdf'):
        clean_title += '.pdf'
    
    return clean_title


def extract_filename_from_url(url: str, default_name: str = "paper") -> str:
    """
    从URL中提取文件名

    Args:
        url: PDF的URL
        default_name: 默认文件名

    Returns:
        str: 文件名
    """
    try:
        parsed_url = urlparse(url)
        # 尝试从URL路径中提取文件名
        if parsed_url.path:
            filename = os.path.basename(parsed_url.path)
            if filename and filename.endswith('.pdf'):
                return sanitize_filename(filename)

        # 如果无法从URL提取，使用默认名称
        return f"{default_name}.pdf"
    except Exception:
        return f"{default_name}.pdf"


def download_pdf(url: str, save_dir: str, filename: str = None, max_retries: int = MAX_RETRY_ATTEMPTS) -> bool:
    """
    下载单个PDF文件，支持重试机制

    Args:
        url: PDF的URL
        save_dir: 保存目录
        filename: 文件名（可选）
        max_retries: 最大重试次数

    Returns:
        bool: 下载是否成功
    """
    # 确保保存目录存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 生成文件名
    if not filename:
        filename = extract_filename_from_url(url)

    file_path = os.path.join(save_dir, filename)

    # 如果文件已存在，跳过下载
    if os.path.exists(file_path):
        logger.info(f"文件已存在，跳过下载: {filename}")
        return True

    # 重试下载逻辑
    for attempt in range(max_retries + 1):
        try:
            # 下载文件
            headers = {'User-Agent': random.choice(USER_AGENTS)}

            if attempt == 0:
                logger.info(f"开始下载: {filename}")
            else:
                logger.info(f"重试下载 (第{attempt}次): {filename}")

            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True)
            response.raise_for_status()

            # 检查响应是否为PDF
            content_type = response.headers.get('content-type', '').lower()
            if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                logger.warning(f"响应不是PDF格式: {content_type} for {url}")

            # 写入文件
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"下载完成: {filename}")
            return True

        except requests.RequestException as e:
            if attempt < max_retries:
                logger.warning(f"下载失败 (第{attempt + 1}次尝试) {url}: {str(e)}，{RETRY_DELAY}秒后重试...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error(f"下载最终失败 {url}: {str(e)} (已重试{max_retries}次)")
                # 如果下载失败，删除可能存在的不完整文件
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        logger.debug(f"已删除不完整文件: {filename}")
                    except OSError:
                        pass
                return False
        except Exception as e:
            logger.error(f"保存文件失败 {filename}: {str(e)}")
            # 如果保存失败，删除可能存在的不完整文件
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.debug(f"已删除不完整文件: {filename}")
                except OSError:
                    pass
            return False

    return False


def download_all_pdfs(pdf_info_list: List[dict], page_id:int = 0, save_dir: str = DEFAULT_SAVE_DIR,
                      delay: float = DEFAULT_DELAY) -> dict:
    """
    下载所有PDF文件

    Args:
        pdf_info_list: PDF信息列表，字典列表（包含url和title）
        page_id: 页面ID
        save_dir: 保存目录
        delay: 每次下载之间的延迟（秒）

    Returns:
        dict: 下载结果统计 {"success": int, "failed": int}
    """
    if not pdf_info_list:
        logger.warning("没有找到PDF链接")
        return {"success": 0, "failed": 0}

    # 确保pdf_info_list是统一格式
    if isinstance(pdf_info_list[0], str):
        # 如果是字符串列表，转换为字典格式
        pdf_info_list = [{"url": url, "title": None} for url in pdf_info_list]

    logger.info(f"开始下载 {len(pdf_info_list)} 个PDF文件到 {save_dir} 目录")

    success_count = 0
    failed_count = 0

    for i, pdf_info in enumerate(pdf_info_list, 1):
        try:
            pdf_url = pdf_info["url"]
            title = pdf_info.get("title")
            
            logger.info(f"进度: {i}/{len(pdf_info_list)} - {pdf_url}")
            if title:
                logger.info(f"论文标题: {title}")

            # 生成文件名
            if title:
                filename = create_filename_from_title(title, page_id, i)
            else:
                filename = f"paper_{i:03d}_{extract_filename_from_url(pdf_url, f'aaai2025_{i:03d}')}"

            # 下载文件
            if download_pdf(pdf_url, save_dir, filename):
                success_count += 1
            else:
                failed_count += 1

            # 添加延迟避免过于频繁的请求
            if i < len(pdf_info_list):  # 最后一个文件不需要延迟
                time.sleep(delay)

        except KeyboardInterrupt:
            logger.info("用户中断下载")
            break
        except Exception as e:
            logger.error(f"处理链接失败 {pdf_info.get('url', 'unknown')}: {str(e)}")
            failed_count += 1

    logger.info(f"下载完成！成功: {success_count}, 失败: {failed_count}")
    return {"success": success_count, "failed": failed_count}


def check_download_results(save_dir: str = DEFAULT_SAVE_DIR) -> dict:
    """
    检查下载结果并显示统计信息

    Args:
        save_dir: 保存目录

    Returns:
        dict: 下载统计信息
    """
    if not os.path.exists(save_dir):
        logger.warning(f"下载目录不存在: {save_dir}")
        return {"count": 0, "total_size_mb": 0}

    # 统计下载的PDF文件
    pdf_files = glob.glob(os.path.join(save_dir, "*.pdf"))

    logger.info(f"下载目录: {save_dir}")
    logger.info(f"已下载PDF文件数量: {len(pdf_files)}")

    total_size_mb = 0
    if pdf_files:
        # 计算文件大小
        total_size = 0
        for pdf_file in pdf_files:
            try:
                size = os.path.getsize(pdf_file)
                total_size += size
            except OSError:
                pass

        # 转换为MB
        total_size_mb = total_size / (1024 * 1024)
        logger.info(f"总文件大小: {total_size_mb:.2f} MB")

        # 显示前10个文件名
        logger.info("下载的文件列表（前10个）:")
        for i, pdf_file in enumerate(pdf_files[:10], 1):
            filename = os.path.basename(pdf_file)
            size_mb = os.path.getsize(pdf_file) / (1024 * 1024)
            logger.info(f"{i:2d}. {filename} ({size_mb:.2f} MB)")

        if len(pdf_files) > 10:
            logger.info(f"... 还有 {len(pdf_files) - 10} 个文件")

    else:
        logger.warning("没有找到已下载的PDF文件")

    return {"count": len(pdf_files), "total_size_mb": total_size_mb}


def main():
    """
    主函数：执行论文下载的完整流程
    """
    logger.info("开始AAAI 2025论文下载程序")
    
    for page_id, url in enumerate(AAAI_ISSUE_URL, 1):
        # 步骤1: 获取页面内容
        logger.info("步骤1: 获取页面内容")
        page_content = fetch_page_content(url)
        
        if not page_content:
            logger.error("获取页面内容失败，程序退出")
            return
        
        logger.info("成功获取页面内容")
        
        # 步骤2: 解析PDF链接和标题
        logger.info("步骤2: 解析PDF链接和标题")
        pdf_info_list = extract_pdf_links_with_titles(page_content, url)
        
        if not pdf_info_list:
            logger.warning("没有找到可下载的PDF链接，请检查网页解析是否正确")
            return
        
        logger.info(f"找到 {len(pdf_info_list)} 个PDF文件")
        
        # 显示前5个作为示例
        for i, info in enumerate(pdf_info_list[:5], 1):
            logger.info(f"{i}. {info['title']}")
        
        if len(pdf_info_list) > 5:
            logger.info(f"... 还有 {len(pdf_info_list) - 5} 个PDF文件")
        
        # 步骤3: 应用下载限制（如果有）
        if MAX_DOWNLOADS and MAX_DOWNLOADS < len(pdf_info_list):
            limited_pdf_info = pdf_info_list[:MAX_DOWNLOADS]
            logger.info(f"限制下载数量为: {MAX_DOWNLOADS}")
        else:
            limited_pdf_info = pdf_info_list
        
        logger.info(f"准备下载的PDF文件数量: {len(limited_pdf_info)}")
        
        # 步骤4: 开始下载
        logger.info("步骤4: 开始下载PDF文件")
        download_results = download_all_pdfs(limited_pdf_info, page_id, DEFAULT_SAVE_DIR, DEFAULT_DELAY)
        
        # 步骤5: 检查下载结果
        logger.info("步骤5: 检查下载结果")
        file_stats = check_download_results(DEFAULT_SAVE_DIR)
        
        # 总结
        logger.info("=" * 50)
        logger.info("下载任务完成总结:")
        logger.info(f"成功下载: {download_results['success']} 个文件")
        logger.info(f"下载失败: {download_results['failed']} 个文件")
        logger.info(f"文件总数: {file_stats['count']} 个")
        logger.info(f"总大小: {file_stats['total_size_mb']:.2f} MB")
        logger.info("=" * 50)
        
    logger.info("AAAI 2025论文下载程序完成")


if __name__ == "__main__":
    main()
