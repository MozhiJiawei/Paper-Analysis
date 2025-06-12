"""
PDF下载器模块
提供批量下载PDF文件的功能
"""

import requests
import random
import logging
import re
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Union
from urllib.parse import urlparse

# 配置日志
logger = logging.getLogger(__name__)


class PDFDownloader:
    """PDF批量下载器"""
    
    def __init__(self, 
                 save_dir: str = "downloads",
                 delay: float = 1.0,
                 max_retries: int = 3,
                 timeout: int = 30,
                 max_filename_length: int = 150,
                 prefix_index: str = None):
        """
        初始化PDF下载器
        
        Args:
            save_dir: 保存目录
            delay: 下载间隔（秒）
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            max_filename_length: 最大文件名长度
            prefix_index: 额外的Prefix索引，用于文件命名
        """
        self.save_dir = save_dir
        self.delay = delay
        self.max_retries = max_retries
        self.timeout = timeout
        self.max_filename_length = max_filename_length
        self.prefix_index = prefix_index
        
        # User-Agent列表
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        ]
        
        # 确保保存目录存在
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
    
    def sanitize_filename(self, filename: str) -> str:
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
        if len(filename) > self.max_filename_length:
            filename = filename[:self.max_filename_length]
        return filename.strip()
    
    def create_filename_from_title(self, title: str, index: int = None, prefix_index: str = None) -> str:
        """
        根据论文标题创建文件名
        
        Args:
            title: 论文标题
            index: 论文序号（可选）
            prefix_index: 额外的XXX索引（可选）
            
        Returns:
            str: 清理后的文件名（包含.pdf扩展名）
        """
        # 清理标题
        clean_title = self.sanitize_filename(title)
        
        # 如果清理后的标题为空，使用默认名称
        if not clean_title:
            clean_title = f"paper_{index:03d}" if index is not None else "paper"
        else:
            # 构建文件名前缀
            prefix_parts = []
            
            # 添加XXX索引（如果提供）
            if prefix_index is not None:
                prefix_parts.append(str(prefix_index))
            
            # 添加序号索引（如果提供）
            if index is not None:
                prefix_parts.append(f"{index:03d}")
            
            # 组合前缀和标题
            if prefix_parts:
                clean_title = "_".join(prefix_parts) + "_" + clean_title
        
        # 确保有.pdf扩展名
        if not clean_title.endswith('.pdf'):
            clean_title += '.pdf'
        
        return clean_title
    
    def extract_filename_from_url(self, url: str, default_name: str = "paper") -> str:
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
                    return self.sanitize_filename(filename)
            
            # 如果无法从URL提取，使用默认名称
            return f"{default_name}.pdf"
        except Exception:
            return f"{default_name}.pdf"
    
    def download_single_pdf(self, url: str, filename: str = None) -> bool:
        """
        下载单个PDF文件，支持重试机制
        
        Args:
            url: PDF的URL
            filename: 文件名（可选）
            
        Returns:
            bool: 下载是否成功
        """
        # 生成文件名
        if not filename:
            filename = self.extract_filename_from_url(url)
        
        file_path = os.path.join(self.save_dir, filename)
        
        # 如果文件已存在，跳过下载
        if os.path.exists(file_path):
            logger.info(f"文件已存在，跳过下载: {filename}")
            return True
        
        # 重试下载逻辑
        for attempt in range(self.max_retries + 1):
            try:
                # 下载文件
                headers = {'User-Agent': random.choice(self.user_agents)}
                
                if attempt == 0:
                    logger.info(f"开始下载: {filename}")
                else:
                    logger.info(f"重试下载 (第{attempt}次): {filename}")
                
                response = requests.get(url, headers=headers, timeout=self.timeout, stream=True)
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
                if attempt < self.max_retries:
                    logger.warning(f"下载失败 (第{attempt + 1}次尝试) {url}: {str(e)}，2秒后重试...")
                    time.sleep(2)
                else:
                    logger.error(f"下载最终失败 {url}: {str(e)} (已重试{self.max_retries}次)")
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
    
    def download_pdfs_from_list(self, pdf_info_list: List[Union[str, Dict]]) -> Dict[str, int]:
        """
        根据PDF信息列表批量下载PDF文件
        
        Args:
            pdf_info_list: PDF信息列表，支持以下格式：
                - 字符串列表: ["url1", "url2", ...]
                - 字典列表: [{"url": "url1", "title": "title1"}, ...]
                
        Returns:
            Dict[str, int]: 下载结果统计 {"success": int, "failed": int}
        """
        if not pdf_info_list:
            logger.warning("没有提供PDF链接")
            return {"success": 0, "failed": 0}
        
        # 统一格式处理
        normalized_list = []
        for item in pdf_info_list:
            if isinstance(item, str):
                normalized_list.append({"url": item, "title": None})
            elif isinstance(item, dict):
                if "url" not in item:
                    logger.warning(f"跳过无效项（缺少url字段）: {item}")
                    continue
                normalized_list.append(item)
            else:
                logger.warning(f"跳过无效项（格式不支持）: {item}")
                continue
        
        logger.info(f"开始下载 {len(normalized_list)} 个PDF文件到 {self.save_dir} 目录")
        
        success_count = 0
        failed_count = 0
        
        for i, pdf_info in enumerate(normalized_list, 1):
            try:
                pdf_url = pdf_info["url"]
                title = pdf_info.get("title")
                
                logger.info(f"进度: {i}/{len(normalized_list)} - {pdf_url}")
                if title:
                    logger.info(f"论文标题: {title}")
                
                # 生成文件名
                if title:
                    filename = self.create_filename_from_title(title, i, self.prefix_index)
                else:
                    filename = f"paper_{i:03d}_{self.extract_filename_from_url(pdf_url, f'paper_{i:03d}')}"
                
                # 下载文件
                if self.download_single_pdf(pdf_url, filename):
                    success_count += 1
                else:
                    failed_count += 1
                
                # 添加延迟避免过于频繁的请求
                if i < len(normalized_list):  # 最后一个文件不需要延迟
                    time.sleep(self.delay)
                    
            except KeyboardInterrupt:
                logger.info("用户中断下载")
                break
            except Exception as e:
                logger.error(f"处理链接失败 {pdf_info.get('url', 'unknown')}: {str(e)}")
                failed_count += 1
        
        logger.info(f"下载完成！成功: {success_count}, 失败: {failed_count}")
        return {"success": success_count, "failed": failed_count}
    
    def get_download_stats(self) -> Dict[str, Union[int, float]]:
        """
        获取下载目录的统计信息
        
        Returns:
            Dict: 统计信息 {"count": int, "total_size_mb": float, "files": List[str]}
        """
        if not os.path.exists(self.save_dir):
            logger.warning(f"下载目录不存在: {self.save_dir}")
            return {"count": 0, "total_size_mb": 0, "files": []}
        
        # 获取所有PDF文件
        pdf_files = []
        for filename in os.listdir(self.save_dir):
            if filename.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.save_dir, filename))
        
        total_size = 0
        file_list = []
        
        for pdf_file in pdf_files:
            try:
                size = os.path.getsize(pdf_file)
                total_size += size
                file_list.append({
                    "name": os.path.basename(pdf_file),
                    "size_mb": size / (1024 * 1024)
                })
            except OSError:
                pass
        
        total_size_mb = total_size / (1024 * 1024)
        
        logger.info(f"下载目录: {self.save_dir}")
        logger.info(f"已下载PDF文件数量: {len(pdf_files)}")
        logger.info(f"总文件大小: {total_size_mb:.2f} MB")
        
        return {
            "count": len(pdf_files),
            "total_size_mb": total_size_mb,
            "files": file_list
        }


# 便捷函数接口
def download_pdfs(pdf_info_list: List[Union[str, Dict]], 
                  save_dir: str = "downloads",
                  delay: float = 1.0,
                  max_retries: int = 3,
                  timeout: int = 30,
                  prefix_index: str = None) -> Dict[str, int]:
    """
    便捷函数：下载PDF列表
    
    Args:
        pdf_info_list: PDF信息列表
        save_dir: 保存目录
        delay: 下载间隔（秒）
        max_retries: 最大重试次数
        timeout: 请求超时时间（秒）
        prefix_index: 额外的XXX索引，用于文件命名
        
    Returns:
        Dict[str, int]: 下载结果统计
    """
    downloader = PDFDownloader(
        save_dir=save_dir,
        delay=delay,
        max_retries=max_retries,
        timeout=timeout,
        prefix_index=prefix_index
    )
    
    return downloader.download_pdfs_from_list(pdf_info_list)


def download_single_pdf(url: str, 
                       save_dir: str = "downloads",
                       filename: str = None,
                       max_retries: int = 3,
                       timeout: int = 30,
                       prefix_index: str = None) -> bool:
    """
    便捷函数：下载单个PDF文件
    
    Args:
        url: PDF的URL
        save_dir: 保存目录
        filename: 文件名（可选）
        max_retries: 最大重试次数
        timeout: 请求超时时间（秒）
        prefix_index: 额外的XXX索引，用于文件命名
        
    Returns:
        bool: 下载是否成功
    """
    downloader = PDFDownloader(
        save_dir=save_dir,
        max_retries=max_retries,
        timeout=timeout,
        prefix_index=prefix_index
    )
    
    return downloader.download_single_pdf(url, filename) 