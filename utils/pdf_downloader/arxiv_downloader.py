import arxiv
import os
import ssl
import urllib.request
import logging
from typing import List, Dict
from pathlib import Path

# 获取模块级别的logger
logger = logging.getLogger(__name__)

# 全局变量跟踪SSL是否已配置
_ssl_configured = False


def _configure_ssl_for_arxiv():
    """
    配置SSL设置以解决arXiv下载的证书验证问题
    在一次脚本执行中仅调用一次
    """
    global _ssl_configured
    
    # 如果已经配置过，直接返回
    if _ssl_configured:
        return True
    
    try:
        # 尝试创建一个更宽松的SSL上下文
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # 设置urllib的全局SSL处理器
        https_handler = urllib.request.HTTPSHandler(context=ssl_context)
        opener = urllib.request.build_opener(https_handler)
        urllib.request.install_opener(opener)
        
        _ssl_configured = True  # 标记为已配置
        logger.info("已配置SSL设置以处理证书验证问题")
        return True
    except Exception as e:
        logger.error(f"SSL配置失败: {e}")
        return False


def download_pdfs_from_arxiv(paper_name_list: List[str], 
    save_dir: str = "downloads") -> Dict[str, int]:
    """
    批量下载arXiv论文
    
    Args:
        paper_name_list: 论文名称列表
        save_dir: 保存目录
        
    Returns:
        Dict[str, int]: 每篇论文的下载状态
            0: 下载成功
            1: 搜索失败（未找到论文）
            2: 下载失败
    """
    results = {}
    
    # 配置SSL设置
    _configure_ssl_for_arxiv()
    
    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    for paper_name in paper_name_list:
        logger.info(f"正在处理论文: {paper_name}")
        result = download_single_pdf_from_arxiv(paper_name, save_dir)
        results.update(result)
    
    return results


def download_single_pdf_from_arxiv(paper_name: str, 
    save_dir: str = "downloads") -> Dict[str, int]:
    """
    下载单篇arXiv论文
    
    Args:
        paper_name: 论文名称或arXiv ID
        save_dir: 保存目录
        
    Returns:
        Dict[str, int]: 论文下载状态
            0: 下载成功
            1: 搜索失败（未找到论文）
            2: 下载失败
    """
    try:
        # 配置SSL设置
        _configure_ssl_for_arxiv()
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 创建搜索客户端
        client = arxiv.Client()
        
        # 如果是arXiv ID格式（如2301.12345），直接按ID搜索
        if paper_name.replace('.', '').replace('v', '').isdigit() or 'arxiv' in paper_name.lower():
            # 提取纯数字ID
            arxiv_id = paper_name.replace('arxiv:', '').replace('arXiv:', '').strip()
            search = arxiv.Search(id_list=[arxiv_id])
            papers = list(client.results(search))
        else:
            # 先尝试精确标题搜索
            keywords = paper_name.replace(':', '').replace('-', ' ')
            search = arxiv.Search(
                query=f'ti:"{keywords}"',  # ti: 表示在标题中搜索
                max_results=5,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            # 执行精确搜索
            papers = list(client.results(search))
            
            # 如果精确搜索失败，尝试更宽松的关键词搜索
            if not papers:
                logger.info(f"精确标题搜索失败，尝试关键词搜索...")
                # 提取关键词进行搜索
                search = arxiv.Search(
                    query=keywords,
                    max_results=10,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                # 重新执行搜索
                papers = list(client.results(search))
        
        if not papers:
            logger.warning(f"未找到论文: {paper_name}")
            return {paper_name: 1}
        
        # 取第一个最相关的结果
        paper = papers[0]
        
        # 清理文件名，移除特殊字符
        safe_title = "".join(c for c in paper.title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')
        
        # 构建文件路径
        filename = f"{safe_title}.pdf"
        filepath = os.path.join(save_dir, filename)
        
        # 如果文件已存在，添加编号
        counter = 1
        while os.path.exists(filepath):
            base_name = safe_title
            filename = f"{base_name}_{counter}.pdf"
            filepath = os.path.join(save_dir, filename)
            counter += 1
        
        logger.info(f"正在下载: {paper.title}")
        logger.info(f"作者: {', '.join([author.name for author in paper.authors])}")
        logger.info(f"发布日期: {paper.published}")
        logger.info(f"保存路径: {filepath}")
        
        # 下载PDF
        paper.download_pdf(dirpath=save_dir, filename=filename)
        
        logger.info(f"下载成功: {filename}")
        return {paper_name: 0}
        
    except Exception as e:
        logger.error(f"下载论文 '{paper_name}' 时发生错误: {str(e)}")
        return {paper_name: 2}
