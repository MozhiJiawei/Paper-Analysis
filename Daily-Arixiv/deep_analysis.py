#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度分析脚本 - 下载AI推理相关论文并生成技术简报
根据解析结果中大模型判别为相关的论文，批量下载PDF文件，解析内容，并生成AI推理加速技术简报

功能:
1. 下载相关论文PDF文件
2. 解析PDF提取标题、作者、摘要、组织信息
3. 为每篇论文生成技术简报
4. 生成最终的Markdown格式技术分析报告

使用方法:
python deep_analysis.py parse_result_directory

其中 parse_result_directory 是包含 ai_inference_related_papers.csv 的目录路径
"""

import os
import sys
import csv
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# 添加utils路径到sys.path以导入模块
current_dir = Path(__file__).parent
utils_dir = current_dir.parent / "utils"
sys.path.insert(0, str(current_dir.parent))

from utils.pdf_downloader.arxiv_downloader import download_pdfs_from_arxiv
from utils.pdf_extractor.pdf_extractor import extract_paper_abstract, abstract_parser
from utils.doubao_api.doubao import call_doubao

def setup_logging(debug: bool = False) -> logging.Logger:
    """设置日志配置"""
    logger = logging.getLogger(__name__)
    
    # 根据debug参数设置日志级别
    log_level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(log_level)
    
    # 避免重复添加handler
    if not logger.handlers:
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # 格式化
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
    
    return logger

def parse_csv_file(csv_path: str) -> Tuple[List[str], List[Dict]]:
    """
    解析CSV文件，提取相关论文信息
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        Tuple[List[str], List[Dict]]: (相关论文标题列表, 所有论文信息列表)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV文件不存在: {csv_path}")
    
    related_papers = []
    all_papers = []
    
    with open(csv_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            paper_info = {
                'title': row.get('标题', '').strip(),
                'relevance_judgment': row.get('大模型相关性判断', '').strip(),
                'summary': row.get('大模型总结', '').strip(),
                'score': row.get('匹配分数', '0')
            }
            
            all_papers.append(paper_info)
            
            # 判断是否相关 - 查找"相关"关键字且不包含"不相关"
            relevance = paper_info['relevance_judgment'].lower()
            if ('相关' in relevance and '不相关' not in relevance) or relevance.startswith('相关'):
                related_papers.append(paper_info['title'])
    
    return related_papers, all_papers

def create_related_paper_directory(parse_result_dir: str) -> str:
    """
    在parse_result目录下创建related_paper目录
    
    Args:
        parse_result_dir: 解析结果目录路径
        
    Returns:
        str: related_paper目录路径
    """
    parse_result_path = Path(parse_result_dir)
    related_paper_dir = parse_result_path / "related_paper"
    
    # 创建目录
    related_paper_dir.mkdir(exist_ok=True)
    
    return str(related_paper_dir)

def check_existing_pdfs(download_dir: str) -> set:
    """
    检查下载目录中已存在的PDF文件
    
    Args:
        download_dir: 下载目录路径
        
    Returns:
        set: 已存在的PDF文件名集合（不含扩展名）
    """
    existing_files = set()
    if os.path.exists(download_dir):
        for filename in os.listdir(download_dir):
            if filename.endswith('.pdf'):
                # 移除扩展名并清理文件名，用于匹配
                base_name = os.path.splitext(filename)[0]
                existing_files.add(base_name)
    return existing_files

def filter_papers_for_download(related_papers: List[str], download_dir: str, logger: logging.Logger) -> Tuple[List[str], List[str]]:
    """
    过滤需要下载的论文，排除已存在的文件
    
    Args:
        related_papers: 相关论文标题列表
        download_dir: 下载目录
        logger: 日志记录器
        
    Returns:
        Tuple[List[str], List[str]]: (需要下载的论文列表, 已存在的论文列表)
    """
    existing_files = check_existing_pdfs(download_dir)
    
    papers_to_download = []
    papers_already_exist = []
    
    for paper_title in related_papers:
        # 清理论文标题，与arxiv_downloader中的逻辑保持一致
        safe_title = "".join(c for c in paper_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_title = safe_title.replace(' ', '_')
        
        # 检查是否已存在（支持带编号的文件）
        file_exists = False
        for existing_file in existing_files:
            if existing_file == safe_title or existing_file.startswith(f"{safe_title}_"):
                file_exists = True
                break
        
        if file_exists:
            papers_already_exist.append(paper_title)
        else:
            papers_to_download.append(paper_title)
    
    return papers_to_download, papers_already_exist

def find_csv_file(parse_result_dir: str) -> str:
    """
    在指定目录中查找ai_inference_related_papers.csv文件
    
    Args:
        parse_result_dir: 解析结果目录路径
        
    Returns:
        str: CSV文件的完整路径
        
    Raises:
        FileNotFoundError: 如果找不到CSV文件
    """
    # 首先检查根目录
    csv_path = os.path.join(parse_result_dir, "ai_inference_related_papers.csv")
    if os.path.exists(csv_path):
        return csv_path
    
    # 如果根目录没有，则在子目录中查找
    if os.path.exists(parse_result_dir):
        for item in os.listdir(parse_result_dir):
            item_path = os.path.join(parse_result_dir, item)
            if os.path.isdir(item_path):
                # 检查是否是ai_acceleration_analysis_开头的目录
                if item.startswith('ai_acceleration_analysis_'):
                    sub_csv_path = os.path.join(item_path, "ai_inference_related_papers.csv")
                    if os.path.exists(sub_csv_path):
                        return sub_csv_path
    
    # 如果都没找到，抛出异常
    raise FileNotFoundError(f"在目录 {parse_result_dir} 及其子目录中未找到 ai_inference_related_papers.csv 文件")

def filter_and_download_papers(parse_result_dir: str, logger: logging.Logger) -> Dict[str, int]:
    """
    主要处理函数：过滤相关论文并下载
    
    Args:
        parse_result_dir: 解析结果目录路径
        logger: 日志记录器
        
    Returns:
        Dict[str, int]: 下载结果统计
    """
    # 1. 查找CSV文件路径
    csv_path = find_csv_file(parse_result_dir)
    logger.info(f"找到CSV文件: {csv_path}")
    
    logger.info(f"正在解析CSV文件: {csv_path}")
    
    # 2. 解析CSV文件
    related_papers, all_papers = parse_csv_file(csv_path)
    
    logger.info(f"总论文数量: {len(all_papers)}")
    logger.info(f"相关论文数量: {len(related_papers)}")
    
    if not related_papers:
        logger.warning("未找到相关论文，无需下载")
        return {}
    
    # 3. 创建下载目录
    download_dir = create_related_paper_directory(parse_result_dir)
    logger.info(f"创建下载目录: {download_dir}")
    
    # 4. 检查重复文件并过滤
    papers_to_download, papers_already_exist = filter_papers_for_download(related_papers, download_dir, logger)
    
    # 5. 打印论文列表信息
    logger.info("=== 论文下载状态检查 ===")
    logger.info(f"检测到相关论文: {len(related_papers)} 篇")
    logger.info(f"已存在文件: {len(papers_already_exist)} 篇")
    logger.info(f"需要下载: {len(papers_to_download)} 篇")
    
    if papers_already_exist:
        logger.info("以下论文已存在，跳过下载:")
        for i, title in enumerate(papers_already_exist, 1):
            logger.info(f"  {i}. {title}")
    
    if papers_to_download:
        logger.info("准备下载以下论文:")
        for i, title in enumerate(papers_to_download, 1):
            logger.info(f"  {i}. {title}")
    
    # 6. 开始下载（只下载不存在的论文）
    download_results = {}
    
    # 为已存在的文件添加"跳过"状态（使用状态码3表示跳过）
    for paper in papers_already_exist:
        download_results[paper] = 3  # 3: 已存在，跳过下载
    
    if papers_to_download:
        logger.info("开始下载新论文PDF...")
        new_download_results = download_pdfs_from_arxiv(papers_to_download, download_dir)
        download_results.update(new_download_results)
    else:
        logger.info("所有相关论文都已存在，无需下载新文件")
    
    # 7. 统计下载结果
    success_count = sum(1 for status in download_results.values() if status == 0)
    not_found_count = sum(1 for status in download_results.values() if status == 1)
    failed_count = sum(1 for status in download_results.values() if status == 2)
    skipped_count = sum(1 for status in download_results.values() if status == 3)
    
    logger.info("=== 下载结果统计 ===")
    logger.info(f"下载成功: {success_count} 篇")
    logger.info(f"未找到: {not_found_count} 篇")
    logger.info(f"下载失败: {failed_count} 篇")
    logger.info(f"已存在跳过: {skipped_count} 篇")
    logger.info(f"总计: {len(download_results)} 篇")
    
    # 8. 详细结果记录
    if download_results:
        logger.info("=== 详细下载结果 ===")
        for paper, status in download_results.items():
            status_text = {0: "成功", 1: "未找到", 2: "失败", 3: "已存在"}
            logger.info(f"{status_text[status]}: {paper}")
    
    return download_results

def parse_downloaded_pdfs(download_dir: str, download_results: Dict[str, int], logger: logging.Logger) -> Dict[str, Dict]:
    """
    解析下载成功的PDF文件
    
    Args:
        download_dir: PDF下载目录
        download_results: 下载结果字典
        logger: 日志记录器
        
    Returns:
        Dict[str, Dict]: 解析结果字典，key为论文标题，value为解析的论文信息
    """
    parsed_results = {}

    abstract_parser(download_dir)

    # 遍历下载目录中的所有PDF文件
    for filename in os.listdir(download_dir):
        if not filename.endswith('.pdf'):
            continue
            
        pdf_path = os.path.join(download_dir, filename)
        logger.info(f"正在解析: {filename}")
        
        try:
            # 调用extract_paper_abstract解析PDF
            paper_info = extract_paper_abstract(pdf_path)
            
            # 使用文件名（去掉.pdf扩展名）作为key
            paper_key = os.path.splitext(filename)[0]
            parsed_results[paper_key] = paper_info
            
            logger.info(f"解析成功: {filename}")
            logger.info(f"  - 标题: {paper_info.get('title', 'N/A')}")
            logger.info(f"  - 作者数量: {len(paper_info.get('authors', []))}")
            logger.info(f"  - 摘要长度: {len(paper_info.get('abstract', ''))}")
            
        except Exception as e:
            logger.error(f"解析PDF文件失败 {filename}: {str(e)}")
            continue
    
    logger.info(f"PDF解析完成，成功解析 {len(parsed_results)} 个文件")
    return parsed_results



def generate_paper_brief(paper_info: Dict, logger: logging.Logger) -> str:
    """
    为单篇论文生成简报
    
    Args:
        paper_info: 论文信息字典
        logger: 日志记录器
        
    Returns:
        str: 生成的简报内容
    """
    title = paper_info.get('title', '未知标题')
    authors = paper_info.get('authors', [])
    affiliations = paper_info.get('affiliations', [])
    abstract = paper_info.get('abstract', '无摘要')
    
    logger.debug(f"正在为论文生成简报: {title}")
    logger.debug(f"作者数量: {len(authors)}")
    logger.debug(f"机构数量: {len(affiliations)}")
    logger.debug(f"摘要长度: {len(abstract)} 字符")
    
    # 提取主要组织
    main_orgs = []
    for author in authors:
        if 'affiliation' in author and author['affiliation']:
            main_orgs.extend(author['affiliation'])
    main_orgs.extend(affiliations)
    unique_orgs = list(set(main_orgs))
    
    logger.debug(f"提取到的机构: {unique_orgs}")
    
    # 构建prompt
    prompt = f"""请你作为AI技术专家，为以下论文生成一份简洁的技术简报。

论文信息:
标题: {title}
主要机构: {', '.join(unique_orgs[:3])}
摘要: {abstract}

请按照以下格式生成简报:
"XXX公司/学校/组织发布了XXX论文，使用XXX技术，解决了XXX问题，达成了XXX效果"

要求:
1. 简洁明了，一句话概括
2. 突出主要技术创新点
3. 说明解决的核心问题
4. 提及关键效果或性能提升
5. 如果涉及多个机构，选择最知名的1-2个

请直接返回简报内容，不需要额外说明。"""
    
    prompt_length = len(prompt)
    logger.debug(f"简报prompt长度: {prompt_length} 字符")
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        logger.debug(f"正在为论文《{title}》调用豆包API生成简报...")
        result = call_doubao(messages)
        
        logger.debug(f"论文《{title}》豆包API响应已接收")
        
        if result["success"]:
            brief_content = result["content"].strip()
            logger.debug(f"论文《{title}》简报生成成功，长度: {len(brief_content)} 字符")
            return brief_content
        else:
            logger.error(f"生成论文简报失败: {result['error']}")
            logger.error(f"论文: {title}")
            logger.error(f"完整错误响应: {result}")
            return f"论文《{title}》简报生成失败"
    except Exception as e:
        logger.error(f"调用豆包API生成简报时发生异常: {str(e)}")
        logger.error(f"论文: {title}")
        logger.error(f"异常类型: {type(e).__name__}")
        import traceback
        logger.error(f"异常堆栈: {traceback.format_exc()}")
        return f"论文《{title}》简报生成异常"

def generate_final_report(parsed_results: Dict[str, Dict], output_dir: str, logger: logging.Logger) -> str:
    """
    生成最终报告
    
    Args:
        parsed_results: 解析的论文结果字典
        output_dir: 输出目录
        logger: 日志记录器
        
    Returns:
        str: 报告文件路径
    """
    logger.info("正在生成最终技术简报...")
    
    # 获取论文key列表（使用原始顺序）
    paper_keys = list(parsed_results.keys())
    
    # 生成每篇论文的简报
    paper_briefs = {}
    for i, paper_key in enumerate(paper_keys, 1):
        paper_info = parsed_results[paper_key]
        logger.info(f"正在生成第{i}篇论文简报...")
        brief = generate_paper_brief(paper_info, logger)
        paper_briefs[paper_key] = brief
    
    # 构建完整报告
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = []
    report_content.append("# AI推理加速技术论文分析报告")
    report_content.append(f"生成时间: {current_time}")
    report_content.append(f"分析论文数量: {len(parsed_results)}篇")
    report_content.append("")
    
    # 1. 论文技术简报部分
    report_content.append("## 论文技术简报")
    report_content.append("")
    
    for i, paper_key in enumerate(paper_keys, 1):
        paper_info = parsed_results[paper_key]
        title = paper_info.get('title', '未知标题')
        brief = paper_briefs[paper_key]
        
        report_content.append(f"### {i}. {title}")
        report_content.append("")
        report_content.append(brief)
        report_content.append("")
    
    # 2. 详细信息部分
    report_content.append("## 论文详细信息")
    report_content.append("")
    
    for i, paper_key in enumerate(paper_keys, 1):
        paper_info = parsed_results[paper_key]
        title = paper_info.get('title', '未知标题')
        authors = paper_info.get('authors', [])
        affiliations = paper_info.get('affiliations', [])
        abstract = paper_info.get('abstract', '无摘要')
        
        # 提取组织信息
        orgs = []
        for author in authors:
            if 'affiliation' in author and author['affiliation']:
                orgs.extend(author['affiliation'])
        orgs.extend(affiliations)
        unique_orgs = list(set(orgs))
        
        report_content.append(f"### {i}. {title}")
        report_content.append("")
        report_content.append(f"**主要机构**: {', '.join(unique_orgs)}")
        report_content.append(f"**作者数量**: {len(authors)}人")
        report_content.append("")
        report_content.append("**摘要**:")
        report_content.append(abstract)
        report_content.append("")
    
    # 保存报告文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"ai_inference_report_{timestamp}.md"
    report_path = os.path.join(output_dir, report_filename)
    
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_content))
        
        logger.info(f"技术简报已保存到: {report_path}")
        return report_path
    except Exception as e:
        logger.error(f"保存报告文件失败: {str(e)}")
        return ""

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="下载AI推理相关论文PDF并生成技术简报",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法 - 指定到包含CSV文件的具体目录
  python deep_analysis.py "2025-09/09-01"
  
  # 启用调试模式（显示详细日志）
  python deep_analysis.py "2025-09/09-01" --debug
  
  
功能说明:
  1. 下载相关论文PDF文件
  2. 解析PDF提取标题、作者、摘要、组织信息
  3. 为每篇论文生成技术简报
  4. 生成最终的Markdown格式技术分析报告

调试选项:
  --debug        启用调试模式，显示详细的API调用信息、网络请求细节、token使用情况等

注意:
  - parse_result目录应包含 ai_inference_related_papers.csv 文件
  - 相关论文将下载到parse_result目录下的related_paper文件夹中
  - 脚本会基于"大模型相关性判断"列过滤相关论文
  - 最终技术简报将保存在parse_result目录下，文件名为 ai_inference_report_YYYYMMDD_HHMMSS.md
        """
    )
    
    parser.add_argument(
        'parse_result_dir',
        help='解析结果目录路径，例如：2025-09/09-01'
    )
    
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='启用调试模式，显示详细日志信息'
    )
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.debug)
    logger.info("启动深度分析脚本")
    logger.info(f"解析结果目录: {args.parse_result_dir}")
    
    try:
        # 验证输入目录
        if not os.path.exists(args.parse_result_dir):
            raise FileNotFoundError(f"解析结果目录不存在: {args.parse_result_dir}")
        
        if not os.path.isdir(args.parse_result_dir):
            raise NotADirectoryError(f"指定路径不是目录: {args.parse_result_dir}")
        
        # 执行主要处理
        download_results = filter_and_download_papers(args.parse_result_dir, logger)

        download_dir = create_related_paper_directory(args.parse_result_dir)

        # 解析PDF文件
        parsed_results = parse_downloaded_pdfs(download_dir, download_results, logger)

        # 生成技术简报
        if parsed_results:
            logger.info("开始生成AI推理加速技术简报...")
            logger.info(f"成功解析的论文数量: {len(parsed_results)}")
            
            # 显示解析成功的论文概览
            logger.info("解析成功的论文列表:")
            for i, (paper_key, paper_info) in enumerate(parsed_results.items(), 1):
                title = paper_info.get('title', '未知标题')
                logger.info(f"  {i}. {title}")
            
            # 生成最终报告 - 保存到parse_result目录而不是download_dir
            logger.info("=== 生成最终报告 ===")
            report_path = generate_final_report(parsed_results, args.parse_result_dir, logger)
            
            if report_path:
                logger.info("=== 技术简报生成完成 ===")
                logger.info(f"简报文件路径: {report_path}")
            else:
                logger.warning("技术简报生成失败")
        else:
            logger.warning("无PDF解析结果，跳过技术简报生成")

        # 输出最终结果
        if download_results:
            success_count = sum(1 for status in download_results.values() if status == 0)
            skipped_count = sum(1 for status in download_results.values() if status == 3)
            total_count = len(download_results)
            
            logger.info(f"任务完成！成功下载 {success_count} 篇新论文，跳过 {skipped_count} 篇已存在论文，总计 {total_count} 篇相关论文")
            if 'parsed_results' in locals():
                logger.info(f"成功解析 {len(parsed_results)} 个PDF文件")
                if 'report_path' in locals() and report_path:
                    logger.info(f"技术简报已生成: {report_path}")
        else:
            logger.info("任务完成！未找到需要下载的相关论文")
            
    except Exception as e:
        logger.error(f"脚本执行失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
