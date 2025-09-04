"""
arXiv Daily论文解析器
解析arxiv_daily.txt文件，提取论文信息并生成ai_acceleration_parse_paper_copilot函数的输入格式
"""

import re
import os
import sys
import random
import argparse
from typing import List, Dict, Optional
from datetime import datetime

# 添加utils路径以便导入ai_acceleration_extractor
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
from utils.ai_acceleration_extractor.ai_acceleration_extractor import ai_acceleration_parse_paper_copilot


def parse_arxiv_daily(file_path: str) -> List[Dict]:
    """
    解析arxiv_daily.txt文件，提取论文信息
    
    Args:
        file_path: arxiv_daily.txt文件路径
    
    Returns:
        List[Dict]: 包含论文信息的字典列表，格式适用于ai_acceleration_parse_paper_copilot函数
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")
    
    papers = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 使用正则表达式分割论文条目
    # 每个论文条目以 arXiv: 开始，到下一个 arXiv: 或文件结尾结束
    # 先按 arXiv: 分割，然后处理每个部分
    sections = re.split(r'(?=arXiv:\d{4}\.\d+)', content)
    
    matches = []
    for section in sections[1:]:  # 跳过第一个空段
        if section.strip():
            matches.append(section)
    
    for match in matches:
        paper_info = _parse_single_paper(match)
        if paper_info:
            papers.append(paper_info)
    
    return papers


def _parse_single_paper(paper_text: str) -> Optional[Dict]:
    """
    解析单篇论文的文本信息
    
    Args:
        paper_text: 单篇论文的文本内容
    
    Returns:
        Dict: 论文信息字典，如果解析失败则返回None
    """
    try:
        lines = paper_text.strip().split('\n')
        
        # 提取arXiv ID
        arxiv_id_match = re.search(r'arXiv:(\d{4}\.\d+)', lines[0])
        if not arxiv_id_match:
            return None
        arxiv_id = arxiv_id_match.group(1)
        
        # 初始化论文信息
        paper_info = {
            'filename': f"arxiv_{arxiv_id}.pdf",
            'arxiv_id': arxiv_id,
            'title': '',
            'authors': '',
            'categories': '',
            'abstract': '',
            'date': '',
            'url': f"https://arxiv.org/abs/{arxiv_id}"
        }
        
        # 解析各个字段
        current_field = None
        abstract_lines = []
        
        for line in lines[1:]:  # 跳过第一行（已处理的arXiv ID）
            line = line.strip()
            
            if line.startswith('Date:'):
                # 提取日期
                date_match = re.search(r'Date:\s*(.+?GMT)', line)
                if date_match:
                    paper_info['date'] = date_match.group(1).strip()
                continue
            
            elif line.startswith('Title:'):
                # 提取标题
                title = line.replace('Title:', '').strip()
                paper_info['title'] = title
                current_field = 'title'
                continue
            
            elif line.startswith('Authors:'):
                # 提取作者
                authors = line.replace('Authors:', '').strip()
                paper_info['authors'] = authors
                current_field = 'authors'
                continue
            
            elif line.startswith('Categories:'):
                # 提取类别
                categories = line.replace('Categories:', '').strip()
                paper_info['categories'] = categories
                current_field = 'categories'
                continue
            
            elif line.startswith('Comments:'):
                # 跳过评论行
                current_field = None
                continue
            
            elif line.startswith('DOI:'):
                # 跳过DOI行
                current_field = None
                continue
            
            elif line.startswith('ACM-class:'):
                # 跳过ACM分类行
                current_field = None
                continue
            
            elif line.startswith('\\\\'):
                # 开始摘要部分
                if len(line) > 2:  # 如果\\ 后面还有内容，那就是摘要的一部分
                    abstract_content = line[2:].strip()
                    if abstract_content and not abstract_content.startswith('( https://arxiv.org/abs/'):
                        abstract_lines.append(abstract_content)
                current_field = 'abstract'
                continue
            
            else:
                # 处理多行内容
                if current_field == 'title' and line and not line.startswith('Authors:') and not line.startswith('Categories:'):
                    # 标题可能跨行
                    paper_info['title'] += ' ' + line
                elif current_field == 'authors' and line and not line.startswith('Categories:') and not line.startswith('Comments:') and not line.startswith('DOI:') and not line.startswith('ACM-class:'):
                    # 作者可能跨行
                    paper_info['authors'] += ' ' + line
                elif current_field == 'abstract' and line:
                    # 摘要内容 - 过滤掉URL和分割线
                    if not (line.startswith('( https://arxiv.org/abs/') or 
                           line.startswith('----------') or
                           'https://arxiv.org/abs/' in line):
                        abstract_lines.append(line)
        
        # 清理和组装摘要
        if abstract_lines:
            paper_info['abstract'] = ' '.join(abstract_lines).strip()
            # 移除摘要中的多余分割线
            paper_info['abstract'] = re.sub(r'-{5,}', '', paper_info['abstract']).strip()
        
        # 清理标题和作者字段
        paper_info['title'] = ' '.join(paper_info['title'].split())
        paper_info['authors'] = ' '.join(paper_info['authors'].split())
        
        # 验证和修正URL
        if paper_info['arxiv_id'] and not paper_info['url'].endswith(paper_info['arxiv_id']):
            paper_info['url'] = f"https://arxiv.org/abs/{paper_info['arxiv_id']}"
        
        # 验证必要字段
        if not paper_info['title']:
            print(f"警告: 论文 {arxiv_id} 缺少标题")
            return None
        
        if not paper_info['abstract']:
            print(f"警告: 论文 {arxiv_id} 缺少摘要")
            
        return paper_info
    
    except Exception as e:
        print(f"解析论文时出错: {e}")
        print(f"问题文本前100字符: {paper_text[:100]}...")
        return None


def load_papers_for_ai_analysis(arxiv_daily_file: str) -> List[Dict]:
    """
    加载arXiv daily文件并返回适用于AI加速分析的论文信息列表
    
    Args:
        arxiv_daily_file: arxiv_daily.txt文件路径
    
    Returns:
        List[Dict]: 论文信息列表，可直接用于ai_acceleration_parse_paper_copilot函数
    """
    print(f"正在解析 {arxiv_daily_file}...")
    papers = parse_arxiv_daily(arxiv_daily_file)
    print(f"成功解析 {len(papers)} 篇论文")
    
    return papers


def analyze_ai_acceleration_papers(papers: List[Dict], parse_dir: str = None) -> None:
    """
    分析AI推理加速相关论文
    
    Args:
        papers: 论文信息列表
        parse_dir: 解析目录路径（如 "2025-09/09-01"），ai_acceleration_results将保存在此目录下
    """
    if not papers:
        print("❌ 没有论文数据需要分析")
        return
    

    # 使用指定的解析目录作为基础，在其下创建ai_acceleration_results文件夹
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(current_dir, parse_dir)

    print(f"\n🚀 开始AI推理加速论文筛选分析...")
    print(f"📊 总论文数: {len(papers)}")
    print(f"📁 输出目录: {output_dir}")
    
    try:
        # 调用ai_acceleration_parse_paper_copilot进行分析
        analysis_result = ai_acceleration_parse_paper_copilot(
            paper_infos=papers,
            output_dir=output_dir,
            output_format="both",
            enable_llm_judge=True,
            match_threshold=5
        )
        
        print(f"\n✅ AI推理加速论文筛选完成!")
        print(f"🎯 找到AI推理加速相关论文: {analysis_result.ai_related_count} 篇")
        print(f"📄 其他论文: {analysis_result.non_ai_related_count} 篇")
        print(f"💾 详细结果已保存到: {output_dir}")
        
        return analysis_result
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        import traceback
        print(f"错误详情: {traceback.format_exc()}")


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description="arXiv Daily论文解析器 - 解析arxiv_daily.txt文件并进行AI推理加速论文筛选",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python parser.py --parse_dir "2025-09/09-01"
  python parser.py --parse_dir "2025-09/09-01" --sample_size 50
        """
    )
    
    parser.add_argument(
        "--parse_dir",
        required=True,
        help="指定要解析的目录路径（相对于当前文件的位置），例如: '2025-09/09-01'"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=30,
        help="随机抽检展示的论文数量（默认: 30）"
    )
    

    
    return parser.parse_args()


def main():
    """
    主函数 - 解析arxiv_daily.txt文件并进行AI推理加速论文筛选
    """
    args = parse_args()
    
    # 构建arxiv_daily.txt文件路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parse_dir_path = os.path.join(current_dir, args.parse_dir)
    arxiv_daily_file = os.path.join(parse_dir_path, "arxiv_daily.txt")
    
    # 检查文件是否存在
    if not os.path.exists(arxiv_daily_file):
        print(f"❌ 错误: 找不到文件 {arxiv_daily_file}")
        print(f"请确保在目录 '{args.parse_dir}' 下存在 'arxiv_daily.txt' 文件")
        return
    
    print(f"📂 解析目录: {args.parse_dir}")
    print(f"📄 目标文件: {arxiv_daily_file}")
    
    # 解析论文
    papers = load_papers_for_ai_analysis(arxiv_daily_file)
    
    if not papers:
        print("❌ 没有成功解析到任何论文")
        return
    
    # 显示解析结果统计
    print(f"\n📊 解析结果统计:")
    print(f"总论文数: {len(papers)}")
    
    # 随机抽检论文信息
    if papers and args.sample_size > 0:
        sample_size = min(args.sample_size, len(papers))
        sample_papers = random.sample(papers, sample_size)
        
        print(f"\n🔍 随机抽检 {sample_size} 篇论文信息:")
        for i, paper in enumerate(sample_papers, 1):
            print(f"\n=== 论文 {i} ===")
            print(f"标题: {paper['title'][:100]}{'...' if len(paper['title']) > 100 else ''}")
            print(f"作者: {paper['authors'][:80]}{'...' if len(paper['authors']) > 80 else ''}")
            print(f"类别: {paper['categories']}")
            print(f"arXiv ID: {paper['arxiv_id']}")
            print(f"日期: {paper['date']}")
            print(f"摘要长度: {len(paper['abstract'])} 字符")
            if paper['abstract']:
                print(f"摘要前100字符: {paper['abstract'][:100]}...")
            else:
                print("摘要: [缺少摘要]")
            print(f"URL: {paper['url']}")
            print("=" * 50)
    
    # 进行AI推理加速论文筛选分析
    print("\n" + "="*60)
    print("🚀 开始AI推理加速论文筛选分析...")
    print("="*60)
    analyze_ai_acceleration_papers(papers, args.parse_dir)


if __name__ == "__main__":
    main()
