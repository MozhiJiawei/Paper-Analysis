import json
import os
import argparse
from utils.ai_acceleration_extractor import ai_acceleration_parse_paper_copilot

def extract_papers_by_status(json_file_path, status_keyword):
    """
    解析JSON文件，根据指定的状态关键字提取论文
    
    Args:
        json_file_path (str): JSON文件的路径
        status_keyword (str): 要筛选的状态关键字（如 'oral', 'spotlight', 'poster' 等）
    
    Returns:
        list: 包含所有匹配状态的论文的字典列表
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as file:
            papers = json.load(file)
        
        # 过滤出Status匹配指定关键字的论文
        filtered_papers = []
        for paper in papers:
            if paper.get('status', '').lower() == status_keyword.lower():
                filtered_papers.append(paper)
        
        return filtered_papers
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file_path}")
        return []
    except json.JSONDecodeError:
        print("错误：JSON文件格式无效")
        return []
    except Exception as e:
        print(f"错误：{e}")
        return []

def print_papers_summary(papers, status_keyword, num_to_print=10):
    """
    打印指定状态论文的摘要信息
    
    Args:
        papers (list): 论文列表
        status_keyword (str): 状态关键字
        num_to_print (int): 要打印的论文数量
    """
    print(f"总共找到 {len(papers)} 篇{status_keyword.upper()}论文")
    print(f"\n前 {min(num_to_print, len(papers))} 篇{status_keyword.upper()}论文详情：")
    print("=" * 80)
    
    for i, paper in enumerate(papers[:num_to_print]):
        print(f"\n第 {i+1} 篇：")
        print(f"标题：{paper.get('title', 'N/A')}")
        print(f"状态：{paper.get('status', 'N/A')}")
        print(f"作者：{paper.get('author', 'N/A')}")
        print(f"主要领域：{paper.get('primary_area', 'N/A')}")
        print(f"关键词：{paper.get('keywords', 'N/A')}")
        print(f"ID：{paper.get('id', 'N/A')}")
        print(f"网站：{paper.get('site', 'N/A')}")
        print("-" * 80)

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='通用论文状态筛选和AI推理加速分析工具')
    parser.add_argument('json_file', help='JSON文件路径')
    parser.add_argument('status', help='要筛选的状态关键字（如 oral, spotlight, poster 等）')
    parser.add_argument('--output-dir', default=None, help='输出目录名称（默认为 {status}_Result）')
    parser.add_argument('--num-preview', type=int, default=10, help='预览显示的论文数量（默认10篇）')
    
    args = parser.parse_args()
    
    json_file_path = args.json_file
    status_keyword = args.status
    output_dir = args.output_dir or f"{status_keyword.capitalize()}_Result"
    num_preview = args.num_preview
    
    # 提取指定状态的论文
    print(f"开始解析 {json_file_path} 论文数据...")
    print(f"筛选状态: {status_keyword.upper()}")
    filtered_papers = extract_papers_by_status(json_file_path, status_keyword)
    
    if filtered_papers:
        # 打印预览结果
        print_papers_summary(filtered_papers, status_keyword, num_preview)
        
        # 保存结果到变量中
        papers_list = filtered_papers
        print(f"\n所有{status_keyword.upper()}论文已保存到papers_list变量中，共 {len(papers_list)} 篇")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 调用ai_acceleration_parse_paper_copilot方法分析所有论文
        print(f"\n开始使用AI推理加速分析器分析所有{status_keyword.upper()}论文...")
        print(f"分析结果将保存在 {output_dir} 目录中")
        
        try:
            analysis_result = ai_acceleration_parse_paper_copilot(
                paper_infos=papers_list,
                output_dir=output_dir,
                output_format="both",
                enable_llm_judge=True,
                match_threshold=5
            )
            
            print(f"\n分析完成！")
            print(f"匹配的AI推理加速相关论文数量: {len(analysis_result.ai_related_papers)}")
            print(f"未匹配的论文数量: {len(analysis_result.non_ai_related_papers)}")
            print(f"详细结果请查看 {output_dir} 目录中的报告文件")
            
        except Exception as e:
            print(f"分析过程中出现错误: {e}")
        
    else:
        print(f"没有找到{status_keyword.upper()}论文或文件读取失败")

if __name__ == "__main__":
    main()
