import json
import os
from utils.ai_acceleration_extractor import ai_acceleration_parse_paper_copilot

def extract_oral_papers(json_file_path):
    """
    解析ICLR 2025 JSON文件，提取所有Status为Oral的论文
    
    Args:
        json_file_path (str): JSON文件的路径
    
    Returns:
        list: 包含所有Oral论文的字典列表
    """
    try:
        # 读取JSON文件
        with open(json_file_path, 'r', encoding='utf-8') as file:
            papers = json.load(file)
        
        # 过滤出Status为Oral的论文
        oral_papers = []
        for paper in papers:
            if paper.get('status', '').lower() == 'poster':
                oral_papers.append(paper)
        
        return oral_papers
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file_path}")
        return []
    except json.JSONDecodeError:
        print("错误：JSON文件格式无效")
        return []
    except Exception as e:
        print(f"错误：{e}")
        return []

def print_oral_papers_summary(oral_papers, num_to_print=10):
    """
    打印Oral论文的摘要信息
    
    Args:
        oral_papers (list): Oral论文列表
        num_to_print (int): 要打印的论文数量
    """
    print(f"总共找到 {len(oral_papers)} 篇Oral论文")
    print(f"\n前 {min(num_to_print, len(oral_papers))} 篇Oral论文详情：")
    print("=" * 80)
    
    for i, paper in enumerate(oral_papers[:num_to_print]):
        print(f"\n第 {i+1} 篇：")
        print(f"标题：{paper.get('title', 'N/A')}")
        print(f"状态：{paper.get('status', 'N/A')}")
        print(f"作者：{paper.get('author', 'N/A')}")
        print(f"主要领域：{paper.get('primary_area', 'N/A')}")
        print(f"关键词：{paper.get('keywords', 'N/A')}")
        print(f"ID：{paper.get('id', 'N/A')}")
        print(f"网站：{paper.get('site', 'N/A')}")
        print("-" * 80)

if __name__ == "__main__":
    # JSON文件路径
    json_file_path = "iclr2025_famous_poster.json"
    
    # 提取Oral论文
    print("开始解析ICLR 2025论文数据...")
    oral_papers = extract_oral_papers(json_file_path)
    
    if oral_papers:
        # 打印前10个结果
        print_oral_papers_summary(oral_papers, 10)
        
        # 保存结果到变量中（按要求使用list(Dict)结构）
        oral_papers_list = oral_papers
        print(f"\n所有Oral论文已保存到oral_papers_list变量中，共 {len(oral_papers_list)} 篇")
        
        # 创建Oral_Result目录
        output_dir = "Famous_Result"
        os.makedirs(output_dir, exist_ok=True)
        
        # 调用ai_acceleration_parse_paper_copilot方法分析所有Oral论文
        print(f"\n开始使用AI推理加速分析器分析所有Oral论文...")
        print(f"分析结果将保存在 {output_dir} 目录中")
        
        try:
            analysis_result = ai_acceleration_parse_paper_copilot(
                paper_infos=oral_papers_list,
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
        print("没有找到Oral论文或文件读取失败")
