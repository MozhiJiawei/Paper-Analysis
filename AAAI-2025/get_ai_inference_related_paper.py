import os
from typing import List, Dict
import sys
from utils.ai_acceleration_extractor import AiAccelerationExtractor

def get_ai_papers_using_parse(papers_dir: str, oral_list_file: str = None, analyze_all: bool = False):
    """
    使用AiAccelerationExtractor的parse方法分析论文，筛选出与AI推理加速相关的论文
    
    Args:
        papers_dir: 论文文件夹路径
        oral_list_file: oral论文列表文件路径 (当analyze_all=False时使用)
        analyze_all: 是否分析全量论文，False表示只分析oral论文
    """
    
    # 创建提取器实例
    extractor = AiAccelerationExtractor(papers_dir, ".")
    
    if analyze_all:
        print("开始分析路径下的全量论文，筛选AI推理加速相关论文...")
        # 使用parse方法分析全量论文
        extractor.parse(analyze_all=True, output_format="csv")
    else:
        print("开始分析所有oral论文，筛选AI推理加速相关论文...")
        
        # 直接从oral_paper_filenames.txt文件中读取oral论文文件名
        oral_filenames_file = "oral_paper_filenames.txt"
        paper_filenames = []
        if os.path.exists(oral_filenames_file):
            with open(oral_filenames_file, 'r', encoding='utf-8') as f:
                paper_filenames = [line.strip() for line in f if line.strip()]
            print(f"从 {oral_filenames_file} 加载了 {len(paper_filenames)} 个oral论文文件名")
        else:
            print(f"错误: 找不到文件 {oral_filenames_file}")
            return
        
        if not paper_filenames:
            print("未找到匹配的oral论文文件")
            return
        
        # 使用parse方法分析指定的oral论文
        extractor.parse(paper_filenames=paper_filenames, analyze_all=False, output_format="csv")

def main():
    """
    主函数，筛选AI推理加速相关的oral论文
    """
    # 设置文件路径
    papers_dir = "AAAI-2025-Papers"
    oral_list_file = "oral_paper_list.txt"
    
    # 检查文件是否存在
    if not os.path.exists(oral_list_file):
        print(f"错误: 找不到文件 {oral_list_file}")
        return
    
    if not os.path.exists(papers_dir):
        print(f"错误: 找不到论文文件夹 {papers_dir}")
        return
    
    # 使用parse方法分析论文
    get_ai_papers_using_parse(papers_dir, oral_list_file, analyze_all=False)

if __name__ == "__main__":
    main()
