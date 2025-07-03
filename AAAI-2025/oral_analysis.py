import os
import shutil
import pandas as pd
from pathlib import Path

import sys
import os
sys.path.append('..')

from utils.pdf_extractor import abstract_parser
from utils.ai_acceleration_extractor import ai_acceleration_parse

def organize_oral_papers():
    """
    根据oral_paper_filenames.txt文件中的信息，将AAAI-2025-Papers中的ORAL论文复制到AAAI-2025-Papers-Oral目录。
    同时复制PDF和XML文件。
    """
    
    # 定义路径
    filenames_file = "oral_paper_filenames.txt"
    xml_source_dir = "AAAI-2025-Papers"
    pdf_source_dir = "AAAI-2025-Papers"
    target_dir = "AAAI-2025-Papers-Oral"
    
    # 读取文件名列表
    try:
        with open(filenames_file, 'r', encoding='utf-8') as f:
            oral_filenames = [line.strip() for line in f.readlines() if line.strip()]
        print(f"成功读取文件名列表，共{len(oral_filenames)}个ORAL论文")
    except Exception as e:
        print(f"读取文件名列表失败: {e}")
        return
    
    # 确保目标目录存在
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    # 处理每个文件
    successful_copies = 0
    failed_copies = 0
    
    for filename in oral_filenames:
        try:
            # 检查文件名是否以.pdf结尾
            if not filename.endswith('.pdf'):
                print(f"跳过非PDF文件: {filename}")
                continue
            
            # 构建XML文件名（将.pdf替换为.grobid.tei.xml）
            xml_filename = filename.replace('.pdf', '.grobid.tei.xml')
            
            # 构建源文件路径
            pdf_source_file = Path(pdf_source_dir) / filename
            xml_source_file = Path(xml_source_dir) / xml_filename
            
            # 构建目标文件路径（保持原文件名，添加ORAL_前缀）
            pdf_target_file = target_path / f"ORAL_{filename}"
            xml_target_file = target_path / f"ORAL_{xml_filename}"
            
            # 复制PDF文件
            if pdf_source_file.exists():
                shutil.copy2(pdf_source_file, pdf_target_file)
                print(f"成功复制PDF: {filename} -> ORAL_{filename}")
                pdf_copied = True
            else:
                print(f"警告: PDF文件不存在 - {pdf_source_file}")
                pdf_copied = False
            
            # 复制XML文件（如果存在）
            if xml_source_file.exists():
                shutil.copy2(xml_source_file, xml_target_file)
                print(f"成功复制XML: {xml_filename} -> ORAL_{xml_filename}")
                xml_copied = True
            else:
                print(f"警告: XML文件不存在 - {xml_source_file}")
                xml_copied = False
            
            # 如果至少复制了一个文件，算作成功
            if pdf_copied or xml_copied:
                successful_copies += 1
            else:
                failed_copies += 1
            
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            failed_copies += 1
            continue
    
    # 输出统计结果
    print(f"\n处理完成:")
    print(f"成功复制: {successful_copies} 个论文")
    print(f"失败: {failed_copies} 个论文")
    print(f"总计: {len(oral_filenames)} 个文件")

if __name__ == "__main__":
    # organize_oral_papers()
    
    # 可选：解析摘要
    # abstract_parser("AAAI-2025-Papers-Oral")

    # 可选：AI加速相关分析
    ai_acceleration_parse("./AAAI-2025-Papers-Oral", "./")
