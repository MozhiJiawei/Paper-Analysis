import os
import shutil
import pandas as pd
from pathlib import Path

import sys
import os
sys.path.append('..')

from utils.pdf_extractor import abstract_parser
from utils.ai_acceleration_extractor import ai_acceleration_parse

def organize_papers_by_company():
    """
    根据CSV文件中的公司信息，将AAAI-2025-Papers中的论文复制到AAAI-2025-Papers-famous目录，
    并在文件名前添加标准化公司名称前缀。同时复制PDF和XML文件。
    """
    
    # 定义路径
    csv_file = "famous_ai_companies_papers_20250629_202208.csv"
    xml_source_dir = "AAAI-2025-Papers"
    pdf_source_dir = "AAAI-2025-Papers"
    target_dir = "AAAI-2025-Papers-famous"
    
    # 读取CSV文件
    try:
        df = pd.read_csv(csv_file, encoding='utf-8')
        print(f"成功读取CSV文件，共{len(df)}条记录")
    except Exception as e:
        print(f"读取CSV文件失败: {e}")
        return
    
    # 确保目标目录存在
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    # 处理每一行数据
    successful_copies = 0
    failed_copies = 0
    
    for index, row in df.iterrows():
        try:
            # 获取信息
            company_name = row['标准化公司名称']
            xml_filename = row['文件名']
            
            # 构建XML源文件路径
            xml_source_file = Path(xml_source_dir) / xml_filename
            
            # 检查XML源文件是否存在
            if not xml_source_file.exists():
                print(f"警告: XML源文件不存在 - {xml_source_file}")
                failed_copies += 1
                continue
            
            # 构建PDF文件名（将.grobid.tei.xml替换为.pdf）
            pdf_filename = xml_filename.replace('.grobid.tei.xml', '.pdf')
            pdf_source_file = Path(pdf_source_dir) / pdf_filename
            
            # 构建目标文件名（添加公司名称前缀）
            xml_file_stem = xml_source_file.stem
            xml_file_suffix = xml_source_file.suffix
            new_xml_filename = f"{company_name}_{xml_file_stem}{xml_file_suffix}"
            
            pdf_file_stem = Path(pdf_filename).stem
            pdf_file_suffix = Path(pdf_filename).suffix
            new_pdf_filename = f"{company_name}_{pdf_file_stem}{pdf_file_suffix}"
            
            # 构建目标文件路径
            xml_target_file = target_path / new_xml_filename
            pdf_target_file = target_path / new_pdf_filename
            
            # 复制XML文件
            shutil.copy2(xml_source_file, xml_target_file)
            print(f"成功复制XML: {xml_filename} -> {new_xml_filename}")
            
            # 复制PDF文件（如果存在）
            if pdf_source_file.exists():
                shutil.copy2(pdf_source_file, pdf_target_file)
                print(f"成功复制PDF: {pdf_filename} -> {new_pdf_filename}")
                successful_copies += 1
            else:
                print(f"警告: PDF文件不存在 - {pdf_source_file}")
                # 即使PDF不存在，我们仍然算作成功（因为XML复制成功了）
                successful_copies += 1
            
        except Exception as e:
            print(f"处理文件 {xml_filename} 时出错: {e}")
            failed_copies += 1
            continue
    
    # 输出统计结果
    print(f"\n处理完成:")
    print(f"成功复制: {successful_copies} 个文件")
    print(f"失败: {failed_copies} 个文件")
    print(f"总计: {len(df)} 个文件")

if __name__ == "__main__":
    # organize_papers_by_company()

    # abstract_parser("AAAI-2025-Papers-famous")

    ai_acceleration_parse("./AAAI-2025-Papers-famous + Oral", "./")
