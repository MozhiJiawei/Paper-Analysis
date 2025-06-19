from grobid_client.grobid_client import GrobidClient
from bs4 import BeautifulSoup
from typing import Dict, Optional
import os

def _get_config_path() -> str:
    """获取config.json文件的绝对路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "config.json")

def abstract_parser(pdf_path: str):
    client = GrobidClient(config_path=_get_config_path())
    client.process("processHeaderDocument", pdf_path, n=20)

def extract_paper_abstract(pdf_file_path: str) -> Dict:
    """
    从PDF文件中提取论文信息
    
    Args:
        pdf_file_path: PDF文件的路径
        
    Returns:
        包含论文信息的字典，包括标题、作者、摘要等
    """
    # 生成对应的XML文件路径
    xml_file_path = os.path.splitext(pdf_file_path)[0] + '.grobid.tei.xml'
    
    # 检查XML文件是否存在
    if os.path.exists(xml_file_path):
        # 如果XML文件存在，直接读取
        with open(xml_file_path, 'r', encoding='utf-8') as f:
            result = f.read()
    else:
        # 如果XML文件不存在，使用Grobid处理PDF
        client = GrobidClient(config_path=_get_config_path())
        result = client.process_pdf("processHeaderDocument",
                                    pdf_file_path,
                                    None, None, None, None, None, None, None)[2]
    
    # 使用BeautifulSoup解析XML
    soup = BeautifulSoup(result, 'xml')
    
    # 提取信息
    paper_info = {
        'title': '',
        'authors': [],
        'abstract': '',
        'affiliations': []
    }
    
    # 提取标题
    title = soup.find('title', {'type': 'main'})
    if title:
        paper_info['title'] = title.text.strip()
    
    # 提取作者信息
    authors = soup.find_all('author')
    for author in authors:
        author_info = {}
        pers_name = author.find('persName')
        if pers_name:
            forename = pers_name.find('forename')
            surname = pers_name.find('surname')
            author_info['name'] = f"{forename.text if forename else ''} {surname.text if surname else ''}".strip()
        
        # 提取作者邮箱
        email = author.find('email')
        if email:
            author_info['email'] = email.text.strip()
            
        # 提取作者单位
        affiliation = author.find('affiliation')
        if affiliation:
            org_names = affiliation.find_all('orgName')
            author_info['affiliation'] = [org.text.strip() for org in org_names]
            
        if author_info:
            paper_info['authors'].append(author_info)
    
    # 提取摘要
    abstract = soup.find('abstract')
    if abstract:
        paper_info['abstract'] = abstract.text.strip()
    
    return paper_info

if __name__ == "__main__":
    # 测试代码
    pdf_path = ".\\001_001_Real-Time_Calibration_Model_for_Low-Cost_Sensor_in_Fine-Grained_Time_Series.pdf"
    result = extract_paper_abstract(pdf_path)
    print(result)

    