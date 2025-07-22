from grobid_client.grobid_client import GrobidClient
from bs4 import BeautifulSoup
from typing import Dict, Optional
import os
import json
import logging

# 获取模块级别的logger
logger = logging.getLogger(__name__)

def _get_config_path() -> str:
    """获取config.json文件的绝对路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "config.json")

def abstract_parser(pdf_path: str):
    client = GrobidClient(config_path=_get_config_path())
    client.process("processHeaderDocument", pdf_path, n=20)

def full_parser(pdf_path: str):
    client = GrobidClient(config_path=_get_config_path())
    client.process("processFulltextDocument", pdf_path, n=20)

def extract_paper_abstract_from_paper_copilot(paper_info: Dict) -> Dict:
    """
    从paper_copilot的解析结果中提取需要的信息

    Args:
        paper_info: 论文的内容，格式与paper_copilot保持一致

    Returns:
        包含论文信息的字典，包括标题、作者、摘要等
    """
    # 初始化返回结果
    result = {
        'title': '',
        'authors': [],
        'abstract': '',
        'affiliations': []
    }
    
    # 提取标题
    result['title'] = paper_info.get('title', '').strip()
    
    # 提取摘要
    result['abstract'] = paper_info.get('abstract', '').strip()
    
    # 提取作者信息
    authors_str = paper_info.get('author', '')
    affiliations_str = paper_info.get('aff', '')
    emails_str = paper_info.get('email', '')
    homepages_str = paper_info.get('homepage', '')
    
    if authors_str:
        # 按分号分割作者、单位、邮箱等信息
        authors_list = [author.strip() for author in authors_str.split(';') if author.strip()]
        affiliations_list = [aff.strip() for aff in affiliations_str.split(';')] if affiliations_str else []
        emails_list = [email.strip() for email in emails_str.split(';')] if emails_str else []
        homepages_list = [homepage.strip() for homepage in homepages_str.split(';')] if homepages_str else []
        
        # 构建作者信息列表
        for i, author_name in enumerate(authors_list):
            author_info = {'name': author_name}
            
            # 添加邮箱信息（如果有）
            if i < len(emails_list) and emails_list[i]:
                author_info['email'] = emails_list[i]
            
            # 添加主页信息（如果有）
            if i < len(homepages_list) and homepages_list[i]:
                author_info['homepage'] = homepages_list[i]
            
            # 添加单位信息（如果有）
            if i < len(affiliations_list) and affiliations_list[i]:
                author_info['affiliation'] = [affiliations_list[i]]
                
            result['authors'].append(author_info)
    
    # 提取所有唯一的单位信息
    if affiliations_str:
        unique_affiliations = list(set([aff.strip() for aff in affiliations_str.split(';') if aff.strip()]))
        result['affiliations'] = unique_affiliations
    
    return result

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
    logger.info(f"XML文件路径: {xml_file_path}")
    
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
    logger.info(f"PDF提取结果: {result}")

    with open("iclr2025_part.json", 'r', encoding='utf-8') as file:
        paper_infos = json.load(file)

    for paper_info in paper_infos:
        result = extract_paper_abstract_from_paper_copilot(paper_info)
        logger.info(f"Paper Copilot提取结果: {result}")
    