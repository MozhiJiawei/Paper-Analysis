#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AAAI 2025 论文组织信息分析工具
使用BeautifulSoup解析XML文件，提取作者组织信息并生成统计报告
"""

import os
import csv
import re
from collections import defaultdict, Counter
from bs4 import BeautifulSoup
import glob
from datetime import datetime


class OrganizationAnalyzer:
    def __init__(self, xml_dir="AAAI-2025-Papers"):
        self.xml_dir = xml_dir
        self.papers_organizations = []  # 论文-组织映射详细清单
        self.organization_counts = Counter()  # 组织发表论文数量统计
        # 知名AI公司映射表 - 用于合并同类项
        self.famous_ai_companies_mapping = {
            'Google': ['Google', 'Alphabet', 'DeepMind', 'Google DeepMind', 'Google Deepmind', 'Google Research'],
            'Microsoft': ['Microsoft', 'Microsoft Research', 'Microsoft Corporation', 'Microsoft (China)', 'Microsoft Redmond', 'Microsoft Research Asia', 'Microsoft Research Cambridge', 'Microsoft AI for Good Research Lab', 'Microsoft IDC', 'MIT-IBM Watson AI Lab'],
            'Meta': ['Meta', 'Facebook', 'Meta Platforms'],
            'Apple': ['Apple'],
            'Amazon': ['Amazon', 'AWS', 'Amazon Web Services', 'Amazon Fashion', 'Amazon Research', 'Amazon Quantum Solutions Lab', 'Amazon Inc', 'Amazon AI Labs', 'Amazon AWS AI Labs', 'Amazon Research Tübingen'],
            'NVIDIA': ['NVIDIA', 'NVIDIA Research', 'NVIDIA AI Technology Center'],
            'Alibaba': ['Alibaba', 'Alibaba Group', 'Ant Group', 'Taobao', 'Tmall', 'Alibaba Cloud Computing', 'Alibaba DAMO', 'Alibaba International Digital Commerce'],
            'Tencent': ['Tencent', 'WeChat', 'Tencent AI Lab', 'Tencent YouTu Lab', 'Tencent Youtu Lab', 'Tencent Hunyuan', 'Tencent PCG', 'Tencent WeChat', 'Tencent Shenzhen', 'Tencent Corporate', 'Tencent Inc', 'Tencent IEG', 'Tencent TEG', 'Tencent Weixin Group', 'WeChat Pay Lab'],
            'ByteDance': ['ByteDance', 'TikTok', 'ByteDance Inc', 'Bytedance Inc', 'Bytedance(Seed)', 'Bytedance Seed'],
            'Baidu': ['Baidu', 'Baidu Inc', 'Baidu Research'],
            'Huawei': ['Huawei', 'Huawei Technologies', 'Huawei Noah\'s Ark Lab', 'Huawei Cloud', 'Huawei Inc', 'Huawei Kirin Solution', 'Huawei International', 'Huawei Technology', 'Huawei Shanghai', 'Huawei Hong Kong Research Center', 'Huawei Hisilicon', 'Huawei Taylor Lab', 'Noah\'s Ark Lab, Huawei Technologies', 'Huawei Technologies Canada', 'Huawei Technologies Riemann Lab'],
            'IBM': ['IBM', 'Red Hat', 'IBM Research', 'IBM T. J. Watson Research Center', 'IBM Software', 'IBM Research Europe', 'IBM Research -Zurich', 'IBM Research New York', 'IBM Research -Israel', 'IBM Research -Europe'],
            'Intel': ['Intel Labs', 'Intel Corporation', 'Intel Labs China'],
            'Adobe': ['Adobe', 'Adobe Research', 'Adobe Inc'],
            'Samsung': ['Samsung', 'Samsung Research', 'Samsung Electronics', 'Samsung R&D Institute China-Beijing', 'Samsung AI Center', 'Samsung Electronic Research Centre of China', 'Samsung AI Center Toronto', 'Samsung Electorics'],
            'Sony': ['Sony', 'Sony Group Corporation', 'Sony Research', 'Sony AI'],
            'OpenAI': ['OpenAI'],
            'Salesforce': ['Salesforce'],
            'Tesla': ['Tesla'],
            'Uber': ['Uber'],
            'Airbnb': ['Airbnb'],
            'Netflix': ['Netflix'],
            'Spotify': ['Spotify', 'Spotify Barcelona'],
            'Qualcomm': ['Qualcomm'],
            'ARM': ['ARM'],
            'Anthropic': ['Anthropic'],
            'Cohere': ['Cohere'],
            'Stability AI': ['Stability AI'],
            'Midjourney': ['Midjourney'],
            'Runway': ['Runway'],
            'Scale AI': ['Scale AI'],
            'Palantir': ['Palantir']
        }
        
    def clean_organization_name(self, org_name):
        """清理组织名称，去除多余的空格和特殊字符"""
        if not org_name:
            return ""
        
        # 去除多余空格和换行符
        org_name = re.sub(r'\s+', ' ', org_name.strip())
        
        # 去除常见的后缀
        suffixes = [', Inc.', ', LLC', ', Ltd.', ', Corp.', ', Corporation', 
                   ', Co.', ', Company', ', Group', ', Team']
        for suffix in suffixes:
            if org_name.endswith(suffix):
                org_name = org_name[:-len(suffix)]
                break
                
        return org_name.strip()
    
    def extract_organizations_from_xml(self, xml_file):
        """从XML文件中提取组织信息"""
        try:
            with open(xml_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'xml')
            
            # 提取论文标题
            title_elem = soup.find('title', {'level': 'a', 'type': 'main'})
            paper_title = title_elem.get_text().strip() if title_elem else "未知标题"
            
            # 提取所有作者的组织信息
            authors = soup.find_all('author')
            organizations = set()
            
            for author in authors:
                # 提取作者姓名
                person_name = author.find('persName')
                if person_name:
                    forename = person_name.find('forename')
                    surname = person_name.find('surname')
                    author_name = ""
                    if forename:
                        author_name += forename.get_text().strip() + " "
                    if surname:
                        author_name += surname.get_text().strip()
                    author_name = author_name.strip()
                else:
                    author_name = "未知作者"
                
                # 提取组织信息
                affiliation = author.find('affiliation')
                if affiliation:
                    # 查找所有组织名称标签
                    org_names = affiliation.find_all('orgName')
                    if org_names:
                        for org_name in org_names:
                            org_text = self.clean_organization_name(org_name.get_text())
                            if org_text:
                                organizations.add(org_text)
                    else:
                        # 如果没有orgName标签，尝试从address或其他地方提取
                        address = affiliation.find('address')
                        if address:
                            settlement = address.find('settlement')
                            if settlement:
                                org_text = self.clean_organization_name(settlement.get_text())
                                if org_text:
                                    organizations.add(org_text)
                        else:
                            # 直接从affiliation文本中提取
                            org_text = self.clean_organization_name(affiliation.get_text())
                            if org_text:
                                organizations.add(org_text)
            
            return paper_title, list(organizations)
            
        except Exception as e:
            print(f"解析文件 {xml_file} 时出错: {e}")
            return None, []
    
    def get_famous_ai_company_canonical_name(self, org_name):
        """获取知名AI公司的标准名称，如果不是知名公司则返回None"""
        org_lower = org_name.lower().strip()
        
        # 遍历所有知名公司的映射，优先精确匹配
        best_match = None
        best_match_length = 0
        
        for canonical_name, variants in self.famous_ai_companies_mapping.items():
            for variant in variants:
                variant_lower = variant.lower().strip()
                
                # 精确匹配
                if variant_lower == org_lower:
                    return canonical_name
                
                # 包含匹配，但要确保是完整的词汇匹配
                if variant_lower in org_lower:
                    # 检查是否为完整词汇（前后是空格、标点或字符串边界）
                    import re
                    pattern = r'\b' + re.escape(variant_lower) + r'\b'
                    if re.search(pattern, org_lower):
                        # 选择最长的匹配
                        if len(variant_lower) > best_match_length:
                            best_match = canonical_name
                            best_match_length = len(variant_lower)
        
        return best_match
    
    def is_famous_ai_company(self, org_name):
        """判断是否为知名AI公司"""
        return self.get_famous_ai_company_canonical_name(org_name) is not None
    
    def analyze_all_papers(self):
        """分析所有论文的组织信息"""
        xml_files = glob.glob(os.path.join(self.xml_dir, "*.xml"))
        
        print(f"找到 {len(xml_files)} 个XML文件")
        
        # 用于去重统计的数据结构：organization -> set of paper titles
        organization_papers = defaultdict(set)
        
        for xml_file in xml_files:
            paper_title, organizations = self.extract_organizations_from_xml(xml_file)
            
            if paper_title and organizations:
                # 为每个组织创建一条记录
                for org in organizations:
                    self.papers_organizations.append({
                        'paper_title': paper_title,
                        'organization': org,
                        'file_name': os.path.basename(xml_file)
                    })
                    
                    # 使用set去重，确保同一篇论文对每个组织只计算一次
                    organization_papers[org].add(paper_title)
        
        # 根据去重后的结果统计每个组织的论文数量
        for org, papers in organization_papers.items():
            self.organization_counts[org] = len(papers)
        
        print(f"成功解析 {len(self.papers_organizations)} 条论文-组织映射记录")
        print(f"发现 {len(self.organization_counts)} 个不同的组织")
        print(f"总计 {len(set(record['paper_title'] for record in self.papers_organizations))} 篇独立论文")
    
    def generate_csv_reports(self):
        """生成CSV报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 论文-组织映射详细清单
        mapping_file = f"paper_organization_mapping_{timestamp}.csv"
        with open(mapping_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['论文标题', '组织名称', '文件名'])
            
            for record in self.papers_organizations:
                writer.writerow([
                    record['paper_title'],
                    record['organization'],
                    record['file_name']
                ])
        
        print(f"已生成论文-组织映射清单: {mapping_file}")
        
        # 2. 组织发表论文数量统计
        stats_file = f"organization_paper_counts_{timestamp}.csv"
        with open(stats_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['组织名称', '论文数量'])
            
            # 按论文数量降序排列
            for org, count in self.organization_counts.most_common():
                writer.writerow([org, count])
        
        print(f"已生成组织论文数量统计: {stats_file}")
        
        # 3. 知名AI公司论文清单（合并同类项）
        famous_companies_file = f"famous_ai_companies_papers_{timestamp}.csv"
        famous_company_papers = []
        
        for record in self.papers_organizations:
            canonical_name = self.get_famous_ai_company_canonical_name(record['organization'])
            if canonical_name:
                famous_company_papers.append({
                    'paper_title': record['paper_title'],
                    'original_organization': record['organization'],
                    'canonical_company': canonical_name,
                    'file_name': record['file_name']
                })
        
        with open(famous_companies_file, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['论文标题', '标准化公司名称', '原始组织名称', '文件名'])
            
            for record in famous_company_papers:
                writer.writerow([
                    record['paper_title'],
                    record['canonical_company'],
                    record['original_organization'],
                    record['file_name']
                ])
        
        print(f"已生成知名AI公司论文清单: {famous_companies_file}")
        
        return mapping_file, stats_file, famous_companies_file
    
    def generate_summary_txt(self, mapping_file, stats_file, famous_companies_file):
        """生成TXT总结报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"organization_analysis_summary_{timestamp}.txt"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("AAAI 2025 论文组织信息分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 基本统计信息
            f.write("一、基本统计信息\n")
            f.write("-" * 30 + "\n")
            f.write(f"总论文数量: {len(set(record['paper_title'] for record in self.papers_organizations))}\n")
            f.write(f"总组织数量: {len(self.organization_counts)}\n")
            f.write(f"论文-组织映射记录数: {len(self.papers_organizations)}\n\n")
            
            # 组织论文数量排行榜（前20）
            f.write("二、组织论文数量排行榜（前20名）\n")
            f.write("-" * 30 + "\n")
            for i, (org, count) in enumerate(self.organization_counts.most_common(20), 1):
                f.write(f"{i:2d}. {org}: {count} 篇\n")
            f.write("\n")
            
            # 知名AI公司统计（合并同类项）
            famous_company_counts = Counter()  # 标准化后的公司统计
            famous_company_details = defaultdict(list)  # 详细的原始组织名称
            famous_company_papers = defaultdict(set)  # 每个公司的论文集合，避免重复计算
            
            for record in self.papers_organizations:
                canonical_name = self.get_famous_ai_company_canonical_name(record['organization'])
                if canonical_name:
                    # 使用论文标题作为唯一标识，避免同一篇论文被重复计算
                    famous_company_papers[canonical_name].add(record['paper_title'])
                    famous_company_details[canonical_name].append(record['organization'])
            
            # 统计每个公司的唯一论文数量
            for company, papers in famous_company_papers.items():
                famous_company_counts[company] = len(papers)
            
            f.write("三、知名AI公司论文发表情况（合并同类项后）\n")
            f.write("-" * 30 + "\n")
            if famous_company_counts:
                f.write(f"知名AI公司总数: {len(famous_company_counts)}\n")
                f.write(f"知名AI公司论文总数: {sum(famous_company_counts.values())}\n\n")
                
                f.write("知名AI公司排行榜（合并同类项后）:\n")
                for i, (company, count) in enumerate(famous_company_counts.most_common(), 1):
                    f.write(f"{i:2d}. {company}: {count} 篇\n")
                
                f.write("\n四、知名AI公司详细组织名称统计\n")
                f.write("-" * 30 + "\n")
                for company in sorted(famous_company_counts.keys()):
                    count = famous_company_counts[company]
                    f.write(f"\n{company} (总计: {count} 篇):\n")
                    
                    # 统计每个原始组织名称的出现次数
                    org_counts = Counter(famous_company_details[company])
                    for org, org_count in org_counts.most_common():
                        f.write(f"  - {org}: {org_count} 篇\n")
            else:
                f.write("未发现知名AI公司发表的论文\n")
            f.write("\n")
            
            # 文件信息
            f.write("五、生成的文件\n")
            f.write("-" * 30 + "\n")
            f.write(f"1. 论文-组织映射清单: {mapping_file}\n")
            f.write(f"2. 组织论文数量统计: {stats_file}\n")
            f.write(f"3. 知名AI公司论文清单（合并同类项）: {famous_companies_file}\n")
            f.write(f"4. 分析总结报告: {summary_file}\n\n")
            
            # 分析说明
            f.write("六、分析说明\n")
            f.write("-" * 30 + "\n")
            f.write("1. 本分析基于AAAI 2025论文的XML文件\n")
            f.write("2. 组织信息从论文作者的机构信息中提取\n")
            f.write("3. 已解决重复统计问题：同一篇论文对每个组织只计算一次，即使该论文有多个作者来自同一组织\n")
            f.write("4. 知名AI公司已实现智能合并同类项，如Huawei、Google、Microsoft等的不同表达方式\n")
            f.write("5. 合并同类项后的统计结果更加准确，避免了重复计算\n")
            f.write("6. CSV文件中包含原始组织名称和标准化公司名称的对应关系\n")
            f.write("7. 建议进一步手工审核和清理组织名称以获得更准确的统计结果\n")
        
        print(f"已生成分析总结报告: {summary_file}")
        return summary_file


def main():
    """主函数"""
    print("开始分析AAAI 2025论文组织信息...")
    
    # 创建分析器实例
    analyzer = OrganizationAnalyzer()
    
    # 检查XML文件目录是否存在
    if not os.path.exists(analyzer.xml_dir):
        print(f"错误: 找不到XML文件目录 '{analyzer.xml_dir}'")
        return
    
    # 分析所有论文
    analyzer.analyze_all_papers()
    
    if not analyzer.papers_organizations:
        print("未找到任何论文-组织信息，请检查XML文件格式")
        return
    
    # 生成CSV报告
    mapping_file, stats_file, famous_companies_file = analyzer.generate_csv_reports()
    
    # 生成TXT总结
    summary_file = analyzer.generate_summary_txt(mapping_file, stats_file, famous_companies_file)
    
    print("\n分析完成！生成的文件:")
    print(f"- {mapping_file}")
    print(f"- {stats_file}")
    print(f"- {famous_companies_file}")
    print(f"- {summary_file}")


if __name__ == "__main__":
    main()
