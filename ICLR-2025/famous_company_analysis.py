#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICLR 2025 知名公司论文筛选工具
从iclr2025.json中筛选出知名AI公司相关的poster论文
"""

import json
import re
from collections import defaultdict
from datetime import datetime
import sys
import os

from utils.famous_company_analysis import FamousCompanyAnalyzer


class ICLR2025FamousCompanyAnalyzer:
    def __init__(self):
        # 使用utils中的FamousCompanyAnalyzer
        self.analyzer = FamousCompanyAnalyzer()
        
    def extract_organizations_from_paper(self, paper):
        """从论文数据中提取组织信息"""
        organizations = set()
        
        # 从aff字段提取组织信息
        if 'aff' in paper and paper['aff']:
            aff_list = paper['aff'].split(';')
            for aff in aff_list:
                if aff.strip():
                    org_name = self.analyzer.clean_organization_name(aff.strip())
                    if org_name:
                        organizations.add(org_name)
        
        # 从aff_unique_norm字段提取组织信息
        if 'aff_unique_norm' in paper and paper['aff_unique_norm']:
            aff_norm_list = paper['aff_unique_norm'].split(';')
            for aff_norm in aff_norm_list:
                if aff_norm.strip():
                    org_name = self.analyzer.clean_organization_name(aff_norm.strip())
                    if org_name:
                        organizations.add(org_name)
        
        return list(organizations)
    
    def filter_famous_company_posters(self, input_file, output_file):
        """筛选知名公司的poster论文"""
        print(f"开始读取文件: {input_file}")
        
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                papers = json.load(f)
        except Exception as e:
            print(f"读取文件时出错: {e}")
            return
        
        print(f"总共读取到 {len(papers)} 篇论文")
        
        # 筛选poster论文
        poster_papers = [paper for paper in papers if paper.get('status', '').lower() == 'poster']
        print(f"找到 {len(poster_papers)} 篇poster论文")
        
        # 筛选知名公司相关的poster论文
        famous_company_posters = []
        company_paper_count = defaultdict(int)
        
        for paper in poster_papers:
            organizations = self.extract_organizations_from_paper(paper)
            paper_famous_companies = set()  # 用于去重，确保每篇论文对每个公司只计算一次
            
            for org in organizations:
                if self.analyzer.is_famous_ai_company(org):
                    canonical_name = self.analyzer.get_famous_ai_company_canonical_name(org)
                    if canonical_name:
                        paper_famous_companies.add(canonical_name)
            
            # 如果论文涉及知名公司，则添加到结果中
            if paper_famous_companies:
                famous_company_posters.append(paper)
                # 为每个涉及的公司增加计数（去重后）
                for company in paper_famous_companies:
                    company_paper_count[company] += 1
        
        print(f"找到 {len(famous_company_posters)} 篇知名公司相关的poster论文")
        
        # 按公司统计
        print("\n知名公司poster论文统计:")
        for company, count in sorted(company_paper_count.items(), key=lambda x: x[1], reverse=True):
            print(f"  {company}: {count} 篇")
        
        # 保存结果
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(famous_company_posters, f, ensure_ascii=False, indent=2)
            print(f"\n已保存到: {output_file}")
        except Exception as e:
            print(f"保存文件时出错: {e}")
            return
        
        # 生成详细报告
        self.generate_detailed_report(famous_company_posters, company_paper_count)
        
        # 额外统计：每篇论文涉及的公司数量分布
        paper_company_count_distribution = defaultdict(int)
        for paper in famous_company_posters:
            organizations = self.extract_organizations_from_paper(paper)
            paper_famous_companies = set()
            for org in organizations:
                canonical_name = self.analyzer.get_famous_ai_company_canonical_name(org)
                if canonical_name:
                    paper_famous_companies.add(canonical_name)
            paper_company_count_distribution[len(paper_famous_companies)] += 1
        
        print("\n论文涉及公司数量分布:")
        for company_count, paper_count in sorted(paper_company_count_distribution.items()):
            print(f"  涉及{company_count}个公司的论文: {paper_count} 篇")
    
    def generate_detailed_report(self, papers, company_counts):
        """生成详细的分析报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"famous_company_poster_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ICLR 2025 知名公司Poster论文分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("一、基本统计信息\n")
            f.write("-" * 30 + "\n")
            f.write(f"知名公司poster论文总数: {len(papers)}\n")
            f.write(f"涉及知名公司数量: {len(company_counts)}\n\n")
            
            f.write("二、知名公司论文数量排行榜\n")
            f.write("-" * 30 + "\n")
            for i, (company, count) in enumerate(sorted(company_counts.items(), key=lambda x: x[1], reverse=True), 1):
                f.write(f"{i:2d}. {company}: {count} 篇\n")
            f.write("\n")
            
            f.write("三、详细论文清单\n")
            f.write("-" * 30 + "\n")
            for i, paper in enumerate(papers, 1):
                f.write(f"\n{i}. {paper.get('title', '未知标题')}\n")
                f.write(f"   状态: {paper.get('status', '未知')}\n")
                f.write(f"   作者: {paper.get('author', '未知')}\n")
                f.write(f"   组织: {paper.get('aff', '未知')}\n")
                f.write(f"   标准化组织: {paper.get('aff_unique_norm', '未知')}\n")
                f.write(f"   ID: {paper.get('id', '未知')}\n")
                
                # 显示匹配的知名公司（去重后）
                organizations = self.extract_organizations_from_paper(paper)
                paper_famous_companies = set()
                famous_companies_detail = []
                
                for org in organizations:
                    canonical_name = self.analyzer.get_famous_ai_company_canonical_name(org)
                    if canonical_name:
                        paper_famous_companies.add(canonical_name)
                        famous_companies_detail.append(f"{canonical_name} (原始: {org})")
                
                if paper_famous_companies:
                    f.write(f"   匹配的知名公司: {'; '.join(sorted(paper_famous_companies))}\n")
                    f.write(f"   详细匹配: {'; '.join(famous_companies_detail)}\n")
                f.write("-" * 50)
        
        print(f"已生成详细报告: {report_file}")


def main():
    """主函数"""
    print("开始筛选ICLR 2025知名公司poster论文...")
    
    # 创建分析器实例
    analyzer = ICLR2025FamousCompanyAnalyzer()
    
    # 输入和输出文件
    input_file = "iclr2025.json"
    output_file = "iclr2025_famous_poster.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 '{input_file}'")
        return
    
    # 筛选知名公司poster论文
    analyzer.filter_famous_company_posters(input_file, output_file)
    
    print("\n筛选完成！")


if __name__ == "__main__":
    main()
