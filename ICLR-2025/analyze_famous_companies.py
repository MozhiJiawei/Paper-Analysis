#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析ICLR 2025数据中的知名公司别称
"""

import json
import re
from collections import defaultdict

# 知名AI公司映射表 - 从orgnization.py复制
famous_ai_companies_mapping = {
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

def clean_organization_name(org_name):
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

def get_famous_ai_company_canonical_name(org_name):
    """获取知名AI公司的标准名称，如果不是知名公司则返回None"""
    org_lower = org_name.lower().strip()
    
    # 遍历所有知名公司的映射，优先精确匹配
    best_match = None
    best_match_length = 0
    
    for canonical_name, variants in famous_ai_companies_mapping.items():
        for variant in variants:
            variant_lower = variant.lower().strip()
            
            # 精确匹配
            if variant_lower == org_lower:
                return canonical_name
            
            # 包含匹配，但要确保是完整的词汇匹配
            if variant_lower in org_lower:
                # 检查是否为完整词汇（前后是空格、标点或字符串边界）
                pattern = r'\b' + re.escape(variant_lower) + r'\b'
                if re.search(pattern, org_lower):
                    # 选择最长的匹配
                    if len(variant_lower) > best_match_length:
                        best_match = canonical_name
                        best_match_length = len(variant_lower)
    
    return best_match

def analyze_famous_companies():
    """分析知名公司的别称"""
    print("开始分析ICLR 2025数据中的知名公司别称...")
    
    # 读取数据
    with open('iclr2025.json', 'r', encoding='utf-8') as f:
        papers = json.load(f)
    
    print(f"总共读取到 {len(papers)} 篇论文")
    
    # 收集所有组织名称
    all_organizations = set()
    for paper in papers:
        if paper.get('aff'):
            aff_list = paper['aff'].split(';')
            for aff in aff_list:
                if aff.strip():
                    org_name = clean_organization_name(aff.strip())
                    if org_name:
                        all_organizations.add(org_name)
    
    print(f"总共找到 {len(all_organizations)} 个不同的组织")
    
    # 分析知名公司的别称
    company_variants = defaultdict(set)
    new_variants = defaultdict(set)
    
    for org in all_organizations:
        canonical_name = get_famous_ai_company_canonical_name(org)
        if canonical_name:
            company_variants[canonical_name].add(org)
        else:
            # 检查是否包含知名公司关键词
            for company, variants in famous_ai_companies_mapping.items():
                for variant in variants:
                    if variant.lower() in org.lower():
                        # 检查是否为完整词汇匹配
                        pattern = r'\b' + re.escape(variant.lower()) + r'\b'
                        if re.search(pattern, org.lower()):
                            new_variants[company].add(org)
                            break
    
    # 输出结果
    print("\n=== 现有知名公司别称统计 ===")
    for company, variants in sorted(company_variants.items()):
        print(f"\n{company}:")
        for variant in sorted(variants):
            print(f"  - {variant}")
    
    print("\n=== 建议添加的新别称 ===")
    for company, variants in sorted(new_variants.items()):
        if variants:
            print(f"\n{company}:")
            for variant in sorted(variants):
                print(f"  - {variant}")
    
    # 生成更新后的映射表
    print("\n=== 更新后的映射表 ===")
    print("famous_ai_companies_mapping = {")
    for company, variants in sorted(famous_ai_companies_mapping.items()):
        # 合并现有和新发现的别称
        all_variants = set(variants)
        if company in new_variants:
            all_variants.update(new_variants[company])
        
        print(f"    '{company}': {list(sorted(all_variants))},")
    print("}")

if __name__ == "__main__":
    analyze_famous_companies() 