from typing import List, Tuple, Optional
import re
from collections import defaultdict


class FamousCompanyAnalyzer:
    """知名公司分析器"""
    
    def __init__(self):
        # 知名AI公司映射表 - 整合AAAI和ICLR的数据
        self.famous_ai_companies_mapping = {
            'Google': ['Google', 'Alphabet', 'DeepMind', 'Google DeepMind', 'Google Deepmind', 'Google Research', 'Google Brain', 'Google Inc.', 'Google Waymo', 'Google/DeepMind', 'Google/Waymo', 'Gooogle DeepMind', 'Research, Google', 'Robotics at Google', 'SystemsResearch@Google', 'Verily (Google Life Sciences)', 'DeepMind Montreal', 'Deepmind', 'Google Brain Robotics & Columbia University', 'Current: Google'],
            'Microsoft': ['Microsoft', 'Microsoft Research', 'Microsoft Corporation', 'Microsoft (China)', 'Microsoft Redmond', 'Microsoft Research Asia', 'Microsoft Research Cambridge', 'Microsoft AI for Good Research Lab', 'Microsoft IDC', 'MIT-IBM Watson AI Lab', 'Microsoft (ABK)', 'Microsoft Azure AI', 'Microsoft GenAI', 'Microsoft Inc', 'Microsoft Research Aisa', 'Microsoft Research, Redmond', 'RedHat AI & MIT-IBM Watson AI Lab'],
            'Meta': ['Meta', 'Facebook', 'Meta Platforms', 'Central Applied Science, Meta', 'FAIR (Meta AI)', 'FAIR (Meta Platforms Inc.)', 'FAIR Labs, Meta AI', 'FAIR at Meta', 'FAIR, Meta AI', 'Facebook A.I. Research', 'Facebook AI', 'Facebook AI Research', 'Facebook AI Research (FAIR) Meta', 'GenAI, Meta', 'META', 'Meta (FAIR)', 'Meta (Facebook) AI', 'Meta AI', 'Meta AI (FAIR)', 'Meta AI Research', 'Meta Ai', 'Meta FAIR', 'Meta Facebook', 'Meta Fundamental AI Research (FAIR)', 'Meta GenAI', 'Meta Inc', 'Meta Inc.', 'Meta Platform Inc.', 'Meta Platforms Inc.', 'Meta Platforms, Inc', 'Meta Reality Labs', 'Meta Research', 'Meta, FAIR', 'Meta, Reality Labs Research', 'Research, Facebook'],
            'Apple': ['Apple', 'Apple AI/ML', 'Apple Inc.', 'Apple/AIML'],
            'Amazon': ['Amazon', 'AWS', 'Amazon Web Services', 'Amazon Fashion', 'Amazon Research', 'Amazon Quantum Solutions Lab', 'Amazon Inc', 'Amazon AI Labs', 'Amazon AWS AI Labs', 'Amazon Research Tübingen', 'AWS AI', 'AWS AI Labs', 'AWS AI Labs (Amazon)', 'Amazon AGI', 'Amazon AGI Autonomy SF Lab', 'Amazon AI Research Lab', 'Amazon AWS', 'Amazon AWS AI', 'Amazon Alexa', 'Amazon Alexa AI', 'Amazon Alexa ASR', 'Amazon Development Center Germany', 'Amazon Prime Video', 'Amazon Robotics', 'Amazon, AWS AI Labs'],
            'NVIDIA': ['NVIDIA', 'NVIDIA Research', 'NVIDIA AI Technology Center', 'NVIDIA AI', 'NVIDIA Corporation'],
            'Alibaba': ['Alibaba', 'Alibaba Group', 'Ant Group', 'Taobao', 'Tmall', 'Alibaba Cloud Computing', 'Alibaba DAMO', 'Alibaba International Digital Commerce', 'Alibaba Cloud', 'Alibaba DAMO Academy', 'Alibaba Group (U.S.)', 'Alibaba Group US', 'Alibaba Group, DAMO Academy', 'Alibaba International Digital Commerce Group', 'Ant Group (AliPay)', 'DAMO Academy, Alibaba Group', 'TAOBAO & TMALL GROUP', 'Tongyi Lab, Alibaba Group', 'ant group'],
            'Tencent': ['Tencent', 'WeChat', 'Tencent AI Lab', 'Tencent YouTu Lab', 'Tencent Youtu Lab', 'Tencent Hunyuan', 'Tencent PCG', 'Tencent WeChat', 'Tencent Shenzhen', 'Tencent Corporate', 'Tencent Inc', 'Tencent IEG', 'Tencent TEG', 'Tencent Weixin Group', 'WeChat Pay Lab', 'AI Lab, Tencent', 'FiT,Tencent', 'Tencent AI Lab Seattle', 'Tencent AI Platform', 'Tencent AI for Life Science Lab', 'Tencent AMS', 'Tencent ARC Lab', 'Tencent America', 'Tencent Content and Platform Group', 'Tencent Ethereal Audio Lab', 'Tencent Games', 'Tencent Hunyuan / AI Lab', 'Tencent Hunyuan Research', 'Tencent Hunyuan Team', 'Tencent Inc.', 'Tencent Jarvis Lab', 'Tencent MoreFun Studios', 'Tencent Music Entertaining', 'Tencent Music Entertainment Lyra Lab', 'Tencent PCG ARC Lab', 'Tencent Robotics X', 'Tencent Technology (Shenzhen) Co.', 'Tencent Technology Company', 'Tencent WeChat AI', 'Tencent Wechat AI', 'Tencent XR Vision Lab', 'Tencent XR Vision Labs', 'Tencent YOUTU Lab', 'Tencent Youtu', 'WeChat AI', 'WeChat AI, Tencent', 'WeChat AI, Tencent Inc', 'WeChat AI, Tencent Inc.', 'WeChat Vision, Tencent Inc.', 'WeChat, Tencent', 'WeChat, Tencent Inc.', 'Wechat, Tencent', 'XR Lab, Tencent Game', 'tencent'],
            'ByteDance': ['ByteDance', 'TikTok', 'ByteDance Inc', 'Bytedance Inc', 'Bytedance(Seed)', 'Bytedance Seed', 'ByteDance AI Lab', 'ByteDance AI Lab Research', 'ByteDance AILab', 'ByteDance Inc.', 'ByteDance Research', 'ByteDance TikTok', 'ByteDance, AI Lab', 'Bytedance', 'Bytedance Inc.', 'Bytedance Research', 'Bytedance US AILab', 'Bytedance inc.', 'Tiktok', 'bytedance', 'bytedance Inc'],
            'Baidu': ['Baidu', 'Baidu Inc', 'Baidu Research', 'Baidu Inc.'],
            'Huawei': ['Huawei', 'Huawei Technologies', 'Huawei Noah\'s Ark Lab', 'Huawei Cloud', 'Huawei Inc', 'Huawei Kirin Solution', 'Huawei International', 'Huawei Technology', 'Huawei Shanghai', 'Huawei Hong Kong Research Center', 'Huawei Hisilicon', 'Huawei Taylor Lab', 'Noah\'s Ark Lab, Huawei Technologies', 'Huawei Technologies Canada', 'Huawei Technologies Riemann Lab', 'Huawei Canada', 'Huawei Canada, Huawei Noah\'s Ark Lab', 'Huawei International.', 'Huawei Noah\'s Ark Lab (AI Lab)', 'Huawei Noah\'s Ark Lab', 'Huawei Tech. Investment Co., Limited', 'Huawei Technologies Ltd', 'Huawei Technologies Ltd.', 'Huawei Technologies Ltd. (Pairs Resaerch Center)', 'Huawei Technologies Research & Development (UK) Ltd', 'Huawei Technologies Research & Development (UK) Ltd.', 'Noah\'s Ark Lab, Huawei', 'Noah\'s Ark Lab, Huawei'],
            'IBM': ['IBM', 'Red Hat', 'IBM Research', 'IBM T. J. Watson Research Center', 'IBM Software', 'IBM Research Europe', 'IBM Research -Zurich', 'IBM Research New York', 'IBM Research -Israel', 'IBM Research -Europe', 'IBM India Pvt Ltd', 'IBM Research - Tokyo, International Business Machines', 'IBM Research AI', 'IBM Research India', 'IBM Research, Thomas J. Watson Research Center', 'IBM TJ Watson Research Center', 'IBM, International Business Machines', 'Red Hat. Inc'],
            'Intel': ['Intel Labs', 'Intel Corporation', 'Intel Labs China', 'Intel Labs, USA'],
            'Adobe': ['Adobe', 'Adobe Research', 'Adobe Inc', 'Adobe Systems'],
            'Samsung': ['Samsung', 'Samsung Research', 'Samsung Electronics', 'Samsung R&D Institute China-Beijing', 'Samsung AI Center', 'Samsung Electronic Research Centre of China', 'Samsung AI Center Toronto', 'Samsung Electorics', 'Samsung - SAIT AI Lab, Montreal', 'Samsung AI', 'Samsung AI Center, Cambridge', 'Samsung AI Lab, Montreal', 'Samsung AI Research Centre', 'Samsung Advanced Institute of Technology', 'Samsung Advanced Institute of Technology (SAIT)', 'Samsung Electronics America', 'Samsung Rearch China-Beijing(SRCB)', 'Samsung Research America', 'Samsung Research China – Beijing', 'Samsung Research UK', 'Samsung SDS', 'Samsung Semiconductor, INC.', 'Samsung advanced institute of technology'],
            'Sony': ['Sony', 'Sony Group Corporation', 'Sony Research', 'Sony AI', 'Sony AI America', 'Sony America', 'Sony Corporation', 'Sony Corporation of America', 'Sony Europe B.V.', 'Sony Europe Ltd.', 'Sony Group Coorporation', 'Sony Group Corp.', 'Sony Group Corporation, Tokyo', 'Sony Interactive Entertainment', 'Sony Interactive Entertainment Europe', 'Sony Research Inc.', 'Sony Semiconductor Solutions'],
            'OpenAI': ['OpenAI'],
            'Salesforce': ['Salesforce', 'SalesForce', 'SalesForce AI Research', 'SalesForce.com', 'Salesforce AI', 'Salesforce AI Research', 'Salesforce Inc.', 'Salesforce Research', 'Salesforce.com', 'salesforce.com'],
            'Tesla': ['Tesla', 'Tesla Inc.'],
            'Uber': ['Uber'],
            'Airbnb': ['Airbnb'],
            'Netflix': ['Netflix', 'NetFlix'],
            'Spotify': ['Spotify', 'Spotify Barcelona'],
            'Qualcomm': ['Qualcomm', 'QualComm', 'Qualcomm AI Research', 'Qualcomm Inc, QualComm', 'Qualcomm Inc, Qualcomm', 'Qualcomm Inc.', 'Qualcomm Technologies'],
            'ARM': ['ARM'],
            'Anthropic': ['Anthropic'],
            'Cohere': ['Cohere', 'Cohere AI', 'Cohere For AI', 'Cohere Labs'],
            'Stability AI': ['Stability AI'],
            'Midjourney': ['Midjourney'],
            'Runway': ['Runway'],
            'Scale AI': ['Scale AI'],
            'Palantir': ['Palantir']
        }
    
    def clean_organization_name(self, org_name: str) -> str:
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
    
    def get_famous_ai_company_canonical_name(self, org_name: str) -> Optional[str]:
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
                    pattern = r'\b' + re.escape(variant_lower) + r'\b'
                    if re.search(pattern, org_lower):
                        # 选择最长的匹配
                        if len(variant_lower) > best_match_length:
                            best_match = canonical_name
                            best_match_length = len(variant_lower)
        
        return best_match
    
    def is_famous_ai_company(self, org_name: str) -> bool:
        """判断是否为知名AI公司"""
        return self.get_famous_ai_company_canonical_name(org_name) is not None


def is_famous_company(aff_list: List[str]) -> Tuple[bool, str]:
    """
    判断公司是否为知名公司
    
    Args:
        aff_list: 组织名称列表
        
    Returns:
        Tuple[bool, str]: (是否为知名公司, 标准化的公司名称)
    """
    analyzer = FamousCompanyAnalyzer()
    
    for aff in aff_list:
        if not aff or not aff.strip():
            continue
            
        # 清理组织名称
        clean_aff = analyzer.clean_organization_name(aff.strip())
        if not clean_aff:
            continue
            
        # 检查是否为知名公司
        canonical_name = analyzer.get_famous_ai_company_canonical_name(clean_aff)
        if canonical_name:
            return True, canonical_name
    
    return False, ""

