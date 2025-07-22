#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试知名公司判定功能
"""

from famous_company import is_famous_company, FamousCompanyAnalyzer


def test_is_famous_company():
    """测试is_famous_company函数"""
    print("开始测试知名公司判定功能...")
    
    # 测试用例
    test_cases = [
        # 知名公司测试
        (["Google Research"], True, "Google"),
        (["Microsoft Research"], True, "Microsoft"),
        (["Meta AI"], True, "Meta"),
        (["Amazon AWS AI Labs"], True, "Amazon"),
        (["NVIDIA Research"], True, "NVIDIA"),
        (["Alibaba Group"], True, "Alibaba"),
        (["Tencent AI Lab"], True, "Tencent"),
        (["ByteDance Inc"], True, "ByteDance"),
        (["Baidu Research"], True, "Baidu"),
        (["Huawei Technologies"], True, "Huawei"),
        (["IBM Research"], True, "IBM"),
        (["Intel Labs"], True, "Intel"),
        (["Adobe Research"], True, "Adobe"),
        (["Samsung Research"], True, "Samsung"),
        (["Sony AI"], True, "Sony"),
        (["OpenAI"], True, "OpenAI"),
        (["Salesforce AI"], True, "Salesforce"),
        (["Tesla"], True, "Tesla"),
        (["Uber"], True, "Uber"),
        (["Airbnb"], True, "Airbnb"),
        (["Netflix"], True, "Netflix"),
        (["Spotify"], True, "Spotify"),
        (["Qualcomm"], True, "Qualcomm"),
        (["ARM"], True, "ARM"),
        (["Anthropic"], True, "Anthropic"),
        (["Cohere AI"], True, "Cohere"),
        (["Stability AI"], True, "Stability AI"),
        (["Midjourney"], True, "Midjourney"),
        (["Runway"], True, "Runway"),
        (["Scale AI"], True, "Scale AI"),
        (["Palantir"], True, "Palantir"),
        
        # 非知名公司测试
        (["MIT"], False, ""),
        (["Stanford University"], False, ""),
        (["University of California, Berkeley"], False, ""),
        (["Tsinghua University"], False, ""),
        (["Peking University"], False, ""),
        (["Some Random Company"], False, ""),
        ([""], False, ""),
        ([], False, ""),
        
        # 混合测试
        (["MIT", "Google Research"], True, "Google"),
        (["Stanford University", "Microsoft Research"], True, "Microsoft"),
        (["Some Company", "Meta AI", "Another Company"], True, "Meta"),
    ]
    
    passed = 0
    failed = 0
    
    for i, (aff_list, expected_is_famous, expected_company) in enumerate(test_cases, 1):
        is_famous, company = is_famous_company(aff_list)
        
        if is_famous == expected_is_famous and company == expected_company:
            print(f"✓ 测试 {i}: 通过 - {aff_list} -> {is_famous}, {company}")
            passed += 1
        else:
            print(f"✗ 测试 {i}: 失败 - {aff_list}")
            print(f"  期望: {expected_is_famous}, {expected_company}")
            print(f"  实际: {is_famous}, {company}")
            failed += 1
    
    print(f"\n测试结果: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("🎉 所有测试通过！")
    else:
        print("❌ 有测试失败，请检查实现")


def test_analyzer_class():
    """测试FamousCompanyAnalyzer类"""
    print("\n开始测试FamousCompanyAnalyzer类...")
    
    analyzer = FamousCompanyAnalyzer()
    
    # 测试清理功能
    test_clean_cases = [
        ("  Google Research  ", "Google Research"),
        ("Microsoft Research, Inc.", "Microsoft Research"),
        ("Meta AI, LLC", "Meta AI"),
        ("Amazon AWS AI Labs, Corp.", "Amazon AWS AI Labs"),
        ("", ""),
        ("   ", ""),
    ]
    
    for input_name, expected in test_clean_cases:
        result = analyzer.clean_organization_name(input_name)
        if result == expected:
            print(f"✓ 清理测试通过: '{input_name}' -> '{result}'")
        else:
            print(f"✗ 清理测试失败: '{input_name}' -> '{result}' (期望: '{expected}')")
    
    # 测试标准名称获取
    test_canonical_cases = [
        ("Google Research", "Google"),
        ("Microsoft Research", "Microsoft"),
        ("Meta AI", "Meta"),
        ("Amazon AWS AI Labs", "Amazon"),
        ("Some Random Company", None),
        ("", None),
    ]
    
    for input_name, expected in test_canonical_cases:
        result = analyzer.get_famous_ai_company_canonical_name(input_name)
        if result == expected:
            print(f"✓ 标准名称测试通过: '{input_name}' -> '{result}'")
        else:
            print(f"✗ 标准名称测试失败: '{input_name}' -> '{result}' (期望: '{expected}')")


if __name__ == "__main__":
    test_is_famous_company()
    test_analyzer_class() 