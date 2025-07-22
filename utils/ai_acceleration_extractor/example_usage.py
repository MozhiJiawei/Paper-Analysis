"""
AI推理加速论文提取器使用示例
"""

import os
from utils.ai_acceleration_extractor import (
    ai_acceleration_parse,
    ai_acceleration_parse_paper_copilot,
    AiAccelerationExtractor
)


def example_pdf_analysis():
    """PDF论文分析示例"""
    print("=== PDF论文分析示例 ===")

    # 检查是否有测试PDF文件
    test_pdf_dir = "./test_pdfs"
    if not os.path.exists(test_pdf_dir):
        print(f"测试PDF目录不存在: {test_pdf_dir}")
        print("请创建测试PDF目录并放入一些PDF文件")
        return

    # 使用便捷函数进行分析
    try:
        ai_acceleration_parse(
            papers_dir=test_pdf_dir,
            output_dir="./results",
            enable_llm_judge=False,  # 禁用LLM以简化示例
            match_threshold=5
        )
        print("✅ PDF分析完成")
    except Exception as e:
        print(f"❌ PDF分析失败: {e}")


def example_paper_copilot_analysis():
    """paper_copilot数据分析示例"""
    print("\n=== paper_copilot数据分析示例 ===")

    # 模拟paper_copilot数据
    paper_infos = [
        {
            "title": "Efficient Inference Acceleration for Large Language Models",
            "author": "John Doe; Jane Smith",
            "aff": "Stanford University; MIT",
            "email": "john@stanford.edu; jane@mit.edu",
            "abstract": "This paper presents a novel approach for accelerating inference in large language models through quantization and pruning techniques. We achieve 3x speedup with minimal accuracy loss.",
            "filename": "test_paper_1"
        },
        {
            "title": "A Study on Climate Change",
            "author": "Alice Johnson",
            "aff": "Harvard University",
            "email": "alice@harvard.edu",
            "abstract": "This paper examines the effects of climate change on global ecosystems.",
            "filename": "test_paper_2"
        },
        {
            "title": "Neural Network Compression for Mobile Deployment",
            "author": "Bob Wilson; Carol Brown",
            "aff": "Google Research; Microsoft Research",
            "email": "bob@google.com; carol@microsoft.com",
            "abstract": "We propose a new method for compressing neural networks to enable efficient deployment on mobile devices. Our approach combines knowledge distillation with quantization.",
            "filename": "test_paper_3"
        }
    ]

    try:
        ai_acceleration_parse_paper_copilot(
            paper_infos=paper_infos,
            output_dir="./results",
            enable_llm_judge=False,  # 禁用LLM以简化示例
            match_threshold=5
        )
        print("✅ paper_copilot分析完成")
    except Exception as e:
        print(f"❌ paper_copilot分析失败: {e}")


def example_advanced_usage():
    """高级使用示例"""
    print("\n=== 高级使用示例 ===")

    # 创建提取器实例
    extractor = AiAccelerationExtractor(
        papers_dir="./test_pdfs",
        output_dir="./results",
        enable_llm_judge=False,
        match_threshold=5
    )

    # 自定义分析
    try:
        # 分析特定文件
        specific_files = ["paper1.pdf", "paper2.pdf"]
        extractor.parse(paper_filenames=specific_files, analyze_all=False)
        print("✅ 特定文件分析完成")

        # 分析所有文件
        extractor.parse(analyze_all=True)
        print("✅ 全量分析完成")

    except Exception as e:
        print(f"❌ 高级使用示例失败: {e}")


def example_component_usage():
    """组件使用示例"""
    print("\n=== 组件使用示例 ===")

    from utils.ai_acceleration_extractor import KeywordMatcher, LLMJudge

    # 使用关键词匹配器
    matcher = KeywordMatcher(threshold=5)
    result = matcher.match_keywords(
        "AI Inference Acceleration",
        "This paper focuses on accelerating AI inference..."
    )
    print(f"关键词匹配结果: 匹配={result.is_match}, 分数={result.keyword_count}")

    # 使用LLM判别器
    judge = LLMJudge()
    # 注意：这里需要实际的LLM API才能工作
    print("LLM判别器已初始化（需要API密钥才能实际使用）")


def main():
    """主函数"""
    print("AI推理加速论文提取器使用示例")
    print("=" * 50)

    # 运行各种示例
    example_pdf_analysis()
    example_paper_copilot_analysis()
    # example_advanced_usage()
    # example_component_usage()

    print("\n" + "=" * 50)
    print("示例运行完成！")
    print("请查看 ./results 目录中的输出文件")


if __name__ == "__main__":
    main() 