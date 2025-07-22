"""
PDF下载器使用示例
演示如何使用pdf_downloader模块下载PDF文件
"""

import logging
from pdf_downloader import PDFDownloader, download_pdfs, download_single_pdf
from arxiv_downloader import download_pdfs_from_arxiv, download_single_pdf_from_arxiv

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# 获取模块级别的logger
logger = logging.getLogger(__name__)

def example_1_download_single_pdf():
    """示例1：下载单个PDF文件"""
    logger.info("=== 示例1：下载单个PDF文件 ===")
    
    # 使用便捷函数下载单个PDF
    url = "https://example.com/sample.pdf"
    success = download_single_pdf(
        url=url,
        save_dir="single_download",
        filename="my_paper.pdf",
        max_retries=3,
        timeout=30
    )
    
    logger.info(f"下载结果: {'成功' if success else '失败'}")
    logger.info("")


def example_2_download_pdf_list_simple():
    """示例2：从简单URL列表下载PDF"""
    logger.info("=== 示例2：从简单URL列表下载PDF ===")
    
    # PDF URL列表
    pdf_urls = [
        "https://example.com/paper1.pdf",
        "https://example.com/paper2.pdf",
        "https://example.com/paper3.pdf"
    ]
    
    # 使用便捷函数下载
    results = download_pdfs(
        pdf_info_list=pdf_urls,
        save_dir="simple_download",
        delay=1.0,
        max_retries=3,
        timeout=30
    )
    
    logger.info(f"下载结果: 成功 {results['success']} 个，失败 {results['failed']} 个")
    logger.info("")


def example_3_download_pdf_list_with_titles():
    """示例3：从包含标题的PDF信息列表下载"""
    logger.info("=== 示例3：从包含标题的PDF信息列表下载 ===")
    
    # 包含标题的PDF信息列表
    pdf_info_list = [
        {
            "url": "https://example.com/paper1.pdf",
            "title": "Deep Learning for Computer Vision"
        },
        {
            "url": "https://example.com/paper2.pdf", 
            "title": "Natural Language Processing with Transformers"
        },
        {
            "url": "https://example.com/paper3.pdf",
            "title": "Reinforcement Learning: An Introduction"
        }
    ]
    
    # 使用便捷函数下载
    results = download_pdfs(
        pdf_info_list=pdf_info_list,
        save_dir="titled_download",
        delay=1.5,
        max_retries=3,
        timeout=30
    )
    
    logger.info(f"下载结果: 成功 {results['success']} 个，失败 {results['failed']} 个")
    logger.info("")


def example_4_use_downloader_class():
    """示例4：使用PDFDownloader类进行更精细的控制"""
    logger.info("=== 示例4：使用PDFDownloader类 ===")
    
    # 创建下载器实例
    downloader = PDFDownloader(
        save_dir="class_download",
        delay=2.0,
        max_retries=5,
        timeout=60,
        max_filename_length=100
    )
    
    # 混合格式的PDF信息列表
    pdf_info_list = [
        "https://example.com/paper1.pdf",  # 简单URL
        {
            "url": "https://example.com/paper2.pdf",
            "title": "Advanced Machine Learning Techniques"
        },
        {
            "url": "https://example.com/paper3.pdf",
            "title": "Quantum Computing and Its Applications"
        }
    ]
    
    # 下载PDF文件
    results = downloader.download_pdfs_from_list(pdf_info_list)
    logger.info(f"下载结果: 成功 {results['success']} 个，失败 {results['failed']} 个")
    
    # 获取下载统计信息
    stats = downloader.get_download_stats()
    logger.info(f"统计信息:")
    logger.info(f"  - 文件数量: {stats['count']}")
    logger.info(f"  - 总大小: {stats['total_size_mb']:.2f} MB")
    logger.info(f"  - 文件列表:")
    for file_info in stats['files'][:5]:  # 只显示前5个
        logger.info(f"    * {file_info['name']} ({file_info['size_mb']:.2f} MB)")
    logger.info("")


def example_5_integrate_with_existing_code():
    """示例5：集成到现有代码中（模拟从download_papers.py获取的数据）"""
    logger.info("=== 示例5：集成到现有代码中 ===")
    
    # 模拟从原有代码中获取的pdf_info_list数据
    # 这通常来自extract_pdf_links_with_titles函数
    simulated_pdf_info_list = [
        {
            "url": "https://ojs.aaai.org/index.php/AAAI/article/download/123/456",
            "title": "A Novel Approach to Deep Reinforcement Learning"
        },
        {
            "url": "https://ojs.aaai.org/index.php/AAAI/article/download/789/012",
            "title": "Transformers for Multi-Modal Learning"
        },
        {
            "url": "https://ojs.aaai.org/index.php/AAAI/article/download/345/678",
            "title": "Efficient Graph Neural Networks"
        }
    ]
    
    # 使用新的下载器替换原有的download_all_pdfs函数
    downloader = PDFDownloader(
        save_dir="AAAI-2025-Papers",  # 保持与原有代码相同的目录
        delay=1.5,
        max_retries=5,
        timeout=30
    )
    
    results = downloader.download_pdfs_from_list(simulated_pdf_info_list)
    
    logger.info(f"下载结果: 成功 {results['success']} 个，失败 {results['failed']} 个")
    
    # 获取统计信息（替换原有的check_download_results函数）
    stats = downloader.get_download_stats()
    logger.info(f"最终统计: {stats['count']} 个文件，总大小 {stats['total_size_mb']:.2f} MB")
    logger.info("")


def example_6_download_single_arxiv_paper():
    """示例6：下载单篇arXiv论文"""
    logger.info("=== 示例6：下载单篇arXiv论文 ===")
    logger.info("注意：如果遇到SSL证书验证错误，程序会自动配置SSL设置来解决")
    logger.info("")
    
    # 方式1：使用论文标题搜索
    logger.info("方式1：使用论文标题搜索")
    result1 = download_single_pdf_from_arxiv(
        paper_name="BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        save_dir="arxiv_downloads"
    )
    status1 = "成功" if list(result1.values())[0] == 0 else "失败"
    logger.info(f"按标题搜索结果: {status1} - {result1}")
    logger.info("")
    
    # 方式2：使用arXiv ID
    logger.info("方式2：使用arXiv ID")
    result2 = download_single_pdf_from_arxiv(
        paper_name="1706.03762",  # Transformer论文的arXiv ID
        save_dir="arxiv_downloads"
    )
    status2 = "成功" if list(result2.values())[0] == 0 else "失败"
    logger.info(f"按ID搜索结果: {status2} - {result2}")
    logger.info("")


def example_7_download_multiple_arxiv_papers():
    """示例7：批量下载多篇arXiv论文"""
    logger.info("=== 示例7：批量下载多篇arXiv论文 ===")
    logger.info("注意：SSL配置只会在第一次调用时执行")
    logger.info("")
    
    # 论文列表（混合使用标题和ID）
    paper_list = [
        "Wide Neural Networks Trained with Weight Decay Provably Exhibit Neural Collapse",
        "A Probabilistic Perspective on Unlearning and Alignment for Large Language Models",
        "Variational Diffusion Posterior Sampling with Midpoint Guidance",
        "Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series",
        "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Parameters for Reasoning",
        "Towards Understanding Why FixMatch Generalizes Better Than Supervised Learning",
        "Exploring The Loss Landscape Of Regularized Neural Networks Via Convex Duality",
        "Turning Up the Heat: Min-p Sampling for Creative and Coherent LLM Outputs"
    ]
    
    # 批量下载
    results = download_pdfs_from_arxiv(
        paper_name_list=paper_list,
        save_dir="arxiv_batch_downloads"
    )
    
    logger.info("下载结果:")
    for paper_name, status in results.items():
        status_text = {
            0: "成功",
            1: "搜索失败",
            2: "下载失败"
        }.get(status, "未知状态")
        logger.info(f"  - {paper_name}: {status_text}")
    
    # 统计结果
    success_count = sum(1 for status in results.values() if status == 0)
    total_count = len(results)
    logger.info(f"\n总计: {success_count}/{total_count} 下载成功")
    logger.info("")



def main():
    """运行所有示例"""
    logger.info("PDF下载器使用示例")
    logger.info("=" * 50)
    
    # 注意：以下示例使用的都是示例URL，实际运行时会失败
    # 在实际使用时，请替换为真实的PDF URL
    
    # example_1_download_single_pdf()
    # example_2_download_pdf_list_simple()
    # example_3_download_pdf_list_with_titles()
    # example_4_use_downloader_class()
    # example_5_integrate_with_existing_code()
    # example_6_download_single_arxiv_paper()
    example_7_download_multiple_arxiv_papers()

    logger.info("所有示例执行完成！")
    logger.info("\n注意：示例中使用的是虚拟URL，实际使用时请替换为真实的PDF链接。")


if __name__ == "__main__":
    main() 