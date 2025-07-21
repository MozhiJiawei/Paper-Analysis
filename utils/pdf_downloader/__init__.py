"""
PDF下载器模块
提供批量下载PDF文件的功能，支持普通URL和arXiv论文下载
"""

from .pdf_downloader import (
    PDFDownloader,
    download_pdfs,
    download_single_pdf
)

from .arxiv_downloader import (
    download_single_pdf_from_arxiv,
    download_pdfs_from_arxiv
)

__all__ = [
    # 通用PDF下载器
    'PDFDownloader',
    'download_pdfs', 
    'download_single_pdf',
    # arXiv论文下载器
    'download_single_pdf_from_arxiv',
    'download_pdfs_from_arxiv'
]

__version__ = "1.1.0" 