"""
PDF下载器模块
提供批量下载PDF文件的功能
"""

from .pdf_downloader import (
    PDFDownloader,
    download_pdfs,
    download_single_pdf
)

__all__ = [
    'PDFDownloader',
    'download_pdfs', 
    'download_single_pdf'
]

__version__ = "1.0.0" 