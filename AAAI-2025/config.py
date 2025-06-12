# AAAI 2025 论文下载器配置文件

# 网站配置
AAAI_ISSUE_URL = "https://ojs.aaai.org/index.php/AAAI/issue/view/624"

# 下载配置
DEFAULT_SAVE_DIR = "AAAI-2025-Papers"
DEFAULT_DELAY = 1.5  # 下载间隔（秒）
MAX_DOWNLOADS = None  # 限制下载数量，None表示下载全部

# 网络配置
REQUEST_TIMEOUT = 30  # 请求超时时间（秒）
PAGE_TIMEOUT = 10     # 页面获取超时时间（秒）

# 重试配置
MAX_RETRY_ATTEMPTS = 5  # 最大重试次数
RETRY_DELAY = 2.0       # 重试间隔（秒）

# 文件名配置
MAX_FILENAME_LENGTH = 150  # 最大文件名长度

# User-Agent列表
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
]

# 日志配置
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# HTML解析配置
ARTICLE_CONTAINER_CLASS = "obj_article_summary"
TITLE_CLASS = "title"
GALLEY_LINKS_CLASS = "galleys_links"
PDF_LINK_CLASS = "obj_galley_link pdf" 