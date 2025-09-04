import os
import yaml
import logging
from pathlib import Path
from volcenginesdkarkruntime import Ark
from typing import List, Dict, Any, Optional

# 获取模块级别的logger
logger = logging.getLogger(__name__)

class DoubaoAPI:
    """豆包API封装类，隐藏内部Ark实现细节"""
    
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        """
        初始化豆包API客户端
        
        Args:
            api_key: API密钥，如果不提供则从配置文件或环境变量读取
            base_url: API基础URL
            model: 模型名称
        """
        # 加载配置文件
        config = self._load_config()
        
        self.api_key = api_key or os.getenv("ARK_API_KEY") or config.get("api_key")
        self.base_url = base_url or config.get("base_url", "https://ark.cn-beijing.volces.com/api/v3")
        self.model = model or config.get("model", "doubao-seed-1-6-250615")
        
        if not self.api_key:
            raise ValueError("API密钥未设置。请在config.yaml文件中设置，或通过ARK_API_KEY环境变量提供，或在初始化时传入。")
        
        # 初始化内部客户端
        self._client = Ark(
            base_url=self.base_url,
            timeout=1800,
            api_key=self.api_key,
        )
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置文件
        
        Returns:
            配置字典
        """
        config_path = Path(__file__).parent / "config.yaml"
        
        if not config_path.exists():
            logger.warning(f"配置文件 {config_path} 不存在。")
            logger.warning("请复制 config_template.yaml 为 config.yaml 并填写您的API密钥。")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('doubao', {}) if config else {}
        except Exception as e:
            logger.warning(f"读取配置文件失败: {e}")
            return {}
    
    def chat(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        发送聊天消息
        
        Args:
            messages: 消息列表，格式为[{"role": "user/assistant", "content": "..."}]
        
        Returns:
            返回处理后的响应结果
        """
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )

            rsp_content = ""
            with response:  # 确保在代码块执行完毕后自动关闭连接，避免链接泄露
                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        rsp_content += chunk.choices[0].delta.content
                        print(chunk.choices[0].delta.content, end="")

            # 提取并返回有用的信息，隐藏内部实现细节
            return {
                "success": True,
                "content": rsp_content,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                } if hasattr(response, 'usage') and response.usage else None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": None
            }


# 创建全局实例
_doubao_instance = DoubaoAPI()


def call_doubao(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    对外提供的豆包API调用接口
    
    Args:
        messages: 消息列表
        
    Returns:
        响应结果字典
    """
    return _doubao_instance.chat(messages)


if __name__ == "__main__":
    # 测试代码
    messages = [
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ]
    
    result = call_doubao(messages)
    logger.info("文本对话结果:")
    logger.info(result)
    
    # 测试图片对话
    messages2 = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://ark-project.tos-cn-beijing.ivolces.com/images/view.jpeg"
                    },
                },
                {"type": "text", "text": "这是哪里？"},
            ],
        }
    ]
    
    result2 = call_doubao(messages2)
    logger.info("\n图片对话结果:")
    logger.info(result2)

