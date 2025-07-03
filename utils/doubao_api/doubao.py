import os
import yaml
from pathlib import Path
from volcenginesdkarkruntime import Ark
from typing import List, Dict, Any, Optional


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
            print(f"警告：配置文件 {config_path} 不存在。")
            print("请复制 config_template.yaml 为 config.yaml 并填写您的API密钥。")
            return {}
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('doubao', {}) if config else {}
        except Exception as e:
            print(f"警告：读取配置文件失败: {e}")
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
                messages=messages
            )
            
            # 提取并返回有用的信息，隐藏内部实现细节
            return {
                "success": True,
                "content": response.choices[0].message.content,
                "model": response.model,
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
        messages: 消息列表，格式示例：
            [
                {
                    "role": "user", 
                    "content": "你好"
                }
            ]
            或包含图片的消息：
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "图片URL"
                            }
                        },
                        {"type": "text", "text": "这是哪里？"}
                    ]
                }
            ]
    
    Returns:
        Dict: 包含响应结果的字典
            {
                "success": bool,
                "content": str,  # AI回复内容
                "model": str,   # 使用的模型名称
                "usage": dict,  # token使用情况
                "error": str    # 错误信息（仅在success为False时存在）
            }
    """
    return _doubao_instance.chat(messages)


# 示例用法（可以删除或注释掉）
if __name__ == "__main__":
    # 文本对话示例
    text_messages = [
        {
            "role": "user",
            "content": "你好，请介绍一下你自己"
        }
    ]
    
    # 图片对话示例
    image_messages = [
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
    
    # 调用API
    result = call_doubao(text_messages)
    print("文本对话结果:")
    print(result)
    
    result2 = call_doubao(image_messages)
    print("\n图片对话结果:")
    print(result2)

