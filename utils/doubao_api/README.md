# 豆包API封装库

## 简介

这是一个对豆包API的Python封装库，提供了简单易用的接口来调用豆包的聊天功能。

## 安装依赖

```bash
pip install volcengine-python-sdk pyyaml
```

## 配置

### 方法1：使用配置文件（推荐）

1. 复制配置文件模板：
   ```bash
   cp config_template.yaml config.yaml
   ```

2. 编辑 `config.yaml` 文件，填写您的API密钥：
   ```yaml
   doubao:
     api_key: "您的实际API密钥"
     base_url: "https://ark.cn-beijing.volces.com/api/v3"
     model: "doubao-seed-1-6-250615"
   ```

### 方法2：使用环境变量

设置环境变量：
```bash
export ARK_API_KEY="您的API密钥"
```

### 方法3：代码中直接传入

```python
from doubao import DoubaoAPI

api = DoubaoAPI(api_key="您的API密钥")
```

## 使用示例

### 简单文本对话

```python
from doubao import call_doubao

messages = [
    {
        "role": "user",
        "content": "你好，请介绍一下你自己"
    }
]

result = call_doubao(messages)
if result["success"]:
    print(result["content"])
else:
    print(f"错误: {result['error']}")
```

### 图片对话

```python
from doubao import call_doubao

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": "图片URL"
                }
            },
            {"type": "text", "text": "这是什么？"}
        ]
    }
]

result = call_doubao(messages)
if result["success"]:
    print(result["content"])
else:
    print(f"错误: {result['error']}")
```

## API参考

### call_doubao(messages)

调用豆包API进行对话。

**参数：**
- `messages`: 消息列表，格式为 `[{"role": "user/assistant", "content": "..."}]`

**返回值：**
```python
{
    "success": bool,        # 是否成功
    "content": str,         # AI回复内容（成功时）
    "model": str,          # 使用的模型名称
    "usage": {             # token使用情况
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int
    },
    "error": str           # 错误信息（失败时）
}
```

## 注意事项

1. **安全性**：`config.yaml` 文件包含敏感的API密钥，已被添加到 `.gitignore` 中，不会被上传到代码仓库。
2. **配置优先级**：代码参数 > 环境变量 > 配置文件
3. **错误处理**：所有API调用都会返回统一格式的结果，请检查 `success` 字段来判断是否成功。

## 故障排除

如果遇到"API密钥未设置"错误，请检查：
1. 是否正确创建了 `config.yaml` 文件
2. 配置文件中的API密钥是否正确
3. 环境变量是否正确设置 