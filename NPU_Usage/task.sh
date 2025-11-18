#!/bin/bash

# 检查NPU资源是否被占用
echo "正在检查NPU资源占用情况..."

# 运行 npu-smi info 命令并捕获输出
NPU_INFO=$(npu-smi info)

# 检查输出中是否包含进程信息
# 如果包含 "No running processes found" 说明所有NPU都没有进程占用
# 如果有进程，会显示类似: | 0       0                 | 805990        | mindie_llm_back          | 23975                   |

# 首先检查是否有进程运行（通过查找进程ID列，排除表头）
# 检查是否存在格式为 | NPU号 Chip号 | 进程ID(6位数字) | 进程名 | 内存 | 的行
if echo "$NPU_INFO" | grep -qE "\|[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]+\|[[:space:]]*[0-9]{5,}[[:space:]]+\|[[:space:]]*[a-zA-Z_]+[[:space:]]+\|"; then
    echo "检测到NPU资源正在被占用，跳过执行。"
    exit 0
fi

# 检查是否所有NPU都显示 "No running processes found"
if echo "$NPU_INFO" | grep -q "No running processes found"; then
    echo "NPU资源未被占用，开始执行任务..."
    
    # 切换到指定目录
    cd /data/z00872399/NPU_PerformanceUsage || {
        echo "错误: 无法切换到目录 /data/z00872399/NPU_PerformanceUsage"
        exit 1
    }
    
    # 执行脚本
    bash NPU_PerformanceUsage.sh NPU_PerformanceUsage
    
    echo "任务执行完成。"
else
    echo "无法确定NPU资源状态，为安全起见跳过执行。"
    exit 1
fi

