#!/bin/bash

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK_SCRIPT="$SCRIPT_DIR/task.sh"

# 检查task.sh是否存在
if [ ! -f "$TASK_SCRIPT" ]; then
    echo "错误: 找不到 $TASK_SCRIPT"
    exit 1
fi

# 确保task.sh有执行权限
chmod +x "$TASK_SCRIPT"

# 创建临时cron文件
CRON_TEMP=$(mktemp)

# 导出当前用户的cron任务
crontab -l > "$CRON_TEMP" 2>/dev/null || true

# 检查是否已经存在相同的任务
CRON_ENTRY="0 22 * * * $TASK_SCRIPT"
if grep -Fxq "$CRON_ENTRY" "$CRON_TEMP" 2>/dev/null; then
    echo "定时任务已存在，无需重复添加。"
    rm -f "$CRON_TEMP"
    exit 0
fi

# 添加新的cron任务
echo "$CRON_ENTRY" >> "$CRON_TEMP"

# 安装新的cron任务
crontab "$CRON_TEMP"

# 清理临时文件
rm -f "$CRON_TEMP"

echo "定时任务已成功设置！"
echo "任务将在每日22:00执行: $TASK_SCRIPT"
echo ""
echo "当前cron任务列表:"
crontab -l

