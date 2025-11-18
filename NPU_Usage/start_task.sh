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

# 定义三个时间点的cron任务
CRON_ENTRY_13="0 13 * * * $TASK_SCRIPT"
CRON_ENTRY_19="0 19 * * * $TASK_SCRIPT"
CRON_ENTRY_24="0 0 * * * $TASK_SCRIPT"

# 检查并添加13:00的任务
if ! grep -Fxq "$CRON_ENTRY_13" "$CRON_TEMP" 2>/dev/null; then
    echo "$CRON_ENTRY_13" >> "$CRON_TEMP"
    echo "已添加13:00的定时任务"
else
    echo "13:00的定时任务已存在，跳过"
fi

# 检查并添加19:00的任务
if ! grep -Fxq "$CRON_ENTRY_19" "$CRON_TEMP" 2>/dev/null; then
    echo "$CRON_ENTRY_19" >> "$CRON_TEMP"
    echo "已添加19:00的定时任务"
else
    echo "19:00的定时任务已存在，跳过"
fi

# 检查并添加24:00（0:00）的任务
if ! grep -Fxq "$CRON_ENTRY_24" "$CRON_TEMP" 2>/dev/null; then
    echo "$CRON_ENTRY_24" >> "$CRON_TEMP"
    echo "已添加24:00（0:00）的定时任务"
else
    echo "24:00（0:00）的定时任务已存在，跳过"
fi

# 安装新的cron任务
crontab "$CRON_TEMP"

# 清理临时文件
rm -f "$CRON_TEMP"

echo ""
echo "定时任务已成功设置！"
echo "任务将在每日13:00、19:00、24:00（0:00）执行: $TASK_SCRIPT"
echo ""
echo "当前cron任务列表:"
crontab -l

