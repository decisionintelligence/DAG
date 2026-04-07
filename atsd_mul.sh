#!/bin/bash
CMD1="sh ./scripts/covariate_forecasting/SWAN_V3_TUNE_part1.sh"
CMD2="sh ./scripts/covariate_forecasting/SWAN_V3_TUNE_part2.sh"

if ! pgrep -f "$CMD1" > /dev/null; then
    echo "错误：程序 $CMD1 没有在运行，退出脚本。"
    exit 1
fi

if ! pgrep -f "$CMD2" > /dev/null; then
    echo "错误：程序 $CMD2 没有在运行，退出脚本。"
    exit 1
fi

echo "所有程序开始运行了"

while pgrep -f "$CMD1" > /dev/null || pgrep -f "$CMD2" > /dev/null; do
    if pgrep -f "$CMD1" > /dev/null; then
        echo "程序 $CMD1 在运行。"
    fi

    if pgrep -f "$CMD2" > /dev/null; then
        echo "程序 $CMD2 在运行。"
    fi
    
    sleep 60
done

echo "所有程序已结束，正在关机..."
/usr/bin/shutdown -h now
