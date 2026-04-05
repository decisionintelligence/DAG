#!/bin/bash
CMD="sh ./scripts/covariate_forecasting/SWAN_grid_stage1.sh"  # 这里填入你启动模型训练时的命令，比如这里我用python run.py指令启动模型
 
 # 判断当前指令是不是在跑
if ! pgrep -f "$CMD" > /dev/null; then
    echo "错误：程序 $CMD 没有在运行，退出脚本。"
    exit 1
else
    echo "程序开始运行了"
fi


while pgrep -f "$CMD" > /dev/null; do
    sleep 60  # 每隔1min检查一次
done
 
echo "程序已结束，正在关机..."
/usr/bin/shutdown -h now