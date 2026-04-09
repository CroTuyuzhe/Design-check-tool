#!/bin/bash
# 设计走查工具 - 一键启动
cd "$(dirname "$0")"

# 检查 Python3
if ! command -v python3 &> /dev/null; then
    osascript -e 'display dialog "请先安装 Python 3\nhttps://www.python.org/downloads/" with title "设计走查工具" buttons {"确定"} default button "确定"'
    exit 1
fi

# 安装依赖
echo "🔧 检查依赖..."
pip3 install -q flask opencv-python-headless numpy Pillow scikit-image 2>/dev/null

# 启动服务
echo "🚀 启动设计走查工具..."
python3 app.py &
SERVER_PID=$!

# 等服务启动
sleep 3

# 打开浏览器
open http://localhost:5002

echo ""
echo "✅ 设计走查工具已启动"
echo "🌐 访问地址: http://localhost:5002"
echo ""
echo "按 Ctrl+C 停止服务"

# 保持运行
trap "kill $SERVER_PID 2>/dev/null; exit" INT TERM
wait $SERVER_PID
