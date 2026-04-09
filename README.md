# 设计走查工具 Design Check

上传设计稿和实现截图，自动检测 UI 元素的位置偏移、尺寸差异、颜色偏差、圆角差异。

## 快速启动

**Mac 用户：** 双击 `启动.command` 即可

**手动启动：**
```bash
pip3 install flask opencv-python-headless numpy Pillow scikit-image
python3 app.py
# 浏览器打开 http://localhost:5002
```

## 功能

- 📍 位置偏移检测 (dp)
- 📐 尺寸差异检测
- 🎨 颜色偏差检测
- ⬜ 圆角差异检测
- 🔄 重合对比视图
- ✏️ 手动添加/编辑标注
- 📥 导出标注报告

## 系统要求

- macOS / Linux / Windows
- Python 3.8+

## 必要依赖

- 通过pip3获取开发者工具
- Python 或者 Pyinstaller
