# 🪙 Ancient Coin Detection & Knowledge System (古钱币智能检测与科普系统)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s-green.svg)](https://github.com/ultralytics/ultralytics)
[![PySide6](https://img.shields.io/badge/GUI-PySide6-orange.svg)](https://pypi.org/project/PySide6/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/Web-React-61dafb.svg)](https://react.dev/)

## 📝 Introduction (项目简介)

本项目是一个集**深度学习目标检测**与**历史文化科普**于一体的古钱币智能识别系统。
系统基于 **YOLOv8** 算法，面向六类中国古代钱币图像完成目标检测。项目保留 **PySide6** 桌面 GUI，同时新增 **FastAPI + React** Web 工作台，用户可实现图片上传、检测结果展示、置信度调节、结果保存及历史背景查询。

当前实验数据集由 395 张古钱币图像组成，使用 Roboflow 完成目标框标注，并划分为训练集 278 张、验证集 60 张、测试集 57 张。项目没有额外生成本地离线增强图片，训练阶段使用 YOLOv8 默认训练流程中的在线增强策略。原始训练图片仅保留在本地 `data/`，不随 GitHub 仓库上传。

## 📸 System Showcase (功能演示)

### 1. 桌面端交互界面 (GUI)

![Training Result](./docs/assets/coin_v8s_768_results.png)
*(最新模型训练曲线)*

### 2. 识别效果 (Detection Result)

![Detection Result](./docs/assets/coin_v8s_768_test_pred.jpg)
*(最新模型在测试集上的画框效果)*

## 🚀 Key Features (核心亮点)

- **Model**: 核心模型采用 YOLOv8s-768，测试集 mAP50 达到 **0.9698**，mAP50-95 达到 **0.9300**。
- **Dataset**: 本地实验数据集包含 395 张古钱币图像，按 YOLO 格式组织为训练集、验证集和测试集。
- **GUI**: 基于 PySide6 开发，支持图片上传、结果渲染、置信度阈值调节和结果保存。
- **Web**: 基于 FastAPI 和 React/Vite 新增本机浏览器工作台，后端负责 YOLO 推理，前端负责上传、预览、检测列表、科普详情和结果下载。
- **Knowledge Base**: 内置六类古钱币历史知识说明，实现“识别+科普”的演示效果。

## 📊 Performance (模型性能对比)

基于本地 `data/test` 的统一评估结果：

| 模型版本              | 输入尺寸 | mAP50     | mAP50-95  | Precision | Recall | 推理速度 (ms) |
|:------------------|:-----|:----------|:----------|:----------|:-------|:----------|
| **YOLOv8s (推荐)** | 768  | **0.9698** | **0.9300** | 0.9524    | 0.9678 | 约 9ms   |
| YOLOv8n           | 768  | 0.9142    | 0.8663    | 0.8241    | 0.9423 | 约 8ms   |
| YOLOv8s           | 640  | 0.9531    | 0.8991    | 0.8676    | 0.9336 | 约 10ms  |
| YOLOv8n           | 640  | 0.9187    | 0.8885    | 0.9286    | 0.8924 | 约 6ms   |

## 📂 Directory Structure (项目结构)

```text
.
├── best_models/          # 四组对比模型与发布版权重
├── config/               # 数据集配置文件 (data.yaml)
├── data/                 # 本地训练/验证/测试数据集（不上传）
├── runs/                 # 实验日志与评估图表 (PR曲线、混淆矩阵)
├── samples/              # GUI与预测演示图片
├── archive/              # 旧数据、原始导出和旧实验结果归档
├── web_backend/          # FastAPI 推理服务
├── frontend/             # React/Vite Web 工作台
├── main_gui.py           # 【核心】GUI 桌面系统启动脚本
├── train.py              # 模型训练脚本
├── dataset_check.py      # 数据集图片、标签和边界框检查脚本
├── compare_models.py     # 模型性能对比评估脚本
├── predict.py            # 命令行预测脚本
├── pyproject.toml        # 项目依赖配置
└── uv.lock               # 依赖锁定文件
```

## 🛠️ Usage (常用命令)

推荐使用 `uv` 运行项目：

```bash
uv --cache-dir .uv-cache sync
```

如果已经存在 `.venv`，日常运行可以跳过同步，避免重复下载依赖：

```bash
uv --cache-dir .uv-cache run --no-sync python dataset_check.py
```

启动桌面识别系统：

```bash
uv --cache-dir .uv-cache run --no-sync python main_gui.py
```

启动 Web 后端：

```bash
uv --cache-dir .uv-cache run --no-sync uvicorn web_backend.app:app --host 127.0.0.1 --port 8000
```

启动 Web 前端：

```bash
cd frontend
npm install
npm run dev
```

默认前端地址为 `http://127.0.0.1:5173`，开发期会将 `/api` 请求代理到 `http://127.0.0.1:8000`。如果先执行 `npm run build`，FastAPI 会在 `frontend/dist` 存在时托管构建后的静态页面。

检查数据集图片与标签是否匹配：

```bash
uv --cache-dir .uv-cache run --no-sync python dataset_check.py
```

生成完整数据集检查报告：

```bash
uv --cache-dir .uv-cache run --no-sync python dataset_check.py --report data_report.txt
```

训练模型：

```bash
uv --cache-dir .uv-cache run --no-sync python train.py --model pretrained/yolov8s.pt --name coin_v8s_768 --imgsz 768 --batch 8 --exist-ok
```

对示例图片运行预测并保存结果：

```bash
uv --cache-dir .uv-cache run --no-sync python predict.py --source samples --save
```

对比多个模型在测试集上的指标：

```bash
uv --cache-dir .uv-cache run --no-sync python compare_models.py
```

当前默认数据集是本地 `data/`，配置入口为 `config/data.yaml`。`best_models/` 里已经整理出四组可直接复现的对比权重，`compare_models.py` 默认就会按各自输入尺寸评估它们。当前 GUI 默认推理模型仍然是发布版 YOLOv8s 768：

```text
best_models/coin_v8s_768_best.pt
```

旧数据集、原始导出、历史实验、历史整理脚本和过时权重主要用于备份追溯；当前训练、预测和 GUI 默认流程不依赖这些归档内容。
