# 模型对比结果追溯说明

本文项目当前保留四组论文对比模型，对应两种网络规模和两种输入尺寸：

| 模型 | 权重路径 | 训练来源 | 评估输入尺寸 | 测试集结果 |
| --- | --- | --- | --- | --- |
| YOLOv8n-640 | `best_models/coin_v8n_640_best.pt` | `archive/slim_20260427/runs/detect/coin_v8n_grouped/weights/best.pt` | 640 | P=0.9286, R=0.8924, mAP50=0.9187, mAP50-95=0.8885 |
| YOLOv8s-640 | `best_models/coin_v8s_640_best.pt` | `archive/slim_20260427/runs/detect/coin_v8s_grouped/weights/best.pt` | 640 | P=0.8676, R=0.9336, mAP50=0.9531, mAP50-95=0.8991 |
| YOLOv8n-768 | `best_models/coin_v8n_768_best.pt` | `runs/detect/coin_v8n_768/weights/best.pt` | 768 | P=0.8241, R=0.9423, mAP50=0.9142, mAP50-95=0.8663 |
| YOLOv8s-768 | `best_models/coin_v8s_768_best.pt` | `runs/detect/coin_v8s_768/weights/best.pt` | 768 | P=0.9524, R=0.9678, mAP50=0.9698, mAP50-95=0.9300 |

统一评估命令如下：

```bash
uv run python compare_models.py
```

如果只验证某两组模型，可以显式指定：

```bash
uv run python compare_models.py --models v8n_640=best_models/coin_v8n_640_best.pt@640 v8s_768=best_models/coin_v8s_768_best.pt@768
```

说明：

- `640` 两组训练原始日志位于 `archive/slim_20260427/`，当前已复制一份权重到 `best_models/`，便于统一管理。
- `768` 两组训练日志与结果图保留在 `runs/detect/`，权重也同步整理到 `best_models/`。
- 论文如需引用实验结果，建议以本文件和 `README.md` 的最新测试集复核数据为准。
