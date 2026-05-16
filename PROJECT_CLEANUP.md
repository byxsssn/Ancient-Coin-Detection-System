# Project Cleanup Notes

## Current Release Scope

- `config/data.yaml`: dataset entry point for training, validation and prediction.
- `best_models/`: selected model weights for GUI use and result reproduction.
- `pretrained/`: YOLOv8 pretrained weights needed to rerun training.
- `runs/detect/`: current experiment outputs, including training curves, PR curves, confusion matrices and prediction examples.
- `docs/assets/` and `论文/配图建议/`: figures used by the README and thesis.
- `samples/`: small demo image set for GUI and command-line prediction.
- `论文/李业恺-22智能01-B20220307112-毕业论文正文.docx`: current thesis document.

## Not Uploaded

- `.venv/`, `.uv-cache/`, `__pycache__/` and `outputs/`: local runtime caches or generated demo outputs.
- `data/`: local training, validation and test dataset. The thesis records its scale and split, but the raw training images are not uploaded to GitHub.
- `archive/`: old datasets, old runs and unused historical weights. It is useful locally, but too large and not required by the current workflow.
- `论文/*_修改前备份_*.docx`: local thesis backup copies.
- `论文/学长论文.pdf`: reference material, not part of this project deliverable.
- `data/**/labels.cache`: generated YOLO cache files that can be rebuilt.

## Reproduce

```bash
# Requires local data/ to be present.
uv --cache-dir .uv-cache run --no-sync python dataset_check.py
uv --cache-dir .uv-cache run --no-sync python train.py --model pretrained/yolov8s.pt --name coin_v8s_768 --imgsz 768 --batch 8 --exist-ok
uv --cache-dir .uv-cache run --no-sync python compare_models.py --split test
uv --cache-dir .uv-cache run --no-sync python main_gui.py
```
