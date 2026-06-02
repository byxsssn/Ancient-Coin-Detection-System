# Project Cleanup Notes

## Current Release Scope

- `config/data.yaml`: dataset entry point for training, validation and prediction.
- `best_models/`: selected model weights for Web inference and result reproduction.
- `pretrained/`: YOLOv8 pretrained weights needed to rerun training.
- `runs/detect/`: current experiment outputs, including training curves, PR curves, confusion matrices and prediction examples.
- `docs/assets/`: figures used by the README.
- `samples/`: small demo image set for Web API checks and command-line prediction.
- `web_backend/` and `frontend/`: FastAPI inference service and React/Vite Web workspace.

## Not Uploaded

- `.venv/`, `.uv-cache/`, `__pycache__/` and `outputs/`: local runtime caches or generated demo outputs.
- `data/`: local training, validation and test dataset. The README records its scale and split, but the raw training images are not uploaded to GitHub.
- `archive/`: old datasets, old runs and unused historical weights. It is useful locally, but too large and not required by the current workflow.
- `data/**/labels.cache`: generated YOLO cache files that can be rebuilt.

## Reproduce

```bash
# Requires local data/ to be present.
uv --cache-dir .uv-cache run --no-sync python dataset_check.py
uv --cache-dir .uv-cache run --no-sync python train.py --model pretrained/yolov8s.pt --name coin_v8s_768 --imgsz 768 --batch 8 --exist-ok
uv --cache-dir .uv-cache run --no-sync python compare_models.py --split test
uv --cache-dir .uv-cache run --no-sync uvicorn web_backend.app:app --host 127.0.0.1 --port 8000
```
