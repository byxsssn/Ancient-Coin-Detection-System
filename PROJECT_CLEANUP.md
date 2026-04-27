# Project Cleanup Notes

## Keep

- `data2_grouped/`: cleaned training dataset with no image/label mismatches and no original-id split leakage.
- `best_models/coin_v8s_768_best.pt`: current release model, selected for best test mAP.
- `runs/detect/coin_v8s_768/weights/best.pt`: source training run for the release model.
- `samples/`: small demo image set for GUI and prediction smoke tests.
- `archive/legacy_20260427/`: old datasets, old runs, and isolated legacy weights kept for reference.
- `archive/slim_20260427/`: original `data2/`, old comparison runs, and unused pretrained weights.

## Optional Archive Or Delete

These are useful for history, but not required for the current default workflow:

- `archive/legacy_20260427/data/`: old dataset with image/label mismatches.
- `archive/slim_20260427/data2/`: original Roboflow export; keep it if you want to regenerate `data2_grouped/`.
- `archive/legacy_20260427/runs/detect/*`: older experiments.
- `archive/slim_20260427/pretrained/`: unused pretrained weights.
- `archive/slim_20260427/best_models/v8_best_899.pt`: old fallback deployment model.

## Regenerate

```bash
uv --cache-dir .uv-cache run --no-sync python split_dataset_by_origin.py --source archive/slim_20260427/data2 --output data2_grouped
uv --cache-dir .uv-cache run --no-sync python dataset_check.py --report data2_grouped_report.txt
uv --cache-dir .uv-cache run --no-sync python train.py --model pretrained/yolov8s.pt --name coin_v8s_768 --imgsz 768 --batch 8 --exist-ok
uv --cache-dir .uv-cache run --no-sync python compare_models.py --split test
```
