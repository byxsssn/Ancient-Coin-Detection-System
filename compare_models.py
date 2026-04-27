import argparse
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODELS = {
    "release_v8s_768": "best_models/coin_v8s_768_best.pt",
    "coin_v8s_768": "runs/detect/coin_v8s_768/weights/best.pt",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate and compare multiple YOLO models.")
    parser.add_argument("--data", default="config/data.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--models", nargs="*", help="Optional entries in name=path format.")
    return parser.parse_args()


def resolve_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / candidate


def parse_model_entries(entries):
    if not entries:
        return DEFAULT_MODELS

    models = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Invalid model entry: {entry}. Use name=path.")
        name, path = entry.split("=", 1)
        models[name.strip()] = path.strip()
    return models


def main():
    args = parse_args()
    data_yaml = resolve_path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    models = parse_model_entries(args.models)
    for name, path in models.items():
        model_path = resolve_path(path)
        print(f"\nEvaluating {name}: {model_path}")
        if not model_path.exists():
            print(f"Skip {name}: file not found.")
            continue

        model = YOLO(str(model_path))
        metrics = model.val(
            data=str(data_yaml),
            split=args.split,
            batch=args.batch,
            imgsz=args.imgsz,
            name=f"val_compare_{name}",
            exist_ok=True,
            plots=True,
        )
        print(
            f"{name}: mAP50={metrics.box.map50:.4f}, "
            f"mAP50-95={metrics.box.map:.4f}, precision={metrics.box.mp:.4f}, recall={metrics.box.mr:.4f}"
        )


if __name__ == "__main__":
    main()
