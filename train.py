import argparse
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Train a YOLO model for ancient coin detection.")
    parser.add_argument("--model", default="pretrained/yolov8s.pt", help="Initial model weights.")
    parser.add_argument("--data", default="config/data.yaml", help="YOLO dataset yaml.")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--name", default="coin_v8s_768")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing run name.")
    return parser.parse_args()


def resolve_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / candidate


def main():
    args = parse_args()
    model_path = resolve_path(args.model)
    data_path = resolve_path(args.data)

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_path}")

    model = YOLO(str(model_path))

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        patience=args.patience,
        imgsz=args.imgsz,
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
