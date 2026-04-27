import argparse
from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO prediction on an image, directory, or video.")
    parser.add_argument("--model", default="best_models/coin_v8s_768_best.pt", help="Model weights.")
    parser.add_argument("--source", default="samples", help="Image, directory, video, or camera source.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--save", action="store_true", help="Save annotated results under runs/detect.")
    parser.add_argument("--show", action="store_true", help="Display prediction window.")
    parser.add_argument("--name", default="predict")
    return parser.parse_args()


def resolve_path(path):
    candidate = Path(path)
    if candidate.is_absolute() or str(path).isdigit():
        return path
    return str(BASE_DIR / candidate)


def main():
    args = parse_args()
    model_path = Path(resolve_path(args.model))
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = YOLO(str(model_path))
    results = model.predict(
        source=resolve_path(args.source),
        conf=args.conf,
        imgsz=args.imgsz,
        save=args.save,
        show=args.show,
        name=args.name,
    )
    print(f"Predicted {len(results)} item(s).")


if __name__ == "__main__":
    main()
