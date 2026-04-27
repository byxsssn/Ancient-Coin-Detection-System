import argparse
from collections import Counter
from pathlib import Path
from shutil import move

import yaml


BASE_DIR = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description="Check YOLO dataset image/label consistency.")
    parser.add_argument("--data", default="config/data.yaml", help="YOLO dataset yaml.")
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--report", default=None, help="Optional path to write the full report.")
    parser.add_argument(
        "--quarantine",
        action="store_true",
        help="Move orphan images/labels to data/_orphaned instead of only reporting them.",
    )
    return parser.parse_args()


def resolve_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / candidate


def resolve_dataset_root(data_yaml, dataset_path=None):
    if dataset_path is None:
        return data_yaml.parent.resolve()

    candidate = Path(dataset_path)
    if candidate.is_absolute():
        return candidate
    return (data_yaml.parent / candidate).resolve()


def image_files_by_stem(directory):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return {path.stem: path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in exts}


def label_files_by_stem(directory):
    return {path.stem: path for path in directory.glob("*.txt")}


def count_classes(label_dir, class_count):
    counts = Counter()
    bad_lines = []
    empty_labels = []

    for label_file in label_dir.glob("*.txt"):
        label_lines = [line.strip() for line in label_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not label_lines:
            empty_labels.append(str(label_file))

        for line_no, stripped in enumerate(label_lines, start=1):
            parts = stripped.split()
            if len(parts) != 5:
                bad_lines.append(f"{label_file}:{line_no}: expected 5 columns, got {len(parts)}")
                continue
            try:
                class_id = int(parts[0])
                bbox = [float(value) for value in parts[1:]]
            except ValueError:
                bad_lines.append(f"{label_file}:{line_no}: invalid class id")
                continue
            if class_id < 0 or class_id >= class_count:
                bad_lines.append(f"{label_file}:{line_no}: class id {class_id} out of range")
                continue
            if any(value < 0 or value > 1 for value in bbox):
                bad_lines.append(f"{label_file}:{line_no}: bbox value out of 0..1 range")
                continue
            if bbox[2] <= 0 or bbox[3] <= 0:
                bad_lines.append(f"{label_file}:{line_no}: bbox width/height must be positive")
                continue
            counts[class_id] += 1

    return counts, bad_lines, empty_labels


def resolve_image_dir(dataset_root, data_yaml, image_rel, split):
    rel_path = Path(image_rel)
    split_dir = "valid" if split == "val" else split
    candidates = [
        dataset_root / rel_path,
        data_yaml.parent / rel_path,
        dataset_root / split_dir / "images",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def add_examples(lines, title, values, max_examples):
    values = sorted(values)
    lines.append(f"  {title}: {len(values)}")
    for value in values[:max_examples]:
        lines.append(f"    - {value}")


def original_id(stem):
    markers = ("_jpg.rf.", "_png.rf.", "_jpeg.rf.", ".rf.")
    for marker in markers:
        if marker in stem:
            return stem.split(marker, 1)[0]
    return stem


def write_report(report_path, lines):
    path = resolve_path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def move_to_quarantine(paths, dataset_root, split, kind):
    quarantine_dir = dataset_root / "_orphaned" / split / kind
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    moved = []
    for path in sorted(paths):
        target = quarantine_dir / path.name
        if target.exists():
            target = quarantine_dir / f"{path.stem}_{path.stat().st_mtime_ns}{path.suffix}"
        move(str(path), str(target))
        moved.append((path, target))
    return moved


def main():
    args = parse_args()
    data_yaml = resolve_path(args.data)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset yaml not found: {data_yaml}")

    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = resolve_dataset_root(data_yaml, config.get("path"))
    names = config["names"]

    lines = [
        f"Dataset: {dataset_root}",
        f"Classes: {len(names)}",
    ]
    quarantine_actions = []
    original_ids_by_split = {}

    for split, image_rel in (("train", config["train"]), ("val", config["val"]), ("test", config.get("test"))):
        if not image_rel:
            continue

        image_dir = resolve_image_dir(dataset_root, data_yaml, image_rel, split)
        label_dir = image_dir.parent / "labels"
        if not image_dir.exists() or not label_dir.exists():
            lines.append(f"\n[{split}] missing directory: images={image_dir.exists()} labels={label_dir.exists()}")
            continue

        image_files = image_files_by_stem(image_dir)
        label_files = label_files_by_stem(label_dir)
        image_stems = set(image_files)
        label_stems = set(label_files)
        missing_labels = image_stems - label_stems
        missing_images = label_stems - image_stems
        class_counts, bad_lines, empty_labels = count_classes(label_dir, len(names))
        original_ids_by_split[split] = {original_id(stem) for stem in image_stems}

        lines.append(f"\n[{split}] images={len(image_files)} labels={len(label_files)}")
        add_examples(lines, "images without labels", missing_labels, args.max_examples)
        add_examples(lines, "labels without images", missing_images, args.max_examples)
        add_examples(lines, "empty label files", empty_labels, args.max_examples)

        lines.append("  class distribution:")
        for class_id, name in enumerate(names):
            lines.append(f"    - {name}: {class_counts[class_id]}")

        if bad_lines:
            add_examples(lines, "bad label lines", bad_lines, args.max_examples)

        if args.quarantine:
            orphan_images = [image_files[stem] for stem in missing_labels]
            orphan_labels = [label_files[stem] for stem in missing_images]
            quarantine_actions.extend(move_to_quarantine(orphan_images, dataset_root, split, "images"))
            quarantine_actions.extend(move_to_quarantine(orphan_labels, dataset_root, split, "labels"))

    leakage = []
    split_names = list(original_ids_by_split)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1:]:
            overlap = original_ids_by_split[left] & original_ids_by_split[right]
            if overlap:
                leakage.append((left, right, overlap))

    lines.append("\n[split leakage check]")
    if not leakage:
        lines.append("  no original ids appear in more than one split")
    for left, right, overlap in leakage:
        lines.append(f"  {left} vs {right}: {len(overlap)} shared original ids")
        for value in sorted(overlap)[:args.max_examples]:
            lines.append(f"    - {value}")

    if quarantine_actions:
        lines.append("\n[quarantine]")
        for source, target in quarantine_actions:
            lines.append(f"  moved: {source} -> {target}")

    output = "\n".join(lines)
    print(output)

    if args.report:
        report_path = write_report(args.report, lines)
        print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
