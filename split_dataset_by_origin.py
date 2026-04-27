import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml


BASE_DIR = Path(__file__).resolve().parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild a YOLO dataset without augmented-image split leakage.")
    parser.add_argument("--source", default="archive/slim_20260427/data2", help="Source dataset directory.")
    parser.add_argument("--output", default="data2_grouped", help="Output dataset directory.")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(path):
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return BASE_DIR / candidate


def original_id(stem):
    markers = ("_jpg.rf.", "_png.rf.", "_jpeg.rf.", ".rf.")
    for marker in markers:
        if marker in stem:
            return stem.split(marker, 1)[0]
    return stem


def read_classes(label_file):
    classes = Counter()
    for line in label_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        classes[int(line.split()[0])] += 1
    return classes


def collect_groups(source_dir):
    groups = defaultdict(list)
    for split in ("train", "valid", "test"):
        image_dir = source_dir / split / "images"
        label_dir = source_dir / split / "labels"
        if not image_dir.exists():
            continue
        for image_path in image_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise FileNotFoundError(f"Missing label for {image_path}")
            groups[original_id(image_path.stem)].append((image_path, label_path))
    return groups


def group_class_counts(items):
    counts = Counter()
    for _, label_path in items:
        counts.update(read_classes(label_path))
    return counts


def assignment_score(assigned_counts, assigned_images, targets, image_targets):
    score = 0
    for split, split_targets in targets.items():
        score += abs(image_targets[split] - assigned_images[split]) * 2
        for class_id, target_count in split_targets.items():
            score += abs(target_count - assigned_counts[split][class_id])
    return score


def assign_groups(groups, ratios, seed):
    group_items = list(groups.items())
    random.Random(seed).shuffle(group_items)
    group_items.sort(key=lambda item: len(item[1]), reverse=True)

    total_counts = Counter()
    group_counts = {}
    for group_id, items in group_items:
        counts = group_class_counts(items)
        group_counts[group_id] = counts
        total_counts.update(counts)

    targets = {
        split: {class_id: count * ratio for class_id, count in total_counts.items()}
        for split, ratio in ratios.items()
    }
    total_images = sum(len(items) for _, items in group_items)
    image_targets = {split: total_images * ratio for split, ratio in ratios.items()}
    assignments = {split: [] for split in ratios}
    assigned_counts = {split: Counter() for split in ratios}
    assigned_images = {split: 0 for split in ratios}

    for group_id, items in group_items:
        counts = group_counts[group_id]
        best_split = None
        best_score = None
        for split in ratios:
            trial_counts = {name: Counter(value) for name, value in assigned_counts.items()}
            trial_images = dict(assigned_images)
            trial_counts[split].update(counts)
            trial_images[split] += len(items)
            score = assignment_score(trial_counts, trial_images, targets, image_targets)
            if best_score is None or score < best_score:
                best_split = split
                best_score = score

        assignments[best_split].append((group_id, items))
        assigned_counts[best_split].update(counts)
        assigned_images[best_split] += len(items)

    return assignments, assigned_counts


def prepare_output(output_dir):
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for split in ("train", "valid", "test"):
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def ensure_safe_output(output_dir, source_dir):
    resolved_output = output_dir.resolve()
    resolved_source = source_dir.resolve()
    protected_paths = {
        BASE_DIR.resolve(),
        BASE_DIR.parent.resolve(),
        resolved_source,
    }
    if resolved_output in protected_paths:
        raise ValueError(f"Refusing to replace protected directory: {resolved_output}")
    if resolved_source in resolved_output.parents:
        raise ValueError(f"Output directory cannot be inside source directory: {resolved_output}")


def copy_assignments(assignments, output_dir):
    for split, groups in assignments.items():
        for _, items in groups:
            for image_path, label_path in items:
                shutil.copy2(image_path, output_dir / split / "images" / image_path.name)
                shutil.copy2(label_path, output_dir / split / "labels" / label_path.name)


def write_yaml(source_dir, output_dir):
    source_yaml = source_dir / "data.yaml"
    config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    config["path"] = "."
    config["train"] = "train/images"
    config["val"] = "valid/images"
    config["test"] = "test/images"
    (output_dir / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")


def main():
    args = parse_args()
    if round(args.train + args.val + args.test, 6) != 1:
        raise ValueError("Split ratios must add up to 1.0")

    source_dir = resolve_path(args.source)
    output_dir = resolve_path(args.output)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dir}")
    ensure_safe_output(output_dir, source_dir)

    groups = collect_groups(source_dir)
    assignments, assigned_counts = assign_groups(
        groups,
        {"train": args.train, "valid": args.val, "test": args.test},
        args.seed,
    )
    prepare_output(output_dir)
    copy_assignments(assignments, output_dir)
    write_yaml(source_dir, output_dir)

    print(f"Rebuilt dataset: {output_dir}")
    for split, split_groups in assignments.items():
        image_count = sum(len(items) for _, items in split_groups)
        print(f"{split}: groups={len(split_groups)} images={image_count} labels={image_count} classes={dict(assigned_counts[split])}")


if __name__ == "__main__":
    main()
