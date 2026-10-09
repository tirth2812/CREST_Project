
from pathlib import Path
import csv
import hashlib
import math
import random
import shutil

import cv2
import numpy as np


BASE = Path(r"C:\Do_Not_Delete_PLC\Conference")

WORK = BASE / "00_Pallet_100_Annotation_Work"

IMAGES = WORK / "images"
LABELS = WORK / "labels_reviewed"
PROGRESS = WORK / "review_progress.csv"
MANIFEST = WORK / "annotation_manifest.csv"

OUTPUT = BASE / "00_Pallet_100_YOLO26_Dataset"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

SEED = 42


def image_files(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
    )


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_csv(path):
    with path.open(
        "r", newline="", encoding="utf-8"
    ) as stream:
        return list(csv.DictReader(stream))


def check_labels(path):
    lines = [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]

    if not lines:
        raise ValueError(
            f"No pallet annotations: {path.name}"
        )

    for line in lines:
        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO row: {path.name}"
            )

        if parts[0] != "0":
            raise ValueError(
                f"Unexpected class: {path.name}"
            )

        try:
            cx, cy, bw, bh = map(float, parts[1:])
        except ValueError:
            raise ValueError(
                f"Invalid coordinates: {path.name}"
            )

        values = (cx, cy, bw, bh)

        if not all(math.isfinite(v) for v in values):
            raise ValueError(
                f"Nonfinite coordinate: {path.name}"
            )

        if not (
            0 <= cx <= 1
            and 0 <= cy <= 1
            and 0 < bw <= 1
            and 0 < bh <= 1
        ):
            raise ValueError(
                f"Out-of-range box: {path.name}"
            )

        tolerance = 0.00001

        if (
            cx - bw / 2 < -tolerance
            or cy - bh / 2 < -tolerance
            or cx + bw / 2 > 1 + tolerance
            or cy + bh / 2 > 1 + tolerance
        ):
            raise ValueError(
                f"Box outside image: {path.name}"
            )

    return len(lines)


def contact_sheet(items, destination):
    columns = 4
    cell_w = 360
    cell_h = 225
    rows = math.ceil(len(items) / columns)

    sheet = np.full(
        (rows * cell_h, columns * cell_w, 3),
        235,
        dtype=np.uint8,
    )

    for index, (split, path) in enumerate(items):
        image = cv2.imread(str(path))

        thumbnail = cv2.resize(
            image, (cell_w, cell_h - 30)
        )

        row = index // columns
        col = index % columns

        x = col * cell_w
        y = row * cell_h

        sheet[
            y + 30:y + cell_h,
            x:x + cell_w
        ] = thumbnail

        caption = f"{split}: {path.stem}"

        cv2.putText(
            sheet,
            caption,
            (x + 5, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    if not cv2.imwrite(
        str(destination), sheet
    ):
        raise RuntimeError(
            "Could not save contact sheet."
        )


def main():

    print("=" * 55)
    print("CREST CONFERENCE - PALLET DATASET FINALIZATION")
    print("=" * 55)

    for folder in (IMAGES, LABELS):
        if not folder.is_dir():
            raise FileNotFoundError(folder)

    for path in (PROGRESS, MANIFEST):
        if not path.is_file():
            raise FileNotFoundError(path)

    if OUTPUT.exists():
        raise RuntimeError(
            "Output dataset already exists. "
            "No files were overwritten."
        )

    images = image_files(IMAGES)

    if len(images) != 100:
        raise RuntimeError(
            f"Expected 100 images, got {len(images)}"
        )

    records = load_csv(MANIFEST)
    progress = load_csv(PROGRESS)

    if len(records) != 100 or len(progress) != 100:
        raise RuntimeError(
            "Manifest/progress does not have 100 rows."
        )

    record_by_name = {
        row["image"]: row for row in records
    }

    reviewed = {
        row["image"]
        for row in progress
        if row["status"] == "REVIEWED"
    }

    if len(reviewed) != 100:
        raise RuntimeError(
            f"Only {len(reviewed)}/100 reviewed."
        )

    image_names = {p.name for p in images}

    if (
        set(record_by_name) != image_names
        or reviewed != image_names
    ):
        raise RuntimeError(
            "Manifest, images, and progress do not match."
        )

    label_files = sorted(LABELS.glob("*.txt"))
    label_stems = {p.stem for p in label_files}

    if len(label_files) != 100:
        raise RuntimeError(
            "Expected exactly 100 reviewed labels."
        )

    if label_stems != {p.stem for p in images}:
        raise RuntimeError(
            "Image and label filenames do not match."
        )

    old_names = []
    new_names = []
    box_counts = {}
    image_hashes = set()

    print("Validating all images and annotations...")

    for image_path in images:

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(
                f"Cannot read image: {image_path.name}"
            )

        height, width = image.shape[:2]

        if (width, height) != (1920, 1080):
            raise RuntimeError(
                f"Unexpected image dimensions: "
                f"{image_path.name}: {width}x{height}"
            )

        digest = sha256(image_path)

        if digest in image_hashes:
            raise RuntimeError(
                "Duplicate image detected."
            )

        image_hashes.add(digest)

        label_path = LABELS / (
            image_path.stem + ".txt"
        )

        box_counts[image_path.name] = (
            check_labels(label_path)
        )

        source_type = record_by_name[
            image_path.name
        ]["source_type"]

        if source_type == "original_20":
            old_names.append(image_path.name)

        elif source_type == "new_80":
            new_names.append(image_path.name)

        else:
            raise RuntimeError(
                f"Unknown source type: {source_type}"
            )

    if len(old_names) != 20:
        raise RuntimeError(
            f"Expected 20 originals, got {len(old_names)}"
        )

    if len(new_names) != 80:
        raise RuntimeError(
            f"Expected 80 new images, got {len(new_names)}"
        )

    print("All 100 annotations passed format checks.")

    # Original 20 remain in training.
    # Validation and test use only new images.

    shuffled = sorted(new_names)
    random.Random(SEED).shuffle(shuffled)

    val_names = set(shuffled[:10])
    test_names = set(shuffled[10:20])

    train_names = (
        set(old_names)
        | set(shuffled[20:])
    )

    assignments = {}

    for name in train_names:
        assignments[name] = "train"

    for name in val_names:
        assignments[name] = "val"

    for name in test_names:
        assignments[name] = "test"

    if (
        len(train_names) != 80
        or len(val_names) != 10
        or len(test_names) != 10
        or len(assignments) != 100
    ):
        raise RuntimeError(
            "Dataset split is incorrect."
        )

    # Create a new dataset.
    # All source files remain unchanged.

    for split in ("train", "val", "test"):
        (OUTPUT / "images" / split).mkdir(
            parents=True, exist_ok=False
        )

        (OUTPUT / "labels" / split).mkdir(
            parents=True, exist_ok=False
        )

    split_records = []
    preview_items = []

    for image_path in images:

        split = assignments[image_path.name]

        destination_image = (
            OUTPUT / "images" / split
            / image_path.name
        )

        source_label = (
            LABELS / (image_path.stem + ".txt")
        )

        destination_label = (
            OUTPUT / "labels" / split
            / source_label.name
        )

        shutil.copy2(
            image_path, destination_image
        )

        shutil.copy2(
            source_label, destination_label
        )

        split_records.append({
            "image": image_path.name,
            "split": split,
            "source_type": record_by_name[
                image_path.name
            ]["source_type"],
            "pallet_boxes": box_counts[
                image_path.name
            ],
            "sha256": sha256(image_path),
        })

        if split in ("val", "test"):
            preview_items.append(
                (split.upper(), image_path)
            )

    yaml_text = (
        f"path: {OUTPUT.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: pallet\n"
    )

    (OUTPUT / "data.yaml").write_text(
        yaml_text, encoding="utf-8"
    )

    with (
        OUTPUT / "split_manifest.csv"
    ).open(
        "w", newline="", encoding="utf-8"
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "image",
                "split",
                "source_type",
                "pallet_boxes",
                "sha256",
            ],
        )

        writer.writeheader()
        writer.writerows(split_records)

    preview_items.sort(
        key=lambda item: (item[0], item[1].name)
    )

    contact_sheet(
        preview_items,
        OUTPUT / "validation_test_overview.jpg",
    )

    print()
    print("=" * 55)
    print("DATASET PREPARED SUCCESSFULLY")
    print("=" * 55)

    for split in ("train", "val", "test"):

        selected = [
            row for row in split_records
            if row["split"] == split
        ]

        boxes = sum(
            row["pallet_boxes"] for row in selected
        )

        print(
            f"{split.upper():5s} "
            f"images={len(selected):3d} "
            f"pallet_boxes={boxes}"
        )

    print()
    print("TOTAL IMAGES: 100")
    print("ALL LABELS: VALID FORMAT")
    print("ORIGINAL 20: INCLUDED IN TRAIN")
    print("TEST IMAGES: NEW IMAGES ONLY")
    print("ORIGINAL FILES: UNCHANGED")
    print()
    print(f"DATASET: {OUTPUT}")
    print()
    print(
        "Review split similarity before training."
    )
    print("NO MODEL WAS TRAINED.")


if __name__ == "__main__":
    main()
