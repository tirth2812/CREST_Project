
from pathlib import Path
import csv
import shutil

import cv2


BASE = Path(r"C:\Do_Not_Delete_PLC\Conference")

WORK = BASE / "00_Pallet_100_Annotation_Work"

OLD_DATASET = BASE / "00_Pallet_100_YOLO26_Dataset"

NEW_DATASET = BASE / "00_Pallet_100_YOLO26_Dataset_v2"

OLD_REVIEWED = WORK / "labels_reviewed"

NEW_REVIEWED = WORK / "labels_corrected_v2"

CORRECTIONS = {
    "pallet_026": 11,
    "pallet_082": 13,
}


def get_lines(path):
    return [
        line.strip()
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


def write_lines(path, lines):
    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():

    if not OLD_DATASET.is_dir():
        raise FileNotFoundError(OLD_DATASET)

    if not OLD_REVIEWED.is_dir():
        raise FileNotFoundError(OLD_REVIEWED)

    if NEW_DATASET.exists():
        raise RuntimeError(
            "Corrected dataset already exists. "
            "Nothing was changed."
        )

    if NEW_REVIEWED.exists():
        raise RuntimeError(
            "Corrected labels already exist. "
            "Nothing was changed."
        )

    locations = {}

    # Verify everything before copying.
    for name, box_number in CORRECTIONS.items():

        reviewed_path = (
            OLD_REVIEWED / f"{name}.txt"
        )

        matches = list(
            (OLD_DATASET / "labels").glob(
                f"*/{name}.txt"
            )
        )

        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one dataset label: {name}"
            )

        dataset_path = matches[0]

        reviewed_lines = get_lines(reviewed_path)
        dataset_lines = get_lines(dataset_path)

        if reviewed_lines != dataset_lines:
            raise RuntimeError(
                f"Reviewed/dataset labels differ: {name}"
            )

        if len(reviewed_lines) != 14:
            raise RuntimeError(
                f"Expected 14 boxes for {name}, "
                f"found {len(reviewed_lines)}"
            )

        if not 1 <= box_number <= len(reviewed_lines):
            raise RuntimeError(
                f"Invalid box number for {name}"
            )

        locations[name] = dataset_path

    print("Original annotations verified.")

    # Make protected copies.
    shutil.copytree(
        OLD_REVIEWED,
        NEW_REVIEWED,
    )

    shutil.copytree(
        OLD_DATASET,
        NEW_DATASET,
    )

    # Remove the two identified boxes.
    for name, box_number in CORRECTIONS.items():

        corrected_path = (
            NEW_REVIEWED / f"{name}.txt"
        )

        lines = get_lines(corrected_path)

        lines.pop(box_number - 1)

        write_lines(corrected_path, lines)

        relative_path = locations[name].relative_to(
            OLD_DATASET
        )

        dataset_label = (
            NEW_DATASET / relative_path
        )

        write_lines(dataset_label, lines)

        print(
            f"{name}: removed box {box_number}, "
            f"remaining boxes = {len(lines)}"
        )

    # Update the YOLO dataset configuration.
    yaml_text = (
        f"path: {NEW_DATASET.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: pallet\n"
    )

    (NEW_DATASET / "data.yaml").write_text(
        yaml_text,
        encoding="utf-8",
    )

    # Update annotation counts in the manifest.
    manifest_path = (
        NEW_DATASET / "split_manifest.csv"
    )

    with manifest_path.open(
        newline="",
        encoding="utf-8",
    ) as stream:

        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        stem = Path(row["image"]).stem

        if stem in CORRECTIONS:
            row["pallet_boxes"] = "13"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    # Generate corrected visual previews.
    preview_folder = (
        NEW_DATASET / "corrected_previews"
    )

    preview_folder.mkdir()

    for name in CORRECTIONS:

        images = list(
            (NEW_DATASET / "images").glob(
                f"*/{name}.*"
            )
        )

        if len(images) != 1:
            raise RuntimeError(
                f"Image not found: {name}"
            )

        image = cv2.imread(str(images[0]))

        if image is None:
            raise RuntimeError(
                f"Cannot open: {name}"
            )

        height, width = image.shape[:2]

        lines = get_lines(
            NEW_REVIEWED / f"{name}.txt"
        )

        for number, line in enumerate(lines, 1):

            parts = line.split()

            cx, cy, bw, bh = map(
                float, parts[1:]
            )

            x1 = int((cx - bw / 2) * width)
            y1 = int((cy - bh / 2) * height)

            x2 = int((cx + bw / 2) * width)
            y2 = int((cy + bh / 2) * height)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3,
            )

            cv2.putText(
                image,
                str(number),
                (x1 + 5, y1 + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        output = (
            preview_folder / f"{name}_corrected.jpg"
        )

        cv2.imwrite(str(output), image)

    # Verify the new dataset counts.
    total_images = 0
    total_boxes = 0

    for split in ("train", "val", "test"):

        image_folder = (
            NEW_DATASET / "images" / split
        )

        label_folder = (
            NEW_DATASET / "labels" / split
        )

        image_count = len(
            list(image_folder.iterdir())
        )

        label_files = list(
            label_folder.glob("*.txt")
        )

        box_count = sum(
            len(get_lines(path))
            for path in label_files
        )

        if image_count != len(label_files):
            raise RuntimeError(
                f"Image/label mismatch: {split}"
            )

        total_images += image_count
        total_boxes += box_count

        print(
            f"{split.upper()}: "
            f"{image_count} images, "
            f"{box_count} boxes"
        )

    if total_images != 100:
        raise RuntimeError(
            f"Expected 100 images, got {total_images}"
        )

    if total_boxes != 972:
        raise RuntimeError(
            f"Expected 972 boxes, got {total_boxes}"
        )

    print()
    print("CORRECTIONS COMPLETE")
    print("TOTAL IMAGES: 100")
    print("TOTAL BOXES: 972")
    print("ORIGINAL DATASET: UNCHANGED")
    print("WORKING MODEL: UNCHANGED")
    print(f"CORRECTED DATASET: {NEW_DATASET}")


if __name__ == "__main__":
    main()
