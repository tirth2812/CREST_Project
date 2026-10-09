
from pathlib import Path
import csv

import cv2


BASE = Path(r"C:\Do_Not_Delete_PLC\Conference")

DATASET = BASE / "00_Pallet_100_YOLO26_Dataset_v2"

OUTPUT = BASE / "00_Pallet_Label_Duplicate_Audit"

PREVIEWS = OUTPUT / "test_label_previews"


def read_boxes(path, width, height):
    boxes = []

    lines = path.read_text(
        encoding="utf-8"
    ).splitlines()

    for line in lines:
        if not line.strip():
            continue

        values = line.split()
        cx, cy, bw, bh = map(float, values[1:])

        boxes.append((
            (cx - bw / 2) * width,
            (cy - bh / 2) * height,
            (cx + bw / 2) * width,
            (cy + bh / 2) * height,
        ))

    return boxes


def overlap(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0, ix2 - ix1)
        * max(0, iy2 - iy1)
    )

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - intersection

    iou = intersection / union if union > 0 else 0

    smaller = min(area_a, area_b)

    coverage = (
        intersection / smaller
        if smaller > 0 else 0
    )

    return iou, coverage


def audit_boxes(boxes):
    pairs = []

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):

            iou, coverage = overlap(
                boxes[i], boxes[j]
            )

            if iou >= 0.55 or coverage >= 0.85:
                pairs.append((i + 1, j + 1))

    return pairs


def draw_preview(image, boxes, pairs, output):
    flagged = set()

    for a, b in pairs:
        flagged.add(a)
        flagged.add(b)

    for index, box in enumerate(boxes, 1):
        x1, y1, x2, y2 = map(int, box)

        color = (
            (0, 140, 255)
            if index in flagged
            else (0, 255, 0)
        )

        cv2.rectangle(
            image, (x1, y1), (x2, y2),
            color, 2
        )

        cv2.putText(
            image,
            str(index),
            (x1 + 3, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

    cv2.imwrite(str(output), image)


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(DATASET)

    if OUTPUT.exists():
        raise RuntimeError(
            "Audit output already exists. "
            "Nothing was overwritten."
        )

    rows = []
    total_boxes = 0
    suspicious_images = 0

    print("CREST PALLET ANNOTATION DUPLICATE AUDIT")
    print("=" * 55)

    for split in ("train", "val", "test"):
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split

        images = sorted(
            p for p in image_dir.iterdir()
            if p.suffix.lower() in (
                ".jpg", ".jpeg", ".png", ".bmp"
            )
        )

        for image_path in images:
            image = cv2.imread(str(image_path))

            if image is None:
                raise RuntimeError(image_path)

            height, width = image.shape[:2]

            label_path = (
                label_dir / f"{image_path.stem}.txt"
            )

            if not label_path.is_file():
                raise FileNotFoundError(label_path)

            boxes = read_boxes(
                label_path, width, height
            )

            pairs = audit_boxes(boxes)

            count = len(boxes)
            total_boxes += count

            if pairs or count > 6:
                suspicious_images += 1

            pair_text = "; ".join(
                f"{a}-{b}" for a, b in pairs
            )

            rows.append({
                "split": split,
                "image": image_path.name,
                "boxes": count,
                "overlap_pairs": len(pairs),
                "box_numbers": pair_text,
            })

    OUTPUT.mkdir()
    PREVIEWS.mkdir()

    with (
        OUTPUT / "duplicate_audit.csv"
    ).open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "split",
                "image",
                "boxes",
                "overlap_pairs",
                "box_numbers",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        if row["split"] != "test":
            continue

        image_path = (
            DATASET / "images" / "test" / row["image"]
        )

        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]

        boxes = read_boxes(
            DATASET / "labels" / "test"
            / f"{image_path.stem}.txt",
            width,
            height,
        )

        pairs = audit_boxes(boxes)

        draw_preview(
            image,
            boxes,
            pairs,
            PREVIEWS / f"{image_path.stem}_audit.jpg",
        )

        print(
            f"{image_path.stem}: "
            f"{len(boxes)} boxes | "
            f"{len(pairs)} overlapping pairs"
        )

    print()
    print("=" * 55)
    print("AUDIT COMPLETE")
    print("=" * 55)
    print("Total annotation boxes:", total_boxes)
    print("Images needing review:", suspicious_images)
    print("Audit report:", OUTPUT / "duplicate_audit.csv")
    print("Test previews:", PREVIEWS)
    print()
    print("No annotations were changed.")
    print("No model was trained.")


if __name__ == "__main__":
    main()
