
from pathlib import Path
import csv
import hashlib
import shutil

import cv2
from ultralytics import YOLO


# ==================================================
# PATHS
# ==================================================

BASE = Path(r"C:\Do_Not_Delete_PLC\Conference")

SOURCE = BASE / "00_Pallet_Detector_100_Clean"

ORIGINAL_20 = (
    BASE / "00_Current_Working_Pallet_Model_20_Images"
)

OLD_LABELS = (
    BASE
    / "02_YOLO_Pallet_Detection"
    / "DATASET"
    / "yolo_dataset"
    / "labels"
)

MODEL = (
    Path(r"C:\Do_Not_Delete_PLC\CREST_YOLO26")
    / "07_Final_Empty_Accepted_Rejected_System"
    / "MODELS"
    / "pallet_yolo26n_best.pt"
)

OUTPUT = BASE / "00_Pallet_100_Annotation_Work"

IMAGE_OUT = OUTPUT / "images"
LABEL_OUT = OUTPUT / "labels_draft"
PREVIEW_OUT = OUTPUT / "previews"

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ==================================================
# HELPERS
# ==================================================

def image_files(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
    )


def file_hash(path):
    digest = hashlib.sha256()

    with open(path, "rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_yolo_labels(path):
    boxes = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        values = line.split()

        if len(values) != 5 or values[0] != "0":
            raise ValueError(
                f"Invalid pallet annotation: {path}"
            )

        coordinates = [
            float(value) for value in values[1:]
        ]

        cx, cy, bw, bh = coordinates

        if not (
            0 <= cx <= 1
            and 0 <= cy <= 1
            and 0 < bw <= 1
            and 0 < bh <= 1
        ):
            raise ValueError(
                f"Invalid box coordinates: {path}"
            )

        boxes.append(coordinates)

    if not boxes:
        raise ValueError(
            f"No pallet boxes in: {path}"
        )

    return boxes


def predict_pallets(model, image_path, width, height):

    result = model.predict(
        source=str(image_path),
        conf=0.10,
        iou=0.50,
        imgsz=960,
        max_det=30,
        verbose=False,
    )[0]

    boxes = []

    if result.boxes is None:
        return boxes

    for detection in result.boxes:

        class_id = int(detection.cls.item())

        if class_id != 0:
            continue

        x1, y1, x2, y2 = (
            detection.xyxy[0].cpu().tolist()
        )

        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            continue

        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height

        boxes.append([cx, cy, bw, bh])

    return boxes


def save_labels(path, boxes):

    with open(path, "w", encoding="utf-8") as file:

        for cx, cy, bw, bh in boxes:
            file.write(
                f"0 {cx:.6f} {cy:.6f} "
                f"{bw:.6f} {bh:.6f}\n"
            )


def save_preview(image, boxes, path, status):

    height, width = image.shape[:2]
    preview = image.copy()

    for cx, cy, bw, bh in boxes:

        x1 = int((cx - bw / 2) * width)
        y1 = int((cy - bh / 2) * height)

        x2 = int((cx + bw / 2) * width)
        y2 = int((cy + bh / 2) * height)

        cv2.rectangle(
            preview,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3,
        )

    cv2.rectangle(
        preview,
        (0, 0),
        (650, 75),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        preview,
        f"{status} | Pallet boxes: {len(boxes)}",
        (15, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )

    if not cv2.imwrite(str(path), preview):
        raise RuntimeError(
            f"Could not save preview: {path}"
        )


# ==================================================
# MAIN
# ==================================================

def main():

    for folder in (SOURCE, ORIGINAL_20, OLD_LABELS):
        if not folder.is_dir():
            raise FileNotFoundError(folder)

    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)

    if OUTPUT.exists():
        raise RuntimeError(
            "Annotation workspace already exists. "
            "Existing work will not be overwritten."
        )

    source_images = image_files(SOURCE)
    original_images = image_files(ORIGINAL_20)

    if len(source_images) != 100:
        raise RuntimeError(
            f"Expected 100 images, got {len(source_images)}"
        )

    if len(original_images) != 20:
        raise RuntimeError(
            "Expected 20 original images."
        )

    original_by_hash = {
        file_hash(path): path
        for path in original_images
    }

    source_hashes = {
        file_hash(path) for path in source_images
    }

    if len(source_hashes) != 100:
        raise RuntimeError(
            "Duplicate source images detected."
        )

    if not set(original_by_hash).issubset(source_hashes):
        raise RuntimeError(
            "Some original 20 images are missing."
        )

    label_lookup = {}

    for path in OLD_LABELS.rglob("*.txt"):
        if path.stem in label_lookup:
            raise RuntimeError(
                f"Duplicate label name: {path.stem}"
            )
        label_lookup[path.stem] = path

    for old_image in original_images:

        old_label = label_lookup.get(old_image.stem)

        if old_label is None:
            raise RuntimeError(
                f"Missing old label: {old_image.name}"
            )

        read_yolo_labels(old_label)

    print("Loading working YOLO26 pallet detector...")
    detector = YOLO(str(MODEL))

    IMAGE_OUT.mkdir(parents=True)
    LABEL_OUT.mkdir()
    PREVIEW_OUT.mkdir()

    manifest = []
    reused_count = 0
    auto_count = 0
    zero_count = 0

    for index, source_image in enumerate(
        source_images, start=1
    ):

        name = (
            f"pallet_{index:03d}"
            f"{source_image.suffix.lower()}"
        )

        destination = IMAGE_OUT / name

        shutil.copy2(source_image, destination)

        image = cv2.imread(str(destination))

        if image is None:
            raise RuntimeError(
                f"Cannot read: {source_image}"
            )

        height, width = image.shape[:2]

        digest = file_hash(source_image)

        if digest in original_by_hash:

            original = original_by_hash[digest]
            old_label = label_lookup[original.stem]

            boxes = read_yolo_labels(old_label)

            status = "ORIGINAL LABEL - REVIEW"
            source_type = "original_20"

            reused_count += 1

        else:

            boxes = predict_pallets(
                detector,
                destination,
                width,
                height,
            )

            status = "AUTO DRAFT - REVIEW"
            source_type = "new_80"

            auto_count += 1

            if not boxes:
                zero_count += 1

        label_path = LABEL_OUT / (
            f"{Path(name).stem}.txt"
        )

        save_labels(label_path, boxes)

        preview_path = PREVIEW_OUT / (
            f"{Path(name).stem}_preview.jpg"
        )

        save_preview(
            image,
            boxes,
            preview_path,
            status,
        )

        manifest.append({
            "image": name,
            "original_file": str(source_image),
            "source_type": source_type,
            "boxes": len(boxes),
            "review_status": "PENDING",
        })

        print(
            f"[{index:03d}/100] "
            f"{source_type} | "
            f"{len(boxes)} boxes"
        )

    manifest_path = OUTPUT / "annotation_manifest.csv"

    with open(
        manifest_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image",
                "original_file",
                "source_type",
                "boxes",
                "review_status",
            ],
        )

        writer.writeheader()
        writer.writerows(manifest)

    print()
    print("=" * 50)
    print("PALLET ANNOTATION DRAFT COMPLETE")
    print("=" * 50)
    print(f"Total images:       {len(manifest)}")
    print(f"Existing labels:    {reused_count}")
    print(f"Auto draft labels:  {auto_count}")
    print(f"Zero-box images:    {zero_count}")
    print(f"Output: {OUTPUT}")
    print()
    print("IMPORTANT: ALL BOXES REQUIRE REVIEW.")
    print("NO MODEL WAS TRAINED.")


if __name__ == "__main__":
    main()
