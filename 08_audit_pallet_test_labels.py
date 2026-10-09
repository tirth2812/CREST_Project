
from pathlib import Path

import cv2


BASE = Path(r"C:\Do_Not_Delete_PLC\Conference")

DATASET = BASE / "00_Pallet_100_YOLO26_Dataset_v2"

IMAGES = DATASET / "images" / "test"
LABELS = DATASET / "labels" / "test"

OUTPUT = BASE / "00_Pallet_Test_Annotation_Audit"


def main():

    images = sorted(
        p for p in IMAGES.iterdir()
        if p.suffix.lower() in (
            ".jpg", ".jpeg", ".png", ".bmp"
        )
    )

    if len(images) != 10:
        raise RuntimeError(
            f"Expected 10 test images, found {len(images)}"
        )

    if OUTPUT.exists():
        raise RuntimeError(
            "Audit folder already exists. "
            "Nothing was overwritten."
        )

    # Verify all labels before creating output.
    for path in images:
        label = LABELS / f"{path.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)

    OUTPUT.mkdir()

    total_boxes = 0

    for image_path in images:

        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(image_path)

        height, width = image.shape[:2]

        label_path = LABELS / (
            image_path.stem + ".txt"
        )

        lines = [
            line.strip()
            for line in label_path.read_text().splitlines()
            if line.strip()
        ]

        for number, line in enumerate(lines, start=1):

            parts = line.split()

            if len(parts) != 5 or parts[0] != "0":
                raise RuntimeError(
                    f"Invalid label: {label_path}"
                )

            cx, cy, bw, bh = map(float, parts[1:])

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
                0.9,
                (0, 0, 255),
                3,
            )

        total_boxes += len(lines)

        output_path = (
            OUTPUT / f"{image_path.stem}_audit.jpg"
        )

        if not cv2.imwrite(
            str(output_path), image
        ):
            raise RuntimeError(
                f"Could not save {output_path}"
            )

        print(
            f"{image_path.stem}: "
            f"{len(lines)} boxes"
        )

    print()
    print("TEST IMAGES:", len(images))
    print("TOTAL TEST BOXES:", total_boxes)
    print("Audit folder:", OUTPUT)
    print("Original dataset unchanged.")


if __name__ == "__main__":
    main()
