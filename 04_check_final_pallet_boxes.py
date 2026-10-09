
from pathlib import Path
import cv2

BASE = Path(
    r"C:\Do_Not_Delete_PLC\Conference"
    r"\00_Pallet_100_Annotation_Work"
)

IMAGES = BASE / "images"
LABELS = BASE / "labels_reviewed"
OUTPUT = BASE / "final_review_previews"

IMAGE_NAMES = [
    "pallet_026",
    "pallet_082",
    "pallet_091",
]

if OUTPUT.exists():
    raise RuntimeError(
        "Preview folder already exists. "
        "No existing files were changed."
    )

OUTPUT.mkdir()

for name in IMAGE_NAMES:

    matches = [
        p for p in IMAGES.glob(name + ".*")
        if p.suffix.lower() in (
            ".jpg", ".jpeg", ".png", ".bmp"
        )
    ]

    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot find image: {name}"
        )

    image = cv2.imread(str(matches[0]))

    if image is None:
        raise RuntimeError(name)

    height, width = image.shape[:2]

    label_path = LABELS / (name + ".txt")

    lines = [
        line.strip()
        for line in label_path.read_text().splitlines()
        if line.strip()
    ]

    for index, line in enumerate(lines, 1):

        parts = line.split()
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
            str(index),
            (x1 + 5, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            3,
        )

    output_path = OUTPUT / (name + "_FINAL.jpg")

    cv2.imwrite(str(output_path), image)

    print(f"{name}: {len(lines)} reviewed boxes")

print()
print("Final reviewed previews created.")
print("Original images and labels unchanged.")
