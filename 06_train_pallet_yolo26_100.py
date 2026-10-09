
from pathlib import Path

import torch
from ultralytics import YOLO


# ==========================================
# CREST CONFERENCE - 100 IMAGE TRAINING
# ==========================================

BASE = Path(
    r"C:\Do_Not_Delete_PLC\Conference"
)

DATASET = (
    BASE / "00_Pallet_100_YOLO26_Dataset_v2"
)

DATA_YAML = DATASET / "data.yaml"

WORKING_MODEL = (
    Path(r"C:\Do_Not_Delete_PLC\CREST_YOLO26")
    / "07_Final_Empty_Accepted_Rejected_System"
    / "MODELS"
    / "pallet_yolo26n_best.pt"
)

OUTPUT = (
    BASE / "00_Pallet_100_YOLO26_Training"
)

RUN_NAME = "pallet_yolo26_100_v2"


# ==========================================
# DATASET VERIFICATION
# ==========================================

def verify_dataset():

    if not DATA_YAML.is_file():
        raise FileNotFoundError(DATA_YAML)

    if not WORKING_MODEL.is_file():
        raise FileNotFoundError(WORKING_MODEL)

    expected = {
        "train": 80,
        "val": 10,
        "test": 10,
    }

    expected_total_boxes = 972
    total_boxes = 0

    for split, expected_count in expected.items():

        image_dir = (
            DATASET / "images" / split
        )

        label_dir = (
            DATASET / "labels" / split
        )

        images = [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in (
                ".jpg", ".jpeg", ".png", ".bmp"
            )
        ]

        labels = list(
            label_dir.glob("*.txt")
        )

        if len(images) != expected_count:
            raise RuntimeError(
                f"{split}: incorrect image count"
            )

        if len(labels) != expected_count:
            raise RuntimeError(
                f"{split}: incorrect label count"
            )

        image_names = {
            p.stem for p in images
        }

        label_names = {
            p.stem for p in labels
        }

        if image_names != label_names:
            raise RuntimeError(
                f"{split}: image/label mismatch"
            )

        boxes = 0

        for label in labels:

            lines = [
                line for line in
                label.read_text().splitlines()
                if line.strip()
            ]

            boxes += len(lines)

        total_boxes += boxes

        print(
            f"{split.upper()}: "
            f"{len(images)} images, "
            f"{boxes} boxes"
        )

    if total_boxes != expected_total_boxes:
        raise RuntimeError(
            f"Expected {expected_total_boxes} "
            f"boxes, found {total_boxes}"
        )

    print()
    print("Dataset structure verified.")
    print("Total images: 100")
    print(f"Total boxes: {total_boxes}")


# ==========================================
# MODEL TRAINING
# ==========================================

def main():

    print("=" * 55)
    print("CREST CONFERENCE YOLO26 PALLET TRAINING")
    print("=" * 55)

    verify_dataset()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. "
            "Activate the GPU environment."
        )

    print()
    print(
        "GPU:",
        torch.cuda.get_device_name(0),
    )

    print()
    print("Starting from current working model.")
    print("Original model will not be modified.")

    model = YOLO(str(WORKING_MODEL))

    results = model.train(
        data=str(DATA_YAML),

        epochs=80,
        patience=20,

        imgsz=960,
        batch=4,

        device=0,
        workers=0,

        optimizer="AdamW",
        lr0=0.0002,

        seed=42,
        deterministic=True,

        project=str(OUTPUT),
        name=RUN_NAME,
        exist_ok=False,

        save=True,
        plots=True,
        verbose=True,
    )

    best_model = (
        Path(results.save_dir)
        / "weights"
        / "best.pt"
    )

    print()
    print("=" * 55)
    print("TRAINING COMPLETE")
    print("=" * 55)

    print("Training output:")
    print(results.save_dir)

    print()
    print("Best trained model:")
    print(best_model)

    print()
    print("Original working model unchanged.")
    print("Held-out test dataset not evaluated.")


if __name__ == "__main__":
    main()
