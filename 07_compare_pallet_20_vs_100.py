
from pathlib import Path
import csv

import torch
from ultralytics import YOLO


BASE = Path(r"C:\Do_Not_Delete_PLC\Conference")

DATASET = BASE / "00_Pallet_100_YOLO26_Dataset_v2"
DATA_YAML = DATASET / "data.yaml"

OLD_MODEL = (
    Path(r"C:\Do_Not_Delete_PLC\CREST_YOLO26")
    / "07_Final_Empty_Accepted_Rejected_System"
    / "MODELS"
    / "pallet_yolo26n_best.pt"
)

NEW_MODEL = (
    BASE
    / "00_Pallet_100_YOLO26_Training"
    / "pallet_yolo26_100_v2"
    / "weights"
    / "best.pt"
)

OUTPUT = BASE / "00_Pallet_20_vs_100_Comparison"


def verify():
    for path in (DATA_YAML, OLD_MODEL, NEW_MODEL):
        if not path.is_file():
            raise FileNotFoundError(path)

    if OUTPUT.exists():
        raise RuntimeError(
            "Comparison output already exists. "
            "Existing results will not be overwritten."
        )

    image_dir = DATASET / "images" / "test"
    label_dir = DATASET / "labels" / "test"

    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in (
            ".jpg", ".jpeg", ".png", ".bmp"
        )
    )

    labels = sorted(label_dir.glob("*.txt"))

    if len(images) != 10 or len(labels) != 10:
        raise RuntimeError(
            "Expected 10 test images and labels."
        )

    if {p.stem for p in images} != {
        p.stem for p in labels
    }:
        raise RuntimeError(
            "Test image/label mismatch."
        )

    manifest_path = DATASET / "split_manifest.csv"

    with manifest_path.open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))

    test_rows = [
        row for row in rows
        if row["split"] == "test"
    ]

    if (
        len(test_rows) != 10
        or any(
            row["source_type"] != "new_80"
            for row in test_rows
        )
    ):
        raise RuntimeError(
            "Unexpected test split."
        )

    print("Test set verified: 10 new images")


def evaluate(model_path, name):
    print()
    print("=" * 55)
    print(name)
    print("=" * 55)

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=960,
        batch=4,
        device=0,
        workers=0,
        conf=0.001,
        iou=0.70,
        max_det=300,
        project=str(OUTPUT),
        name=name,
        exist_ok=False,
        plots=True,
        verbose=False,
    )

    values = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    for key, value in values.items():
        print(f"{key}: {value:.4f}")

    return values


def main():
    print("CREST - PALLET MODEL COMPARISON")
    print("Original model remains unchanged.")
    print("No training will be performed.")

    verify()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable.")

    print("GPU:", torch.cuda.get_device_name(0))

    old = evaluate(
        OLD_MODEL,
        "original_20_image_model",
    )

    new = evaluate(
        NEW_MODEL,
        "conference_100_image_model",
    )

    report = [
        "CREST PALLET DETECTION COMPARISON",
        "=" * 55,
        "Dataset: 10 common held-out test images",
        "Evaluation image size: 960",
        "",
        "Metric          Original20    New100    Difference",
    ]

    for key in (
        "precision",
        "recall",
        "map50",
        "map50_95",
    ):
        difference = new[key] - old[key]

        report.append(
            f"{key:15s} "
            f"{old[key]:9.4f} "
            f"{new[key]:9.4f} "
            f"{difference:+9.4f}"
        )

    report.extend([
        "",
        "This is a model-level test comparison.",
        "Visual annotation and split audits are",
        "required before reporting final conclusions.",
        "",
        f"Original model: {OLD_MODEL}",
        f"Conference model: {NEW_MODEL}",
        "",
        "Neither model was modified.",
    ])

    report_path = OUTPUT / "comparison_results.txt"

    report_path.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print()
    print("\n".join(report))
    print()
    print("Saved:", report_path)


if __name__ == "__main__":
    main()
