"""CREST Phase 58: evaluate the frozen 100-image YOLO26 pallet detector.

Evaluation ONLY: no model training, no camera, no RFID, no PLC calls.

Reads human annotations from Phase 57; validates EVERY capture and label;
creates isolated, copied test data so Ultralytics caches cannot alter Phase 57;
then runs Ultralytics detection validation on the frozen best.pt checkpoint.

The 61 frames are correlated captures from one day, NOT 61 independent
physical pallet passes. Do not use this test to tune the model or thresholds.
"""

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(r"C:\Do_Not_Delete_PLC\Conference\09_Independent_Blind_Test_Captures")
LABEL_ROOT = ROOT / "_57_Manual_Pallet_Test_Labels"
MODEL = Path(
    r"C:\Do_Not_Delete_PLC\Conference\00_Pallet_100_YOLO26_Training"
    r"\pallet_yolo26_100_v2\weights\best.pt"
)
OUT_BASE = ROOT / "_58_Frozen_100_Image_Detector_Test"
EXPECTED_SHA256 = "14c62c306c1f6182ba0f2713f4a992e15f27649acc0f5c9aa58acc6a440f8c30"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_labels(label_path, image_name):
    boxes = []
    for line_no, raw in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        fields = raw.split()
        if len(fields) != 5:
            raise ValueError(f"{image_name}: label line {line_no} needs 5 fields")
        try:
            category = int(fields[0])
            x, y, w, h = (float(v) for v in fields[1:])
        except ValueError as exc:
            raise ValueError(f"{image_name}: invalid label line {line_no}: {exc}") from exc
        if category != 0:
            raise ValueError(f"{image_name}: non-pallet class {category}")
        if not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f"{image_name}: nonfinite box on line {line_no}")
        if not (0 < x < 1 and 0 < y < 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{image_name}: invalid normalized box on line {line_no}")
        eps = 0.00001
        if (x - w / 2 < -eps or x + w / 2 > 1 + eps or
                y - h / 2 < -eps or y + h / 2 > 1 + eps):
            raise ValueError(f"{image_name}: bounding box exceeds image on line {line_no}")
        boxes.append((x, y, w, h))
    return boxes


def preflight(label_root):
    manifest = label_root / "ground_truth_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing Phase 57 manifest: {manifest}")
    with manifest.open(newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise ValueError("Phase 57 manifest is empty")

    images = label_root / "images" / "test"
    labels = label_root / "labels" / "test"
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError("Missing Phase 57 images/test or labels/test folder")

    issues = []
    checked = []
    seen = set()
    for record in records:
        status = record.get("status", "")
        file_text = record.get("evaluation_image", "")
        original = record.get("source_image", "")
        if status not in {"labeled", "empty"}:
            issues.append(f"Unreviewed/unusable capture: {original}, status={status}")
            continue
        if not file_text:
            issues.append(f"Missing evaluation image in manifest: {original}")
            continue
        image = Path(file_text)
        if not image.is_file():
            issues.append(f"Image missing: {image}")
            continue
        if image.parent.resolve() != images.resolve():
            issues.append(f"Image not in Phase 57 test folder: {image}")
            continue
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            issues.append(f"Not a supported image: {image}")
            continue
        if image.name in seen:
            issues.append(f"Duplicate manifest image: {image.name}")
            continue
        seen.add(image.name)
        if not Path(original).is_file():
            issues.append(f"Original capture missing: {original}")
            continue
        label = labels / f"{image.stem}.txt"
        if not label.is_file():
            issues.append(f"Ground-truth file missing: {label}")
            continue
        try:
            boxes = validate_labels(label, image.name)
            expected = int(record.get("pallet_boxes", "-1"))
            if len(boxes) != expected:
                issues.append(f"Label count mismatch: {image.name}: {len(boxes)} != {expected}")
            if status == "labeled" and not boxes:
                issues.append(f"Labeled image has no boxes: {image.name}")
            if status == "empty" and boxes:
                issues.append(f"Empty image has labeled boxes: {image.name}")
        except (ValueError, OSError) as exc:
            issues.append(str(exc))
            continue
        checked.append((record, image, label, len(boxes)))

    extra_images = [p.name for p in images.iterdir() if p.is_file() and p.name not in seen]
    extra_labels = [p.name for p in labels.glob("*.txt")
                    if p.stem not in {image.stem for _, image, _, _ in checked}]
    if extra_images:
        issues.append(f"Unexpected test images: {extra_images}")
    if extra_labels:
        issues.append(f"Unexpected test labels: {extra_labels}")
    if len(checked) != len(records):
        issues.append(f"Only {len(checked)}/{len(records)} captures are validated")
    if issues:
        raise ValueError("GROUND-TRUTH PREFLIGHT FAILED:\n - " + "\n - ".join(issues))
    return checked


def json_convert(value):
    if isinstance(value, dict):
        return {str(k): json_convert(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_convert(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (int, float, bool, str)) or value is None:
        return value
    return str(value)


def main():
    parser = argparse.ArgumentParser(description="Frozen YOLO26 detector independent-session test")
    parser.add_argument("--label-root", type=Path, default=LABEL_ROOT,
                        help="Phase 57 annotation folder (testing only)")
    parser.add_argument("--model", type=Path, default=MODEL, help="Frozen detector best.pt")
    parser.add_argument("--output-root", type=Path, default=OUT_BASE)
    parser.add_argument("--check-only", action="store_true", help="Verify labels; no YOLO evaluation")
    args = parser.parse_args()

    print("CREST PHASE 58 - FROZEN PALLET DETECTOR HELD-OUT EVALUATION")
    print("Read-only source data. No training, camera, PLC or RFID.")
    checked = preflight(args.label_root)
    n_boxes = sum(item[3] for item in checked)
    counts = Counter(item[3] for item in checked)
    print(f"GROUND TRUTH VERIFIED: {len(checked)} images, {n_boxes} pallet boxes")
    print(f"Pallets per image: {dict(sorted(counts.items()))}")
    print("IMPORTANT: repeated frames/physical pallets are correlated observations.")

    if args.check_only:
        print("CHECK-ONLY PASSED: evaluation not started.")
        return

    if not args.model.is_file():
        raise FileNotFoundError(f"Deployed detector not found: {args.model}")
    model_hash = sha256_file(args.model)
    print("MODEL SHA256: " + model_hash)
    if args.model.resolve() == MODEL.resolve() and model_hash != EXPECTED_SHA256:
        raise RuntimeError(
            "Frozen model SHA256 differs from Oct 8 report; STOP before evaluation."
            f"\nExpected: {EXPECTED_SHA256}\nActual:   {model_hash}"
        )

    from ultralytics import YOLO
    import torch

    run_dir = args.output_root / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    if run_dir.exists():
        raise FileExistsError(f"Output run exists: {run_dir}")
    staged = run_dir / "test_dataset"
    target_images = staged / "images" / "test"
    target_labels = staged / "labels" / "test"
    target_images.mkdir(parents=True, exist_ok=False)
    target_labels.mkdir(parents=True, exist_ok=False)

    print("Creating isolated evaluation copies; sources remain untouched...")
    inventory = []
    for record, image, label, boxes in checked:
        staged_image = target_images / image.name
        staged_label = target_labels / label.name
        shutil.copy2(image, staged_image)
        shutil.copy2(label, staged_label)
        inventory.append({
            "session": record["session"], "image": image.name,
            "box_count": boxes, "image_sha256": sha256_file(image),
            "label_sha256": sha256_file(label),
        })
    (run_dir / "frozen_test_manifest.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8")
    shutil.copy2(args.label_root / "ground_truth_manifest.csv", run_dir / "original_ground_truth_manifest.csv")

    # Ultralytics requires a train key in detection dataset YAML, even for val().
    # All three keys point only to the FROZEN test copy; NO train() is ever invoked.
    yaml_path = staged / "data_test_only.yaml"
    yaml_path.write_text(
        "# Phase 58 TEST ONLY. The required train key is a schema placeholder.\n"
        "# NEVER use this YAML with YOLO.train().\n"
        f"path: '{staged.as_posix()}'\n"
        "train: images/test\n"
        "val: images/test\n"
        "test: images/test\n"
        "names:\n  0: pallet\n",
        encoding="utf-8"
    )

    device = 0 if torch.cuda.is_available() else "cpu"
    config = {
        "purpose": "Frozen model evaluation on freshly captured manually labeled test frames",
        "model": str(args.model), "model_sha256": model_hash,
        "source_dataset": str(args.label_root),
        "evaluation_dataset": str(staged), "images": len(checked),
        "ground_truth_boxes": n_boxes, "device": device,
        "imgsz": 960, "conf": 0.001, "iou": 0.70,
        "max_det": 300, "batch": 1, "split": "test",
        "augmentation": False, "training": False,
        "independence_note": "Multiple same-day captures of same physical pallets; correlated frames",
    }
    (run_dir / "evaluation_configuration.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("Frozen evaluation data: " + str(staged))
    print(f"Testing model with imgsz=960, conf=0.001, IoU=0.70, batch=1, device={device}")
    print("This is detection validation, NOT live three-model accuracy.")
    print("Beginning YOLO.val()...")
    detector = YOLO(str(args.model))
    metrics = detector.val(
        data=str(yaml_path), split="test", imgsz=960, batch=1,
        conf=0.001, iou=0.70, max_det=300, device=device,
        workers=0, augment=False, plots=True, verbose=True,
        project=str(run_dir), name="ultralytics_test", exist_ok=False,
    )
    box = metrics.box
    precision = float(box.mp) * 100
    recall = float(box.mr) * 100
    map50 = float(box.map50) * 100
    map5095 = float(box.map) * 100
    results = {
        "total_images": len(checked), "total_ground_truth_boxes": n_boxes,
        "precision_percent": precision, "recall_percent": recall,
        "mAP50_percent": map50, "mAP50_95_percent": map5095,
        "model_sha256": model_hash, "raw_metrics": json_convert(metrics.results_dict),
        "comparison_warning": "Historical detector test used DIFFERENT images; do not treat as paired improvement",
        "independence_warning": "61 frames do not mean 61 independent physical pallet passes",
    }
    (run_dir / "test_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = (
        "CREST PHASE 58 - FROZEN DETECTOR NEW SESSION TEST\n"
        "================================================\n"
        f"Images evaluated:        {len(checked)}\n"
        f"Ground truth boxes:      {n_boxes}\n"
        f"Precision:               {precision:.2f}%\n"
        f"Recall:                  {recall:.2f}%\n"
        f"mAP50:                   {map50:.2f}%\n"
        f"mAP50-95:                {map5095:.2f}%\n\n"
        "Historical 10-frame results (different test):\n"
        "Precision 77.29%, recall 65.71%, mAP50 77.01%, mAP50-95 72.54%.\n"
        "These frames were captured in one day; several may share physical passes.\n"
        "NO thresholds or model weights were tuned on this held-out set.\n"
        "NOT an end-to-end EMPTY/ACCEPTED/REJECTED system evaluation.\n"
        "Run output: " + str(run_dir) + "\n"
    )
    (run_dir / "test_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nPHASE 58 STOPPED: {exc}", file=sys.stderr)
        raise
