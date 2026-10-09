from pathlib import Path
from collections import Counter
from datetime import datetime
import csv
import hashlib
import platform
import re
import sys

import cv2
import torch
import ultralytics
from ultralytics import YOLO


ROOT = Path(r"C:\Do_Not_Delete_PLC\Conference")
FINAL = ROOT / "07_Final_Empty_Accepted_Rejected_System"
PALLET_DATA = ROOT / "00_Pallet_100_YOLO26_Dataset_v2"
PALLET_RUN = (
    ROOT / "00_Pallet_100_YOLO26_Training"
    / "pallet_yolo26_100_v2"
)
PALLET_MODEL = PALLET_RUN / "weights" / "best.pt"
OBJECT_DATA = (
    ROOT / "03_Empty_Object_Classifier"
    / "DATASET" / "prepared_yolo26"
)
OBJECT_MODEL = (
    ROOT / "03_Empty_Object_Classifier"
    / "MODEL" / "empty_object_yolo26n_cls_best.pt"
)
SYMBOL_DATA = ROOT / "00_Current_Working_X_NO_X_Dataset"
SYMBOL_MODEL = (
    FINAL / "MODELS"
    / "x_no_x_yolo26n_medium_focus_final_hard_best.pt"
)
OLD_MODEL = (
    FINAL / "MODELS" / "pallet_yolo26n_best.pt"
)
COMPARISON = (
    ROOT / "00_Pallet_20_vs_100_Comparison"
    / "comparison_results.txt"
)
HARD_RESULTS = (
    FINAL / "RESULTS" / "Final_Model_Training"
    / "final_hard_finetune_results.txt"
)
HARD_CONFIG = (
    FINAL / "RESULTS" / "Final_Model_Training"
    / "final_hard_finetune_config.json"
)
CONFERENCE_LIVE = (
    FINAL / "CODE" / "40_conference_100_pallet_live.py"
)
OUT = ROOT / "00_Conference_Paper_Model_Report"
EXT = {".jpg", ".jpeg", ".png", ".bmp"}
DEVICE = 0 if torch.cuda.is_available() else "cpu"
LINES = []


def say(*parts):
    message = " ".join(str(part) for part in parts)
    print(message, flush=True)
    LINES.append(message)


def section(name):
    say()
    say("=" * 72)
    say(name)
    say("=" * 72)


def files_in(folder):
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in EXT
    )


def file_hash(path):
    hash_obj = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def describe_model(title, path):
    say(title)
    say("  Path:", path)
    if not path.is_file():
        say("  STATUS: MODEL FILE MISSING")
        return None
    say("  Size (MB):", f"{path.stat().st_size / (1024 * 1024):.2f}")
    say("  SHA256:", file_hash(path))
    try:
        model = YOLO(str(path))
        say("  Task:", model.task)
        say("  Classes:", model.names)
        network = model.model
        say("  Parameters:", f"{sum(p.numel() for p in network.parameters()):,}")
        say("  Trainable parameters:",
            f"{sum(p.numel() for p in network.parameters() if p.requires_grad):,}")
        say("  Model input size (checkpoint metadata):",
            getattr(model, "overrides", {}).get("imgsz", "not recorded"))
        return model
    except Exception as exc:
        say("  COULD NOT READ MODEL METADATA:", repr(exc))
        return None


def describe_detection_data():
    section("2. PALLET DETECTION DATASET AND ANNOTATIONS")
    say("Dataset:", PALLET_DATA)
    total_boxes = 0
    all_images = 0
    for split in ("train", "val", "test"):
        images = files_in(PALLET_DATA / "images" / split)
        labels_dir = PALLET_DATA / "labels" / split
        box_count = 0
        missing = 0
        class_counts = Counter()
        for image in images:
            label_path = labels_dir / (image.stem + ".txt")
            if not label_path.is_file():
                missing += 1
                continue
            for text in label_path.read_text(
                encoding="utf-8"
            ).splitlines():
                if text.strip():
                    box_count += 1
                    class_counts[text.split()[0]] += 1
        sizes = Counter()
        for image in images:
            picture = cv2.imread(str(image))
            if picture is not None:
                h, w = picture.shape[:2]
                sizes[f"{w}x{h}"] += 1
        say(f"  {split.upper()}: {len(images)} images; "
            f"{box_count} annotated boxes; "
            f"missing labels={missing}; sizes={dict(sizes)}; "
            f"class IDs={dict(class_counts)}")
        total_boxes += box_count
        all_images += len(images)
    say("TOTAL:", all_images, "images;", total_boxes, "bounding boxes")
    say("Annotation format: YOLO detection, 0=centered pallet class;")
    say("  coordinates are normalized x_center, y_center, width, height.")
    manifest = PALLET_DATA / "split_manifest.csv"
    if manifest.is_file():
        with manifest.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        counts = Counter((r.get("split"), r.get("source_type")) for r in rows)
        say("Dataset origin by split:")
        for (split, origin), count in sorted(counts.items()):
            say(f"  {split} / {origin}: {count} images")
    say("NOTE: All 100 are dataset images, NOT 100 gradient-training images.")
    say("NOTE: Training used 80; validation 10; locked test 10.")
    say("Image/annotation quality: manually reviewed; no independent audit recorded.")


def describe_classification_data(title, path, labels):
    section(title)
    say("Dataset:", path)
    total = 0
    for split in ("train", "val", "test"):
        parts = []
        for label in labels:
            count = len(files_in(path / split / label))
            total += count
            parts.append(f"{label}={count}")
        say(f"  {split.upper()}: " + ", ".join(parts))
    say("TOTAL images:", total)
    for split in ("train", "val", "test"):
        imgs = []
        for label in labels:
            imgs.extend(files_in(path / split / label))
        if imgs:
            sizes = Counter()
            for image in imgs[:30]:
                pic = cv2.imread(str(image))
                if pic is not None:
                    h, w = pic.shape[:2]
                    sizes[f"{w}x{h}"] += 1
            say(f"  {split} first {min(30, len(imgs))} image dimensions:",
                dict(sizes))


def training_details():
    section("6. TRAINING SETTINGS AND TRAINING CURVE")
    args_path = PALLET_RUN / "args.yaml"
    say("Pallet-detector training settings:", args_path)
    if args_path.is_file():
        try:
            import yaml
            values = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
            keys = (
                "model", "data", "epochs", "patience", "batch", "imgsz",
                "optimizer", "lr0", "weight_decay", "seed", "deterministic",
                "device", "workers", "pretrained", "mosaic", "scale",
                "translate", "degrees", "fliplr", "flipud", "hsv_h",
                "hsv_s", "hsv_v", "auto_augment", "mixup", "erasing"
            )
            for key in keys:
                if key in values:
                    say(f"  {key}: {values[key]}")
        except Exception as exc:
            say("  Could not read args.yaml:", repr(exc))
    else:
        say("  Training arguments NOT FOUND")

    csv_path = PALLET_RUN / "results.csv"
    say("Pallet training history:", csv_path)
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            records = [
                {k.strip(): v.strip() for k, v in row.items() if k}
                for row in csv.DictReader(stream)
            ]
        say("  Logged epochs:", len(records))
        if records:
            last = records[-1]
            say("  Last recorded epoch:", last.get("epoch", "unknown"))
            important = (
                "train/box_loss", "train/cls_loss", "train/dfl_loss",
                "val/box_loss", "val/cls_loss", "val/dfl_loss",
                "metrics/precision(B)", "metrics/recall(B)",
                "metrics/mAP50(B)", "metrics/mAP50-95(B)"
            )
            for key in important:
                if key in last:
                    say(f"  Last epoch {key}: {last[key]}")
            key = "metrics/mAP50-95(B)"
            valid = []
            for record in records:
                try:
                    valid.append((float(record[key]), record))
                except (KeyError, TypeError, ValueError):
                    pass
            if valid:
                value, record = max(valid, key=lambda pair: pair[0])
                say("  Peak logged validation mAP50-95:",
                    f"{value:.4f}", "at epoch", record.get("epoch", "unknown"))
                say("  NOTE: Selected checkpoint is best.pt, not last.pt.")
    else:
        say("  Training results.csv NOT FOUND")

    say("X/NO_X final hard fine-tuning config:", HARD_CONFIG)
    if HARD_CONFIG.is_file():
        try:
            import json
            config = json.loads(HARD_CONFIG.read_text(encoding="utf-8"))
            for key in (
                "purpose", "start_model", "final_model", "imgsz",
                "epochs", "patience", "batch", "optimizer", "lr0",
                "weight_decay", "seed", "selection_rule"
            ):
                if key in config:
                    say(f"  {key}: {config[key]}")
        except Exception as exc:
            say("  Could not read symbol config:", repr(exc))

    say("EMPTY/OBJECT training configuration:")
    object_args = sorted((ROOT / "03_Empty_Object_Classifier").rglob("args.yaml"))
    if object_args:
        for candidate in object_args[:5]:
            say("  Available file:", candidate)
        say("  Original training run identity must be confirmed before citing its settings.")
    else:
        say("  Not found in copied Conference artifacts.")


def historical_symbol_results():
    section("8. RECORDED X/NO_X FINAL-HARD RESULTS")
    say("Source:", HARD_RESULTS)
    if not HARD_RESULTS.is_file():
        say("NOT FOUND. Do not invent these numbers.")
        return
    text = HARD_RESULTS.read_text(encoding="utf-8", errors="replace")
    for name in ("VAL", "TEST"):
        pattern = (
            r"FINAL YOLO26 HARD FINE-TUNE - " + name
            + r"\s*\n=+\s*\n(.*?)\nPER-IMAGE PREDICTIONS"
        )
        match = re.search(pattern, text, re.DOTALL)
        if match:
            say(name + " from saved training report:")
            for line in match.group(1).splitlines():
                if line.strip() and not set(line.strip()) <= {"-", "="}:
                    say("  " + line.strip())
        else:
            say(name, "summary not found in file format.")
    say("This is saved test evidence, separate from re-evaluation below.")


def old_vs_new():
    section("7. PALLET DETECTOR: COMMON HELD-OUT TEST COMPARISON")
    say("Source:", COMPARISON)
    if not COMPARISON.is_file():
        say("No saved 20-vs-100 comparison found.")
        return
    say("Test dataset: 10 new full-frame images; evaluation imgsz=960")
    for line in COMPARISON.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        match = re.match(
            r"^(precision|recall|map50|map50_95)\s+"
            r"([0-9.]+)\s+([0-9.]+)\s+([-+0-9.]+)", line
        )
        if match:
            label, old, new, difference = match.groups()
            say(f"  {label}: old={float(old)*100:.2f}% "
                f"new={float(new)*100:.2f}% "
                f"change={float(difference)*100:+.2f} percentage points")
    say("Comparison columns: 20-image baseline, conference 100-image")
    say("expanded-dataset model, and new-minus-old difference.")
    say("Test labels began as proposals from the old detector and were")
    say("manually reviewed. State this possible evaluation bias in the paper.")
    say("Metrics are box detection metrics, NOT whole-system accuracy.")
    for folder in (
        PALLET_RUN,
        ROOT / "00_Pallet_20_vs_100_Comparison" / "original_20_image_model",
        ROOT / "00_Pallet_20_vs_100_Comparison" / "conference_100_image_model"
    ):
        if folder.is_dir():
            say("Additional training/test graphs in:", folder)
            for name in (
                "results.png", "confusion_matrix.png",
                "confusion_matrix_normalized.png", "BoxPR_curve.png",
                "BoxF1_curve.png", "BoxP_curve.png", "BoxR_curve.png",
                "val_batch0_pred.jpg", "val_batch0_labels.jpg"
            ):
                if (folder / name).is_file():
                    say("  FIGURE:", folder / name)


def evaluate_classifier(title, model, folder, classes, split):
    say()
    say(f"{title} / {split.upper()} HOLDOUT OR VALIDATION")
    if model is None:
        say("  SKIPPED: model unavailable")
        return
    cases = []
    for label in classes:
        for image in files_in(folder / split / label):
            cases.append((label, image))
    if not cases:
        say("  SKIPPED: no images found")
        return
    size = getattr(model, "overrides", {}).get("imgsz")
    try:
        size = int(size) if size is not None else 224
    except (ValueError, TypeError):
        size = 224
    say("  Images:", len(cases), "input imgsz:", size, "device:", DEVICE)
    say("  Test-time augmentation: disabled")
    say("  Ground truth is the class folder name.")
    matrix = {truth: Counter() for truth in classes}
    errors = []
    confidences = []
    try:
        outputs = model.predict(
            source=[str(path) for _, path in cases],
            imgsz=size,
            batch=8,
            device=DEVICE,
            stream=True,
            verbose=False,
            augment=False,
        )
        observed = 0
        for (truth, path), result in zip(cases, outputs):
            if result.probs is None:
                raise RuntimeError("Model returned no classification probabilities")
            predicted = str(result.names[int(result.probs.top1)]).upper()
            predicted = predicted.replace(" ", "_").replace("-", "_")
            confidence = float(result.probs.top1conf)
            matrix[truth][predicted] += 1
            confidences.append(confidence)
            observed += 1
            if predicted != truth:
                errors.append((path.name, truth, predicted, confidence))
        if observed != len(cases):
            raise RuntimeError(f"Only got {observed} predictions for {len(cases)} images")
    except Exception as exc:
        say("  EVALUATION ERROR:", repr(exc))
        say("  No metrics reported for this split.")
        return
    correct = sum(matrix[label][label] for label in classes)
    say("  CORRECT:", f"{correct}/{len(cases)}")
    say("  ACCURACY:", f"{correct / len(cases) * 100:.2f}%")
    say("  CONFUSION MATRIX (rows=TRUE, columns=PREDICTED):")
    say("  Predicted columns:", ", ".join(classes))
    for truth in classes:
        say("  " + truth + ": " + ", ".join(
            str(matrix[truth][prediction]) for prediction in classes
        ))
    f1_values = []
    for label in classes:
        tp = matrix[label][label]
        fn = sum(matrix[label].values()) - tp
        fp = sum(matrix[other][label] for other in classes if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        say(f"  {label}: support={tp + fn} TP={tp} FP={fp} FN={fn} "
            f"precision={precision:.4f} recall={recall:.4f} F1={f1:.4f}")
    say("  Macro F1:", f"{sum(f1_values) / len(f1_values):.4f}")
    say("  Mean top-1 confidence:",
        f"{sum(confidences) / len(confidences):.4f}")
    say("  Misclassified images:", len(errors))
    for name, truth, prediction, score in errors[:30]:
        say(f"    {name}: true={truth} predicted={prediction} confidence={score:.4f}")
    if len(errors) > 30:
        say("    More error filenames omitted;", len(errors) - 30, "additional")


def main():
    section("CREST - COMPLETE CONFERENCE MODEL EVIDENCE REPORT")
    say("Report generated:", datetime.now().isoformat(timespec="seconds"))
    say("Conference root:", ROOT)
    say("Scope: CONFERENCE artifacts only. NO TRAINING. NO MODEL CHANGES.")
    say("Pipeline:")
    say("  RGB camera -> YOLO26 pallet detector -> tracking ->")
    say("  EMPTY/OBJECT_PRESENT classifier -> X/NO_X classifier ->")
    say("  EMPTY / ACCEPTED / REJECTED decisions.")
    say("  X = REJECTED, NO_X = ACCEPTED.")
    say("Live integration code:", CONFERENCE_LIVE)
    say("Live code present:", CONFERENCE_LIVE.is_file())

    section("1. HARDWARE, SOFTWARE, MODEL ARCHITECTURE")
    say("Python:", sys.version.split()[0])
    say("OS:", platform.platform())
    say("Torch:", torch.__version__)
    say("Ultralytics:", ultralytics.__version__)
    say("OpenCV:", cv2.__version__)
    say("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        say("GPU:", torch.cuda.get_device_name(0))
        say("CUDA runtime:", torch.version.cuda)
    say("Camera information from project history:")
    say("  Single RGB camera; 1920x1080 frames; 30 FPS requested.")
    say("  Requested FPS is NOT a measured end-to-end FPS.")
    pallet = describe_model("100-image CONFERENCE pallet detector:", PALLET_MODEL)
    obj = describe_model("EMPTY vs OBJECT_PRESENT classifier:", OBJECT_MODEL)
    sym = describe_model("X vs NO_X classifier:", SYMBOL_MODEL)
    say("Archived conference copy of original pallet model:", OLD_MODEL)

    describe_detection_data()
    describe_classification_data(
        "3. EMPTY/OBJECT_PRESENT CLASSIFICATION DATASET",
        OBJECT_DATA, ("EMPTY", "OBJECT_PRESENT")
    )
    describe_classification_data(
        "4. X/NO_X CLASSIFICATION DATASET",
        SYMBOL_DATA, ("X", "NO_X")
    )
    section("5. DATASET ORIGIN AND EXPERIMENTAL SPLITS")
    say("Pallet: 100 unique full-frame images (80 train, 10 val, 10 test).")
    say("  Current-20 source frames included among the 80 training images.")
    say("EMPTY/OBJECT and X/NO_X datasets use separate labeled CROPS.")
    say("  The 100 detector images were NOT automatically used to train classifiers.")
    say("  Test images must not be reused in training or tuning.")
    say("Reusable 276-image EMPTY/ACCEPTED/REJECTED folder was not confirmed")
    say("  as the training source for these deployed classifiers.")

    training_details()
    old_vs_new()
    historical_symbol_results()

    section("9. LIVE RE-EVALUATION: CLASSIFIERS ON THEIR OWN DATASET SPLITS")
    say("Below numbers are calculated from actual classifier predictions")
    say("against folder labels. These are individual-model results, not")
    say("end-to-end conveyor accuracy.")
    evaluate_classifier("EMPTY/OBJECT", obj, OBJECT_DATA,
                        ("EMPTY", "OBJECT_PRESENT"), "val")
    evaluate_classifier("EMPTY/OBJECT", obj, OBJECT_DATA,
                        ("EMPTY", "OBJECT_PRESENT"), "test")
    evaluate_classifier("X/NO_X", sym, SYMBOL_DATA,
                        ("X", "NO_X"), "val")
    evaluate_classifier("X/NO_X", sym, SYMBOL_DATA,
                        ("X", "NO_X"), "test")

    section("10. END-TO-END SYSTEM AND PAPER REPORTING CHECKLIST")
    say("Conference live pipeline: user reported visually working in lab.")
    say("Measured end-to-end correct pallets / total: NOT RECORDED.")
    say("Measured end-to-end classification accuracy: NOT RECORDED.")
    say("Measured end-to-end FPS/latency: NOT RECORDED.")
    say("RFID / PLC / actuation performance: NOT EVALUATED BY THIS SCRIPT.")
    say("Do not substitute pallet mAP or X/NO_X accuracy for end-to-end accuracy.")
    say("Do not describe 100 detector images as 100 classifier training images.")
    say("Caveats for paper: limited 10-image detector test set;")
    say("  4-image EMPTY/OBJECT test set; potential old-detector pseudo-label bias;")
    say("  visual review of overlapping pallet annotations is not independently verified.")
    say("PAPER SECTIONS: system architecture; camera/acquisition; dataset")
    say("  collection and split; annotation; model architecture; preprocessing;")
    say("  training hyperparameters; validation and held-out results; baseline")
    say("  comparison; qualitative live testing; limitations and future work.")
    say("NO weights, original labels, or main CREST files were modified.")

    OUT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUT / f"conference_full_model_report_{timestamp}.txt"
    if report_path.exists():
        raise RuntimeError("Report path already exists; refusing overwrite")
    report_path.write_text("\n".join(LINES) + "\n", encoding="utf-8")
    say()
    say("SAVED REPORT:", report_path)
    say("Copy terminal text above OR open the saved .txt file.")


if __name__ == "__main__":
    main()
