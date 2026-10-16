"""CREST Phase 59: test frozen EMPTY/OBJECT classifier on new empty-pallet crops.

Input is the 61 freshly captured empty-pallet frames plus Phase 57's hand-drawn
pallet labels. The physical contents were established by the lab operator as
EMPTY before capture. No new labels are inferred from the model predictions.

IMPORTANT: This is an EMPTY-only new-session test using manually drawn boxes.
It does NOT evaluate OBJECT_PRESENT, X/NO_X, or Phase 40's exact crop policy.
The 366 boxes are correlated observations of six physical pallets.
Nothing in the original dataset, original models, or Phase 40 is modified.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(r"C:\Do_Not_Delete_PLC\Conference\09_Independent_Blind_Test_Captures")
ANNOTATED = ROOT / "_57_Manual_Pallet_Test_Labels"
IMAGE_DIR = ANNOTATED / "images" / "test"
LABEL_DIR = ANNOTATED / "labels" / "test"
MANIFEST = ANNOTATED / "ground_truth_manifest.csv"
MODEL = Path(
    r"C:\Do_Not_Delete_PLC\Conference\03_Empty_Object_Classifier"
    r"\MODEL\empty_object_yolo26n_cls_best.pt"
)
EXPECTED_HASH = "366ecc39ef8e6566774ac68200548218b51cc4246352e56301f8cd7e8cb31083"
OUTPUT = ROOT / "_59_Frozen_EMPTY_Classifier_New_Session"


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_yolo_boxes(path, width, height):
    boxes = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Invalid class or box in {path} line {line_no}")
        cx, cy, bw, bh = map(float, parts[1:])
        if not (0 < cx < 1 and 0 < cy < 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"Invalid normalized box in {path} line {line_no}")
        if cx - bw / 2 < -1e-5 or cx + bw / 2 > 1 + 1e-5:
            raise ValueError(f"Box exceeds frame width in {path} line {line_no}")
        if cy - bh / 2 < -1e-5 or cy + bh / 2 > 1 + 1e-5:
            raise ValueError(f"Box exceeds frame height in {path} line {line_no}")
        x1 = max(0, round((cx - bw / 2) * width))
        y1 = max(0, round((cy - bh / 2) * height))
        x2 = min(width, round((cx + bw / 2) * width))
        y2 = min(height, round((cy + bh / 2) * height))
        if x2 - x1 < 8 or y2 - y1 < 8:
            raise ValueError(f"Box too small in {path} line {line_no}")
        boxes.append((x1, y1, x2, y2))
    return boxes


def collect_examples():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Phase 57 manifest missing: {MANIFEST}")
    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as stream:
        records = list(csv.DictReader(stream))
    if not records:
        raise RuntimeError("No annotated frames in manifest")
    examples = []
    sessions = Counter()
    for rec in records:
        if rec.get("status") != "labeled":
            raise RuntimeError("All frames must be labeled empty-PALLET scenes: " + str(rec))
        image = Path(rec["evaluation_image"])
        if image.parent.resolve() != IMAGE_DIR.resolve() or not image.is_file():
            raise RuntimeError(f"Invalid/missing evaluation image: {image}")
        frame = cv2.imread(str(image))
        if frame is None:
            raise RuntimeError(f"Unreadable frame: {image}")
        h, w = frame.shape[:2]
        label = LABEL_DIR / (image.stem + ".txt")
        if not label.is_file():
            raise RuntimeError(f"Missing ground-truth label: {label}")
        boxes = read_yolo_boxes(label, w, h)
        if len(boxes) != int(rec["pallet_boxes"]):
            raise RuntimeError(f"Count mismatch for {image.name}")
        if len(boxes) != 6:
            raise RuntimeError(f"Expected six pallets in {image.name}; found {len(boxes)}")
        session = rec["session"]
        sessions[session] += 1
        for index, coords in enumerate(boxes, start=1):
            x1, y1, x2, y2 = coords
            crop = frame[y1:y2, x1:x2].copy()
            if crop.size == 0:
                raise RuntimeError(f"Empty crop from {image.name} box {index}")
            examples.append((session, image.name, index, coords, crop))
    return examples, sessions, len(records)


def main():
    parser = argparse.ArgumentParser(description="Frozen EMPTY-only classifier new-session test")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    print("CREST PHASE 59 - NEW EMPTY-PALLET CLASSIFIER TEST")
    print("READ ONLY: original models, camera, Phase 40, PLC and datasets untouched")
    print("Source: new empty-pallet camera captures with Phase 57 manual boxes")
    examples, sessions, image_count = collect_examples()
    print(f"CHECKED: {image_count} full frames; {len(examples)} empty pallet crops; {len(sessions)} sessions")
    print("WARNING: Correlated crops from six pallets, NOT independent experiments.")
    print("WARNING: Manually labeled box crops, NOT verified Phase 40 production crops.")
    if args.check_only:
        print("CHECK-ONLY PASSED: no classifier run yet")
        return

    if not MODEL.is_file():
        raise FileNotFoundError(f"Deployed EMPTY/OBJECT model missing: {MODEL}")
    actual_hash = hash_file(MODEL)
    if actual_hash != EXPECTED_HASH:
        raise RuntimeError(
            "STOP: Deployed classifier hash changed. Expected " + EXPECTED_HASH
            + "; got " + actual_hash
        )

    from ultralytics import YOLO
    import torch

    classifier = YOLO(str(MODEL))
    names = {int(k): str(v).upper() for k, v in classifier.names.items()}
    if set(names.values()) != {"EMPTY", "OBJECT_PRESENT"}:
        raise RuntimeError(f"Unexpected classifier class names: {names}")
    empty_idx = next(k for k, v in names.items() if v == "EMPTY")
    object_idx = next(k for k, v in names.items() if v == "OBJECT_PRESENT")
    device = 0 if torch.cuda.is_available() else "cpu"

    run_dir = OUTPUT / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    failures_dir = run_dir / "wrong_crops"
    failures_dir.mkdir()
    results_file = run_dir / "per_pallet_predictions.csv"
    fields = [
        "session", "image", "pallet_index", "x1", "y1", "x2", "y2",
        "ground_truth", "predicted", "prob_empty", "prob_object",
        "predicted_confidence", "correct",
    ]

    frame_hits = defaultdict(list)
    session_counts = Counter()
    correct = 0
    with results_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for start in range(0, len(examples), 8):
            block = examples[start:start + 8]
            predictions = classifier.predict(
                source=[e[4] for e in block],
                imgsz=320, device=device, batch=len(block),
                verbose=False, augment=False,
            )
            if len(predictions) != len(block):
                raise RuntimeError("Model results count does not match crop count")
            for (session, name, index, coords, crop), prediction in zip(block, predictions):
                if prediction.probs is None:
                    raise RuntimeError("Expected classification probabilities")
                probs = prediction.probs.data.detach().cpu().tolist()
                predicted_id = int(prediction.probs.top1)
                predicted_label = names[predicted_id]
                good = predicted_label == "EMPTY"
                correct += int(good)
                frame_hits[name].append(good)
                session_counts[(session, "total")] += 1
                session_counts[(session, "correct")] += int(good)
                x1, y1, x2, y2 = coords
                writer.writerow({
                    "session": session, "image": name, "pallet_index": index,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "ground_truth": "EMPTY", "predicted": predicted_label,
                    "prob_empty": f"{float(probs[empty_idx]):.6f}",
                    "prob_object": f"{float(probs[object_idx]):.6f}",
                    "predicted_confidence": f"{float(prediction.probs.top1conf):.6f}",
                    "correct": int(good),
                })
                if not good:
                    cv2.imwrite(str(failures_dir / f"{Path(name).stem}__pallet_{index}.png"), crop)
            print(f"Evaluated {min(start + 8, len(examples))}/{len(examples)} crops", flush=True)

    wrong = len(examples) - correct
    all_correct_frames = sum(all(hits) for hits in frame_hits.values())
    per_session = [
        {"session": key, "frames": sessions[key],
         "correct_crops": session_counts[(key, "correct")],
         "total_crops": session_counts[(key, "total")]}
        for key in sorted(sessions)
    ]
    summary = {
        "scope": "EMPTY-only frozen classifier; manually annotated pallet crops",
        "model_path": str(MODEL), "model_sha256": actual_hash,
        "imgsz": 320, "device": str(device), "image_count": image_count,
        "number_of_physical_pallets": 6,
        "session_count": len(sessions), "crop_count": len(examples),
        "correct_empty": correct, "incorrect_object_present": wrong,
        "empty_class_accuracy_percent": 100 * correct / len(examples),
        "all_six_empty_frames": all_correct_frames,
        "all_six_empty_frame_rate_percent": 100 * all_correct_frames / image_count,
        "tested_object_present": 0,
        "independent_physical_pass_count": "not established",
        "note": "Cannot report binary-class accuracy/sensitivity using EMPTY-only data; "
                "these frames are correlated and manual crops differ from Phase 40 crops.",
        "per_session": per_session,
    }
    (run_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "CREST PHASE 59 - FROZEN EMPTY/OBJECT TEST ON NEW EMPTY-PALLET CAPTURES",
        "=" * 67,
        f"Frozen checkpoint SHA256: {actual_hash}",
        f"Full-frame captures: {image_count}",
        f"Camera recording sessions: {len(sessions)}",
        f"Crops evaluated (correlated): {len(examples)}",
        f"Correct EMPTY: {correct}",
        f"Incorrect OBJECT_PRESENT: {wrong}",
        f"EMPTY recall on these manual-box crops: {100 * correct / len(examples):.2f}%",
        f"Frames with all six correctly EMPTY: {all_correct_frames}/{image_count}",
        f"Input classification imgsz: 320 (unchanged)",
        "OBJECT_PRESENT images tested: 0",
        "IMPORTANT: Not overall classifier accuracy, not end-to-end live accuracy.",
        "IMPORTANT: Six recurring physical pallets; captures are correlated.",
        "IMPORTANT: Manual ground-truth crops may differ from live Phase 40 crops.",
        "IMPORTANT: Frozen weights and test set remain unchanged.",
        f"Output: {run_dir}",
    ]
    report = "\n".join(lines) + "\n"
    (run_dir / "results.txt").write_text(report, encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"PHASE 59 STOPPED: {exc}", file=sys.stderr)
        sys.exit(1)
