"""CREST Phase 60: human-labeled EMPTY/OBJECT test on existing Phase 57 crops.

Review is independent of model predictions. The original 61 camera frames,
366 manually drawn pallet boxes, model weights and live Phase 40 stay unchanged.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

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
OUTPUT = ROOT / "_60_Corrected_EMPTY_OBJECT_Test"
GROUND_TRUTH = OUTPUT / "human_reviewed_ground_truth.csv"
CLASSES = ("EMPTY", "OBJECT_PRESENT")
LABEL_FIELDS = [
    "session", "image", "pallet_index", "x1", "y1", "x2", "y2",
    "ground_truth", "reviewed_at"
]

TILE_W, TILE_H = 350, 280
HEADER, FOOTER = 85, 75
WINDOW = "CREST Phase 60 - Human Ground Truth Review"


def checksum(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_boxes(label_path, width, height):
    boxes = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Invalid pallet label: {label_path}:{line_number}")
        cx, cy, bw, bh = map(float, parts[1:])
        if not (0 < cx < 1 and 0 < cy < 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise ValueError(f"Invalid normalized box: {label_path}:{line_number}")
        if cx - bw / 2 < -1e-5 or cx + bw / 2 > 1 + 1e-5:
            raise ValueError(f"Box outside frame: {label_path}:{line_number}")
        if cy - bh / 2 < -1e-5 or cy + bh / 2 > 1 + 1e-5:
            raise ValueError(f"Box outside frame: {label_path}:{line_number}")
        box = (
            max(0, round((cx - bw / 2) * width)),
            max(0, round((cy - bh / 2) * height)),
            min(width, round((cx + bw / 2) * width)),
            min(height, round((cy + bh / 2) * height)),
        )
        if box[2] - box[0] < 8 or box[3] - box[1] < 8:
            raise ValueError(f"Crop too small: {label_path}:{line_number}")
        boxes.append(box)
    return boxes


def load_frames():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Missing Phase 57 manifest: {MANIFEST}")
    with MANIFEST.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise RuntimeError("Phase 57 manifest is empty")
    frames = []
    used_names = set()
    for rec in records:
        if rec.get("status") != "labeled":
            raise RuntimeError("Unlabeled frame in Phase 57 manifest")
        image = Path(rec["evaluation_image"])
        if image.parent.resolve() != IMAGE_DIR.resolve() or not image.is_file():
            raise FileNotFoundError(f"Unexpected/missing evaluation frame: {image}")
        if image.name in used_names:
            raise RuntimeError(f"Duplicate image name in manifest: {image.name}")
        used_names.add(image.name)
        frame = cv2.imread(str(image))
        if frame is None:
            raise RuntimeError(f"Cannot read {image}")
        h, w = frame.shape[:2]
        label = LABEL_DIR / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing label: {label}")
        boxes = read_boxes(label, w, h)
        if len(boxes) != 6 or len(boxes) != int(rec["pallet_boxes"]):
            raise RuntimeError(f"Expected six boxes in {image.name}")
        frames.append({
            "image": image.name, "path": image,
            "session": rec["session"], "boxes": boxes,
        })
    return frames


def expected_rows(frame, labels, reviewed_at):
    rows = []
    for index, (coords, label) in enumerate(zip(frame["boxes"], labels), 1):
        if label not in CLASSES:
            raise ValueError(f"Bad human label for {frame['image']} pallet {index}")
        rows.append({
            "session": frame["session"], "image": frame["image"],
            "pallet_index": index,
            "x1": coords[0], "y1": coords[1],
            "x2": coords[2], "y2": coords[3],
            "ground_truth": label, "reviewed_at": reviewed_at,
        })
    return rows


def read_reviews(frames):
    if not GROUND_TRUTH.is_file():
        return {}
    with GROUND_TRUTH.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    by_image = {}
    for row in rows:
        by_image.setdefault(row["image"], []).append(row)
    allowed = {f["image"] for f in frames}
    if set(by_image) - allowed:
        raise RuntimeError("Review CSV contains unknown image names")
    lookup = {f["image"]: f for f in frames}
    for name, found in by_image.items():
        frame = lookup[name]
        if len(found) != 6:
            raise RuntimeError(f"Incomplete review: {name}")
        found.sort(key=lambda r: int(r["pallet_index"]))
        for idx, (actual, expected) in enumerate(
            zip(found, expected_rows(frame, ["EMPTY"] * 6, "")), 1
        ):
            for key in ("session", "image"):
                if actual[key] != expected[key]:
                    raise RuntimeError(f"Review mismatch: {name} #{idx} {key}")
            for key in ("pallet_index", "x1", "y1", "x2", "y2"):
                if int(actual[key]) != int(expected[key]):
                    raise RuntimeError(f"Boxes changed since review: {name} #{idx}")
            if actual["ground_truth"] not in CLASSES:
                raise RuntimeError(f"Invalid class in reviews: {name} #{idx}")
            if not actual.get("reviewed_at"):
                raise RuntimeError(f"Unconfirmed review: {name} #{idx}")
    return by_image


def save_reviews(reviews, frames):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    temp = GROUND_TRUTH.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LABEL_FIELDS)
        writer.writeheader()
        for frame in frames:
            for row in reviews.get(frame["image"], []):
                writer.writerow(row)
    temp.replace(GROUND_TRUTH)


def create_board(frame, image_data, labels, frame_num, total):
    canvas = np.full((HEADER + 2 * TILE_H + FOOTER, 3 * TILE_W, 3), 28, dtype=np.uint8)
    cv2.putText(canvas, f"REVIEW {frame_num}/{total}  -  {frame['image'][:52]}",
                (13, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (255, 255, 255), 2)
    cv2.putText(canvas, "Click the pallets containing objects. All others stay EMPTY.",
                (13, 61), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (210, 210, 210), 1)
    for i, ((x1, y1, x2, y2), label) in enumerate(zip(frame["boxes"], labels)):
        left = (i % 3) * TILE_W
        top = HEADER + (i // 3) * TILE_H
        color = (70, 190, 70) if label == "EMPTY" else (55, 80, 240)
        cv2.rectangle(canvas, (left + 3, top + 3),
                      (left + TILE_W - 3, top + TILE_H - 3), color, 3)
        cv2.putText(canvas, f"{i + 1}: {label}", (left + 16, top + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        crop = image_data[y1:y2, x1:x2]
        maxw, maxh = TILE_W - 22, TILE_H - 73
        scale = min(maxw / crop.shape[1], maxh / crop.shape[0])
        nw = max(1, round(crop.shape[1] * scale))
        nh = max(1, round(crop.shape[0] * scale))
        resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
        xx = left + (TILE_W - nw) // 2
        yy = top + 48 + (maxh - nh) // 2
        canvas[yy:yy + nh, xx:xx + nw] = resized
    footer_y = HEADER + 2 * TILE_H
    cv2.putText(canvas, "SPACE = confirm/save next    B = previous    Q = save/quit",
                (13, footer_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.64,
                (255, 255, 255), 2)
    cv2.putText(canvas, "Keys 1-6 also toggle object presence. Review EVERY frame.",
                (13, footer_y + 57), cv2.FONT_HERSHEY_SIMPLEX, 0.57,
                (180, 180, 180), 1)
    return canvas


def review(frames):
    reviews = read_reviews(frames)
    first = next((i for i, f in enumerate(frames) if f["image"] not in reviews), len(frames))
    if first == len(frames):
        print("All frames already reviewed. Run --evaluate to get results.")
        return
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    position = first
    try:
        while 0 <= position < len(frames):
            frame = frames[position]
            img = cv2.imread(str(frame["path"]))
            if img is None:
                raise RuntimeError(f"Cannot read {frame['path']}")
            old = reviews.get(frame["image"])
            labels = [row["ground_truth"] for row in old] if old else ["EMPTY"] * 6

            def click(event, x, y, flags, param):
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                if not (HEADER <= y < HEADER + 2 * TILE_H and 0 <= x < 3 * TILE_W):
                    return
                tile = ((y - HEADER) // TILE_H) * 3 + x // TILE_W
                labels[tile] = "OBJECT_PRESENT" if labels[tile] == "EMPTY" else "EMPTY"

            cv2.setMouseCallback(WINDOW, click)
            while True:
                cv2.imshow(WINDOW, create_board(
                    frame, img, labels, position + 1, len(frames)))
                key = cv2.waitKey(40) & 0xFF
                if ord("1") <= key <= ord("6"):
                    i = key - ord("1")
                    labels[i] = "OBJECT_PRESENT" if labels[i] == "EMPTY" else "EMPTY"
                elif key == 32:
                    now = datetime.now().isoformat(timespec="seconds")
                    reviews[frame["image"]] = expected_rows(frame, labels, now)
                    save_reviews(reviews, frames)
                    print(f"Saved {position + 1}/{len(frames)}: "
                          f"{labels.count('OBJECT_PRESENT')} OBJECT_PRESENT; "
                          f"{labels.count('EMPTY')} EMPTY", flush=True)
                    position += 1
                    break
                elif key in (ord("b"), ord("B")):
                    position = max(0, position - 1)
                    break
                elif key in (ord("q"), ord("Q"), 27):
                    print("Review paused; confirmed frames are saved.")
                    return
        print("Review finished. Now run the --evaluate command.")
        print(f"Labels: {GROUND_TRUTH}")
    finally:
        cv2.destroyAllWindows()


def evaluate(frames):
    reviews = read_reviews(frames)
    missing = [f["image"] for f in frames if f["image"] not in reviews]
    if missing:
        raise RuntimeError(f"Review all frames first! Missing {len(missing)}. Run --review.")
    if not MODEL.is_file():
        raise FileNotFoundError(MODEL)
    actual_hash = checksum(MODEL)
    if actual_hash != EXPECTED_HASH:
        raise RuntimeError("Frozen model SHA256 mismatch; refusing evaluation")

    from ultralytics import YOLO
    import torch
    classifier = YOLO(str(MODEL))
    names = {int(i): str(name).upper() for i, name in classifier.names.items()}
    if set(names.values()) != set(CLASSES):
        raise RuntimeError(f"Unexpected model classes: {names}")
    empty_idx = next(i for i, name in names.items() if name == "EMPTY")
    obj_idx = next(i for i, name in names.items() if name == "OBJECT_PRESENT")
    device = 0 if torch.cuda.is_available() else "cpu"

    examples = []
    for frame in frames:
        for row in reviews[frame["image"]]:
            examples.append((frame, row))
    run_dir = OUTPUT / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    wrong_dir = run_dir / "wrong_crops"
    wrong_dir.mkdir()
    fields = ["session", "image", "pallet_index", "ground_truth", "predicted",
              "prob_empty", "prob_object", "predicted_confidence", "correct"]
    counts = Counter()
    frames_correct = Counter()
    with (run_dir / "per_pallet_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for start in range(0, len(examples), 8):
            block = examples[start:start + 8]
            crops = []
            for frame, row in block:
                img = cv2.imread(str(frame["path"]))
                if img is None:
                    raise RuntimeError(f"Missing frame: {frame['path']}")
                x1, y1, x2, y2 = (int(row[k]) for k in ("x1", "y1", "x2", "y2"))
                crops.append(img[y1:y2, x1:x2].copy())
            predictions = classifier.predict(
                source=crops, imgsz=320, device=device, batch=len(crops),
                verbose=False, augment=False,
            )
            if len(predictions) != len(block):
                raise RuntimeError("Prediction count mismatch")
            for (frame, row), crop, pred in zip(block, crops, predictions):
                if pred.probs is None:
                    raise RuntimeError("Classifier returned no probabilities")
                probs = pred.probs.data.detach().cpu().tolist()
                truth = row["ground_truth"]
                decision = names[int(pred.probs.top1)]
                correct = truth == decision
                counts[(truth, decision)] += 1
                frames_correct[frame["image"]] += int(correct)
                writer.writerow({
                    "session": frame["session"], "image": frame["image"],
                    "pallet_index": row["pallet_index"],
                    "ground_truth": truth, "predicted": decision,
                    "prob_empty": f"{probs[empty_idx]:.6f}",
                    "prob_object": f"{probs[obj_idx]:.6f}",
                    "predicted_confidence": f"{float(pred.probs.top1conf):.6f}",
                    "correct": int(correct),
                })
                if not correct:
                    fname = f"{Path(frame['image']).stem}__pallet_{row['pallet_index']}.png"
                    cv2.imwrite(str(wrong_dir / fname), crop)
            print(f"Evaluated {min(start + 8, len(examples))}/{len(examples)}", flush=True)

    total = len(examples)
    correct = sum(counts[(name, name)] for name in CLASSES)
    actual = {name: sum(counts[(name, pred)] for pred in CLASSES) for name in CLASSES}
    recall = {
        name: (100.0 * counts[(name, name)] / actual[name] if actual[name] else None)
        for name in CLASSES
    }
    with (run_dir / "confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["ground_truth / predicted", *CLASSES])
        for ground_truth in CLASSES:
            writer.writerow([ground_truth, *(counts[(ground_truth, p)] for p in CLASSES)])

    report = [
        "CREST PHASE 60 - HUMAN-REVIEWED EMPTY/OBJECT TEST",
        "=" * 57,
        f"Full frames: {len(frames)}; manually reviewed crops: {total}",
        f"Frozen model SHA256: {actual_hash}",
        f"Ground-truth EMPTY: {actual['EMPTY']}",
        f"Ground-truth OBJECT_PRESENT: {actual['OBJECT_PRESENT']}",
        f"Correct predictions: {correct}/{total}",
        f"Overall crop accuracy: {100.0 * correct / total:.2f}%",
        f"EMPTY recall: {recall['EMPTY']:.2f}%" if recall["EMPTY"] is not None else "EMPTY recall: N/A",
        f"OBJECT_PRESENT recall: {recall['OBJECT_PRESENT']:.2f}%" if recall["OBJECT_PRESENT"] is not None else "OBJECT_PRESENT recall: N/A",
        "Confusion matrix (rows = actual, columns = predicted):",
        "                    Pred EMPTY   Pred OBJECT_PRESENT",
        f"Actual EMPTY          {counts[('EMPTY','EMPTY')]:5d}       {counts[('EMPTY','OBJECT_PRESENT')]:5d}",
        f"Actual OBJECT_PRESENT {counts[('OBJECT_PRESENT','EMPTY')]:5d}       {counts[('OBJECT_PRESENT','OBJECT_PRESENT')]:5d}",
        f"All six correct in {sum(n == 6 for n in frames_correct.values())}/{len(frames)} frames",
        "LIMITATION: Six recurring pallets and multiple correlated views.",
        "LIMITATION: Manual boxes do not establish Phase 40 live pipeline accuracy.",
        "Original images, labels, checkpoints and Phase 40 were not changed.",
        f"Output: {run_dir}",
    ]
    text = "\n".join(report) + "\n"
    (run_dir / "results.txt").write_text(text, encoding="utf-8")
    (run_dir / "results.json").write_text(json.dumps({
        "frames": len(frames), "crops": total, "correct": correct,
        "accuracy_percent": 100.0 * correct / total,
        "ground_truth_counts": actual, "recall_percent": recall,
        "model_sha256": actual_hash,
        "matrix": {g: {p: counts[(g, p)] for p in CLASSES} for g in CLASSES},
        "human_ground_truth": str(GROUND_TRUTH),
    }, indent=2), encoding="utf-8")
    print("\n" + text)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--review", action="store_true", help="Label existing crops manually")
    mode.add_argument("--evaluate", action="store_true", help="Evaluate after review")
    args = parser.parse_args()
    frames = load_frames()
    print(f"Existing frames: {len(frames)}; existing pallet boxes: {len(frames)*6}")
    if args.review:
        review(frames)
    else:
        evaluate(frames)


if __name__ == "__main__":
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print(f"PHASE 60 STOPPED: {exc}", file=sys.stderr)
        sys.exit(1)
