"""
CREST Phase 60 -- ONE current camera frame, frozen EMPTY/OBJECT test.

Capture exactly ONE full-conveyor RGB frame, run the frozen 100-image
pallet detector and the frozen EMPTY/OBJECT classifier, and show/save
predictions for each detected pallet.  No video is recorded. No training,
PLC commands, dataset edits, or modifications to Phase 40.

This is a SINGLE-FRAME DIAGNOSTIC, not the full tracking/voting pipeline.
Predictions alone are NOT measured accuracy or ground-truth labels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(r"C:\Do_Not_Delete_PLC\Conference")
PALLET_MODEL = (
    ROOT / "00_Pallet_100_YOLO26_Training"
    / "pallet_yolo26_100_v2" / "weights" / "best.pt"
)
OBJECT_MODEL = (
    ROOT / "03_Empty_Object_Classifier" / "MODEL"
    / "empty_object_yolo26n_cls_best.pt"
)
EXPECTED_OBJECT_SHA256 = (
    "366ecc39ef8e6566774ac68200548218b51cc4246352e56301f8cd7e8cb31083"
)
OUTPUT_ROOT = (
    ROOT / "09_Independent_Blind_Test_Captures"
    / "_60_One_Frame_Live_Empty_Object_Test"
)

CAMERA_INDICES = (1, 0, 2, 3, 4)
FRAME_WIDTH = 1920
FRAME_HEIGHT = 1080
FRAME_FPS = 30
FRAME_EXPOSURE = -7.0
DETECT_IMGSZ = 960
DETECT_CONF = 0.10
DETECT_IOU = 0.60
CLASSIFY_IMGSZ = 320
CROP_PADDING = 0.03
PREVIEW_SIZE = (1280, 720)


# Same gamma/CLAHE pallet-view design used in the working CREST pipeline.
GAMMA_LUT = np.clip(
    255 * (np.arange(256, dtype=np.float32) / 255.0) ** 0.60,
    0, 255,
).astype(np.uint8)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def pallet_view(frame):
    corrected = cv2.LUT(frame, GAMMA_LUT)
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def open_camera():
    for camera_index in CAMERA_INDICES:
        # Windows DirectShow is usually more reliable for the CREST camera.
        candidate = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not candidate.isOpened():
            candidate.release()
            candidate = cv2.VideoCapture(camera_index)
        if not candidate.isOpened():
            candidate.release()
            continue
        candidate.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        candidate.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        candidate.set(cv2.CAP_PROP_FPS, FRAME_FPS)
        candidate.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        candidate.set(cv2.CAP_PROP_EXPOSURE, FRAME_EXPOSURE)
        ret, image = candidate.read()
        if ret and image is not None and image.size:
            print(
                f"Camera opened: index={camera_index}, "
                f"resolution={image.shape[1]}x{image.shape[0]}"
            )
            return candidate
        candidate.release()
    raise RuntimeError(
        "Cannot read the live camera. Close Phase 40 or other camera "
        "applications, then run Phase 60 again."
    )


def capture_one_frame():
    cap = open_camera()
    window = "CREST Phase 60 - SPACE: CAPTURE ONE FRAME   Q: EXIT"
    try:
        print("Preview is live; NO VIDEO WILL BE SAVED.")
        print("Press SPACE when all the pallets/objects are in view.")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, *PREVIEW_SIZE)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Live camera stopped supplying images.")
            preview = cv2.resize(frame, PREVIEW_SIZE)
            cv2.putText(
                preview, "SPACE = capture ONE frame | Q = quit",
                (22, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2, cv2.LINE_AA,
            )
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                return frame.copy()
            if key in (ord("q"), ord("Q"), 27):
                return None
    finally:
        cap.release()
        cv2.destroyAllWindows()


def padded_crop(image, box):
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    px = int((x2 - x1) * CROP_PADDING)
    py = int((y2 - y1) * CROP_PADDING)
    left = max(0, x1 - px)
    top = max(0, y1 - py)
    right = min(w, x2 + px)
    bottom = min(h, y2 + py)
    return image[top:bottom, left:right].copy()


def main():
    print("\nCREST PHASE 60 - LIVE SINGLE-FRAME EMPTY/OBJECT TEST")
    print("ONE frame only. Frozen models. No video, no training, no PLC.")
    for model_path in (PALLET_MODEL, OBJECT_MODEL):
        if not model_path.is_file():
            raise FileNotFoundError(f"Required model missing: {model_path}")
    actual_hash = sha256(OBJECT_MODEL)
    if actual_hash != EXPECTED_OBJECT_SHA256:
        raise RuntimeError(
            "The EMPTY/OBJECT checkpoint is different from Phase 59; "
            f"expected {EXPECTED_OBJECT_SHA256}, found {actual_hash}."
        )

    frame = capture_one_frame()
    if frame is None:
        print("Quit without capturing or saving anything.")
        return

    import torch
    device = 0 if torch.cuda.is_available() else "cpu"
    detector = YOLO(str(PALLET_MODEL))
    classifier = YOLO(str(OBJECT_MODEL))

    names = {int(k): str(v).upper() for k, v in classifier.names.items()}
    if set(names.values()) != {"EMPTY", "OBJECT_PRESENT"}:
        raise RuntimeError(f"Unexpected classification class names: {names}")
    empty_id = next(i for i, v in names.items() if v == "EMPTY")
    object_id = next(i for i, v in names.items() if v == "OBJECT_PRESENT")

    # The detector processes the full brightened conveyor image.
    bright = pallet_view(frame)
    det_result = detector.predict(
        source=bright, imgsz=DETECT_IMGSZ,
        conf=DETECT_CONF, iou=DETECT_IOU,
        max_det=30, device=device, verbose=False,
        augment=False,
    )[0]

    detections = []
    if det_result.boxes is not None:
        for box in det_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().tolist())
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            detections.append({
                "box": (x1, y1, x2, y2),
                "detector_confidence": float(box.conf[0].item()),
            })

    # Stable display order: top row left-to-right, then lower row.
    detections.sort(key=lambda d: (
        (d["box"][1] + d["box"][3]) // 180,
        (d["box"][0] + d["box"][2]) // 2,
    ))

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"run_{now}"
    run_dir.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(run_dir / "one_full_camera_frame.jpg"), frame)
    annotated = frame.copy()
    records = []

    if detections:
        crops = [padded_crop(bright, d["box"]) for d in detections]
        predictions = classifier.predict(
            source=crops, imgsz=CLASSIFY_IMGSZ,
            batch=min(len(crops), 8), device=device,
            augment=False, verbose=False,
        )
        if len(predictions) != len(detections):
            raise RuntimeError("Classification result count mismatch")

        for number, (detection, crop, prediction) in enumerate(
            zip(detections, crops, predictions), 1
        ):
            probs = prediction.probs
            if probs is None:
                raise RuntimeError("Model did not return class probabilities")
            probability = probs.data.detach().cpu().tolist()
            label = names[int(probs.top1)]
            score = float(probs.top1conf.item())
            x1, y1, x2, y2 = detection["box"]
            color = (50, 210, 50) if label == "EMPTY" else (0, 170, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            title = f"P{number} {label} {score:.2f}"
            text_y = max(30, y1 - 9)
            cv2.putText(
                annotated, title, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                color, 2, cv2.LINE_AA,
            )
            cv2.imwrite(str(run_dir / f"pallet_{number:02d}_input.jpg"), crop)
            records.append({
                "pallet_number_in_this_frame": number,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "detector_confidence": round(detection["detector_confidence"], 6),
                "prediction": label,
                "predicted_confidence": round(score, 6),
                "prob_empty": round(float(probability[empty_id]), 6),
                "prob_object_present": round(float(probability[object_id]), 6),
            })
            print(
                f"P{number}: {label:<14} "
                f"confidence={score:.2%} "
                f"detector={detection['detector_confidence']:.2%}"
            )

    empty_count = sum(r["prediction"] == "EMPTY" for r in records)
    object_count = sum(r["prediction"] == "OBJECT_PRESENT" for r in records)
    print("\n--- CURRENT FULL-CONVEYOR FRAME ---")
    print(f"Pallets detected:               {len(records)}")
    print(f"Predicted EMPTY:                {empty_count}")
    print(f"Predicted OBJECT_PRESENT:       {object_count}")
    print("Known physical setup from user: approximately 2 object pallets.")
    print("This is a model prediction, NOT ground-truth accuracy.")
    if len(records) != 6:
        print("WARNING: Detector did not return exactly 6 pallets.")
    if object_count != 2:
        print("CHECK: Predicted OBJECT_PRESENT count differs from expected 2.")

    cv2.imwrite(str(run_dir / "one_frame_predictions.jpg"), annotated)
    summary = {
        "test_type": "current live camera, one full frame, no video",
        "models_frozen": True,
        "not_end_to_end": True,
        "accuracy_measured": False,
        "ground_truth_not_collected": True,
        "classify_imgsz": CLASSIFY_IMGSZ,
        "detect_imgsz": DETECT_IMGSZ,
        "object_checkpoint_sha256": actual_hash,
        "pallet_count_detected": len(records),
        "predicted_empty": empty_count,
        "predicted_object_present": object_count,
        "per_pallet": records,
    }
    (run_dir / "results.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if records:
        with (run_dir / "per_pallet_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    print(f"Saved result: {run_dir / 'one_frame_predictions.jpg'}")
    print(f"Saved table:  {run_dir / 'per_pallet_predictions.csv'}")
    print("Press any key in the result image to close it.")
    cv2.namedWindow("CREST Phase 60 - ONE FRAME PREDICTIONS", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CREST Phase 60 - ONE FRAME PREDICTIONS", *PREVIEW_SIZE)
    cv2.imshow("CREST Phase 60 - ONE FRAME PREDICTIONS", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"PHASE 60 ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
