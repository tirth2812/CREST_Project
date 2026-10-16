"""CREST Phase 61 v2: read-only RFID1 -> live YOLO26 track mapping experiment.

- Existing Phase 40 and model checkpoints are NEVER edited or imported.
- PLC access is READ ONLY, RFID1 words 96-99. No actuator controls.
- Shows RFID physical ID 1..6 only for unique geometry/time matches.
- Other tracks remain UNKNOWN. Mapping needs physical verification.
- Uses frozen detector and EMPTY/OBJECT classifier for diagnostic labels,
  NOT the Phase 40 full tracking / X voting / final ACCEPTED/REJECTED logic.
- No video is saved; only optional snapshots and a CSV event audit.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"C:\Do_Not_Delete_PLC\Conference")
CODE_DIR = ROOT / "07_Final_Empty_Accepted_Rejected_System" / "CODE"
CONFIG = ROOT / "07_Final_Empty_Accepted_Rejected_System" / "CONFIG" / "rfid_stations.json"
DETECTOR = ROOT / "00_Pallet_100_YOLO26_Training" / "pallet_yolo26_100_v2" / "weights" / "best.pt"
CLASSIFIER = ROOT / "03_Empty_Object_Classifier" / "MODEL" / "empty_object_yolo26n_cls_best.pt"
RESULTS = ROOT / "07_Final_Empty_Accepted_Rejected_System" / "RESULTS" / "61_RFID1_Vision_Mapping_Readonly"

PLC_IP = "192.168.82.231"
RFID1_TAGS = [f"IO_Link_Master_ST1_to_ST4:I1.Data[{i}]" for i in range(96, 100)]
RFID_MAP = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}
NO_TAG = (0, 0, 0, 0)
RPM_MOTOR, RPM_DRIVE = 1561.0, 35.0
CAMERA_INDEXES = (1, 0, 2, 3, 4)
CAMERA_W, CAMERA_H = 1920, 1080
SENSOR_BOX = (1123, 578, 1183, 675)  # prior manually verified RFID1 ROI
SENSOR_X_GAP = 125.0  # tentative pixel gate; verify against logged evidence
SENSOR_Y_GAP = 85.0
MATCH_MARGIN = 50.0  # reject competing plausible candidates
# RFID can appear before the pallet reaches the center of its marked station.
# RFID1 dwell was historically about 1.34s, so do not restrict to +/-0.45s.
ASSOCIATION_WAIT_SEC = 2.0
ASSOCIATION_LOOKBACK_SEC = 0.50
# Drawing-only persistence: does NOT modify YOLO detection/tracking predictions.
BOX_DISPLAY_HOLD_SEC = 1.25
VIEW_W, VIEW_H = 1280, 720

_G = np.arange(256, dtype=np.float32) / 255.0
GAMMA_LUT = np.clip(255 * _G ** 0.60, 0, 255).astype(np.uint8)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance(frame):
    lab = cv2.cvtColor(cv2.LUT(frame, GAMMA_LUT), cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = CLAHE.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def open_camera():
    for idx in CAMERA_INDEXES:
        for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            cap.set(cv2.CAP_PROP_EXPOSURE, -7)
            ok, frame = cap.read()
            if ok and frame is not None and frame.size:
                print(f"Camera index={idx}, resolution={frame.shape[1]}x{frame.shape[0]}")
                return cap
            cap.release()
    raise RuntimeError("Cannot read camera; close any other camera program first.")


def station_box(width, height):
    raw = SENSOR_BOX
    if CONFIG.is_file():
        try:
            obj = json.loads(CONFIG.read_text(encoding="utf-8"))
            box = obj["rfid1"]["box"]
            raw = tuple(int(box[k]) for k in ("x1", "y1", "x2", "y2"))
            source_w = int(obj.get("image_width", CAMERA_W))
            source_h = int(obj.get("image_height", CAMERA_H))
            return (int(raw[0] * width / source_w),
                    int(raw[1] * height / source_h),
                    int(raw[2] * width / source_w),
                    int(raw[3] * height / source_h))
        except (ValueError, KeyError, TypeError, ZeroDivisionError) as exc:
            print(f"Invalid station JSON ({exc}); using recorded RFID1 box.")
    return (int(raw[0] * width / CAMERA_W),
            int(raw[1] * height / CAMERA_H),
            int(raw[2] * width / CAMERA_W),
            int(raw[3] * height / CAMERA_H))


def read_rfid1(comm):
    """READ ONLY. Try one batched read; fall back to four single-tag reads.

    The legacy working PLC reader used separate comm.Read(tag) calls. Some
    pylogix/PLC installations behave differently for batch-read requests.
    No PLC writes are performed.
    """
    batch_reason = ""
    try:
        responses = comm.Read(RFID1_TAGS)
        if isinstance(responses, (tuple, list)) and len(responses) == 4:
            values = []
            for resp in responses:
                if getattr(resp, "Status", None) != "Success" or getattr(resp, "Value", None) is None:
                    raise ValueError(f"batch status={getattr(resp, 'Status', None)}")
                values.append(int(resp.Value))
            return tuple(values), "BATCH"
        batch_reason = f"bad batch shape {type(responses).__name__}"
    except Exception as exc:
        batch_reason = f"batch failed: {exc}"

    try:
        values = []
        for tag in RFID1_TAGS:
            response = comm.Read(tag)
            if getattr(response, "Status", None) != "Success" or getattr(response, "Value", None) is None:
                raise ValueError(f"{tag}: {getattr(response, 'Status', None)}")
            values.append(int(response.Value))
        return tuple(values), "SINGLE"
    except Exception as exc:
        raise RuntimeError(f"{batch_reason}; single failed: {exc}") from exc


def rfid_worker(stop, events, stats):
    try:
        from pylogix import PLC
    except ImportError:
        stats["error"] = "pylogix missing: python -m pip install pylogix"
        print("RFID1 ERROR:", stats["error"], flush=True)
        return
    last = NO_TAG
    consecutive_empty = 0
    last_error_print = 0.0
    try:
        with PLC() as comm:
            comm.IPAddress = PLC_IP
            comm.SocketTimeout = 3
            while not stop.is_set():
                try:
                    raw, method = read_rfid1(comm)
                except Exception as exc:
                    stats["connected"] = False
                    stats["failures"] += 1
                    stats["error"] = f"PLC read: {str(exc)[:160]}"
                    now = time.monotonic()
                    if now - last_error_print > 3.0:
                        print("RFID1 ERROR:", stats["error"], flush=True)
                        last_error_print = now
                    stop.wait(0.20)
                    continue
                if not stats["connected"]:
                    print(f"RFID1 PLC ONLINE via {method}. Waiting for a tagged pallet.", flush=True)
                stats["connected"] = True
                stats["error"] = ""
                stats["read_method"] = method
                stats["last_raw"] = raw
                stats["reads"] += 1
                if raw == NO_TAG:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        last = NO_TAG
                else:
                    consecutive_empty = 0
                    if raw != last:
                        physical_id = RFID_MAP.get(raw)
                        event = {"time": time.monotonic(), "pallet": physical_id,
                                 "raw": raw}
                        events.put(event)
                        print(f"RFID1 read: physical={physical_id or 'UNKNOWN TAG'} raw={raw}", flush=True)
                        last = raw
                stop.wait(0.035)
    except Exception as exc:
        stats["connected"] = False
        stats["error"] = f"PLC connection: {exc}"
        print("RFID1 ERROR:", stats["error"], flush=True)


def remove_duplicate_boxes(boxes):
    """Suppress near-identical detections, preferring higher confidence."""
    kept = []
    for obj in sorted(boxes, key=lambda b: b["confidence"], reverse=True):
        ax1, ay1, ax2, ay2 = obj["box"]
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        duplicate = False
        for seen in kept:
            bx1, by1, bx2, by2 = seen["box"]
            inter = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            iou = inter / max(1, area_a + area_b - inter)
            smaller_overlap = inter / max(1, min(area_a, area_b))
            if iou > 0.25 or smaller_overlap > 0.55:
                duplicate = True
                break
        if not duplicate:
            kept.append(obj)
    return kept


def candidates_for_event(tracks, sensor, width, height):
    """Test where RFID1 can plausibly see a right-to-left pallet.

    RFID event occurs when the leading/left side of the inbound pallet is
    close to the reader. A pallet already far left of the reader is excluded.
    This is a CALIBRATION HYPOTHESIS, not proof of correct association.
    """
    sx = (sensor[0] + sensor[2]) / 2
    sy = (sensor[1] + sensor[3]) / 2
    x_limit = SENSOR_X_GAP * width / CAMERA_W
    y_limit = SENSOR_Y_GAP * height / CAMERA_H
    choices = []
    for item in tracks:
        x1, y1, x2, y2 = item["box"]
        # Pallet must intersect the reader's vertical neighborhood.
        vertical_gap = max(0.0, y1 - sy, sy - y2)
        if vertical_gap > y_limit:
            continue
        # RFID1 is on upper left-moving belt; leading edge x1 arrives first.
        left_edge_gap = abs(x1 - sx)
        if left_edge_gap > x_limit:
            continue
        if item["vx"] > 100.0:  # reliably moving RIGHT is incompatible
            continue
        score = left_edge_gap + 0.6 * vertical_gap
        choices.append((round(score, 2), item["id"]))
    return sorted(choices)


def select_match(tracks, sensor, width, height):
    scored = candidates_for_event(tracks, sensor, width, height)
    if not scored:
        return None, "NO_TRACK_NEAR_READER", scored
    gap_margin = MATCH_MARGIN * width / CAMERA_W
    if len(scored) >= 2 and scored[1][0] - scored[0][0] < gap_margin:
        return None, "AMBIGUOUS_MULTIPLE_TRACKS", scored
    return scored[0][1], "PROVISIONAL_GEOMETRY_MATCH", scored


def match_event_from_history(evt, snapshots, sensor, width, height):
    """Use the full RFID-to-camera delay window, not only the closest frame.

    Return only a unique, plausible station passage, otherwise UNKNOWN.
    This is a provisional association, not a verified physical identity.
    """
    options = {}
    for timestamp, tracks in snapshots:
        if not (evt["time"] - ASSOCIATION_LOOKBACK_SEC <= timestamp
                <= evt["time"] + ASSOCIATION_WAIT_SEC):
            continue
        for score, tid in candidates_for_event(tracks, sensor, width, height):
            previous = options.get(tid)
            if previous is None or score < previous:
                options[tid] = score
    scored = sorted((round(score, 2), tid) for tid, score in options.items())
    if not scored:
        return None, "NO_TRACK_NEAR_READER", scored
    if len(scored) > 1:
        gap = MATCH_MARGIN * width / CAMERA_W
        if scored[1][0] - scored[0][0] < gap:
            return None, "AMBIGUOUS_MULTIPLE_TRACKS", scored
    return scored[0][1], "PROVISIONAL_TIME_GEOMETRY_MATCH", scored


def build_display_objects(found, cache, now):
    """Drawing-only box hold for short missed detections (not tracker logic)."""
    live_ids = {obj["id"] for obj in found}
    for obj in found:
        cache[obj["id"]] = (dict(obj), now)
    visible = [dict(obj, held=False) for obj in found]
    live_boxes = [obj["box"] for obj in found]
    for tid, (item, last_seen) in list(cache.items()):
        age = now - last_seen
        if age > BOX_DISPLAY_HOLD_SEC:
            del cache[tid]
            continue
        if tid in live_ids:
            continue
        # Do not draw a stale box on top of a different current detection.
        ax1, ay1, ax2, ay2 = item["box"]
        held_area = max(1, (ax2-ax1)*(ay2-ay1))
        overlapping_live = False
        for bx1, by1, bx2, by2 in live_boxes:
            intersect = max(0, min(ax2, bx2)-max(ax1, bx1))*max(0, min(ay2, by2)-max(ay1, by1))
            if intersect / held_area > 0.45:
                overlapping_live = True
                break
        if not overlapping_live:
            visible.append(dict(item, held=True))
    return visible


def crop_for_classifier(frame, box):
    x1, y1, x2, y2 = box
    pad_x = int(0.03 * (x2 - x1))
    pad_y = int(0.03 * (y2 - y1))
    return frame[max(0, y1-pad_y):min(frame.shape[0], y2+pad_y),
                 max(0, x1-pad_x):min(frame.shape[1], x2+pad_x)].copy()


def check_rfid_only():
    """Read-only PLC diagnostic, no camera or trained models needed."""
    try:
        from pylogix import PLC
    except ImportError as exc:
        raise RuntimeError("Missing pylogix: python -m pip install pylogix") from exc
    print(f"RFID1 READ-ONLY CHECK: PLC {PLC_IP}, words 96-99", flush=True)
    with PLC() as comm:
        comm.IPAddress = PLC_IP
        comm.SocketTimeout = 3
        for i in range(5):
            try:
                words, method = read_rfid1(comm)
                physical = RFID_MAP.get(words)
                state = (f"PALLET {physical}" if physical is not None else
                         "NO TAG" if words == NO_TAG else "UNRECOGNIZED TAG")
                print(f"Read {i+1}/5: {state} {words} via {method}", flush=True)
            except Exception as exc:
                print(f"Read {i+1}/5 ERROR: {exc}", flush=True)
            if i < 4:
                time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="test matching rules without camera/PLC")
    parser.add_argument("--check-rfid", action="store_true", help="read RFID1 five times, no camera")
    args = parser.parse_args()
    if args.check_rfid:
        check_rfid_only()
        return
    if args.self_test:
        fake = [
            {"id": 4, "box": (1182, 488, 1345, 602), "vx": -280.0},
            {"id": 5, "box": (938, 499, 1099, 603), "vx": -275.0},
        ]
        got, why, options = select_match(fake, SENSOR_BOX, CAMERA_W, CAMERA_H)
        print(f"SELF TEST: match={got} status={why} candidates={options}")
        assert got == 4
        got2, why2, _ = select_match([], SENSOR_BOX, CAMERA_W, CAMERA_H)
        assert got2 is None and why2 == "NO_TRACK_NEAR_READER"
        from types import SimpleNamespace
        class MockPLC:
            def Read(self, arg):
                if isinstance(arg, list):
                    return SimpleNamespace(Status="Error", Value=None)
                idx = RFID1_TAGS.index(arg)
                return SimpleNamespace(Status="Success", Value=RFID1_WORDS[idx])
        RFID1_WORDS = (-8188, 336, -18499, 21259)
        value, method = read_rfid1(MockPLC())
        assert value == RFID1_WORDS and method == "SINGLE"
        event = {"time": 10.0, "pallet": 3}
        fake_snapshot = [(10.1, []), (10.85, fake)]
        found_id, found_status, _ = match_event_from_history(
            event, fake_snapshot, SENSOR_BOX, CAMERA_W, CAMERA_H)
        assert found_id == 4 and found_status == "PROVISIONAL_TIME_GEOMETRY_MATCH"
        persist = {}
        drawn = build_display_objects(fake, persist, 100.0)
        assert len(drawn) == 2 and not any(x["held"] for x in drawn)
        drawn = build_display_objects([], persist, 100.5)
        assert len(drawn) == 2 and all(x["held"] for x in drawn)
        assert not build_display_objects([], persist, 102.0)
        print("SELF TEST PASSED: read fallback, time matching, drawing hold (synthetic)")
        return

    from ultralytics import YOLO
    import torch
    for path in (DETECTOR, CLASSIFIER):
        if not path.is_file():
            raise FileNotFoundError(f"Frozen model missing: {path}")
    print("CREST PHASE 61 - RFID1 PHYSICAL PALLET MAPPING (READ ONLY)")
    print("PLC reads RFID1 only. No PLC writes, motor or piston control.")
    print("Motor RPM 1561, drive RPM 35 => ratio 44.6:1.")
    print("Roller diameter unknown: NO assumed mm/s belt speed.")
    print("Computer vision movement/timestamps used for association.")
    print("Matching is EXPERIMENTAL; verify physical labels on real pallets.")
    gpu = 0 if torch.cuda.is_available() else "cpu"
    detector = YOLO(str(DETECTOR))
    classifier = YOLO(str(CLASSIFIER))
    names = {int(k): str(v).upper() for k, v in classifier.names.items()}
    if set(names.values()) != {"EMPTY", "OBJECT_PRESENT"}:
        raise RuntimeError(f"Unexpected classifier classes: {names}")
    cap = open_camera()
    run_dir = RESULTS / ("run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=False)
    csv_file = (run_dir / "rfid1_association_events.csv").open("w", newline="", encoding="utf-8")
    columns = ["time", "rfid_pallet", "raw_words", "matched_track", "status", "candidates"]
    writer = csv.DictWriter(csv_file, fieldnames=columns)
    writer.writeheader()

    stop = threading.Event()
    events = queue.Queue()
    stats = {"connected": False, "failures": 0, "error": "",
             "read_method": "-", "reads": 0, "last_raw": None}
    thread = threading.Thread(target=rfid_worker, args=(stop, events, stats), daemon=True)
    thread.start()
    identities = {}         # temporary ByteTrack id => physical pallet ID
    label_cache = {}        # temporary ByteTrack id => (classification, confidence, timestamp)
    history = {}            # track id => (cx, monotonic time)
    snapshots = deque(maxlen=90)
    display_cache = {}  # only for visible overlays, not inference/matching
    pending = []
    latest_event = "Waiting for RFID1"
    count = 0
    window = "CREST Phase 61: RFID1 read-only | S snapshot | Q quit"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, VIEW_W, VIEW_H)
    try:
        while True:
            ok, frame = cap.read()
            captured_at = time.monotonic()
            if not ok or frame is None:
                raise RuntimeError("Camera stopped returning frames")
            height, width = frame.shape[:2]
            sensor = station_box(width, height)
            bright = enhance(frame)
            prediction = detector.track(
                bright, imgsz=960, conf=0.10, iou=0.60,
                persist=True, tracker="bytetrack.yaml", max_det=30,
                device=gpu, verbose=False,
            )[0]
            found = []
            if prediction.boxes is not None and prediction.boxes.id is not None:
                for box in prediction.boxes:
                    obj_id = int(box.id.item())
                    bb = tuple(int(n) for n in box.xyxy[0].cpu().tolist())
                    if bb[2]-bb[0] < 15 or bb[3]-bb[1] < 15:
                        continue
                    cx = 0.5 * (bb[0] + bb[2])
                    last = history.get(obj_id)
                    vx = (cx-last[0]) / max(0.001, captured_at-last[1]) if last else 0.0
                    history[obj_id] = (cx, captured_at)
                    found.append({"id": obj_id, "box": bb,
                                  "confidence": float(box.conf.item()), "vx": vx})
            found = remove_duplicate_boxes(found)
            snapshots.append((captured_at, [dict(b) for b in found]))
            while True:
                try:
                    pending.append(events.get_nowait())
                except queue.Empty:
                    break
            remains = []
            for evt in pending:
                age = time.monotonic() - evt["time"]
                # RFID1 typically activates BEFORE the pallet center reaches
                # the camera's station mark. Accumulate passage observations.
                if age < ASSOCIATION_WAIT_SEC:
                    remains.append(evt)
                    continue
                if evt["pallet"] is None:
                    matched, status, candidates = None, "UNRECOGNIZED_RFID", []
                else:
                    matched, status, candidates = match_event_from_history(
                        evt, snapshots, sensor, width, height)
                    if matched is not None:
                        old_identity = identities.get(matched)
                        conflicting_track = next((k for k, v in identities.items()
                                                  if v == evt["pallet"] and k != matched
                                                  and any(o["id"] == k for o in found)), None)
                        if old_identity is not None and old_identity != evt["pallet"]:
                            matched, status = None, "CONFLICT_EXISTING_TRACK_ID"
                        elif conflicting_track is not None:
                            matched, status = None, "CONFLICT_DUPLICATE_PHYSICAL_ID"
                        else:
                            identities[matched] = evt["pallet"]
                latest_event = f"RFID {evt['pallet']} -> track {matched if matched is not None else 'UNKNOWN'} [{status}]"
                print(latest_event, flush=True)
                writer.writerow({"time": datetime.now().isoformat(timespec="milliseconds"),
                                 "rfid_pallet": evt["pallet"], "raw_words": str(evt["raw"]),
                                 "matched_track": matched, "status": status,
                                 "candidates": str(candidates)})
                csv_file.flush()
            pending = remains
            # Refresh the classification periodically; it is not Phase 40's voting logic.
            count += 1
            to_classify = [b for b in found if count % 5 == 0 or b["id"] not in label_cache]
            if to_classify:
                crops = [crop_for_classifier(bright, b["box"]) for b in to_classify]
                predictions = classifier.predict(crops, imgsz=320, device=gpu,
                                                 batch=min(8, len(crops)), verbose=False)
                for box, classification in zip(to_classify, predictions):
                    if classification.probs is not None:
                        cid = int(classification.probs.top1)
                        label_cache[box["id"]] = (names[cid],
                            float(classification.probs.top1conf.item()), captured_at)
            shown = frame.copy()
            cv2.rectangle(shown, sensor[:2], sensor[2:], (255, 0, 255), 2)
            cv2.putText(shown, "RFID1", (sensor[0], max(25, sensor[1]-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            # Drawing fallback only: no change to model/tracker outputs.
            display_objects = build_display_objects(found, display_cache, captured_at)
            for obj in display_objects:
                tid, (x1,y1,x2,y2) = obj["id"], obj["box"]
                physical = identities.get(tid)
                classification = label_cache.get(tid, ("CHECKING", 0, 0))[0]
                label = f"RFID {physical if physical is not None else 'UNKNOWN'} | T{tid} | {classification}"
                if obj.get("held"):
                    label += " [HOLD]"
                color = (0, 200, 0) if physical is not None else (0, 170, 255)
                cv2.rectangle(shown, (x1,y1), (x2,y2), color, 2)
                cv2.putText(shown, label, (x1, max(25, y1-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            if stats["connected"]:
                link_state = f"ONLINE({stats['read_method']})"
            elif stats["error"]:
                link_state = "READ ERROR - see CMD"
            else:
                link_state = "CONNECTING"
            header = f"RFID1 {link_state} | {latest_event[:55]}"
            cv2.rectangle(shown, (0, 0), (width, 55), (20, 20, 20), -1)
            cv2.putText(shown, header, (12, 38), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255,255,255), 2, cv2.LINE_AA)
            preview = cv2.resize(shown, (VIEW_W, VIEW_H))
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xff
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("s"), ord("S")):
                out = run_dir / ("mapping_snapshot_" + datetime.now().strftime("%H%M%S") + ".jpg")
                cv2.imwrite(str(out), shown)
                print(f"Saved snapshot: {out}")
            # The RFID worker prints a throttled diagnostic immediately on failure.
    finally:
        stop.set()
        cap.release()
        cv2.destroyAllWindows()
        thread.join(timeout=4)
        csv_file.close()
        print("RFID mapping CSV:", run_dir / "rfid1_association_events.csv")
        print("Remember: geometry matches are provisional until physically checked.")


if __name__ == "__main__":
    main()
