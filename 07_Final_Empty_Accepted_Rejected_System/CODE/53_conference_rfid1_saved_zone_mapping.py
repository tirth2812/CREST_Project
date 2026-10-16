"""CREST Phase 53 - saved RFID1 TOP-zone to moving-pallet box association.

Experimental, READ-ONLY hardware observer. Does NOT modify Phase 40/52 or model
weights and does NOT read RFID2. An RFID~N overlay is a hypothesis; it is NOT
independent proof of the pallet's physical identity.

Reuses the RFID event decoder and visual motion tracker of the working Phase 52,
but removes its 13-read/two-lap/pallet-spacing requirement.
"""
from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from queue import Empty, Queue
from threading import Event

import cv2

CODE_DIR = Path(__file__).resolve().parent
SYSTEM_DIR = CODE_DIR.parent
BASE_FILE = CODE_DIR / "40_conference_100_pallet_live.py"
TRACKER_FILE = CODE_DIR / "52_conference_rfid1_pattern_mapping.py"
RESULT_DIR = SYSTEM_DIR / "RESULTS" / "53_RFID1_Saved_Zone_Mapping"

# This is the SAVED hardware bounding box, not a proven pallet tag center.
# Check the camera overlay before trusting any predicted mapping.
RFID_STATION_CANDIDATES = [
    SYSTEM_DIR / "CONFIG" / "rfid_stations.json",
    Path(r"C:\Do_Not_Delete_PLC\CREST_YOLO26\07_Final_Empty_Accepted_Rejected_System\CONFIG\rfid_stations.json"),
    Path(r"C:\Do_Not_Delete_PLC\00_CREST_Main_Project_Report_Files\07_Final_Empty_Accepted_Rejected_System\CONFIG\rfid_stations.json"),
    Path(r"C:\Do_Not_Delete_PLC\FINAL\07_Final_Empty_Accepted_Rejected_System\CONFIG\rfid_stations.json"),
]

# Adjustable matching safeguards. These are exploratory, not calibrated.
POST_READ_WAIT_SEC = 0.65
CAMERA_SAMPLE_MAX_DELTA_SEC = 0.35
OBSERVATION_HISTORY_SEC = 12.0
MIN_CAMERA_OBSERVATIONS = 3
MAX_READER_TO_PALLET_EDGE_PX = 185.0
UNIQUE_MATCH_MARGIN_PX = 85.0
MIN_TOP_LANE_PAD_ABOVE = 150
MIN_TOP_LANE_PAD_BELOW = 25
EVENT_TIMEOUT_SEC = 4.0


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_station_config():
    override = os.environ.get("CREST_RFID_STATIONS_FILE", "").strip()
    paths = [Path(override)] if override else RFID_STATION_CANDIDATES
    for filename in paths:
        if not filename.is_file():
            continue
        data = json.loads(filename.read_text(encoding="utf-8-sig"))
        entry = data.get("rfid1")
        if not isinstance(entry, dict):
            raise ValueError(f"rfid1 region not found in {filename}")
        if entry.get("location", "TOP").upper() != "TOP":
            raise ValueError(f"Saved RFID1 location is NOT TOP in {filename}; refusing to guess")
        direction = entry.get("direction", "RIGHT_TO_LEFT").upper()
        if direction != "RIGHT_TO_LEFT":
            raise ValueError(f"Unexpected RFID1 direction {direction} in {filename}")
        rect = entry.get("box", {})
        coords = tuple(float(rect[key]) for key in ("x1", "y1", "x2", "y2"))
        if coords[0] >= coords[2] or coords[1] >= coords[3]:
            raise ValueError(f"Invalid saved RFID1 coordinates: {coords}")
        width = float(data.get("image_width", 1920))
        height = float(data.get("image_height", 1080))
        if width <= 0 or height <= 0:
            raise ValueError("Saved RFID station dimensions must be positive")
        return filename, coords, width, height
    attempted = "\n".join(str(item) for item in paths)
    raise FileNotFoundError(
        "Cannot find the previously saved rfid_stations.json. "
        "Please locate it; do NOT guess a new RFID box. Checked:\n" + attempted
    )


def scaled_rect(rect, original_w, original_h, current_w, current_h):
    sx = current_w / original_w
    sy = current_h / original_h
    return tuple(int(round(v * (sx if i % 2 == 0 else sy)))
                 for i, v in enumerate(rect))


def observation_at(samples, event_time):
    """Return interpolated x/y and box with an evidence time error, if possible."""
    if not samples:
        return None
    ordered = list(samples)
    before = [sample for sample in ordered if sample[0] <= event_time]
    after = [sample for sample in ordered if sample[0] >= event_time]
    if before and after:
        a = before[-1]
        b = after[0]
        if a[0] == b[0]:
            return (a[1], a[2], a[3], 0.0)
        if (event_time - a[0] <= CAMERA_SAMPLE_MAX_DELTA_SEC
                and b[0] - event_time <= CAMERA_SAMPLE_MAX_DELTA_SEC):
            ratio = (event_time - a[0]) / (b[0] - a[0])
            x = a[1] + ratio * (b[1] - a[1])
            y = a[2] + ratio * (b[2] - a[2])
            nearer = a if ratio < 0.5 else b
            return (x, y, nearer[3], max(event_time - a[0], b[0] - event_time))
    nearest = min(ordered, key=lambda sample: abs(sample[0] - event_time))
    difference = abs(nearest[0] - event_time)
    if difference > CAMERA_SAMPLE_MAX_DELTA_SEC:
        return None
    return (nearest[1], nearest[2], nearest[3], difference)


def rank_candidates(event_time, history, rect, scale=1.0):
    """Use reader X + TOP conveyor lane, not hardware Y overlap as a box center."""
    x1, y1, x2, y2 = rect
    antenna_x = (x1 + x2) / 2.0
    hardware_h = max(1, y2 - y1)
    top_min = y1 - max(MIN_TOP_LANE_PAD_ABOVE * scale, 1.7 * hardware_h)
    top_max = y2 + max(MIN_TOP_LANE_PAD_BELOW * scale, 0.35 * hardware_h)
    candidates = []
    for uid, samples in history.items():
        if len(samples) < MIN_CAMERA_OBSERVATIONS:
            continue
        prediction = observation_at(samples, event_time)
        if prediction is None:
            continue
        x, y, box, uncertainty = prediction
        # Exclude bottom-conveyor detections regardless of the RFID hardware
        # rectangle's overlap with them.
        if not top_min <= y <= top_max:
            continue
        left, _, right, _ = box
        # The tag may not be at the pallet bounding-box CENTER. Compare
        # the antenna X against the visible pallet's horizontal extent.
        outside_distance = max(left - antenna_x, antenna_x - right, 0.0)
        max_outside = MAX_READER_TO_PALLET_EDGE_PX * scale
        if outside_distance > max_outside:
            continue
        candidates.append({
            "uid": uid,
            "center_x": round(float(x), 1),
            "center_y": round(float(y), 1),
            "box": list(box),
            "outside_distance_px": round(outside_distance, 1),
            "frame_time_error_s": round(float(uncertainty), 3),
            "score": round(float(outside_distance + uncertainty * 65 * scale), 2),
        })
    candidates.sort(key=lambda item: (item["score"], item["uid"]))
    return candidates


class CameraProxy:
    def __init__(self, original, exp):
        self.original = original
        self.exp = exp

    def read(self):
        ok, frame = self.original.read()
        if ok:
            self.exp.last_frame_time = time.monotonic()
        return ok, frame

    def __getattr__(self, name):
        return getattr(self.original, name)


class Experiment:
    def __init__(self, helpers):
        self.helpers = helpers
        self.config_path, self.rect, self.config_w, self.config_h = load_station_config()
        self.ratio = 1.0
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = RESULT_DIR / f"phase53_{stamp}.csv"
        self.file = self.csv_path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=[
            "wall_time", "tag", "raw_words", "first_valid_mono",
            "confirmed_mono", "status", "assigned_visual_uid",
            "candidate_json", "note",
        ])
        self.writer.writeheader()
        self.queue = Queue()
        self.stop = Event()
        self.reader = helpers.RFIDReader(self.queue, self.stop)
        self.path = helpers.ConveyorPath(
            SYSTEM_DIR / "CONFIG" / "conveyor_boundary.json")
        self.visual = helpers.VisualTracker(self.path)
        self.history = defaultdict(lambda: deque(maxlen=300))
        self.pending = deque()
        self.rfid_for_uid = {}        # motion UID -> (RFID number, last event time)
        self.uid_for_rfid = {}       # RFID number -> motion UID
        self.last_frame_time = None
        self.last_frame_update = -1
        self.last_status = "WAIT_RFID1"
        self.last_warning = False
        self.latest_tag = None
        self.last_log_time = 0.0

    def start(self):
        print("PHASE 53 COMPLETE SCRIPT ACTIVE - SYNTAX VERIFIED")
        print("PHASE 53: SAVED RFID1 TOP STATION / READ ONLY")
        print("RFID1=TOP (right to left); RFID2=BOTTOM (not polled)")
        print("SAVED CONFIG:", self.config_path)
        print("SAVED RFID1 RECT:", self.rect)
        if self.config_path != RFID_STATION_CANDIDATES[0]:
            print("WARNING: RFID station box loaded from older project copy;")
            print("verify its drawn position matches your CURRENT camera.")
        print("RFID~N = TENTATIVE, NOT PHYSICALLY VERIFIED")
        print("No two-lap wait; no assumptions about fixed pallet spacing.")
        print("CSV:", self.csv_path)
        self.reader.start()

    def collect(self):
        while True:
            try:
                event = self.queue.get_nowait()
            except Empty:
                break
            self.pending.append(event)
            t0, t1, number, words = event
            self.latest_tag = number
            self.last_status = f"RFID1 TAG {number}: ALIGNING CAMERA"
            print(f"RFID1 tag={number} at={t0:.3f} pending_match")

    def log(self, event, status, uid=None, candidates=None, note=""):
        t0, t1, number, words = event
        self.writer.writerow({
            "wall_time": time.strftime("%H:%M:%S"),
            "tag": number,
            "raw_words": repr(words),
            "first_valid_mono": f"{t0:.6f}",
            "confirmed_mono": f"{t1:.6f}",
            "status": status,
            "assigned_visual_uid": "" if uid is None else uid,
            "candidate_json": json.dumps(candidates or []),
            "note": note,
        })
        self.file.flush()
        self.last_status = f"RFID1 {number}: {status}"
        print(self.last_status + (f" uid={uid}" if uid is not None else ""))

    def process_event(self, event, picture):
        t0, t1, number, words = event
        height, width = picture.shape[:2]
        rect = scaled_rect(self.rect, self.config_w, self.config_h, width, height)
        scale = width / self.config_w
        candidates = rank_candidates(t0, self.history, rect, scale)
        if not candidates:
            self.log(event, "NO_UNIQUE_TOP_PALLET", candidates=candidates,
                     note="No sufficiently recent visual track in saved RFID1 X corridor")
            return
        if len(candidates) > 1 and (candidates[1]["score"] - candidates[0]["score"]
                                    < UNIQUE_MATCH_MARGIN_PX * scale):
            self.log(event, "AMBIGUOUS_TOP_PALLETS", candidates=candidates)
            return
        selected = candidates[0]
        uid = selected["uid"]
        old_tag = self.rfid_for_uid.get(uid)
        if old_tag is not None and old_tag[0] != number:
            self.log(event, "CONFLICT_SAME_VISUAL_TRACK", candidates=candidates,
                     note=f"UID {uid} previously tentatively mapped to RFID {old_tag[0]}")
            return
        old_uid = self.uid_for_rfid.get(number)
        if old_uid is not None and old_uid != uid and old_uid in self.visual.tracks:
            self.log(event, "CONFLICT_DUPLICATE_RFID", candidates=candidates,
                     note=f"RFID {number} already attached to live UID {old_uid}")
            return
        self.rfid_for_uid[uid] = (number, t0)
        self.uid_for_rfid[number] = uid
        self.log(event, "TENTATIVE_ZONE_MATCH", uid=uid, candidates=candidates,
                 note="Needs independent physical verification of reader/pallet offset")

    def show(self, title, frame, original_show):
        caller = inspect.currentframe().f_back
        variables = {}
        for _ in range(7):
            if caller is None:
                break
            if "confirmed_tracks" in caller.f_locals:
                variables = caller.f_locals
                break
            caller = caller.f_back
        self.collect()
        now = self.last_frame_time
        image = frame.copy()
        live = []
        if now is not None and now > self.last_frame_update:
            tracks = variables.get("confirmed_tracks")
            if isinstance(tracks, (list, tuple)):
                self.last_frame_update = now
                live = self.visual.update(now, tracks)
                for item in live:
                    left, top, right, bottom = item.box
                    center = ((left + right) / 2, (top + bottom) / 2)
                    self.history[item.uid].append((now, center[0], center[1], item.box))
                for uid, history in list(self.history.items()):
                    if history and now - history[-1][0] > OBSERVATION_HISTORY_SEC:
                        del self.history[uid]
                for uid in list(self.rfid_for_uid):
                    if uid not in self.visual.tracks:
                        n = self.rfid_for_uid.pop(uid)[0]
                        if self.uid_for_rfid.get(n) == uid:
                            del self.uid_for_rfid[n]
            elif not self.last_warning:
                self.last_warning = True
                print("WARNING: Phase 40 confirmed_tracks not available at imshow")

        # The saved rectangle is shown exactly, scaled from its saved image size.
        height, width = image.shape[:2]
        sx1, sy1, sx2, sy2 = scaled_rect(
            self.rect, self.config_w, self.config_h, width, height)
        cv2.rectangle(image, (sx1, sy1), (sx2, sy2), (0, 255, 255), 2)
        cv2.putText(image, "SAVED RFID1 TOP", (sx1, max(22, sy1 - 9)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 255), 2)
        for track in live:
            number_info = self.rfid_for_uid.get(track.uid)
            tag = f"RFID~{number_info[0]}" if number_info else "RFID ?"
            left, top, right, bottom = track.box
            # Phase 40's ACCEPTED/REJECTED/EMPTY label is untouched.
            cv2.putText(image, tag, (max(0, left), min(height - 10, bottom + 34)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.57,
                        (0, 255, 255) if number_info else (245, 245, 245), 2)

        current = time.monotonic()
        while self.pending and (
            now is not None and now >= self.pending[0][0] + POST_READ_WAIT_SEC
            or current - self.pending[0][0] >= EVENT_TIMEOUT_SEC
        ):
            event = self.pending.popleft()
            if now is None:
                self.log(event, "NO_CAMERA_TIMESTAMP")
            else:
                self.process_event(event, image)
        cv2.rectangle(image, (4, height - 58),
                      (min(width - 5, 1230), height - 3), (0, 0, 0), -1)
        cv2.putText(image, "PHASE 53 | RFID1 TOP SAVED BOX | NO RFID2 | READ ONLY",
                    (12, height - 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (255, 255, 255), 2)
        cv2.putText(image, self.last_status[:110], (12, height - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
        original_show(title, image)

    def close(self):
        self.stop.set()
        if self.reader.is_alive():
            self.reader.join(timeout=3)
        self.collect()
        while self.pending:
            self.log(self.pending.popleft(), "CLOSED_BEFORE_EVENT_MATCH")
        self.file.close()
        print("Phase 53 results saved:", self.csv_path)


def main():
    for path in (BASE_FILE, TRACKER_FILE,
                 SYSTEM_DIR / "CONFIG" / "conveyor_boundary.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Required working reference file missing: {path}")
    helpers = load_module("crest_phase52_helpers", TRACKER_FILE)
    base = load_module("crest_phase40_base", BASE_FILE)
    if not hasattr(base, "open_camera") or not hasattr(base, "main"):
        raise RuntimeError("Phase 40 API differs. Do not modify Phase 40.")
    experiment = Experiment(helpers)
    open_original = base.open_camera
    show_original = cv2.imshow

    def camera_hook():
        cap, index = open_original()
        return CameraProxy(cap, experiment), index

    def show_hook(title, frame):
        experiment.show(title, frame, show_original)

    base.open_camera = camera_hook
    cv2.imshow = show_hook
    try:
        experiment.start()
        base.main()
    finally:
        cv2.imshow = show_original
        base.open_camera = open_original
        experiment.close()


if __name__ == "__main__":
    main()
