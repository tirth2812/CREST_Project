"""CREST Phase 50 - RFID1 ground-truth LOGGER (no mapping logic).

Phase 40 is loaded and NOT changed. No number is assigned to any box.
Records on ONE clock: every frame's boxes (track, centre, vision label)
and every RFID1 appear/leave event with a snapshot of all boxes.
RFID1 only (words 96..99). PLC READ ONLY. RFID2 never touched.

Run:   python -u 50_conference_rfid1_ground_truth_log.py
Quit:  q or Esc in the camera window.
"""

import csv
import importlib.util
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(
    r"C:\Do_Not_Delete_PLC\Conference"
    r"\07_Final_Empty_Accepted_Rejected_System"
)
LIVE_PATH = ROOT / "CODE" / "40_conference_100_pallet_live.py"
OUT_DIR = ROOT / "RESULTS" / "50_RFID1_Ground_Truth_Log"

PLC_IP = "192.168.82.231"
RFID_TAGS = [
    f"IO_Link_Master_ST1_to_ST4:I1.Data[{i}]" for i in range(96, 100)
]
NO_TAG = (0, 0, 0, 0)
RFID_MAP = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}


def rfid1_reader(events, stop):
    try:
        from pylogix import PLC
    except ImportError:
        print("ERROR: pylogix missing. Install: python -m pip install pylogix")
        return

    seen_no_tag = False
    absent_streak = 0
    pending_words = None
    first_valid_time = None
    valid_streak = 0
    present_number = None
    last_error = 0.0
    notified_wait = False

    while not stop.is_set():
        try:
            with PLC() as plc:
                plc.IPAddress = PLC_IP
                plc.SocketTimeout = 3
                print("RFID1 ONLY: connected", PLC_IP, "words [96..99]", flush=True)

                while not stop.is_set():
                    words = []
                    for tag in RFID_TAGS:
                        reading = plc.Read(tag)
                        if reading.Status != "Success":
                            raise RuntimeError(f"PLC read {tag}: {reading.Status}")
                        words.append(int(reading.Value))

                    t = time.monotonic()
                    words = tuple(words)

                    if words == NO_TAG:
                        absent_streak += 1
                        valid_streak = 0
                        pending_words = None
                        if absent_streak >= 2:
                            if present_number is not None:
                                events.put(
                                    {"type": "leave", "number": present_number, "time": t}
                                )
                                present_number = None
                            if not seen_no_tag:
                                print("RFID1 READY: sensor clear", flush=True)
                            seen_no_tag = True

                    elif words in RFID_MAP:
                        absent_streak = 0
                        if not seen_no_tag:
                            if not notified_wait:
                                print("RFID1 WAIT: start with a clear sensor", flush=True)
                                notified_wait = True
                            continue

                        if words != pending_words:
                            pending_words = words
                            valid_streak = 1
                            first_valid_time = t
                        else:
                            valid_streak += 1

                        if valid_streak >= 2 and present_number is None:
                            present_number = RFID_MAP[words]
                            events.put(
                                {
                                    "type": "appear",
                                    "number": present_number,
                                    "time": first_valid_time,
                                    "confirmed_at": t,
                                }
                            )
                    else:
                        absent_streak = 0
                        valid_streak = 0
                        pending_words = None

                    stop.wait(0.025)

        except Exception as error:
            now = time.monotonic()
            if now - last_error > 3.0:
                print("RFID1 PLC ERROR:", error, flush=True)
                last_error = now
            stop.wait(1.0)


def status_of(data):
    label = str(data.get("label", "")).upper()
    for state in ("REJECTED", "ACCEPTED", "EMPTY", "CHECKING"):
        if state in label:
            return state
    return "UNKNOWN"


class GroundTruthLog:
    def __init__(self, event_queue):
        self.events = event_queue
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frames_file = OUT_DIR / f"frames_{stamp}.csv"
        self.events_file = OUT_DIR / f"events_{stamp}.csv"
        self.start = time.monotonic()
        self.recent = deque(maxlen=400)
        self.frame_count = 0
        self.last_banner = ("", 0.0)

        with self.frames_file.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                ["frame", "t_monotonic", "t_rel_sec", "track", "cx", "cy", "w", "h",
                 "label", "missed", "confirmed"]
            )
        with self.events_file.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                ["event", "tag", "t_monotonic", "t_rel_sec", "confirmed_at_rel_sec",
                 "snapshot_frame_t_rel_sec", "n_boxes", "boxes(track:cx,cy:label:missed)"]
            )
        self._frames_handle = self.frames_file.open("a", encoding="utf-8", newline="")
        self._frames_writer = csv.writer(self._frames_handle)
        self._events_handle = self.events_file.open("a", encoding="utf-8", newline="")
        self._events_writer = csv.writer(self._events_handle)

    def close(self):
        for handle in (self._frames_handle, self._events_handle):
            try:
                handle.flush()
                handle.close()
            except Exception:
                pass

    def rel(self, t):
        return round(t - self.start, 3)

    def record_frame(self, t, scope):
        confirmed_ids = set()
        for tr in scope.get("confirmed_tracks", []):
            try:
                confirmed_ids.add(int(tr["track_id"]))
            except Exception:
                pass

        boxes = []
        for tid, data in scope.get("visible_track_cache", {}).items():
            try:
                x1, y1, x2, y2 = [float(v) for v in data["box"]]
            except Exception:
                continue
            boxes.append(
                {
                    "track": int(tid),
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                    "w": x2 - x1,
                    "h": y2 - y1,
                    "label": status_of(data),
                    "missed": int(data.get("missed", 0)),
                    "confirmed": int(int(tid) in confirmed_ids),
                }
            )

        self.frame_count += 1
        self.recent.append((t, boxes))
        for box in boxes:
            self._frames_writer.writerow(
                [self.frame_count, round(t, 4), self.rel(t), box["track"],
                 round(box["cx"], 1), round(box["cy"], 1), round(box["w"], 1),
                 round(box["h"], 1), box["label"], box["missed"], box["confirmed"]]
            )
        if self.frame_count % 15 == 0:
            self._frames_handle.flush()

    def snapshot_near(self, t):
        best = None
        for frame_t, boxes in self.recent:
            if best is None or abs(frame_t - t) < abs(best[0] - t):
                best = (frame_t, boxes)
        return best

    def drain_events(self, now):
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            snap = self.snapshot_near(event["time"])
            if snap is None:
                frame_t, boxes = None, []
            else:
                frame_t, boxes = snap
            text = " | ".join(
                f"T{b['track']}:{b['cx']:.0f},{b['cy']:.0f}:{b['label']}:{b['missed']}"
                for b in boxes
            )
            self._events_writer.writerow(
                [event["type"], event["number"], round(event["time"], 4),
                 self.rel(event["time"]),
                 self.rel(event["confirmed_at"]) if "confirmed_at" in event else "",
                 self.rel(frame_t) if frame_t is not None else "",
                 len(boxes), text]
            )
            self._events_handle.flush()
            print(
                f"RFID1 {event['type'].upper()} tag {event['number']} at "
                f"{self.rel(event['time'])}s | boxes: {text}",
                flush=True,
            )
            if event["type"] == "appear":
                self.last_banner = (f"RFID1 read: tag {event['number']}", now)

    def display(self, scope, acquired_t):
        now = time.monotonic()
        t = acquired_t if acquired_t is not None else now
        raw = scope.get("symbol_frame")
        if raw is None:
            raw = scope.get("frame")
        if raw is None:
            raise RuntimeError("Phase40 frame unavailable")
        image = raw.copy()

        self.record_frame(t, scope)
        self.drain_events(now)

        for name, color in (("outer_polygon", (0, 255, 0)), ("inner_polygon", (0, 255, 255))):
            poly = scope.get(name)
            if poly is not None:
                cv2.polylines(image, [poly], True, color, 2)

        for tid, data in scope.get("visible_track_cache", {}).items():
            x1, y1, x2, y2 = [int(v) for v in data["box"]]
            status = status_of(data)
            color = {
                "EMPTY": (0, 255, 0), "ACCEPTED": (255, 255, 0),
                "REJECTED": (0, 0, 255), "CHECKING": (0, 255, 255),
            }.get(status, (255, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2 if data.get("missed", 0) else 4)
            cv2.putText(image, f"? {status} [T{tid}]", (max(0, x1), max(25, y1 - 9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        banner, banner_t = self.last_banner
        recent = banner if banner and now - banner_t < 4.0 else "waiting for tag"
        title = f"RFID1 GROUND-TRUTH LOG (no mapping) | frames {self.frame_count} | {recent}"
        cv2.rectangle(image, (5, 5), (1500, 52), (0, 0, 0), -1)
        cv2.putText(image, title, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return image


class CameraClock:
    def __init__(self, original):
        self.original = original
        self.acquired_at = None

    def read(self):
        result = self.original.read()
        self.acquired_at = time.monotonic() if result[0] else None
        return result

    def __getattr__(self, name):
        return getattr(self.original, name)


def main():
    if not LIVE_PATH.is_file():
        raise FileNotFoundError(LIVE_PATH)

    print("RFID1 only. RFID2: NOT READ OR USED. PLC: read only.")
    print("Phase 40 detection and classification: UNCHANGED. No number is assigned to any box.")

    sys.path.insert(0, str(LIVE_PATH.parent))
    spec = importlib.util.spec_from_file_location("crest_conference_live50", str(LIVE_PATH))
    live = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = live
    spec.loader.exec_module(live)

    events = queue.Queue()
    stop = threading.Event()
    log = GroundTruthLog(events)

    real_open = live.open_camera
    real_show = cv2.imshow
    clock = {"obj": None}

    def open_camera_wrapper():
        cap, index = real_open()
        clock["obj"] = CameraClock(cap)
        return clock["obj"], index

    def show_wrapper(name, frame):
        if "CREST" not in str(name).upper():
            return real_show(name, frame)
        scope = sys._getframe(1).f_locals
        acquired = clock["obj"].acquired_at if clock["obj"] else None
        return real_show(name, log.display(scope, acquired))

    live.open_camera = open_camera_wrapper
    cv2.imshow = show_wrapper

    worker = threading.Thread(target=rfid1_reader, args=(events, stop), daemon=True)
    worker.start()

    print()
    print("START: ONE pallet with an object (ACCEPTED/REJECTED), all others EMPTY.")
    print("Let the sensor become clear, then run 2+ laps. Press q to stop.")
    print()

    try:
        live.main()
    finally:
        stop.set()
        worker.join(timeout=2)
        live.open_camera = real_open
        cv2.imshow = real_show
        log.drain_events(time.monotonic())
        log.close()
        print("Frames CSV:", log.frames_file)
        print("Events CSV:", log.events_file)


if __name__ == "__main__":
    main()