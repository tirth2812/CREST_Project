
"""CREST Phase45 RFID1 timing calibration; never maps to physical ID.

Phase40 YOLO, tracker, and classification stay unchanged. One camera.
No PLC writes. Only RFID1 [96..99].
"""

import csv
import importlib.util
import json
import queue
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(
    r"C:\Do_Not_Delete_PLC\Conference"
    r"\07_Final_Empty_Accepted_Rejected_System"
)
LIVE_PATH = (
    ROOT / "CODE" / "40_conference_100_pallet_live.py"
)
CONFIG_PATH = ROOT / "CONFIG" / "rfid_stations.json"
OUT_DIR = ROOT / "RESULTS" / "45_RFID1_Timing_Calibration"

PLC_IP = "192.168.82.231"

RFID_TAGS = [
    f"IO_Link_Master_ST1_to_ST4:I1.Data[{i}]"
    for i in range(96, 100)
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

DEFAULT_BOX = (1123, 578, 1183, 675)


def station_box():
    if not CONFIG_PATH.is_file():
        print(
            "Using saved verified RFID1 default:",
            DEFAULT_BOX
        )
        return DEFAULT_BOX

    with CONFIG_PATH.open(encoding="utf-8") as handle:
        cfg = json.load(handle)

    b = cfg.get("rfid1", {}).get("box", {})

    if not all(
        k in b
        for k in ("x1", "y1", "x2", "y2")
    ):
        return DEFAULT_BOX

    sx = 1920.0 / float(
        cfg.get("image_width", 1920)
    )
    sy = 1080.0 / float(
        cfg.get("image_height", 1080)
    )

    return tuple(
        round(
            float(b[k]) * (
                sx if k[0] == "x" else sy
            )
        )
        for k in ("x1", "y1", "x2", "y2")
    )


class CameraClock:
    """Use the existing camera, never open a second camera."""

    def __init__(self, original):
        self.original = original
        self.acquired_at = None

    def read(self):
        result = self.original.read()

        self.acquired_at = (
            time.monotonic()
            if result[0]
            else None
        )

        return result

    def __getattr__(self, name):
        return getattr(self.original, name)


def rfid1_reader(events, stop):
    try:
        from pylogix import PLC

    except ImportError:
        print(
            "ERROR: pylogix missing. "
            "Install using python -m pip install pylogix"
        )
        return

    seen_no_tag = False
    absent_streak = 0
    valid_streak = 0

    pending_words = None
    first_valid_time = None
    emitted_during_presence = False

    last_error = 0.0
    last_start_notice = False

    while not stop.is_set():

        try:
            with PLC() as plc:

                plc.IPAddress = PLC_IP
                plc.SocketTimeout = 3

                print(
                    "RFID1 ONLY: connected",
                    PLC_IP,
                    "words [96..99]",
                    flush=True
                )

                while not stop.is_set():

                    words = []

                    for tag in RFID_TAGS:

                        reading = plc.Read(tag)

                        if reading.Status != "Success":
                            raise RuntimeError(
                                f"PLC read {tag}: "
                                f"{reading.Status}"
                            )

                        words.append(
                            int(reading.Value)
                        )

                    t = time.monotonic()
                    words = tuple(words)

                    if words == NO_TAG:

                        absent_streak += 1
                        valid_streak = 0
                        pending_words = None

                        if absent_streak >= 2:

                            if not seen_no_tag:
                                print(
                                    "RFID1 READY: sensor "
                                    "returned to 0,0,0,0",
                                    flush=True
                                )

                            seen_no_tag = True
                            emitted_during_presence = False

                    elif words in RFID_MAP:

                        absent_streak = 0

                        if not seen_no_tag:

                            if not last_start_notice:
                                print(
                                    "RFID1 WAIT: start with "
                                    "a clear sensor",
                                    flush=True
                                )
                                last_start_notice = True

                            continue

                        if words != pending_words:

                            pending_words = words
                            valid_streak = 1
                            first_valid_time = t

                        else:
                            valid_streak += 1

                        if (
                            valid_streak >= 2
                            and not emitted_during_presence
                        ):

                            events.put({
                                "number": RFID_MAP[words],
                                "words": words,
                                "time": first_valid_time,
                                "confirmed_at": t,
                            })

                            emitted_during_presence = True

                    else:
                        # Ignore incomplete PLC words.
                        absent_streak = 0
                        valid_streak = 0
                        pending_words = None

                    stop.wait(0.025)

        except Exception as error:

            now = time.monotonic()

            if now - last_error > 3.0:
                print(
                    "RFID1 PLC ERROR:",
                    error,
                    flush=True
                )
                last_error = now

            stop.wait(1.0)


def status_of(data):

    label = str(
        data.get("label", "")
    ).upper()

    for state in (
        "REJECTED",
        "ACCEPTED",
        "EMPTY",
        "CHECKING"
    ):

        if state in label:
            return state

    return "UNKNOWN"


class Calibration:

    def __init__(self, event_queue, box):

        self.events = event_queue
        self.box = box

        self.last_snapshot = None
        self.pending = []

        self.latest_number = None
        self.latest_time = 0.0
        self.last_image = None

        self.event_counter = 0

        self.csv_file = (
            OUT_DIR
            / (
                "rfid1_frames_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                + ".csv"
            )
        )

        self.columns = [
            "event",
            "pallet",
            "capture",
            "frame_minus_read_s",
            "frame_acquired_monotonic",
            "read_monotonic",
            "reader_confirm_delay_s",
            "track_id",
            "classification",
            "x1",
            "y1",
            "x2",
            "y2",
            "vx",
            "vy",
            "box_left_minus_sensor_x",
            "box_right_minus_sensor_x",
            "box_bottom_minus_sensor_y1",
            "sensor_x_inside_box",
        ]

        with self.csv_file.open(
            "w",
            encoding="utf-8",
            newline=""
        ) as f:

            csv.writer(f).writerow(
                self.columns
            )

    def store_capture(
        self,
        event,
        snapshot,
        name
    ):

        if snapshot is None or event.get(name):
            return

        delta = (
            snapshot["t"] - event["time"]
        )

        if name == "BEFORE" and delta > 0:
            return

        if name == "AFTER" and delta < 0:
            return

        filename = (
            f"RFID1_P{event['number']}"
            f"_E{event['id']:03d}"
            f"_{name}"
            f"_{delta:+.2f}s.png"
        )

        path = OUT_DIR / filename
        image = snapshot["image"].copy()

        cv2.putText(
            image,
            (
                f"RFID1 READ={event['number']} "
                f"| {name} frame "
                f"dt={delta:+.2f}s"
            ),
            (16, 111),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78,
            (0, 255, 255),
            2
        )

        cv2.imwrite(
            str(path),
            image
        )

        event[name] = True

        sx = (
            self.box[0] + self.box[2]
        ) / 2.0

        print(
            f"RFID1 {name}: "
            f"P{event['number']} "
            f"frame/read dt={delta:+.3f}s "
            f"-> {path.name}",
            flush=True
        )

        nearby = []

        with self.csv_file.open(
            "a",
            encoding="utf-8",
            newline=""
        ) as f:

            writer = csv.writer(f)

            if not snapshot["tracks"]:

                row = [
                    event["id"],
                    event["number"],
                    name,
                    round(delta, 3),
                    round(snapshot["t"], 3),
                    round(event["time"], 3),
                    round(
                        event["confirmed_at"]
                        - event["time"],
                        3
                    ),
                ]

                row += (
                    [""] * (
                        len(self.columns)
                        - len(row)
                    )
                )

                writer.writerow(row)

            for tr in snapshot["tracks"]:

                x1, y1, x2, y2 = (
                    tr["box"]
                )

                writer.writerow([
                    event["id"],
                    event["number"],
                    name,
                    round(delta, 3),
                    round(snapshot["t"], 3),
                    round(event["time"], 3),
                    round(
                        event["confirmed_at"]
                        - event["time"],
                        3
                    ),
                    tr["id"],
                    tr["status"],
                    x1,
                    y1,
                    x2,
                    y2,
                    round(tr["vx"], 2),
                    round(tr["vy"], 2),
                    round(x1 - sx, 1),
                    round(x2 - sx, 1),
                    y2 - self.box[1],
                    int(x1 <= sx <= x2),
                ])

                if (
                    tr["box"][1]
                    < self.box[1] + 100
                    and
                    tr["box"][3]
                    > self.box[1] - 150
                ):

                    nearby.append((
                        abs(x1 - sx),
                        tr["id"],
                        tr["box"],
                        tr["vx"],
                    ))

        print(
            "  top-lane box observations "
            "(diagnostic, not mapping):",
            sorted(nearby)[:5],
            flush=True
        )

    def display(
        self,
        scope,
        acquisition_t
    ):

        now = time.monotonic()

        raw = scope.get(
            "symbol_frame"
        )

        if raw is None:
            raw = scope.get("frame")

        if raw is None:
            raise RuntimeError(
                "Phase40 frame unavailable"
            )

        image = raw.copy()

        h, w = image.shape[:2]

        bx1, by1, bx2, by2 = self.box

        sx = w / 1920.0
        sy = h / 1080.0

        bx1, by1, bx2, by2 = (
            int(round(bx1 * sx)),
            int(round(by1 * sy)),
            int(round(bx2 * sx)),
            int(round(by2 * sy)),
        )

        for polygon_name, color in (
            ("outer_polygon", (0, 255, 0)),
            ("inner_polygon", (0, 255, 255)),
        ):

            polygon = scope.get(
                polygon_name
            )

            if polygon is not None:

                cv2.polylines(
                    image,
                    [polygon],
                    True,
                    color,
                    2
                )

        cv2.rectangle(
            image,
            (bx1, by1),
            (bx2, by2),
            (0, 255, 255),
            3
        )

        cv2.putText(
            image,
            "RFID1 SENSOR",
            (bx1, max(18, by1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2
        )

        sensor_x = (
            bx1 + bx2
        ) // 2

        cv2.line(
            image,
            (
                sensor_x,
                max(0, by1 - 150)
            ),
            (sensor_x, by2),
            (0, 200, 255),
            1
        )

        current_tracks = scope.get(
            "confirmed_tracks",
            []
        )

        status_by_id = {
            int(tid): status_of(data)
            for tid, data in scope.get(
                "visible_track_cache",
                {}
            ).items()
        }

        diagnostic_tracks = []

        for t in current_tracks:

            tid = int(
                t["track_id"]
            )

            coords = tuple(
                int(v)
                for v in t["box"]
            )

            diagnostic_tracks.append({
                "id": tid,
                "box": coords,
                "status": status_by_id.get(
                    tid,
                    "UNKNOWN"
                ),
                "vx": float(
                    t.get("velocity_x", 0)
                ),
                "vy": float(
                    t.get("velocity_y", 0)
                ),
            })

        visible = scope.get(
            "visible_track_cache",
            {}
        )

        colors = {
            "EMPTY": (0, 255, 0),
            "ACCEPTED": (255, 255, 0),
            "REJECTED": (0, 0, 255),
            "CHECKING": (0, 255, 255),
            "UNKNOWN": (255, 255, 255),
        }

        for tid, data in visible.items():

            x1, y1, x2, y2 = [
                int(v)
                for v in data["box"]
            ]

            status = status_of(data)

            color = colors.get(
                status,
                (255, 255, 255)
            )

            thickness = (
                2
                if data.get("missed", 0)
                else 4
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                thickness
            )

            # T indicates internal diagnostic ID.
            # It is NOT a physical pallet number.
            label = (
                f"? {status}  [T{tid}]"
            )

            cv2.putText(
                image,
                label,
                (
                    max(0, x1),
                    max(25, y1 - 9)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.64,
                color,
                2
            )

        if (
            self.latest_number is not None
            and
            now - self.latest_time < 3.0
        ):

            text = (
                "RFID1 CALIBRATION ONLY "
                f"| Latest tag: "
                f"{self.latest_number}"
            )

        else:

            text = (
                "RFID1 CALIBRATION ONLY "
                "| Waiting for new RFID1 tag"
            )

        cv2.rectangle(
            image,
            (5, 5),
            (1030, 86),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            image,
            text,
            (18, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            (
                "? = NOT MAPPED | "
                "[T...] = INTERNAL DIAGNOSTIC "
                "| S save | Q quit"
            ),
            (18, 66),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (255, 255, 255),
            1
        )

        self.last_image = image

        snapshot = {
            "t": (
                acquisition_t
                if acquisition_t is not None
                else now
            ),
            "image": image.copy(),
            "tracks": diagnostic_tracks,
        }

        # Receive RFID1 events.
        while True:

            try:
                reading = (
                    self.events.get_nowait()
                )

            except queue.Empty:
                break

            self.event_counter += 1

            reading["id"] = (
                self.event_counter
            )

            self.latest_number = (
                reading["number"]
            )

            self.latest_time = now

            self.pending.append(
                reading
            )

            delay = (
                reading["confirmed_at"]
                - reading["time"]
            )

            print(
                f"\nRFID1 READ P"
                f"{reading['number']} "
                f"| event E"
                f"{reading['id']:03d} "
                f"| reader confirmation "
                f"delay={delay:.3f}s",
                flush=True
            )

            if self.last_snapshot is not None:

                self.store_capture(
                    reading,
                    self.last_snapshot,
                    "BEFORE"
                )

        # Save first available frame on
        # either side of each RFID reading.
        for event in list(self.pending):

            self.store_capture(
                event,
                snapshot,
                "BEFORE"
            )

            self.store_capture(
                event,
                snapshot,
                "AFTER"
            )

            if (
                event.get("AFTER")
                or now - event["time"] >= 8.0
            ):

                if (
                    not event.get("BEFORE")
                    or not event.get("AFTER")
                ):

                    print(
                        f"RFID1 E"
                        f"{event['id']:03d}: "
                        "missing frame on one side; "
                        "inspect available evidence",
                        flush=True
                    )

                self.pending.remove(
                    event
                )

        self.last_snapshot = snapshot

        return image


def main():

    if not LIVE_PATH.is_file():

        raise FileNotFoundError(
            LIVE_PATH
        )

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    box = station_box()

    print(
        "RFID1 box:",
        box
    )

    print(
        "RFID2: NOT READ OR USED"
    )

    print(
        "Phase40 detection and "
        "classification: UNCHANGED"
    )

    print(
        "NO AUTOMATIC RFID-to-box "
        "assignment in calibration mode"
    )

    sys.path.insert(
        0,
        str(LIVE_PATH.parent)
    )

    spec = (
        importlib.util.spec_from_file_location(
            "crest_conference_live45",
            str(LIVE_PATH)
        )
    )

    live = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[spec.name] = live

    spec.loader.exec_module(live)

    events = queue.Queue()
    stop = threading.Event()

    app = Calibration(
        events,
        box
    )

    real_open = live.open_camera
    real_show = cv2.imshow
    real_key = cv2.waitKey

    clock = None

    def open_camera_wrapper():

        nonlocal clock

        cap, camera_idx = real_open()

        clock = CameraClock(cap)

        return clock, camera_idx

    def show_wrapper(name, frame):

        if "CREST" not in str(name).upper():

            return real_show(
                name,
                frame
            )

        scope = (
            sys._getframe(1).f_locals
        )

        acquired = (
            clock.acquired_at
            if clock
            else None
        )

        return real_show(
            name,
            app.display(
                scope,
                acquired
            )
        )

    def key_wrapper(delay=0):

        key = real_key(delay)

        if (
            key & 0xFF == ord("s")
            and app.last_image is not None
        ):

            filename = (
                "manual_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
                + ".png"
            )

            destination = (
                OUT_DIR / filename
            )

            cv2.imwrite(
                str(destination),
                app.last_image
            )

            print(
                "Manual screenshot:",
                destination
            )

            return 255

        return key

    # Temporary hooks.
    # The original Phase 40 file is untouched.
    live.open_camera = (
        open_camera_wrapper
    )

    cv2.imshow = (
        show_wrapper
    )

    cv2.waitKey = (
        key_wrapper
    )

    worker = threading.Thread(
        target=rfid1_reader,
        args=(events, stop),
        daemon=True
    )

    worker.start()

    print()
    print(
        "START: Let RFID1 become clear, "
        "then pass known pallets through it."
    )
    print()

    try:
        live.main()

    finally:

        stop.set()

        worker.join(
            timeout=2
        )

        live.open_camera = real_open
        cv2.imshow = real_show
        cv2.waitKey = real_key

        print(
            "Evidence CSV:",
            app.csv_file
        )

        print(
            "Evidence images:",
            OUT_DIR
        )


if __name__ == "__main__":
    main()
