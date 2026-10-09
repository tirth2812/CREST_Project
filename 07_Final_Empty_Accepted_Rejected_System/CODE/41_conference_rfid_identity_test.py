
"""CREST conference RFID identity test.

Runs the existing Phase 40 conference model without changing it.

UNKNOWN physical pallet -> ? EMPTY / ? ACCEPTED / ? REJECTED
Verified physical pallet -> 1-6 with the existing classification

RFID-to-vision association requires:
1. A directional crossing at the calibrated RFID station.
2. A valid PLC RFID reading from the same station.
3. A unique timestamp match.

No PLC writes, piston control, or conveyor stopping.

Keys:
S = save screenshot
R = original model manual recheck
Q = quit
"""

import csv
import importlib.util
import json
import queue
import sys
import threading
import time

from collections import deque
from datetime import datetime
from pathlib import Path

import cv2


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\Do_Not_Delete_PLC\Conference"
)

SYSTEM_ROOT = (
    ROOT
    / "07_Final_Empty_Accepted_Rejected_System"
)

LIVE_FILE = (
    SYSTEM_ROOT
    / "CODE"
    / "40_conference_100_pallet_live.py"
)

CONFIG_FILE = (
    SYSTEM_ROOT
    / "CONFIG"
    / "rfid_stations.json"
)

SAVE_DIR = (
    SYSTEM_ROOT
    / "RESULTS"
    / "41_RFID_Identity_Test"
)


# ============================================================
# RFID PLC CONFIGURATION
# ============================================================

PLC_IP = "192.168.82.231"

MATCH_SECONDS = 1.5

VERTICAL_MARGIN = 35


# ============================================================
# VERIFIED PHYSICAL RFID MAPPING
# ============================================================

RFID_MAP = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}


# ============================================================
# RFID STATIONS
# ============================================================

STATIONS = {
    "RFID1": {
        "box": (
            1123, 578, 1183, 675
        ),
        "direction": -1,
        "tags": [
            (
                "IO_Link_Master_ST1_to_ST4:"
                f"I1.Data[{i}]"
            )
            for i in range(96, 100)
        ],
    },

    "RFID2": {
        "box": (
            823, 734, 884, 815
        ),
        "direction": 1,
        "tags": [
            (
                "IO_Link_Master_ST1_to_ST4:"
                f"I1.Data[{i}]"
            )
            for i in range(128, 132)
        ],
    },
}


# ============================================================
# LOAD EXISTING RFID BOUNDARIES
# ============================================================

def load_boxes():

    if not CONFIG_FILE.is_file():

        print(
            "Using previously verified "
            "RFID station boxes."
        )

        return

    with CONFIG_FILE.open(
        encoding="utf-8"
    ) as handle:

        config = json.load(handle)

    scale_x = (
        1920
        / float(config.get("image_width", 1920))
    )

    scale_y = (
        1080
        / float(config.get("image_height", 1080))
    )

    for station_name, setting in STATIONS.items():

        saved = (
            config.get(
                station_name.lower(), {}
            ).get("box", {})
        )

        required = (
            "x1", "y1", "x2", "y2"
        )

        if all(key in saved for key in required):

            setting["box"] = (
                int(round(saved["x1"] * scale_x)),
                int(round(saved["y1"] * scale_y)),
                int(round(saved["x2"] * scale_x)),
                int(round(saved["y2"] * scale_y)),
            )

    print(
        "RFID station boxes:",
        {
            name: setting["box"]
            for name, setting in STATIONS.items()
        }
    )


# ============================================================
# RFID READING THREAD
# ============================================================

def rfid_worker(events, stop_event):

    try:
        from pylogix import PLC

    except ImportError:

        print(
            "RFID requires pylogix."
        )

        print(
            "Install using: "
            "python -m pip install pylogix"
        )

        return

    previous = {
        name: (0, 0, 0, 0)
        for name in STATIONS
    }

    last_error = 0.0

    while not stop_event.is_set():

        try:

            with PLC() as comm:

                comm.IPAddress = PLC_IP

                comm.SocketTimeout = 3

                print(
                    "RFID PLC polling started:",
                    PLC_IP
                )

                while not stop_event.is_set():

                    for station, info in STATIONS.items():

                        values = []

                        for tag in info["tags"]:

                            result = comm.Read(tag)

                            if result.Status != "Success":

                                raise IOError(
                                    f"{tag}: {result.Status}"
                                )

                            values.append(
                                int(result.Value)
                            )

                        words = tuple(values)

                        # Only generate an event when
                        # the RFID reading changes.
                        if words != previous[station]:

                            previous[station] = words

                            if words != (0, 0, 0, 0):

                                events.put({
                                    "station": station,
                                    "number": RFID_MAP.get(words),
                                    "words": words,
                                    "time": time.monotonic(),
                                })

                    stop_event.wait(0.05)

        except Exception as error:

            now = time.monotonic()

            if now - last_error > 4:

                print(
                    "RFID read unavailable:",
                    error
                )

                last_error = now

            stop_event.wait(1.0)


# ============================================================
# TRACK GEOMETRY
# ============================================================

def box_center(box):

    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0
    )


def current_items(local_vars):

    visible = local_vars.get(
        "visible_track_cache"
    )

    if not isinstance(visible, dict):

        visible = local_vars.get(
            "visible"
        )

    if not isinstance(visible, dict):

        raise RuntimeError(
            "Cannot locate the live track states "
            "at the display stage. "
            "The original model was not modified."
        )

    active = local_vars.get(
        "current_track_ids"
    )

    items = []

    for track_id, data in visible.items():

        if isinstance(data, dict):

            box = data.get("box")

            label = data.get(
                "label", ""
            )

            status = data.get(
                "status", ""
            )

            missed = data.get(
                "missed", 0
            )

        else:

            box = getattr(
                data, "box", None
            )

            label = getattr(
                data, "label", ""
            )

            status = getattr(
                data, "status", ""
            )

            missed = getattr(
                data, "missed", 0
            )

        if box is None:
            continue

        valid_states = (
            "EMPTY",
            "ACCEPTED",
            "REJECTED",
            "CHECKING"
        )

        if status not in valid_states:

            status = next(
                (
                    state
                    for state in (
                        "REJECTED",
                        "ACCEPTED",
                        "EMPTY",
                        "CHECKING"
                    )
                    if state in str(label).upper()
                ),
                "CHECKING"
            )

        currently_matched = (
            (
                active is None
                or track_id in active
            )
            and missed == 0
        )

        items.append((
            track_id,
            data,
            tuple(box),
            status,
            currently_matched
        ))

    return items


# ============================================================
# RFID ASSOCIATION
# ============================================================

class Association:

    def __init__(self, events):

        self.events = events

        self.tracks = {}

        self.crossings = deque()

        self.reads = deque()

        self.last_cross = {}

        self.last_frame = None

        filename = (
            "rfid_events_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".csv"
        )

        self.log_file = (
            SAVE_DIR / filename
        )

        with self.log_file.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            csv.writer(file).writerow([
                "time",
                "event",
                "station",
                "rfid_pallet",
                "vision_track",
                "time_difference_s",
                "note"
            ])

    def log(
        self,
        event,
        station="",
        number="",
        track_id="",
        difference="",
        note=""
    ):

        message = (
            f"{event}: "
            f"{station} "
            f"RFID={number} "
            f"TRACK={track_id} "
            f"dt={difference} "
            f"{note}"
        )

        print(
            message,
            flush=True
        )

        with self.log_file.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            csv.writer(file).writerow([
                datetime.now().isoformat(
                    timespec="seconds"
                ),
                event,
                station,
                number,
                track_id,
                difference,
                note
            ])

    def observe(self, items):

        now = time.monotonic()

        # --------------------------------------------
        # Track each object's actual state instance.
        # If the existing state bridge transfers it
        # to a new internal tracker ID, identity
        # follows that same state instance.
        # --------------------------------------------

        for (
            track_id,
            data,
            box,
            status,
            matched
        ) in items:

            key = id(data)

            existing = self.tracks.get(
                key
            )

            if (
                existing is None
                or existing["object"] is not data
            ):

                existing = {
                    "object": data,
                    "number": None,
                    "center": None,
                    "seen": 0.0,
                    "track_id": track_id
                }

                self.tracks[key] = existing

            previous_center = (
                existing["center"]
            )

            previous_time = (
                existing["seen"]
            )

            current_center = (
                box_center(box)
            )

            if (
                matched
                and previous_center is not None
                and now - previous_time < 2.0
            ):

                for (
                    station,
                    setting
                ) in STATIONS.items():

                    x1, y1, x2, y2 = (
                        setting["box"]
                    )

                    station_x = (
                        x1 + x2
                    ) / 2.0

                    inside_station = (
                        y1 - VERTICAL_MARGIN
                        <= previous_center[1]
                        <= y2 + VERTICAL_MARGIN
                        and
                        y1 - VERTICAL_MARGIN
                        <= current_center[1]
                        <= y2 + VERTICAL_MARGIN
                    )

                    direction = setting[
                        "direction"
                    ]

                    if direction == -1:

                        crossed = (
                            previous_center[0]
                            > station_x
                            >= current_center[0]
                        )

                    else:

                        crossed = (
                            previous_center[0]
                            < station_x
                            <= current_center[0]
                        )

                    cooldown_key = (
                        key, station
                    )

                    last_time = self.last_cross.get(
                        cooldown_key,
                        -1000000.0
                    )

                    if (
                        inside_station
                        and crossed
                        and now - last_time >= 1.5
                    ):

                        self.crossings.append({
                            "station": station,
                            "key": key,
                            "track_id": track_id,
                            "time": now
                        })

                        self.last_cross[
                            cooldown_key
                        ] = now

                        self.log(
                            "VISION_CROSS",
                            station,
                            track_id=track_id
                        )

            if matched:

                existing["center"] = (
                    current_center
                )

                existing["seen"] = now

                existing["track_id"] = (
                    track_id
                )

        # --------------------------------------------
        # Collect RFID readings from PLC thread.
        # --------------------------------------------

        while True:

            try:

                event = (
                    self.events.get_nowait()
                )

            except queue.Empty:
                break

            if event["number"] is None:

                self.log(
                    "UNRECOGNIZED_TAG",
                    event["station"],
                    note=str(event["words"])
                )

            else:

                self.reads.append(
                    event
                )

                self.log(
                    "RFID_READ",
                    event["station"],
                    event["number"],
                    note=str(event["words"])
                )

        # --------------------------------------------
        # Require an unambiguous station crossing.
        # Do not map to the nearest box.
        # --------------------------------------------

        while (
            self.reads
            and now - self.reads[0]["time"]
            > MATCH_SECONDS
        ):

            reading = (
                self.reads.popleft()
            )

            candidates = [
                crossing
                for crossing in self.crossings
                if (
                    crossing["station"]
                    == reading["station"]
                    and
                    abs(
                        crossing["time"]
                        - reading["time"]
                    ) <= MATCH_SECONDS
                )
            ]

            if len(candidates) != 1:

                self.log(
                    "AMBIGUOUS_OR_UNMATCHED",
                    reading["station"],
                    reading["number"],
                    note=(
                        f"crossings={len(candidates)}"
                    )
                )

                continue

            crossing = candidates[0]

            candidate = self.tracks.get(
                crossing["key"]
            )

            if (
                candidate is None
                or now - candidate["seen"] > 2.5
            ):

                self.log(
                    "LOST_TRACK",
                    reading["station"],
                    reading["number"],
                    crossing["track_id"]
                )

                continue

            # Prevent two recent RFID events from
            # being assigned to the same crossing.

            multiple_reads = any(
                other["station"]
                == reading["station"]
                and
                abs(
                    other["time"]
                    - crossing["time"]
                ) <= MATCH_SECONDS
                for other in self.reads
            )

            if multiple_reads:

                self.log(
                    "MULTIPLE_RFID_EVENTS",
                    reading["station"],
                    reading["number"],
                    crossing["track_id"]
                )

                continue

            # One active state owns a physical
            # pallet number at a time.

            for other in self.tracks.values():

                if (
                    other is not candidate
                    and
                    other["number"]
                    == reading["number"]
                ):

                    other["number"] = None

            old_number = candidate[
                "number"
            ]

            candidate["number"] = (
                reading["number"]
            )

            self.crossings.remove(
                crossing
            )

            if old_number is None:

                event_type = "MAP"

            elif old_number == reading["number"]:

                event_type = "VERIFY"

            else:

                event_type = "CORRECT"

            difference = round(
                crossing["time"]
                - reading["time"],
                3
            )

            self.log(
                event_type,
                reading["station"],
                reading["number"],
                candidate["track_id"],
                difference,
                f"old={old_number}"
            )

        # --------------------------------------------
        # Remove old association history.
        # --------------------------------------------

        self.crossings = deque(
            crossing
            for crossing in self.crossings
            if now - crossing["time"] <= 4
        )

        for key, state in list(
            self.tracks.items()
        ):

            if now - state["seen"] > 20:

                del self.tracks[key]

                for station in STATIONS:

                    self.last_cross.pop(
                        (key, station),
                        None
                    )

    # ========================================================
    # DRAW RFID IDENTITY WITH EXISTING CLASSIFICATION
    # ========================================================

    def draw(self, local_vars, items):

        base = local_vars.get(
            "symbol_frame"
        )

        if base is None:

            base = local_vars.get(
                "frame"
            )

        if base is None:

            raise RuntimeError(
                "No camera frame available"
            )

        image = base.copy()

        # Draw existing RFID station boxes.

        for station, setting in STATIONS.items():

            x1, y1, x2, y2 = (
                setting["box"]
            )

            color = (
                (0, 255, 255)
                if station == "RFID1"
                else (255, 255, 0)
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                image,
                station,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        colors = {
            "EMPTY": (0, 255, 0),
            "ACCEPTED": (255, 255, 0),
            "REJECTED": (0, 0, 255),
            "CHECKING": (0, 255, 255)
        }

        # Draw the unchanged classification
        # with the RFID physical pallet identity.

        for (
            track_id,
            data,
            box,
            status,
            matched
        ) in items:

            owner = self.tracks.get(
                id(data)
            )

            number = None

            if (
                owner is not None
                and owner["object"] is data
            ):

                number = owner[
                    "number"
                ]

            identity = (
                str(number)
                if number is not None
                else "?"
            )

            label = (
                f"{identity} {status}"
            )

            x1, y1, x2, y2 = [
                int(value)
                for value in box
            ]

            color = colors.get(
                status,
                (255, 255, 255)
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                4 if matched else 2
            )

            font = (
                cv2.FONT_HERSHEY_SIMPLEX
            )

            (text_width, text_height), _ = (
                cv2.getTextSize(
                    label,
                    font,
                    0.65,
                    2
                )
            )

            tx = max(
                0,
                min(
                    x1,
                    image.shape[1]
                    - text_width - 12
                )
            )

            ty = max(
                text_height + 7,
                y1 - 10
            )

            cv2.rectangle(
                image,
                (
                    tx,
                    ty - text_height - 5
                ),
                (
                    tx + text_width + 8,
                    ty + 5
                ),
                (0, 0, 0),
                -1
            )

            cv2.putText(
                image,
                label,
                (tx + 4, ty),
                font,
                0.65,
                color,
                2
            )

        # Count physically mapped pallets.

        mapped = 0

        for (
            track_id,
            data,
            box,
            status,
            matched
        ) in items:

            owner = self.tracks.get(
                id(data)
            )

            if (
                owner is not None
                and owner["number"] is not None
            ):

                mapped += 1

        cv2.rectangle(
            image,
            (8, 8),
            (800, 76),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            image,
            (
                f"CREST RFID TEST | "
                f"MAPPED {mapped}/{len(items)}"
            ),
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            (
                "? = RFID not verified | "
                "S save | R recheck | Q quit"
            ),
            (20, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1
        )

        return image


# ============================================================
# RUN ORIGINAL CONFERENCE MODEL WITH RFID OVERLAY
# ============================================================

def main():

    if not LIVE_FILE.is_file():

        raise FileNotFoundError(
            str(LIVE_FILE)
        )

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    load_boxes()

    sys.path.insert(
        0,
        str(LIVE_FILE.parent)
    )

    spec = importlib.util.spec_from_file_location(
        "crest_conference_phase40",
        str(LIVE_FILE)
    )

    live = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = live

    spec.loader.exec_module(
        live
    )

    if not callable(
        getattr(live, "main", None)
    ):

        raise RuntimeError(
            "Phase 40 has no main() function"
        )

    messages = queue.Queue()

    stop = threading.Event()

    layer = Association(
        messages
    )

    thread = threading.Thread(
        target=rfid_worker,
        args=(messages, stop),
        daemon=True
    )

    original_show = cv2.imshow

    original_key = cv2.waitKey

    # Keep original model processing unchanged.
    # Replace only the final image shown to user.

    def show(window_name, image):

        if "CREST" not in str(
            window_name
        ).upper():

            return original_show(
                window_name,
                image
            )

        frame_vars = (
            sys._getframe(1).f_locals
        )

        items = current_items(
            frame_vars
        )

        layer.observe(
            items
        )

        display = layer.draw(
            frame_vars,
            items
        )

        layer.last_frame = display

        return original_show(
            window_name,
            display
        )

    def key(delay=0):

        code = original_key(
            delay
        )

        if (
            code & 0xFF == ord("s")
            and layer.last_frame is not None
        ):

            filename = (
                "rfid_live_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
                + ".png"
            )

            path = (
                SAVE_DIR / filename
            )

            cv2.imwrite(
                str(path),
                layer.last_frame
            )

            print(
                "SAVED:",
                path
            )

            # Prevent saving the old overlay.
            return 255

        return code

    cv2.imshow = show

    cv2.waitKey = key

    thread.start()

    print()
    print(
        "CREST RFID IDENTITY TEST"
    )

    print(
        "? remains until a unique "
        "RFID/vision match."
    )

    print(
        "No PLC writes. "
        "No model changes."
    )

    print()

    try:

        live.main()

    finally:

        stop.set()

        thread.join(
            timeout=3
        )

        cv2.imshow = original_show

        cv2.waitKey = original_key

        print(
            "RFID evidence saved:",
            layer.log_file
        )


if __name__ == "__main__":
    main()
