
"""CREST Conference Phase 44: RFID1-only identity test.

Runs the original Phase 40 camera/vision pipeline.
Only reads RFID1 PLC tags [96..99].

RFID2 is not read, matched, or displayed.

Unknown physical identity: ? ACCEPTED / ? REJECTED / ? EMPTY
Matched physical identity: 1-6 with the original classification

No PLC writes, piston operations, or conveyor control.
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
    r"\07_Final_Empty_Accepted_Rejected_System"
)

LIVE = (
    ROOT / "CODE"
    / "40_conference_100_pallet_live.py"
)

STATION_CONFIG = (
    ROOT / "CONFIG" / "rfid_stations.json"
)

RESULTS = (
    ROOT / "RESULTS" / "44_RFID1_Only_Test"
)


# ============================================================
# RFID1 ONLY
# ============================================================

PLC_IP = "192.168.82.231"

RFID1_WORDS = [
    (
        "IO_Link_Master_ST1_to_ST4:"
        f"I1.Data[{i}]"
    )
    for i in range(96, 100)
]

TAG_TO_PALLET = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}


# ============================================================
# STATION POSITION
# ============================================================

DEFAULT_BOX = (1123, 578, 1183, 675)

SAMPLE_BEFORE = 0.75
SAMPLE_AFTER = 0.55

# Test the leading edge of pallets moving
# RIGHT -> LEFT toward RFID1.
INCOMING_EDGE_PIXELS = 55

STATION_HORIZONTAL_MARGIN = 6

MAX_SAMPLE_AGE = 3.0
IDENTITY_EXPIRY = 10.0


def station_box():

    if not STATION_CONFIG.exists():

        print(
            "Using confirmed RFID1 box:",
            DEFAULT_BOX
        )

        return DEFAULT_BOX

    with STATION_CONFIG.open(
        "r", encoding="utf-8"
    ) as file:

        config = json.load(file)

    entry = config.get(
        "rfid1", {}
    ).get("box", {})

    required = ("x1", "y1", "x2", "y2")

    if not all(key in entry for key in required):

        print(
            "Using confirmed RFID1 box:",
            DEFAULT_BOX
        )

        return DEFAULT_BOX

    sx = 1920.0 / float(
        config.get("image_width", 1920)
    )

    sy = 1080.0 / float(
        config.get("image_height", 1080)
    )

    box = (
        round(entry["x1"] * sx),
        round(entry["y1"] * sy),
        round(entry["x2"] * sx),
        round(entry["y2"] * sy),
    )

    print("Loaded RFID1 box:", box)

    return box


# ============================================================
# BACKGROUND RFID1 READER
# ============================================================

def poll_rfid1(events, stop):

    try:

        from pylogix import PLC

    except ImportError:

        print(
            "RFID1 ERROR: pylogix missing."
        )

        return

    last_emitted = None
    last_sample = None

    consecutive = 0
    absence_count = 0

    last_error = 0.0

    while not stop.is_set():

        try:

            with PLC() as plc:

                plc.IPAddress = PLC_IP
                plc.SocketTimeout = 3

                print(
                    "RFID1-only polling:",
                    PLC_IP,
                    "words [96..99]",
                    flush=True
                )

                while not stop.is_set():

                    values = []

                    for tag in RFID1_WORDS:

                        result = plc.Read(tag)

                        if result.Status != "Success":

                            raise RuntimeError(
                                f"{tag}: {result.Status}"
                            )

                        values.append(
                            int(result.Value)
                        )

                    words = tuple(values)
                    timestamp = time.monotonic()

                    if words in TAG_TO_PALLET:

                        absence_count = 0

                        if words == last_sample:
                            consecutive += 1

                        else:
                            consecutive = 1
                            last_sample = words

                        # Require two consecutive
                        # identical valid RFID reads.
                        if (
                            consecutive >= 2
                            and words != last_emitted
                        ):

                            last_emitted = words

                            events.put({
                                "number":
                                    TAG_TO_PALLET[words],
                                "time": timestamp,
                                "raw": words,
                            })

                    else:

                        # Invalid partial PLC words
                        # are not accepted as tags.
                        consecutive = 0
                        last_sample = None

                        absence_count += 1

                        if absence_count >= 3:
                            last_emitted = None

                    stop.wait(0.04)

        except Exception as error:

            now = time.monotonic()

            if now - last_error >= 3.0:

                print(
                    "RFID1 read/connection problem:",
                    error,
                    flush=True
                )

                last_error = now

            stop.wait(1.0)


# ============================================================
# RFID ASSOCIATION
# ============================================================

class Mapper:

    def __init__(self, events, box):

        self.events = events
        self.box = box

        self.sightings = deque()
        self.pending = deque()

        # IMPORTANT:
        # Actual numeric tracker IDs.
        # Never use id(data).
        self.owner = {}

        self.last_seen = {}
        self.last_boxes = {}

        self.last_image = None

        self.current_read = None
        self.current_read_time = 0.0

        filename = (
            "rfid1_only_"
            + datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )
            + ".csv"
        )

        self.logfile = RESULTS / filename

        with self.logfile.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            csv.writer(file).writerow([
                "time",
                "event",
                "pallet",
                "track",
                "details"
            ])

    # ========================================================
    # LOG
    # ========================================================

    def log(
        self,
        kind,
        number="",
        track="",
        detail=""
    ):

        print(
            f"RFID1 {kind}: "
            f"pallet={number} "
            f"track={track} "
            f"{detail}",
            flush=True
        )

        with self.logfile.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as file:

            csv.writer(file).writerow([
                datetime.now().isoformat(
                    timespec="seconds"
                ),
                kind,
                number,
                track,
                detail,
            ])

    # ========================================================
    # ID TRANSFER THROUGH ORIGINAL STATE BRIDGE
    # ========================================================

    def transfer(self, previous, current):

        if previous == current:
            return

        if previous not in self.owner:
            return

        number = self.owner.pop(previous)

        if (
            current in self.owner
            and self.owner[current] != number
        ):

            self.owner.pop(current, None)

            self.log(
                "BRIDGE_CONFLICT_BOTH_UNKNOWN",
                number,
                current
            )

            return

        self.owner[current] = number

        self.log(
            "STATE_BRIDGE",
            number,
            current,
            f"from={previous}"
        )

    # ========================================================
    # DIAGNOSTIC POSITIONS
    # ========================================================

    def diagnose(self, event_time):

        sx = (
            self.box[0] + self.box[2]
        ) / 2.0

        by = self.box[1]

        latest = []

        for (
            track_id,
            (timestamp, box)
        ) in self.last_boxes.items():

            x1, y1, x2, y2 = box

            cy = (
                y1 + y2
            ) / 2.0

            if (
                by - 175
                <= cy
                <= by + 95
            ):

                latest.append((
                    abs(x1 - sx),
                    track_id,
                    x1,
                    x2,
                    int(cy),
                    round(
                        timestamp - event_time,
                        2
                    ),
                ))

        latest.sort()

        return latest[:5]

    # ========================================================
    # MATCH RFID1 READ TO A PALLET
    # ========================================================

    def find_match(self, event, now):

        event_time = event["time"]
        number = event["number"]

        sx = (
            self.box[0] + self.box[2]
        ) / 2.0

        by = self.box[1]

        # One camera observation nearest in time
        # for each actual tracker ID.
        closest = {}

        for item in self.sightings:

            delta = (
                item["t"] - event_time
            )

            if not (
                -SAMPLE_BEFORE
                <= delta
                <= SAMPLE_AFTER
            ):
                continue

            x1, y1, x2, y2 = item["box"]

            cy = (
                y1 + y2
            ) / 2.0

            # RFID1 is on the upper conveyor.
            # Reject lower-conveyor tracks.
            if not (
                by - 175
                <= cy
                <= by + 95
            ):
                continue

            tid = item["track"]

            if (
                tid not in closest
                or abs(delta)
                < abs(closest[tid][0])
            ):

                closest[tid] = (
                    delta,
                    item["box"]
                )

        inside = []
        approaching = []

        for (
            tid,
            (dt, box)
        ) in closest.items():

            x1, y1, x2, y2 = box

            # Case A:
            # The pallet box spans the
            # RFID1 sensing X position.
            if (
                x1 - STATION_HORIZONTAL_MARGIN
                <= sx
                <= x2 + STATION_HORIZONTAL_MARGIN
            ):

                inside.append((
                    tid,
                    abs(
                        (x1 + x2) / 2.0 - sx
                    ),
                    dt,
                    box
                ))

            # Case B:
            # Pallet approaching from the right,
            # moving toward the RFID1 station.
            elif (
                sx < x1
                <= sx + INCOMING_EDGE_PIXELS
            ):

                approaching.append((
                    tid,
                    x1 - sx,
                    dt,
                    box
                ))

        # ====================================================
        # CASE A: PALLET AT RFID1
        # ====================================================

        if len(inside) == 1:

            candidate = inside[0]
            mode = "BOX_AT_SENSOR"

        elif len(inside) > 1:

            self.log(
                "AMBIGUOUS",
                number,
                detail=(
                    "multiple_boxes_at_sensor="
                    f"{inside}"
                )
            )

            return

        # ====================================================
        # CASE B: APPROACHING PALLET
        # ====================================================

        elif len(approaching) == 1:

            candidate = approaching[0]
            mode = "APPROACHING_RFID1"

        elif len(approaching) > 1:

            self.log(
                "AMBIGUOUS",
                number,
                detail=(
                    "multiple_approaching="
                    f"{approaching}"
                )
            )

            return

        else:

            self.log(
                "NO_UNIQUE_MATCH",
                number,
                detail=(
                    "nearby="
                    f"{self.diagnose(event_time)}"
                )
            )

            return

        # ====================================================
        # VERIFY SELECTED TRACK
        # ====================================================

        tid, distance, dt, box = candidate

        if (
            now - self.last_seen.get(
                tid, -1e9
            ) > 2.0
        ):

            self.log(
                "STALE_CANDIDATE",
                number,
                tid,
                f"mode={mode}"
            )

            return

        previous = self.owner.get(tid)

        # A physical pallet cannot suddenly
        # become another RFID pallet number.
        if (
            previous is not None
            and previous != number
        ):

            self.owner.pop(tid, None)

            self.log(
                "CONFLICT_RESET_TO_QUESTION",
                number,
                tid,
                (
                    f"previous={previous} "
                    f"mode={mode} "
                    f"dt={dt:.2f} "
                    f"box={box}"
                )
            )

            return

        # One RFID pallet number must have
        # only one active tracker owner.
        for (
            old_tid,
            old_number
        ) in list(self.owner.items()):

            if (
                old_tid != tid
                and old_number == number
            ):

                self.owner.pop(
                    old_tid, None
                )

                self.log(
                    "OLD_TRACK_UNMAPPED",
                    number,
                    old_tid
                )

        self.owner[tid] = number

        action = (
            "MAP"
            if previous is None
            else "VERIFY"
        )

        self.log(
            action,
            number,
            tid,
            (
                f"mode={mode} "
                f"time_offset={dt:.2f}s "
                f"dist={distance:.0f}px "
                f"box={box}"
            )
        )

    # ========================================================
    # OBSERVE ORIGINAL PHASE 40 TRACKS
    # ========================================================

    def observe(self, scope):

        now = time.monotonic()

        frame_time = float(
            scope.get("frame_now", now)
        )

        actual_tracks = scope.get(
            "confirmed_tracks", ()
        )

        self.last_boxes = {}

        for track in actual_tracks:

            tid = int(
                track["track_id"]
            )

            box = tuple(
                int(value)
                for value in track["box"]
            )

            self.last_seen[tid] = frame_time

            self.last_boxes[tid] = (
                frame_time,
                box
            )

            self.sightings.append({
                "t": frame_time,
                "track": tid,
                "box": box,
            })

        # Collect RFID1 events.
        while True:

            try:

                event = (
                    self.events.get_nowait()
                )

            except queue.Empty:
                break

            self.pending.append(event)

            self.current_read = (
                event["number"]
            )

            self.current_read_time = now

            self.log(
                "READ",
                event["number"],
                detail=str(event["raw"])
            )

        # Wait briefly for camera observations.
        while (
            self.pending
            and frame_time
            - self.pending[0]["time"]
            >= SAMPLE_AFTER
        ):

            event = self.pending.popleft()

            self.find_match(
                event,
                frame_time
            )

        # Delete expired camera observations.
        while (
            self.sightings
            and now - self.sightings[0]["t"]
            > MAX_SAMPLE_AGE
        ):

            self.sightings.popleft()

        # Remove long-lost tracker identities.
        for (
            tid,
            seen
        ) in list(self.last_seen.items()):

            if now - seen > IDENTITY_EXPIRY:

                self.last_seen.pop(
                    tid, None
                )

                self.owner.pop(
                    tid, None
                )

    # ========================================================
    # DRAW ORIGINAL CLASSIFICATIONS WITH RFID ID
    # ========================================================

    def draw(self, scope):

        source = scope.get(
            "symbol_frame"
        )

        if source is None:

            source = scope.get(
                "frame"
            )

        if source is None:

            raise RuntimeError(
                "Phase 40 camera frame unavailable."
            )

        output = source.copy()

        # Draw original conveyor boundary.
        for (
            polygon_name,
            color
        ) in (
            (
                "outer_polygon",
                (0, 255, 0)
            ),
            (
                "inner_polygon",
                (0, 255, 255)
            ),
        ):

            polygon = scope.get(
                polygon_name
            )

            if polygon is not None:

                cv2.polylines(
                    output,
                    [polygon],
                    True,
                    color,
                    2
                )

        # Draw RFID1 only.
        x1, y1, x2, y2 = self.box

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 255),
            2
        )

        cv2.putText(
            output,
            "RFID1 ONLY",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        sx = (x1 + x2) // 2

        cv2.line(
            output,
            (sx, max(0, y1 - 130)),
            (sx, y2),
            (0, 190, 255),
            1
        )

        # Original Phase 40 display cache.
        visible = scope.get(
            "visible_track_cache", {}
        )

        mapped = 0

        for (
            track_id,
            data
        ) in visible.items():

            tid = int(track_id)

            box = tuple(
                int(value)
                for value in data["box"]
            )

            old_label = str(
                data.get("label", "")
            ).upper()

            status = next(
                (
                    word
                    for word in (
                        "REJECTED",
                        "ACCEPTED",
                        "EMPTY",
                        "CHECKING"
                    )
                    if word in old_label
                ),
                "UNKNOWN"
            )

            number = self.owner.get(tid)

            if number is not None:
                mapped += 1

            identity = (
                str(number)
                if number is not None
                else "?"
            )

            label = (
                f"{identity} {status}"
            )

            color = data.get(
                "color",
                (255, 255, 255)
            )

            bx1, by1, bx2, by2 = box

            thickness = (
                2
                if data.get("missed", 0)
                else 4
            )

            cv2.rectangle(
                output,
                (bx1, by1),
                (bx2, by2),
                color,
                thickness
            )

            cv2.putText(
                output,
                label,
                (
                    max(0, bx1),
                    max(24, by1 - 8)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

        # Status panel.
        cv2.rectangle(
            output,
            (8, 8),
            (700, 85),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            output,
            (
                "RFID1 TEST | "
                f"MAPPED {mapped}/{len(visible)}"
            ),
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.73,
            (255, 255, 255),
            2
        )

        if (
            self.current_read is not None
            and time.monotonic()
            - self.current_read_time < 3
        ):

            detail = (
                f"RFID1 READ: {self.current_read}"
                " | Q quit | S save | R recheck"
            )

        else:

            detail = (
                "? = UNVERIFIED RFID"
                " | Q quit | S save | R recheck"
            )

        cv2.putText(
            output,
            detail,
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1
        )

        self.last_image = output

        return output


# ============================================================
# MAIN - RUN ORIGINAL CONFERENCE PHASE 40
# ============================================================

def main():

    if not LIVE.is_file():

        raise FileNotFoundError(
            f"Conference Phase 40 missing:\n{LIVE}"
        )

    try:
        import pylogix

    except ImportError as error:

        raise RuntimeError(
            "Install pylogix using: "
            "python -m pip install pylogix"
        ) from error

    RESULTS.mkdir(
        parents=True,
        exist_ok=True
    )

    box = station_box()

    spec = (
        importlib.util.spec_from_file_location(
            "crest_phase40_for_rfid1_only",
            str(LIVE)
        )
    )

    if spec is None or spec.loader is None:

        raise RuntimeError(
            "Could not load Conference Phase 40."
        )

    live = (
        importlib.util.module_from_spec(spec)
    )

    sys.modules[spec.name] = live

    spec.loader.exec_module(live)

    if not hasattr(
        live, "transfer_bridged_state"
    ):

        raise RuntimeError(
            "Phase 40 state bridge unavailable."
        )

    events = queue.Queue()
    stop = threading.Event()

    mapper = Mapper(events, box)

    worker = threading.Thread(
        target=poll_rfid1,
        args=(events, stop),
        daemon=True
    )

    old_bridge = (
        live.transfer_bridged_state
    )

    old_show = cv2.imshow
    old_wait = cv2.waitKey

    # Preserve RFID number when the original
    # model transfers a lost track's state.
    def bridge_wrapper(
        old_id,
        new_id,
        *args,
        **kwargs
    ):

        result = old_bridge(
            old_id,
            new_id,
            *args,
            **kwargs
        )

        mapper.transfer(
            int(old_id),
            int(new_id)
        )

        return result

    # Display new identity without changing
    # the original model's classification.
    def show_wrapper(
        window_name,
        frame
    ):

        if (
            "CREST" not in
            str(window_name).upper()
        ):

            return old_show(
                window_name,
                frame
            )

        variables = (
            sys._getframe(1).f_locals
        )

        mapper.observe(variables)

        display = mapper.draw(variables)

        return old_show(
            window_name,
            display
        )

    # Save screenshot of the RFID overlay.
    def wait_wrapper(delay=0):

        key = old_wait(delay)

        if (
            key & 0xFF == ord("s")
            and mapper.last_image is not None
        ):

            filename = (
                "rfid1_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
                + ".png"
            )

            path = RESULTS / filename

            cv2.imwrite(
                str(path),
                mapper.last_image
            )

            print(
                "Saved:",
                path
            )

            return 255

        return key

    # Temporary wrappers.
    live.transfer_bridged_state = (
        bridge_wrapper
    )

    cv2.imshow = show_wrapper
    cv2.waitKey = wait_wrapper

    worker.start()

    print()
    print("=" * 55)
    print("CREST PHASE 44: RFID1 ONLY")
    print("=" * 55)
    print(
        "RFID2: NOT POLLED, "
        "NOT MATCHED, NOT DRAWN"
    )
    print(
        "Phase 40 vision: UNCHANGED"
    )
    print(
        "Previous RFID mappings: NOT REUSED"
    )
    print(
        "PLC operations: READ ONLY"
    )
    print(
        "Q=quit | S=screenshot | R=recheck"
    )
    print("=" * 55)
    print()

    try:

        live.main()

    finally:

        stop.set()

        worker.join(timeout=2)

        live.transfer_bridged_state = (
            old_bridge
        )

        cv2.imshow = old_show
        cv2.waitKey = old_wait

        print(
            "RFID1 log:",
            mapper.logfile
        )


if __name__ == "__main__":
    main()
