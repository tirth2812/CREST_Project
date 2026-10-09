
"""CREST Phase 43: RFID-to-pallet mapping.

Runs the original Conference Phase 40 system unchanged.

Unknown RFID identity:
    ? EMPTY
    ? ACCEPTED
    ? REJECTED

Mapped RFID identity:
    1 EMPTY
    3 ACCEPTED
    5 REJECTED

Uses actual tracker IDs and timestamped station observations.
No PLC writes, conveyor stopping, or piston commands.
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

LIVE_FILE = (
    ROOT
    / "CODE"
    / "40_conference_100_pallet_live.py"
)

STATION_FILE = (
    ROOT
    / "CONFIG"
    / "rfid_stations.json"
)

SAVE_DIR = (
    ROOT
    / "RESULTS"
    / "43_RFID_Track_Mapping"
)


# ============================================================
# RFID CONFIGURATION
# ============================================================

PLC_IP = "192.168.82.231"

SECONDS_BEFORE = 1.65
SECONDS_AFTER = 0.95

MIN_UNIQUE_SCORE_MARGIN = 25.0


RFID_MAP = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}


STATIONS = {
    "RFID1": {
        "box": (1123, 578, 1183, 675),
        "offset": 96,
    },

    "RFID2": {
        "box": (823, 734, 884, 815),
        "offset": 128,
    },
}


# ============================================================
# LOAD PREVIOUSLY CALIBRATED RFID STATION BOXES
# ============================================================

def load_station_boxes():

    if not STATION_FILE.is_file():

        print(
            "Station configuration not found; "
            "using verified 1920x1080 boxes."
        )

        return

    with STATION_FILE.open(
        encoding="utf-8"
    ) as handle:

        config = json.load(handle)

    sx = 1920.0 / float(
        config.get("image_width", 1920)
    )

    sy = 1080.0 / float(
        config.get("image_height", 1080)
    )

    for name, station in STATIONS.items():

        entry = config.get(
            name.lower(), {}
        ).get("box", {})

        if all(
            key in entry
            for key in ("x1", "y1", "x2", "y2")
        ):

            station["box"] = (
                int(round(entry["x1"] * sx)),
                int(round(entry["y1"] * sy)),
                int(round(entry["x2"] * sx)),
                int(round(entry["y2"] * sy)),
            )

    print(
        "RFID station boxes:",
        {
            name: setting["box"]
            for name, setting in STATIONS.items()
        }
    )


# ============================================================
# READ RFID FROM PLC
# ============================================================

def rfid_reader(event_queue, stop):

    from pylogix import PLC

    previous = {
        name: None
        for name in STATIONS
    }

    empty_count = {
        name: 0
        for name in STATIONS
    }

    while not stop.is_set():

        try:

            with PLC() as comm:

                comm.IPAddress = PLC_IP
                comm.SocketTimeout = 3

                print(
                    "RFID PLC connected:",
                    PLC_IP
                )

                while not stop.is_set():

                    for name, info in STATIONS.items():

                        base = info["offset"]
                        words = []

                        for i in range(
                            base,
                            base + 4
                        ):

                            result = comm.Read(
                                "IO_Link_Master_ST1_to_ST4:"
                                f"I1.Data[{i}]"
                            )

                            if result.Status != "Success":

                                raise IOError(
                                    f"{name} word {i}: "
                                    f"{result.Status}"
                                )

                            words.append(
                                int(result.Value)
                            )

                        words = tuple(words)

                        # Wait for confirmed tag absence
                        # before allowing another read
                        # of the same pallet.

                        if words == (0, 0, 0, 0):

                            empty_count[name] += 1

                            if empty_count[name] >= 2:
                                previous[name] = None

                            continue

                        # Ignore incomplete / transient
                        # four-word PLC readings.

                        if words not in RFID_MAP:
                            continue

                        empty_count[name] = 0

                        number = RFID_MAP[words]

                        if number != previous[name]:

                            previous[name] = number

                            event_queue.put({
                                "station": name,
                                "number": number,
                                "time": time.monotonic(),
                                "raw": words,
                            })

                    stop.wait(0.05)

        except Exception as error:

            print(
                "RFID PLC reconnect:",
                error
            )

            stop.wait(1.0)


# ============================================================
# GEOMETRY
# ============================================================

def center(box):

    return (
        (box[0] + box[2]) / 2.0,
        (box[1] + box[3]) / 2.0,
    )


# ============================================================
# RFID MAPPING ENGINE
# ============================================================

class RFIDMapper:

    def __init__(self, events):

        self.events = events

        self.samples = deque()
        self.pending = deque()

        # IMPORTANT:
        # Use persistent numeric tracker IDs.
        # Do not use id(data), because Phase 40
        # recreates its display dictionary.
        self.owners = {}

        self.last_seen = {}
        self.last_boxes = {}
        self.last_assign = {}

        self.last_image = None

        self.log_path = (
            SAVE_DIR
            / (
                "rfid_phase43_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S"
                )
                + ".csv"
            )
        )

        with self.log_path.open(
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            csv.writer(file).writerow([
                "time",
                "event",
                "station",
                "pallet",
                "track",
                "detail",
            ])

    # ========================================================
    # EVENT LOGGING
    # ========================================================

    def log(
        self,
        event,
        station="",
        number="",
        track="",
        detail=""
    ):

        print(
            f"{event} "
            f"station={station} "
            f"RFID={number} "
            f"track={track} "
            f"{detail}",
            flush=True
        )

        with self.log_path.open(
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
                track,
                detail,
            ])

    # ========================================================
    # TRANSFER RFID ID THROUGH EXISTING MODEL STATE BRIDGE
    # ========================================================

    def bridge(self, old_id, new_id):

        if old_id == new_id:
            return

        if old_id not in self.owners:
            return

        number = self.owners.pop(old_id)

        if (
            new_id in self.owners
            and self.owners[new_id] != number
        ):

            self.log(
                "BRIDGE_CONFLICT",
                number=number,
                track=new_id
            )

            return

        self.owners[new_id] = number

        self.log(
            "RFID_ID_BRIDGE",
            number=number,
            track=new_id,
            detail=f"old_track={old_id}"
        )

    # ========================================================
    # STATION GEOMETRY
    # ========================================================

    def eligible(self, sample, name):

        x1, y1, x2, y2 = sample["box"]

        bx1, by1, bx2, by2 = (
            STATIONS[name]["box"]
        )

        station_x = (
            bx1 + bx2
        ) / 2.0

        cx, cy = center(
            sample["box"]
        )

        # The station boxes are near the belt edge.
        # Pallet boxes are mostly above RFID1
        # and below RFID2.

        if name == "RFID1":

            in_lane = (
                by1 - 180
                <= cy
                <= by2 + 30
            )

        else:

            in_lane = (
                by1 - 30
                <= cy
                <= by2 + 185
            )

        if not in_lane:
            return False

        # Require horizontal alignment, not
        # a 50% rectangle-overlap threshold.

        return (
            x1 - 35
            <= station_x
            <= x2 + 35
        )

    # ========================================================
    # MATCH ONE RFID EVENT
    # ========================================================

    def match(self, event, now):

        name = event["station"]
        number = event["number"]
        event_time = event["time"]

        candidates = {}

        # Compare each physical track's position
        # against the RFID reader timestamp.

        for sample in self.samples:

            dt = (
                sample["time"]
                - event_time
            )

            if not (
                -SECONDS_BEFORE
                <= dt
                <= SECONDS_AFTER
            ):
                continue

            if not self.eligible(
                sample,
                name
            ):
                continue

            track_id = sample["track"]

            cx, cy = center(
                sample["box"]
            )

            bx1, _, bx2, _ = (
                STATIONS[name]["box"]
            )

            station_x = (
                bx1 + bx2
            ) / 2.0

            # Lower score = better alignment.
            # Position and time both contribute.

            score = (
                abs(cx - station_x)
                + 32.0 * abs(dt)
            )

            old = candidates.get(
                track_id
            )

            # Keep only the best observation
            # for each tracker ID.

            if (
                old is None
                or score < old["score"]
            ):

                candidates[track_id] = {
                    "score": score,
                    "dt": dt,
                    "box": sample["box"],
                }

        ranked = sorted(
            candidates.items(),
            key=lambda item: (
                item[1]["score"]
            )
        )

        diagnostics = "; ".join(
            (
                f"T{tid} "
                f"score={data['score']:.1f} "
                f"dt={data['dt']:.2f} "
                f"box={data['box']}"
            )
            for tid, data in ranked[:4]
        )

        # ----------------------------------------
        # NO PALLET IN STATION AREA
        # ----------------------------------------

        if not ranked:

            closest = []

            for track_id, box in (
                self.last_boxes.items()
            ):

                cx, cy = center(box)

                bx1, _, bx2, _ = (
                    STATIONS[name]["box"]
                )

                distance = abs(
                    cx - (bx1 + bx2) / 2.0
                )

                closest.append((
                    distance,
                    track_id,
                    (int(cx), int(cy))
                ))

            closest.sort()

            self.log(
                "NO_CANDIDATE",
                name,
                number,
                detail=(
                    "closest_current="
                    f"{closest[:3]}"
                )
            )

            return

        # ----------------------------------------
        # REQUIRE UNIQUE BEST MATCH
        # ----------------------------------------

        winner, best = ranked[0]

        if len(ranked) > 1:

            second_score = (
                ranked[1][1]["score"]
            )

            margin = (
                second_score
                - best["score"]
            )

            if (
                margin
                < MIN_UNIQUE_SCORE_MARGIN
            ):

                self.log(
                    "AMBIGUOUS",
                    name,
                    number,
                    detail=(
                        f"margin={margin:.1f}; "
                        f"{diagnostics}"
                    )
                )

                return

        # ----------------------------------------
        # REQUIRE RECENTLY OBSERVED TRACK
        # ----------------------------------------

        if (
            now - self.last_seen.get(
                winner, 0.0
            )
            > 3.0
        ):

            self.log(
                "STALE_TRACK",
                name,
                number,
                winner,
                diagnostics
            )

            return

        # ----------------------------------------
        # PREVENT RAPID CONFLICTING REMAPPING
        # ----------------------------------------

        previous = (
            self.last_assign.get(winner)
        )

        if (
            previous
            and previous[0] != number
            and now - previous[1] < 3.0
        ):

            self.log(
                "CONFLICT",
                name,
                number,
                winner,
                (
                    f"recent_RFID={previous[0]}; "
                    f"{diagnostics}"
                )
            )

            return

        # ----------------------------------------
        # ONE ACTIVE OWNER PER RFID NUMBER
        # ----------------------------------------

        for (
            track_id,
            owned_number
        ) in list(self.owners.items()):

            if (
                owned_number == number
                and track_id != winner
            ):

                del self.owners[track_id]

                self.log(
                    "CLEAR_OLD_OWNER",
                    name,
                    number,
                    track_id
                )

        old_number = (
            self.owners.get(winner)
        )

        self.owners[winner] = number

        self.last_assign[winner] = (
            number,
            now
        )

        if old_number is None:

            action = "MAP"

        elif old_number == number:

            action = "VERIFY"

        else:

            action = "CORRECT"

        self.log(
            action,
            name,
            number,
            winner,
            (
                f"old={old_number}; "
                f"score={best['score']:.1f}; "
                f"dt={best['dt']:.2f}; "
                f"candidates={len(ranked)}; "
                f"{diagnostics}"
            )
        )

    # ========================================================
    # OBSERVE ACTUAL CONFERENCE TRACKS
    # ========================================================

    def observe(self, local_vars):

        now = time.monotonic()

        tracks = local_vars.get(
            "confirmed_tracks",
            []
        )

        self.last_boxes = {}

        for track in tracks:

            track_id = int(
                track["track_id"]
            )

            box = tuple(
                int(value)
                for value in track["box"]
            )

            self.last_seen[track_id] = now

            self.last_boxes[track_id] = box

            self.samples.append({
                "time": now,
                "track": track_id,
                "box": box,
            })

        # ----------------------------------------
        # CONSUME RFID EVENTS
        # ----------------------------------------

        while True:

            try:

                event = (
                    self.events.get_nowait()
                )

            except queue.Empty:
                break

            self.pending.append(event)

            self.log(
                "RFID_READ",
                event["station"],
                event["number"],
                detail=str(
                    event["raw"]
                )
            )

        # ----------------------------------------
        # WAIT FOR CAMERA OBSERVATIONS
        # AFTER THE RFID READ
        # ----------------------------------------

        while (
            self.pending
            and
            now - self.pending[0]["time"]
            >= SECONDS_AFTER
        ):

            event = self.pending.popleft()

            self.match(
                event,
                now
            )

        # ----------------------------------------
        # REMOVE OLD CAMERA HISTORY
        # ----------------------------------------

        while (
            self.samples
            and
            now - self.samples[0]["time"]
            > 5.0
        ):

            self.samples.popleft()

        # ----------------------------------------
        # REMOVE VERY OLD TRACK IDENTITIES
        # ----------------------------------------

        for (
            track_id,
            seen_time
        ) in list(
            self.last_seen.items()
        ):

            if now - seen_time > 20.0:

                self.last_seen.pop(
                    track_id, None
                )

                self.owners.pop(
                    track_id, None
                )

                self.last_assign.pop(
                    track_id, None
                )

    # ========================================================
    # DRAW ACTUAL MODEL CLASSIFICATION + RFID ID
    # ========================================================

    def draw(self, local_vars):

        image = (
            local_vars["symbol_frame"].copy()
        )

        for polygon_name, color in (
            (
                "outer_polygon",
                (0, 255, 0)
            ),
            (
                "inner_polygon",
                (0, 255, 255)
            ),
        ):

            polygon = local_vars.get(
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

        # ----------------------------------------
        # DRAW RFID STATIONS
        # ----------------------------------------

        for name, settings in (
            STATIONS.items()
        ):

            x1, y1, x2, y2 = (
                settings["box"]
            )

            color = (
                (0, 255, 255)
                if name == "RFID1"
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
                name,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

        # ----------------------------------------
        # DRAW PALLETS
        # ----------------------------------------

        visible = local_vars.get(
            "visible_track_cache",
            {}
        )

        mapped = 0

        for track_id, data in (
            visible.items()
        ):

            box = data["box"]

            old_label = str(
                data.get("label", "")
            ).upper()

            status = next(
                (
                    state
                    for state in (
                        "REJECTED",
                        "ACCEPTED",
                        "EMPTY",
                        "CHECKING",
                    )
                    if state in old_label
                ),
                "CHECKING"
            )

            physical_number = (
                self.owners.get(
                    int(track_id)
                )
            )

            if physical_number is not None:
                mapped += 1

            identity = (
                str(physical_number)
                if physical_number is not None
                else "?"
            )

            label = (
                f"{identity} {status}"
            )

            color = data.get(
                "color",
                (255, 255, 255)
            )

            x1, y1, x2, y2 = (
                int(value)
                for value in box
            )

            thickness = (
                4
                if data.get(
                    "missed", 0
                ) == 0
                else 2
            )

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                color,
                thickness
            )

            cv2.putText(
                image,
                label,
                (
                    max(0, x1),
                    max(25, y1 - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2
            )

        # ----------------------------------------
        # STATUS PANEL
        # ----------------------------------------

        cv2.rectangle(
            image,
            (8, 8),
            (790, 75),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            image,
            (
                "RFID PHASE 43 | "
                f"MAPPED {mapped}/{len(visible)}"
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
                "? UNKNOWN RFID | "
                "S SAVE | R RECHECK | Q QUIT"
            ),
            (20, 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1
        )

        self.last_image = image

        return image


# ============================================================
# MAIN - RUN ORIGINAL PHASE 40
# ============================================================

def main():

    if not LIVE_FILE.is_file():

        raise FileNotFoundError(
            str(LIVE_FILE)
        )

    from pylogix import PLC

    SAVE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    load_station_boxes()

    sys.path.insert(
        0,
        str(LIVE_FILE.parent)
    )

    spec = (
        importlib.util.spec_from_file_location(
            "crest_conference_live40",
            LIVE_FILE
        )
    )

    live = (
        importlib.util.module_from_spec(
            spec
        )
    )

    sys.modules[
        spec.name
    ] = live

    spec.loader.exec_module(
        live
    )

    messages = queue.Queue()

    stop = threading.Event()

    mapper = RFIDMapper(
        messages
    )

    thread = threading.Thread(
        target=rfid_reader,
        args=(messages, stop),
        daemon=True
    )

    original_show = cv2.imshow
    original_key = cv2.waitKey

    original_bridge = (
        live.transfer_bridged_state
    )

    # ----------------------------------------
    # PRESERVE RFID ID ON AN EXISTING
    # VERIFIED STATE BRIDGE
    # ----------------------------------------

    def bridge_wrapper(
        old_track_id,
        new_track_id,
        *args,
        **kwargs
    ):

        result = original_bridge(
            old_track_id,
            new_track_id,
            *args,
            **kwargs
        )

        mapper.bridge(
            old_track_id,
            new_track_id
        )

        return result

    # ----------------------------------------
    # REPLACE DISPLAY ONLY
    # ----------------------------------------

    def show_wrapper(
        window_name,
        image
    ):

        if (
            "CREST"
            not in str(window_name).upper()
        ):

            return original_show(
                window_name,
                image
            )

        frame_variables = (
            sys._getframe(1).f_locals
        )

        mapper.observe(
            frame_variables
        )

        display = mapper.draw(
            frame_variables
        )

        return original_show(
            window_name,
            display
        )

    # ----------------------------------------
    # SCREENSHOT HANDLING
    # ----------------------------------------

    def key_wrapper(delay=0):

        key = original_key(
            delay
        )

        if (
            key & 0xFF == ord("s")
            and mapper.last_image is not None
        ):

            path = (
                SAVE_DIR
                / (
                    "rfid_image_"
                    + datetime.now().strftime(
                        "%Y%m%d_%H%M%S_%f"
                    )
                    + ".png"
                )
            )

            cv2.imwrite(
                str(path),
                mapper.last_image
            )

            print(
                "Screenshot saved:",
                path
            )

            return 255

        return key

    # ----------------------------------------
    # APPLY TEMPORARY WRAPPERS
    # ----------------------------------------

    live.transfer_bridged_state = (
        bridge_wrapper
    )

    cv2.imshow = show_wrapper
    cv2.waitKey = key_wrapper

    thread.start()

    print()
    print("=" * 60)
    print("CREST CONFERENCE - RFID PHASE 43")
    print("=" * 60)
    print("Original Phase 40 model: UNCHANGED")
    print("Original classifications: UNCHANGED")
    print("RFID association: TRACK ID + TIME + POSITION")
    print("Unknown physical identity: ?")
    print("PLC access: READ ONLY")
    print("S = Save | R = Recheck | Q = Quit")
    print("=" * 60)
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

        live.transfer_bridged_state = (
            original_bridge
        )

        print(
            "RFID log:",
            mapper.log_path
        )


if __name__ == "__main__":
    main()
