
"""CREST Conference - Phase 42 RFID Mapping Test.

Runs the existing Phase 41 wrapper and Phase 40 model.
Replaces only RFID-to-vision association logic.

Unknown identity: ? EMPTY / ? ACCEPTED / ? REJECTED
Matched identity: 1-6 with the existing classification

No PLC writes, piston commands, or conveyor stopping.

S = save image
R = existing model recheck
Q = quit
"""

from collections import deque
from pathlib import Path
import importlib.util
import queue
import sys
import time


PHASE41 = Path(
    r"C:\Do_Not_Delete_PLC\Conference"
    r"\07_Final_Empty_Accepted_Rejected_System"
    r"\CODE\41_conference_rfid_identity_test.py"
)

# Observe camera frames before and after an RFID event.
BEFORE_SECONDS = 1.25
AFTER_SECONDS = 0.85

# Minimum fraction of station rectangle overlapped
# by a detected pallet box.
MIN_STATION_COVERAGE = 0.10

TRACK_EXPIRY_SECONDS = 20.0


def load_previous_test():

    if not PHASE41.is_file():
        raise FileNotFoundError(
            "Missing Phase 41:\n" + str(PHASE41)
        )

    sys.path.insert(
        0, str(PHASE41.parent)
    )

    spec = importlib.util.spec_from_file_location(
        "crest_conference_rfid41",
        str(PHASE41)
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot load Phase 41 RFID script."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


def station_coverage(
    pallet_box,
    station_box
):
    """Fraction of the RFID station covered by a pallet."""

    ax1, ay1, ax2, ay2 = pallet_box
    bx1, by1, bx2, by2 = station_box

    overlap_width = max(
        0.0,
        min(ax2, bx2) - max(ax1, bx1)
    )

    overlap_height = max(
        0.0,
        min(ay2, by2) - max(ay1, by1)
    )

    station_area = max(
        1.0,
        (bx2 - bx1) * (by2 - by1)
    )

    return (
        overlap_width * overlap_height
        / station_area
    )


def add_station_area_matching(phase41):

    original_association = phase41.Association

    class AreaAssociation(
        original_association
    ):

        def __init__(self, events):

            super().__init__(events)

            self.sightings = deque()
            self.waiting_reads = deque()

            self.latest_boxes = {}
            self.last_assigned = {}

            self.mapped_events = 0

        def observe(self, items):

            now = time.monotonic()

            self.latest_boxes.clear()

            # ------------------------------------
            # Observe existing confirmed tracks.
            # No change to YOLO or classification.
            # ------------------------------------

            for (
                track_id,
                data,
                box,
                status,
                matched
            ) in items:

                if not matched:
                    continue

                key = id(data)

                record = self.tracks.get(key)

                if (
                    record is None
                    or record["object"] is not data
                ):

                    record = {
                        "object": data,
                        "number": None,
                        "center": None,
                        "seen": 0.0,
                        "track_id": track_id
                    }

                    self.tracks[key] = record

                record["seen"] = now

                record["track_id"] = track_id

                record["center"] = (
                    phase41.box_center(box)
                )

                self.latest_boxes[track_id] = (
                    tuple(box)
                )

                # Check overlap with each existing
                # calibrated RFID station box.

                for (
                    station,
                    settings
                ) in phase41.STATIONS.items():

                    coverage = station_coverage(
                        box,
                        settings["box"]
                    )

                    if (
                        coverage
                        >= MIN_STATION_COVERAGE
                    ):

                        self.sightings.append({
                            "station": station,
                            "key": key,
                            "track_id": track_id,
                            "time": now,
                            "coverage": coverage
                        })

            # ------------------------------------
            # Receive RFID readings from PLC.
            # ------------------------------------

            while True:

                try:
                    reading = (
                        self.events.get_nowait()
                    )

                except queue.Empty:
                    break

                number = reading.get(
                    "number"
                )

                station = reading.get(
                    "station", ""
                )

                if number not in range(1, 7):

                    self.log(
                        "UNKNOWN_RFID",
                        station,
                        note=str(
                            reading.get("words")
                        )
                    )

                    continue

                self.waiting_reads.append(
                    reading
                )

                self.log(
                    "RFID_READ",
                    station,
                    number,
                    note=str(
                        reading.get("words")
                    )
                )

            # ------------------------------------
            # Associate an RFID read only if
            # exactly ONE pallet overlaps the
            # station during its time window.
            # ------------------------------------

            while (
                self.waiting_reads
                and
                now
                - self.waiting_reads[0]["time"]
                >= AFTER_SECONDS
            ):

                reading = (
                    self.waiting_reads.popleft()
                )

                station = reading["station"]
                number = reading["number"]
                event_time = reading["time"]

                matches = {}

                for sight in self.sightings:

                    time_difference = (
                        sight["time"]
                        - event_time
                    )

                    valid_time = (
                        -BEFORE_SECONDS
                        <= time_difference
                        <= AFTER_SECONDS
                    )

                    if (
                        sight["station"] == station
                        and valid_time
                    ):

                        key = sight["key"]

                        if (
                            key not in matches
                            or sight["coverage"]
                            >
                            matches[key]["coverage"]
                        ):

                            matches[key] = sight

                # --------------------------------
                # No match or multiple matches:
                # retain UNKNOWN identity.
                # --------------------------------

                if len(matches) != 1:

                    sx1, sy1, sx2, sy2 = (
                        phase41.STATIONS[
                            station
                        ]["box"]
                    )

                    sx = (sx1 + sx2) / 2.0
                    sy = (sy1 + sy2) / 2.0

                    nearby = sorted(
                        (
                            (
                                (
                                    phase41.box_center(
                                        box
                                    )[0] - sx
                                ) ** 2
                                +
                                (
                                    phase41.box_center(
                                        box
                                    )[1] - sy
                                ) ** 2,
                                track_id,
                                box
                            )
                            for (
                                track_id,
                                box
                            ) in self.latest_boxes.items()
                        ),
                        key=lambda item: item[0]
                    )[:3]

                    nearby_boxes = [
                        (
                            track_id,
                            tuple(
                                int(v) for v in box
                            )
                        )
                        for (
                            _,
                            track_id,
                            box
                        ) in nearby
                    ]

                    self.log(
                        "NO_UNIQUE_STATION_OVERLAP",
                        station,
                        number,
                        note=(
                            "overlapping_track_count="
                            f"{len(matches)}; "
                            "nearest_boxes_for_DIAGNOSIS_ONLY="
                            f"{nearby_boxes}"
                        )
                    )

                    continue

                # --------------------------------
                # Exactly one eligible track.
                # --------------------------------

                key, sight = next(
                    iter(matches.items())
                )

                candidate = (
                    self.tracks.get(key)
                )

                if (
                    candidate is None
                    or
                    now - candidate["seen"] > 3.0
                ):

                    self.log(
                        "CANDIDATE_TRACK_LOST",
                        station,
                        number,
                        sight["track_id"]
                    )

                    continue

                # Prevent conflicting numbers from
                # rapidly reusing the same station
                # and vision track.

                prior = self.last_assigned.get(
                    (station, key)
                )

                if (
                    prior is not None
                    and
                    event_time - prior[0] < 1.5
                    and
                    prior[1] != number
                ):

                    self.log(
                        "CONFLICTING_RFID_TOO_SOON",
                        station,
                        number,
                        candidate["track_id"]
                    )

                    continue

                # One physical number cannot belong
                # to two simultaneously tracked pallets.

                for other in self.tracks.values():

                    if (
                        other is not candidate
                        and
                        other["number"] == number
                    ):

                        other["number"] = None

                old_number = (
                    candidate["number"]
                )

                candidate["number"] = number

                if old_number is None:
                    event_name = "MAP"

                elif old_number == number:
                    event_name = "VERIFY"

                else:
                    event_name = "CORRECT"

                self.last_assigned[
                    (station, key)
                ] = (
                    event_time,
                    number
                )

                self.mapped_events += 1

                self.log(
                    event_name,
                    station,
                    number,
                    candidate["track_id"],
                    round(
                        sight["time"] - event_time,
                        3
                    ),
                    (
                        f"old={old_number}; "
                        "station_coverage="
                        f"{sight['coverage']:.2f}"
                    )
                )

            # ------------------------------------
            # Clean expired observations.
            # ------------------------------------

            while (
                self.sightings
                and
                now - self.sightings[0]["time"]
                >
                BEFORE_SECONDS
                + AFTER_SECONDS + 2.0
            ):

                self.sightings.popleft()

            for (
                key,
                record
            ) in list(self.tracks.items()):

                if (
                    now - record["seen"]
                    > TRACK_EXPIRY_SECONDS
                ):

                    del self.tracks[key]

    # Override only the association class
    # used by Phase 41.
    phase41.Association = AreaAssociation


def main():

    phase41 = load_previous_test()

    add_station_area_matching(
        phase41
    )

    print()
    print("=" * 60)
    print("CREST CONFERENCE - RFID PHASE 42")
    print("=" * 60)

    print(
        "100-image pallet model: UNCHANGED"
    )

    print(
        "EMPTY/OBJECT classifier: UNCHANGED"
    )

    print(
        "X/NO_X classifier: UNCHANGED"
    )

    print(
        "RFID association: STATION OVERLAP TEST"
    )

    print(
        "Unknown remains ? until unique match."
    )

    print(
        "PLC: READ ONLY"
    )

    print(
        "S = Save | R = Recheck | Q = Quit"
    )

    print()

    phase41.main()


if __name__ == "__main__":
    main()

