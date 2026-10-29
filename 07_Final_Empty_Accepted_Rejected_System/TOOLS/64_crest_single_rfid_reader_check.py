"""CREST: read-only physical pallet check for ONE RFID reader per run.

No camera, YOLO, PLC writes, motor commands, or piston control.
Examples:
    python 64_crest_single_rfid_reader_check.py --reader 1
    python 64_crest_single_rfid_reader_check.py --reader 2
"""

import argparse
import time
from datetime import datetime

PLC_IP = "192.168.82.231"
PLC_TAG_PREFIX = "IO_Link_Master_ST1_to_ST4:I1.Data"
READER_START = {1: 96, 2: 128}

RFID_MAP = {
    (-8188, 336, -18499, -14824): 1,
    (-8188, 336, -18498, 7750): 2,
    (-8188, 336, -18499, 21259): 3,
    (-8188, 336, -18498, 11138): 4,
    (-8188, 336, -18498, 5934): 5,
    (-8188, 336, -18499, 12969): 6,
}
NO_TAG = (0, 0, 0, 0)


def read_one_reader(plc, tags):
    """Read four words of exactly one RFID reader (read-only)."""
    responses = plc.Read(tags)
    if not isinstance(responses, (list, tuple)):
        raise RuntimeError(
            f"Unexpected PLC response: {getattr(responses, 'Status', responses)}"
        )
    if len(responses) != 4:
        raise RuntimeError(f"Expected four PLC words, received {len(responses)}")

    values = []
    for tag, response in zip(tags, responses):
        if response.Status != "Success" or response.Value is None:
            raise RuntimeError(f"{tag}: {response.Status}")
        values.append(int(response.Value))
    return tuple(values)


def stamp():
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def main():
    parser = argparse.ArgumentParser(
        description="Check physical pallets using one CREST RFID reader at a time"
    )
    parser.add_argument("--reader", type=int, choices=(1, 2), required=True,
                        help="1 = PLC words 96-99; 2 = PLC words 128-131")
    args = parser.parse_args()

    start = READER_START[args.reader]
    tags = [f"{PLC_TAG_PREFIX}[{i}]" for i in range(start, start + 4)]

    print(f"CREST RFID READER {args.reader} - READ ONLY", flush=True)
    print(f"PLC IP: {PLC_IP}", flush=True)
    print(f"Input words: {start}-{start + 3}", flush=True)
    print("Only one reader is polled in this run.", flush=True)
    print("Watch whether the tagged pallet is at the TOP or BOTTOM reader.", flush=True)
    print("Press Ctrl+C to stop. No PLC writes are used.\n", flush=True)

    try:
        from pylogix import PLC
    except ImportError:
        raise SystemExit(
            "pylogix is missing. In vlm_env run: python -m pip install pylogix"
        )

    last_pallet = None
    consecutive_empty = 0
    consecutive_errors = 0
    unrecognized_last = None

    try:
        with PLC() as plc:
            plc.IPAddress = PLC_IP
            plc.SocketTimeout = 3
            while True:
                try:
                    words = read_one_reader(plc, tags)
                    if consecutive_errors:
                        print(f"[{stamp()}] PLC communication restored", flush=True)
                    consecutive_errors = 0
                except Exception as exc:
                    consecutive_errors += 1
                    if consecutive_errors == 1 or consecutive_errors % 10 == 0:
                        print(f"[{stamp()}] PLC READ ERROR: {exc}", flush=True)
                    time.sleep(0.3)
                    continue

                if words == NO_TAG:
                    consecutive_empty += 1
                    if consecutive_empty >= 2:
                        last_pallet = None
                        unrecognized_last = None
                else:
                    consecutive_empty = 0
                    pallet = RFID_MAP.get(words)
                    if pallet is not None:
                        unrecognized_last = None
                        if pallet != last_pallet:
                            print(
                                f"[{stamp()}] RFID{args.reader} -> "
                                f"PHYSICAL PALLET {pallet} | raw={words}",
                                flush=True,
                            )
                            last_pallet = pallet
                    elif words != unrecognized_last:
                        print(
                            f"[{stamp()}] RFID{args.reader} -> "
                            f"UNRECOGNIZED/transient raw={words}",
                            flush=True,
                        )
                        unrecognized_last = words

                time.sleep(0.08)
    except KeyboardInterrupt:
        print("\nStopped. No PLC values were changed.", flush=True)


if __name__ == "__main__":
    main()
