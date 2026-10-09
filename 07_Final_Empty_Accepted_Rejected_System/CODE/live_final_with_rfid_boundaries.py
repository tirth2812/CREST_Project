import importlib.util
from pathlib import Path

import cv2


CODE_DIR = Path(__file__).resolve().parent

VISION_FILE = (
    CODE_DIR
    / "live_final_crest_yolo26.py"
)


# ============================================================
# RFID STATION BOUNDARIES ONLY
# ============================================================

RFID1_BOX = (
    1123,
    578,
    1183,
    675
)

RFID2_BOX = (
    823,
    734,
    884,
    815
)


# ============================================================
# LOAD EXISTING FINAL MODEL
# ============================================================

def load_vision_model():

    spec = importlib.util.spec_from_file_location(
        "crest_final_model",
        str(VISION_FILE)
    )

    vision = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        vision
    )

    return vision


# ============================================================
# DRAW RFID BOUNDARIES
# ============================================================

def draw_rfid_boundaries(frame):

    if frame is None:
        return frame

    # RFID 1
    x1, y1, x2, y2 = RFID1_BOX

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "RFID1",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # RFID 2
    x1, y1, x2, y2 = RFID2_BOX

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        frame,
        "RFID2",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    vision = load_vision_model()

    original_imshow = cv2.imshow


    def imshow_with_rfid_boundaries(
        window_name,
        frame
    ):

        frame = draw_rfid_boundaries(
            frame
        )

        original_imshow(
            window_name,
            frame
        )


    cv2.imshow = (
        imshow_with_rfid_boundaries
    )


    vision.main()


if __name__ == "__main__":
    main()