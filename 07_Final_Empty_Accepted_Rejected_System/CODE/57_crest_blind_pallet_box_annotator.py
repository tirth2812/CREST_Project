"""CREST Phase 57 - manually label fresh pallet-detector test frames.

No models, camera, PLC, training-data changes, or automatic predictions.
Saves new YOLO class-0 box labels and copies of reviewed test images only.
"""
import csv
import json
import shutil
from pathlib import Path

import cv2

ROOT = Path(r"C:\Do_Not_Delete_PLC\Conference\09_Independent_Blind_Test_Captures")
OUT = ROOT / "_57_Manual_Pallet_Test_Labels"
IMAGES = OUT / "images" / "test"
LABELS = OUT / "labels" / "test"
STATE = OUT / "annotation_progress.json"
MANIFEST = OUT / "ground_truth_manifest.csv"
WINDOW = "CREST 57 - Draw every PALLET (not the item inside)"
MAX_PREVIEW_WIDTH = 1280
MAX_PREVIEW_HEIGHT = 720


def find_sources():
    return sorted(
        p for p in ROOT.glob("session_*/test_frames/*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )


def unique_name(path):
    return path.parent.parent.name + "__" + path.name


def read_progress():
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def save_progress(state, sources):
    OUT.mkdir(parents=True, exist_ok=True)
    temp = STATE.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temp.replace(STATE)
    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as fh:
        columns = ["session", "source_image", "evaluation_image", "status", "pallet_boxes", "note"]
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for path in sources:
            name = unique_name(path)
            entry = state.get(name, {})
            writer.writerow({
                "session": path.parent.parent.name,
                "source_image": str(path),
                "evaluation_image": str(IMAGES / name) if entry.get("status") in ("labeled", "empty") else "",
                "status": entry.get("status", "pending"),
                "pallet_boxes": len(entry.get("boxes", [])),
                "note": entry.get("note", ""),
            })


def update_evaluation_copy(path, status, boxes):
    name = unique_name(path)
    dst_image = IMAGES / name
    dst_label = LABELS / (Path(name).stem + ".txt")
    if status == "unusable":
        # Remove only previous derived evaluation copies, never the originals.
        for file in (dst_image, dst_label):
            if file.exists():
                file.unlink()
        return
    IMAGES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    if not dst_image.exists() or dst_image.stat().st_size != path.stat().st_size:
        shutil.copy2(path, dst_image)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    h, w = image.shape[:2]
    lines = []
    for x1, y1, x2, y2 in boxes:
        xc, yc = (x1 + x2) / (2 * w), (y1 + y2) / (2 * h)
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"0 {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
    dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def save_data_yaml():
    yaml_path = OUT / "data.yaml"
    yaml_path.write_text(
        "path: '" + OUT.as_posix() + "'\n"
        "val: images/test\n"
        "test: images/test\n"
        "names:\n  0: pallet\n",
        encoding="utf-8"
    )
    return yaml_path


class DrawState:
    def __init__(self, image, prior_boxes):
        self.height, self.width = image.shape[:2]
        factor = min(MAX_PREVIEW_WIDTH / self.width, MAX_PREVIEW_HEIGHT / self.height, 1.0)
        self.preview_w = max(1, round(self.width * factor))
        self.preview_h = max(1, round(self.height * factor))
        self.background = cv2.resize(image, (self.preview_w, self.preview_h))
        self.boxes = [list(b) for b in prior_boxes]
        self.start = None
        self.cursor = None

    def orig(self, x, y):
        return (max(0, min(self.width - 1, round(x * self.width / self.preview_w))),
                max(0, min(self.height - 1, round(y * self.height / self.preview_h))))

    def mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start = self.orig(x, y)
            self.cursor = self.start
        elif event == cv2.EVENT_MOUSEMOVE and self.start is not None:
            self.cursor = self.orig(x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.start is not None:
            xx, yy = self.orig(x, y)
            x1, x2 = sorted((self.start[0], xx))
            y1, y2 = sorted((self.start[1], yy))
            if x2 - x1 >= 12 and y2 - y1 >= 12:
                self.boxes.append([x1, y1, x2, y2])
            self.start = None
            self.cursor = None

    def display(self, number, total, name, previous_status, notice):
        canvas = self.background.copy()
        for i, (x1, y1, x2, y2) in enumerate(self.boxes, 1):
            a = (round(x1 * self.preview_w / self.width), round(y1 * self.preview_h / self.height))
            b = (round(x2 * self.preview_w / self.width), round(y2 * self.preview_h / self.height))
            cv2.rectangle(canvas, a, b, (0, 255, 0), 2)
            cv2.putText(canvas, str(i), (a[0], max(65, a[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        if self.start is not None and self.cursor is not None:
            a = (round(self.start[0] * self.preview_w / self.width),
                 round(self.start[1] * self.preview_h / self.height))
            b = (round(self.cursor[0] * self.preview_w / self.width),
                 round(self.cursor[1] * self.preview_h / self.height))
            cv2.rectangle(canvas, a, b, (255, 255, 0), 2)
        cv2.rectangle(canvas, (0, 0), (self.preview_w, 62), (30, 30, 30), -1)
        captions = [
            f"IMAGE {number}/{total} | {name} | PALLETS: {len(self.boxes)} | previous: {previous_status}",
            "DRAG box | ENTER/S save+next | Z undo | R clear | B back | E empty | X unusable | Q quit",
            notice,
        ]
        for n, txt in enumerate(captions):
            cv2.putText(canvas, txt[:112], (8, 18 + 19 * n),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                        (0, 220, 255) if n == 2 else (240, 240, 240),
                        1, cv2.LINE_AA)
        return canvas


def main():
    sources = find_sources()
    if not sources:
        raise SystemExit("No Phase 55 test_frames found at " + str(ROOT))
    state = read_progress()
    unknown = [p for p in state if p not in {unique_name(s) for s in sources}]
    if unknown:
        print("WARNING: Saved annotations for absent source frames:", unknown)
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    index = next((i for i, path in enumerate(sources)
                  if state.get(unique_name(path), {}).get("status") not in
                  ("labeled", "empty", "unusable")), 0)
    print(f"CREST PHASE 57 - MANUAL BLIND BOX ANNOTATION ({len(sources)} frames)")
    print("Draw tight boxes around EVERY visible pallet, not the object/mark on it.")
    print("ENTER/S saves boxes. E confirms genuinely zero pallets; X marks truly unusable.")
    print("Z undo, R clear, B previous, Q quit and resume later.")
    print("Never use YOLO predictions to choose the ground-truth boxes.")
    try:
        while 0 <= index < len(sources):
            path = sources[index]
            name = unique_name(path)
            frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if frame is None:
                print("UNREADABLE: " + str(path) + "; flag X and continue")
                index += 1
                continue
            prior = state.get(name, {})
            screen = DrawState(frame, prior.get("boxes", []))
            cv2.setMouseCallback(WINDOW, screen.mouse)
            notice = ""
            while True:
                cv2.imshow(WINDOW, screen.display(
                    index + 1, len(sources), name, prior.get("status", "pending"), notice))
                key = cv2.waitKey(35) & 0xFF
                if key == 255:
                    continue
                if key in (ord("z"), ord("Z")) and screen.boxes:
                    screen.boxes.pop()
                elif key in (ord("r"), ord("R")):
                    screen.boxes.clear()
                elif key in (ord("b"), ord("B")):
                    index = max(0, index - 1)
                    break
                elif key in (ord("q"), ord("Q"), 27):
                    index = len(sources)
                    break
                elif key in (13, 10, ord("s"), ord("S")):
                    if not screen.boxes:
                        notice = "NO BOXES: draw pallets; press E only if truly no pallet."
                        continue
                    status = "labeled"
                    state[name] = {"status": status, "boxes": screen.boxes, "note": ""}
                    update_evaluation_copy(path, status, screen.boxes)
                    save_progress(state, sources)
                    print(f"SAVED {index + 1}/{len(sources)}: {name}, {len(screen.boxes)} pallets", flush=True)
                    index += 1
                    break
                elif key in (ord("e"), ord("E")):
                    if screen.boxes:
                        notice = "Press R to clear boxes BEFORE confirming an empty image."
                        continue
                    state[name] = {"status": "empty", "boxes": [], "note": "confirmed no pallets"}
                    update_evaluation_copy(path, "empty", [])
                    save_progress(state, sources)
                    print(f"SAVED {index + 1}/{len(sources)}: confirmed EMPTY image", flush=True)
                    index += 1
                    break
                elif key in (ord("x"), ord("X")):
                    state[name] = {"status": "unusable", "boxes": [],
                                   "note": "visually unassessable capture; needs review"}
                    update_evaluation_copy(path, "unusable", [])
                    save_progress(state, sources)
                    print(f"FLAGGED {index + 1}/{len(sources)}: unusable (not silently discarded)")
                    index += 1
                    break
    finally:
        cv2.destroyAllWindows()
        save_progress(state, sources)
        yaml_path = save_data_yaml()
        counts = {kind: sum(state.get(unique_name(p), {}).get("status") == kind
                            for p in sources) for kind in ("labeled", "empty", "unusable")}
        completed = sum(counts.values())
        print("\nCREST PHASE 57 SUMMARY")
        print(f"Total captures: {len(sources)} | Reviewed: {completed} | Remaining: {len(sources)-completed}")
        print(f"Labeled with pallets: {counts['labeled']} | Confirmed no pallets: {counts['empty']}")
        print(f"Flagged unusable: {counts['unusable']} (must be reported)")
        print("Annotation manifest: " + str(MANIFEST))
        print("Test data YAML: " + str(yaml_path))
        print("Original Phase 55 recordings and trained models remain untouched.")
        if completed != len(sources):
            print("Run this same script again to resume at the first unfinished image.")
        else:
            print("All images reviewed. Send this output before detector evaluation.")


if __name__ == "__main__":
    main()
