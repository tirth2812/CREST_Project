
from pathlib import Path
import csv

import cv2
import numpy as np


BASE = Path(r"C:\Do_Not_Delete_PLC\Conference")
WORK = BASE / "00_Pallet_100_Annotation_Work"

IMAGES = WORK / "images"
DRAFTS = WORK / "labels_draft"
REVIEWED = WORK / "labels_reviewed"
MANIFEST = WORK / "annotation_manifest.csv"
PROGRESS = WORK / "review_progress.csv"

WINDOW = "CREST - Pallet Annotation Review"
HEADER = 115
MIN_BOX = 10


def screen_size():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        size = (
            root.winfo_screenwidth(),
            root.winfo_screenheight(),
        )
        root.destroy()
        return size
    except Exception:
        return 1366, 768


def read_boxes(path, width, height):
    result = []

    for line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5 or parts[0] != "0":
            raise ValueError(f"Invalid label: {path}")

        cx, cy, bw, bh = map(float, parts[1:])

        x1 = (cx - bw / 2) * width
        y1 = (cy - bh / 2) * height
        x2 = (cx + bw / 2) * width
        y2 = (cy + bh / 2) * height

        result.append([
            max(0, x1),
            max(0, y1),
            min(width, x2),
            min(height, y2),
        ])

    return result


def write_boxes(path, boxes, width, height):
    lines = []

    for x1, y1, x2, y2 in boxes:
        if x2 - x1 < MIN_BOX:
            raise ValueError("Box is too narrow.")

        if y2 - y1 < MIN_BOX:
            raise ValueError("Box is too short.")

        cx = ((x1 + x2) / 2) / width
        cy = ((y1 + y2) / 2) / height
        bw = (x2 - x1) / width
        bh = (y2 - y1) / height

        lines.append(
            f"0 {cx:.6f} {cy:.6f} "
            f"{bw:.6f} {bh:.6f}"
        )

    temp = path.with_suffix(".tmp")
    temp.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


class Reviewer:

    def __init__(self):
        for folder in (IMAGES, DRAFTS):
            if not folder.is_dir():
                raise FileNotFoundError(folder)

        if not MANIFEST.is_file():
            raise FileNotFoundError(MANIFEST)

        self.images = sorted(
            p for p in IMAGES.iterdir()
            if p.suffix.lower() in (
                ".jpg", ".jpeg", ".png", ".bmp"
            )
        )

        if len(self.images) != 100:
            raise RuntimeError(
                f"Expected 100 images, "
                f"found {len(self.images)}"
            )

        for image in self.images:
            if not (DRAFTS / (
                image.stem + ".txt"
            )).is_file():
                raise FileNotFoundError(
                    f"Missing draft for {image.name}"
                )

        with MANIFEST.open(
            newline="", encoding="utf-8"
        ) as stream:
            self.manifest = {
                row["image"]: row
                for row in csv.DictReader(stream)
            }

        self.done = set()

        if PROGRESS.exists():
            with PROGRESS.open(
                newline="", encoding="utf-8"
            ) as stream:
                for row in csv.DictReader(stream):
                    if row["status"] == "REVIEWED":
                        self.done.add(row["image"])

        self.screen_w, self.screen_h = screen_size()

        self.index = 0
        self.boxes = []
        self.undo_stack = []
        self.selected = -1
        self.mode = None
        self.corner = None
        self.start = None
        self.old_box = None
        self.preview = None
        self.dirty = False
        self.message = ""

        self.load(0)

    def load(self, index):
        self.index = index
        self.path = self.images[index]
        self.image = cv2.imread(str(self.path))

        if self.image is None:
            raise RuntimeError(
                f"Cannot read {self.path}"
            )

        self.height, self.width = (
            self.image.shape[:2]
        )

        max_w = max(500, self.screen_w - 120)
        max_h = max(
            350,
            self.screen_h - 180 - HEADER,
        )

        self.scale = min(
            1.0,
            1500 / self.width,
            800 / self.height,
            max_w / self.width,
            max_h / self.height,
        )

        self.view_w = int(self.width * self.scale)
        self.view_h = int(self.height * self.scale)

        label = REVIEWED / (
            self.path.stem + ".txt"
        )

        if not label.is_file():
            label = DRAFTS / (
                self.path.stem + ".txt"
            )

        self.boxes = read_boxes(
            label, self.width, self.height
        )

        self.undo_stack = []
        self.selected = -1
        self.mode = None
        self.preview = None
        self.dirty = False
        self.message = ""

    def point(self, x, y):
        px = max(
            0,
            min(self.width, x / self.scale),
        )
        py = max(
            0,
            min(
                self.height,
                (y - HEADER) / self.scale,
            ),
        )
        return px, py

    def corners(self, box):
        x1, y1, x2, y2 = box
        return [
            (x1, y1),
            (x2, y1),
            (x1, y2),
            (x2, y2),
        ]

    def hit(self, x, y):
        threshold = 13 / self.scale

        for i in range(len(self.boxes) - 1, -1, -1):
            box = self.boxes[i]

            for corner, (cx, cy) in enumerate(
                self.corners(box)
            ):
                if (
                    abs(x - cx) <= threshold
                    and abs(y - cy) <= threshold
                ):
                    return i, corner

        for i in range(len(self.boxes) - 1, -1, -1):
            x1, y1, x2, y2 = self.boxes[i]

            if x1 <= x <= x2 and y1 <= y <= y2:
                return i, None

        return -1, None

    def snapshot(self):
        self.undo_stack.append(
            [box.copy() for box in self.boxes]
        )

    def on_mouse(self, event, x, y, flags, param):
        if y < HEADER and event in (
            cv2.EVENT_LBUTTONDOWN,
            cv2.EVENT_RBUTTONDOWN,
        ):
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            px, py = self.point(x, y)
            index, _ = self.hit(px, py)

            if index >= 0:
                self.snapshot()
                self.boxes.pop(index)
                self.selected = -1
                self.dirty = True
                self.message = "Box deleted - save to confirm"
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            px, py = self.point(x, y)
            index, corner = self.hit(px, py)

            self.selected = index
            self.corner = corner
            self.start = (px, py)
            self.preview = None
            self.message = ""

            if index == -1:
                self.mode = "new"
            else:
                self.snapshot()
                self.old_box = self.boxes[index].copy()

                if corner is None:
                    self.mode = "move"
                else:
                    self.mode = "resize"

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.mode is None:
                return

            px, py = self.point(x, y)
            sx, sy = self.start

            if self.mode == "new":
                self.preview = [
                    min(sx, px),
                    min(sy, py),
                    max(sx, px),
                    max(sy, py),
                ]

            elif self.mode == "move":
                x1, y1, x2, y2 = self.old_box
                bw = x2 - x1
                bh = y2 - y1

                nx1 = max(
                    0,
                    min(self.width - bw, x1 + px - sx),
                )
                ny1 = max(
                    0,
                    min(self.height - bh, y1 + py - sy),
                )

                self.boxes[self.selected] = [
                    nx1, ny1, nx1 + bw, ny1 + bh
                ]

                self.dirty = True

            elif self.mode == "resize":
                box = self.old_box.copy()
                minimum = MIN_BOX

                if self.corner in (0, 2):
                    box[0] = min(px, box[2] - minimum)
                else:
                    box[2] = max(px, box[0] + minimum)

                if self.corner in (0, 1):
                    box[1] = min(py, box[3] - minimum)
                else:
                    box[3] = max(py, box[1] + minimum)

                self.boxes[self.selected] = box
                self.dirty = True

        elif event == cv2.EVENT_LBUTTONUP:
            if self.mode == "new":
                if self.preview is not None:
                    x1, y1, x2, y2 = self.preview

                    if (
                        x2 - x1 >= MIN_BOX
                        and y2 - y1 >= MIN_BOX
                    ):
                        self.snapshot()
                        self.boxes.append(
                            self.preview.copy()
                        )
                        self.selected = len(
                            self.boxes
                        ) - 1
                        self.dirty = True

            self.mode = None
            self.preview = None

    def save_progress(self):
        temp = PROGRESS.with_suffix(".tmp")

        with temp.open(
            "w", newline="", encoding="utf-8"
        ) as stream:

            writer = csv.writer(stream)
            writer.writerow([
                "image",
                "source_type",
                "status",
            ])

            for image in self.images:
                source_type = self.manifest.get(
                    image.name, {}
                ).get("source_type", "unknown")

                status = (
                    "REVIEWED"
                    if image.name in self.done
                    else "PENDING"
                )

                writer.writerow([
                    image.name,
                    source_type,
                    status,
                ])

        temp.replace(PROGRESS)

    def save(self):
        if not self.boxes:
            self.message = (
                "No boxes! Add pallets before saving."
            )
            return False

        REVIEWED.mkdir(
            parents=True, exist_ok=True
        )

        output = REVIEWED / (
            self.path.stem + ".txt"
        )

        write_boxes(
            output,
            self.boxes,
            self.width,
            self.height,
        )

        self.done.add(self.path.name)
        self.save_progress()
        self.dirty = False

        print(
            f"REVIEWED {len(self.done)}/100: "
            f"{self.path.name} | "
            f"{len(self.boxes)} boxes"
        )

        self.message = "Saved and reviewed"
        return True

    def draw(self):
        view = cv2.resize(
            self.image,
            (self.view_w, self.view_h),
        )

        for i, box in enumerate(self.boxes):
            x1, y1, x2, y2 = [
                int(v * self.scale)
                for v in box
            ]

            color = (
                (0, 255, 255)
                if i == self.selected
                else (0, 255, 0)
            )

            cv2.rectangle(
                view,
                (x1, y1),
                (x2, y2),
                color,
                2,
            )

            if i == self.selected:
                for cx, cy in self.corners(box):
                    cv2.circle(
                        view,
                        (
                            int(cx * self.scale),
                            int(cy * self.scale),
                        ),
                        5,
                        (0, 255, 255),
                        -1,
                    )

        if self.preview is not None:
            x1, y1, x2, y2 = [
                int(v * self.scale)
                for v in self.preview
            ]

            cv2.rectangle(
                view,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2,
            )

        status = (
            "REVIEWED"
            if self.path.name in self.done
            else "PENDING"
        )

        if self.dirty:
            status += " / UNSAVED"

        source_type = self.manifest.get(
            self.path.name, {}
        ).get("source_type", "unknown")

        lines = [
            (
                f"Image {self.index + 1}/100 | "
                f"{self.path.name} | {source_type} | "
                f"Boxes: {len(self.boxes)} | {status}"
            ),
            (
                "LEFT drag: add / move / resize | "
                "RIGHT click: delete"
            ),
            (
                "N: save + next | S: save | "
                "P: previous | Z: undo | "
                "B: draft | C: discard | Q: quit"
            ),
            (
                f"Reviewed: {len(self.done)}/100 | "
                f"{self.message}"
            ),
        ]

        header = np.zeros(
            (HEADER, self.view_w, 3),
            dtype=np.uint8,
        )

        for i, line in enumerate(lines):
            cv2.putText(
                header,
                line,
                (10, 23 + i * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return np.vstack([header, view])

    def run(self):
        cv2.namedWindow(
            WINDOW, cv2.WINDOW_AUTOSIZE
        )
        cv2.setMouseCallback(
            WINDOW, self.on_mouse
        )

        print()
        print("CREST PALLET ANNOTATION REVIEW")
        print("Review all 100 images.")
        print("Original labels are protected.")
        print()

        while True:
            cv2.imshow(WINDOW, self.draw())
            key = cv2.waitKey(25) & 0xFF

            if key == ord("s"):
                self.save()

            elif key == ord("n"):
                if self.save():
                    if self.index < 99:
                        self.load(self.index + 1)
                    else:
                        self.message = (
                            "Last image saved. "
                            "Press Q to finish."
                        )

            elif key == ord("p"):
                if self.dirty:
                    self.message = (
                        "Unsaved! Press S or C first."
                    )
                elif self.index > 0:
                    self.load(self.index - 1)

            elif key == ord("z"):
                if self.undo_stack:
                    self.boxes = self.undo_stack.pop()
                    self.selected = -1
                    self.dirty = True
                    self.message = "Undo applied"

            elif key == ord("b"):
                self.snapshot()
                draft = DRAFTS / (
                    self.path.stem + ".txt"
                )
                self.boxes = read_boxes(
                    draft, self.width, self.height
                )
                self.selected = -1
                self.dirty = True
                self.message = (
                    "Restored draft - save if correct"
                )

            elif key == ord("c"):
                self.load(self.index)
                self.message = (
                    "Unsaved edits discarded"
                )

            elif key in (ord("q"), 27):
                if self.dirty:
                    self.message = (
                        "Unsaved! Press S or C first."
                    )
                else:
                    break

            if cv2.getWindowProperty(
                WINDOW, cv2.WND_PROP_VISIBLE
            ) < 1:
                break

        cv2.destroyAllWindows()

        print()
        print(f"REVIEWED: {len(self.done)}/100")
        print(f"REMAINING: {100 - len(self.done)}")

        if len(self.done) == 100:
            print("ALL ANNOTATIONS REVIEWED.")
        else:
            print("Run again to continue reviewing.")


if __name__ == "__main__":
    Reviewer().run()
