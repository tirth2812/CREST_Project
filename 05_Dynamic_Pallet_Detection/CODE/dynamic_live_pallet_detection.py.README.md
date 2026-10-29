# dynamic_live_pallet_detection.py

**Stage:** Dynamic detection and tracking
**Source path in package:** `05_Dynamic_Pallet_Detection/CODE/dynamic_live_pallet_detection.py`

## Purpose

Required dynamic tracking module loaded by Phase 40 and Phase 65 at runtime.

## Original file

Copied from `C:\Do_Not_Delete_PLC\Conference\05_Dynamic_Pallet_Detection\CODE\dynamic_live_pallet_detection.py`. This README is generated; the Python file is unchanged.

## Required at runtime

The final scripts load this module from their `CODE` directory. Keep this file alongside Phase 40 and Phase 65 in the archived Stage 07 `CODE/` folder.

The final application uses tracker objects and boundary helpers from this module; do not modify it during documentation cleanup.

## Source code map

These entries are extracted from the copied file; the text in the last column is an overview, not a replacement for reading implementation details.

### Classes

| Class | Line | Responsibility |
|---|---:|---|
| `DynamicTracker` | 905 | Implements `DynamicTracker`; see the indicated code section. |

### Top-level functions

| Function | Line | Responsibility |
|---|---:|---|
| `write_readme()` | 52 | Handles write readme; precise behavior is in the source. |
| `point_distance()` | 239 | Handles point distance; precise behavior is in the source. |
| `normalize_points()` | 246 | Handles normalize points; precise behavior is in the source. |
| `denormalize_points()` | 256 | Handles denormalize points; precise behavior is in the source. |
| `clean_polygon()` | 270 | Handles clean polygon; precise behavior is in the source. |
| `save_boundary()` | 304 | Handles save boundary; precise behavior is in the source. |
| `load_boundary()` | 329 | Handles load boundary; precise behavior is in the source. |
| `create_ring_mask()` | 368 | Handles create ring mask; precise behavior is in the source. |
| `calibration_screen()` | 403 | Handles calibration screen; precise behavior is in the source. |
| `box_iou()` | 755 | Calculate intersection-over-union between boxes. |
| `ring_overlap_ratio()` | 804 | Handles ring overlap ratio; precise behavior is in the source. |
| `box_center_inside_ring()` | 865 | Handles box center inside ring; precise behavior is in the source. |
| `get_yolo_detections()` | 1038 | Handles get yolo detections; precise behavior is in the source. |
| `filter_by_ring()` | 1093 | Handles filter by ring; precise behavior is in the source. |
| `draw_live_frame()` | 1134 | Handles draw live frame; precise behavior is in the source. |
| `main()` | 1318 | Run the program's main processing loop. |

## Imported Python modules

`cv2`, `json`, `math`, `numpy`, `pathlib`, `time`, `ultralytics`

## Model filenames visible in source

`pallet_yolo11n_best.pt`

## Preservation

Only this Markdown document was generated; the program source has not been edited.
