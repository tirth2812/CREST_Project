# live_object_detection_on_pallet.py

**Stage:** Intermediate finalized live system
**Source path in package:** `06_Live_Empty_Object_Detection_Final/CODE/live_object_detection_on_pallet.py`

## Purpose

Source module named `live_object_detection_on_pallet`. Its responsibilities should be verified from the functions and source references below; the original file has no module-level description.

## Original file

Copied from `C:\Do_Not_Delete_PLC\Conference\06_Live_Empty_Object_Detection_Final\CODE\live_object_detection_on_pallet.py`. This README is generated; the Python file is unchanged.

## Operational context

This module is retained as a development, annotation, training, live-stage or evaluation artifact. Do not assume it is the default live entrypoint. Read its arguments and source before using it with lab hardware.

## Source code map

These entries are extracted from the copied file; the text in the last column is an overview, not a replacement for reading implementation details.

### Top-level functions

| Function | Line | Responsibility |
|---|---:|---|
| `load_pallet_stage()` | 75 | Handles load pallet stage; precise behavior is in the source. |
| `write_readme()` | 100 | Handles write readme; precise behavior is in the source. |
| `crop_pallet()` | 222 | Extract a pallet region from an image. |
| `normalize_class_name()` | 277 | Handles normalize class name; precise behavior is in the source. |
| `classify_pallet()` | 308 | Handles classify pallet; precise behavior is in the source. |
| `draw_result()` | 354 | Handles draw result; precise behavior is in the source. |
| `main()` | 595 | Run the program's main processing loop. |

## Imported Python modules

`cv2`, `importlib.util`, `pathlib`, `time`, `ultralytics`

## Model filenames visible in source

`improved_empty_vs_object_yolo11n_cls_best.pt`, `pallet_yolo11n_best.pt`

## Preservation

Only this Markdown document was generated; the program source has not been edited.
