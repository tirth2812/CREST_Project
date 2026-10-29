# train_yolo26_pallet_localizer.py

**Stage:** Earlier pallet detection training and evaluation
**Source path in package:** `02_YOLO_Pallet_Detection/CODE/train_yolo26_pallet_localizer.py`

## Purpose

Source module named `train_yolo26_pallet_localizer`. Its responsibilities should be verified from the functions and source references below; the original file has no module-level description.

## Original file

Copied from `C:\Do_Not_Delete_PLC\Conference\02_YOLO_Pallet_Detection\CODE\train_yolo26_pallet_localizer.py`. This README is generated; the Python file is unchanged.

## Operational context

This module is retained as a development, annotation, training, live-stage or evaluation artifact. Do not assume it is the default live entrypoint. Read its arguments and source before using it with lab hardware.

## Source code map

These entries are extracted from the copied file; the text in the last column is an overview, not a replacement for reading implementation details.

### Top-level functions

| Function | Line | Responsibility |
|---|---:|---|
| `load_json()` | 45 | Handles load json; precise behavior is in the source. |
| `recreate_directory()` | 52 | Handles recreate directory; precise behavior is in the source. |
| `find_source_image()` | 58 | Handles find source image; precise behavior is in the source. |
| `get_annotation_entry()` | 72 | Handles get annotation entry; precise behavior is in the source. |
| `yolo_label_line()` | 86 | Handles yolo label line; precise behavior is in the source. |
| `create_subset()` | 103 | Handles create subset; precise behavior is in the source. |
| `metric_value()` | 155 | Handles metric value; precise behavior is in the source. |
| `evaluate_model()` | 160 | Handles evaluate model; precise behavior is in the source. |
| `save_test_visuals()` | 176 | Handles save test visuals; precise behavior is in the source. |
| `main()` | 192 | Run the program's main processing loop. |

## Imported Python modules

`cv2`, `json`, `pathlib`, `shutil`, `torch`, `ultralytics`

## Model filenames visible in source

`best.pt`, `pallet_yolo26n_best.pt`, `yolo26n.pt`

## Preservation

Only this Markdown document was generated; the program source has not been edited.
