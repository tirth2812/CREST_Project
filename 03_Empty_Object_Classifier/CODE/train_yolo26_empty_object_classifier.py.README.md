# train_yolo26_empty_object_classifier.py

**Stage:** EMPTY versus OBJECT_PRESENT classification
**Source path in package:** `03_Empty_Object_Classifier/CODE/train_yolo26_empty_object_classifier.py`

## Purpose

Source module named `train_yolo26_empty_object_classifier`. Its responsibilities should be verified from the functions and source references below; the original file has no module-level description.

## Original file

Copied from `C:\Do_Not_Delete_PLC\Conference\03_Empty_Object_Classifier\CODE\train_yolo26_empty_object_classifier.py`. This README is generated; the Python file is unchanged.

## Operational context

This module is retained as a development, annotation, training, live-stage or evaluation artifact. Do not assume it is the default live entrypoint. Read its arguments and source before using it with lab hardware.

## Source code map

These entries are extracted from the copied file; the text in the last column is an overview, not a replacement for reading implementation details.

### Top-level functions

| Function | Line | Responsibility |
|---|---:|---|
| `collect_images()` | 185 | Handles collect images; precise behavior is in the source. |
| `normalize_class()` | 206 | Handles normalize class; precise behavior is in the source. |
| `verify_dataset()` | 237 | Handles verify dataset; precise behavior is in the source. |
| `evaluate_validation()` | 412 | Handles evaluate validation; precise behavior is in the source. |
| `main()` | 631 | Run the program's main processing loop. |

## Imported Python modules

`csv`, `json`, `pathlib`, `shutil`, `torch`, `ultralytics`

## Model filenames visible in source

`best.pt`, `empty_object_yolo26n_cls_best.pt`, `empty_object_yolo26n_cls_best_BEFORE_FAIR_RETRAIN.pt`, `yolo26n-cls.pt`

## Preservation

Only this Markdown document was generated; the program source has not been edited.
