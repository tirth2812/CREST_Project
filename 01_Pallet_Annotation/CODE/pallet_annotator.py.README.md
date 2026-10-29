# pallet_annotator.py

**Stage:** Pallet annotation and label review
**Source path in package:** `01_Pallet_Annotation/CODE/pallet_annotator.py`

## Purpose

Source module named `pallet_annotator`. Its responsibilities should be verified from the functions and source references below; the original file has no module-level description.

## Original file

Copied from `C:\Do_Not_Delete_PLC\Conference\01_Pallet_Annotation\CODE\pallet_annotator.py`. This README is generated; the Python file is unchanged.

## Operational context

This module is retained as a development, annotation, training, live-stage or evaluation artifact. Do not assume it is the default live entrypoint. Read its arguments and source before using it with lab hardware.

## Source code map

These entries are extracted from the copied file; the text in the last column is an overview, not a replacement for reading implementation details.

### Classes

| Class | Line | Responsibility |
|---|---:|---|
| `PalletAnnotator` | 329 | Implements `PalletAnnotator`; see the indicated code section. |

### Top-level functions

| Function | Line | Responsibility |
|---|---:|---|
| `list_class_images()` | 35 | Handles list class images; precise behavior is in the source. |
| `image_feature()` | 54 | Handles image feature; precise behavior is in the source. |
| `select_diverse_images()` | 103 | Handles select diverse images; precise behavior is in the source. |
| `load_annotations()` | 180 | Handles load annotations; precise behavior is in the source. |
| `save_annotations()` | 208 | Handles save annotations; precise behavior is in the source. |
| `build_or_restore_selection()` | 222 | Handles build or restore selection; precise behavior is in the source. |
| `class_values()` | 306 | Handles class values; precise behavior is in the source. |
| `class_color()` | 320 | Handles class color; precise behavior is in the source. |
| `main()` | 844 | Run the program's main processing loop. |

## Imported Python modules

`__future__`, `cv2`, `json`, `numpy`, `pathlib`

## Preservation

Only this Markdown document was generated; the program source has not been edited.
