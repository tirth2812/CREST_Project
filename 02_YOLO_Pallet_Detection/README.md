# Earlier pallet detection training and evaluation

**Folder:** `02_YOLO_Pallet_Detection/`

## Goal

Document the original detector baseline before the Conference 100-image retraining.

## Process

Prepare YOLO dataset -> train/evaluate pallet detector -> save baseline best weights -> inspect detections.

## Inputs and artifacts

CODE, DATASET, and MODEL are copied from Conference where available, with CREST as fallback for missing stage folders.

## Stage inventory

- 2 code/source files
- 20 image files
- 1 model checkpoint files

## Browse folders

- [CODE](CODE/README.md)
- [DATASET](DATASET/README.md)
- [MODEL](MODEL/README.md)

## Code files and documentation

- `CODE/evaluate_yolo_pallet_localizer_test.py` — [source explanation](CODE/DOCS/evaluate_yolo_pallet_localizer_test.py.README.md)
- `CODE/train_yolo26_pallet_localizer.py` — [source explanation](CODE/DOCS/train_yolo26_pallet_localizer.py.README.md)

## Checks and limitations

This stage is a baseline; do not substitute its detector checkpoint for the final Stage 00 checkpoint without testing.
