# EMPTY versus OBJECT_PRESENT classification

**Folder:** `03_Empty_Object_Classifier/`

## Goal

Classify detected pallet crops as EMPTY or OBJECT_PRESENT using a separate image classifier.

## Process

Prepare labeled pallet crops -> preserve train/val/test splits -> train/evaluate classifier -> use frozen checkpoint in live inference.

## Inputs and artifacts

DATASET contains classifier data, MODEL contains its best checkpoint, and CODE stores stage-specific programs.

## Stage inventory

- 1 code/source files
- 148 image files
- 1 model checkpoint files

## Browse folders

- [CODE](CODE/README.md)
- [DATASET](DATASET/README.md)
- [MODEL](MODEL/README.md)

## Code files and documentation

- `CODE/train_yolo26_empty_object_classifier.py` — [source explanation](CODE/DOCS/train_yolo26_empty_object_classifier.py.README.md)

## Checks and limitations

Cross-check class-label ordering with the normalization used by the final inference source; confirm checkpoint before deployment.
