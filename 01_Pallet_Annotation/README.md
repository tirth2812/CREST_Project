# Pallet annotation and label review

**Folder:** `01_Pallet_Annotation/`

## Goal

Prepare reviewed pallet bounding boxes and annotation tools before model training.

## Process

Collect images -> annotate pallet boxes -> review and correct labels -> export YOLO-format text labels -> check image/label correspondence.

## Inputs and artifacts

DATA and CONFERENCE_100/CORRECTED_LABELS preserve source materials and corrected labels. The corrected labels are historical evidence; the final train/val/test split is in Stage 00.

## Stage inventory

- 1 code/source files
- 20 image files
- 0 model checkpoint files

## Browse folders

- [CODE](CODE/README.md)
- [CONFERENCE_100](CONFERENCE_100/README.md)
- [DATA](DATA/README.md)

## Code files and documentation

- `CODE/pallet_annotator.py` — [source explanation](CODE/DOCS/pallet_annotator.py.README.md)

## Checks and limitations

Open matching image and label filenames and check coordinates and class IDs against the dataset YAML.
