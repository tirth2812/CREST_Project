# Conference 100-image pallet-detector training

**Folder:** `00_Pallet_100_YOLO26_Training/`

## Goal

Build and evaluate the revised pallet-localization detector used by the Conference live pipeline.

## Process

Review corrected pallet labels -> preserve train/val/test split -> train YOLO26 pallet detector -> select best.pt -> compare against the original 20-image model.

## Inputs and artifacts

DATASET/100_IMAGE_DETECTOR stores images, labels and data.yaml from the corrected Conference 100-image dataset. The training-run directory stores its weights and evaluation artifacts.

## Stage inventory

- 0 code/source files
- 118 image files
- 2 model checkpoint files

## Browse folders

- [DATASET](DATASET/README.md)
- [RESULTS](RESULTS/README.md)
- [pallet_yolo26_100_v2](pallet_yolo26_100_v2/README.md)

## Checks and limitations

Compare DATASET/100_IMAGE_DETECTOR/data.yaml, the training args.yaml, and the weights/best.pt file. No fresh retraining is claimed here.
