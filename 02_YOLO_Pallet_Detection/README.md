# Stage 02 — YOLO26 Pallet Detection

**CREST | Model 1 of the final vision pipeline**

[Previous: Stage 01](../01_Pallet_Annotation/README.md) · [Project overview](../README.md) · **Next: [Stage 03 — Empty/Object Classification](../03_Empty_Object_Classifier/README.md)**

## 1. What this stage is for

The goal of Stage 02 was to train a vision model to **locate all visible pallets in an image**. We used **YOLO26n object detection**. Its result is a rectangular box and detection confidence for each proposed pallet, not an empty/accepted/rejected label.

This detection step is the foundation of the entire project: the later classifiers need a pallet image crop before they can inspect what is on the pallet.

## 2. How the detector works in CREST

```text
Full conveyor camera image
   ↓
YOLO26n pallet detector
   ↓
Candidate pallet boxes + detection confidence
   ↓
Pallet crops and tracking in later stages
```

The trained model learns from annotated full frames. At runtime, it can identify several pallets in the same image. The tracking system in Stage 05 handles continuity between frames; **detection and tracking are not the same task**.

## 3. Training data

The detector dataset recorded for the final experiment has **100 full-frame images**, each at **1920 × 1080** resolution, with the single class `pallet`.

| Split | Images | Ground-truth pallet boxes |
|---|---:|---:|
| Train | 80 | 761 |
| Validation | 10 | 106 |
| Held-out test | 10 | 105 |
| **Total** | **100** | **972** |

The held-out test split is for assessing predictions on images not used in model training. The classifier datasets in Stage 03 and Stage 07 are separate collections.

## 4. Recorded training setup

| Setting | Recorded value |
|---|---|
| Task | Object detection, one pallet class |
| Model family | YOLO26n |
| Input size | 960 |
| Batch size | 4 |
| Optimizer | AdamW |
| Initial learning rate | 0.0002 |
| Weight decay | 0.0005 |
| Requested epochs | 80 |
| Logged training epochs | 70 |
| Early-stopping patience | 20 |
| Random seed | 42 |
| Output | Best validation checkpoint (`best.pt`) |

The source training run used augmentation to expose the detector to image variation. The requested epoch limit is not the number of completed epochs: the recorded log contains 70 training entries.

## 5. Recorded held-out detector results

The historical 10-image held-out evaluation of this detector reported:

| Detection measure | Recorded result |
|---|---:|
| Precision | 77.29% |
| Recall | 65.71% |
| mAP at IoU 0.50 | 77.01% |
| mAP across IoU 0.50–0.95 | 72.54% |

**What this means:** precision measures how often predicted pallets were correct; recall measures how many annotated pallets were found. Average precision measures both localization and classification quality as the confidence threshold varies. These are **object detection results**, not the overall final-system inspection accuracy.

Subsequent live-frame evaluations used different camera captures and should be reported separately rather than replacing these historical figures.

## 6. Code and results

The `main` branch has been observed to contain these Stage 02 files:

- `CODE/train_yolo26_pallet_localizer.py` — detector training script.
- `CODE/evaluate_yolo_pallet_localizer_test.py` — detector testing/evaluation script.

Check the **file inventory below** to see which files are actually tracked in `conveyor-upload`. The training scripts may contain lab-specific dataset paths, model paths, and GPU settings, so inspect those paths before executing them on a different PC. You do **not** need to retrain the detector to run the final Phase 70 system.

Common saved artifacts from the detector training include `best.pt` weights, training curves or CSV logs, evaluation plots, and dataset YAML configuration, where retained.

## 7. What this stage delivered

**Output:** a pallet localization model that returns detection boxes and allows later software to crop pallets. The model does **not** determine whether a pallet is accepted/rejected or provide the permanent RFID pallet number.

## 8. Important limitations

Pallets at the turns change apparent shape and size. Close pallets or temporary occlusion can cause missing, merged, or duplicate boxes. These cases are addressed *partly* by the tracking and decision logic in later stages, but they are not guaranteed to disappear.

## 9. For the professor / reviewer

This stage demonstrates **how the first neural network was trained and evaluated**. Stage 03 explains the next model, which classifies the contents of each detected pallet.

---

## Files actually included in this Git branch

The list below was generated from Git-tracked files on `conveyor-upload` when this README was installed. It is an inventory, not an instruction to run every script.

**Tracked files in this stage (excluding this README): 59**

### Folder overview
- `CODE/` — 6 tracked file(s)
- `DATASET/` — 51 tracked file(s)
- `MODEL/` — 2 tracked file(s)

### Python source files
- `CODE/evaluate_yolo_pallet_localizer_test.py`
- `CODE/train_yolo26_pallet_localizer.py`

### Model checkpoints
- `MODEL/pallet_yolo26n_best.pt`

### Configuration files
- `DATASET/yolo_dataset/data.yaml`
