# Stage 03 — Empty vs Object-Present Classification

**CREST | Model 2 of the final vision pipeline**

[Previous: Stage 02](../02_YOLO_Pallet_Detection/README.md) · [Project overview](../README.md) · **Next: [Stage 04 — Live Empty/Object System](../04_Live_Empty_Object_System/README.md)**

## 1. What this stage is for

Once the detector finds a pallet, the next question is simple: **is the pallet empty, or is an inspection object on top of it?** Stage 03 trains a **YOLO26n classification model** to answer that question.

The two classes are:

- **`EMPTY`** — the pallet has no inspection object.
- **`OBJECT_PRESENT`** — an object is visible on the pallet and may need a later symbol check.

This classifier does **not** decide ACCEPTED vs REJECTED. Those final categories depend on an additional `X` / `NO_X` marking classifier introduced in the later integrated system.

## 2. Why it is a separate model

The first model detects the pallet's location in the *whole image*. The second model receives an image **crop of an individual pallet** and judges its contents. This separation prevents the object classifier from needing to search the entire conveyor image.

```text
Stage 02 pallet box
   ↓
Crop image around the pallet
   ↓
YOLO26n EMPTY / OBJECT_PRESENT classifier
   ↓
EMPTY  or  OBJECT_PRESENT
```

## 3. Recorded classification dataset

The Stage 03 dataset consists of **148 labeled classification images**, evenly divided between the two categories.

| Split | EMPTY | OBJECT_PRESENT | Total |
|---|---:|---:|---:|
| Training | 57 | 57 | 114 |
| Validation | 15 | 15 | 30 |
| Test | 2 | 2 | 4 |
| **Total** | **74** | **74** | **148** |

These are labeled image crops or classification samples, **not** the 100 full camera frames from the detector training set. Source images can have different pixel sizes; the inference input is resized/prepared for the model.

## 4. Recorded classifier setup

| Setting | Recorded value |
|---|---|
| Model family | YOLO26n classification (`-cls`) |
| Classes | `EMPTY`, `OBJECT_PRESENT` |
| Input size | 320 |
| Configured maximum epochs | 100 |
| Early-stopping patience | 20 |
| Batch size | 16 |
| Initial learning rate | 0.0003 |
| Weight decay | 0.0005 |
| Random seed | 42 |

These are the recorded settings of the available training implementation. The selected deployment checkpoint and completed-epoch history should be checked against saved run artifacts when reporting exact training duration.

## 5. Original validation and held-out results

| Evaluation | Correct / total | Accuracy |
|---|---:|---:|
| Validation | 30 / 30 | 100% |
| Held-out test | 4 / 4 | 100% |

Both test-class counts were **2/2 correct**. The held-out test set is very small, so these results do **not** establish perfect performance in all live conditions.

Live tests on moving pallets are a separate evaluation. Occlusions, lighting, box crops, and newly placed/removed objects can make live operation harder than testing isolated saved images.

## 6. Code and project files

- **`CODE/`** — dataset preparation, training, evaluation, or inference scripts, where tracked.
- **Dataset folders** — classification images organized by class and split.
- **Model or weights folders** — trained classifier checkpoint(s), normally with the `.pt` extension.
- **Results or training-output folders** — logs, validation plots, and test results, where included.

The model checkpoint is used by the later live stages. A `.pt` file contains trained weights; the `.py` files are the editable process for training or inference. Consult the inventory below for filenames actually present on `conveyor-upload`.

## 7. What this stage delivered

**Output:** a trained binary classifier that differentiates empty pallets from pallets carrying an inspection object. Its prediction can be displayed for a live tracked pallet after a crop is available.

## 8. Important limitations

The classifier depends on the detector's crop. If the detector includes the wrong part of the conveyor or misses part of a pallet, the classification input can be less reliable. When a pallet is partly hidden by a pole or an object is being placed/removed, a single-frame prediction may not represent the true stable condition. Later stages address this using repeated observations and decision management.

## 9. For the professor / reviewer

This stage documents the **second trained neural network**. Stage 04 shows how the detector and this classifier were combined with the real camera.

---

## Files actually included in this Git branch

The list below was generated from Git-tracked files on `conveyor-upload` when this README was installed. It is an inventory, not an instruction to run every script.

**Tracked files in this stage (excluding this README): 165**

### Folder overview
- `CODE/` — 4 tracked file(s)
- `DATASET/` — 159 tracked file(s)
- `MODEL/` — 2 tracked file(s)

### Python source files
- `CODE/train_yolo26_empty_object_classifier.py`

### Model checkpoints
- `MODEL/empty_object_yolo26n_cls_best.pt`

### Configuration files
- No YAML/JSON file is tracked inside this stage.
