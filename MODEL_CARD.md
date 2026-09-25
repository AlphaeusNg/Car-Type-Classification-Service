# Model card

Serving model for the Car Type Classification API. This card describes the
artifact authenticated by `model_manifest.json`. It is not a training report
for every experiment in the repository.

Numbers below are copied from the manifest, `training_runs/mild-aug-v1/`,
the run console log, `PROGRESS.md`, `README.md`, `data/README.md`, or the
ResNet50 notebook output. Where a figure was not recorded, the value is
**unavailable**.

## Selected artifact

| Item | Value |
| --- | --- |
| File | `best_car_model.keras` (gitignored; not distributed by Git) |
| SHA-256 | `a97b7d139d86c9f9fea7b9886e9bc73921e7528295a1644a3d342973f343029c` |
| Size | 347,612,129 bytes |
| Selected | 2026-09-12 from run `mild-aug-v1` |
| Selection metric | validation accuracy only |
| Test split role | final report only; it did not select the checkpoint |
| Loader | Keras 3.10.0 with `compile=False` |
| Class mapping | `class_mapping.json`, SHA-256 `5fc7e7690897eed7a20fcd0971db84181bdf21a661179f353df6b1b7a1d74511`, 15,838 bytes |

`PROGRESS.md` Cycle 44 records that this file was copied from
`training_runs/mild-aug-v1/best_val.keras` outside the trainer. The trainer
does not write the serving file. Keras 3.11.3, 3.12.3, and 3.15.0 fail
deserialization of the current artifacts (`README.md`). A candidate re-export
must be compared with `tools/check_reexport_equivalence.py` and is not promoted
by that tool.

ResNet50 is the original notebook path. It is not this artifact.

## Architecture

Run `mild-aug-v1` printed model `stanford_cars_efficientnetv2s`:

- Input `(None, 224, 224, 3)`.
- In-graph `Rescaling` layer `efficientnet_v2_preprocess`. The manifest input
  contract is RGB float32 in `[0, 1]`; the graph rescales to `[-1, 1]`.
- `efficientnetv2-s` functional backbone, 20,331,360 parameters, output
  `(None, 7, 7, 1280)`.
- Global average pooling `gap`.
- Dropout layer `head_dropout`. The dropout rate is **unavailable** (the run
  summary does not print it).
- Dense `predictions`, 251,076 parameters, output width 196. The run summary
  does not print activation or dtype (**unavailable** there). `README.md`
  describes the served head as a 196-way softmax. Cycle 45 reloaded the
  serving file through `api.utils` and recorded 20,582,436 parameters, input
  `(None, 224, 224, 3)`, output `(None, 196)`, and in-graph
  `Rescaling(2, offset=-1)`.
- Total parameters printed at training time: 20,582,436.

The console log's fine-tune line says the first 230 of 512 backbone layers
were frozen and batch-normalization layers stayed frozen. Head-phase learning
rate is **unavailable** in `epochs.jsonl`. Fine-tune steps in the console log
record `learning_rate: 3e-5`. `PROGRESS.md` Cycle 44 states AdamW, label
smoothing 0.1, and batch size 32. The log shows 230 training steps per epoch
and 252 official-test steps, which is consistent with that batch size, but the
log does not print a batch-size flag.

Augmentation documented for this run (`PROGRESS.md` Cycle 44 and `README.md`):
horizontal flip plus mild brightness/contrast. Numeric jitter magnitudes are
**unavailable** in the run log. Mixup, rotation, and random crop were not used
for this selected run; an earlier candidate that used them is reported below
and was not deployed.

## Preprocessing at serving time

`api.utils.preprocess_image` is the request contract:

1. Accept decoded JPEG or PNG only.
2. Reject non-positive dimensions and images above 50,000,000 pixels.
3. Apply EXIF orientation, convert to RGB, resize to 224×224.
4. Emit one `float32` batch in `[0, 1]` by dividing by 255.

The network then applies the in-graph `[-1, 1]` rescaling. Uploads are limited
to 10 MiB, and the HTTP body is limited to that plus a 64 KiB multipart
allowance. The API returns the top five labels. The JSON field `confidence` is
the softmax class score, not a calibrated probability that the label is correct.

## Label order

The label order is exactly `class_mapping.json` `index_to_class`, string keys
`"0"` through `"195"`. Index `i` is the softmax column `i`. The manifest hash
above authenticates that file.

- `0`: `AM General Hummer SUV 2000`
- `195`: `smart fortwo Convertible 2012`

Labels are Stanford Cars class folder names (make, model, and year), not a
separate taxonomy. Per-class precision and recall are **unavailable**.

## Dataset provenance

Stanford Cars images arranged by class folder, as described in `data/README.md`
and the manifest:

- Source recorded in this repo: Kaggle dataset
  `cyizhuo/stanford-cars-by-classes-folder`.
- 196 classes, 16,185 images: 8,144 under `data/train` and 8,041 under
  `data/test`.
- The images are not part of the Git distribution. License text for the
  dataset is **unavailable** in this repository; `data/README.md` only says to
  comply with the dataset's license.

The selected run's console log reports the same file counts at train time:
8,144 training-directory files, of which 7,330 were the training subset and
814 were the validation subset, plus 8,041 test files.

## Measured metrics

### Selected checkpoint, train-directory holdout

- **Split:** 814 images, the validation subset printed by the run
  (`Using 814 files for validation` out of 8,144 training-directory files).
- **Procedure:** after fine-tuning, the run restored weights from best epoch
  29 (`Restoring model weights from the end of the best epoch: 29`) and
  `evaluate()` printed `val: loss=1.0432 top1=98.28% top5=99.51%`.
  `training_runs/mild-aug-v1/metrics.json` stores that same evaluation.
  Checkpoint choice during training used `val_accuracy`. This holdout is not
  the official Stanford Cars test split.
- **Results:** accuracy `0.9828009605407715`, top-5 accuracy
  `0.9950860142707825`, loss `1.0432065725326538`.

The final fine-tune epoch logged in `status.json` (`epoch` 30) is a different
procedure (that epoch's fit logs). Its validation top-5
(`0.9938575029373169`) and validation loss (`1.0485754013061523`) do not match
the restored-checkpoint evaluation above. Do not quote epoch 30 as the
selected checkpoint.

### Selected checkpoint, official Stanford Cars test

- **Split:** 8,041 images under `data/test`, the official test directories.
- **Procedure:** final report after selection. The trainer prints
  `test: loss=1.6275 top1=81.88% top5=95.54%`. `metrics.json` and
  `model_manifest.json` store this evaluation. The manifest field
  `test_role` is `final_report_only`.
- **Results:** accuracy `0.818803608417511`, top-5 accuracy
  `0.9553537964820862`, loss `1.6275264024734497` (loss is in `metrics.json`;
  the manifest records the two accuracies and the image count only).

### Second full test load (Cycle 45)

- **Split:** 8,041 images (`PROGRESS.md` Cycle 45).
- **Procedure:** the serving file was loaded through `api.utils` and evaluated
  again. This is a separate measurement from `metrics.json`.
- **Results:** top-1 `81.89%`, top-5 `95.55%` as rounded in `PROGRESS.md`.
  Unrounded values are **unavailable**.

### Single-image smoke checks

These are not split metrics.

- Cycle 44: a Hummer test JPEG was predicted correctly at `0.73` after an API
  load. The exact path and unrounded score are **unavailable**.
- Cycle 45: the same style of check is described as a correct Hummer
  prediction. The score for that repetition is **unavailable**.

### Original ResNet50 notebook (not the serving artifact)

`model_training.ipynb` cell "MODEL EVALUATION AND RESULTS" writes
`model.evaluate` results. The cell source uses a 20% `validation_split` of the
training directory with seed 123, then the test directory. The captured stdout
prints validation loss `1.9403`, accuracy `0.5688`, top-5 `0.8237`, and then a
distinct test block: loss `1.8129`, accuracy `0.5869`, top-5 `0.8439`. The
stdout also repeats the validation figures under an earlier "Test Set Results"
heading; that repeated block is not a second test result. Image counts for
this notebook run are **unavailable** in the captured output. `README.md`
rounds the same validation top-1 to `56.88%` and the test result to `58.69%`
top-1 / `84.39%` top-5.

`PROGRESS.md` Cycle 44 says the API's `[0, 1]` preprocessing did not match
that ResNet artifact's `[0, 255]` training. The notebook markdown also says
pixels were scaled to `[0, 1]`. Those statements disagree, and the saved
ResNet graph was not re-measured for this card, so the historical input scale
is **not reconciled** here. `best_car_model.previous.keras` is the retained
previous weights (`PROGRESS.md`). It is not the manifest-selected model.

### Rejected EfficientNet candidate (not deployed)

`training_run/metrics.json` records a different run with `"promoted": false`:

- Validation: accuracy `0.9410319328308105`, top-5 `0.9631449580192566`,
  loss `1.3021519184112549`. Image count **unavailable** in that file.
- Test: accuracy `0.4307921826839447`, top-5 `0.6944409608840942`, loss
  `2.997074604034424`. Image count **unavailable** in that file.
- Procedure details beyond the stored `val` and `test` blocks are the
  Cycle 43 note: this experiment used mixup, rotation, and random crop, reached
  about `94.10%` validation accuracy and `43.08%` test accuracy, and was not
  deployed. `training_history.json` is that experimental trace, not the
  serving checkpoint's history.

## Limits

- Official-test top-1 (`0.8188`) is much lower than the 814-image train
  holdout (`0.9828`). The holdout must not be quoted as field accuracy.
- Softmax scores are not calibrated. Expected calibration error and any
  reliability diagram are **unavailable**. Do not treat `confidence` as the
  probability that the predicted class is correct.
- No per-class, geographic, or demographic metrics are recorded.
- The service accepts JPEG and PNG only, resizes to 224×224, and will not
  score other formats or images above the pixel or byte limits.
- Near-duplicate body styles and model years remain confusable; the label set
  is fine-grained make/model/year.
- Keras newer than 3.10.0 is not a drop-in loader for this file.
- Weights are local. A clone with only the manifest cannot serve predictions.
- One process admits one model call at a time and two preprocess workers.
  Shutdown waits for work that already holds a lane; it does not make that
  work preemptable.
