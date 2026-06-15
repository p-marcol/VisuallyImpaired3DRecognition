# VI3DR Training

This package is responsible for training and testing YOLO models. The server
package consumes trained artifacts later; do not put server runtime changes here.

## Dataset layout

Keep `dataset.yaml` next to the dataset, including on an external disk. The
repository only keeps an example in `examples/dataset.example.yaml`.

```yaml
path: .
train: train.txt
val: val.txt
test: test.txt
nc: <number-of-classes>
names: [...]
```

The split text files contain image paths relative to `path` or absolute image
paths. Expected labels are resolved by replacing the `images` path component
with `labels`:

```text
datasets/vi3dr/
  dataset.yaml
  train.txt
  val.txt
  test.txt
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
```

## Train

Install the base dependencies for CPU/MPS environments:

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

For NVIDIA CUDA environments, install the CUDA PyTorch stack first, then the
base requirements:

```bash
./.venv/bin/python -m pip install -r requirements-cuda.txt
./.venv/bin/python -m pip install -r requirements.txt
```

```bash
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --dry-run
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset
```

You can also point directly to the YAML file:

```bash
./.venv/bin/python train.py --data /Volumes/Data/vi3dr-dataset/dataset.yaml
```

The default model source is configured in `config.py` as `yolov8n.pt`.
Use `--model` to pick a different pretrained Ultralytics model or to fine-tune
from an existing checkpoint:

```bash
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --model yolov8s.pt
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --model /path/to/best.pt
```

Use `--from-scratch` to train from random initialization instead of pretrained
weights. By default this uses `yolov8n.yaml`; pass `--scratch-model` to choose a
different YOLO architecture YAML:

```bash
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --from-scratch
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --from-scratch --scratch-model yolov8s.yaml
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --from-scratch --scratch-model /path/to/custom-model.yaml
```

Runs are saved under `runs/` by default. If `--name` is omitted, `train.py`
generates a timestamped run folder name from the model, dataset, device, image
size, learning rate, and seed, for example
`yolo26s_dataset1_cuda_imgsz1024_lr0.01_seed42_11_06_2026_T_13_06`.
Choose the output location with `--runs-dir` and override the run folder name
with `--name`:

```bash
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --runs-dir /Volumes/Data/vi3dr-runs --name experiment-001
```

By default, training uses the dataset as-is. Pass a Python filter file to
`--input-filter` to create a filtered dataset before training. The filter is not
attached to the model during training. Instead, `train.py` writes the filtered
images under `<dataset>/filters/<filter-name>/`, copies matching labels there,
copies the filter source as `filter.py`, and trains on the generated
`dataset.yaml`.

```bash
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --input-filter filters/grayscale.py
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --input-filter filters/sobel.py
```

Generated filtered datasets use the same split-file layout:

```yaml
path: .
train: train.txt
val: val.txt
test: test.txt
nc: 6
names: ["cube", "sphere", "cylinder", "cuboid", "tetrahedron", "cone"]
filter: filter.py
```

Re-run with `--rebuild-filtered-dataset` to recompute existing filtered images
or replace a different copied `filter.py`. If you point `--data` or
`--dataset-dir` directly at a dataset whose YAML contains `filter: filter.py`,
the run name automatically includes the filter name and no extra
`--input-filter` argument is needed.

After filtered training, the run contains two checkpoint forms:

```text
weights/best.pt
weights/best_with_filter.pt
```

`best.pt` is the plain YOLO model and expects already filtered images.
`best_with_filter.pt` has the filter attached to the first layer and expects
original RGB images.

Preview input filters on random images from all dataset splits without writing
files:

```bash
./.venv/bin/python preview_input_filter.py --dataset-dir /Volumes/Data/vi3dr-dataset --input-filter filters/grayscale.py
```

Press space to show another random image. Press `q` or Escape to close the
preview.

Images are loaded by Ultralytics dataloaders. By default this project uses
`--image-cache none`, so images are read lazily from disk batch by batch. You
can override this explicitly:

```bash
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --image-cache auto
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --image-cache none
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --image-cache ram
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --image-cache disk
```

`auto` estimates the selected split after resize and caches decoded images in
RAM only when there is enough available memory with a safety margin.
`ram` is fastest for small datasets but uses memory for decoded/resized images.
`disk` writes Ultralytics `.npy` cache files next to the dataset images and is a
deterministic alternative when disk space is available. `none` avoids
whole-split caching.

Before training, `train.py` writes a normalized Ultralytics dataset config under
`<runs-dir>/_dataset_configs/`. This keeps datasets on external disks working
when `dataset.yaml` or `train.txt`/`val.txt` use relative paths.

At the end of training, `train.py` computes F1 from the final validation
precision and recall in `results.csv`, prints it, and writes `f1_score.json` in
the run directory. It also writes a short `run_stats.json` with `best_epoch`,
best/final precision, recall, F1, `mAP`, `mAP50`, and `mAP50_95`.

## Test

Evaluate a trained checkpoint on the `test` split:

```bash
./.venv/bin/python test.py \
  --dataset-dir /Volumes/Data/vi3dr-dataset \
  --model runs/vi3dr-yolo/weights/best.pt
```

`test.py` accepts the same `--image-cache auto|none|ram|disk` option. In `auto`
mode it bases the decision on the `test` split.

`test.py` loads and normalizes `dataset.yaml` the same way as `train.py`, then
runs Ultralytics validation with `split=test`. Results are written to
`<run-dir>/test`, where `<run-dir>` is inferred from
`<run-dir>/weights/best.pt`. If that directory already contains the expected
test plots, `labels.jpg`, and `f1_score.json`, the script exits without running
evaluation again. The generated `labels.jpg` describes the `test` split label
distribution.
After a fresh evaluation, `test.py` also writes `<run-dir>/test/f1_score.json`
from the test precision and recall.

For filtered runs, evaluate `weights/best.pt` with the filtered dataset and
`weights/best_with_filter.pt` with the original dataset. If `test.py` detects a
likely mismatch, it prints a warning and asks you to type `continue` before it
runs evaluation anyway.

## Epochs, Early Stopping And Resume

YOLO training uses `--epochs` as an upper limit, not as a guarantee that all
epochs will run. Early stopping is controlled by `--patience`: if validation
fitness stops improving for that many epochs, training ends early and keeps the
best checkpoint.

For object detection, the stop decision should be based on validation metrics
such as mAP/precision/recall through Ultralytics fitness, not only on validation
loss. Validation loss is useful for diagnostics, but it is not the main quality
criterion for detections.

Recommended default workflow:

```bash
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --epochs 300 --patience 25
```

The default `--device` is detected from PyTorch in this order: CUDA, MPS, CPU.
You can also force a device explicitly:

```bash
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --device cuda
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --device cuda:0
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --device cpu
```

There is no true unlimited epoch mode. Use a high `--epochs` ceiling with
`--patience`, or use `--patience 0` to disable early stopping and stop manually.
Manual stop keeps checkpoints from completed epochs; `last.pt` is written after
each epoch and `best.pt` is written when validation fitness improves.

Resume an interrupted run from `last.pt`:

```bash
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --resume runs/vi3dr-yolo/weights/last.pt
```

Fine-tune from an existing trained model by using it as `--model`:

```bash
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --model runs/vi3dr-yolo/weights/best.pt --name vi3dr-yolo-finetune
```

## Predict With Hooks

`predict.py` exposes the same pre/post-prediction hook shape that the server can
reuse later.

```bash
./.venv/bin/python predict.py image.jpg --model runs/vi3dr-yolo/weights/best.pt
```

Hooks are regular callables that accept `hooks.PredictionContext`. Configure
defaults in `config.py`:

```python
PRE_PREDICT_HOOKS = ["hooks.clahe_luminance"]
POST_PREDICT_HOOKS = ["hooks.draw_yolo_overlay"]
```

Use pre hooks to alter the image passed into YOLO, and post hooks to alter the
output image or metadata after prediction.
