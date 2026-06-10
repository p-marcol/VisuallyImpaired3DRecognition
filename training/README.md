# VI3DR Training

This package is responsible for training and testing YOLO models. The server
package consumes trained artifacts later; do not put server runtime changes here.

## Dataset layout

Keep `dataset.yaml` next to the dataset, including on an external disk. The
repository only keeps an example in `examples/dataset.example.yaml`.

```yaml
path: .
train: images/train
val: images/val
test: images/test
nc: <number-of-classes>
names: [...]
```

Expected labels live next to the images:

```text
datasets/vi3dr/
  images/train/
  images/val/
  labels/train/
  labels/val/
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

Runs are saved under `runs/vi3dr-yolo` by default. Choose the output location
with `--runs-dir` and the run folder name with `--name`:

```bash
./.venv/bin/python train.py --dataset-dir /Volumes/Data/vi3dr-dataset --runs-dir /Volumes/Data/vi3dr-runs --name experiment-001
```

Before training, `train.py` writes a normalized Ultralytics dataset config under
`<runs-dir>/_dataset_configs/`. This keeps datasets on external disks working
when `dataset.yaml` or `train.txt`/`val.txt` use relative paths.

## Test

Evaluate a trained checkpoint on the `test` split:

```bash
./.venv/bin/python test.py \
  --dataset-dir /Volumes/Data/vi3dr-dataset \
  --model runs/vi3dr-yolo/weights/best.pt
```

`test.py` loads and normalizes `dataset.yaml` the same way as `train.py`, then
runs Ultralytics validation with `split=test`. Results are written to
`<run-dir>/test`, where `<run-dir>` is inferred from
`<run-dir>/weights/best.pt`. If that directory already contains the expected
test plots and `labels.jpg`, the script exits without running evaluation again.
The generated `labels.jpg` describes the `test` split label distribution.

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
