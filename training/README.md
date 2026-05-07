# VI3DR Training

This package is responsible for training and testing YOLO models. The server
package consumes trained artifacts later; do not put server runtime changes here.

## Dataset layout

`dataset.yaml` follows the Ultralytics YOLO format:

```yaml
path: ./datasets/vi3dr
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

```bash
./.venv/bin/python train.py --dry-run
./.venv/bin/python train.py
```

The default model source is configured in `config.py` as `yolov8n.pt`.

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
