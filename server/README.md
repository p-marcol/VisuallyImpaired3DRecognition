# VI3DR Server

The server receives camera frames from a phone over WebSocket, publishes the service over mDNS, and runs a desktop UI in `PySide6` with an embedded HTML frontend.

## Requirements

- Python 3.11
- local `venv` environment

## Install dependencies

```bash
./venv/bin/pip install -r requirements.txt
```

If `venv` does not exist yet:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

## Run the desktop UI

This is the main application mode. It starts both the backend runtime and the `PySide6` desktop shell.

```bash
./venv/bin/python app.py
```

## Run headless

This mode starts only the backend without the desktop UI. In this variant, preview remains on the `OpenCV` side.

```bash
./venv/bin/python server_main.py
```

## Current behavior

- listens for phone connections over WebSocket,
- accepts JPEG frames,
- loads a YOLO detector through Ultralytics/PyTorch,
- draws YOLO detection boxes into the preview frame,
- shows live preview in the desktop UI,
- closes the current session after the `stop` command,
- sends `client_stop` when the server initiates connection shutdown.

## Detection configuration

Detection is enabled by default. The server uses Ultralytics YOLO and can load either a named pretrained model or a local `.pt` file.

In desktop mode, use the Detection panel and the `Choose model` button to load a `.pt` model without restarting the app.

```bash
VI3DR_YOLO_MODEL=path/to/model.pt ./.venv/bin/python app.py
```

Useful environment variables:

- `VI3DR_DETECTION_ENABLED=0` - disable YOLO processing
- `VI3DR_YOLO_MODEL=yolo11n.pt` - model name or local model path
- `VI3DR_YOLO_CONFIDENCE=0.25` - confidence threshold
- `VI3DR_YOLO_IMAGE_SIZE=640` - inference image size
- `VI3DR_YOLO_DEVICE=cpu` - optional device override, for example `cpu`, `cuda:0`, `mps`

## Important files

- `app.py` - desktop entrypoint
- `server_main.py` - headless entrypoint
- `modules/runtime.py` - shared backend runtime
- `modules/capture/` - image transport and capture session
- `modules/ui/` - desktop shell and HTML frontend

## Next steps

- add an overlay layer on top of preview,
- support editing colors of shape walls.
