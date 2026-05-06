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
- shows live preview in the desktop UI,
- closes the current session after the `stop` command,
- sends `client_stop` when the server initiates connection shutdown.

## Important files

- `app.py` - desktop entrypoint
- `server_main.py` - headless entrypoint
- `modules/runtime.py` - shared backend runtime
- `modules/capture/` - image transport and capture session
- `modules/ui/` - desktop shell and HTML frontend

## Next steps

- wire YOLO results into the UI,
- add an overlay layer on top of preview,
- support editing colors of shape walls.
