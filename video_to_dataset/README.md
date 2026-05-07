# Video to Dataset

Extract frames from a video into a dataset directory.

## Usage

From this directory:

```bash
python3 cli.py --video path/to/video.mp4 --dataset path/to/dataset --every 10
```

This writes `frame_<number>.jpeg` files. If the dataset directory already contains files such as
`frame_1.jpeg` and `frame_2.jpeg`, new frames continue from `frame_3.jpeg`.

`--every 1` extracts all frames. `--every 10` extracts frame 0, 10, 20, and so on.
