from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


FRAME_NAME_PATTERN = re.compile(r"^frame_(\d+)\.jpeg$", re.IGNORECASE)


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool:
        ...

    def read(self) -> tuple[bool, object]:
        ...

    def release(self) -> None:
        ...


@dataclass(frozen=True)
class ExtractionResult:
    video_path: Path
    dataset_dir: Path
    every: int
    first_frame_number: int
    saved_frames: int
    read_frames: int


def require_cv2():
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed. Install dependencies from requirements.txt.")
    return cv2


def find_next_frame_number(dataset_dir: Path) -> int:
    highest_number = 0

    if not dataset_dir.exists():
        return 1

    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")

    for path in dataset_dir.iterdir():
        match = FRAME_NAME_PATTERN.match(path.name)
        if match:
            highest_number = max(highest_number, int(match.group(1)))

    return highest_number + 1


def extract_frames(
    video_path: Path,
    dataset_dir: Path,
    every: int,
    *,
    capture: VideoCaptureLike | None = None,
) -> ExtractionResult:
    if every < 1:
        raise ValueError("--every must be greater than or equal to 1")

    video_path = video_path.expanduser().resolve()
    dataset_dir = dataset_dir.expanduser().resolve()

    if not video_path.is_file():
        raise FileNotFoundError(f"Video file does not exist: {video_path}")

    dataset_dir.mkdir(parents=True, exist_ok=True)
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_dir}")

    first_frame_number = find_next_frame_number(dataset_dir)
    next_frame_number = first_frame_number
    owned_capture = capture is None
    opencv = require_cv2()
    video_capture = capture if capture is not None else opencv.VideoCapture(str(video_path))

    if not video_capture.isOpened():
        if owned_capture:
            video_capture.release()
        raise RuntimeError(f"Could not open video file: {video_path}")

    read_frames = 0
    saved_frames = 0

    try:
        while True:
            ok, frame = video_capture.read()
            if not ok:
                break

            if read_frames % every == 0:
                output_path = dataset_dir / f"frame_{next_frame_number}.jpeg"
                if not opencv.imwrite(str(output_path), frame):
                    raise RuntimeError(f"Could not write frame to: {output_path}")

                saved_frames += 1
                next_frame_number += 1

            read_frames += 1
    finally:
        video_capture.release()

    return ExtractionResult(
        video_path=video_path,
        dataset_dir=dataset_dir,
        every=every,
        first_frame_number=first_frame_number,
        saved_frames=saved_frames,
        read_frames=read_frames,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 vtd.py",
        description="Wycina klatki z filmu do katalogu datasetu.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Przyklady:\n"
            "  python3 vtd.py --video film.mp4 --dataset dataset --every 10\n"
            "  python3 vtd.py --video ./input/film.mov --dataset ./frames --every 1\n"
            "\n"
            "Nazwy plikow:\n"
            "  Obrazy sa zapisywane jako frame_<numer>.jpeg.\n"
            "  Jesli w katalogu datasetu istnieja juz np. frame_1.jpeg i frame_2.jpeg,\n"
            "  nowe pliki zaczna sie od frame_3.jpeg.\n"
            "\n"
            "Klatki:\n"
            "  --every 1 zapisuje kazda klatke.\n"
            "  --every 10 zapisuje klatki 0, 10, 20 itd."
        ),
    )
    parser.add_argument(
        "--video",
        required=True,
        type=Path,
        metavar="SCIEZKA",
        help="sciezka do pliku wideo, np. film.mp4",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        metavar="KATALOG",
        help="katalog, do ktorego zostana zapisane klatki",
    )
    parser.add_argument(
        "--every",
        required=True,
        type=int,
        metavar="N",
        help="zapisz co N-ta klatke; 1 oznacza wszystkie klatki",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = extract_frames(args.video, args.dataset, args.every)
    except (FileNotFoundError, NotADirectoryError, RuntimeError, ValueError) as err:
        parser.error(str(err))

    last_frame_number = result.first_frame_number + result.saved_frames - 1
    if result.saved_frames:
        frame_range = f"frame_{result.first_frame_number}.jpeg..frame_{last_frame_number}.jpeg"
    else:
        frame_range = "no frames"

    print(
        "Saved "
        f"{result.saved_frames} frame(s) from {result.read_frames} read frame(s) "
        f"to {result.dataset_dir} ({frame_range})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
