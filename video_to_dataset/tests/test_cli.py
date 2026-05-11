import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from video_to_dataset.vtd import extract_frames, find_next_frame_number


class FakeCapture:
    def __init__(self, frames, opened=True):
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class FrameExtractionTests(unittest.TestCase):
    def test_next_frame_number_continues_after_existing_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            (dataset_dir / "frame_1.jpeg").write_text("", encoding="utf-8")
            (dataset_dir / "frame_7.jpeg").write_text("", encoding="utf-8")
            (dataset_dir / "frame_notes.jpeg").write_text("", encoding="utf-8")
            (dataset_dir / "other_99.jpeg").write_text("", encoding="utf-8")

            self.assertEqual(find_next_frame_number(dataset_dir), 8)

    def test_extracts_first_and_every_nth_frame_with_continued_numbering(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "source.mp4"
            dataset_dir = tmp_path / "dataset"
            video_path.write_bytes(b"fake video")
            dataset_dir.mkdir()
            (dataset_dir / "frame_3.jpeg").write_text("", encoding="utf-8")
            capture = FakeCapture(["a", "b", "c", "d", "e"])
            written = []

            def fake_imwrite(path, frame):
                written.append((Path(path).name, frame))
                return True

            fake_cv2 = type("FakeCV2", (), {"imwrite": staticmethod(fake_imwrite)})

            with patch("video_to_dataset.vtd.require_cv2", return_value=fake_cv2):
                result = extract_frames(video_path, dataset_dir, 2, capture=capture)

            self.assertTrue(capture.released)
            self.assertEqual(result.read_frames, 5)
            self.assertEqual(result.saved_frames, 3)
            self.assertEqual(result.first_frame_number, 4)
            self.assertEqual(
                written,
                [
                    ("frame_4.jpeg", "a"),
                    ("frame_5.jpeg", "c"),
                    ("frame_6.jpeg", "e"),
                ],
            )

    def test_rejects_invalid_every_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "source.mp4"
            video_path.write_bytes(b"fake video")

            with self.assertRaises(ValueError):
                extract_frames(video_path, Path(tmp) / "dataset", 0, capture=FakeCapture([]))


if __name__ == "__main__":
    unittest.main()
