"""CLI helper for extracting dataset frames from videos."""

from .vtd import ExtractionResult, extract_frames, find_next_frame_number

__all__ = ["ExtractionResult", "extract_frames", "find_next_frame_number"]
