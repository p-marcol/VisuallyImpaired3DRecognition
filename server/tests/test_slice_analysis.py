import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import webcolors

from modules.analysis import UNRECOGNIZED_SURFACE_MESSAGE, SliceAnalyzer
from modules.detection import DetectionBox, DetectionResult
from modules.knowledge import (
    ColorMatchingConfig,
    KnowledgeBase,
    SolidKnowledge,
    SurfaceKnowledge,
    load_knowledge_base,
)
from modules.runtime import ApplicationRuntime


class SliceAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.analyzer = SliceAnalyzer()

    def test_cube_purple_crop_returns_purple_surface_message(self):
        result = self.analyzer.analyze(
            _solid_image("purple"),
            "cube",
            load_knowledge_base(),
        )

        self.assertEqual(result.text, "ściana fioletowa")

    def test_cube_red_crop_returns_red_surface_message(self):
        result = self.analyzer.analyze(
            _solid_image("red"),
            "cube",
            load_knowledge_base(),
        )

        self.assertEqual(result.text, "ściana czerwona")

    def test_unmatched_surface_returns_internal_message(self):
        knowledge = KnowledgeBase(
            solids={
                "cube": SolidKnowledge(
                    id="cube",
                    name="cube",
                    color_matching=ColorMatchingConfig(space="lab", max_distance=1.0),
                    surfaces=(SurfaceKnowledge("red", "red", "red"),),
                )
            }
        )

        result = self.analyzer.analyze(_solid_image("black"), "cube", knowledge)

        self.assertEqual(result.text, UNRECOGNIZED_SURFACE_MESSAGE)

    def test_invalid_crop_returns_internal_message(self):
        result = self.analyzer.analyze(None, "cube", load_knowledge_base())

        self.assertEqual(result.text, UNRECOGNIZED_SURFACE_MESSAGE)

    def test_analysis_uses_detection_snapshot_from_info_request(self):
        runtime = ApplicationRuntime(preview_enabled=False)
        runtime.knowledge = KnowledgeBase(
            solids={
                "cube": SolidKnowledge(
                    id="cube",
                    name="cube",
                    color_matching=ColorMatchingConfig(space="lab", max_distance=35.0),
                    surfaces=(
                        SurfaceKnowledge("purple", "purple", "snapshot cube purple"),
                    ),
                ),
                "cylinder": SolidKnowledge(
                    id="cylinder",
                    name="cylinder",
                    color_matching=ColorMatchingConfig(space="lab", max_distance=35.0),
                    surfaces=(
                        SurfaceKnowledge("red", "red", "current cylinder red"),
                    ),
                ),
            }
        )
        runtime.detector._store_processed_detection(
            _solid_image("purple"),
            DetectionResult("cube", 0.90, DetectionBox(0, 0, 10, 10)),
        )
        context = runtime.create_analysis_context("all")

        runtime.detector._store_processed_detection(
            _solid_image("red"),
            DetectionResult("cylinder", 0.95, DetectionBox(0, 0, 10, 10)),
        )

        self.assertEqual(runtime.provide_analysis_response(context), "snapshot cube purple")

    def test_runtime_reload_knowledge_base_affects_next_analysis(self):
        runtime = ApplicationRuntime(preview_enabled=False)
        knowledge_path = _write_knowledge_json("new purple message")
        try:
            runtime.reload_knowledge_base(knowledge_path)
            runtime.detector._store_processed_detection(
                _solid_image("purple"),
                DetectionResult("cube", 0.90, DetectionBox(0, 0, 10, 10)),
            )
            context = runtime.create_analysis_context("all")

            self.assertEqual(runtime.provide_analysis_response(context), "new purple message")
        finally:
            knowledge_path.unlink(missing_ok=True)


def _solid_image(color_name: str, height: int = 10, width: int = 10) -> np.ndarray:
    rgb = webcolors.name_to_rgb(color_name)
    bgr = np.array([rgb.blue, rgb.green, rgb.red], dtype=np.uint8)
    return np.tile(bgr, (height, width, 1))


def _write_knowledge_json(message: str) -> Path:
    data = [
        {
            "id": "cube",
            "name": "cube",
            "color_matching": {
                "space": "lab",
                "max_distance": 35.0,
            },
            "surfaces": [
                {
                    "id": "purple",
                    "color": "purple",
                    "message": message,
                }
            ],
        }
    ]
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False)
    try:
        json.dump(data, handle, ensure_ascii=False)
        return Path(handle.name)
    finally:
        handle.close()


if __name__ == "__main__":
    unittest.main()
