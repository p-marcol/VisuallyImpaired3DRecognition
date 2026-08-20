import unittest

import numpy as np
import webcolors

from modules.knowledge import (
    ColorMatchingConfig,
    KnowledgeBase,
    KnowledgeLoadError,
    SolidKnowledge,
    SurfaceKnowledge,
    load_knowledge_base,
)
from modules.surface_matching import SurfaceMatcher


class SurfaceMatcherTest(unittest.TestCase):
    def setUp(self):
        self.matcher = SurfaceMatcher()

    def test_cube_red_image_matches_red_surface(self):
        result = self.matcher.match(_solid_image("red"), "cube", load_knowledge_base())

        self.assertIsNotNone(result)
        self.assertEqual(result.surface.id, "red")
        self.assertEqual(result.surface.message, "ściana czerwona")
        self.assertEqual(result.matched_pixels, 100)

    def test_cube_purple_image_matches_purple_surface(self):
        result = self.matcher.match(_solid_image("purple"), "cube", load_knowledge_base())

        self.assertIsNotNone(result)
        self.assertEqual(result.surface.id, "purple")
        self.assertEqual(result.matched_pixels, 100)

    def test_cube_mixed_image_matches_dominant_yellow_surface(self):
        image = np.vstack(
            (
                _solid_image("yellow", height=7, width=10),
                _solid_image("red", height=3, width=10),
            )
        )

        result = self.matcher.match(image, "cube", load_knowledge_base())

        self.assertIsNotNone(result)
        self.assertEqual(result.surface.id, "yellow")
        self.assertEqual(result.matched_pixels, 70)

    def test_low_max_distance_rejects_far_color(self):
        knowledge = KnowledgeBase(
            solids={
                "cube": SolidKnowledge(
                    id="cube",
                    name="cube",
                    color_matching=ColorMatchingConfig(space="lab", max_distance=1.0),
                    surfaces=(
                        SurfaceKnowledge("red", "red", "red"),
                        SurfaceKnowledge("yellow", "yellow", "yellow"),
                    ),
                )
            }
        )

        result = self.matcher.match(_solid_image("black"), "cube", knowledge)

        self.assertIsNone(result)

    def test_missing_yolo_class_raises_knowledge_error(self):
        with self.assertRaisesRegex(KnowledgeLoadError, "Knowledge object not found: missing"):
            self.matcher.match(_solid_image("red"), "missing", load_knowledge_base())

    def test_matcher_uses_only_surfaces_for_yolo_class(self):
        knowledge = KnowledgeBase(
            solids={
                "cube": SolidKnowledge(
                    id="cube",
                    name="cube",
                    color_matching=ColorMatchingConfig(space="lab", max_distance=35.0),
                    surfaces=(SurfaceKnowledge("red", "red", "red"),),
                ),
                "cylinder": SolidKnowledge(
                    id="cylinder",
                    name="cylinder",
                    color_matching=ColorMatchingConfig(space="lab", max_distance=35.0),
                    surfaces=(SurfaceKnowledge("green", "green", "green"),),
                ),
            }
        )

        result = self.matcher.match(_solid_image("green"), "cube", knowledge)

        self.assertIsNone(result)


def _solid_image(color_name: str, height: int = 10, width: int = 10) -> np.ndarray:
    rgb = webcolors.name_to_rgb(color_name)
    bgr = np.array([rgb.blue, rgb.green, rgb.red], dtype=np.uint8)
    return np.tile(bgr, (height, width, 1))


if __name__ == "__main__":
    unittest.main()
