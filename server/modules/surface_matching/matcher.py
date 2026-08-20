from dataclasses import dataclass

import cv2
import numpy as np
import webcolors

from modules.knowledge import KnowledgeBase, SolidKnowledge, SurfaceKnowledge


class SurfaceMatcherError(RuntimeError):
    pass


@dataclass(frozen=True)
class SurfaceMatchResult:
    surface: SurfaceKnowledge
    matched_pixels: int


class SurfaceMatcher:
    def match(
        self,
        image_crop: np.ndarray,
        class_id: str,
        knowledge: KnowledgeBase,
    ) -> SurfaceMatchResult | None:
        solid = knowledge.get(class_id)
        self._validate_color_space(solid)
        self._validate_image_crop(image_crop)

        surfaces = solid.surfaces
        if not surfaces:
            return None

        references_lab = self._build_reference_colors_lab(surfaces)
        pixels_lab = cv2.cvtColor(image_crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

        diff = pixels_lab[:, None, :] - references_lab[None, :, :]
        distance_squared = np.sum(diff * diff, axis=2)
        nearest_surface_indexes = np.argmin(distance_squared, axis=1)
        nearest_distances_squared = np.min(distance_squared, axis=1)

        accepted_pixels = nearest_distances_squared <= solid.color_matching.max_distance**2
        if not np.any(accepted_pixels):
            return None

        counts = np.bincount(
            nearest_surface_indexes[accepted_pixels],
            minlength=len(surfaces),
        )
        best_surface_index = int(np.argmax(counts))
        matched_pixels = int(counts[best_surface_index])
        if matched_pixels == 0:
            return None

        return SurfaceMatchResult(
            surface=surfaces[best_surface_index],
            matched_pixels=matched_pixels,
        )

    @staticmethod
    def _validate_color_space(solid: SolidKnowledge):
        if solid.color_matching.space.strip().lower() != "lab":
            raise SurfaceMatcherError(
                f"Unsupported color matching space for {solid.id}: {solid.color_matching.space}"
            )

    @staticmethod
    def _validate_image_crop(image_crop: np.ndarray):
        if not isinstance(image_crop, np.ndarray):
            raise SurfaceMatcherError("image_crop must be a numpy.ndarray")
        if image_crop.ndim != 3 or image_crop.shape[2] != 3:
            raise SurfaceMatcherError("image_crop must be a BGR image with shape H x W x 3")
        if image_crop.size == 0:
            raise SurfaceMatcherError("image_crop must not be empty")
        if image_crop.dtype != np.uint8:
            raise SurfaceMatcherError("image_crop must use uint8 BGR values")

    @staticmethod
    def _build_reference_colors_lab(surfaces: tuple[SurfaceKnowledge, ...]) -> np.ndarray:
        bgr_colors = np.array(
            [SurfaceMatcher._css_color_to_bgr(surface.color) for surface in surfaces],
            dtype=np.uint8,
        ).reshape(-1, 1, 3)
        return cv2.cvtColor(bgr_colors, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)

    @staticmethod
    def _css_color_to_bgr(color_name: str) -> tuple[int, int, int]:
        try:
            rgb = webcolors.name_to_rgb(color_name)
        except ValueError as err:
            raise SurfaceMatcherError(f"Unsupported CSS color name: {color_name}") from err
        return rgb.blue, rgb.green, rgb.red
