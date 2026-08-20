from dataclasses import dataclass

import numpy as np

from modules.knowledge import KnowledgeBase, KnowledgeLoadError
from modules.surface_matching import SurfaceMatcher, SurfaceMatcherError


UNRECOGNIZED_SURFACE_MESSAGE = "Nie udało się rozpoznać widocznej powierzchni."


@dataclass(frozen=True)
class SliceAnalysisResult:
    text: str


def analyze_slice(
    crop,
    class_id: str,
    knowledge: KnowledgeBase,
    surface_matcher: SurfaceMatcher | None = None,
    request=None,
) -> SliceAnalysisResult:
    if not _is_valid_crop(crop):
        return SliceAnalysisResult(UNRECOGNIZED_SURFACE_MESSAGE)

    matcher = surface_matcher or SurfaceMatcher()
    try:
        match = matcher.match(crop, class_id, knowledge)
    except (KnowledgeLoadError, SurfaceMatcherError):
        return SliceAnalysisResult(UNRECOGNIZED_SURFACE_MESSAGE)

    if match is None:
        return SliceAnalysisResult(UNRECOGNIZED_SURFACE_MESSAGE)

    return SliceAnalysisResult(match.surface.message)


class SliceAnalyzer:
    def __init__(self, surface_matcher: SurfaceMatcher | None = None):
        self.surface_matcher = surface_matcher or SurfaceMatcher()

    def analyze(
        self,
        crop,
        class_id: str,
        knowledge: KnowledgeBase,
        request=None,
    ) -> SliceAnalysisResult:
        return analyze_slice(
            crop,
            class_id,
            knowledge,
            surface_matcher=self.surface_matcher,
            request=request,
        )


def _is_valid_crop(crop) -> bool:
    return (
        isinstance(crop, np.ndarray)
        and crop.ndim == 3
        and crop.shape[2] == 3
        and crop.size > 0
        and crop.dtype == np.uint8
    )
