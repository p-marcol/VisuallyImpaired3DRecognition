import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


DEFAULT_KNOWLEDGE_PATH = Path(__file__).resolve().parents[2] / "knowledge_db" / "knowledge.json"


class KnowledgeLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class ColorMatchingConfig:
    space: str
    max_distance: float


@dataclass(frozen=True)
class SurfaceKnowledge:
    id: str
    color: str
    message: str


@dataclass(frozen=True)
class SolidKnowledge:
    id: str
    name: str
    color_matching: ColorMatchingConfig
    surfaces: tuple[SurfaceKnowledge, ...]


@dataclass(frozen=True)
class KnowledgeBase:
    solids: dict[str, SolidKnowledge]

    def get(self, solid_id: str) -> SolidKnowledge:
        try:
            return self.solids[solid_id]
        except KeyError as err:
            raise KnowledgeLoadError(f"Knowledge object not found: {solid_id}") from err


def load_knowledge_base(path: str | Path = DEFAULT_KNOWLEDGE_PATH) -> KnowledgeBase:
    return _load_knowledge_base_once(str(Path(path)))


def reload_knowledge_base(path: str | Path = DEFAULT_KNOWLEDGE_PATH) -> KnowledgeBase:
    _load_knowledge_base_once.cache_clear()
    return load_knowledge_base(path)


@lru_cache(maxsize=1)
def _load_knowledge_base_once(path: str) -> KnowledgeBase:
    knowledge_path = Path(path)
    try:
        with knowledge_path.open("r", encoding="utf-8") as file:
            raw_data = json.load(file)
    except FileNotFoundError as err:
        raise KnowledgeLoadError(f"Knowledge database file not found: {knowledge_path}") from err
    except json.JSONDecodeError as err:
        raise KnowledgeLoadError(
            f"Knowledge database JSON is invalid at line {err.lineno}, column {err.colno}: {err.msg}"
        ) from err
    except OSError as err:
        raise KnowledgeLoadError(f"Cannot read knowledge database: {knowledge_path}: {err}") from err

    try:
        return _parse_knowledge_base(raw_data)
    except (TypeError, ValueError) as err:
        raise KnowledgeLoadError(f"Knowledge database has invalid structure: {err}") from err


def _parse_knowledge_base(raw_data) -> KnowledgeBase:
    if not isinstance(raw_data, list):
        raise TypeError("root value must be a list of objects")

    solids = {}
    for index, raw_solid in enumerate(raw_data):
        solid = _parse_solid(raw_solid, f"[{index}]")
        if solid.id in solids:
            raise ValueError(f"duplicate solid id: {solid.id}")
        solids[solid.id] = solid

    return KnowledgeBase(solids=solids)


def _parse_solid(raw_solid, location: str) -> SolidKnowledge:
    if not isinstance(raw_solid, dict):
        raise TypeError(f"{location} must be an object")

    solid_id = _require_string(raw_solid, "id", location)
    name = _require_string(raw_solid, "name", location)
    color_matching = _parse_color_matching(
        _require_dict(raw_solid, "color_matching", location),
        f"{location}.color_matching",
    )
    surfaces = tuple(
        _parse_surface(raw_surface, f"{location}.surfaces[{index}]")
        for index, raw_surface in enumerate(_require_list(raw_solid, "surfaces", location))
    )

    return SolidKnowledge(
        id=solid_id,
        name=name,
        color_matching=color_matching,
        surfaces=surfaces,
    )


def _parse_color_matching(raw_config, location: str) -> ColorMatchingConfig:
    return ColorMatchingConfig(
        space=_require_string(raw_config, "space", location),
        max_distance=_require_number(raw_config, "max_distance", location),
    )


def _parse_surface(raw_surface, location: str) -> SurfaceKnowledge:
    if not isinstance(raw_surface, dict):
        raise TypeError(f"{location} must be an object")

    return SurfaceKnowledge(
        id=_require_string(raw_surface, "id", location),
        color=_require_string(raw_surface, "color", location),
        message=_require_string(raw_surface, "message", location),
    )


def _require_dict(data: dict, key: str, location: str) -> dict:
    value = _require_key(data, key, location)
    if not isinstance(value, dict):
        raise TypeError(f"{location}.{key} must be an object")
    return value


def _require_list(data: dict, key: str, location: str) -> list:
    value = _require_key(data, key, location)
    if not isinstance(value, list):
        raise TypeError(f"{location}.{key} must be a list")
    return value


def _require_string(data: dict, key: str, location: str) -> str:
    value = _require_key(data, key, location)
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{location}.{key} must be a non-empty string")
    return value


def _require_number(data: dict, key: str, location: str) -> float:
    value = _require_key(data, key, location)
    if not isinstance(value, (int, float)):
        raise TypeError(f"{location}.{key} must be a number")
    return float(value)


def _require_key(data: dict, key: str, location: str):
    if key not in data:
        raise ValueError(f"{location}.{key} is required")
    return data[key]
