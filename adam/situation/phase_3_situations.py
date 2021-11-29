"""A container to hold perception-processed scenes for Phase 3."""
from pathlib import Path
from typing import Tuple

from attr import attrs

from adam.situation import Situation


@attrs(slots=True, repr=False)
class SimulationSituation(Situation):
    language: Tuple[str]
    scene_images_png: Tuple[Path]
    scene_point_cloud: Tuple[Path]
