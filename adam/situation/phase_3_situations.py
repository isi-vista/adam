"""A container to hold perception-processed scenes for Phase 3."""
from itertools import chain
from pathlib import Path
from typing import Tuple, Sequence, Iterable

from attr import attrs, attrib
from attr.validators import deep_iterable, instance_of
from immutablecollections.converter_utils import _to_tuple

from adam.situation import Situation


@attrs(slots=True, repr=False)
class SimulationSituation(Situation):
    language: Tuple[str] = attrib(converter=_to_tuple)
    features: Sequence[Path] = attrib(validator=deep_iterable(instance_of(Path)))
    scene_images_png: Sequence[Path] = attrib(validator=deep_iterable(instance_of(Path)))
    scene_point_cloud: Sequence[Path] = attrib(validator=deep_iterable(instance_of(Path)))
    depth_pngs: Sequence[Path] = attrib(validator=deep_iterable(instance_of(Path)))
    semantic_pngs: Sequence[Path] = attrib(validator=deep_iterable(instance_of(Path)))
    pdc_semantic_plys: Sequence[Path] = attrib(validator=deep_iterable(instance_of(Path)))
    strokes: Sequence[Path] = attrib(validator=deep_iterable(instance_of(Path)))
    stroke_graphs: Sequence[Path] = attrib(validator=deep_iterable(instance_of(Path)))

    def all_files(self) -> Iterable[Path]:
        for file_path in chain(
            self.features,
            self.scene_images_png,
            self.scene_point_cloud,
            self.depth_pngs,
            self.semantic_pngs,
            self.pdc_semantic_plys,
            self.strokes,
            self.stroke_graphs,
        ):
            yield file_path
