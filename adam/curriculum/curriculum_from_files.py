from typing import Tuple, MutableSequence

import yaml
from vistautils.parameters import Parameters

from adam.curriculum import InstanceGroup, ExplicitWithSituationInstanceGroup
from adam.language import TokenSequenceLinguisticDescription
from adam.language.language_generator import (
    LanguageGenerator,
    InSituationLanguageGenerator,
)
from adam.paths import (
    TRAINING_CURRICULUM_DIR,
    TESTING_CURRICULUM_DIR,
    CURRICULUM_INFO_FILE,
    SITUATION_DIR_NAME,
    SITUATION_DESCRIPTION_FILE,
    SCENE_JSON,
)
from adam.perception import PerceptualRepresentationFrame
from adam.perception.visual_perception import VisualPerceptionFrame
from adam.situation.phase_3_situations import SimulationSituation

PHASE_3_TRAINING_CURRICULUM_OPTIONS = [
    "phase3-m4-core",
    "phase3-m4-stretch",
]

PHASE_3_TESTING_CURRICULUM_OPTIONS = ["phase3-m4-eval"]

TRAINING_CUR = "training"
TESTING_CUR = "testing"


def phase3_load_from_disk(  # pylint: disable=unused-argument
    num_samples: int,
    num_noise_objects: int,
    language_generator: LanguageGenerator[
        SimulationSituation, TokenSequenceLinguisticDescription
    ] = InSituationLanguageGenerator,  # type: ignore
    *,
    params: Parameters = Parameters.empty(),
) -> InstanceGroup[
    SimulationSituation,
    TokenSequenceLinguisticDescription,
    PerceptualRepresentationFrame,
]:
    curriculum_type = params.string(
        "curriculum_type", valid_options=[TRAINING_CUR, TESTING_CUR], default=TRAINING_CUR
    )
    curriculum_to_load = params.string(
        "curriculum",
        valid_options=PHASE_3_TRAINING_CURRICULUM_OPTIONS
        if curriculum_type == TRAINING_CUR
        else PHASE_3_TESTING_CURRICULUM_OPTIONS,
    )

    if curriculum_type == TRAINING_CUR:
        root_dir = TRAINING_CURRICULUM_DIR
    else:
        root_dir = TESTING_CURRICULUM_DIR

    curriculum_dir = root_dir / curriculum_to_load

    if not curriculum_dir.exists():
        raise RuntimeError(
            f"Curriculum to load does not exist! Tried to load {curriculum_dir}"
        )

    with open(
        curriculum_dir / CURRICULUM_INFO_FILE, encoding="utf=8"
    ) as curriculum_info_yaml:
        curriculum_params = yaml.safe_load(curriculum_info_yaml)

    instances: MutableSequence[
        Tuple[
            SimulationSituation,
            TokenSequenceLinguisticDescription,
            PerceptualRepresentationFrame,
        ]
    ] = []
    for situation_num in range(curriculum_params["num_dirs"]):
        situation_dir = curriculum_dir / SITUATION_DIR_NAME.format(num=situation_num)
        with open(
            situation_dir / SITUATION_DESCRIPTION_FILE, encoding="utf-8"
        ) as situation_description_file:
            situation_description = yaml.safe_load(situation_description_file)
        language_tuple = tuple(situation_description["language"].split(" "))
        situation = SimulationSituation(
            language=language_tuple,
            scene_images_png=tuple(),
            scene_point_cloud=tuple(),
        )
        language = TokenSequenceLinguisticDescription(tokens=language_tuple)
        perception = VisualPerceptionFrame.from_json(situation_dir / SCENE_JSON)
        instances.append((situation, language, perception))

    return ExplicitWithSituationInstanceGroup(
        name=curriculum_to_load,
        instances=tuple(instances),
    )
