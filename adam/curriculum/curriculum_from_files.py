from typing import Tuple, MutableSequence, Sequence

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
)
from adam.perception.visual_perception import (
    VisualPerceptionFrame,
    VisualPerceptionRepresentation,
)
from adam.situation.phase_3_situations import SimulationSituation

PHASE_3_TRAINING_CURRICULUM_OPTIONS = [
    "m4_core",
    "m4_stretch",
]

PHASE_3_TESTING_CURRICULUM_OPTIONS = ["m4_core_eval", "m4_stretch_eval"]

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
) -> Sequence[
    InstanceGroup[
        SimulationSituation,
        TokenSequenceLinguisticDescription,
        VisualPerceptionFrame,
    ]
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
    color_is_rgb = params.boolean("color_is_rgb", default=False)

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
            VisualPerceptionRepresentation[VisualPerceptionFrame],
        ]
    ] = []
    for situation_num in range(curriculum_params["num_dirs"]):
        situation_dir = curriculum_dir / SITUATION_DIR_NAME.format(num=situation_num)
        language_tuple: Tuple[str, ...] = tuple()
        if curriculum_type == TRAINING_CUR:
            with open(
                situation_dir / SITUATION_DESCRIPTION_FILE, encoding="utf-8"
            ) as situation_description_file:
                situation_description = yaml.safe_load(situation_description_file)
            language_tuple = tuple(situation_description["language"].split(" "))
        feature_yamls = sorted(situation_dir.glob("feature_*"))
        situation = SimulationSituation(
            language=language_tuple,
            scene_images_png=sorted(situation_dir.glob("rgb_*")),
            scene_point_cloud=tuple(situation_dir.glob("pdc_rgb_*")),
            depth_pngs=sorted(situation_dir.glob("depth_*")),
            pdc_semantic_plys=sorted(situation_dir.glob("pdc_semantic_*")),
            semantic_pngs=sorted(situation_dir.glob("semantic_*")),
            features=feature_yamls,
            strokes=sorted(situation_dir.glob("stroke_[0-9]*_[0-9]*.png")),
            stroke_graphs=sorted(situation_dir.glob("stroke_graph_*")),
            actions=sorted(situation_dir.glob("action_*")),
        )
        language = TokenSequenceLinguisticDescription(tokens=language_tuple)
        if len(feature_yamls) == 1:
            perception = VisualPerceptionRepresentation.single_frame(
                VisualPerceptionFrame.from_yaml(
                    situation_dir / feature_yamls[0],
                    color_is_rgb=color_is_rgb,
                )
            )
        else:
            # If we have more than one feature yaml there is also some information in a
            # separate file relating to action feature extraction
            with open(situation_dir / "action.yaml", encoding="utf-8") as action_yaml:
                action_features = yaml.safe_load(action_yaml)
            perception = VisualPerceptionRepresentation.multi_frame(
                frames=[
                    VisualPerceptionFrame.from_yaml(
                        situation_dir / feature_file, color_is_rgb=color_is_rgb
                    )
                    for feature_file in feature_yamls
                ],
                action_features=action_features,
            )
        instances.append((situation, language, perception))  # type: ignore

    return [
        ExplicitWithSituationInstanceGroup(  # type: ignore
            name=curriculum_to_load,
            instances=tuple(instances),
        )
    ]
