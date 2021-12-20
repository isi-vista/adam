from pathlib import Path
from typing import Any, MutableMapping

import yaml

from adam.curriculum.curriculum_utils import Phase3InstanceGroup
from adam.curriculum.phase3_curriculum import (
    phase_3_one_objects_curriculum,
    phase_3_one_core_objects_curriculum,
    phase_3_one_stretch_objects_curriculum,
    phase_3_m4_core_eval,
    phase_3_m4_stretch_eval,
)
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_3_LANGUAGE_GENERATOR,
)
from adam.paths import (
    TRAINING_CURRICULUM_DIR,
    GENERATION_YAML_DIR_NAME,
    TESTING_CURRICULUM_DIR,
)


def curriculum_to_yaml(
    curriculum: Phase3InstanceGroup,
    yaml_dir: Path,
    *,
    num_generated_per_instance: int = 5,
):
    yaml_dir.mkdir(exist_ok=True, parents=True)
    for i, scene in enumerate(curriculum.instances()):
        situation, language, _ = scene
        if not situation:
            raise RuntimeError("Situation missing.")
        output_dict: MutableMapping[str, Any] = dict()
        output_dict["num_scenes_to_generate"] = num_generated_per_instance
        output_dict["language"] = language.as_token_string()
        output_dict["objects"] = [
            {"type": obj.ontology_node.handle} for obj in situation.all_objects
        ]
        with open(yaml_dir / f"outline_{i}.yaml", "w", encoding="utf-8") as yaml_file:
            yaml.dump(output_dict, yaml_file)


# TODO: Adapt this entry point to take a parameters file to build a set of curriculum
# So we can only rebuild the curriculum we need to rather than all curriculum every time
# See: https://github.com/isi-vista/adam/issues/1071
def main():
    # Training Curriculums
    for curriculum_builder in [
        phase_3_one_objects_curriculum,
        phase_3_one_core_objects_curriculum,
        phase_3_one_stretch_objects_curriculum,
    ]:
        curriculum = curriculum_builder(1, 5, GAILA_PHASE_3_LANGUAGE_GENERATOR)
        curriculum_to_yaml(
            curriculum,
            TRAINING_CURRICULUM_DIR / curriculum.name() / GENERATION_YAML_DIR_NAME,
            num_generated_per_instance=10,
        )

    # Testing Curriculums
    for curriculum_builder in [phase_3_m4_stretch_eval, phase_3_m4_core_eval]:
        curriculum = curriculum_builder(1, 5, GAILA_PHASE_3_LANGUAGE_GENERATOR)
        curriculum_to_yaml(
            curriculum,
            TESTING_CURRICULUM_DIR / curriculum.name() / GENERATION_YAML_DIR_NAME,
            num_generated_per_instance=1,
        )


if __name__ == "__main__":
    main()
