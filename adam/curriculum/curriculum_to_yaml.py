from pathlib import Path
from typing import Any, MutableMapping

import yaml

from adam.curriculum.curriculum_utils import Phase3InstanceGroup
from adam.curriculum.phase3_curriculum import Phase3OneObjectsCurriculum
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_3_LANGUAGE_GENERATOR,
)
from adam.paths import TRAINING_CURRICULUM_DIR, GENERATION_YAML_DIR_NAME


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


def main():
    phase_3_one_object = Phase3OneObjectsCurriculum()
    curriculum = phase_3_one_object(1, 5, GAILA_PHASE_3_LANGUAGE_GENERATOR)
    curriculum_to_yaml(
        curriculum, TRAINING_CURRICULUM_DIR / curriculum.name() / GENERATION_YAML_DIR_NAME
    )


if __name__ == "__main__":
    main()
