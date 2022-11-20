"""A script to filter a given curriculum along a given condition.

Default filtering is for Object Training Curriculums where this filter ensures
only a single object is present in the YAML data."""
import argparse
import logging
from pathlib import Path
from typing import Mapping, Any, Sequence, MutableMapping

import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


def clean_object_yaml(object_yaml: MutableMapping[str, Any]) -> Mapping[str, Any]:
    """Cleans an object yaml of relationship fields."""
    object_yaml["distance"] = {}
    object_yaml["relative_distance"] = {}
    object_yaml["relative_size"] = {}

    return object_yaml


def filter_by_largest_box_area(
    objects_: Sequence[MutableMapping[str, Any]]
) -> Mapping[str, Any]:
    """Filter a set of objects by the largest box area constraint."""

    return clean_object_yaml(max(objects_, key=lambda x: x["size"]["box_area"]))


def main() -> None:
    """Script to filter curriculum to single objects for training."""
    parser = argparse.ArgumentParser(
        description="Utility script to reorganize Action curricula"
    )
    parser.add_argument(
        "curriculum_path", type=Path, help="Curriculum to apply filtering to."
    )
    args = parser.parse_args()

    curriculum_dir: Path = args.curriculum_path

    with open(curriculum_dir / "info.yaml", encoding="utf=8") as curriculum_info_yaml:
        curriculum_params = yaml.safe_load(curriculum_info_yaml)
    logger.info("Input curriculum has %d dirs/situations.", curriculum_params["num_dirs"])

    for situation_num in tqdm(
        range(curriculum_params["num_dirs"]),
        desc="Situations processed",
        total=curriculum_params["num_dirs"],
    ):
        feature_path = curriculum_dir / f"situation_{situation_num}" / "feature.yaml"
        with feature_path.open(encoding="utf-8") as feature_yaml_file:
            features = yaml.safe_load(feature_yaml_file)

        features["objects"] = (
            [filter_by_largest_box_area(features["objects"])]
            if len(features["objects"]) > 0
            else []
        )
        features["touching"] = []

        with feature_path.open("w", encoding="utf-8") as feature_out_file:
            yaml.safe_dump(features, feature_out_file)


if __name__ == "__main__":
    main()
