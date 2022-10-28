"""Script to organize a set of image/object pairs as a train/test curriculums"""
import argparse
import logging
from pathlib import Path
from typing import Tuple, Generator, Dict

import cv2
import yaml

from adam.ontology.phase3_ontology import PHASE_3_CURRICULUM_OBJECTS

logger = logging.getLogger(__name__)


def img_iter(source_img_dir: Path) -> Generator[Tuple[str, Path], None, None]:
    """Function to load concept name and image as a single iterable."""
    for object_concept in PHASE_3_CURRICULUM_OBJECTS:
        concept_path = source_img_dir / object_concept.handle

        if not concept_path.is_dir():
            logger.warning(
                "Concept %s not found in given directory", object_concept.handle
            )
            continue

        for img_path in sorted(concept_path.glob("*.png")):
            yield object_concept.handle, img_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Utility script to reorganize object curricula"
    )
    parser.add_argument("input_img_dir", type=Path, help="Input set of images.")
    parser.add_argument(
        "base_curriculum_dir", type=Path, help="Base directory for curriculum storage."
    )
    parser.add_argument(
        "curriculum_name", type=str, help="The name of the curriculum to create."
    )
    parser.add_argument(
        "--train-num",
        type=int,
        default=10,
        help="The number of samples to put in the training curriculum. The rest of the samples will go the test curriculum.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Enable writing over an existing curriculum directory.",
    )
    parser.add_argument(
        "--image-size",
        nargs=2,
        metavar=("img_width", "img_height"),
        help="The values to resize the give curriculum images to. Must be 16x9 & both divisible by 8.",
        default=(384, 216),
        type=int,
    )
    args = parser.parse_args()

    source_img_dir: Path = args.input_img_dir
    train_curriculum_dir: Path = args.base_curriculum_dir / "train" / args.curriculum_name
    test_curriculum_dir: Path = (
        args.base_curriculum_dir / "test" / f"{args.curriculum_name}_eval"
    )

    img_width = args.image_size[0]
    img_height = args.image_size[1]

    # We assert this to validate that the STEGO API will not crash
    assert img_width % 16 == 0
    assert img_height % 8 == 0 and img_height % 9 == 0

    train_curriculum_dir.mkdir(exist_ok=args.overwrite, parents=True)
    test_curriculum_dir.mkdir(exist_ok=args.overwrite, parents=True)

    train_scene_count = 0
    test_scene_count = 0
    img2 = None
    object_names_to_count: Dict[str, int] = {}
    for object_name, scene_img in img_iter(source_img_dir):
        if object_names_to_count.get(object_name, 0) < args.train_num:
            sit_dir = train_curriculum_dir / f"situation_{train_scene_count}"
            train_scene_count += 1
        else:
            sit_dir = test_curriculum_dir / f"situation_{test_scene_count}"
            test_scene_count += 1
        sit_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(scene_img))
        img = cv2.resize(img, (img_width, img_height))
        try:
            img2 = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            logger.debug(f"Image {scene_img} is RGBA")
        except ValueError:
            logger.debug(f"Image {scene_img} is RGB")
            img2 = img
        finally:
            cv2.imwrite(str(sit_dir / "rgb_0.png"), img2)

        with (sit_dir / "description.yaml").open("w", encoding="utf-8") as desc_file:
            yaml.dump({"language": f"a {object_name}"}, desc_file)

        object_names_to_count[object_name] = object_names_to_count.get(object_name, 0) + 1

    with (train_curriculum_dir / "info.yaml").open("w", encoding="utf-8") as info_file:
        yaml.dump(
            {"curriculum": args.curriculum_name, "num_dirs": train_scene_count},
            info_file,
        )

    with (test_curriculum_dir / "info.yaml").open("w", encoding="utf-8") as info_file:
        yaml.dump(
            {"curriculum": f"{args.curriculum_name}_eval", "num_dirs": test_scene_count},
            info_file,
        )

    logger.info("Processing complete. Object names to scenes processed:")
    for object_name, count in object_names_to_count.items():
        if count <= args.train_num:
            logger.error(
                f"Concept {object_name} did not reach the number of training samples requested or has 0 test samples."
            )
        logger.info(f"{object_name} - {count}")


if __name__ == "__main__":
    main()
