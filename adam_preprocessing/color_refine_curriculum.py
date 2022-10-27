"""
Perform color segmentation and refinement on an entire input curriculum.
"""
from argparse import ArgumentParser
import json
import logging
import os.path
from pathlib import Path
import re
import shutil

import cv2
from tqdm import tqdm
import yaml

from color_refinement import refined_segmentations, refine_segmentation_simple

logger = logging.getLogger(__name__)
file_dir = os.path.dirname(__file__)


IMAGE_NUMBER_REGEX = re.compile(r"[a-zA-Z_]+_(\d+).png")


def parse_image_number(filename: str) -> int:
    """
    Parse image number from a filename.

    The filename (sans extension) should consist of ASCII letters and underscores and contain just
    one integer. The integer must immediately precede the .png.
    """
    return int(re.sub(IMAGE_NUMBER_REGEX, r"\1", filename))


def refine_segmentation_file(
    *,
    semantic: Path,
    color_segmentation: Path,
    save_to_dir: Path,
    name_format: str,
) -> None:
    """
    Given the path to an object segmentation and a color segmentation image, refine to the given
    dir.

    Parameters:
        semantic: Path to an object segmentation image.
        color_segmentation: Path to the corresponding color segmentation image.
        save_to_dir:
            Directory where the color-refined segmentation files should be saved as output.
        name_format:
            Format string to use for the output file names. Use {} or {0} for the image-specific
            object ID number. Note that these do NOT necessarily correspond to mask numbers or to
            the indices/IDs used in `feature.yaml`!
    """
    semantic_data = cv2.imread(str(semantic))
    color_segmentation_data = cv2.imread(str(color_segmentation))
    color_refined_segmentations = refined_segmentations(
        color_segmentation_data, semantic_data
    )

    for mask_id, color_refined_segmentation in enumerate(color_refined_segmentations):
        cv2.imwrite(
            str(save_to_dir / name_format.format(mask_id)),
            color_refined_segmentation,
        )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "curriculum_path",
        type=Path,
        help="The curriculum path to process.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="The output curriculum dir.",
    )
    parser.add_argument(
        "--dir-num",
        default=None,
        help="A specific situation directory number to process. If provided only this directory is processed."
    )
    parser.add_argument(
        "--color-seed",
        type=int,
        default=42,
        help="The seed to use when randomizing the color segmentation region colors."
    )
    parser.add_argument(
        "--multifile-output",
        action="store_true",
        help="When enabled, save K files per segmentation (K being the number of objects)."
    )
    parser.add_argument(
        "--no-multifile-output",
        action="store_false",
        dest="multifile_output",
        help="When passed, save 1 output file per segmentation input."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # copied and edited from phase3_load_from_disk() -- see adam.curriculum.curriculum_from_files
    if args.dir_num:
        logger.info(f"Processing directory number {args.dir_num}")
    else:
        with open(
            args.curriculum_path / "info.yaml", encoding="utf=8"
        ) as curriculum_info_yaml:
            curriculum_params = yaml.safe_load(curriculum_info_yaml)
        logger.info("Input curriculum has %d dirs/situations.", curriculum_params["num_dirs"])

    for situation_num in tqdm(
        range(curriculum_params["num_dirs"]) if args.dir_num is None else [args.dir_num],
        desc="Situations processed",
        total=curriculum_params["num_dirs"] if args.dir_num is None else 1,
    ):
        situation_dir = args.curriculum_path / f"situation_{situation_num}"
        output_situation_dir = args.output_dir / f"situation_{situation_num}"
        output_situation_dir.mkdir(exist_ok=True, parents=True)

        # Refine segmentation files assuming one segmentation file per RGB image
        # This is how we do things right now (as of the August demo)
        # On the contrary, we save as output one refined segmentation mask per object
        color_segmentation_metadata = json.loads(
            output_situation_dir.joinpath("color_segmentation_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        failed_image_numbers = color_segmentation_metadata["failed_image_numbers"]
        rgb_images = sorted(situation_dir.glob("rgb_*.png"))
        if not rgb_images:
            logger.warning(
                "No RGB images for situation number %d; not doing anything...", situation_num
            )
        for rgb_image in rgb_images:
            number = parse_image_number(rgb_image.name)
            semantic_image = situation_dir / f"semantic_{number}.png"
            color_segmentation_image = output_situation_dir / f"color_segmentation_{number}.png"
            if number in failed_image_numbers:
                logger.warning(
                    "Skipping refinement for image number %d (%s) in situation %d because color "
                    "segmentation failed.",
                    number,
                    semantic_image,
                    situation_num,
                )
                if args.multifile_output:
                    raise ValueError("Don't know what to do here.")
                else:
                    shutil.copy(
                        semantic_image,
                        output_situation_dir / f"combined_color_refined_semantic_{number}.png",
                    )
                continue
            elif not semantic_image.exists():
                logger.warning(
                    "Skipping refinement for image number %d (%s) in situation %d because instance "
                    "segmentation file (%s) doesn't exist.",
                    number,
                    rgb_image,
                    semantic_image,
                    situation_num,
                )
                continue
            if args.multifile_output:
                refine_segmentation_file(
                    semantic=semantic_image,
                    color_segmentation=output_situation_dir / f"color_segmentation_{number}.png",
                    save_to_dir=output_situation_dir,
                    name_format=f"color_refined_semantic_{number}_{{}}.png",
                )
            else:
                assert semantic_image.exists()
                assert color_segmentation_image.exists()
                semantic_data = cv2.imread(str(semantic_image))
                color_segmentation_data = cv2.imread(str(color_segmentation_image))
                color_refined_segmentation = refine_segmentation_simple(
                    color_segmentation_data, semantic_data
                )
                cv2.imwrite(
                    str(output_situation_dir / f"combined_color_refined_semantic_{number}.png"),
                    color_refined_segmentation,
                )


if __name__ == "__main__":
    main()
