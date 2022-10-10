"""
Perform color segmentation and refinement on an entire input curriculum.
"""
from argparse import ArgumentParser
import json
import logging
import os.path
from pathlib import Path
import re

from tqdm import tqdm
import yaml

from color_segmentation import segment_rgb_file

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

        # Segment RGB files assuming one segmentation file per RGB image
        failed_image_numbers = []
        for rgb_image in sorted(situation_dir.glob("rgb_*.png")):
            number = parse_image_number(rgb_image.name)
            try:
                segment_rgb_file(
                    rgb=rgb_image,
                    save_with_proper_colors_to=output_situation_dir / f"original_colors_color_segmentation_{number}.png",
                    save_with_random_colors_to=output_situation_dir / f"color_segmentation_{number}.png",
                    color_seed=args.color_seed,
                )
            # jac: For some images we may not be able to refine the segmentation due to errors in
            # the Matlab code. For now, ignore such errors and move on.
            except ValueError:
                logger.debug(
                    "Couldn't segment image %s in situation %d, continuing...",
                    rgb_image.name,
                    situation_num,
                )
                failed_image_numbers.append(number)
                continue

        output_situation_dir.joinpath("color_segmentation_metadata.json").write_text(
            json.dumps({"failed_image_numbers": failed_image_numbers}) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
