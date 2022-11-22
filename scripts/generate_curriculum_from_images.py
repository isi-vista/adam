import argparse
import random
from pathlib import Path

from PIL.Image import Resampling
import yaml

IMAGE_SIZE = (640, 360)
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(
        "Create new curriculum from directory full of objects. Take language "
        "descriptions from filenames. "
    )
    parser.add_argument(
        "images_dir", type=Path, help="directory from which to draw images")
    parser.add_argument(
        "--base-dir", type=Path, default="data/curriculum",
        help="directory into which to place generated train & test curricula")
    parser.add_argument(
        "--curriculum-name", type=str, default="multi_obj")
    parser.add_argument("--seed", type=int, default=123, help="random seed")
    parser.add_argument(
        "--train-split", type=float, default=0.5,
        help="split that goes to train")
    args = parser.parse_args()

    # Split into train & test
    random.seed(args.seed)
    image_files = sorted(args.images_dir.glob('*.jpg'))
    random.shuffle(image_files)
    num_train = int(args.train_split * len(image_files))
    splits = {'train': image_files[:num_train], 'test': image_files[num_train:]}

    # Save situations to curricula
    for split, images in splits.items():

        split_name = args.curriculum_name + '_eval' if split == 'test' \
            else args.curriculum_name

        # Create dir for splits
        situations_count = 0
        split_path: Path = args.base_dir / split / split_name
        split_path.mkdir(exist_ok=True)
        for image_file in images:
            input_image = Image.open(image_file)
            description = image_file.stem
            save_situation_to_dir(split_path, description, input_image,
                                  situations_count)
            situations_count += 1

        # Output info file
        with open(split_path / "info.yaml", "w") as info_file:
            yaml.dump(
                {'curriculum': split_name, 'num_dirs': situations_count},
                info_file
            )


def save_situation_to_dir(curriculum_dir: Path, description: str, input_image: Image.Image, situation_num: int):
    """Save an image and its description in ADAM curriculum format.
    """
    # Make situation directory
    situation_dir = curriculum_dir / f"situation_{situation_num}"
    situation_dir.mkdir()

    # Save description
    with open(situation_dir / "description.yaml", "w") as description_file:
        yaml.dump({'language': description}, description_file)

    # Output RGB image of situation
    resized_image = input_image.resize(IMAGE_SIZE, Resampling.LANCZOS)
    resized_image.save(situation_dir / "rgb_0.png")
    return situation_num


if __name__ == "__main__":
    main()
