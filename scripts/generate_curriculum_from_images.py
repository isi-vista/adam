import argparse
from pathlib import Path

from PIL.Image import Resampling
from yaml import load, dump

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
    parser.add_argument("images_dir", type=Path)
    parser.add_argument(
        "--output_dir", type=Path, default="data/multi_obj")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True)

    situations_count = 0
    images_dir: Path = args.images_dir
    for image_file in images_dir.glob('*.jpg'):
        input_image = Image.open(image_file)
        description = image_file.name[:-len('.jpg')]

        # Make situation directory
        situation_dir = args.output_dir / f"situation_{situations_count}"
        situation_dir.mkdir()
        situations_count += 1

        # Save description
        with open(situation_dir / "description.yaml", "w") as description_file:
            dump({'language': description}, description_file, Dumper=Dumper)

        # Output RGB image of situation
        resized_image = input_image.resize(IMAGE_SIZE, Resampling.LANCZOS)
        resized_image.save(situation_dir / "rgb_0.png")

    # Output info file
    with open(args.output_dir / "info.yaml", "w") as info_file:
        dump(
            {'curriculum': str(args.output_dir), 'num_dirs': situations_count},
            info_file
        )


if __name__ == "__main__":
    main()
