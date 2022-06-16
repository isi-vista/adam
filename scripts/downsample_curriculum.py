import argparse
import logging
from pathlib import Path
from shutil import copytree, rmtree
from typing import Mapping, Sequence
from random import sample, seed

import yaml

situation_num: int = 0


def get_situation_num(situation_dir: Path) -> int:
    """Given a Path for a situation directory, return the associated number"""
    return int(situation_dir.name.split('_')[-1])


def copy_directory(start_path: Path, end_path: Path) -> None:
    """Function to recursively copy directory from start_path to end_path"""
    copytree(str(start_path), str(end_path))


def copy_all_directories(output_dir: Path, dirs_to_copy: Sequence[Path]) -> None:
    """Move all directories in dirs_to_move to their corresponding location in the target directory"""
    global situation_num
    sorted_dirs = sorted(dirs_to_copy, key=get_situation_num)
    for directory in sorted_dirs:
        copy_directory(directory, output_dir / f'situation_{situation_num}')
        situation_num += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Script to downsample preprocessed curricula"
    )
    parser.add_argument("--input-dir", type=Path, help='Directory to read training curriculum from')
    parser.add_argument("--output-dir", type=Path, help='Outer directory to output downsampled curriculum to')
    parser.add_argument("--num-samples", type=int, help='Number of samples to retrieve from input curriculum')
    parser.add_argument("--random-seed", required=False, type=int, help='Optional random seed.')
    parser.add_argument("--curriculum-name", required=False, type=str, help="Name of training curriculum")
    parser.add_argument(
        "-f", "--force",
        action='store_true',
        required=False,
        help='Force overwrite of target directory. By default, the script exits with an error if there already exists '
             'a directory at the target destination.'
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    num_samples: int = args.num_samples
    random_seed: int = args.random_seed
    force_overwrite: int = args.force
    curriculum_name: str = args.curriculum_name
    if not curriculum_name:
        curriculum_name = str(output_dir).split('/')[-1]

    if not input_dir.exists():
        logging.warning("Input directory does not exist")
        raise FileNotFoundError(str(input_dir))

    if output_dir.is_dir():
        if force_overwrite:
            rmtree(str(output_dir))
        else:
            logging.warning("There already exists a directory in the target location")
            raise FileExistsError(str(output_dir))

    if num_samples < 0:
        logging.warning("Attempting to retrieve a negative number of samples")
        raise ValueError(num_samples)

    if random_seed:
        seed(random_seed)

    object_to_potential_samples: Mapping[str, Sequence[Path]] = dict()
    selected_sample_count: int = 0
    for feature_file_path in sorted(input_dir.glob("situation*")):
        with open(feature_file_path / 'description.yaml', encoding="utf-8") as description_file:
            description_yaml = yaml.safe_load(description_file)
            target_object = description_yaml['language']
            object_to_potential_samples.setdefault(target_object, []).append(feature_file_path)

    logging.info("Downsampling Objects")

    for obj, potential_samples in sorted(object_to_potential_samples.items(), key=lambda x: x[0]):
        logging.info(f"Now processing {obj}")
        selected_samples: Sequence[Path]
        if num_samples > len(potential_samples):
            logging.warning(f"Attempting to retrieve more samples than exist for {obj}. Default to using all samples")
            selected_samples = potential_samples
        else:
            selected_samples = sample(potential_samples, k=num_samples)
        copy_all_directories(output_dir, selected_samples)
        selected_sample_count += len(selected_samples)

    logging.info("Finished copying selected samples")
    logging.info("Updating info.yaml")
    with open(output_dir / 'info.yaml', 'w', encoding="utf-8") as info_file:
        yaml.dump({'curriculum': curriculum_name, 'num_dirs': selected_sample_count}, info_file)
    logging.info("Finished updating info.yaml")


if __name__ == "__main__":
    main()
