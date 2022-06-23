import argparse
import logging
from pathlib import Path
from shutil import rmtree, copytree
from typing import Mapping, Sequence

import yaml

situation_num: int = 0


def get_situation_num(situation_dir: Path) -> int:
    """Given a Path for a situation directory, return the associated number"""
    return int(situation_dir.name.split('_')[-1])


def copy_directory(start_path: Path, end_path: Path) -> None:
    """Function to recursively copy directory from start_path to end_path"""
    copytree(str(start_path), str(end_path))


def copy_all_directories(output_dir: Path, dirs_to_copy: Sequence[Path]):
    """Move all directories in dirs_to_copy to their corresponding location in the target directory"""
    global situation_num
    sorted_dirs = sorted(dirs_to_copy, key=get_situation_num)
    for directory in sorted_dirs:
        copy_directory(directory, output_dir / f'situation_{situation_num}')
        situation_num += 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Utility to filter out specified objects from curriculum"
    )
    parser.add_argument('--input-dir', type=Path, help="Directory with curriculum samples")
    parser.add_argument('--output-dir', type=Path, help="Directory to write remaining samples to")
    parser.add_argument("--curriculum-name", required=False, type=str, help="Name of output curriculum")
    parser.add_argument(
        "-f", "--force",
        action='store_true',
        required=False,
        help='Force overwrite of target directory. By default, the script exits with an error if there already exists '
             'a directory at the target destination.'
    )
    parser.add_argument(
        '-u', '--unknown-objects',
        type=str,
        nargs='+',
        required=True,
        help="List of unknown objects to be removed from curriculum"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    force_overwrite: bool = args.force
    curriculum_name: str = args.curriculum_name
    unknown_objects: Sequence[str] = args.unknown_objects
    if not curriculum_name:
        curriculum_name = output_dir.name

    if not input_dir.exists():
        logging.warning("Input directory does not exist")
        raise FileNotFoundError(str(input_dir))

    if output_dir.is_dir():
        if force_overwrite:
            rmtree(str(output_dir))
        else:
            logging.warning("There already exists a directory in the target location")
            raise FileExistsError(str(output_dir))

    object_to_sample_directories: Mapping[str, Sequence[Path]] = dict()
    selected_sample_count: int = 0
    for feature_file_path in sorted(input_dir.glob("situation*")):
        with open(feature_file_path / 'description.yaml', encoding="utf-8") as description_file:
            description_yaml = yaml.safe_load(description_file)
            target_object = description_yaml['language'].split()[-1]  # retrieve only object name
            object_to_sample_directories.setdefault(target_object, []).append(feature_file_path)

    samples_to_copy_over: Mapping[str, Sequence[Path]] = dict()
    for obj in object_to_sample_directories:
        if obj not in unknown_objects:
            samples_to_copy_over[obj] = object_to_sample_directories[obj]
        else:
            logging.info(f"Removing {obj} from sample pool")

    sorted_samples_to_copy_over = sorted(samples_to_copy_over.items(), key=lambda x: x[0])
    for obj, samples in sorted_samples_to_copy_over:
        logging.info(f"Copying {obj} samples")
        copy_all_directories(output_dir, samples)
        selected_sample_count += len(samples)

    logging.info("Finished copying samples")
    logging.info("Updating info.yaml")
    with open(output_dir / 'info.yaml', 'w', encoding="utf-8") as info_file:
        yaml.dump({'curriculum': curriculum_name, 'num_dirs': selected_sample_count}, info_file)
    logging.info("Finished updating info.yaml")


if __name__ == '__main__':
    main()
