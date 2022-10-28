import argparse
import csv
import pathlib
import sys

import yaml
from yaml import Loader


def main():
    """
    Search for object (by name) in descriptions of situations in curriculum.
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('curriculum_dir', type=pathlib.Path, help='directory containing curriculum to scan for object')
    parser.add_argument('object_name', type=str, help='name of object to search for')
    args = parser.parse_args()

    sorted_dirs = sorted(args.curriculum_dir.iterdir())

    all_descriptions, all_directories = zip(
        *[(yaml.load(open(x / 'description.yaml').read(), Loader=Loader)['language'], x)
        for x in sorted_dirs if x.name.startswith('situation')]
    )

    relevant_idxs = [
        i for i, description in enumerate(all_descriptions)
        if args.object_name in description
    ]
    for idx in relevant_idxs:
        print("Occurence:")
        print(all_directories[idx])
        print(all_descriptions[idx])
