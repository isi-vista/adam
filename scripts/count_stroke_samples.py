import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Utility to count number of samples with successful stroke extraction"
    )
    parser.add_argument("--input-dir", type=Path, help="An input curriculum directory to read objects from.")
    args = parser.parse_args()
    input_dir: Path = args.input_dir
    object_to_stroke_count: dict = dict()

    for feature_file_path in sorted(input_dir.glob("situation*")):
        print(f"Working on file: {feature_file_path}")
        with open(feature_file_path / 'feature.yaml', encoding="utf-8") as feature_file:
            feature_yaml = yaml.safe_load(feature_file)
        with open(feature_file_path / 'description.yaml', encoding="utf-8") as description_file:
            description_yaml = yaml.safe_load(description_file)
            target_object = description_yaml['language']
        if 'stroke_graph' in feature_yaml.get('objects')[0]:
            # Meant to check if stroke graph is in decode, but not really necessary since preprocess filters objects
            # without stroke graph
            object_to_stroke_count[target_object] = object_to_stroke_count.setdefault(target_object, 0) + 1
        else:
            print('missing stroke for', target_object)

    spaces: int = max(len(obj) for obj in object_to_stroke_count) # Total amount of space needed for object name
    sorted_object_to_stroke_count = sorted(object_to_stroke_count.items(), key=lambda x: x[1], reverse=True)
    for obj, stroke_count in sorted_object_to_stroke_count:
        print(f'|\t{obj}' + (spaces - len(obj)) * ' ' + f'\t|\t{stroke_count}\t|')


if __name__ == "__main__":
    main()
