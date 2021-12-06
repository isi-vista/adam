import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Utility script to convert feature YAMLs"
    )
    parser.add_argument("--input-dir", type=Path, help="An input directory to convert all 'feature.yaml' files in.")
    args = parser.parse_args()
    input_dir: Path = args.input_dir

    for feature_file_path in sorted(input_dir.glob("*/feature*.yaml")):
        print(f"Working on file: {feature_file_path}")
        with open(feature_file_path, encoding="utf-8") as feature_file:
            feature_yaml = yaml.safe_load(feature_file)

        output_dict = {
            "objects": [object_ for object_ in feature_yaml]
        }

        feature_file_output_parts = "/".join(feature_file_path.parts[:-1])
        with open(Path(feature_file_output_parts) / "feature.yaml", "w", encoding="utf-8") as feature_file:
            yaml.dump(output_dict, feature_file)


if __name__ == "__main__":
    main()
