import argparse
import shutil
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(
        description="Utility script to reorganize curricula"
    )
    parser.add_argument("--input-feature-dir", type=Path, help="An input directory of the features", required=True)
    parser.add_argument("--input-cur-dir", type=Path, help="An input directory of the curriculum", required=True)
    parser.add_argument("--output-dir", type=Path, help="The curriculum output directory", required=True)
    args = parser.parse_args()

    output_dir: Path = args.output_dir

    situation_num = 0
    for object_name, range_examples in (("cube", 14), ("sphere", 11), ("pyramid", 11)):
        input_curriculum_dir: Path = args.input_cur_dir / object_name
        input_feature_dir: Path = args.input_feature_dir / object_name
        for num in range(1, range_examples):
            output_situation = output_dir / f"situation_{situation_num}"
            output_situation.mkdir(parents=True)
            # Depth Files
            for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"depth__{num}_*"))):
                shutil.copy(file, output_situation / f"depth_{idx}.png")
            # PDC Files
            for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"pdc_rgb__{num}_*"))):
                shutil.copy(file, output_situation / f"pdc_rgb_{idx}.ply")
            # RGB Files
            for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"rgb__{num}_*"))):
                shutil.copy(file, output_situation / f"rgb_{idx}.png")
            # PDC Semantic Files
            for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"pdc_semantic__{num}_*"))):
                shutil.copy(file, output_situation / f"pdc_semantic_{idx}.ply")
            # Semantic Files
            for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"semantic__{num}_*"))):
                shutil.copy(file, output_situation / f"semantic_{idx}.png")
            # Feature File
            for file in sorted(input_feature_dir.glob(f"feature_{num}*")):
                with open(file, encoding="utf-8") as feature_file:
                    feature_yaml = yaml.safe_load(feature_file)

                output_dict = {
                    "objects": [object_ for object_ in feature_yaml]
                }

                with open(output_situation / "feature.yaml", "w", encoding="utf-8") as feature_file:
                    yaml.dump(output_dict, feature_file)
            # Stroke File
            for idx, file in enumerate(sorted(input_feature_dir.glob(f"stroke_{num}_*"))):
                shutil.copy(file, output_situation / f"stroke_{idx}_{idx}.png")
            # Stroke Graph File
            for idx, file in enumerate(sorted(input_feature_dir.glob(f"stroke_graph_{num}_*"))):
                shutil.copy(file, output_situation / f"stroke_graph_{idx}.png")
            # Description file
            with open(output_situation / "description.yaml", "w", encoding="utf-8") as description_file:
                yaml.dump({"language": f"a {object_name}"}, description_file)

            situation_num += 1

    with open(output_dir / "info.yaml", "w", encoding="utf-8") as info_file:
        yaml.dump({"curriculum": output_dir.stem, "num_dirs": situation_num}, info_file)


if __name__ == "__main__":
    main()
