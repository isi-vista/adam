import argparse
import itertools as itt
import logging
import shutil
from pathlib import Path

import yaml


ACTIONS_LIST = (
    "take",
    "sit",
    "running",
    "eat",
    "put",
    "open",
    "jump",
    "drinking",
    "go",
    "close",
    "falling",
    "writing",
    "stack",
    "throw",
    "shake",
    "give",
    "spin",
    "walking",
    "spill",
    "shove",
)
N_EXAMPLES_PER_ACTION = 10
N_EXAMPLES_PER_TEST_ACTION = 1
N_CAMERAS = 3


ACTION_DEBUG_NAME_TO_NAME = {
    "running": "run",
    "walking": "walk",
    "falling": "fall",
    "drinking": "drink",
    "writing": "write",
}

ACTION_TO_TWO_CONCEPT_AFFORDANCE = {
    "take": "person take box",
    "sit": "person sit sofa",
    "eat": "person eat apple",
    "put": "person put box",
    "open": "person open box",
    "drink": "person drink water",
    "close": "person close box",
    "stack": "person stack box",
    "throw": "person throw box",
    "shake": "person shake apple",
    "give": "person give box",
    "spin": "person spin apple",
    "spill": "person spill water",
    "shove": "person shove box",
}


def main():
    parser = argparse.ArgumentParser(
        description="Utility script to reorganize Action curricula"
    )
    parser.add_argument("--input-feature-dir", type=Path, help="An input directory of the features", required=True)
    parser.add_argument("--input-cur-dir", type=Path, help="An input directory of the curriculum", required=True)
    parser.add_argument("--input-split", type=str, help="The input curriculum split to process", required=True)
    parser.add_argument("--annotation-files", type=Path, nargs="*", help="The annotation files to combine as object lookup", default=list())
    parser.add_argument("--output-dir", type=Path, help="The curriculum output directory", required=True)
    parser.add_argument("--language-person-only", action="store_true", help="Flag to build language for the scene which is only in the form 'a person X' where X is the action.")
    parser.add_argument("--language-dual-concept", action="store_true", help="Flag to build test affordance curriculum with two concepts.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir: Path = args.output_dir

    if not args.language_person_only and args.language_dual_concept:
        raise RuntimeError("Can't combine slot fill generation with dual concept templates.")

    objects_to_append = dict()
    for file in args.annotation_files:
        if not file.is_file():
            raise ValueError(f"Input file ({file} is not a valid file.")
        with open(file) as annotations:
            for key, value in yaml.safe_load(annotations).items():
                if key in objects_to_append:
                    logging.warning("Annotation file %s overwriting key (%s) with value (%s) with new value (%s)", file.name, key, objects_to_append[key], value)
                objects_to_append[key] = value

    situation_num = 0
    actions_covered = set()
    for action_debug_name, range_examples, n_cameras in zip(
        ACTIONS_LIST, itt.repeat(N_EXAMPLES_PER_ACTION if args.input_split.lower() == "train" else N_EXAMPLES_PER_TEST_ACTION), itt.repeat(N_CAMERAS)
    ):
        for ex in range(range_examples):
            for cam in range(n_cameras):
                input_curriculum_dir: Path = args.input_cur_dir / action_debug_name / args.input_split / str(ex) / f"cam{cam}"
                # print(f"Processing {input_curriculum_dir}")
                if not input_curriculum_dir.exists():
                    logging.warning(
                        "Input curriculum dir %s does not exist, so globs on this directory will fail.",
                        input_curriculum_dir
                    )
                # Check if trajectory extraction succeeded
                action_feature_file = args.input_feature_dir / f"{action_debug_name}_{args.input_split}_{ex}_{cam}.yaml"
                if not action_feature_file.is_file():
                    logging.warning(
                        "Missing features for action %s (camera %d, example %d); skipping...",
                        action_debug_name,
                        cam,
                        ex,
                    )
                    continue

                with open(action_feature_file, encoding="utf-8") as feature_file:
                    action_features = yaml.safe_load(feature_file)
                if all(
                    all(point[0] == point[1] == point[2] == -1 for point in coordinates) for coordinates in action_features["stroke_graph"]["joint_points_world_coords"]
                ):
                    logging.warning(
                        "Trajectory processing for action %s (camera %d, example %d) failed; skipping...",
                        action_debug_name,
                        cam,
                        ex,
                    )
                    continue

                output_situation = output_dir / f"situation_{situation_num}"
                output_situation.mkdir(parents=True)
                # Semantic Files
                semantic_glob = "semantic_*"
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(semantic_glob))):
                    shutil.copy(file, output_situation / f"semantic_{idx}.png")
                if not any(input_curriculum_dir.glob(semantic_glob)):
                    logging.warning(
                        "Missing semantic image for object %s (camera %d, example %d).",
                        action_debug_name,
                        cam,
                        ex,
                    )
                # RGB Files
                rgb_glob = "rgb_*"
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(rgb_glob))):
                    shutil.copy(file, output_situation / f"rgb_{idx}.png")
                if not any(input_curriculum_dir.glob(rgb_glob)):
                    logging.warning(
                        "Missing RGB image for object %s (camera %d, example %d).",
                        action_debug_name,
                        cam,
                        ex,
                    )
                # PCD Semantic Files
                for idx, file in enumerate(sorted(input_curriculum_dir.glob("pcd_semantic*"))):
                    shutil.copy(file, output_situation / f"pcd_semantic_{idx}.ply")
                # PCD RGB Files
                for idx, file in enumerate(sorted(input_curriculum_dir.glob("pdc_rgb*"))):
                    shutil.copy(file, output_situation / f"pdc_rgb_{idx}.ply")
                # Optical Flow Files
                for idx, file in enumerate(sorted(input_curriculum_dir.glob("opticalflow_*"))):
                    shutil.copy(file, output_situation / f"opticalflow_{idx}.png")
                # Optical Flow Vis Files
                for idx, file in enumerate(sorted(input_curriculum_dir.glob("opticalflowvis_*"))):
                    shutil.copy(file, output_situation / f"opticalflowvis_{idx}.png")
                # Surface Normal Files
                for idx, file in enumerate(sorted(input_curriculum_dir.glob("normal_*"))):
                    shutil.copy(file, output_situation / f"normal_{idx}.png")
                # Infrared Files
                for idx, file in enumerate(sorted(input_curriculum_dir.glob("infrared_*"))):
                    shutil.copy(file, output_situation / f"infrared_{idx}.png")
                # Depth Vis Files
                for idx, file in enumerate(sorted(input_curriculum_dir.glob("depth_vis_*"))):
                    shutil.copy(file, output_situation / f"depthvis_{idx}.png")
                # Depth Files
                for idx, depth_file in enumerate(file for file in sorted(input_curriculum_dir.glob("depth_*")) if "vis" not in file.stem):
                    shutil.copy(depth_file, output_situation / f"depth_{idx}.png")
                # Feature Files
                # Fake Object Files
                object_features = {"objects": [
                    {
                        "color": [0, 0, 0],
                        "object_name": f"object{idx}",
                        "sub_part": None,
                        "subobject_id": '0',
                        "texture": None,
                        "distance": None,
                        "viewpoint_id": cam,
                        "stroke_graph": {
                            "adjacency_matrix": [],
                            "concept_name": object_name,
                            "confidence_score": 1.0,
                            "stroke_mean_x": 150.0,
                            "stroke_mean_y": 100.0,
                            "stroke_std": 2.0,
                            "strokes_normalized_coordinates": []
                        }
                    }
                    for idx, object_name in enumerate(itt.chain(["person"], objects_to_append.get(f"{args.input_split}_{action_debug_name}_{ex}_{cam}", [])))
                ]}
                with open(output_situation / "feature_0.yaml", "w", encoding="utf-8") as feature_obj_0:
                    yaml.dump(object_features, feature_obj_0)
                with open(output_situation / "feature_1.yaml", "w", encoding="utf-8") as feature_obj_1:
                    yaml.dump(object_features, feature_obj_1)

                # Action Feature copy
                with open(action_feature_file, encoding="utf-8") as action_feature_file:
                    action_features = yaml.safe_load(action_feature_file)
                action_features["object_name"] = "0"
                annotated_objects = objects_to_append.get(f"{args.input_split}_{action_debug_name}_{ex}_{cam}", [])
                action_features["objects"] = [
                    str(value + 1) for value in range(len(annotated_objects))
                ] if action_features["objects"] is None else action_features["objects"]
                with open(output_situation / "action.yaml", "w", encoding="utf-8") as action_feature_out:
                    yaml.dump(action_features, action_feature_out)

                # Build output language
                action_name = ACTION_DEBUG_NAME_TO_NAME.get(action_debug_name, action_debug_name)
                default_language = language = f"a person {action_name}"
                if args.language_dual_concept:
                    language = ACTION_TO_TWO_CONCEPT_AFFORDANCE.get(action_name, default_language)
                else:
                    if not args.language_person_only and annotated_objects:
                        language = f"{default_language} a {annotated_objects[0]}"

                # Description file
                with open(output_situation / "description.yaml", "w", encoding="utf-8") as description_file:
                    yaml.dump({"language": language}, description_file)

                situation_num += 1
                actions_covered.add(action_debug_name)

    with open(output_dir / "info.yaml", "w", encoding="utf-8") as info_file:
        yaml.dump({"curriculum": output_dir.stem, "num_dirs": situation_num}, info_file)
    logging.info("Saved %d situations.", situation_num)
    logging.info("Actions covered: %s", sorted(actions_covered))
    logging.info("Actions not covered: %s", sorted(set(ACTIONS_LIST) - actions_covered))


if __name__ == "__main__":
    main()
