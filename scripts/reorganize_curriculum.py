import argparse
import itertools as itt
import logging
import shutil
from pathlib import Path

import yaml


OBJECTS_LIST = (
    "apple",
    "ball",
    "banana",
    "book",
    "box",
    "chair",
    "cubeblock",
    "cup",
    "desk",
    "floor",
    "mug",
    "orange",
    "paper",
    "sofa",
    "sphereblock",
    "table",
    "toysedan",
    "toytruck",
    "triangleblock",
    "window",
)
N_EXAMPLES_PER_OBJECT = 10
N_CAMERAS = 3


# These must also match the ontology node labels due to how the GNN recognizer works.
OBJECT_DEBUG_NAME_TO_NAME = {
    "toysedan": "toy_sedan",
    "toytruck": "toy_truck",
    "cubeblock": "cube_block",
    "sphereblock": "sphere_block",
    "triangleblock": "pyramid_block",
}


def _removeprefix(string: str, prefix: str) -> str:
    """
    Shim for Python 3.9+ removeprefix.
    """
    if string[:len(prefix)] == prefix:
        return string[len(prefix):]
    else:
        return string


def main():
    parser = argparse.ArgumentParser(
        description="Utility script to reorganize curricula"
    )
    parser.add_argument("--input-feature-dir", type=Path, help="An input directory of the features", required=True)
    parser.add_argument("--input-cur-dir", type=Path, help="An input directory of the curriculum", required=True)
    parser.add_argument("--input-split", type=str, help="The input curriculum split to process", required=True)
    parser.add_argument(
        "--input-slice",
        type=str,
        help='The input curriculum slice to use, as a string. This could be "small_single_" '
        'meaning use the complete set of generated curriculum files. It can also be the empty '
        'string "", the subset of files the processing code uses). This is the bit between the '
        'split name and the object name in the curriculum file directory names, i.e. '
        '{split}_{prefix}{object_name}. Defaults to "small_single_", because all such files are '
        'uploaded to the Google Drive while the subsetted versions may not be.',
        default="small_single_"
    )
    parser.add_argument("--output-dir", type=Path, help="The curriculum output directory", required=True)
    args = parser.parse_args()

    output_dir: Path = args.output_dir

    situation_num = 0
    for object_debug_name, range_examples, n_cameras in zip(
        OBJECTS_LIST, itt.repeat(N_EXAMPLES_PER_OBJECT), itt.repeat(N_CAMERAS)
    ):
        for cam in range(n_cameras):
            input_curriculum_dir: Path = args.input_cur_dir / f"{args.input_split}_{args.input_slice}{object_debug_name}" / f"cam{cam}"
            if not input_curriculum_dir.exists():
                logging.warning(
                    "Input curriculum dir %s does not exist, so globs on this directory will fail.",
                    input_curriculum_dir
                )
            for ex in range(range_examples):
                # Check for strokes
                if not any(
                    args.input_feature_dir.glob(
                        f"stroke_{args.input_split}_{object_debug_name}_{cam}_{ex}*"
                    )
                ):
                    logging.warning(
                        "Missing strokes for object %s (camera %d, example %d); skipping...",
                        object_debug_name,
                        cam,
                        ex,
                    )
                    continue

                output_situation = output_dir / f"situation_{situation_num}"
                output_situation.mkdir(parents=True)
                # Depth Files
                # Absent in M5 curriculum
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"depth_*_{ex}.png"))):
                    # We handle these files separately
                    if file.name.startswith("depth_vis"):
                        continue
                    shutil.copy(file, output_situation / f"depth_{idx}.png")
                # Depth Vis Files
                # In M5 curriculum, only present in small_single_ slice
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"depth_vis_*_{ex}.png"))):
                    shutil.copy(file, output_situation / f"depthvis_{idx}.png")
                # PCD Files
                # In M5 curriculum, only present in small_single_ slice
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"pcd_rgb*_{ex}.ply"))):
                    shutil.copy(file, output_situation / f"pdc_rgb_{idx}.ply")
                # RGB Files
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"rgb_*_{ex}.png"))):
                    shutil.copy(file, output_situation / f"rgb_{idx}.png")
                if not any(input_curriculum_dir.glob(f"rgb__{ex}*")):
                    logging.warning(
                        "Missing RGB image for object %s (camera %d, example %d).",
                        object_debug_name,
                        cam,
                        ex,
                    )
                # Infrared Files
                # In M5 curriculum, only present in small_single_ slice
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"infrared_*_{ex}.png"))):
                    shutil.copy(file, output_situation / f"infrared_{idx}.png")
                # Normal Files
                # In M5 curriculum, only present in small_single_ slice
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"normal_*_{ex}.png"))):
                    shutil.copy(file, output_situation / f"normal_{idx}.png")
                # Flow? Files
                # Maybe related to optical flow? Not present for in mug samples
                # In M5 curriculum, only present in small_single_ slice
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"flow_*_{ex}.flo"))):
                    shutil.copy(file, output_situation / f"flow_{idx}.flo")
                # Flowvis? Files
                # Maybe related to optical flow? Not present for in mug samples
                # In M5 curriculum, only present in small_single_ slice
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"flowvis_*_{ex}.png"))):
                    shutil.copy(file, output_situation / f"flowvis_{idx}.png")
                # Optical Flow Files
                # In M5 curriculum, only present in small_single_ slice
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"opticalflow_*_{ex}.png"))):
                    shutil.copy(file, output_situation / f"optical_flow_{idx}.png")
                # Optical Flow Vis Files
                # In M5 curriculum, only present in small_single_ slice
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"opticalflowvis_*_{ex}.png"))):
                    shutil.copy(file, output_situation / f"optical_flowvis_{idx}.png")
                # PCD Semantic Files
                # In M5 curriculum, only present in small_single_ slice
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"pcd_semantic_*_{ex}.ply"))):
                    shutil.copy(file, output_situation / f"pdc_semantic_{idx}.ply")
                # Semantic Files
                for idx, file in enumerate(sorted(input_curriculum_dir.glob(f"semantic_*_{ex}.png"))):
                    shutil.copy(file, output_situation / f"semantic_{idx}.png")
                if not any(input_curriculum_dir.glob(f"semantic__{ex}*")):
                    logging.warning(
                        "Missing semantic image for object %s (camera %d, example %d).",
                        object_debug_name,
                        cam,
                        ex,
                    )
                # Feature File
                features_found = None
                for file in sorted(
                    # Note the input slice isn't used in the feature directory or file names.
                    args.input_feature_dir.glob(
                        f"feature_{args.input_split}_{object_debug_name}_{cam}_{ex}*"
                    )
                ):
                    with open(file, encoding="utf-8") as feature_file:
                        feature_yaml = yaml.safe_load(feature_file)

                    if features_found:
                        logging.warning(
                            "Multiple feature files for object %s (camera %d, example %d): "
                            "Got %s and %s",
                            object_debug_name,
                            cam,
                            ex,
                            feature_file,
                            features_found,
                        )
                    features_found = True

                    for object in feature_yaml:
                        # Clean up the object name so it matches the internal ones
                        object["stroke_graph"]["concept_name"] = _removeprefix(
                            object["stroke_graph"]["concept_name"], args.input_split + "_"
                        )
                        # And modify it so it matches what's in phase3_ontology
                        object["stroke_graph"]["concept_name"] = OBJECT_DEBUG_NAME_TO_NAME.get(
                            object["stroke_graph"]["concept_name"], object["stroke_graph"]["concept_name"]
                        )

                    output_dict = {
                        "objects": [object_ for object_ in feature_yaml]
                    }

                    with open(output_situation / "feature.yaml", "w", encoding="utf-8") as feature_file:
                        yaml.dump(output_dict, feature_file)
                # Stroke File
                for idx, file in enumerate(
                    sorted(
                        args.input_feature_dir.glob(
                            f"stroke_{args.input_split}_{object_debug_name}_{cam}_{ex}*"
                        )
                    )
                ):
                    shutil.copy(file, output_situation / f"stroke_{idx}_{idx}.png")
                # Stroke Graph File
                for idx, file in enumerate(
                    sorted(
                        args.input_feature_dir.glob(
                            f"stroke_graph_{args.input_split}_{object_debug_name}_{cam}_{ex}*"
                        )
                    )
                ):
                    shutil.copy(file, output_situation / f"stroke_graph_{idx}.png")
                if not any(
                    args.input_feature_dir.glob(
                        f"stroke_graph_{args.input_split}_{object_debug_name}_{cam}_{ex}*"
                    )
                ):
                    logging.warning(
                        "Missing stroke graph for object %s (camera %d, example %d).",
                        object_debug_name,
                        cam,
                        ex,
                    )
                # Description file
                object_name = OBJECT_DEBUG_NAME_TO_NAME.get(object_debug_name, object_debug_name)
                with open(output_situation / "description.yaml", "w", encoding="utf-8") as description_file:
                    yaml.dump({"language": f"a {object_name}"}, description_file)

                situation_num += 1

    with open(output_dir / "info.yaml", "w", encoding="utf-8") as info_file:
        yaml.dump({"curriculum": output_dir.stem, "num_dirs": situation_num}, info_file)


if __name__ == "__main__":
    main()
