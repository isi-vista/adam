import argparse
import base64
import json
import pickle
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml
from vistautils.parameters import Parameters

from adam.curriculum.curriculum_from_files import phase3_process_scene_dir
from adam.experiment.observer import YAMLLogger
from adam.learner.integrated_learner import SimulatedIntegratedTemplateLearner
from adam.paths import (
    TESTING_CURRICULUM_DIR,
    LEARNERS_DIR,
    PREPROCESSING_DIR,
    TRAINING_CURRICULUM_DIR,
)

MENU_PROMPT = "ADAM Demo UI Valid Commands: (d)ecode, (e)xit, (r)eset curriculum: "
DECODE_PROMPT = "Filepath to the image to decode: "


def configure_learner(
    learner_path: Path,
) -> SimulatedIntegratedTemplateLearner:
    """Load a learner from a pickle file."""
    try:
        return pickle.load(open(learner_path, "rb"))
    # if the learner can't be loaded, just instantiate the default learner and notify the user
    except OSError as err:
        raise RuntimeError(
            "Unable to load existing learner, exiting demo script"
        ) from err


@dataclass
class DemoCurriculum:
    curriculum_name: str
    curriculum_path: Path
    situation_number: int = 0
    color_seed: int = 42

    def initialize_curriculum(self) -> None:
        """Create the curriculum directory"""
        if self.curriculum_path.is_dir():
            print(
                "Curriculum already exists. Configuring situation number automatically."
            )
            info_path = self.curriculum_path / "info.yaml"
            if info_path.exists():
                with open(info_path, encoding="utf-8") as info_file:
                    info_yaml = yaml.safe_load(info_file)
                self.situation_number = int(info_yaml["num_dirs"])

        self.curriculum_path.mkdir(parents=True, exist_ok=True)

    def reset(self, *, initialize_after_reset: bool = True) -> None:
        """Reset the curriculum directory, optionally reinitialize it."""
        if self.curriculum_path.is_dir():
            shutil.rmtree(self.curriculum_path)

        if initialize_after_reset:
            self.initialize_curriculum()

    def preprocess(
        self, image_path: Path, host: str, stroke_model_path: Path, stroke_python: Path
    ) -> Path:
        """Preprocess an image with stroke extraction to configure the curriculum scene."""
        situation_dir = self.curriculum_path / f"situation_{self.situation_number}"
        situation_dir.mkdir(parents=True, exist_ok=True)
        situation_img_path = situation_dir / "rgb_0.png"
        segmentation_path = situation_dir / "semantic_0.png"
        img = cv2.imread(str(image_path))
        img2 = None
        try:
            img2 = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            print("Image is RGBA")
        except ValueError:
            print("Image is RGB")
            img2 = img
        finally:
            cv2.imwrite(str(situation_img_path), img2)

        # Generate a Semantic Segmentation Image
        with situation_img_path.open("rb") as img_file:
            im_bytes = img_file.read()
        im_b64 = base64.b64encode(im_bytes).decode("utf8")
        headers = {"Content-type": "application/json", "Accept": "text/plain"}
        payload = json.dumps({"image": im_b64, "other_key": "value"})
        api = host + "instanceseg"
        print(f"Processing Image Segmentation from host: {api}")
        resp = requests.post(api, data=payload, headers=headers)

        data = resp.json()

        print("Generating Segmentation Mask")
        # 3D array: (N, W, H) where N is number of masks
        masks = np.asarray(data["masks"], dtype=np.uint8)
        print(f"Got {len(masks)} masks.")

        # Check for and warn about overlapping masks. We warn because this is an edge case
        # where we don't have a good solution. Currently, we deal with it using an arbitrary hack
        # of "prefer the later mask", which might give weird results sometimes.
        for i in range(len(masks)):  # pylint: disable=consider-using-enumerate
            for j in range(i + 1, len(masks)):  # pylint: disable=consider-using-enumerate
                if np.any(masks[i] & masks[j]):
                    print(
                        f"Warning: Masks {i} and {j} overlap. Preferring later mask {j}."
                    )

        # Compute pixelwise segmentation/mask IDs
        #
        # That is, an input-image-sized array (shape (W, H)) where each entry is either 0 (meaning
        # not in any mask) or a positive integer identifying which of the `N` output masks this pixel
        # belongs to. When a pixel lies in the overlap of several masks, we arbitrarily take the
        # highest-numbered mask.
        segmentation_ids = (
            np.max(
                masks
                * (1 + np.arange(masks.shape[0], dtype=masks.dtype))[
                    :, np.newaxis, np.newaxis
                ],
                axis=0,
            )
            if masks.size > 0
            else np.zeros(img.shape[:2], dtype=np.uint32)
        )
        # Create segmentation mask image by assigning random colors to each mask
        rng = np.random.default_rng(self.color_seed)
        segment_colors = np.zeros((len(masks) + 1, 3), dtype=np.uint8)
        segment_colors[1:, :] = rng.uniform(0, 256, size=(len(masks), 3)).astype(np.uint8)
        bgr_segmentation_ids = np.take(segment_colors, indices=segmentation_ids, axis=0)
        print("Images processed.")

        cv2.imwrite(str(segmentation_path), bgr_segmentation_ids)

        curriculum_path_str = shlex.quote(str(self.curriculum_path))
        # Make a request to stroke extraction
        print("Object Stroke Extraction processing")
        subprocess.run(
            f"{stroke_python} shape_stroke_extraction.py {curriculum_path_str} {curriculum_path_str} --dir-num {self.situation_number}",
            shell=True,
            check=True,
            cwd=PREPROCESSING_DIR,
        )

        # Then decode the strokes & generate feature file
        print("Object Stroke GNN Decode")
        subprocess.run(
            f"{stroke_python} shape_stroke_graph_inference.py {shlex.quote(str(stroke_model_path))} {curriculum_path_str} --save_outputs_to {curriculum_path_str} --dir-num {self.situation_number}",
            shell=True,
            check=True,
            cwd=PREPROCESSING_DIR,
        )

        self.situation_number += 1
        self.info_yaml()
        print("Situation preprocessing complete.")
        return situation_dir

    def info_yaml(self) -> None:
        """Generate the curriculum info.yaml file."""
        info_path = self.curriculum_path / "info.yaml"

        with open(info_path, "w", encoding="utf-8") as info_file:
            yaml.dump(
                {
                    "curriculum": self.curriculum_name,
                    "num_dirs": str(self.situation_number),
                },
                info_file,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "learner_path",
        type=Path,
        help="Path to a pickled ADAM learner to use for live decode.",
    )
    parser.add_argument(
        "--curriculum-name",
        type=str,
        default="m6_live_demo",
        help="Name of the curriculum to save decode into.",
    )
    parser.add_argument(
        "--reset-curriculum",
        action="store_true",
        help="Reset the given curriculum folder rather than appending to it.",
    )
    parser.add_argument(
        "--color-is-rgb", action="store_true", help="Flag to treat color as an RGB value."
    )
    parser.add_argument(
        "--segmentation-api",
        default="http://localhost:5000/",
        help="The web API server to make image segmentation requests to.",
    )
    parser.add_argument(
        "--stroke-model-path",
        required=True,
        type=Path,
        help="Path to the stroke decode mode to use.",
    )
    parser.add_argument(
        "--stroke-python-bin",
        required=True,
        type=Path,
        help="Path to the Stroke Extraction python environment's bin.",
    )
    parser.add_argument(
        "--no-silent-errors",
        action="store_true",
        help="Don't catch errors in scene decode, instead crash.",
    )
    args = parser.parse_args()

    learner_path: Path = args.learner_path
    stroke_model_path: Path = args.stroke_model_path
    stroke_python_env = Path(args.stroke_python_bin) / "python3"

    if not learner_path.is_file():
        raise OSError(
            f"Expected an ADAM learner pickle file path but got: {learner_path}."
        )

    # Prepare the Learner for Decode
    learner: SimulatedIntegratedTemplateLearner = configure_learner(learner_path)

    # Now that we have a learner loaded, we need an observer
    observer_args = Parameters.from_mapping(
        {
            "experiment_output_path": str(
                LEARNERS_DIR
                / "demo_learner"
                / "experiments"
                / args.curriculum_name
                / "test_curriculums"
                / args.curriculum_name
            ),
            "file_name": "post_decode",
            "calculate_accuracy_by_language": True,
            "calculate_overall_accuracy": True,
        }
    )
    observer = YAMLLogger.from_params("test_observer", observer_args)  # type: ignore

    # Configure our 'curriculum' handling to wrap relevant functionality
    faux_train_curriculum = DemoCurriculum(
        args.curriculum_name, TRAINING_CURRICULUM_DIR / args.curriculum_name
    )
    faux_train_curriculum.initialize_curriculum()
    curriculum = DemoCurriculum(
        args.curriculum_name, TESTING_CURRICULUM_DIR / args.curriculum_name
    )
    curriculum.initialize_curriculum()

    # Now we're ready to handle user input
    while True:
        user_input = input(MENU_PROMPT).lower()
        if user_input == "e":
            print("Exiting Live Demo")
            break
        elif user_input == "d":
            # Loop to handle acquiring a file input from the user.
            while True:
                file_path_str = input(DECODE_PROMPT)
                file_path = Path(file_path_str)
                if not file_path.exists() and not file_path.is_file():
                    print("File does not exist. Please try again.")
                    continue
                break
            try:
                # Now we need to preprocess the file, this configures the curriculum scene on disk
                scene_dir = curriculum.preprocess(
                    file_path, args.segmentation_api, stroke_model_path, stroke_python_env
                )

                # Now we need to load this scene into the correct perception format
                (
                    situation,
                    linguistic_description,
                    perception_frame,
                ) = phase3_process_scene_dir(
                    scene_dir, curriculum_type="testing", color_is_rgb=args.color_is_rgb
                )

                # Now we describe the given input
                descriptions_from_learner = learner.describe(perception_frame)

                # The observer should run now
                if not observer:
                    raise RuntimeError(f"Missing observer from arguments {observer_args}")
                observer.observe(
                    situation,
                    linguistic_description,
                    perception_frame,
                    descriptions_from_learner,
                )
                observer.report()

                print(f"Demo scene {curriculum.situation_number} has been processed.")
            # We purposely catch broad exceptions here because we don't want the interactive session to end
            # If an exception occurs in processing a given file
            except Exception as e:  # pylint: disable=broad-except
                print(f"Error encountered while processing situation. {e}")

                if args.no_silent_errors:
                    raise e

        elif user_input == "r":
            observer = YAMLLogger.from_params("test_observer", observer_args)
            curriculum.reset()


if __name__ == "__main__":
    main()
