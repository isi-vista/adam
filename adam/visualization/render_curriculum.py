import os
from pathlib import Path
from typing import List

from adam.curriculum.phase1_curriculum import (
    build_gaila_phase_1_curriculum as build_curriculum,
)

# helper function to locating a curriculum
from adam.experiment.log_experiment import curriculum_from_params
from adam.language_specific.english.english_language_generator import (
    GAILA_PHASE_1_LANGUAGE_GENERATOR,
)
from adam.visualization.make_scenes import main as make_scenes
from adam.visualization.panda3d_interface import SituationVisualizer
from vistautils.parameters import Parameters, YAMLParametersLoader
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

USAGE_MESSAGE = """"""

# TODO: remove this or log its use as an error
#       https://github.com/isi-vista/adam/issues/689
EXCLUDED_CURRICULA = {
    "objects-in-other-objects",
    "behind_in_front_curriculum",
    "rolling",
    "addressee_curriculum",
    "drinking",
    "sitting",
    "eating",
    "spin",
    "come",
    "putting-on-body-part-addressee-speaker",
}


def main(params: Parameters):
    viz = SituationVisualizer()
    # try to get the directory for rendering for an experiment
    adam_root = params.existing_directory("adam_root")
    root_output_directory = params.optional_creatable_directory("experiment_group_dir")
    if root_output_directory is not None:
        m9_experiments_dir = adam_root / "parameters" / "experiments" / "m9"
        param_files: List[Path] = []

        if params.boolean("include_objects"):
            param_files.append(m9_experiments_dir / "objects.params")

        if params.boolean("include_attributes"):
            param_files.append(m9_experiments_dir / "attributes.params")

        if params.boolean("include_relations"):
            param_files.append(m9_experiments_dir / "relations.params")

        if params.boolean("include_events"):
            param_files.append(m9_experiments_dir / "events.params")

        # This activates a special "debug" curriculum,
        # which is meant to be edited in the code by a developer to do fine-grained debugging.
        if params.boolean("include_debug", default=False):
            param_files.append(m9_experiments_dir / "debug.params")

        # loop over all experiment params files
        for param_file in param_files:
            experiment_params = YAMLParametersLoader().load(param_file)
            if "curriculum" in experiment_params:
                # get the experiment curriculum list (if there is one)

                curriculum = curriculum_from_params(experiment_params)[0]
                directory_name = experiment_params.string("experiment") + "/renders"
                if not os.path.isdir(root_output_directory / directory_name):
                    os.mkdir(root_output_directory / directory_name)
                for instance_group in curriculum:
                    try:
                        make_scenes(
                            params,
                            [instance_group],
                            root_output_directory / directory_name,
                            viz,
                        )
                    except RuntimeError as err:
                        print(f"uncaught exception: {err}")

    else:
        # render phase 1 scenes:
        root_output_directory = params.optional_creatable_directory(
            "screenshot_directory"
        )
        assert root_output_directory is not None
        if not os.path.isdir(root_output_directory):
            os.mkdir(root_output_directory)
        for idx, instance_group in enumerate(
            build_curriculum(None, None, GAILA_PHASE_1_LANGUAGE_GENERATOR)
        ):
            # do any filtering here
            if instance_group.name() in EXCLUDED_CURRICULA:
                continue
            directory_name = f"{idx:03}-{instance_group.name()}"
            if not os.path.isdir(root_output_directory / directory_name):
                os.mkdir(root_output_directory / directory_name)  # type: ignore

            # then call some function from make_scenes.py to run the curriculum
            make_scenes(
                params, [instance_group], root_output_directory / directory_name, viz
            )


if __name__ == "__main__":
    parameters_only_entry_point(main, usage_message=USAGE_MESSAGE)
