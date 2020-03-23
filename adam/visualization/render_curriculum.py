from adam.curriculum.phase1_curriculum import (
    build_gaila_phase_1_curriculum as build_curriculum,
)
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from vistautils.parameters import Parameters

import os

from adam.visualization.make_scenes import main as make_scenes
from adam.visualization.panda3d_interface import SituationVisualizer

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
    root_output_directory = params.string("screenshot_directory")
    if not os.path.isdir(root_output_directory):
        os.mkdir(root_output_directory)
    for idx, instance_group in enumerate(build_curriculum()):
        # do any filtering here
        if instance_group.name() in EXCLUDED_CURRICULA:
            continue
        directory_name = f"{idx:03}-{instance_group.name()}"
        if not os.path.isdir(root_output_directory + "/" + directory_name):
            os.mkdir(root_output_directory + "/" + directory_name)  # type: ignore

        # then call some function from make_scenes.py to run the curriculum
        make_scenes(
            params, [instance_group], root_output_directory + "/" + directory_name, viz
        )


if __name__ == "__main__":
    parameters_only_entry_point(main, usage_message=USAGE_MESSAGE)