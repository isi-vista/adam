import logging
from typing import Callable, Optional, Mapping, Iterable, Tuple
import random
from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point


def create_gaze_ablation_entry_point(params: Parameters) -> None:
    """This function creates all possible gaze ablation param files within a given range"""
    # get the parameters directory, which must be non-null
    parameters_dir = params.optional_creatable_directory("parameters_directory")
    if not parameters_dir:
        raise RuntimeError(
            "Must specify a directory where you wish to write your param files"
        )
    # get the minimum and maximum number of objects in a scene
    # TODO: is 1 a good default since it won't make a difference or no?
    # min_num_objects = params.integer("min_num_objects", default=1)
    # max_num_objects = params.integer("max_num_objects", default=7)


if __name__ == "__main__":
    parameters_only_entry_point(create_gaze_ablation_entry_point)
