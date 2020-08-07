from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
import numpy as np


def create_gaze_ablation_entry_point(params: Parameters) -> None:
    """This function creates all possible gaze ablation param files within a given range"""
    # get the parameters directory, which must be non-null
    parameters_dir = params.optional_creatable_directory("parameters_directory")
    if not parameters_dir:
        raise RuntimeError(
            "Must specify a directory where you wish to write your param files"
        )
    # get the minimum and maximum number of objects in a scene
    min_num_objects = params.integer("min_num_objects", default=1)
    max_num_objects = params.integer("max_num_objects", default=7)

    # this gets the number of different accuracies to try; default = increment by 0.1
    num_accuracy_increments = params.integer("num_increments", default=11)
    values_for_accuracy = np.linspace(0, 1, num_accuracy_increments)

    # the number of noise instances to be included
    min_num_noise_instances = params.integer("min_num_noise", default=0)
    max_num_noise_instances = params.integer("max_num_noise", default=0)

    # get the number of instances in the entire curriculum
    min_num_instances_in_curriculum = params.integer("min_instances", default=10)
    max_num_instances_in_curriculum = params.integer("max_instances", default=20)

    # all possible numbers of noise instances
    for num_noise_instances in range(
        min_num_noise_instances, max_num_noise_instances + 1
    ):
        # all possible numbers of instances in the curriculum
        for num_instances in range(
            min_num_instances_in_curriculum, max_num_instances_in_curriculum + 1
        ):
            # all possible numbers of instances
            for num_objects_in_instance in range(min_num_objects, max_num_objects + 1):
                # all possible accuracies
                for prob_given in values_for_accuracy:
                    for prob_not_given in values_for_accuracy:
                        for add_gaze in [True, False]:
                            file_name = (
                                f"pursuit-ablating-gaze-{num_instances}_instances-{num_noise_instances}_noise_instances-"
                                f"{num_objects_in_instance}_objects_in_instance-{prob_given}_given-{prob_not_given}_not_given-{add_gaze}"
                                f"_gaze"
                            )
                            param_file_string = f"""_includes:
   - "../../root.params"
   - "m13.params"

experiment: '{file_name}'
curriculum: pursuit
learner: pursuit-gaze
accuracy_to_txt : True

pursuit:
   learning_factor: 0.05
   graph_match_confirmation_threshold: 0.7
   lexicon_entry_threshold: 0.7
   smoothing_parameter: .001

pursuit-curriculum-params:
   num_instances: {num_instances}
   num_noise_instances: {num_noise_instances}
   num_objects_in_instance: {num_objects_in_instance}
   add_gaze : {add_gaze}
   prob_given : {prob_given}
   prob_not_given : {prob_not_given}"""

                        with open(f"{parameters_dir}/{file_name}", "a") as f:
                            f.write(param_file_string)


if __name__ == "__main__":
    parameters_only_entry_point(create_gaze_ablation_entry_point)
