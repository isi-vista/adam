from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from pegasus_wrapper import (
    initialize_vista_pegasus_wrapper,
    directory_for,
    run_python_on_parameters,
    Locator,
    write_workflow_description,
)
import numpy as np
from adam.experiment.log_experiment import log_experiment_entry_point


def gaze_ablation_runner_entry_point(params: Parameters) -> None:
    """This function creates all possible gaze ablation param files within a given range"""
    initialize_vista_pegasus_wrapper(params)

    # Get the baseline experiment parameters for gaze ablation -- these are things common to all of
    # the experiments, like:
    #
    #     include_image_links: true
    #     sort_learner_descriptions_by_length: True
    #     num_pretty_descriptions: 5
    baseline_parameters = params.namespace("gaze_ablation")

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
                        # both ignoring and perceiving gaze
                        for add_gaze in [True, False]:
                            # add the required arguments to create a unique filename
                            experiment_name = Locator(
                                EXPERIMENT_NAME_FORMAT.format(
                                    num_instances=num_instances,
                                    num_noise_instances=num_noise_instances,
                                    num_objects_in_instance=num_objects_in_instance,
                                    prob_given=prob_given,
                                    prob_not_given=prob_not_given,
                                    add_gaze=add_gaze,
                                ).split('-')
                            )

                            # Note that the input parameters should include the root params and
                            # anything else we want.
                            experiment_params = baseline_parameters.unify(FIXED_PARAMETERS).unify({
                                "experiment_group_dir": directory_for(experiment_name),
                                "hypothesis_log_dir": directory_for(experiment_name) / "hypotheses",
                                "pursuit-curriculum-params": {
                                    "num_instances": num_instances,
                                    "num_noise_instances": num_noise_instances,
                                    "num_objects_in_instance": num_objects_in_instance,
                                    "add_gaze": add_gaze,
                                    "prob_given": prob_given,
                                    "prob_not_given": prob_not_given,
                                }
                            })

                            run_python_on_parameters(
                                experiment_name,
                                log_experiment_entry_point,
                                experiment_params,
                            )

    write_workflow_description()


EXPERIMENT_NAME_FORMAT = (
    "{num_instances:d}_instances-{num_noise_instances:d}_noise_instances-{num_objects_in_instance:d}_"
    "objects_in_instance-{prob_given:.3f}_given-{prob_not_given:.3f}_not_given-{add_gaze}_gaze"
)

FIXED_PARAMETERS = {
    "curriculum": "pursuit",
    "learner": "pursuit-gaze",
    "pursuit": {
        "learning_factor": 0.05,
        "graph_match_confirmation_threshold": 0.7,
        "lexicon_entry_threshold": 0.7,
        "smoothing_parameter": 0.001,
    },
}

if __name__ == "__main__":
    parameters_only_entry_point(gaze_ablation_runner_entry_point)
