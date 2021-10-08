from typing import Any, Dict, Tuple, List

import numpy as np

from vistautils.parameters import Parameters

from pegasus_wrapper import (
    initialize_vista_pegasus_wrapper,
    directory_for,
    run_python_on_parameters,
    Locator,
    write_workflow_description,
    limit_jobs_for_category,
)
from pegasus_wrapper.resource_request import SlurmResourceRequest

from vistautils.parameters_only_entrypoint import parameters_only_entry_point
import adam.experiment.log_experiment as log_experiment_script


def object_language_ablation_runner_entry_point(params: Parameters) -> None:
    """This function creates all possible object language ablation param files within a given range"""
    initialize_vista_pegasus_wrapper(params)

    baseline_parameters = params.namespace("object_language_ablation")
    pursuit_resource_request_params = params.namespace("pursuit_resource_request")

    # get the minimum and maximum number of objects in a scene
    min_num_objects = params.integer("min_num_objects", default=1)
    max_num_objects = params.integer("max_num_objects", default=7)

    # get the minimum and maximum accuracy of the language with the situation
    min_language_accuracy = params.floating_point("min_language_accuracy", default=0.1)
    max_language_accuracy = params.floating_point("max_language_accuracy", default=0.5)
    num_language_accuracy_increment = params.integer(
        "num_language_accuracy_increment", default=5
    )
    values_for_accuracy = np.linspace(
        min_language_accuracy, max_language_accuracy, num_language_accuracy_increment
    )

    limit_jobs_for_category(
        "pursuit", params.integer("num_pursuit_learners_active", default=8)
    )

    for num_objects in range(min_num_objects, max_num_objects + 1):
        for language_accuracy in values_for_accuracy:
            for (learner_type, params_value) in LEARNER_VALUES_TO_PARAMS.items():
                for params_str, learner_params in params_value:
                    experiment_name_string = EXPERIMENT_NAME_FORMAT.format(
                        num_objects=num_objects,
                        language_accuracy=language_accuracy,
                        learner_type=learner_type,
                        learner_params=params_str,
                    )
                    experiment_name = Locator(experiment_name_string.split("-"))

                    # Note that the input parameters should include the root params and
                    # anything else we want.
                    experiment_params = baseline_parameters.unify(FIXED_PARAMETERS).unify(
                        {
                            "experiment": experiment_name_string,
                            "experiment_group_dir": directory_for(experiment_name),
                            "hypothesis_log_dir": directory_for(experiment_name)
                            / "hypotheses",
                            "learner_logging_path": directory_for(experiment_name),
                            "log_learner_state": True,
                            "resume_from_latest_logged_state": True,
                            "train_curriculum": {
                                "accurate_language_percentage": float(language_accuracy)
                            },
                            "object_learner_type": learner_type,
                            "object_learner": learner_params,
                            # We subtract one because the target object is a given
                            "num_noise_objects": num_objects - 1,
                        }
                    )

                    run_python_on_parameters(
                        experiment_name,
                        log_experiment_script,
                        experiment_params,
                        depends_on=[],
                        resource_request=SlurmResourceRequest.from_parameters(
                            pursuit_resource_request_params
                        )
                        if learner_type == "pursuit"
                        else None,
                        category=learner_type,
                    )

    write_workflow_description()


EXPERIMENT_NAME_FORMAT = "{num_objects:d}_objects-{language_accuracy:.2f}_language_accuracy-{learner_type}_object_learner-{learner_params}_params"

# ["subset", "pbv", "cross_situational", "pursuit"]
LEARNER_VALUES_TO_PARAMS: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {
    "subset": [("subset", {"learner_type": "subset", "ontology": "phase2"})],
    "pbv": [
        (
            "0.9_graph_match_conf",
            {
                "learner_type": "pbv",
                "random_seed": 0,
                "graph_match_confirmation_threshold": 0.9,
            },
        ),
        (
            "0.95_graph_match_conf",
            {
                "learner_type": "pbv",
                "random_seed": 0,
                "graph_match_confirmation_threshold": 0.95,
            },
        ),
        (
            "1.0_graph_match_conf",
            {
                "learner_type": "pbv",
                "random_seed": 0,
                "graph_match_confirmation_threshold": 1.0,
            },
        ),
    ],
    "cross_situational": [
        (
            "0.9_graph_match_conf",
            {
                "learner_type": "cross-situational",
                "lexicon_entry_threshold": 0.7,
                "smoothing_parameter": 0.001,
                "graph_match_confirmation_threshold": 0.9,
            },
        ),
        (
            "0.95_graph_match_conf",
            {
                "learner_type": "cross-situational",
                "lexicon_entry_threshold": 0.7,
                "smoothing_parameter": 0.001,
                "graph_match_confirmation_threshold": 0.95,
            },
        ),
        (
            "1.0_graph_match_conf",
            {
                "learner_type": "cross-situational",
                "lexicon_entry_threshold": 0.7,
                "smoothing_parameter": 0.001,
                "graph_match_confirmation_threshold": 1.0,
            },
        ),
    ],
    "pursuit": [
        (
            "0.9_graph_match_conf",
            {
                "learner_type": "pursuit",
                "learning_factor": 0.02,
                "graph_match_confirmation_threshold": 0.9,
                "lexicon_entry_threshold": 0.7,
                "smoothing_parameter": 0.001,
            },
        ),
        (
            "0.95_graph_match_conf",
            {
                "learner_type": "pursuit",
                "learning_factor": 0.02,
                "graph_match_confirmation_threshold": 0.95,
                "lexicon_entry_threshold": 0.7,
                "smoothing_parameter": 0.001,
            },
        ),
        (
            "1.0_graph_match_conf",
            {
                "learner_type": "pursuit",
                "learning_factor": 0.02,
                "graph_match_confirmation_threshold": 1.0,
                "lexicon_entry_threshold": 0.7,
                "smoothing_parameter": 0.001,
            },
        ),
    ],
}

FIXED_PARAMETERS = {
    "curriculum": "m15-object-noise-experiments",
    "learner": "integrated-learner-params",
    "post_observer": {
        "include_acc_observer": False,
        "include_pr_observer": True,
        "log_pr": True,
    },
    "attribute_learner": {"learner_type": "none"},
    "relation_learner": {"learner_type": "none"},
    "action_learner": {"learner_type": "none"},
    "plural_learner": {"learner_type": "none"},
    "include_functional_learner": False,
    "include_generics_learner": False,
    "test_observer": {"accuracy_to_txt": True},
}

if __name__ == "__main__":
    parameters_only_entry_point(object_language_ablation_runner_entry_point)
