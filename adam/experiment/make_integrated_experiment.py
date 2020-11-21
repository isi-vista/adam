from pegasus_wrapper import (
    initialize_vista_pegasus_wrapper,
    limit_jobs_for_category,
    Locator,
    directory_for,
    run_python_on_parameters,
    write_workflow_description,
)
from pegasus_wrapper.resource_request import SlurmResourceRequest

from vistautils.parameters import Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

import adam.experiment.generate_curriculum as generate_curriculum_script
import adam.experiment.log_experiment as log_experiment_script
from immutablecollections import immutableset


def integrated_experiment_entry_point(params: Parameters) -> None:
    initialize_vista_pegasus_wrapper(params)

    baseline_parameters = params.namespace("integrated_learners_experiment")
    pursuit_resource_request_params = params.namespace("pursuit_resource_request")

    # This code is commented out but may be used in the near future to add language ablation
    # Capabilities to this curriculum.

    # get the minimum and maximum accuracy of the language with the situation
    # min_language_accuracy = params.floating_point("min_language_accuracy", default=0.1)
    # max_language_accuracy = params.floating_point("max_language_accuracy", default=0.5)
    # num_language_accuracy_increment = params.integer(
    #    "num_language_accuracy_increment", default=5
    # )
    # values_for_accuracy = np.linspace(
    #    min_language_accuracy, max_language_accuracy, num_language_accuracy_increment
    # )

    # Get if attributes or relations should be included
    include_attributes = params.boolean("include_attributes", default=True)
    include_relations = params.boolean("include_relations", default=True)

    limit_jobs_for_category(
        "pursuit_job_limit", params.integer("num_pursuit_learners_active", default=8)
    )

    curriculum_repository_path = params.creatable_directory("curriculum_repository_path")

    # Job to build desired curriculum(s) which our learners use

    curriculum_dependencies = immutableset(
        (
            CURRICULUM_NAME_FORMAT.format(
                noise=add_noise,
                shuffled=shuffle,
                relations=include_relations,
                attributes=include_attributes,
            ),
            run_python_on_parameters(
                Locator(
                    CURRICULUM_NAME_FORMAT.format(
                        noise=add_noise,
                        shuffled=shuffle,
                        relations=include_relations,
                        attributes=include_attributes,
                    ).split("-")
                ),
                generate_curriculum_script,
                baseline_parameters.unify(
                    {
                        "train_curriculum": Parameters.from_mapping(CURRICULUM_PARAMS)
                        .unify(
                            {
                                "add_noise": add_noise,
                                "shuffled": shuffle,
                                "include_attributes": include_attributes,
                                "include_relations": include_relations,
                            }
                        )
                        .as_mapping()
                    }
                )
                .unify(FIXED_PARAMETERS)
                .unify({"curriculum_repository_path": curriculum_repository_path}),
                depends_on=[],
            ),
            Parameters.from_mapping(CURRICULUM_PARAMS).unify(
                {
                    "add_noise": add_noise,
                    "shuffled": shuffle,
                    "include_attributes": include_attributes,
                    "include_relations": include_relations,
                }
            ),
        )
        for add_noise in (True, False)
        for shuffle in (True, False)
    )

    # jobs to build experiment
    for (curriculum_str, curriculum_dep, curr_params) in curriculum_dependencies:
        object_learner_type = params.string(
            "object_learner.learner_type",
            valid_options=["pursuit", "subset", "pbv"],
            default="pursuit",
        )
        attribute_learner_type = params.string(
            "attribute_learner.learner__type",
            valid_options=["none", "pursuit", "subset"],
            default="pursuit",
        )
        relation_learner_type = params.string(
            "relation_learner.learner_type",
            valid_options=["none", "pursuit", "subset"],
            default="pursuit",
        )
        experiment_name_string = EXPERIMENT_NAME_FORMAT.format(
            curriculum_name=curriculum_str.replace("-", "+"),
            object_learner=object_learner_type,
            attribute_learner=attribute_learner_type,
            relation_learner=relation_learner_type,
        )
        experiment_name = Locator(experiment_name_string.split("-"))

        # Note that the input parameters should include the root params and
        # anything else we want.
        experiment_params = baseline_parameters.unify(FIXED_PARAMETERS).unify(
            {
                "experiment": experiment_name_string,
                "experiment_group_dir": directory_for(experiment_name),
                "hypothesis_log_dir": directory_for(experiment_name) / "hypotheses",
                "learner_logging_path": directory_for(experiment_name),
                "log_learner_state": True,
                "resume_from_latest_logged_state": True,
                "load_from_curriculum_repository": curriculum_repository_path,
                "train_curriculum": curr_params,
            }
        )

        run_python_on_parameters(
            experiment_name,
            log_experiment_script,
            experiment_params,
            depends_on=[curriculum_dep],
            resource_request=SlurmResourceRequest.from_parameters(
                pursuit_resource_request_params
            )
            if "pursuit"
            in [object_learner_type, attribute_learner_type, relation_learner_type]
            else None,
            category="pursuit"
            if "pursuit"
            in [object_learner_type, attribute_learner_type, relation_learner_type]
            else "subset",
            use_pypy=True,
        )

    write_workflow_description()


EXPERIMENT_NAME_FORMAT = "{curriculum_name}-object_learner:{object_learner}-attribute_learner:{attribute_learner}-relation_learner:{relation_learner}"
CURRICULUM_NAME_FORMAT = (
    "noise@{noise}-shuffled@{shuffled}-attributes@{attributes}-relations@{relations}"
)

CURRICULUM_PARAMS = {
    "block_multiple_of_same_type": True,
    "include_targets_in_noise": False,
    "min_noise_objects": 1,
    "max_noise_objects": 10,
    "min_noise_relations": 0,
    "max_noise_relations": 5,
    "random_seed": 0,
    "chooser_seed": 0,
}

FIXED_PARAMETERS = {
    "curriculum": "m18-integrated-learners-experiment",
    "learner": "integrated-learner-params",
    "action_learner": {"learner_type": "none"},
    "plural_learner": {"learner_type": "none"},
    "include_functional_learner": False,
    "include_generics_learner": False,
}


if __name__ == "__main__":
    parameters_only_entry_point(integrated_experiment_entry_point)
