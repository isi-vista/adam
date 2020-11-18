from adam.experiment.curriculum_repository import _build_curriculum_path
from adam.learner import LanguageMode

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
        "pursuit", params.integer("num_pursuit_learners_active", default=8)
    )

    language_mode = params.enum(
        "language_mode", LanguageMode, default=LanguageMode.ENGLISH
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
            _build_curriculum_path(
                curriculum_repository_path,
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
                ).unify(FIXED_PARAMETERS),
                language_mode,
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
    for (
        curriculum_str,
        _curriculum_path,
        curriculum_dep,
        curr_params,
    ) in curriculum_dependencies:
        object_learner_type = params.string(
            "object_learner_type", valid_options=LEARNER_TO_PARAMS.keys()
        )
        attribute_learner_type = params.string(
            "attribute_learner_type", valid_options=LEARNER_TO_PARAMS.keys()
        )
        relation_learner_type = params.string(
            "relation_learner_type", valid_options=LEARNER_TO_PARAMS.keys()
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
                "object_learner": LEARNER_TO_PARAMS[object_learner_type],
                "attribute_learner": LEARNER_TO_PARAMS[attribute_learner_type],
                "relation_learner": LEARNER_TO_PARAMS[relation_learner_type],
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
                PURSUIT_RESOURCE_REQUEST
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

LEARNER_TO_PARAMS = {
    "subset": {"learner_type": "subset", "ontology": "integrated_experiment"},
    "pursuit": {
        "learner_type": "pursuit",
        "ontology": "integrated_experiment",
        "random_seed": 0,
        "learning_factor": 0.02,
        "graph_match_confirmation_threshold": 0.9,
        "lexicon_entry_threshold": 0.7,
        "smoothing_parameter": 0.001,
    },
    "none": {"learner_type": "none"},
}

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
    "post_observer": {
        "include_acc_observer": False,
        "include_pr_observer": True,
        "log_pr": True,
    },
    "test_observer": {"accuracy_to_txt": True},
    "action_learner": {"learner_type": "none"},
    "plural_learner": {"learner_type": "none"},
    "include_functional_learner": False,
    "include_generics_learner": False,
}

PURSUIT_RESOURCE_REQUEST = Parameters.from_mapping(
    {
        "exclude_list": f"saga01,saga02,saga03,saga04,saga05,saga06,saga07,saga08,saga10,saga11,saga12,saga13,"
        f"saga14,saga15,saga16,saga17,saga18,saga19,saga20,saga21,saga22,saga23,saga24,saga25,saga26,"
        f"gaia01,gaia02",
        "partition": "ephemeral",
    }
)


if __name__ == "__main__":
    parameters_only_entry_point(integrated_experiment_entry_point)
