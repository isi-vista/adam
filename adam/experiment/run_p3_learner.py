import logging
from pathlib import Path
from typing import List

from pegasus_wrapper import (
    initialize_vista_pegasus_wrapper,
    directory_for,
    run_python_on_parameters,
    Locator,
    write_workflow_description,
)

from vistautils.parameters import Parameters, YAMLParametersLoader
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

import adam.experiment.log_experiment as log_experiment_script
from adam.experiment.log_experiment import log_experiment_entry_point


def main(params: Parameters):
    adam_root = params.existing_directory("adam_root")
    p3_experiments_dir = adam_root / "parameters" / "experiments" / "p3"
    use_pegasus = params.boolean("use_pegasus", default=False)
    if use_pegasus:
        initialize_vista_pegasus_wrapper(params)

    param_files: List[Path] = []

    # This activates a special "debug" curriculum,
    # which is meant to be edited in the code by a developer to do fine-grained debugging.
    if params.boolean("include_debug", default=False):
        param_files.append(p3_experiments_dir / "debug.params")

    # If any of the param files don't exist, bail out earlier instead of making the user
    # wait for the error.
    for param_file in param_files:
        if not param_file.exists():
            raise RuntimeError(f"Expected param file {param_file} does not exist")

    for param_file in param_files:
        logging.info("Running %s", param_file)
        experiment_params = YAMLParametersLoader().load(param_file)
        if not use_pegasus:
            log_experiment_entry_point(experiment_params)
        else:
            experiment_name = Locator(experiment_params.string("experiment"))
            experiment_params = experiment_params.unify(
                {
                    "experiment_group_dir": directory_for(experiment_name) / "output",
                    "hypothesis_log_dir": directory_for(experiment_name) / "hypotheses",
                    # State pickles will go under experiment_name/learner_state
                    "learner_logging_path": directory_for(experiment_name),
                    "log_learner_state": True,
                    "resume_from_latest_logged_state": True,
                    "log_hypothesis_every_n_steps": params.integer(
                        "save_state_every_n_steps"
                    ),
                    "debug_learner_pickling": params.boolean(
                        "debug_learner_pickling", default=False
                    ),
                }
            )

            run_python_on_parameters(
                experiment_name, log_experiment_script, experiment_params, depends_on=[]
            )

    if use_pegasus:
        write_workflow_description()


if __name__ == "__main__":
    parameters_only_entry_point(main)
