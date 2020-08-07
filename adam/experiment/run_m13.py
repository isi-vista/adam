import logging
from pathlib import Path
from typing import List

from vistautils.parameters import Parameters, YAMLParametersLoader
from vistautils.parameters_only_entrypoint import parameters_only_entry_point
from pegasus_wrapper import (
    initialize_vista_pegasus_wrapper,
    run_python_on_parameters,
    Locator,
    write_workflow_description,
)

import adam.experiment.log_experiment as log_experiment_script
from adam.experiment.log_experiment import log_experiment_entry_point


def main(params: Parameters):
    adam_root = params.existing_directory("adam_root")
    m13_experiments_dir = adam_root / "parameters" / "experiments" / "m13"
    use_pegasus = params.boolean("use_pegasus", default=False)
    if use_pegasus:
        initialize_vista_pegasus_wrapper(params)

    param_files: List[Path] = []

    if params.boolean("include_objects", default=True):
        param_files.append(m13_experiments_dir / "objects.params")

    if params.boolean("include_imprecise_size", default=True):
        param_files.append(m13_experiments_dir / "imprecise_size.params")

    if params.boolean("include_imprecise_temporal", default=True):
        param_files.append(m13_experiments_dir / "imprecise_temporal.params")

    if params.boolean("include_subtle_verb", default=True):
        param_files.append(m13_experiments_dir / "subtle_verb.params")

    if params.boolean("include_object_restrictions", default=True):
        param_files.append(m13_experiments_dir / "object_restrictions.params")

    if params.boolean("include_functionally_defined_objects", default=True):
        param_files.append(m13_experiments_dir / "functionally_defined_objects.params")

    if params.boolean("include_relations", default=True):
        param_files.append(m13_experiments_dir / "relations.params")

    if params.boolean("include_generics", default=True):
        param_files.append(m13_experiments_dir / "generics.params")

    if params.boolean("include_verbs_with_dynamic_prepositions", default=True):
        param_files.append(
            m13_experiments_dir / "events_with_dynamic_prepositions.params"
        )

    if params.boolean("include_m9_complete", default=False):
        param_files.append(m13_experiments_dir / "m9_complete.params")

    if params.boolean("include_m13_complete", default=False):
        param_files.append(m13_experiments_dir / "m13_complete.params")

    if params.boolean("include_m13_shuffled", default=False):
        param_files.append(m13_experiments_dir / "m13_shuffled.params")

    # This activates a special "debug" curriculum,
    # which is meant to be edited in the code by a developer to do fine-grained debugging.
    if params.boolean("include_debug", default=False):
        param_files.append(m13_experiments_dir / "debug.params")

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
            run_python_on_parameters(
                experiment_name,
                log_experiment_script,
                experiment_params,
                depends_on=[],
            )

    if use_pegasus:
        write_workflow_description()


if __name__ == "__main__":
    parameters_only_entry_point(main)
