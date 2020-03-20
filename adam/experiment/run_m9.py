import logging

from vistautils.parameters import Parameters, YAMLParametersLoader
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.experiment.log_experiment import log_experiment_entry_point


def main(params: Parameters):
    adam_root = params.existing_directory("adam_root")
    m9_experiments_dir = adam_root / "parameters" / "experiments" / "m9"
    param_files = [
        m9_experiments_dir / "objects.params",
        m9_experiments_dir / "attributes.params",
        m9_experiments_dir / "relations.params",
        m9_experiments_dir / "events.params",
    ]

    # If any of the param files don't exist, bail out earlier instead of making the user
    # wait for the error.
    for param_file in param_files:
        if not param_file.exists():
            raise RuntimeError(f"Expected param file {param_file} does not exist")

    for param_file in param_files:
        logging.info("Running %s", param_file)
        experiment_params = YAMLParametersLoader().load(param_file)
        log_experiment_entry_point(experiment_params)


if __name__ == "__main__":
    parameters_only_entry_point(main)
