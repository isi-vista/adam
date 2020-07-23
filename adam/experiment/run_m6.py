import logging

from adam.experiment.log_experiment import log_experiment_entry_point
from vistautils.parameters import Parameters, YAMLParametersLoader
from vistautils.parameters_only_entrypoint import parameters_only_entry_point


def main(params: Parameters):
    adam_root = params.existing_directory("adam_root")
    m6_experiments_dir = adam_root / "parameters" / "experiments" / "m6"
    param_files = [
        m6_experiments_dir / "each-object-by-itself.pursuit.params",
        m6_experiments_dir / "pursuit-single-noise.params",
        m6_experiments_dir / "static-prepositions.params",
        m6_experiments_dir / "pursuit-double-noise.params",
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
