import logging
from pathlib import Path
from typing import List

from vistautils.parameters import Parameters, YAMLParametersLoader
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.experiment.log_experiment import log_experiment_entry_point


def main(params: Parameters):
    adam_root = params.existing_directory("adam_root")
    experiments_dir = adam_root / "parameters" / "experiments"
    param_file = experiments_dir / "object_restrictions.params"

    if not param_file.exists():
        raise RuntimeError(f"Expected param file {param_file} does not exist")

    logging.info("Running %s", param_file)
    experiment_params = YAMLParametersLoader().load(param_file)
    log_experiment_entry_point(experiment_params)


if __name__ == "__main__":
    parameters_only_entry_point(main)
