import logging

from vistautils.parameters import YAMLParametersLoader, Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.experiment.log_experiment import log_experiment_entry_point

CONFIGURATION_STRING = """
experiment: {experiment}
curriculum: {curriculum}
learner: {learner}
"""


def main(params: Parameters):
    for experiment, curriculum, learner in [
        (
            "generics-with-learner",
            "actions-and-generics-curriculum",
            "integrated-learner-recognizer",
        ),
        (
            "generics-without-learner",
            "actions-and-generics-curriculum",
            "integrated-learner-recognizer-without-generics",
        ),
    ]:
        logging.info("Running %s", experiment)
        setup_specifications = YAMLParametersLoader().load_string(CONFIGURATION_STRING.format(
            experiment=experiment, learner=learner, curriculum=curriculum
        ))
        experiment_params = params.unify(setup_specifications)
        print('Configuration specifications: \n', experiment_params)
        log_experiment_entry_point(experiment_params)


if __name__ == "__main__":
    parameters_only_entry_point(main)
