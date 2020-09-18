import logging

from vistautils.parameters import YAMLParametersLoader, Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.experiment.log_experiment import log_experiment_entry_point


PARAM_STRING = """
adam_root: '/Users/isiboston/ISI/ADAM/adam'
adam_experiment_root: '/Users/isiboston/ISI/ADAM/adam/experiment_logs'
    
experiment_group_dir: '%adam_experiment_root%/generics/'
hypothesis_log_dir: "%experiment_group_dir%/{experiment}-hypotheses"
include_image_links: true

sort_learner_descriptions_by_length: True
num_pretty_descriptions: 5

experiment: {experiment}
curriculum: {curriculum}
learner: {learner}
"""


def main(params: Parameters):
    # pylint: disable=unused-argument
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
        param_string = PARAM_STRING.format(
            experiment=experiment, learner=learner, curriculum=curriculum
        )
        logging.info("Running %s", experiment)
        experiment_params = YAMLParametersLoader().load_string(param_string)
        log_experiment_entry_point(experiment_params)


if __name__ == "__main__":
    parameters_only_entry_point(main)
