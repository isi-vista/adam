import logging

from vistautils.parameters import YAMLParametersLoader, Parameters
from vistautils.parameters_only_entrypoint import parameters_only_entry_point

from adam.experiment.log_experiment import log_experiment_entry_point

CONFIGURATION_STRING = """
experiment: {experiment}
curriculum: {curriculum}
learner: {learner}
accuracy_to_txt : True

pursuit:
   learning_factor: 0.05
   graph_match_confirmation_threshold: 0.7
   lexicon_entry_threshold: 0.7
   smoothing_parameter: .001
"""


def main(params: Parameters):
    for curriculum in ["actions-and-generics-curriculum"]:
        for learner in ["integrated-learner-recognizer","integrated-learner-recognizer-without-generics"]:
            experiment = f'kind-semantics_{curriculum}_{learner}'
            logging.info("Running %s", experiment)
            setup_specifications = YAMLParametersLoader().load_string(CONFIGURATION_STRING.format(
                experiment=experiment, learner=learner, curriculum=curriculum
            ))
            experiment_params = params.unify(setup_specifications)
            print('Configuration specifications: \n', experiment_params)
            log_experiment_entry_point(experiment_params)


if __name__ == "__main__":
    parameters_only_entry_point(main)
