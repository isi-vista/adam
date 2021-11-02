import random
from pathlib import Path

import numpy as np
import pandas as pd
from immutablecollections import immutableset

from adam.curriculum.curriculum_utils import CHOOSER_FACTORY
from adam.curriculum.phase1_curriculum import (
    _make_kind_predicates_curriculum,
    _make_each_object_by_itself_curriculum,
    _make_generic_statements_curriculum,
    _make_eat_curriculum,
    _make_drink_curriculum,
    _make_sit_curriculum,
    _make_jump_curriculum,
    _make_fly_curriculum,
    _make_plural_objects_curriculum,
    _make_objects_with_colors_curriculum,
    _make_colour_predicates_curriculum,
)
from adam.language import TokenSequenceLinguisticDescription
from adam.language.language_utils import phase2_language_generator
from adam.learner import LearningExample
from adam.learner.attributes import SubsetAttributeLearner
from adam.learner.generics import SimpleGenericsLearner
from adam.learner.integrated_learner import SymbolicIntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.objects import SubsetObjectLearner
from adam.learner.plurals import SubsetPluralLearner
from adam.learner.semantics_utils import SemanticsManager
from adam.learner.verbs import SubsetVerbLearner
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY, GROUND
from adam.ontology.phase2_ontology import GAILA_PHASE_2_ONTOLOGY
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_2_PERCEPTION_GENERATOR,
)
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from tabulate import tabulate


def integrated_learner_factory(language_mode: LanguageMode):
    rng = random.Random()
    rng.seed(0)
    return SymbolicIntegratedTemplateLearner(
        object_learner=SubsetObjectLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        attribute_learner=SubsetAttributeLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        plural_learner=SubsetPluralLearner(
            ontology=GAILA_PHASE_2_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        action_learner=SubsetVerbLearner(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        generics_learner=SimpleGenericsLearner(),
    )


def run_experiment(learner, curricula, experiment_id):
    # Teach each pretraining curriculum
    for curriculum in curricula:
        print("Teaching", curriculum.name(), "curriculum")
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in curriculum.instances():
            # Get the object matches first - prepositison learner can't learn without already recognized objects
            # print('Observation: ',' '.join(linguistic_description.as_token_sequence()))
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

    # Teach each kind member
    empty_situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_2_ONTOLOGY,
        salient_objects=immutableset(
            [
                SituationObject.instantiate_ontology_node(
                    ontology_node=GROUND,
                    debug_handle=GROUND.handle,
                    ontology=GAILA_PHASE_1_ONTOLOGY,
                )
            ]
        ),
    )
    empty_perception = GAILA_PHASE_2_PERCEPTION_GENERATOR.generate_perception(
        empty_situation, CHOOSER_FACTORY()
    )
    pseudoword_to_kind = {"wug": "animal", "vonk": "food", "snarp": "people"}

    print("Teaching new objects in known categories")
    for word, kind in pseudoword_to_kind.items():
        print("Observation: ", word, "s", "are", kind, "s")
        learner.observe(
            LearningExample(
                empty_perception,
                TokenSequenceLinguisticDescription(tokens=(word, "s", "are", kind, "s"))
                if kind != "people"
                else TokenSequenceLinguisticDescription(
                    tokens=(word, "s", "are", kind, "s")
                ),
            )
        )

    semantics_manager: SemanticsManager = SemanticsManager(
        semantics_graph=learner.semantics_graph
    )
    complete_results = []
    print("\nResults for ", experiment_id)
    for word, _ in pseudoword_to_kind.items():
        results = [
            (kind, semantics_manager.evaluate_kind_membership(word, kind))
            for kind in pseudoword_to_kind.values()
        ]
        complete_results.append(results)

    results_df = pd.DataFrame(
        [[np.asscalar(i[1]) for i in line] for line in complete_results],
        columns=["Animal", "Food", "People"],
    )
    results_df.insert(0, "Words", pseudoword_to_kind.keys())

    # print(results_df.to_csv(index=False))
    print(tabulate(results_df, headers="keys", tablefmt="psql"))

    learner.log_hypotheses(Path(f"./renders/{experiment_id}"))

    learner.render_semantics_to_file(
        graph=learner.semantics_graph,
        graph_name="semantics",
        output_file=Path(f"./renders/{experiment_id}/semantics.png"),
    )


def main():
    for lm in [LanguageMode.ENGLISH]:
        language_generator = phase2_language_generator(lm)
        num_samples = 200
        pretraining_curricula = {
            "objects-and-kinds": [
                _make_each_object_by_itself_curriculum(
                    num_samples, 0, language_generator
                ),
                _make_kind_predicates_curriculum(None, None, language_generator),
            ],
            "kinds-and-generics": [
                _make_each_object_by_itself_curriculum(
                    num_samples, 0, language_generator
                ),
                _make_kind_predicates_curriculum(None, None, language_generator),
                _make_generic_statements_curriculum(
                    num_samples=3, noise_objects=0, language_generator=language_generator
                ),
            ],
            "obj-actions-kinds-generics": [
                _make_each_object_by_itself_curriculum(
                    num_samples, 0, language_generator
                ),
                # Actions - verbs in generics
                _make_eat_curriculum(10, 0, language_generator),
                _make_drink_curriculum(10, 0, language_generator),
                _make_sit_curriculum(10, 0, language_generator),
                _make_jump_curriculum(10, 0, language_generator),
                _make_fly_curriculum(10, 0, language_generator),
                # Plurals
                _make_plural_objects_curriculum(None, 0, language_generator),
                # Color attributes
                _make_objects_with_colors_curriculum(None, None, language_generator),
                # Predicates
                _make_colour_predicates_curriculum(None, None, language_generator),
                _make_kind_predicates_curriculum(None, None, language_generator),
                # Generics
                _make_generic_statements_curriculum(
                    num_samples=3, noise_objects=0, language_generator=language_generator
                ),
            ],
        }

        for curricula_name, pretraining_curriculum in pretraining_curricula.items():
            # Run experiment
            experiment = f"kind_semantics_ns-{num_samples}_cur-{curricula_name}"
            print("\nRunning Category Semantics Experiment:", experiment, "\n")
            integrated_learner = integrated_learner_factory(lm)
            run_experiment(
                learner=integrated_learner,
                curricula=pretraining_curriculum,
                experiment_id=experiment,
            )


if __name__ == "__main__":
    main()
