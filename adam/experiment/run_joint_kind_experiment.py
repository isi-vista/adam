import random
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from immutablecollections import immutableset

from adam.curriculum.curriculum_utils import PHASE1_CHOOSER_FACTORY
from adam.curriculum.phase1_curriculum import (
    _make_kind_predicates_curriculum,
    _make_each_object_by_itself_curriculum,
    _make_generic_statements_curriculum,
    _make_eat_curriculum,
    _make_drink_curriculum,
    _make_sit_curriculum,
    _make_jump_curriculum,
    _make_fly_curriculum,
    _make_objects_with_colors_curriculum,
    _make_colour_predicates_curriculum,
)
from adam.language import TokenSequenceLinguisticDescription
from adam.language.language_utils import phase2_language_generator
from adam.learner import LearningExample
from adam.learner.attributes import SubsetAttributeLearnerNew
from adam.learner.generics import SimpleGenericsLearner
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.objects import SubsetObjectLearnerNew
from adam.learner.plurals import SubsetPluralLearnerNew
from adam.learner.semantics_utils import SemanticsManager, cos_sim
from adam.learner.verbs import SubsetVerbLearnerNew
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
    GROUND,
    CHICKEN,
    BEEF,
    COW,
)
from adam.ontology.phase2_ontology import GAILA_PHASE_2_ONTOLOGY
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    GAILA_PHASE_2_PERCEPTION_GENERATOR,
)
from adam.semantics import Concept
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation


def integrated_learner_factory(language_mode: LanguageMode):
    rng = random.Random()
    rng.seed(0)
    return IntegratedTemplateLearner(
        object_learner=SubsetObjectLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        attribute_learner=SubsetAttributeLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        plural_learner=SubsetPluralLearnerNew(
            ontology=GAILA_PHASE_2_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        action_learner=SubsetVerbLearnerNew(
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
        ),
        generics_learner=SimpleGenericsLearner(),
    )


def run_experiment(learner, curricula, experiment_id):
    # Teach each pretraining curriculum
    for curriculum in curricula:
        print("Teaching", curriculum.name())
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in curriculum.instances():
            # Get the object matches first - prepositison learner can't learn without already recognized objects
            # print(' '.join(linguistic_description.as_token_sequence()))
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
        empty_situation, PHASE1_CHOOSER_FACTORY()
    )
    pseudoword_to_kind = {"wug": "animal", "vonk": "food", "snarp": "people"}
    for word, kind in pseudoword_to_kind.items():
        print(word, "s", "are", kind, "s")
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
    print("Results for ", experiment_id)
    for word, _ in pseudoword_to_kind.items():
        results = [
            (kind, semantics_manager.evaluate_kind_membership(word, kind))
            for kind in pseudoword_to_kind.values()
        ]
        complete_results.append(results)

    results_df = pd.DataFrame(
        [[np.asscalar(i[1]) for i in l] for l in complete_results],
        columns=["Animal", "Food", "People"],
    )
    results_df.insert(0, "Words", pseudoword_to_kind.keys())
    print(results_df.to_csv(index=False))
    learner.log_hypotheses(Path(f"./renders/{experiment_id}"))
    learner.render_semantics_to_file(
        graph=learner.semantics_graph,
        graph_name="semantics",
        output_file=Path(f"./renders/{experiment_id}/semantics.png"),
    )


if __name__ == "__main__":
    for lm in [LanguageMode.ENGLISH]:
        language_generator = phase2_language_generator(lm)
        num_samples = 200
        ban_all = [CHICKEN, BEEF, COW]
        condition_and_banned_objects = {
            "without-chicken-beef-cow": [CHICKEN, BEEF, COW],
            "chicken": [BEEF, COW],
            "beef-cow": [CHICKEN],
            "chicken-beef-cow": immutableset(),
        }
        for condition, banned_objects in condition_and_banned_objects.items():
            pretraining_curricula = [
                _make_each_object_by_itself_curriculum(
                    num_samples,
                    0,
                    language_generator,
                    banned_ontology_types=banned_objects,
                ),
                # Actions - verbs in generics
                _make_eat_curriculum(
                    10, 0, language_generator, banned_ontology_types=banned_objects
                ),
                _make_drink_curriculum(
                    10, 0, language_generator, banned_ontology_types=banned_objects
                ),
                _make_sit_curriculum(
                    10, 0, language_generator, banned_ontology_types=banned_objects
                ),
                _make_jump_curriculum(
                    10, 0, language_generator, banned_ontology_types=banned_objects
                ),
                _make_fly_curriculum(
                    10, 0, language_generator, banned_ontology_types=banned_objects
                ),
                # Color attributes
                _make_objects_with_colors_curriculum(
                    None, None, language_generator, banned_ontology_types=banned_objects
                ),
                # Predicates
                _make_colour_predicates_curriculum(
                    None, None, language_generator, banned_ontology_types=banned_objects
                ),
                _make_kind_predicates_curriculum(
                    None, None, language_generator, banned_ontology_types=banned_objects
                ),
                # Generics
                _make_generic_statements_curriculum(
                    num_samples=3,
                    noise_objects=0,
                    language_generator=language_generator,
                    banned_ontology_types=banned_objects,
                ),
            ]

            # Run experiment
            experiment = (
                f"joint_kind_semantics_ns-{num_samples}_cur-{condition}"
            )
            print("\nRunning experiment:", experiment)
            integrated_learner = integrated_learner_factory(lm)
            run_experiment(
                learner=integrated_learner,
                curricula=pretraining_curricula,
                experiment_id=experiment,
            )
