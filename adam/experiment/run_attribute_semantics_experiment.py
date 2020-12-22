import random
from pathlib import Path

from adam.curriculum.phase1_curriculum import (
    _make_each_object_by_itself_curriculum,
    _make_objects_with_colors_curriculum,
    _make_colour_predicates_curriculum,
)
from adam.language.language_utils import phase1_language_generator
from adam.learner import LearningExample
from adam.learner.attributes import SubsetAttributeLearnerNew
from adam.learner.generics import SimpleGenericsLearner
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.objects import SubsetObjectLearnerNew
from adam.learner.plurals import SubsetPluralLearnerNew
from adam.learner.semantics_utils import get_concept_node_from_graph
from adam.learner.verbs import SubsetVerbLearnerNew
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.ontology.phase2_ontology import GAILA_PHASE_2_ONTOLOGY
from adam.semantics import AttributeConcept


def integrated_learner_factory(language_mode: LanguageMode):
    rng = random.Random()
    rng.seed(0)
    return IntegratedTemplateLearner(
        # object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
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
    english_color_dictionary = {
        "watermelon": "green",
        "cookie": "light brown",
        "paper": "white",
    }

    # Teach pretraining curriculum
    for curriculum in curricula:
        print("\nTeaching", curriculum.name())
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in curriculum.instances():
            # Get the object matches first - preposition learner can't learn without already recognized objects
            print("Observation: ", " ".join(linguistic_description.as_token_sequence()))
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

    # Evaluate assocations before generics
    print("\nColor assocations - Before Generics")
    for word, _ in english_color_dictionary.items():
        word_concept = get_concept_node_from_graph(word, learner.semantics_graph)
        if not word_concept:
            continue
        results = [
            (
                color_concept.debug_string,
                learner.semantics_graph[word_concept][color_concept]["weight"],
            )
            for color_concept in learner.semantics_graph.neighbors(word_concept)
            if isinstance(color_concept, AttributeConcept)
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        print(f"\nObject:", word)
        print(
            f"Associated Colors:", [(r[0].replace("_slot1", ""), r[1]) for r in results]
        )
        # for r in results:
        #     print(f'{word}, {color}, {r[0].replace("_slot1","")}, {r[1]}')

    # Teach generics
    color_predicates = _make_colour_predicates_curriculum(None, None, language_generator)

    print("\nTeaching color predicates")
    for (
        _,
        linguistic_description,
        perceptual_representation,
    ) in color_predicates.instances():
        # Get the object matches first - preposition learner can't learn without already recognized objects
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )
        print("Observation:", " ".join(linguistic_description.as_token_sequence()))

    # Evaluate assocations after generics
    print("\nColor assocations - After Generics")
    for word, _ in english_color_dictionary.items():
        word_concept = get_concept_node_from_graph(word, learner.semantics_graph)
        if not word_concept:
            continue
        results = [
            (
                color_concept.debug_string,
                learner.semantics_graph[word_concept][color_concept]["weight"],
            )
            for color_concept in learner.semantics_graph.neighbors(word_concept)
            if isinstance(color_concept, AttributeConcept)
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        print(f"\nObject:", word)
        print(
            f"Associated Colors:", [(r[0].replace("_slot1", ""), r[1]) for r in results]
        )

    learner.log_hypotheses(Path(f"./renders/{experiment_id}"))
    learner.render_semantics_to_file(
        graph=learner.semantics_graph,
        graph_name="semantics",
        output_file=Path(f"./renders/{experiment_id}/semantics.png"),
    )


if __name__ == "__main__":
    for lm in [LanguageMode.ENGLISH]:
        language_generator = phase1_language_generator(lm)
        for num_samples in [100]:
            pretraining_curriculas = {
                "objects-and-colors": [
                    _make_each_object_by_itself_curriculum(
                        num_samples, 0, language_generator
                    ),
                    _make_objects_with_colors_curriculum(
                        num_samples, None, language_generator
                    ),
                ]
            }

            for curricula_name, pretraining_curricula in pretraining_curriculas.items():
                # Run experiment
                experiment = f"attribute_semantics_ns-{num_samples}_cur-{curricula_name}"
                print("Running Attribute Semantics Experiment:", experiment)
                integrated_learner = integrated_learner_factory(lm)
                run_experiment(
                    learner=integrated_learner,
                    curricula=pretraining_curricula,
                    experiment_id=experiment,
                )
