import random
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

from adam.curriculum.phase1_curriculum import (
    _make_each_object_by_itself_curriculum,
    _make_objects_with_colors_curriculum,
    _make_colour_predicates_curriculum)
from adam.language.language_utils import phase1_language_generator
from adam.learner import LearningExample
from adam.learner.attributes import SubsetAttributeLearnerNew
from adam.learner.generics import SimpleGenericsLearner
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.learner_utils import (
    cos_sim,
    get_concept_node_from_graph)
from adam.learner.objects import SubsetObjectLearnerNew
from adam.learner.plurals import SubsetPluralLearnerNew
from adam.learner.verbs import SubsetVerbLearnerNew
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.semantics import Concept, AttributeConcept
from tests.learner import LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER


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
            ontology=GAILA_PHASE_1_ONTOLOGY, beam_size=5, language_mode=language_mode
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
        for (
            _,
            linguistic_description,
            perceptual_representation,
        ) in curriculum.instances():
            # Get the object matches first - preposition learner can't learn without already recognized objects
            learner.observe(
                LearningExample(perceptual_representation, linguistic_description)
            )

    print(learner.semantics_graph.nodes)

    # Evaluate assocations before generics
    for word, color in english_color_dictionary.items():
        print(word, color)
        word_concept = get_concept_node_from_graph(word, learner.semantics_graph)
        if not word_concept: continue
        results = [
            (color_concept.debug_string, learner.semantics_graph[word_concept][color_concept]['weight'])
            for color_concept in learner.semantics_graph.neighbors(word_concept) if isinstance(color_concept, AttributeConcept)
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        print(results)

    # Teach generics
    color_predicates = _make_colour_predicates_curriculum(None, None, language_generator)
    for (
            _,
            linguistic_description,
            perceptual_representation,
    ) in color_predicates.instances():
        # Get the object matches first - preposition learner can't learn without already recognized objects
        learner.observe(
            LearningExample(perceptual_representation, linguistic_description)
        )
        print(' '.join(linguistic_description.as_token_sequence()))

    # Evaluate assocations after generics
    for word, color in english_color_dictionary.items():
        print(word, color)
        word_concept = get_concept_node_from_graph(word, learner.semantics_graph)
        if not word_concept: continue
        results = [
            (color_concept.debug_string, learner.semantics_graph[word_concept][color_concept]['weight'])
            for color_concept in learner.semantics_graph.neighbors(word_concept) if
            isinstance(color_concept, AttributeConcept)
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        print(results)

    #
    #
    #     embeddings = semantics_as_weighted_adjacency_matrix(learner.semantics_graph)
    #     objects_to_embeddings = {
    #         n: embeddings[i]
    #         for i, n in enumerate(learner.semantics_graph.nodes)
    #         if isinstance(n, ObjectConcept)
    #     }
    #     generate_heatmap(objects_to_embeddings, experiment_id)

    learner.log_hypotheses(Path(f"./renders/{experiment_id}"))
    # generate_similarities(semantic_matrix, list(learner.semantics_graph.nodes()), ObjectConcept)


def generate_heatmap(nodes_to_embeddings: Dict[Concept, Any], filename: str):
    if not nodes_to_embeddings:
        return
    similarity_matrix = np.zeros((len(nodes_to_embeddings), len(nodes_to_embeddings)))
    for i, (_, embedding_1) in enumerate(nodes_to_embeddings.items()):
        for j, (_, embedding_2) in enumerate(nodes_to_embeddings.items()):
            similarity_matrix[i][j] = cos_sim(embedding_1, embedding_2)
    names = [n.debug_string for n in nodes_to_embeddings.keys()]
    df = pd.DataFrame(data=similarity_matrix, index=names, columns=names)
    plt.rcParams["figure.figsize"] = (20.0, 20.0)
    plt.rcParams["font.family"] = "serif"
    # sb.heatmap(df)
    # sb.clustermap(df)
    sb.clustermap(df, row_cluster=True, col_cluster=True)
    # cm.ax_row_dendrogram.set_visible(False)
    # cm.ax_col_dendrogram.set_visible(False)
    # plt.show()
    plt.savefig(f"plots/{filename}.png")
    plt.close()


if __name__ == "__main__":
    for lm in [LanguageMode.ENGLISH]:
        language_generator = phase1_language_generator(lm)
        for num_samples in [100]:
            pretraining_curriculas = {
                "objects-and-colors": [
                    _make_each_object_by_itself_curriculum(
                        num_samples, 0, language_generator
                    ),
                    _make_objects_with_colors_curriculum(num_samples, None, language_generator),
                ],
                # "obj-actions-kinds-generics": [
                #     _make_each_object_by_itself_curriculum(
                #         num_samples, 0, language_generator
                #     ),
                #     # Actions - verbs in generics
                #     _make_eat_curriculum(num_samples, 0, language_generator),
                #     _make_drink_curriculum(num_samples, 0, language_generator),
                #     _make_sit_curriculum(num_samples, 0, language_generator),
                #     _make_jump_curriculum(num_samples, 0, language_generator),
                #     _make_fly_curriculum(num_samples, 0, language_generator),
                #     # Plurals
                #     _make_plural_objects_curriculum(num_samples, 0, language_generator),
                #     # plurals,
                #     # Color attributes
                #     _make_objects_with_colors_curriculum(None, None, language_generator),
                #     # Predicates
                #     _make_colour_predicates_curriculum(None, None, language_generator),
                #     _make_kind_predicates_curriculum(None, None, language_generator),
                #     # Generics
                #     _make_generic_statements_curriculum(
                #         num_samples=3,
                #         noise_objects=0,
                #         language_generator=language_generator,
                #     ),
                # ],
                # build_gaila_m13_curriculum(num_samples=num_samples, num_noise_objects=0, language_generator=language_generator)
            }

            for curricula_name, pretraining_curricula in pretraining_curriculas.items():
                # Run experiment
                experiment = f"attribute_semantics_lang-{lm}_num-samples-{num_samples}_cur-{curricula_name}"
                print("Running experiment:", experiment)
                integrated_learner = integrated_learner_factory(lm)
                run_experiment(
                    learner=integrated_learner,
                    curricula=pretraining_curricula,
                    experiment_id=experiment,
                )
