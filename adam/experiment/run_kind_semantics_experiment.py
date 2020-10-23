import random
from typing import List, Optional, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb
from more_itertools import first
from networkx import Graph
from sklearn.metrics.pairwise import cosine_similarity

from adam.curriculum.phase1_curriculum import (
    _make_kind_predicates_curriculum,
    _make_each_object_by_itself_curriculum, _make_generic_statements_curriculum, _make_eat_curriculum,
    _make_drink_curriculum, _make_sit_curriculum, _make_jump_curriculum, _make_fly_curriculum,
    _make_plural_objects_curriculum, _make_objects_with_colors_curriculum, _make_colour_predicates_curriculum)
from adam.language import TokenSequenceLinguisticDescription
from adam.language.language_utils import (
    phase2_language_generator,
)
from adam.learner import LearningExample
from adam.learner.attributes import SubsetAttributeLearnerNew
from adam.learner.generics import SimpleGenericsLearner
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.learner.learner_utils import semantics_as_weighted_adjacency_matrix
from adam.learner.plurals import SubsetPluralLearnerNew
from adam.learner.verbs import SubsetVerbLearnerNew
from adam.ontology.phase1_ontology import (
    GAILA_PHASE_1_ONTOLOGY,
)
from adam.semantics import Concept, KindConcept
from tests.learner import LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER


def integrated_learner_factory(language_mode: LanguageMode):
    rng = random.Random()
    rng.seed(0)
    return IntegratedTemplateLearner(
        object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
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


def run_experiment(learner, curricula):
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

    pseudoword_to_kind = {'wug':'animal', 'vonk': 'food', 'snarp': 'people'}
    for word, kind in pseudoword_to_kind.items():
        learner.observe(
            LearningExample(
                perceptual_representation,
                TokenSequenceLinguisticDescription(tokens=(word, "s", "are", kind, "s")) if kind != 'people' \
                    else TokenSequenceLinguisticDescription(tokens=(word, "s", "are", kind, "s")),
            )
        )

    for word, gold_kind in pseudoword_to_kind.items():
        print(word, gold_kind)
        results = [(kind, evaluate_kind_membership(learner.semantics_graph, word, kind)) for kind in pseudoword_to_kind.values()]
        results.sort(key=lambda x: x[1], reverse=True)
        print(results)

    # learner.log_hypotheses(Path(f"./renders/{language_mode.name}"))
    # generate_similarities(semantic_matrix, list(learner.semantics_graph.nodes()), ObjectConcept)


def concept_embedding(concept: Concept, graph: Graph) -> Any:
    # Get a numpy array weighted adjacency embedding of the concept from the graph
    semantics_matrix = semantics_as_weighted_adjacency_matrix(graph)
    nodes = list(graph.nodes)
    return semantics_matrix[nodes.index(concept)]


def kind_embedding(concept: KindConcept, graph: Graph) -> Any:
    # Get a numpy array weighted adjacency embedding averaging the members of a kind concept in the graph
    member_embeddings = np.vstack([concept_embedding(member, graph) for member in graph.neighbors(concept)])
    return np.mean(member_embeddings, axis=0)


def get_concept_node_from_graph(identifier: str, semantics_graph: Graph) -> Optional[Concept]:
    return first([n for n in semantics_graph.nodes if n.debug_string == identifier], None)


def cos_sim(a, b) -> float:
    return cosine_similarity(
        a.reshape(1, -1), b.reshape(1, -1)
    )

def evaluate_kind_membership(semantics: Graph, word: str, kind: str) -> float:
    semantics_graph = semantics.to_undirected()
    kind_concept = get_concept_node_from_graph(kind, semantics_graph)
    word_concept = get_concept_node_from_graph(word, semantics_graph)
    if not kind_concept or not word_concept: return 0
    return cos_sim(concept_embedding(word_concept, semantics_graph), kind_embedding(kind_concept, semantics_graph))


def generate_similarities(embedding_matrix, nodes: List[Concept], type):
    relevant_nodes = [n for n in nodes if isinstance(n, type)]
    similarity_matrix = np.zeros((len(relevant_nodes), len(relevant_nodes)))
    for i, node in enumerate(relevant_nodes):
        e_i = nodes.index(node)
        for j, node_2 in enumerate(relevant_nodes):
            e_j = nodes.index(node_2)
            similarity_matrix[i][j] = cos_sim(embedding_matrix[e_i], embedding_matrix[e_j])
    names = [n.debug_string for n in relevant_nodes]
    df = pd.DataFrame(data=similarity_matrix, index=names, columns=names)
    fig, ax = plt.subplots(figsize=(20, 20))
    # sb.heatmap(df)
    # sb.clustermap(df)
    cm = sb.clustermap(df, row_cluster=True, col_cluster=True)
    cm.ax_row_dendrogram.set_visible(False)
    cm.ax_col_dendrogram.set_visible(False)
    plt.show()

    # Most similar
    # ranking = [(node_2.debug_string, cosine_similarity(a, matrix[j])) for j, node_2 in
    #            enumerate(nodes) if isinstance(node_2, Concept)]
    # ranking.sort(key=lambda x: x[1], reverse=True)
    # print(ranking[:3])


if __name__ == "__main__":
    language_mode = LanguageMode.ENGLISH
    learner = integrated_learner_factory(language_mode)
    language_generator = phase2_language_generator(language_mode)
    for num_samples in [1,3,5]:
        pretraining_curriculas = {
            'just-objects':[
                _make_each_object_by_itself_curriculum(num_samples, 0, language_generator),
                # _make_kind_predicates_curriculum(None, None, language_generator),
             ],
            'objects-and-kinds':[
                _make_each_object_by_itself_curriculum(num_samples, 0, language_generator),
                _make_kind_predicates_curriculum(None, None, language_generator),
             ],
             'kinds-and-generics':[
                 _make_each_object_by_itself_curriculum(num_samples, 0, language_generator),
                 _make_kind_predicates_curriculum(None, None, language_generator),
                 _make_generic_statements_curriculum(
                     num_samples=3, noise_objects=0, language_generator=language_generator
                 ),
             ]
            # [
            #      # Actions - verbs in generics
            #      _make_eat_curriculum(num_samples, 0, language_generator),
            #      _make_drink_curriculum(num_samples, 0, language_generator),
            #      _make_sit_curriculum(num_samples, 0, language_generator),
            #      _make_jump_curriculum(num_samples, 0, language_generator),
            #      _make_fly_curriculum(num_samples, 0, language_generator),
            #      # Plurals
            #      _make_plural_objects_curriculum(num_samples, 0, language_generator),
            #      # plurals,
            #      # Color attributes
            #      _make_objects_with_colors_curriculum(None, None, language_generator),
            #      # Predicates
            #      _make_colour_predicates_curriculum(None, None, language_generator),
            #      _make_kind_predicates_curriculum(None, None, language_generator),
            #      # Generics
            #      _make_generic_statements_curriculum(
            #          num_samples=3, noise_objects=0, language_generator=language_generator
            #      ),
            # ],
        # build_gaila_m13_curriculum(num_samples=num_samples, num_noise_objects=0, language_generator=language_generator)
        }

        for curricula_name, curricula in pretraining_curriculas.items():
            # Run experiment
            print('Number of samples:', num_samples)
            print('Curricula:', curricula_name)
            run_experiment(learner, curricula)
