from typing import Optional, Any

import numpy as np
from more_itertools import first
from networkx import Graph, to_numpy_matrix

from adam.semantics import Concept, KindConcept


def semantics_as_weighted_adjacency_matrix(semantics_graph: Graph) -> Any:
    return to_numpy_matrix(semantics_graph)


def concept_embedding(concept: Concept, graph: Graph) -> Any:
    # Get a numpy array weighted adjacency embedding of the concept from the graph
    semantics_matrix = semantics_as_weighted_adjacency_matrix(graph)
    nodes = list(graph.nodes)
    return semantics_matrix[nodes.index(concept)]


def kind_embedding(concept: KindConcept, graph: Graph) -> Any:
    # Get a numpy array weighted adjacency embedding averaging the members of a kind concept in the graph
    member_embeddings = np.vstack(
        [concept_embedding(member, graph) for member in graph.neighbors(concept)]
    )
    return np.mean(member_embeddings, axis=0)


def get_concept_node_from_graph(
    identifier: str, semantics_graph: Graph
) -> Optional[Concept]:
    return first([n for n in semantics_graph.nodes if n.debug_string == identifier], None)


def cos_sim(a, b) -> float:
    dot = np.dot(a.reshape(1, -1), b.reshape(-1, 1))
    norma = np.linalg.norm(a.reshape(1, -1))
    normb = np.linalg.norm(b.reshape(1, -1))
    return dot / (norma * normb)


def evaluate_kind_membership(semantics: Graph, word: str, kind: str) -> float:
    semantics_graph = semantics.to_undirected()
    kind_concept = get_concept_node_from_graph(kind, semantics_graph)
    word_concept = get_concept_node_from_graph(word, semantics_graph)
    if not kind_concept or not word_concept or not isinstance(kind_concept, KindConcept):
        return 0
    return cos_sim(
        concept_embedding(word_concept, semantics_graph),
        kind_embedding(kind_concept, semantics_graph),
    )
