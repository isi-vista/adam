from collections import defaultdict
from typing import Optional, Any

import numpy as np
from more_itertools import first
from networkx import Graph, to_numpy_matrix

from adam.semantics import Concept, KindConcept, ObjectConcept, ActionConcept


class SemanticsManager:

    def __init__(self, semantics_graph: Graph):
        self.semantics_graph = Graph()
        for u, v, data, in semantics_graph.edges(data=True):
            syntactic_position = data['slot']
            new_u = self.concept_as_string(u, syntactic_position) if isinstance(u, ActionConcept) else self.concept_as_string(u)
            new_v = self.concept_as_string(v, syntactic_position) if isinstance(v, ActionConcept) else self.concept_as_string(v)
            self.semantics_graph.add_edge(new_u, new_v, weight=data['weight'])

        self.nodes = list(self.semantics_graph.nodes)
        self.semantics_matrix = to_numpy_matrix(self.semantics_graph)

    def object_concept_embedding(self, concept: str) -> Any:
        # Get a numpy array weighted adjacency embedding of the concept from the graph
        return self.semantics_matrix[self.nodes.index(concept)]

    def kind_concept_embedding(self, concept: str) -> Any:
        # Get a numpy array weighted adjacency embedding averaging the members of a kind concept in the graph
        member_embeddings = np.vstack(
            [self.object_concept_embedding(member) for member in self.semantics_graph.neighbors(concept)]
        )
        return np.mean(member_embeddings, axis=0)

    def get_concept_node_with_id(self, identifier: str) -> Optional[Concept]:
        return first([n for n in self.nodes if identifier == n], None)

    def evaluate_kind_membership(self, word: str, kind: str) -> float:
        word_node = self.concept_as_string(ObjectConcept(word))
        kind_node = self.concept_as_string(KindConcept(kind))
        print(word_node, kind_node)
        if kind_node not in self.nodes or word_node not in self.nodes:
            return 0
        return cos_sim(
            self.object_concept_embedding(word_node),
            self.kind_concept_embedding(kind_node)
        )

    @staticmethod
    def concept_as_string(concept: Concept, syntactic_position='') -> str:
        if syntactic_position:
            return f'{concept.debug_string}_{str(type(concept))}_{syntactic_position}'
        else:
            return f'{concept.debug_string}_{str(type(concept))}'


def cos_sim(a, b) -> float:
    dot = np.dot(a.reshape(1, -1), b.reshape(-1, 1))
    norma = np.linalg.norm(a.reshape(1, -1))
    normb = np.linalg.norm(b.reshape(1, -1))
    return dot / (norma * normb)
