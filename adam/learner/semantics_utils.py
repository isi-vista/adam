from typing import Optional, Any, Dict

import numpy as np
import pandas as pd
from more_itertools import first
from networkx import Graph, to_numpy_matrix
import matplotlib.pyplot as plt
import seaborn as sb

from adam.semantics import Concept, KindConcept, ObjectConcept, ActionConcept


class SemanticsManager:
    def __init__(self, semantics_graph: Graph) -> None:
        self.semantics_graph: Graph = Graph()
        # Create a new type of edge for each edge in the original semantics graph
        # If any of the nodes is an action concept, we want to make a distinct new node to track syntax
        for u, v, data in semantics_graph.edges(data=True):
            syntactic_position = data["slot"]
            new_u = (
                self.concept_as_str_node(u, syntactic_position)
                if isinstance(u, ActionConcept)
                else self.concept_as_str_node(u)
            )
            new_v = (
                self.concept_as_str_node(v, syntactic_position)
                if isinstance(v, ActionConcept)
                else self.concept_as_str_node(v)
            )
            self.semantics_graph.add_edge(new_u, new_v, weight=data["weight"])

        self.nodes = list(self.semantics_graph.nodes)
        self.semantics_matrix = to_numpy_matrix(self.semantics_graph)

    def object_concept_embedding(self, concept: str) -> Any:
        # Get a numpy array weighted adjacency embedding of the concept from the graph
        return self.semantics_matrix[self.nodes.index(concept)]

    def kind_concept_embedding(self, concept: str) -> Any:
        # Get a numpy array weighted adjacency embedding averaging the members of a kind concept in the graph
        member_embeddings = np.vstack(
            [
                self.object_concept_embedding(member)
                for member in self.semantics_graph.neighbors(concept)
            ]
        )
        return np.mean(member_embeddings, axis=0)

    def evaluate_kind_membership(self, word: str, kind: str) -> float:
        word_node = self.concept_as_str_node(ObjectConcept(word))
        kind_node = self.concept_as_str_node(KindConcept(kind))
        if kind_node not in self.nodes or word_node not in self.nodes:
            return 0
        return cos_sim(
            self.object_concept_embedding(word_node),
            self.kind_concept_embedding(kind_node),
        )

    @staticmethod
    def concept_as_str_node(concept: Concept, syntactic_position="") -> str:
        if syntactic_position:
            return f"{concept.debug_string}_{str(type(concept))}_{syntactic_position}"
        else:
            return f"{concept.debug_string}_{str(type(concept))}"


def get_concept_node_from_graph(
    identifier: str, semantics_graph: Graph
) -> Optional[Concept]:
    return first([n for n in semantics_graph.nodes if n.debug_string == identifier], None)


def cos_sim(a, b) -> float:
    dot = np.dot(a.reshape(1, -1), b.reshape(-1, 1))
    norma = np.linalg.norm(a.reshape(1, -1))
    normb = np.linalg.norm(b.reshape(1, -1))
    return dot / (norma * normb)


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
