from typing import Tuple, Dict, List

from immutablecollections.converter_utils import _to_immutableset

from adam.ontology import OntologyNode
from adam.relation import Relation


def build_size_relationships(
    relative_size_nodes: Tuple[Tuple[OntologyNode, ...], ...],
    *,
    relation_type: OntologyNode,
    opposite_type: OntologyNode,
):
    """
    Build a dictionary to represent opposite relation_types between OntologyNodes

    Used primarily to represent relative size_of relationships, this function takes a ranking of
    `OntologyNode` objects by *relative_size_nodes* which are then assigned the appropriate
    *relation_type* and *opposite_type* respectively.

    For use see GAILA_PHASE_1_ONTOLOGY.
    """
    node_to_relations: Dict[OntologyNode, List[Relation[OntologyNode]]] = {}
    bigger: List[OntologyNode] = []
    for nodes in relative_size_nodes:
        for node in nodes:
            if node not in node_to_relations.keys():
                node_to_relations.update({node: []})
            for entry in bigger:
                node_to_relations[node].append(
                    Relation(
                        relation_type=opposite_type, first_slot=node, second_slot=entry
                    )
                )
                node_to_relations[entry].append(
                    Relation(
                        relation_type=relation_type, first_slot=entry, second_slot=node
                    )
                )
        bigger.extend(nodes)
    returner = []
    for node in node_to_relations:
        returner.append((node, _to_immutableset(node_to_relations[node])))
    return returner
