from typing import List, Tuple

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
    node_to_relations: List[Relation[OntologyNode]] = []
    bigger: List[OntologyNode] = []
    for nodes in relative_size_nodes:
        for node in nodes:
            for entry in bigger:
                node_to_relations.append(
                    Relation(
                        relation_type=opposite_type, first_slot=node, second_slot=entry
                    )
                )
                node_to_relations.append(
                    Relation(
                        relation_type=relation_type, first_slot=entry, second_slot=node
                    )
                )
        bigger.extend(nodes)
    return node_to_relations
