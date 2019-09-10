from abc import ABC, abstractmethod
from typing import AbstractSet, Iterable

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from vistautils.preconditions import check_arg

from adam.ontology import Ontology, OntologyNode, THING


class OntologyNodeSelector(ABC):
    r"""
    A method for specifying a subset of the notes in an `Ontology`.

    This is for use in specifying `SituationTemplate`\ s.
    """

    @abstractmethod
    def select_nodes(self, ontology: Ontology) -> AbstractSet[OntologyNode]:
        """
        Select and return some subset of the nodes in *ontology*.
        """


@attrs(frozen=True, slots=True)
class Is(OntologyNodeSelector):
    """
    An `OntologyNodeSelector` which always selects exactly the specified node.
    """

    _nodes: ImmutableSet[OntologyNode] = attrib(validator=instance_of(OntologyNode))

    def select_nodes(self, ontology: Ontology) -> AbstractSet[OntologyNode]:
        for node in self._nodes:
            check_arg(node in ontology, f"{node} is not in the ontology")
        return self._nodes


@attrs(frozen=True, slots=True)
class ByHierarchyAndProperties(OntologyNodeSelector):
    """
    An `OntologyNodeSelector` which selects all nodes which possess certain properties.
    """

    _descendents_of: OntologyNode = attrib(validator=instance_of(OntologyNode))
    _required_properties: ImmutableSet[OntologyNode] = attrib(
        converter=_to_immutableset, default=immutableset()
    )

    def select_nodes(self, ontology: Ontology) -> AbstractSet[OntologyNode]:
        return ontology.nodes_with_properties(
            self._descendents_of, self._required_properties
        )


def object_variable(
    root_node: OntologyNode = THING,
    with_properties: Iterable[OntologyNode] = immutableset(),
):
    return ByHierarchyAndProperties(
        descendents_of=root_node, required_properties=with_properties
    )
