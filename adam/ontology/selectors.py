"""
Utilities for specifying a sub-set of the nodes in an `Ontology`
"""
from abc import ABC, abstractmethod
from typing import AbstractSet

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from vistautils.preconditions import check_arg

from adam.ontology import OntologyNode
from adam.ontology.ontology import Ontology


class OntologyNodeSelector(ABC):
    r"""
    A method for specifying a subset of the notes in an `Ontology`.

    This is for use in specifying `SituationTemplate`\ s.
    """

    def select_nodes(
        self, ontology: Ontology, *, require_non_empty_result=False
    ) -> AbstractSet[OntologyNode]:
        """
        Select and return some subset of the nodes in *ontology*.
        """
        ret = self._select_nodes(ontology)
        if require_non_empty_result and not ret:
            raise RuntimeError(f"No node in {ontology} satisfied {self}")
        return ret

    @abstractmethod
    def _select_nodes(self, ontology: Ontology) -> AbstractSet[OntologyNode]:
        """
        Method sub-classes should override to perform the actual selection.
        """


@attrs(frozen=True, slots=True)
class Is(OntologyNodeSelector):
    """
    An `OntologyNodeSelector` which always selects exactly the specified node.
    """

    _nodes: ImmutableSet[OntologyNode] = attrib(converter=_to_immutableset)

    def _select_nodes(self, ontology: Ontology) -> AbstractSet[OntologyNode]:
        for node in self._nodes:
            check_arg(node in ontology, f"{node} is not in the ontology")
        return self._nodes


@attrs(frozen=True, slots=True)
class ByHierarchyAndProperties(OntologyNodeSelector):
    """
    An `OntologyNodeSelector` which selects all nodes
     which are descendents of *descendents_of*,
     which possess all of *required_properties*,
     and which possess none of *banned_properties*.
    """

    _descendents_of: OntologyNode = attrib(validator=instance_of(OntologyNode))
    _required_properties: ImmutableSet[OntologyNode] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    _banned_properties: ImmutableSet[OntologyNode] = attrib(
        converter=_to_immutableset, default=immutableset()
    )

    def _select_nodes(self, ontology: Ontology) -> AbstractSet[OntologyNode]:
        return ontology.nodes_with_properties(
            self._descendents_of,
            self._required_properties,
            banned_properties=self._banned_properties,
        )


@attrs(frozen=True, slots=True)
class AndOntologySelector(OntologyNodeSelector):
    _sub_selectors: ImmutableSet[OntologyNodeSelector] = attrib(
        converter=_to_immutableset, default=immutableset()
    )

    def _select_nodes(self, ontology: Ontology) -> AbstractSet[OntologyNode]:
        # we don't just use set.intersection here because we want to guarantee
        # deterministic iteration ordering
        matches = [selector.select_nodes(ontology) for selector in self._sub_selectors]

        first_matches = matches[0]
        later_matches = matches[1:]

        return immutableset(
            x
            for x in first_matches
            if all(x in later_match_set for later_match_set in later_matches)
        )

    def __attrs_post_init__(self) -> None:
        check_arg(
            len(self._sub_selectors) > 1, "_And requires at least two sub-selectors"
        )


@attrs(frozen=True, slots=True)
class SubcategorizationSelector(OntologyNodeSelector):
    required_subcategorization_frame: ImmutableSet[OntologyNode] = attrib(
        converter=_to_immutableset
    )

    def _select_nodes(self, ontology: Ontology) -> AbstractSet[OntologyNode]:
        return immutableset(
            action
            for (action, action_description) in ontology.action_to_description.items()
            if any(
                frame.roles_to_entities.keys() == self.required_subcategorization_frame
                for frame in action_description.frames
            )
        )
