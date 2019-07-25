r"""
Representations for simple ontologies.

These ontologies are intended to be used when describing `Situation`\ s and writing
`SituationTemplate`\ s.
"""
from attr import attrs, attrib
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from networkx import DiGraph


@attrs(frozen=True, slots=True, repr=False)
class OntologyProperty:
    r"""
    A property which a node in an `Ontology` may bear, such as "animate".

    A `OntologyProperty` has a *handle*, which is a user-facing description used for debugging only.
    """

    _handle: str = attrib(validator=instance_of(str))

    def __repr__(self) -> str:
        return f"+{self._handle}"


@attrs(frozen=True, slots=True)
class OntologyNode:
    r"""
    A node in an ontology representing some type of object, action, or relation, such as
    "animate object" or "transfer action."

    An `OntologyNode` has a *handle*, which is a user-facing description used for debugging only.

    It may also have a set of *local_properties* which are inherited by all child nodes.
    """

    _handle: str = attrib(validator=instance_of(str))
    _local_properties: ImmutableSet[OntologyProperty] = attrib(
        converter=_to_immutableset, default=immutableset()
    )


@attrs(frozen=True, slots=True)
class Ontology:
    r"""
    A collection of `OntologyNode`\ s with parent-child relationships.

    This cannot yet be used for anything.
    """

    _graph: DiGraph = attrib(validator=instance_of(DiGraph))

    @staticmethod
    def from_directed_graph(graph: DiGraph) -> "Ontology":
        r"""
        Create an ontology from a NetworkX ::class`DiGraph`.

        Args:
            graph: a NetworkX graph representing the ontology.  Sub-class
                `OntologyNode`\ s should have edges pointing to super-class `OntologyNode`\ s.
                If the graph is not acyclic, the result is undefined.

        Returns:
            The `Ontology` encoding the relationships in the `networkx.DiGraph`.
        """
        return Ontology(graph)
