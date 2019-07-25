from attr import attrs, attrib
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from networkx import Graph

@attrs(frozen=True, slots=True)
class OntologyProperty:
    handle: str = attrib(validator=instance_of(str))


@attrs(frozen=True, slots=True)
class OntologyNode:
    handle: str = attrib(validator=instance_of(str))
    properties: ImmutableSet[OntologyProperty] = attrib(
        converter=_to_immutableset, default=immutableset())


@attrs(frozen=True, slots=True)
class Ontology:
    _graph: Graph = attrib(validator=instance_of(Graph))

