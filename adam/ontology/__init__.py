r"""
Representations for simple ontologies.

These ontologies are intended to be used when describing `Situation`\ s and writing `SituationTemplate`\ s.
"""

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from networkx import DiGraph

from adam.ontology.phase1_spatial_relations import Distance, Region


@attrs(frozen=True, slots=True, repr=False)
class OntologyNode:
    r"""
    A node in an ontology representing some type of object, action, or relation, such as
    "animate object" or "transfer action."

    An `OntologyNode` has a *handle*, which is a user-facing description used for debugging
    and testing only.

    It may also have a set of *local_properties* which are inherited by all child nodes.
    """

    handle: str = attrib(validator=instance_of(str))
    """
    A simple human-readable description of this node,
    used for debugging and testing only.
    """
    inheritable_properties: ImmutableSet["OntologyNode"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    Properties of the `OntologyNode`, as a set of `OntologyNode`\ s
    which should be inherited by its children.
    """
    non_inheritable_properties: ImmutableSet["OntologyNode"] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    r"""
    Properties of the `OntologyNode`, as a set of `OntologyNode`\ s
    which should not be inherited by its children.
    """

    def __repr__(self) -> str:
        if self.inheritable_properties:
            local_properties = ",".join(
                str(local_property) for local_property in self.inheritable_properties
            )
            properties_string = f"[{local_properties}]"
        else:
            properties_string = ""
        return f"{self.handle}{properties_string}"


@attrs(frozen=True, slots=True)
class OntologyNodeRelation:
    r"""
    This class defines the relationships between `OntologyNode`\ s
    """

    relation_type: OntologyNode = attrib(validator=instance_of(OntologyNode))
    """
    An `OntologyNode` which gives the relationship type between the args
    """
    arg1: OntologyNode = attrib(validator=instance_of(OntologyNode))
    """
    An `OntologyNode` which is the first argument of the relation_type
    """
    arg2: OntologyNode = attrib(validator=instance_of(OntologyNode))
    """
    An `OntologyNode` which is the second argument of the relation_type
    """


_OneOrMoreOntologyNodes = Union[OntologyNode, Iterable[OntologyNode]]


def build_size_relationships(
    relative_size_nodes: Tuple[Tuple[OntologyNode, ...], ...],
    *,
    relation_type: OntologyNode,
    opposite_type: OntologyNode,
) -> ImmutableDict[OntologyNode, ImmutableSet[OntologyNodeRelation]]:
    node_to_relations: Dict[OntologyNode, List[OntologyNodeRelation]] = {}
    bigger: List[OntologyNode] = []
    for nodes in relative_size_nodes:
        for node in nodes:
            if node not in node_to_relations.keys():
                node_to_relations.update({node: []})
            for entry in bigger:
                node_to_relations[node].append(
                    OntologyNodeRelation(
                        relation_type=opposite_type, arg1=node, arg2=entry
                    )
                )
                node_to_relations[entry].append(
                    OntologyNodeRelation(
                        relation_type=relation_type, arg1=entry, arg2=node
                    )
                )
        bigger.append(flatten(nodes))
    return _to_immutabledict(node_to_relations)


# by convention, the following should appear in all Ontologies

CAN_FILL_TEMPLATE_SLOT = OntologyNode("can-fill-template-slot")
r"""
A property indicating that a node can be instantiated in a scene.

The ontology contains many nodes which, 
while useful for various purposes,
do not themselves form part of our primary concept vocabulary.
This property distinguishes the elements of our core "concept vocabulary"
from such auxiliary concepts.

For example, PERSON is one of our core concepts; 
we have a concept of ARM which is used in defining the `ObjectStructuralSchema` of PERSON
but disembodied arms should never be instantiated in templates directly.
"""

THING = OntologyNode("thing")
r"""
Ancestor of all objects in an `Ontology`.

By convention this should appear in all `Ontology`\ s.
"""
RELATION = OntologyNode("relation")
r"""
Ancestor of all relations in an `Ontology`.

By convention this should appear in all `Ontology`\ s.
"""
ACTION = OntologyNode("action")
r"""
Ancestor of all actions in an `Ontology`.

By convention this should appear in all `Ontology`\ s.
"""
PROPERTY = OntologyNode("property")
r"""
Ancestor of all properties in an `Ontology`.

By convention this should appear in all `Ontology`\ s.
"""

META_PROPERTY = OntologyNode("meta-property")
r"""
A property of a property.

For example, whether it is perceivable or binary.

By convention this should appear in all `Ontology`\ s.
"""

IN_REGION = OntologyNode("in-region")
"""
Indicates that an object is located in a `Region`.

This is used to support the Landau and Jackendoff interpretation of prepositions.
"""

REQUIRED_ONTOLOGY_NODES = immutableset(
    [THING, RELATION, ACTION, PROPERTY, META_PROPERTY, CAN_FILL_TEMPLATE_SLOT, IN_REGION]
)


def minimal_ontology_graph():
    """
    Get the NetworkX DiGraph corresponding to the minimum legal ontology,
    containing all required nodes.

    This is useful as a convenient foundation for building your own ontologies.
    """
    ret = DiGraph()
    for node in REQUIRED_ONTOLOGY_NODES:
        ret.add_node(node)
    ret.add_edge(IN_REGION, RELATION)
    return ret
