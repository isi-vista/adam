"""
Classes to represent semantics from the learner's point-of-view.

Really this and `HighLevelSemanticsSituation` should somehow be refactored together,
but it's not worth the trouble at this point.
"""
from typing import Iterable, Mapping

from more_itertools import flatten, one
from typing_extensions import Protocol, runtime

from attr import attrib, attrs
from attr.validators import deep_mapping, in_, instance_of
from immutablecollections import (
    ImmutableDict,
    ImmutableSet,
    ImmutableSetMultiDict,
    immutabledict,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import _to_immutabledict, _to_immutableset
from vistautils.range import Range


@runtime
class Concept(Protocol):
    debug_string: str


@attrs(frozen=True, slots=True)
class SyntaxSemanticsVariable:
    """
    A variable portion of a a `SurfaceTemplate` or of a learner semantic structure.
    """

    name: str = attrib(validator=instance_of(str))


@attrs(frozen=True, eq=False)
class ObjectConcept(Concept):
    debug_string: str = attrib(validator=instance_of(str))


@attrs(frozen=True, eq=False)
class AttributeConcept(Concept):
    debug_string: str = attrib(validator=instance_of(str))


@attrs(frozen=True, eq=False)
class RelationConcept(Concept):
    debug_string: str = attrib(validator=instance_of(str))


@attrs(frozen=True, eq=False)
class ActionConcept(Concept):
    debug_string: str = attrib(validator=instance_of(str))


@attrs(frozen=True, eq=False)
class NumberConcept(Concept):
    """
    The concept of some number of things.
    """

    number: int = attrib(validator=in_(Range.at_least(1)))
    debug_string: str = attrib(validator=instance_of(str))


GROUND_OBJECT_CONCEPT = ObjectConcept("ground")


@runtime
class SemanticNode(Protocol):
    concept: Concept
    slot_fillings: ImmutableDict[SyntaxSemanticsVariable, "ObjectSemanticNode"]

    @staticmethod
    def for_concepts_and_arguments(
        concept: Concept,
        slots_to_fillers: Mapping[SyntaxSemanticsVariable, "ObjectSemanticNode"],
    ):
        if isinstance(concept, ObjectConcept):
            if slots_to_fillers:
                raise RuntimeError(
                    f"Objects should not have arguments, but got concept "
                    f"{concept} with arguments {slots_to_fillers}"
                )
            return ObjectSemanticNode(concept)
        elif isinstance(concept, AttributeConcept):
            return AttributeSemanticNode(concept, slots_to_fillers)
        elif isinstance(concept, RelationConcept):
            return RelationSemanticNode(concept, slots_to_fillers)
        elif isinstance(concept, ActionConcept):
            return ActionSemanticNode(concept, slots_to_fillers)
        else:
            raise RuntimeError(
                f"Don't know how to make a semantic node from concept " f"{concept}"
            )


@attrs(frozen=True, eq=False)
class ObjectSemanticNode(SemanticNode):
    concept: ObjectConcept = attrib(validator=instance_of(ObjectConcept))
    slot_fillings: ImmutableDict[SyntaxSemanticsVariable, "ObjectSemanticNode"] = attrib(
        init=False, default=immutabledict()
    )

    # def __attrs_post_init__(self) -> None:
    #     for template in self.templates:
    #         check_arg(template.num_slots == 0)


@attrs(frozen=True, eq=False)
class AttributeSemanticNode(SemanticNode):
    concept: AttributeConcept = attrib(validator=instance_of(AttributeConcept))
    slot_fillings: ImmutableDict[SyntaxSemanticsVariable, "ObjectSemanticNode"] = attrib(
        converter=_to_immutabledict,
        validator=deep_mapping(
            instance_of(SyntaxSemanticsVariable), instance_of(ObjectSemanticNode)
        ),
    )

    # def __attrs_post_init__(self) -> None:
    #     for template in self.templates:
    #         check_arg(template.num_slots == 1)


@attrs(frozen=True, eq=False)
class RelationSemanticNode(SemanticNode):
    concept: RelationConcept = attrib(validator=instance_of(RelationConcept))
    slot_fillings: ImmutableDict[SyntaxSemanticsVariable, "ObjectSemanticNode"] = attrib(
        converter=_to_immutabledict,
        validator=deep_mapping(
            instance_of(SyntaxSemanticsVariable), instance_of(ObjectSemanticNode)
        ),
    )

    # def __attrs_post_init__(self) -> None:
    #     for template in self.templates:
    #         check_arg(template.num_slots == 2)


@attrs(frozen=True, eq=False)
class ActionSemanticNode(SemanticNode):
    concept: ActionConcept = attrib(validator=instance_of(ActionConcept))
    slot_fillings: ImmutableDict[SyntaxSemanticsVariable, "ObjectSemanticNode"] = attrib(
        converter=_to_immutabledict,
        validator=deep_mapping(
            instance_of(SyntaxSemanticsVariable), instance_of(ObjectSemanticNode)
        ),
    )

    # def __attrs_post_init__(self) -> None:
    #     for template in self.templates:
    #         check_arg(template.num_slots >= 1)


@attrs(frozen=True, slots=True)
class QuantificationSemanticNode(SemanticNode):
    concept: NumberConcept = attrib(validator=instance_of(NumberConcept))
    slot_fillings: ImmutableDict[SyntaxSemanticsVariable, "ObjectSemanticNode"] = attrib(
        converter=_to_immutabledict,
        validator=deep_mapping(
            instance_of(SyntaxSemanticsVariable), instance_of(ObjectSemanticNode)
        ),
    )

    def __attrs_post_init__(self) -> None:
        if len(self.slot_fillings) != 1:
            raise RuntimeError(
                f"QuantificationSemanticNode should have only one slot but got "
                f"{self.slot_fillings}"
            )


@attrs(frozen=True)
class LearnerSemantics:
    """
    Represent's the learner's semantic (rather than perceptual) understanding of a situation.

    The learner is assumed to view the situation as a collection of *objects* which possess
    *attributes*, have *relations* to one another, and serve as the arguments of *actions*.
    """

    objects: ImmutableSet[ObjectSemanticNode] = attrib(converter=_to_immutableset)
    attributes: ImmutableSet[AttributeSemanticNode] = attrib(converter=_to_immutableset)
    relations: ImmutableSet[RelationSemanticNode] = attrib(converter=_to_immutableset)
    actions: ImmutableSet[ActionSemanticNode] = attrib(converter=_to_immutableset)
    quantifiers: ImmutableSet[QuantificationSemanticNode] = attrib(
        converter=_to_immutableset
    )

    objects_to_attributes: ImmutableSetMultiDict[
        ObjectSemanticNode, AttributeSemanticNode
    ] = attrib(init=False)
    objects_to_relation_in_slot1: ImmutableSetMultiDict[
        ObjectSemanticNode, RelationSemanticNode
    ] = attrib(init=False)
    objects_to_actions: ImmutableSetMultiDict[
        ObjectSemanticNode, ActionSemanticNode
    ] = attrib(init=False)

    @staticmethod
    def from_nodes(semantic_nodes: Iterable[SemanticNode]) -> "LearnerSemantics":
        semantic_nodes_tuple = tuple(semantic_nodes)
        for node in semantic_nodes_tuple:
            if not isinstance(node, SemanticNode):
                raise RuntimeError(
                    f"Tried to add something which is not a semantic node to "
                    f"LearnerSemantics: {node}"
                )
        return LearnerSemantics(
            objects=[
                node
                for node in semantic_nodes_tuple
                if isinstance(node, ObjectSemanticNode)
            ],
            attributes=[
                node
                for node in semantic_nodes_tuple
                if isinstance(node, AttributeSemanticNode)
            ],
            relations=[
                node
                for node in semantic_nodes_tuple
                if isinstance(node, RelationSemanticNode)
            ],
            actions=[
                node
                for node in semantic_nodes_tuple
                if isinstance(node, ActionSemanticNode)
            ],
            quantifiers=[
                node
                for node in semantic_nodes_tuple
                if isinstance(node, QuantificationSemanticNode)
            ],
        )

    @objects_to_attributes.default
    def _init_objects_to_attributes(
        self
    ) -> ImmutableSetMultiDict[ObjectSemanticNode, AttributeSemanticNode]:
        return immutablesetmultidict(
            (one(attribute.slot_fillings.values()), attribute)
            for attribute in self.attributes
        )

    @objects_to_relation_in_slot1.default
    def _init_objects_to_relations(
        self
    ) -> ImmutableSetMultiDict[ObjectSemanticNode, AttributeSemanticNode]:
        return immutablesetmultidict(
            flatten(
                [
                    (slot_filler, relation)
                    for slot_filler in relation.slot_fillings.values()
                ]
                for relation in self.relations
            )
        )

    @objects_to_actions.default
    def _init_objects_to_actions(
        self
    ) -> ImmutableSetMultiDict[ObjectSemanticNode, AttributeSemanticNode]:
        return immutablesetmultidict(
            flatten(
                [(slot_filler, action) for slot_filler in action.slot_fillings.values()]
                for action in self.actions
            )
        )
