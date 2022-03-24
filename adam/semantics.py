"""
Classes to represent semantics from the learner's point-of-view.

Really this and `HighLevelSemanticsSituation` should somehow be refactored together,
but it's not worth the trouble at this point.
"""
from more_itertools import one, flatten
from typing import Mapping, Iterable, Optional

from typing_extensions import Protocol, runtime

from attr import attrib, attrs
from attr.validators import deep_mapping, instance_of, optional
from immutablecollections import (
    ImmutableDict,
    immutabledict,
    ImmutableSet,
    ImmutableSetMultiDict,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import _to_immutabledict, _to_immutableset


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
class KindConcept(Concept):
    debug_string: str = attrib(validator=instance_of(str))


@attrs(frozen=True, eq=False)
class RelationConcept(Concept):
    debug_string: str = attrib(validator=instance_of(str))


@attrs(frozen=True, eq=False)
class ActionConcept(Concept):
    debug_string: str = attrib(validator=instance_of(str))


@attrs(frozen=True, eq=False)
class FunctionalObjectConcept(ObjectConcept):
    debug_string: str = attrib(validator=instance_of(str))


@attrs(frozen=True, eq=False)
class GenericConcept(Concept):
    debug_string: str = attrib(validator=instance_of(str))


@attrs(frozen=True, eq=False)
class AffordanceConcept(Concept):
    debug_string: str = attrib(validator=instance_of(str))


GROUND_OBJECT_CONCEPT = ObjectConcept("ground")


@runtime
class SemanticNode(Protocol):
    concept: Concept
    slot_fillings: Mapping[SyntaxSemanticsVariable, "ObjectSemanticNode"]
    confidence: float
    original_node_id: Optional[str] = None

    @staticmethod
    def for_concepts_and_arguments(
        concept: Concept,
        slots_to_fillers: Mapping[SyntaxSemanticsVariable, "ObjectSemanticNode"],
        confidence: float,
        *,
        node_id: Optional[str] = None,
    ):
        if isinstance(concept, ObjectConcept):
            if slots_to_fillers:
                raise RuntimeError(
                    f"Objects should not have arguments, but got concept "
                    f"{concept} with arguments {slots_to_fillers}"
                )
            return ObjectSemanticNode(
                concept, confidence=confidence, original_node_id=node_id
            )
        elif isinstance(concept, AttributeConcept):
            return AttributeSemanticNode(
                concept, slots_to_fillers, confidence=confidence, original_node_id=node_id
            )
        elif isinstance(concept, RelationConcept):
            return RelationSemanticNode(
                concept, slots_to_fillers, confidence=confidence, original_node_id=node_id
            )
        elif isinstance(concept, ActionConcept):
            return ActionSemanticNode(
                concept, slots_to_fillers, confidence=confidence, original_node_id=node_id
            )
        elif isinstance(concept, AffordanceConcept):
            return AffordanceSemanticNode(
                concept, slots_to_fillers, confidence=confidence
            )
        else:
            raise RuntimeError(
                f"Don't know how to make a semantic node from concept " f"{concept}"
            )


@attrs(frozen=True, eq=False)
class ObjectSemanticNode(SemanticNode):
    concept: ObjectConcept = attrib(validator=instance_of(ObjectConcept))
    slot_fillings: Mapping[SyntaxSemanticsVariable, "ObjectSemanticNode"] = attrib(
        init=False, default=dict()
    )
    confidence: float = attrib(validator=instance_of(float))
    original_node_id: Optional[str] = attrib(
        default=None, validator=optional(instance_of(str))
    )

    # def __attrs_post_init__(self) -> None:
    #     for template in self.templates:
    #         check_arg(template.num_slots == 0)


@attrs(frozen=True)
class AttributeSemanticNode(SemanticNode):
    concept: AttributeConcept = attrib(validator=instance_of(AttributeConcept))
    slot_fillings: Mapping[SyntaxSemanticsVariable, "ObjectSemanticNode"] = attrib(
        validator=deep_mapping(
            instance_of(SyntaxSemanticsVariable), instance_of(ObjectSemanticNode)
        ),
    )
    confidence: float = attrib(validator=instance_of(float))
    original_node_id: Optional[str] = attrib(
        default=None, validator=optional(instance_of(str))
    )

    # def __attrs_post_init__(self) -> None:
    #     for template in self.templates:
    #         check_arg(template.num_slots == 1)


@attrs(frozen=True)
class RelationSemanticNode(SemanticNode):
    concept: RelationConcept = attrib(validator=instance_of(RelationConcept))
    slot_fillings: Mapping[SyntaxSemanticsVariable, "ObjectSemanticNode"] = attrib(
        validator=deep_mapping(
            instance_of(SyntaxSemanticsVariable), instance_of(ObjectSemanticNode)
        ),
    )
    confidence: float = attrib(validator=instance_of(float))
    original_node_id: Optional[str] = attrib(
        default=None, validator=optional(instance_of(str))
    )

    # def __attrs_post_init__(self) -> None:
    #     for template in self.templates:
    #         check_arg(template.num_slots == 2)


@attrs(frozen=True)
class ActionSemanticNode(SemanticNode):
    concept: ActionConcept = attrib(validator=instance_of(ActionConcept))
    slot_fillings: Mapping[SyntaxSemanticsVariable, "ObjectSemanticNode"] = attrib(
        validator=deep_mapping(
            instance_of(SyntaxSemanticsVariable), instance_of(ObjectSemanticNode)
        ),
    )
    confidence: float = attrib(validator=instance_of(float))
    original_node_id: Optional[str] = attrib(
        default=None, validator=optional(instance_of(str))
    )

    # def __attrs_post_init__(self) -> None:
    #     for template in self.templates:
    #         check_arg(template.num_slots >= 1)


@attrs(frozen=True)
class AffordanceSemanticNode(SemanticNode):
    concept: AffordanceConcept = attrib(validator=instance_of(AffordanceConcept))
    slot_fillings: ImmutableDict[SyntaxSemanticsVariable, ObjectSemanticNode] = attrib(
        converter=_to_immutabledict,
        validator=deep_mapping(
            instance_of(SyntaxSemanticsVariable), instance_of(ObjectSemanticNode)
        ),
    )
    confidence: float = attrib(validator=instance_of(float))


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

    functional_concept_to_object_concept: ImmutableDict[
        FunctionalObjectConcept, ObjectConcept
    ] = attrib(
        converter=_to_immutabledict,
        validator=deep_mapping(
            instance_of(FunctionalObjectConcept), instance_of(ObjectConcept)
        ),
        default=immutabledict(),
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
    def from_nodes(
        semantic_nodes: Iterable[SemanticNode],
        *,
        concept_map: ImmutableDict[
            FunctionalObjectConcept, ObjectConcept
        ] = immutabledict(),
    ) -> "LearnerSemantics":
        return LearnerSemantics(
            objects=[
                node for node in semantic_nodes if isinstance(node, ObjectSemanticNode)
            ],
            attributes=[
                node for node in semantic_nodes if isinstance(node, AttributeSemanticNode)
            ],
            relations=[
                node for node in semantic_nodes if isinstance(node, RelationSemanticNode)
            ],
            actions=[
                node for node in semantic_nodes if isinstance(node, ActionSemanticNode)
            ],
            functional_concept_to_object_concept=concept_map,
        )

    @objects_to_attributes.default
    def _init_objects_to_attributes(
        self,
    ) -> ImmutableSetMultiDict[ObjectSemanticNode, AttributeSemanticNode]:
        return immutablesetmultidict(
            (one(attribute.slot_fillings.values()), attribute)
            for attribute in self.attributes
        )

    @objects_to_relation_in_slot1.default
    def _init_objects_to_relations(
        self,
    ) -> ImmutableSetMultiDict[ObjectSemanticNode, AttributeSemanticNode]:
        return immutablesetmultidict(
            flatten(
                [
                    (slot_filler, relation)  # type: ignore
                    for slot_filler in relation.slot_fillings.values()
                ]
                for relation in self.relations
            )
        )

    @objects_to_actions.default
    def _init_objects_to_actions(
        self,
    ) -> ImmutableSetMultiDict[ObjectSemanticNode, AttributeSemanticNode]:
        return immutablesetmultidict(
            flatten(
                [(slot_filler, action) for slot_filler in action.slot_fillings.values()]  # type: ignore
                for action in self.actions
            )
        )
