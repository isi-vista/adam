"""
Classes to represent semantics from the learner's point-of-view.

Really this and `HighLevelSemanticsSituation` should somehow be refactored together,
but it's not worth the trouble at this point.
"""
from typing import Mapping

from typing_extensions import Protocol, runtime

from attr import attrib, attrs
from attr.validators import deep_mapping, instance_of
from immutablecollections import ImmutableDict, immutabledict
from immutablecollections.converter_utils import _to_immutabledict


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
