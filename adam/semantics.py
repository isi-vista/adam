"""
Classes to represent semantics from the learner's point-of-view.

Really this and `HighLevelSemanticsSituation` should somehow be refactored together,
but it's not worth the trouble at this point.
"""
from typing import Protocol, Tuple

from adam.learner.surface_templates import SurfaceTemplateVariable
from attr import attrib, attrs
from attr.validators import deep_mapping, instance_of
from immutablecollections import ImmutableDict
from immutablecollections.converter_utils import _to_immutabledict


class Concept:
    pass


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


class LearnerSemanticNode(Protocol):
    concept: Concept
    slot_fillings: ImmutableDict[SurfaceTemplateVariable, "ObjectSemanticNode"]


@attrs(frozen=True, eq=False)
class ObjectSemanticNode(LearnerSemanticNode):
    concept: ObjectConcept = attrib(validator=instance_of(ObjectConcept))
    slot_fillings: ImmutableDict[SurfaceTemplateVariable, "ObjectSemanticNode"] = attrib(
        converter=_to_immutabledict
    )

    # def __attrs_post_init__(self) -> None:
    #     for template in self.templates:
    #         check_arg(template.num_slots == 0)


@attrs(frozen=True, eq=False)
class AttributeSemanticNode(LearnerSemanticNode):
    concept: AttributeConcept = attrib(validator=instance_of(AttributeConcept))
    slot_fillings: ImmutableDict[SurfaceTemplateVariable, "ObjectSemanticNode"] = attrib(
        converter=_to_immutabledict,
        validator=deep_mapping(
            instance_of(SurfaceTemplateVariable), instance_of(ObjectSemanticNode)
        ),
    )

    # def __attrs_post_init__(self) -> None:
    #     for template in self.templates:
    #         check_arg(template.num_slots == 1)


@attrs(frozen=True, eq=False)
class RelationSemanticNode(LearnerSemanticNode):
    concept: RelationConcept = attrib(validator=instance_of(RelationConcept))
    slot_fillings: ImmutableDict[SurfaceTemplateVariable, "ObjectSemanticNode"] = attrib(
        converter=_to_immutabledict,
        validator=deep_mapping(
            instance_of(SurfaceTemplateVariable), instance_of(ObjectSemanticNode)
        ),
    )

    # def __attrs_post_init__(self) -> None:
    #     for template in self.templates:
    #         check_arg(template.num_slots == 2)


@attrs(frozen=True, eq=False)
class ActionSemanticNode(LearnerSemanticNode):
    concept: ActionConcept = attrib(validator=instance_of(ActionConcept))
    slot_fillings: ImmutableDict[SurfaceTemplateVariable, "ObjectSemanticNode"] = attrib(
        converter=_to_immutabledict,
        validator=deep_mapping(
            instance_of(SurfaceTemplateVariable), instance_of(ObjectSemanticNode)
        ),
    )

    # def __attrs_post_init__(self) -> None:
    #     for template in self.templates:
    #         check_arg(template.num_slots >= 1)
