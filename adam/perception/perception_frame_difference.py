from attr import attrib, attrs
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset

from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    ObjectPerception,
    PropertyPerception,
    RelationPerception,
)


@attrs(slots=True, frozen=True)
class DevelopmentalPrimitivePerceptionFrameDiff:
    r"""
    Represents the difference between a `LanguageLearner`\ 's perception
    of two different moments as sets of added and removed relations, perceived_objects, and property assertions.
    """

    added_objects: ImmutableSet[ObjectPerception] = attrib(
        kw_only=True, converter=_to_immutableset, default=immutableset()
    )
    r"""
    the set of `ObjectPerception`\ s, that were present on the second perception frame, but not the first.
    """
    removed_objects: ImmutableSet[ObjectPerception] = attrib(
        kw_only=True, converter=_to_immutableset, default=immutableset()
    )
    r"""
    the set of `ObjectPerception`\ s, that were present on the first perception frame, but not the second.
    """
    added_property_assertions: ImmutableSet[PropertyPerception] = attrib(
        kw_only=True, converter=_to_immutableset, default=immutableset()
    )
    r"""
    the set of `PropertyPerception`\ s, that were present on the second perception frame, but not the first.
    """
    removed_property_assertions: ImmutableSet[PropertyPerception] = attrib(
        kw_only=True, converter=_to_immutableset, default=immutableset()
    )
    r"""
    the set of `PropertyPerception`\ s, that were present on the first perception frame, but not the second.
    """
    added_relations: ImmutableSet[RelationPerception] = attrib(
        kw_only=True, converter=_to_immutableset, default=immutableset()
    )
    r"""
    the set of `RelationPerception`\ s, that were present on the second perception frame, but not the first.
    """
    removed_relations: ImmutableSet[RelationPerception] = attrib(
        kw_only=True, converter=_to_immutableset, default=immutableset()
    )
    r"""
    the set of `RelationPerception`\ s, that were present on the first perception frame, but not the second
    """


def diff_primitive_perception_frames(
    before: DevelopmentalPrimitivePerceptionFrame,
    after: DevelopmentalPrimitivePerceptionFrame,
) -> DevelopmentalPrimitivePerceptionFrameDiff:
    r"""
    Given a before and an after frame, computes the difference between two frames and
    returns a `DevelopmentalPrimitivePerceptionFrameDiff` object.
    """
    return DevelopmentalPrimitivePerceptionFrameDiff(
        added_objects=after.perceived_objects.difference(before.perceived_objects),
        removed_objects=before.perceived_objects.difference(after.perceived_objects),
        added_property_assertions=after.property_assertions.difference(
            before.property_assertions
        ),
        removed_property_assertions=before.property_assertions.difference(
            after.property_assertions
        ),
        added_relations=after.relations.difference(before.relations),
        removed_relations=before.relations.difference(after.relations),
    )
