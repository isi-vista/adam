from abc import ABC
from typing import Generic, TypeVar

from attr import attrib, attrs
from attr.validators import in_, instance_of
from immutablecollections import ImmutableSet
from adam.perception.developmental_primitive_perception import DevelopmentalPrimitivePerceptionFrame, ObjectPerception, \
    PropertyPerception, RelationPerception


class PerceptionFrameDiff(ABC):
    r"""
    Represents the difference between a `LanguageLearner`\ 's perception
    of two different moments.
    """


@attrs(slots=True, frozen=True)
class DevelopmentalPrimitivePerceptionFrameDiff(PerceptionFrameDiff):
    r"""
    Represents the difference between a `LanguageLearner`\ 's perception
    of two different moments as sets of added and removed relations, perceived_objects, and property assertions.
    """

    added_objects: ImmutableSet["ObjectPerception"] = attrib(validator=instance_of(ImmutableSet))
    r"""
    the set of `ObjectPerception`\ s, that were present on the second perception frame, but not the first.
    """
    removed_objects: ImmutableSet["ObjectPerception"] = attrib(validator=instance_of(ImmutableSet))
    r"""
    the set of `ObjectPerception`\ s, that were present on the first perception frame, but not the second.
    """
    added_property_assertions: ImmutableSet["PropertyPerception"] = attrib(validator=instance_of(ImmutableSet))
    r"""
    the set of `PropertyPerception`\ s, that were present on the second perception frame, but not the first.
    """
    removed_property_assertions: ImmutableSet["PropertyPerception"] = attrib(validator=instance_of(ImmutableSet))
    r"""
    the set of `PropertyPerception`\ s, that were present on the first perception frame, but not the second.
    """
    added_relations: ImmutableSet["RelationPerception"] = attrib(validator=instance_of(ImmutableSet))
    r"""
    the set of `RelationPerception`\ s, that were present on the second perception frame, but not the first.
    """
    removed_relations: ImmutableSet["RelationPerception"] = attrib(validator=instance_of(ImmutableSet))
    r"""
    the set of `RelationPerception`\ s, that were present on the first perception frame, but not the second
    """

def diff_primitive_perception_frames(before: DevelopmentalPrimitivePerceptionFrame,
                                     after: DevelopmentalPrimitivePerceptionFrame) \
        -> DevelopmentalPrimitivePerceptionFrameDiff:
    r"""
    Given a before and an after frame, computes the difference between two frames and
    returns a `DevelopmentalPrimitivePerceptionFrameDiff` object.
    """

    added_objects: ImmutableSet["ObjectPerception"] = \
        after.perceived_objects.difference(before.perceived_objects)
    removed_objects: ImmutableSet["ObjectPerception"] = \
        before.perceived_objects.difference(after.perceived_objects)
    added_property_assertions: ImmutableSet["PropertyPerception"] = \
        after.property_assertions.difference(before.property_assertions)
    removed_property_assertions: ImmutableSet["PropertyPerception"] = \
        before.property_assertions.difference(after.property_assertions)
    added_relations: ImmutableSet["RelationPerception"] = \
        after.relations.difference(before.relations)
    removed_relations: ImmutableSet["RelationPerception"] = \
        before.relations.difference(after.relations)

    return DevelopmentalPrimitivePerceptionFrameDiff(added_objects, removed_objects, added_property_assertions,
                                                     removed_property_assertions, added_relations, removed_relations)