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


PerceptionDiffT = TypeVar("PerceptionDiffT", bound="PerceptionFrameDiff")


@attrs(slots=True, frozen=True)
class DevelopmentalPrimitivePerceptionFrameDiff(PerceptionFrameDiff):
    r"""
    Represents the difference between a `LanguageLearner`\ 's perception
    of two different moments as sets of added and removed relations, perceived_objects, and property assertions.
    """

    before_frame: DevelopmentalPrimitivePerceptionFrame = attrib(
        validator=instance_of(DevelopmentalPrimitivePerceptionFrame))
    r"""
    a `DevelopmentalPrimitivePerceptionFrame` representing the first perception frame.
    """
    after_frame: DevelopmentalPrimitivePerceptionFrame = attrib(
        validator=instance_of(DevelopmentalPrimitivePerceptionFrame))
    r"""
    a `DevelopmentalPrimitivePerceptionFrame` representing the second perception frame.
    """

    def get_added_perceived_objects(self) -> ImmutableSet["ObjectPerception"]:
        r"""
        returns the set of `ObjectPerception`\ s, that were present on the second perception frame, but not the first.
        """
        return self.after_frame.perceived_objects.difference(self.before_frame.perceived_objects)

    def get_removed_perceived_objects(self) -> ImmutableSet["ObjectPerception"]:
        r"""
        returns the set of `ObjectPerception`\ s, that were present on the first perception frame, but not the second.
        """
        return self.before_frame.perceived_objects.difference(self.after_frame.perceived_objects)

    def get_added_property_assertions(self) -> ImmutableSet["PropertyPerception"]:
        r"""
        returns the set of `PropertyPerception`\ s, that were present on the second perception frame, but not the first.
        """
        return self.after_frame.property_assertions.difference(self.before_frame.property_assertions)

    def get_removed_property_assertions(self) -> ImmutableSet["PropertyPerception"]:
        r"""
        returns the set of `PropertyPerception`\ s, that were present on the first perception frame, but not the second.
        """
        return self.before_frame.property_assertions.difference(self.after_frame.property_assertions)

    def get_added_relations(self) -> ImmutableSet["RelationPerception"]:
        r"""
        returns the set of `RelationPerception`\ s, that were present on the second perception frame, but not the first.
        """
        return self.after_frame.relations.difference(self.before_frame.relations)

    def get_removed_relations(self) -> ImmutableSet["RelationPerception"]:
        r"""
        returns the set of `RelationPerception`\ s, that were present on the first perception frame, but not the second
        """
        return self.before_frame.relations.difference(self.after_frame.relations)