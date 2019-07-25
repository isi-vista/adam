"""
Structures for describing situations in the world at an abstacted, human-friendly level.
"""
from abc import ABC
from typing import Mapping

from attr import attrs, attrib
from attr.validators import instance_of
from immutablecollections import immutableset, ImmutableSet, immutabledict

# noinspection PyProtectedMember
from immutablecollections.converter_utils import _to_immutableset, _to_immutabledict

from adam.math_3d import Point
from adam.ontology import OntologyProperty


class Situation(ABC):
    r"""
    A situation is a high-level representation of a configuration of objects, possibly including
    changes in the states of objects across time.

    A Curriculum is a sequence of situations.

    Situations are a high-level description intended to make it easy for human beings to specify
    curricula.  Situations will be transformed into pairs of `PerceptualRepresentation`\ s and
    `LinguisticDescription`\ s for input to a `LanguageLearner`.
    """


@attrs(frozen=True)
class BagOfFeaturesSituationRepresentation(Situation):
    r"""
    Represents a `Situation` as an unstructured set of features.

    For testing purposes only.
    """
    features: ImmutableSet[str] = attrib(converter=immutableset)


@attrs(frozen=True, slots=True, hash=None, cmp=False)
class SituationObject:
    """
    An object present in some situation.

    Unlike most of our classes, `SituationObject` has *id*-based hashing and equality.  This is
    because two objects with identical properties are nonetheless distinct.
    """

    type_tag: str = attrib(validator=instance_of(str))
    properties: ImmutableSet[OntologyProperty] = attrib(
        converter=_to_immutableset, default=immutableset()
    )


@attrs(frozen=True, slots=True)
class LocatedObjectSituation(Situation):
    """
    A representation of a situation as a set of objects located at particular points.
    """

    objects_to_locations: Mapping[SituationObject, Point] = attrib(
        converter=_to_immutabledict, default=immutabledict()
    )
