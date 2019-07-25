"""
Tools for working with situation templates, which allow a human to compactly describe a large
number of possible situations.
"""
from _random import Random
from abc import ABC, abstractmethod
from typing import AbstractSet, Generic, List, Sequence, Tuple, TypeVar

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableDict, ImmutableSet, ImmutableSetMultiDict
from immutablecollections.converter_utils import (
    _to_immutabledict,
    _to_immutableset,
    _to_immutablesetmultidict,
)

from adam.ontology import OntologyNode, OntologyProperty
from adam.situation import Situation


class SituationTemplate(ABC):
    """
    A compact description for a large number of situations.
    """


_SituationTemplateT = TypeVar("_SituationTemplateT", bound=SituationTemplate)


class SituationTemplateProcessor(ABC, Generic[_SituationTemplateT]):
    r"""
    Turns a `SituationTemplate` into one or more `Situation`\ s.
    """

    @abstractmethod
    def generate_situations(
        self,
        template: _SituationTemplateT,
        *,
        num_instantiations: int = 1,
        rng: Random = Random(0)
    ) -> AbstractSet[Situation]:
        r"""
        Generates one or more `Situation`\ s from a `SituationTemplate`\ .

        The behavior of this method should be deterministic conditional upon
        an identically initialized `Random` being supplied.

        Args:
            template: the template to instantiate
            num_instantiations: the number of instantiations to produce
            rng: the random number generator to use.

        Returns:
            A set of instantiated `Situation`\ s with size at most *num_instantiations*.
        """


@attrs(slots=True, frozen=True)
class SituationTemplateObject:
    """
    An object in a situation template.

    Every object has a string *handle* which is used to name it for debugging purposes only
    """

    _handle: str = attrib(validator=instance_of(str))


@attrs(slots=True, frozen=True)
class SituationTemplatePropertyConstraint:
    obj: SituationTemplateObject = attrib(validator=instance_of(SituationTemplateObject))
    property: OntologyProperty = attrib(validator=instance_of(OntologyProperty))


@attrs(slots=True, frozen=True)
class SimpleSituationTemplate:
    objects: ImmutableSet[SituationTemplateObject] = attrib(converter=_to_immutableset)
    objects_to_properties: ImmutableSetMultiDict[
        SituationTemplateObject, OntologyProperty
    ] = attrib(converter=_to_immutablesetmultidict)
    objects_to_ontology_types: ImmutableDict[
        SituationTemplateObject, OntologyNode
    ] = attrib(converter=_to_immutabledict)

    @attrs(frozen=True, slots=True)
    class Builder:
        objects: List[SituationTemplateObject] = attrib(init=False, default=Factory(list))
        objects_to_properties: List[
            Tuple[SituationTemplateObject, OntologyProperty]
        ] = attrib(init=False, default=Factory(list))
        objects_to_ontology_types: List[
            Tuple[SituationTemplateObject, OntologyNode]
        ] = attrib(init=False, default=Factory(list))

        def object(
            self,
            handle: str,
            ontology_type: OntologyNode,
            properties: Sequence[OntologyProperty] = (),
        ) -> SituationTemplateObject:
            obj = SituationTemplateObject(handle)
            self.objects.append(obj)

            self.objects_to_ontology_types.append((obj, ontology_type))

            for _property in properties:
                self.objects_to_properties.append((obj, _property))

            return obj

        def build(self) -> "SimpleSituationTemplate":
            return SimpleSituationTemplate(
                self.objects, self.objects_to_properties, self.objects_to_ontology_types
            )


# user can make object variable placeholder objects for each object in the scene
# user can assert the object bear certain properties
# user can assert certain relations must hold between the objects
# user can make action variables and assert they obey certain properties
# user can specify how object variables are related to actions
