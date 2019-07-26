"""
Tools for working with situation templates, which allow a human to compactly describe a large
number of possible situations.
"""
import random
import sys
from abc import ABC, abstractmethod
from typing import AbstractSet, Generic, List, Sequence, Tuple, TypeVar

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import (
    ImmutableDict,
    ImmutableSet,
    ImmutableSetMultiDict,
    immutableset,
    immutabledict,
)
from immutablecollections.converter_utils import (
    _to_immutabledict,
    _to_immutableset,
    _to_immutablesetmultidict,
)

from adam import ontology
from adam.random_utils import SequenceChooser, RandomChooser
from adam.math_3d import Point
from adam.ontology import OntologyNode, OntologyProperty, Ontology
from adam.situation import Situation, LocatedObjectSituation, SituationObject


class SituationTemplate(ABC):
    """
    A compact description for a large number of situations.
    """


_SituationTemplateT = TypeVar("_SituationTemplateT", bound=SituationTemplate)


def _fixed_random_factory() -> SequenceChooser:
    ret = random.Random()
    ret.seed(0)
    return RandomChooser(ret)


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
        chooser: SequenceChooser = Factory(_fixed_random_factory),
    ) -> AbstractSet[Situation]:
        r"""
        Generates one or more `Situation`\ s from a `SituationTemplate`\ .

        The behavior of this method should be deterministic conditional upon
        an identically initialized ::class`random.Random` being supplied.

        Args:
            template: the template to instantiate
            num_instantiations: the number of instantiations requested
            chooser: the means of making any random selections the generator may need.

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
class SimpleSituationTemplate(SituationTemplate):
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


@attrs(frozen=True, slots=True)
class SimpleSituationTemplateProcessor(
    SituationTemplateProcessor[SimpleSituationTemplate]
):
    """
    A trivial situation template processor for testing use.

    This cannot handle anything in situation templates except object variables.  This object
    variables are instantiated with random compatible objects from the provided `Ontology` ;
    they are positioned in a line one meter apart.
    """

    _ontology: Ontology = attrib(validator=instance_of(Ontology))

    def generate_situations(
        self,
        template: SimpleSituationTemplate,
        *,
        num_instantiations: int = 1,
        chooser: SequenceChooser = Factory(_fixed_random_factory),
    ) -> ImmutableSet[LocatedObjectSituation]:
        assert num_instantiations >= 1

        instantiations = []

        for _ in range(num_instantiations):
            instantiated_objects = immutableset(
                self._instantiate_object(obj, template, chooser)
                for obj in template.objects
            )
            instantiations.append(
                LocatedObjectSituation(
                    immutabledict(
                        zip(
                            instantiated_objects,
                            SimpleSituationTemplateProcessor._locations_in_a_line_1m_apart(),
                        )
                    )
                )
            )
        return immutableset(instantiations)

    def _instantiate_object(
        self,
        template_object: SituationTemplateObject,
        template: SimpleSituationTemplate,
        chooser: SequenceChooser,
    ) -> SituationObject:
        object_supertype = template.objects_to_ontology_types[template_object]
        required_properties = template.objects_to_properties[template_object]

        compatible_ontology_types = self._ontology.nodes_with_properties(
            superclass=object_supertype, required_properties=required_properties
        )

        if compatible_ontology_types:
            ontology_node = chooser.choice(compatible_ontology_types)
            return SituationObject(
                ontology_node.handle, self._ontology.properties_for_node(ontology_node)
            )
        else:
            raise RuntimeError(
                f"When attempting to instantiate object {template_object} in"
                f" {template}: no node at or under {object_supertype} with "
                f"properties {required_properties} exists in the ontology "
                f"{ontology}"
            )

    @staticmethod
    def _locations_in_a_line_1m_apart():
        for x_coordinate in range(0, sys.maxsize):
            yield Point(float(x_coordinate), 0.0, 0.0)
