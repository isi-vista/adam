r"""
Tools for working with situation templates, which allow a human to compactly describe a large
number of possible situations.

Note that we provide convenience functions like `random_situation_templates` and
`one_situation_per_template` to assist in supplying `Situation`\ s to `Experiment`\ s.
"""
import sys
from abc import ABC, abstractmethod
from typing import Generic, Iterable, List, Sequence, Tuple, TypeVar

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import (
    ImmutableDict,
    ImmutableSet,
    ImmutableSetMultiDict,
    immutabledict,
    immutableset,
)

# noinspection PyProtectedMember
from immutablecollections.converter_utils import (
    _to_immutabledict,
    _to_immutableset,
    _to_immutablesetmultidict,
)
from more_itertools import take

from adam.language.language_generator import SituationT
from adam.math_3d import Point
from adam.ontology import OntologyNode
from adam.ontology.ontology import Ontology
from adam.random_utils import RandomChooser, SequenceChooser
from adam.situation import LocatedObjectSituation, SituationObject


class SituationTemplate(ABC):
    r"""
    A compact description for a large number of `Situation`\ s.
    """


_SituationTemplateT = TypeVar("_SituationTemplateT", bound=SituationTemplate)


class SituationTemplateProcessor(ABC, Generic[_SituationTemplateT, SituationT]):
    r"""
    Turns a `SituationTemplate` into one or more `Situation`\ s.
    """

    @abstractmethod
    def generate_situations(
        self,
        template: _SituationTemplateT,
        *,
        chooser: SequenceChooser = Factory(RandomChooser.for_seed),
    ) -> Iterable[SituationT]:
        r"""
        Generates one or more `Situation`\ s from a `SituationTemplate`\ .

        The behavior of this method should be deterministic conditional upon
        an identically initialized and deterministic `SequenceChooser` being supplied.

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
    An object in a `SituationTemplate`.

    For example, a particular ball or person.
    """

    handle: str = attrib(validator=instance_of(str))
    """
    Every object has a string *handle* which is used to name it for debugging purposes only
    """


@attrs(slots=True, frozen=True)
class SimpleSituationTemplate(SituationTemplate):
    """
    A minimal implementation of a situation template for objects only (no actions or relations).

    It is usually easiest to create a `SimpleSituationTemplate` using
    `SimpleSituationTemplate.Builder` .
    """

    objects: ImmutableSet[SituationTemplateObject] = attrib(converter=_to_immutableset)
    r"""
    The `SituationObject`\ s present in the `Situation`, e.g. {a person, a ball, a table}
    """
    objects_to_required_properties: ImmutableSetMultiDict[
        SituationTemplateObject, OntologyNode
    ] = attrib(converter=_to_immutablesetmultidict)
    r"""
    A mapping of each `SituationObject` in the `Situation` to `OntologyNode`\ s it is
    required to have.
    """
    objects_to_ontology_types: ImmutableDict[
        SituationTemplateObject, OntologyNode
    ] = attrib(converter=_to_immutabledict)
    r"""
    A mapping of each `SituationObject` in the `Situation` to the `OntologyNode`\ s its type
    must correspond to either directly or by inheritance.
    """

    @attrs(frozen=True, slots=True)
    class Builder:
        """
        The preferred means of creating a `SimpleSituationTemplate`
        """

        _objects: List[SituationTemplateObject] = attrib(
            init=False, default=Factory(list)
        )
        _objects_to_properties: List[
            Tuple[SituationTemplateObject, OntologyNode]
        ] = attrib(init=False, default=Factory(list))
        _objects_to_ontology_types: List[
            Tuple[SituationTemplateObject, OntologyNode]
        ] = attrib(init=False, default=Factory(list))

        def object_variable(
            self,
            handle: str,
            ontology_type: OntologyNode,
            properties: Sequence[OntologyNode] = (),
        ) -> SituationTemplateObject:
            r"""
            Add an object to a `SimpleSituationTemplate` being built.

            Args:
                handle: the debugging handle of the object
                ontology_type: the `OntologyNode` which any object filling this slot must match.
                properties: the `OntologyNode`\ s any object filling this slot must have.

            Returns:
                the `SituationTemplateObject` which was created.
            """
            obj = SituationTemplateObject(handle)
            self._objects.append(obj)

            self._objects_to_ontology_types.append((obj, ontology_type))

            for _property in properties:
                self._objects_to_properties.append((obj, _property))

            return obj

        def build(self) -> "SimpleSituationTemplate":
            return SimpleSituationTemplate(
                self._objects,
                self._objects_to_properties,
                self._objects_to_ontology_types,
            )


@attrs(frozen=True, slots=True)
class SimpleSituationTemplateProcessor(
    SituationTemplateProcessor[SimpleSituationTemplate, LocatedObjectSituation]
):
    """
    A trivial situation template processor for testing use.

    This cannot handle anything in situation templates except object variables.  This object
    variables are instantiated with random compatible objects from the provided
    `Ontology` ; they are positioned in a line one meter apart.
    """

    _ontology: Ontology = attrib(validator=instance_of(Ontology))

    def generate_situations(
        self,
        template: SimpleSituationTemplate,
        *,
        num_instantiations: int = 1,
        chooser: SequenceChooser = Factory(RandomChooser.for_seed),
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
        required_properties = template.objects_to_required_properties[template_object]

        compatible_ontology_types = self._ontology.nodes_with_properties(
            root_node=object_supertype, required_properties=required_properties
        )

        if compatible_ontology_types:
            ontology_node = chooser.choice(compatible_ontology_types)
            return SituationObject(
                ontology_node,
                properties=self._ontology.properties_for_node(ontology_node),
                debug_handle=template_object.handle,
            )
        else:
            raise RuntimeError(
                f"When attempting to instantiate object {template_object} in"
                f" {template}: no node at or under {object_supertype} with "
                f"properties {required_properties} exists in the ontology "
                f"{self._ontology}"
            )

    @staticmethod
    def _locations_in_a_line_1m_apart():
        for x_coordinate in range(0, sys.maxsize):
            yield Point(float(x_coordinate), 0.0, 0.0)


def random_situation_templates(
    situation_templates: Sequence[SituationTemplate],
    num_random_templates: int,
    sequence_chooser: SequenceChooser,
) -> Iterable[SituationTemplate]:
    r"""
    Get *num_random_templates* random (with replacement) `SituationTemplate`\ s
    from *situation_templates*, where "random" choices are given by
    the provided `SequenceChooser`.
    """
    return (
        sequence_chooser.choice(situation_templates)
        for _ in range(0, num_random_templates)
    )


def one_situation_per_template(
    situation_templates: Sequence[_SituationTemplateT],
    situation_template_processor: SituationTemplateProcessor[
        _SituationTemplateT, SituationT
    ],
    sequence_chooser: SequenceChooser,
) -> Iterable[SituationT]:
    return (
        take(
            1,
            situation_template_processor.generate_situations(
                template=situation_template, chooser=sequence_chooser
            ),
        )
        for situation_template in situation_templates
    )
