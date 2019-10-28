"""
Structures for describing situations in the world at an abstracted, human-friendly level.
"""
from abc import ABC
from typing import Generic, List, Mapping, Optional, TypeVar, Union, Iterable, Dict

from more_itertools import first, only
from vistautils.preconditions import check_arg

from adam.axis import GeonAxis
from adam.ontology.action_description import ActionDescriptionVariable
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.ontology.structural_schema import ObjectStructuralSchema
from attr import attrib, attrs
from attr.validators import instance_of, optional, deep_mapping
from immutablecollections import (
    ImmutableDict,
    ImmutableSet,
    ImmutableSetMultiDict,
    immutabledict,
    immutableset,
    immutablesetmultidict,
)

# noinspection PyProtectedMember
from immutablecollections.converter_utils import (
    _to_immutabledict,
    _to_immutableset,
    _to_immutablesetmultidict,
)

from adam.axes import Axes, HasAxes, WORLD_AXES
from adam.math_3d import Point
from adam.ontology import OntologyNode, IS_SUBSTANCE
from adam.ontology.during import DuringAction
from adam.ontology.phase1_spatial_relations import Region


class Situation(ABC):
    r"""
    A situation is a high-level representation of a configuration of objects, possibly including
    changes in the states of objects across time.

    An example situation might represent
    a person holding a toy truck and then putting it on a table.

    A curriculum is a sequence of `Situation`\ s.

    Situations are a high-level description intended to make it easy for human beings to specify
    curricula.

    Situations will be transformed into pairs of `PerceptualRepresentation`\ s and
    `LinguisticDescription`\ s for input to a `LanguageLearner`
    by `PerceptualRepresentationGenerator`\ s and `LanguageGenerator`\ s, respectively.
    """


@attrs(frozen=True)
class BagOfFeaturesSituationRepresentation(Situation):
    r"""
    Represents a `Situation` as an unstructured set of features.

    For testing purposes only.
    """
    features: ImmutableSet[str] = attrib(converter=immutableset, default=immutableset())
    """
    The set of string features which describes this situation.
    """


@attrs(frozen=True, slots=True, hash=None, eq=False, repr=False)
class SituationObject(HasAxes):
    r"""
    An object present in some situation.

    Every object must refer to an `OntologyNode` linking it to a type in an ontology.

    Unlike most of our classes, `SituationObject` has *id*-based hashing and equality.  This is
    because two objects with identical properties are nonetheless distinct.

    `SituationObject`\ should not be directly instantiated.
    Instead use `instantiate_ontology_node`.
    """

    ontology_node: OntologyNode = attrib(
        validator=instance_of(OntologyNode)  # , default=None
    )
    """
    The `OntologyNode` specifying the type of thing this object is.
    """
    # note to readers: the two axis-related fields have to be placed first to make PyCharm happy,
    # but you should read the other fields first.
    axes: Axes = attrib(validator=instance_of(Axes), kw_only=True)
    schema_axis_to_object_axis: Mapping[GeonAxis, GeonAxis] = attrib(
        validator=deep_mapping(instance_of(GeonAxis), instance_of(GeonAxis)),
        kw_only=True,
        converter=_to_immutabledict,
    )
    """
    Provides a mapping between the axes of an `ObjectStructuralSchema` 
    (which are abstract and generic - i.e. the axes of "tire"s in general)
    and the concrete instantiations of those axes which are stored in the *axes* field
    of this object.
    We need to track this information to keep the object axes in sync with the `Geon` axes
    during perceptual generation.

    Note that this mapping may be empty if this situation object was not derived from 
    an `ObjectStructuralSchema`.

    Rather than setting this field by hand, we recommend using the static factory method
    `from_structural_schema`.
    """
    properties: ImmutableSet[OntologyNode] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    r"""
    The `OntologyNode`\ s representing the properties this object has.
    """
    debug_handle: str = attrib(validator=instance_of(str), kw_only=True)

    def __attrs_post_init__(self) -> None:
        # disabled warning below is due to a PyCharm bug
        # noinspection PyTypeChecker
        for property_ in self.properties:
            if not isinstance(property_, OntologyNode):
                raise ValueError(
                    f"Situation object property {property_} is not an " f"OntologyNode"
                )
        for concrete_axis in self.schema_axis_to_object_axis.values():
            check_arg(concrete_axis in self.axes.all_axes)

    @debug_handle.default
    def _default_debug_handle(self) -> str:
        return f"{self.ontology_node.handle}"

    def __repr__(self) -> str:
        if self.properties:
            additional_properties = ", ".join(repr(prop) for prop in self.properties)
            additional_properties_string = f"[{additional_properties}]"
        else:
            additional_properties_string = ""

        if self.ontology_node and not self.debug_handle.startswith(
            self.ontology_node.handle
        ):
            handle_string = f"[{self.ontology_node.handle}]"
        else:
            handle_string = ""

        return f"{self.debug_handle}{handle_string}{additional_properties_string}"

    @staticmethod
    def instantiate_ontology_node(
        ontology_node: OntologyNode,
        *,
        properties: Iterable[OntologyNode] = immutableset(),
        debug_handle: Optional[str] = None,
        ontology: Ontology,
    ) -> "SituationObject":
        """
        Make a `SituationObject` from the object type *ontology_node*
        with properties *properties*.
        The *properties* and ontology node must all come from *ontology*.
        """
        schema_axis_to_object_axis: Mapping[GeonAxis, GeonAxis]
        if ontology.has_property(ontology_node, IS_SUBSTANCE):
            # it's not clear what the object_concrete_axes should be for substances,
            # so we just use the world object_concrete_axes for now
            schema_axis_to_object_axis = immutabledict()
            object_concrete_axes = WORLD_AXES
        else:
            structural_schemata = ontology.structural_schemata(ontology_node)
            if not structural_schemata:
                raise RuntimeError(f"No structural schema found for {ontology_node}")
            if len(structural_schemata) > 1:
                raise RuntimeError(
                    f"Multiple structural schemata available for {ontology_node}, "
                    f"please construct the SituationObject manually: "
                    f"{structural_schemata}"
                )
            schema_abstract_axes = only(structural_schemata).axes
            # The copy is needed or e.g. all tires of a truck
            # would share the same axis objects.
            object_concrete_axes = schema_abstract_axes.copy()
            schema_axis_to_object_axis = immutabledict(
                zip(schema_abstract_axes.all_axes, object_concrete_axes.all_axes)
            )

        return SituationObject(
            ontology_node=ontology_node,
            properties=properties,
            debug_handle=debug_handle if debug_handle else ontology_node.handle,
            axes=object_concrete_axes,
            schema_axis_to_object_axis=schema_axis_to_object_axis,
        )


SituationRegion = Region[SituationObject]  # pylint:disable=invalid-name


@attrs(frozen=True, slots=True)
class LocatedObjectSituation(Situation):
    r"""
    A representation of a `Situation` as a set of objects located at particular `Point`\ s.
    """

    objects_to_locations: Mapping[SituationObject, Point] = attrib(
        converter=_to_immutabledict, default=immutabledict()
    )
    r"""
    A mapping of `SituationObject`\ s to `Point`\ s giving their locations.
    """


_ActionTypeT = TypeVar("_ActionTypeT")
_ObjectT = TypeVar("_ObjectT")


@attrs(frozen=True, repr=False)
class Action(Generic[_ActionTypeT, _ObjectT]):
    r"""
    An action.

    This can be bound to `SituationObject` to represent actions in `Situation`\ s
    or to `TemplateObjectVariable`\ s to represent actions in situation templates.
    """
    action_type: _ActionTypeT = attrib()
    argument_roles_to_fillers: ImmutableSetMultiDict[
        OntologyNode, Union[_ObjectT, Region[_ObjectT]]
    ] = attrib(converter=_to_immutablesetmultidict, default=immutablesetmultidict())
    r"""
    A mapping of semantic roles (given as `OntologyNode`\ s) to their fillers.

    There may be multiple fillers for the same semantic role 
    (e.g. conjoined arguments).
    """
    # the optional below seems to confuse mypy?
    during: Optional[DuringAction[_ObjectT]] = attrib(  # type: ignore
        validator=optional(instance_of(DuringAction)), default=None, kw_only=True
    )
    auxiliary_variable_bindings: ImmutableDict[
        ActionDescriptionVariable, _ObjectT
    ] = attrib(converter=_to_immutabledict, default=immutabledict(), kw_only=True)
    """
    A mapping of action variables from *action_type*'s `ActionDescription`
    to the items which should fill them.
    """

    def accumulate_referenced_objects(self, object_accumulator: List[_ObjectT]) -> None:
        for (_, filler) in self.argument_roles_to_fillers.items():
            if isinstance(filler, Region):
                filler.accumulate_referenced_objects(object_accumulator)
            else:
                object_accumulator.append(filler)
        if self.during:
            self.during.accumulate_referenced_objects(object_accumulator)
        for aux_var_binding in self.auxiliary_variable_bindings.values():
            if isinstance(aux_var_binding, Region):
                aux_var_binding.accumulate_referenced_objects(object_accumulator)
            else:
                object_accumulator.append(aux_var_binding)

    def __repr__(self) -> str:
        parts = [str(self.argument_roles_to_fillers)]
        if self.during:
            parts.append(f"during={self.during}")
        if self.auxiliary_variable_bindings:
            parts.append(
                f"auxiliary_variable_bindings={self.auxiliary_variable_bindings}"
            )
        return f"{self.action_type}({', '.join(parts)})"
