r"""
Our strategy for `SituationTemplate`\ s in Phase 1 of ADAM.
"""
import random
from _random import Random
from abc import ABC, abstractmethod
from itertools import product
from typing import AbstractSet, Iterable, Mapping, Sequence, TypeVar, Union

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableDict, ImmutableSet, immutabledict, immutableset
from immutablecollections.converter_utils import _to_immutabledict, _to_immutableset
from more_itertools import take
from typing_extensions import Protocol
from vistautils.preconditions import check_arg

from adam.ontology import CAN_FILL_TEMPLATE_SLOT, OntologyNode, PROPERTY, THING
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import COLOR, GAILA_PHASE_1_ONTOLOGY, LEARNER, GROUND
from adam.ontology.selectors import ByHierarchyAndProperties, Is, OntologyNodeSelector
from adam.random_utils import RandomChooser, SequenceChooser
from adam.relation import Relation
from adam.situation import SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates import (
    SituationTemplate,
    SituationTemplateObject,
    SituationTemplateProcessor,
)


@attrs(frozen=True, slots=True)
class Phase1SituationTemplate(SituationTemplate):
    r"""
    The `SituationTemplate` implementation used in Phase 1 of the ADAM project.

    Currently, this can only be a collection of `TemplateObjectVariable`\ s.

    Phase1SituationTemplateGenerator will translate these
    to a sequence `HighLevelSemanticsSituation`\ s corresponding
    to the Cartesian product of the possible values of the *object_variables*.

    Beware that this can be very large if the number of object variables
    or the number of possible values of the variables is even moderately large.
    """
    object_variables: ImmutableSet["TemplateObjectVariable"] = attrib(
        converter=_to_immutableset
    )
    asserted_relations: ImmutableSet[Relation["TemplateObjectVariable"]] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    """
    This are relations we assert to hold true in the situation.
    This should be used to specify additional relations
    which cannot be deduced from the types of the objects alone.
    """

    def __attrs_post_init__(self) -> None:
        check_arg(self.object_variables, "A situation must contain at least one object")


def all_possible(
    situation_template: Phase1SituationTemplate,
    *,
    ontology: Ontology,
    chooser: SequenceChooser
) -> Iterable[HighLevelSemanticsSituation]:
    """
    Generator for all possible instantiations of *situation_template* with *ontology*.
    """
    return _Phase1SituationTemplateGenerator(
        ontology=ontology, variable_assigner=_CrossProductVariableAssigner()
    ).generate_situations(situation_template, chooser=chooser)


def sampled(
    situation_template: Phase1SituationTemplate,
    *,
    ontology: Ontology,
    chooser: SequenceChooser,
    max_to_sample: int
) -> Iterable[HighLevelSemanticsSituation]:
    """
    Gets *max_to_sample* instantiations of *situation_template* with *ontology*
    """
    check_arg(max_to_sample > 0)
    return take(
        max_to_sample,
        _Phase1SituationTemplateGenerator(
            ontology=ontology, variable_assigner=_SamplingVariableAssigner()
        ).generate_situations(situation_template, chooser=chooser),
    )


@attrs(frozen=True, slots=True)
class _Phase1SituationTemplateGenerator(
    SituationTemplateProcessor[Phase1SituationTemplate, HighLevelSemanticsSituation]
):
    r"""
    Generates `HighLevelSemanticsSituation`\ s from `Phase1SituationTemplate`\ s
    by sampling a valid filler for each object variable.

    This can potentially generate an infinite stream of `Situation`\ s,
    so be sure to wrap this in more_itertools.take .
    """
    _variable_assigner: "_VariableAssigner" = attrib(kw_only=True)
    # can be set to something besides GAILA_PHASE_1_ONTOLOGY for testing purposes
    ontology: Ontology = attrib(default=GAILA_PHASE_1_ONTOLOGY, kw_only=True)

    def generate_situations(
        self,
        template: Phase1SituationTemplate,
        *,
        chooser: SequenceChooser = Factory(
            RandomChooser.for_seed
        )  # pylint:disable=unused-argument
    ) -> Iterable[HighLevelSemanticsSituation]:
        # gather property variables from object variables
        property_variables = immutableset(
            property_
            for obj_var in template.object_variables
            for property_ in obj_var.asserted_properties
            if isinstance(property_, TemplatePropertyVariable)
        )

        for variable_assignment in self._variable_assigner.variable_assignments(
            ontology=self.ontology,
            object_variables=template.object_variables,
            property_variables=property_variables,
            chooser=chooser,
        ):
            yield HighLevelSemanticsSituation(
                ontology=self.ontology,
                objects=[
                    # instantiate all objects in the situation according to the variable assignment.
                    SituationObject(
                        ontology_node=variable_assignment.object_variables_to_fillers[
                            obj_var
                        ],
                        properties=[
                            # instantiate any property variables associated with this object
                            variable_assignment.property_variables_to_fillers[prop_var]
                            for prop_var in obj_var.asserted_properties
                        ],
                    )
                    for obj_var in template.object_variables
                ],
                relations=[relation.copy_remapping_objects(variable_assignment.object_variables_to_fillers) for relation in
                           template.asserted_relations]
            )


class _TemplateVariable(Protocol):
    """
    This is not for public use; use `object_variable` and `property_variable` instead.
    """

    node_selector: OntologyNodeSelector


@attrs(frozen=True, slots=True, cmp=False)
class TemplateObjectVariable(SituationTemplateObject, _TemplateVariable):
    r"""
    A variable in a `Phase1SituationTemplate`
    which could be filled by any object
    whose `OntologyNode` is selected by *node_selector*.

    *asserted_properties* allows you to specify what properties
    should be asserted for this object in generated `Situation`\ s.
    This is for specifying properties which are not intrinsic to an object
    (e.g. if your object variable is constrained to be a sub-type of person
    you don't need to and shouldn't specify *ANIMATE* as an asserted property)
    and are not used to filter what object can fill this variable.
    For example, if you wanted to specify that whatever fills this variable,
    you want to make it red in this situation, you would specify *RED*
    in *asserted_properties*.

    We provide `object_variable` to make creating `TemplateObjectVariable`\ s more convenient.

    `TemplateObjectVariable`\ s with the same node selector are *not* equal to one another
    so that you can have multiple objects in a `Situation` which obey the same constraints.
    """

    node_selector: OntologyNodeSelector = attrib(
        validator=instance_of(OntologyNodeSelector)
    )
    asserted_properties: ImmutableSet["TemplatePropertyVariable"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )


@attrs(frozen=True, slots=True, cmp=False)
class TemplatePropertyVariable(SituationTemplateObject, _TemplateVariable):
    r"""
    A variable in a `Phase1SituationTemplate`
    which could be filled by any property
    whose `OntologyNode` is selected by *node_selector*.

    We provide `property_variable` to make creating `TemplatePropertyVariable`\ s more convenient.

    `TemplatePropertyVariable`\ s with the same node selector are *not* equal to one another
    so that you can have multiple objects in a `Situation` which obey the same constraints.
    """

    node_selector: OntologyNodeSelector = attrib(
        validator=instance_of(OntologyNodeSelector)
    )


def object_variable(
    debug_handle: str,
    root_node: OntologyNode = THING,
    *,
    required_properties: Iterable[OntologyNode] = immutableset(),
    added_properties: Iterable[
        Union[OntologyNode, TemplatePropertyVariable]
    ] = immutableset()
) -> TemplateObjectVariable:
    r"""
    Create a `TemplateObjectVariable` with the specified *debug_handle*
    which can be filled by any object whose `OntologyNode` is a descendant of
    (or is exactly) *root_node*
    and which possesses all properties in *required_properties*.

    Additionally, the template will add all properties in *added_properties*
    to the object.
    Use *required_properties* for things like
    "anything filling this variable should be animate."
    Use *added_properties* for things like
    "whatever fills this variable, make it red."
    """
    real_required_properties = list(required_properties)
    if root_node != LEARNER and root_node != GROUND:
        # the learner and the ground are special cases of things we want to be
        # explicitly instantiable but not instantiable by variable.
        real_required_properties.append(CAN_FILL_TEMPLATE_SLOT)

    return TemplateObjectVariable(
        debug_handle,
        ByHierarchyAndProperties(
            descendents_of=root_node, required_properties=real_required_properties
        ),
        asserted_properties=[
            property_
            if isinstance(property_, TemplatePropertyVariable)
            else Is([property_])
            for property_ in added_properties
        ],
    )


def property_variable(
    debug_handle: str,
    root_node: OntologyNode = PROPERTY,
    with_meta_properties: Iterable[OntologyNode] = immutableset(),
) -> TemplatePropertyVariable:
    r"""
    Create a `TemplatePropertyVariable` with the specified *debug_handle*
    which can be filled by any property whose `OntologyNode` is a descendant of
    (or is exactly) *root_node*
    and which possesses all properties in *with_properties*.
    """
    real_required_properties = list(with_meta_properties)
    real_required_properties.append(CAN_FILL_TEMPLATE_SLOT)

    return TemplatePropertyVariable(
        debug_handle,
        ByHierarchyAndProperties(
            descendents_of=root_node, required_properties=real_required_properties
        ),
    )


def color_variable(debug_handle: str) -> TemplatePropertyVariable:
    r"""
    Create a `TemplatePropertyVariable` with the specified *debug_handle*
    which ranges over all colors in the ontology.
    """
    return property_variable(debug_handle, COLOR)


@attrs(frozen=True, slots=True)
class _VariableAssignment:
    """
    An assignment of ontology types to object and property variables in a situation.
    """

    object_variables_to_fillers: ImmutableDict[
        "TemplateObjectVariable", OntologyNode
    ] = attrib(converter=_to_immutabledict, default=immutabledict())
    property_variables_to_fillers: ImmutableDict[
        "TemplatePropertyVariable", OntologyNode
    ] = attrib(converter=_to_immutabledict, default=immutabledict())


class _VariableAssigner(ABC):
    @abstractmethod
    def variable_assignments(
        self,
        *,
        ontology: Ontology,
        object_variables: AbstractSet["TemplateObjectVariable"],
        property_variables: AbstractSet["TemplatePropertyVariable"],
        chooser: SequenceChooser
    ) -> Iterable[_VariableAssignment]:
        r"""
        Produce a (potentially infinite) stream of `_VariableAssignment`\ s of nodes from *ontology*
        to the given object and property variables.
        """


_VarT = TypeVar("_VarT", bound=_TemplateVariable)


class _CrossProductVariableAssigner(_VariableAssigner):
    """
    Iterates over all possible assignments to the given variables.
    """

    def variable_assignments(
        self,
        *,
        ontology: Ontology,
        object_variables: AbstractSet["TemplateObjectVariable"],
        property_variables: AbstractSet["TemplatePropertyVariable"],
        chooser: SequenceChooser  # pylint: disable=unused-argument
    ) -> Iterable[_VariableAssignment]:
        # TODO: fix hard-coded rng
        # https://github.com/isi-vista/adam/issues/123
        rng = Random()
        rng.seed(0)

        for object_combination in self._all_combinations(
            object_variables, ontology=ontology, rng=rng
        ):
            for property_combination in self._all_combinations(
                property_variables, ontology=ontology, rng=rng
            ):
                yield _VariableAssignment(
                    object_variables_to_fillers=object_combination,
                    property_variables_to_fillers=property_combination,
                )

    def _all_combinations(
        self, variables: AbstractSet[_VarT], *, ontology: Ontology, rng: Random
    ) -> Iterable[Mapping[_VarT, OntologyNode]]:
        var_to_options = {
            # tuple() needed to make it hashable
            var: tuple(
                _shuffled(
                    var.node_selector.select_nodes(
                        ontology, require_non_empty_result=True
                    ),
                    rng,
                )
            )
            for var in variables
        }

        if var_to_options:
            for combination in product(*var_to_options.values()):
                # this makes a dictionary where the keys are the variables and the values
                # correspond to one of the possible assignments.
                yield immutabledict(zip(var_to_options.keys(), combination))

        else:
            # in this case, there are no variables, so the only possible assignment
            # is the empty assignment.
            yield dict()


class _SamplingVariableAssigner(_VariableAssigner):
    """
    Provides an infinite stream of variable assignments
    where each variable is randomly sampled from its possible values.
    """

    def variable_assignments(
        self,
        *,
        ontology: Ontology,
        object_variables: AbstractSet["TemplateObjectVariable"],
        property_variables: AbstractSet["TemplatePropertyVariable"],
        chooser: SequenceChooser
    ) -> Iterable[_VariableAssignment]:
        for object_combination in self._sample_combinations(
            object_variables, ontology=ontology, chooser=chooser
        ):
            for property_combination in self._sample_combinations(
                property_variables, ontology=ontology, chooser=chooser
            ):
                yield _VariableAssignment(
                    object_variables_to_fillers=object_combination,
                    property_variables_to_fillers=property_combination,
                )

    def _sample_combinations(
        self,
        variables: AbstractSet[_VarT],
        *,
        ontology: Ontology,
        chooser: SequenceChooser
    ) -> Iterable[Mapping[_VarT, OntologyNode]]:
        var_to_options = {
            # beware - the values in this map are infinite generators!
            var: _samples(
                var.node_selector.select_nodes(ontology, require_non_empty_result=True),
                chooser,
            )
            for var in variables
        }

        if var_to_options:
            for combination in zip(*var_to_options.values()):
                #  this makes a dictionary where the keys are the variables and the values
                # correspond to one of the possible assignments.
                yield immutabledict(zip(var_to_options.keys(), combination))
        else:
            while True:
                # if there are no variables to assign, the only possible assignment
                # is the empty assignment
                yield dict()


_T = TypeVar("_T")


def _shuffled(items: Iterable[_T], rng: Random) -> Sequence[_T]:
    """
    Return the elements of *items* in shuffled order,
    using *rng* as the source of randomness.

    This should eventually get shifted to VistaUtils.
    """
    items_list = list(items)
    random.shuffle(items_list, rng.random)
    return items_list


def _samples(items: Iterable[_T], chooser: SequenceChooser) -> Iterable[_T]:
    """
    Return an infinite stream of samples (with replacement) from *items*
    using *chooser* as the source of randomness.

    This should eventually get shifted to VistaUtils.
    """
    items_list = list(items)
    while True:
        yield chooser.choice(items_list)
