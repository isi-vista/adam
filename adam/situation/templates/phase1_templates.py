r"""
Our strategy for `SituationTemplate`\ s in Phase 1 of ADAM.
"""
import random
from _random import Random
from abc import ABC, abstractmethod
from collections import Counter
from itertools import chain, product
from typing import (
    AbstractSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
    Tuple,
)

from attr.validators import instance_of
from immutablecollections import ImmutableDict, ImmutableSet, immutabledict, immutableset
from immutablecollections.converter_utils import _to_immutabledict, _to_immutableset
from more_itertools import only, take
from typing_extensions import Protocol
from vistautils.preconditions import check_arg

from adam.axes import AxesInfo, HorizontalAxisOfObject
from adam.ontology import (
    ACTION,
    CAN_FILL_TEMPLATE_SLOT,
    IS_ADDRESSEE,
    OntologyNode,
    PROPERTY,
    THING,
)
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    COLOR,
    GAILA_PHASE_1_ONTOLOGY,
    GROUND,
    LEARNER,
    TRANSPARENT,
    is_recognized_particular,
)
from adam.ontology.phase1_spatial_relations import Region
from adam.ontology.selectors import (
    AndOntologySelector,
    ByHierarchyAndProperties,
    FilterOut,
    OntologyNodeSelector,
    SubcategorizationSelector,
)
from adam.random_utils import RandomChooser, SequenceChooser
from adam.relation import Relation, flatten_relations
from adam.situation import Action, SituationObject, SituationRegion
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates import (
    SituationTemplate,
    SituationTemplateObject,
    SituationTemplateProcessor,
)
from attr import Factory, attrib, attrs

_ExplicitOrVariableActionType = Union[OntologyNode, "TemplateActionTypeVariable"]


class _TemplateVariable(Protocol):
    """
    This is not for public use; use `object_variable` and `property_variable` instead.
    """

    node_selector: OntologyNodeSelector


@attrs(frozen=True, slots=True, eq=False, repr=False)
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
    asserted_properties: ImmutableSet[
        Union[OntologyNode, "TemplatePropertyVariable"]
    ] = attrib(converter=_to_immutableset, default=immutableset())

    def __repr__(self) -> str:
        props: List[str] = []
        props.append(str(self.node_selector))
        props.extend(f"assert({str(prop_var)})" for prop_var in self.asserted_properties)
        return f"{self.handle}[{' ,'.join(props)}]"


TemplateRegion = Region[TemplateObjectVariable]  # pylint:disable=invalid-name


@attrs(frozen=True, slots=True, eq=False)
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


@attrs(frozen=True, slots=True, eq=False)
class TemplateActionTypeVariable(SituationTemplateObject, _TemplateVariable):
    r"""
    A variable in a `Phase1SituationTemplate`
    which could be filled by any action type
    whose `OntologyNode` is selected by *node_selector*.

    We provide `action_variable` to make creating `TemplateActionTypeVariable`\ s more convenient.
    """

    node_selector: OntologyNodeSelector = attrib(
        validator=instance_of(OntologyNodeSelector)
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
    name: str = attrib(validator=instance_of(str))
    salient_object_variables: ImmutableSet["TemplateObjectVariable"] = attrib(
        converter=_to_immutableset
    )
    background_object_variables: ImmutableSet["TemplateObjectVariable"] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    asserted_always_relations: ImmutableSet[Relation["TemplateObjectVariable"]] = attrib(
        converter=flatten_relations, default=immutableset(), kw_only=True
    )
    """
    This are relations we assert to hold true in the situation.
    This should be used to specify additional relations
    which cannot be deduced from the types of the objects alone.
    """
    constraining_relations: ImmutableSet[Relation["TemplateObjectVariable"]] = attrib(
        converter=flatten_relations, default=immutableset(), kw_only=True
    )
    """
    These are relations which we required to be true
    and are used in selecting assignments to object variables.
    Our ability to enforce these constraints efficiently is very limited,
    so don't make them too complex or constraining!
    """
    actions: ImmutableSet[
        Action[_ExplicitOrVariableActionType, "TemplateObjectVariable"]
    ] = attrib(converter=_to_immutableset, default=immutableset(), kw_only=True)
    syntax_hints: ImmutableSet[str] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    """
    A temporary hack to allow control of language generation decisions
    using the situation template language.
    
    See https://github.com/isi-vista/adam/issues/222 .
    """
    all_object_variables: ImmutableSet[TemplateObjectVariable] = attrib(init=False)
    r"""
    All `TemplateObjectVariable`s in the situation, 
    both salient and auxiliary to actions.
    """
    gazed_objects: ImmutableSet[TemplateObjectVariable] = attrib(
        converter=_to_immutableset, kw_only=True
    )
    """
    A set of `TemplateObjectVariable`s which are the focus of the speaker. 
    Defaults to all semantic role fillers of situation actions.
    """
    before_action_relations: ImmutableSet[Relation[TemplateObjectVariable]] = attrib(
        converter=flatten_relations, kw_only=True, default=immutableset()
    )
    """
    The relations which hold in this `SituationTemplate`,
    before, but not necessarily after, any actions which occur.
    
    It is not necessary to state every relationship which holds in a situation.
    Rather this should contain the salient relationships
    which should be expressed in the linguistic description.
    
    Do not specify those relations here which are *implied* by any actions which occur.
    Those are handled automatically. 
    """
    after_action_relations: ImmutableSet[Relation[TemplateObjectVariable]] = attrib(
        converter=flatten_relations, kw_only=True, default=immutableset()
    )
    """
    The relations which hold in this `SituationTemplate`,
    after, but not necessarily before, any actions which occur.

    It is not necessary to state every relationship which holds in a situation.
    Rather this should contain the salient relationships
    which should be expressed in the linguistic description.

    Do not specify those relations here which are *implied* by any actions which occur.
    Those are handled automatically. 
    """

    @property
    def has_relations(self):
        return (
            self.before_action_relations
            or self.after_action_relations
            or self.asserted_always_relations
        )

    def __attrs_post_init__(self) -> None:
        check_arg(
            self.salient_object_variables, "A situation must contain at least one object"
        )

        # ensure all objects referenced anywhere are on the object list
        objects_referenced_accumulator = list(self.salient_object_variables)
        for relation in chain(
            self.constraining_relations, self.asserted_always_relations
        ):
            relation.accumulate_referenced_objects(objects_referenced_accumulator)
        for action in self.actions:
            action.accumulate_referenced_objects(objects_referenced_accumulator)

        unique_objects_referenced = immutableset(objects_referenced_accumulator)
        missing_objects = unique_objects_referenced - self.all_object_variables
        if missing_objects:
            raise RuntimeError(
                f"Set of referenced objects {unique_objects_referenced} does not match "
                f"object variables {self.all_object_variables} for template {self}: "
                f"the following are missing {missing_objects}"
            )

    @all_object_variables.default
    def _init_all_object_variables(self) -> ImmutableSet[TemplateObjectVariable]:
        ret: List[TemplateObjectVariable] = []

        for action in self.actions:
            action.accumulate_referenced_objects(ret)

        for relation in chain(
            self.constraining_relations, self.asserted_always_relations
        ):
            relation.accumulate_referenced_objects(ret)

        ret.extend(self.salient_object_variables)
        ret.extend(self.background_object_variables)

        for obj_var in ret:
            if not isinstance(obj_var, TemplateObjectVariable):
                raise RuntimeError(
                    f"Got non-object variable {obj_var} in template {self}"
                )

        return immutableset(ret)

    @gazed_objects.default
    def _determine_gazed_objects(self):
        return immutableset(
            object_
            for action in self.actions
            for (_, object_) in action.argument_roles_to_fillers.items()
            if not isinstance(object_, Region)
        )


def all_possible(
    situation_template: Phase1SituationTemplate,
    *,
    ontology: Ontology,
    chooser: SequenceChooser,
    default_addressee_node: OntologyNode = LEARNER,
) -> Iterable[HighLevelSemanticsSituation]:
    """
    Generator for all possible instantiations of *situation_template* with *ontology*.
    """
    return list(
        _Phase1SituationTemplateGenerator(
            ontology=ontology, variable_assigner=_CrossProductVariableAssigner()
        ).generate_situations(
            situation_template,
            chooser=chooser,
            default_addressee_node=default_addressee_node,
        )
    )


def sampled(
    situation_template: Phase1SituationTemplate,
    *,
    ontology: Ontology,
    chooser: SequenceChooser,
    max_to_sample: int,
    default_addressee_node: OntologyNode = LEARNER,
    block_multiple_of_the_same_type: bool,
) -> Iterable[HighLevelSemanticsSituation]:
    """
    Gets *max_to_sample* instantiations of *situation_template* with *ontology*
    """
    check_arg(max_to_sample >= 0)
    return list(
        take(
            max_to_sample,
            _Phase1SituationTemplateGenerator(
                ontology=ontology,
                variable_assigner=_SamplingVariableAssigner(),
                block_multiple_objects_of_the_same_type=block_multiple_of_the_same_type,
            ).generate_situations(
                situation_template,
                chooser=chooser,
                default_addressee_node=default_addressee_node,
            ),
        )
    )


def fixed_assignment(
    situation_template: Phase1SituationTemplate,
    assignment: "TemplateVariableAssignment",
    *,
    ontology: Ontology,
    chooser: SequenceChooser,
    default_addressee_node: OntologyNode = LEARNER,
) -> Iterable[HighLevelSemanticsSituation]:
    return list(
        _Phase1SituationTemplateGenerator(
            ontology=ontology, variable_assigner=_FixedVariableAssigner(assignment)
        ).generate_situations(
            situation_template,
            chooser=chooser,
            default_addressee_node=default_addressee_node,
        )
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
    block_multiple_objects_of_the_same_type = attrib(
        default=True, validator=instance_of(bool), kw_only=True
    )

    def generate_situations(
        self,
        template: Phase1SituationTemplate,
        *,
        num_instantiations: int = 1,  # pylint: disable=unused-argument
        chooser: SequenceChooser = Factory(RandomChooser.for_seed),
        include_ground: bool = True,  # pylint: disable=unused-argument
        default_addressee_node: Optional[OntologyNode] = LEARNER,
    ) -> Iterable[HighLevelSemanticsSituation]:
        check_arg(isinstance(template, Phase1SituationTemplate))
        if default_addressee_node is None:
            default_addressee_node = LEARNER
        try:
            # gather property variables from object variables
            property_variables = immutableset(
                property_
                for obj_var in template.salient_object_variables
                for property_ in obj_var.asserted_properties
                if isinstance(property_, TemplatePropertyVariable)
            )

            action_type_variables = immutableset(
                action.action_type
                for action in template.actions
                if isinstance(action.action_type, TemplateActionTypeVariable)
            )

            failures_in_a_row = 0

            for variable_assignment in self._variable_assigner.variable_assignments(
                ontology=self.ontology,
                object_variables=template.all_object_variables,
                property_variables=property_variables,
                action_variables=action_type_variables,
                chooser=chooser,
            ):
                # instantiate all objects in the situation according to the variable assignment.
                object_var_to_instantiations = self._instantiate_objects(
                    template,
                    variable_assignment,
                    default_addressee_node=default_addressee_node,
                )

                # Cannot have multiple instantiations of the same recognized particular.
                # e.g. "Dad gave Dad a box"
                if self._has_multiple_recognized_particulars(
                    object_var_to_instantiations
                ):
                    continue
                if self.block_multiple_objects_of_the_same_type:
                    object_instantiations_ontology_nodes = [
                        object_instantiation.ontology_node
                        for object_instantiation in object_var_to_instantiations.values()
                    ]
                    if len(set(object_instantiations_ontology_nodes)) != len(
                        object_instantiations_ontology_nodes
                    ):
                        # There must be two objects of the same ontology type.
                        continue

                # use them to instantiate the entire situation
                situation = self._instantiate_situation(
                    template, variable_assignment, object_var_to_instantiations
                )
                if self._satisfies_constraints(
                    template, situation, object_var_to_instantiations
                ):
                    failures_in_a_row = 0
                    yield situation
                else:
                    failures_in_a_row += 1
                    if failures_in_a_row >= 250:
                        raise RuntimeError(
                            f"Failed to find a satisfying variable assignment "
                            f"for situation template constraints after "
                            f"{failures_in_a_row} consecutive attempts."
                            f"Try shifting constraints from relations to properties."
                        )
                    continue
        except Exception as e:
            raise RuntimeError(
                f"Exception while generating from situation template {template}"
            ) from e

    def _instantiate_objects(
        self,
        template: Phase1SituationTemplate,
        variable_assignment: "TemplateVariableAssignment",
        *,
        default_addressee_node: OntologyNode,
    ):
        has_addressee = any(
            IS_ADDRESSEE in object_.asserted_properties
            for object_ in template.all_object_variables
        )

        object_var_to_instantiations_mutable: List[
            Tuple[TemplateObjectVariable, SituationObject]
        ] = [
            (
                obj_var,
                self._instantiate_object(
                    obj_var,
                    variable_assignment,
                    has_addressee=has_addressee,
                    default_addressee_node=default_addressee_node,
                ),
            )
            for obj_var in template.all_object_variables
        ]
        if (
            default_addressee_node
            not in immutableset(
                object_.ontology_node
                for (_, object_) in object_var_to_instantiations_mutable
            )
            and not has_addressee
        ):
            object_var_to_instantiations_mutable.append(
                (
                    object_variable(
                        default_addressee_node.handle, default_addressee_node
                    ),
                    SituationObject.instantiate_ontology_node(
                        default_addressee_node,
                        properties=[IS_ADDRESSEE],
                        debug_handle=default_addressee_node.handle + "_default_addressee",
                        ontology=self.ontology,
                    ),
                )
            )
        return immutabledict(object_var_to_instantiations_mutable)

    def _instantiate_object(
        self,
        object_var: TemplateObjectVariable,
        variable_assignment: "TemplateVariableAssignment",
        *,
        has_addressee: bool,
        default_addressee_node: OntologyNode,
    ) -> SituationObject:
        object_type = variable_assignment.object_variables_to_fillers[object_var]
        asserted_properties = object_var.asserted_properties
        if object_type == default_addressee_node and not has_addressee:
            asserted_properties = immutableset(
                object_var.asserted_properties.union({IS_ADDRESSEE})
            )
        return SituationObject.instantiate_ontology_node(
            ontology_node=object_type,
            properties=[
                # instantiate any property variables associated with this object
                variable_assignment.property_variables_to_fillers[asserted_property]
                if isinstance(asserted_property, TemplatePropertyVariable)
                else asserted_property
                for asserted_property in asserted_properties
            ],
            ontology=self.ontology,
        )

    def _instantiate_situation(
        self,
        template: Phase1SituationTemplate,
        variable_assignment: "TemplateVariableAssignment",
        object_var_to_instantiations,
    ) -> HighLevelSemanticsSituation:
        return HighLevelSemanticsSituation(
            from_template=template.name,
            ontology=self.ontology,
            salient_objects=[
                object_var_to_instantiations[obj_var]
                for obj_var in template.salient_object_variables
            ],
            other_objects=[  # type: ignore
                object_var_to_instantiations[obj_var]  # type: ignore
                for obj_var in (
                    immutableset(object_var_to_instantiations.keys()).difference(
                        template.salient_object_variables
                    )
                )
                # We use the keys of the mapping in case a default addressee was added
            ],
            always_relations=[
                relation.copy_remapping_objects(object_var_to_instantiations)
                for relation in template.asserted_always_relations
            ],
            actions=[
                self._instantiate_action(
                    action,
                    object_var_to_instantiations,
                    variable_assignment.action_variables_to_fillers,
                )
                for action in template.actions
            ],
            before_action_relations=[
                relation.copy_remapping_objects(object_var_to_instantiations)
                for relation in template.before_action_relations
            ],
            after_action_relations=[
                relation.copy_remapping_objects(object_var_to_instantiations)
                for relation in template.after_action_relations
            ],
            syntax_hints=template.syntax_hints,
            axis_info=self._compute_axis_info(object_var_to_instantiations),
            gazed_objects=immutableset(
                object_var_to_instantiations[object_]
                for object_ in template.gazed_objects
            ),
        )

    def _has_multiple_recognized_particulars(
        self, variable_binding: Mapping["TemplateObjectVariable", SituationObject]
    ) -> bool:
        # First, we check for a universal constraint that a situation
        # cannot contain multiple recognized particulars.
        recognized_particular_counts = Counter(
            object_binding.ontology_node
            for object_binding in variable_binding.values()
            if is_recognized_particular(self.ontology, object_binding.ontology_node)
        )
        return bool(
            recognized_particular_counts
            and max(recognized_particular_counts.values()) > 1
        )

    def _satisfies_constraints(
        self,
        template: Phase1SituationTemplate,
        instantiated_situation: HighLevelSemanticsSituation,
        variable_binding: Mapping["TemplateObjectVariable", SituationObject],
    ) -> bool:
        for constraining_relation in template.constraining_relations:
            # the constraint is satisfied if it is explicitly-specified as true
            relation_bound_to_situation_objects = (
                constraining_relation.copy_remapping_objects(variable_binding)
            )
            relation_explicitly_specified = instantiated_situation.relation_always_holds(
                relation_bound_to_situation_objects
            )
            # or if we can deduce it is true from general relations in the ontology
            # (e.g. general tendencies like people being larger than balls)
            # Note we do not currently allow overriding relations derived from the ontology.
            # See https://github.com/isi-vista/adam/issues/229
            relation_implied_by_ontology_relations: bool
            if isinstance(
                relation_bound_to_situation_objects.second_slot, SituationObject
            ):
                # second slot could have been a region, in which case ontological relations
                # do not apply
                relation_in_terms_of_object_types = relation_bound_to_situation_objects.copy_remapping_objects(
                    {
                        relation_bound_to_situation_objects.first_slot: relation_bound_to_situation_objects.first_slot.ontology_node,
                        relation_bound_to_situation_objects.second_slot: relation_bound_to_situation_objects.second_slot.ontology_node,
                    }
                )
                relation_implied_by_ontology_relations = (
                    relation_in_terms_of_object_types in self.ontology.relations
                )
            else:
                relation_implied_by_ontology_relations = False
            if not (
                relation_explicitly_specified or relation_implied_by_ontology_relations
            ):
                return False
        return True

    def _instantiate_action(
        self,
        action: Action[_ExplicitOrVariableActionType, "TemplateObjectVariable"],
        object_var_to_instantiations: Mapping["TemplateObjectVariable", SituationObject],
        action_variables_to_fillers: Mapping["TemplateActionTypeVariable", OntologyNode],
    ) -> Action[OntologyNode, SituationObject]:
        def map_action_type() -> OntologyNode:
            if isinstance(action.action_type, OntologyNode):
                return action.action_type
            else:
                return action_variables_to_fillers[action.action_type]

        def map_action_variable_binding(
            x: Union[TemplateObjectVariable, TemplateRegion]
        ) -> Union[SituationObject, SituationRegion]:
            if isinstance(x, Region):
                return x.copy_remapping_objects(object_var_to_instantiations)
            else:
                return object_var_to_instantiations[x]

        # new_aux_bindings = []
        # for (auxiliary_variable, auxiliary_variable_binding) in \
        #         action.auxiliary_variable_bindings.items():
        #     new_aux_bindings.append((auxiliary_variable, object_var_to_instantiations[auxiliary_variable_binding]))

        return Action(
            action_type=map_action_type(),
            argument_roles_to_fillers=[
                (role, map_action_variable_binding(arg))
                for (role, arg) in action.argument_roles_to_fillers.items()
            ],
            during=action.during.copy_remapping_objects(object_var_to_instantiations)
            if action.during
            else None,
            auxiliary_variable_bindings=[
                (
                    auxiliary_variable,
                    map_action_variable_binding(auxiliary_variable_binding),
                )
                for (
                    auxiliary_variable,
                    auxiliary_variable_binding,
                ) in action.auxiliary_variable_bindings.items()
            ],
        )

    def _compute_axis_info(
        self,
        object_var_to_instantiations: Mapping[TemplateObjectVariable, SituationObject],
    ) -> AxesInfo[SituationObject]:
        # if there is an addressee, then we determine which axis
        # of each object faces the addressee
        addressees = immutableset(
            obj
            for obj in object_var_to_instantiations.values()
            if IS_ADDRESSEE in obj.properties
        )
        addressee = only(
            addressees, too_long=RuntimeError("Multiple addressees not supported")
        )
        if not addressee:
            return AxesInfo()

        return AxesInfo(
            addressee=addressee,
            axes_facing=[
                (
                    addressee,
                    # TODO: fix this hack
                    HorizontalAxisOfObject(obj, index=1).to_concrete_axis(  # type: ignore
                        None
                    ),
                )
                for obj in object_var_to_instantiations.values()
                if obj.axes
            ],
        )


def object_variable(
    debug_handle: str,
    root_node: OntologyNode = THING,
    *,
    required_properties: Iterable[OntologyNode] = immutableset(),
    banned_properties: Iterable[OntologyNode] = immutableset(),
    added_properties: Iterable[
        Union[OntologyNode, TemplatePropertyVariable]
    ] = immutableset(),
    banned_ontology_types: Iterable[OntologyNode] = immutableset(),
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

    You can optionally specify *banned_ontology_types* to block this variable
    from being filled by those ontology types or any of their descendants.
    """
    real_required_properties = list(required_properties)
    if root_node != LEARNER and root_node != GROUND:
        # the learner and the ground are special cases of things we want to be
        # explicitly instantiable but not instantiable by variable.
        real_required_properties.append(CAN_FILL_TEMPLATE_SLOT)

    return TemplateObjectVariable(
        debug_handle,
        ByHierarchyAndProperties(
            descendents_of=root_node,
            required_properties=real_required_properties,
            banned_properties=banned_properties,
            banned_ontology_types=banned_ontology_types,
        ),
        asserted_properties=added_properties,
    )


def property_variable(
    debug_handle: str,
    root_node: OntologyNode = PROPERTY,
    *,
    with_meta_properties: Iterable[OntologyNode] = immutableset(),
    banned_values: Iterable[OntologyNode] = immutableset(),
) -> TemplatePropertyVariable:
    r"""
    Create a `TemplatePropertyVariable` with the specified *debug_handle*
    which can be filled by any property whose `OntologyNode` is a descendant of
    (or is exactly) *root_node*
    and which possesses all properties in *with_properties*.
    """
    real_required_properties = list(with_meta_properties)
    # real_required_properties.append(CAN_FILL_TEMPLATE_SLOT)

    hierarchy_selector = ByHierarchyAndProperties(
        descendents_of=root_node, required_properties=real_required_properties
    )
    selector: OntologyNodeSelector
    if banned_values:
        selector = FilterOut(hierarchy_selector, bad_values=banned_values)
    else:
        selector = hierarchy_selector
    return TemplatePropertyVariable(debug_handle, selector)


def action_variable(
    debug_handle: str,
    root_node: OntologyNode = ACTION,
    *,
    with_subcategorization_frame: Optional[Iterable[OntologyNode]] = None,
    with_properties: Iterable[OntologyNode] = immutableset(),
) -> TemplateActionTypeVariable:
    r"""
    Create a `TemplatePropertyVariable` with the specified *debug_handle*
    which can be filled by any property whose `OntologyNode` is a descendant of
    (or is exactly) *root_node*
    and which possesses all properties in *with_properties*.
    """
    hierarchy_and_properties_selector = ByHierarchyAndProperties(
        descendents_of=root_node, required_properties=with_properties
    )

    selector: OntologyNodeSelector
    # it could be empty for e.g. rain or snow
    if with_subcategorization_frame is not None:
        selector = AndOntologySelector(
            [
                hierarchy_and_properties_selector,
                SubcategorizationSelector(with_subcategorization_frame),
            ]
        )
    else:
        selector = hierarchy_and_properties_selector
    return TemplateActionTypeVariable(debug_handle, selector)


def color_variable(
    debug_handle: str, *, required_properties: Iterable[OntologyNode] = immutableset()
) -> TemplatePropertyVariable:
    r"""
    Create a `TemplatePropertyVariable` with the specified *debug_handle*
    which ranges over all colors in the ontology.
    """
    return property_variable(
        debug_handle,
        COLOR,
        banned_values=[COLOR, TRANSPARENT],
        with_meta_properties=required_properties,
    )


@attrs(frozen=True, slots=True)
class TemplateVariableAssignment:
    """
    An assignment of ontology types to object and property variables in a situation.
    """

    object_variables_to_fillers: ImmutableDict[
        "TemplateObjectVariable", OntologyNode
    ] = attrib(converter=_to_immutabledict, default=immutabledict())
    property_variables_to_fillers: ImmutableDict[
        "TemplatePropertyVariable", OntologyNode
    ] = attrib(converter=_to_immutabledict, default=immutabledict())
    action_variables_to_fillers: ImmutableDict[
        "TemplateActionTypeVariable", OntologyNode
    ] = attrib(converter=_to_immutabledict, default=immutabledict())


class _VariableAssigner(ABC):
    @abstractmethod
    def variable_assignments(
        self,
        *,
        ontology: Ontology,
        object_variables: AbstractSet["TemplateObjectVariable"],
        property_variables: AbstractSet["TemplatePropertyVariable"],
        action_variables: AbstractSet["TemplateActionTypeVariable"],
        chooser: SequenceChooser,
    ) -> Iterable[TemplateVariableAssignment]:
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
        action_variables: AbstractSet["TemplateActionTypeVariable"],
        chooser: SequenceChooser,  # pylint: disable=unused-argument
    ) -> Iterable[TemplateVariableAssignment]:
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
                for action_combination in self._all_combinations(
                    action_variables, ontology=ontology, rng=rng
                ):
                    yield TemplateVariableAssignment(
                        object_variables_to_fillers=object_combination,
                        property_variables_to_fillers=property_combination,
                        action_variables_to_fillers=action_combination,
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
        action_variables: AbstractSet["TemplateActionTypeVariable"],
        chooser: SequenceChooser,
    ) -> Iterable[TemplateVariableAssignment]:
        # we need to do the zip() below instead of using nested for loops
        # or you will get a bunch of propery combinations for the same object combination.
        object_combinations = self._sample_combinations(
            object_variables, ontology=ontology, chooser=chooser
        )
        property_combinations = self._sample_combinations(
            property_variables, ontology=ontology, chooser=chooser
        )
        action_combinations = self._sample_combinations(
            action_variables, ontology=ontology, chooser=chooser
        )

        concatenated_combinations = zip(
            object_combinations, property_combinations, action_combinations
        )

        for (
            object_combination,
            property_combination,
            action_combination,
        ) in concatenated_combinations:
            yield TemplateVariableAssignment(
                object_variables_to_fillers=object_combination,
                property_variables_to_fillers=property_combination,
                action_variables_to_fillers=action_combination,
            )

    def _sample_combinations(
        self,
        variables: AbstractSet[_VarT],
        *,
        ontology: Ontology,
        chooser: SequenceChooser,
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


@attrs(slots=True, frozen=True)
class _FixedVariableAssigner(_VariableAssigner):
    """
    A `_VariableAssigner` which always returns the same variable assignment.
    """

    assignment: TemplateVariableAssignment = attrib(
        validator=instance_of(TemplateVariableAssignment)
    )

    def variable_assignments(
        self,
        *,
        ontology: Ontology,  # pylint:disable=unused-argument
        object_variables: AbstractSet["TemplateObjectVariable"],
        property_variables: AbstractSet["TemplatePropertyVariable"],
        action_variables: AbstractSet["TemplateActionTypeVariable"],
        chooser: SequenceChooser,  # pylint:disable=unused-argument
    ) -> Iterable[TemplateVariableAssignment]:
        check_arg(
            all(
                obj_var in self.assignment.object_variables_to_fillers
                for obj_var in object_variables
            )
        )
        check_arg(
            all(
                prop_var in self.assignment.property_variables_to_fillers
                for prop_var in property_variables
            )
        )
        check_arg(
            all(
                action_var in self.assignment.action_variables_to_fillers
                for action_var in action_variables
            )
        )

        return (self.assignment,)


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
