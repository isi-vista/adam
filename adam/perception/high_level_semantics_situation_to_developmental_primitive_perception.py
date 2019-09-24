from itertools import chain
from typing import AbstractSet, Dict, List, Mapping, Optional, Union, cast

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import (
    ImmutableDict,
    ImmutableSet,
    immutabledict,
    immutableset,
    ImmutableSetMultiDict,
    immutablesetmultidict,
)
from more_itertools import only, quantify
from vistautils.preconditions import check_arg

from adam.ontology import IN_REGION, OntologyNode
from adam.ontology.action_description import ActionDescription
from adam.ontology.during import DuringAction
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    BINARY,
    COLOR,
    COLORS_TO_RGBS,
    GAILA_PHASE_1_ONTOLOGY,
    GAZED_AT,
    GROUND,
    HOLLOW,
    IS_SPEAKER,
    LIQUID,
    PART_OF,
    PERCEIVABLE,
    TWO_DIMENSIONAL,
    LEARNER,
    on,
)
from adam.ontology.phase1_spatial_relations import INTERIOR, Region, SpatialPath
from adam.ontology.structural_schema import ObjectStructuralSchema, SubObject
from adam.perception import (
    ObjectPerception,
    PerceptualRepresentation,
    PerceptualRepresentationGenerator,
)
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    HasBinaryProperty,
    HasColor,
    PropertyPerception,
    RgbColorPerception,
)
from adam.random_utils import SequenceChooser
from adam.relation import Relation
from adam.situation import Action, SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation


@attrs(frozen=True, slots=True)
class HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
    PerceptualRepresentationGenerator[
        HighLevelSemanticsSituation, DevelopmentalPrimitivePerceptionFrame
    ]
):
    r"""
    Produces `PerceptualRepresentation`\ s with `DevelopmentalPrimitivePerceptionFrame`\ s
    for `HighLevelSemanticsSituation`\ s.

    This is the primary generator of perceptual representations for Phase 1 of ADAM.
    """

    ontology: Ontology = attrib(validator=instance_of(Ontology))
    """
    The `Ontology` assumed to be used by both the `HighLevelSemanticsSituation`
    and the output `DevelopmentalPrimitivePerceptionFrame`.
    """

    def generate_perception(
        self, situation: HighLevelSemanticsSituation, chooser: SequenceChooser
    ) -> PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]:
        check_arg(
            situation.ontology == self.ontology,
            "Cannot generate perceptions "
            "for a situation with a mis-matched "
            "ontology.",
        )
        # all the work is done in a stateful _PerceptionGeneration object
        return _PerceptionGeneration(self, situation, chooser).do()


@attrs(frozen=True, slots=True)
class _ObjectHandleGenerator:
    """
    Generates debug handles for objects.

    There is a little complexity to this because we want to use subscripts
    to distinguish distinct objects of the same type
    (e.g. multiple arms of a person).

    This is only for internal use by `_PerceptionGeneration`.
    """

    _object_handles_seen: List[str] = attrib(init=False, default=Factory(list))

    def subscripted_handle(self, object_schema: ObjectStructuralSchema) -> str:
        unsubscripted_handle = object_schema.ontology_node.handle
        # using count() here makes subscript computation linear time
        # in the number of objects in a situation,
        # but this should be small enough not to matter.
        subscript = self._object_handles_seen.count(unsubscripted_handle)
        self._object_handles_seen.append(unsubscripted_handle)
        return f"{unsubscripted_handle}_{subscript}"


@attrs(frozen=True, slots=True)
class _PerceptionGeneration:
    """
    A stateful object which performs all the work for the stateless
    `HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator`
    """

    _generator: HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator = attrib(
        validator=instance_of(
            HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator
        )
    )
    _situation: HighLevelSemanticsSituation = attrib(
        validator=instance_of(HighLevelSemanticsSituation)
    )
    _chooser: SequenceChooser = attrib(validator=instance_of(SequenceChooser))
    _objects_to_perceptions: Dict[SituationObject, ObjectPerception] = attrib(
        init=False, default=Factory(dict)
    )
    r"""
    Maps `SituationObject`\ s to the learner's `ObjectPerception`\ s of them.
    """
    _object_perceptions: List[ObjectPerception] = attrib(
        init=False, default=Factory(list)
    )
    _object_perceptions_to_ontology_nodes: Dict[ObjectPerception, OntologyNode] = attrib(
        init=False, default=Factory(dict)
    )
    r"""
    Maps `ObjectPerception`\ s to the learner's `OntologyNode`\ s of them.
    """
    _object_handle_generator: _ObjectHandleGenerator = attrib(
        init=False, default=Factory(_ObjectHandleGenerator)
    )
    """
    Used for tracking sub-scripts of objects.
    """
    _regions_to_perceptions: Dict[
        Region[SituationObject], Region[ObjectPerception]
    ] = attrib(init=False, default=Factory(dict))
    """
    Tracks the correspondence between spatial regions in the situation description
    and those in the perceptual representation.
    """
    _relation_perceptions: List[Relation[ObjectPerception]] = attrib(
        init=False, default=Factory(list)
    )
    r"""
    `Relation`\ s perceived by the learner.
    """
    _property_assertion_perceptions: List[PropertyPerception] = attrib(
        init=False, default=Factory(list)
    )
    r"""
    `PropertyPerception`\ s perceived by the learner.
    """

    def do(self) -> PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]:
        try:
            return self._real_do()
        except Exception as e:
            raise RuntimeError(
                f"Error while generating perceptions " f"for situation {self._situation}"
            ) from e

    def _real_do(self) -> PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]:
        self._sanity_check_situation()

        # The first step is to determine what objects are perceived.
        self._perceive_objects()
        # Next we perceive any regions referred to by the situation
        self._perceive_regions()
        # Then we perceive the properties those objects possess
        self._perceive_property_assertions()
        # Perceive any relations explicitly specified in the situation
        # as holding both before and after any actions
        self._relation_perceptions.extend(
            self._perceive_relation(relation)
            for relation in self._situation.always_relations
        )

        # Handle implicit size relations
        # self._perceive_implicit_size()

        # Other relations implied by actions will be handled during action translation below.

        if not self._situation.actions:
            self._perceive_ground_relations()
            return PerceptualRepresentation.single_frame(
                DevelopmentalPrimitivePerceptionFrame(
                    perceived_objects=self._object_perceptions,
                    relations=self._relation_perceptions,
                    property_assertions=self._property_assertion_perceptions,
                )
            )

        # finally, if there are actions, we perceive the before and after states of the action
        _action_perception = self._perceive_action()
        # sometimes additional before and after relations will be given explicitly by the user
        explicit_before_relations = [
            self._perceive_relation(relation)
            for relation in self._situation.before_action_relations
        ]
        explicit_after_relations = [
            self._perceive_relation(relation)
            for relation in self._situation.after_action_relations
        ]

        first_frame = DevelopmentalPrimitivePerceptionFrame(
            perceived_objects=self._object_perceptions,
            relations=chain(
                self._relation_perceptions,
                explicit_before_relations,
                _action_perception.before_relations,
            ),
            property_assertions=self._property_assertion_perceptions,
        )
        second_frame = DevelopmentalPrimitivePerceptionFrame(
            perceived_objects=self._object_perceptions,
            relations=chain(
                self._relation_perceptions,
                explicit_after_relations,
                _action_perception.after_relations,
            ),
            property_assertions=self._property_assertion_perceptions,
        )

        return PerceptualRepresentation(
            frames=(first_frame, second_frame),
            during=self._compute_during(
                during_from_action_description=_action_perception.during_action
            ),
        )

    def _sanity_check_situation(self) -> None:
        if (
            quantify(
                property_ == IS_SPEAKER
                for object_ in self._situation.all_objects
                for property_ in object_.properties
            )
            > 1
        ):
            raise TooManySpeakersException(
                f"Situations with multiple speakers are not supported: {self._situation}"
            )

    def _perceive_regions(self) -> None:
        # gather all places a region can appear in the situation representation
        possible_regions = chain(
            (relation.second_slot for relation in self._situation.always_relations),
            (
                filler
                for action in self._situation.actions
                for (_, filler) in action.argument_roles_to_fillers.items()
            ),
        )
        regions = immutableset(
            possible_region
            for possible_region in possible_regions
            if isinstance(possible_region, Region)
        )

        # map each of these to a region perception
        self._regions_to_perceptions.update(
            {
                region: region.copy_remapping_objects(self._objects_to_perceptions)
                for region in regions
            }
        )

    def _perceive_region(
        self, region: Region[SituationObject]
    ) -> Region[ObjectPerception]:
        return region.copy_remapping_objects(self._objects_to_perceptions)

    @attrs(frozen=True, slots=True)
    class _ActionPerception:
        before_relations: ImmutableSet[Relation[ObjectPerception]] = attrib()
        after_relations: ImmutableSet[Relation[ObjectPerception]] = attrib()
        during_action: Optional[DuringAction[ObjectPerception]] = attrib()

    def _perceive_action(self) -> "_PerceptionGeneration._ActionPerception":
        # Extract relations from action
        situation_action = self._situation.actions[0]
        # TODO: Handle multiple actions
        if len(self._situation.actions) > 1:
            raise RuntimeError("Cannot handle multiple situation actions")

        # e.g: SituationAction(PUT, ((AGENT, mom),(THEME, ball),(DESTINATION, SituationRelation(
        # IN_REGION, ball, ON(TABLE)))))
        # Get description from PUT (PUT is action_type)
        action_description: ActionDescription = GAILA_PHASE_1_ONTOLOGY.required_action_description(
            situation_action.action_type,
            situation_action.argument_roles_to_fillers.keys(),
        )

        action_objects_variables_to_perceived_objects = self._bind_action_objects_variables_to_perceived_objects(
            situation_action, action_description
        )

        enduring_relations = self._perceive_action_relations(
            conditions=action_description.enduring_conditions,
            action_object_variables_to_object_perceptions=action_objects_variables_to_perceived_objects,
        )
        before_relations = self._perceive_action_relations(
            conditions=action_description.preconditions,
            action_object_variables_to_object_perceptions=action_objects_variables_to_perceived_objects,
        )
        after_relations = self._perceive_action_relations(
            conditions=action_description.postconditions,
            action_object_variables_to_object_perceptions=action_objects_variables_to_perceived_objects,
        )

        self._perceive_ground_relations()

        return _PerceptionGeneration._ActionPerception(
            before_relations=immutableset(chain(enduring_relations, before_relations)),
            after_relations=immutableset(chain(enduring_relations, after_relations)),
            during_action=action_description.during.copy_remapping_objects(
                cast(
                    Mapping[SituationObject, ObjectPerception],
                    action_objects_variables_to_perceived_objects,
                )
            )
            if action_description.during
            else None,
        )

    def _bind_action_objects_variables_to_perceived_objects(
        self,
        situation_action: Action[OntologyNode, SituationObject],
        action_description: ActionDescription,
    ) -> Mapping[SituationObject, Union[Region[ObjectPerception], ObjectPerception]]:
        if any(
            len(fillers) > 1
            for fillers in situation_action.argument_roles_to_fillers.value_groups()
        ):
            raise RuntimeError("Cannot handle multiple fillers for an argument role yet.")

        bindings: Dict[
            SituationObject, Union[ObjectPerception, Region[ObjectPerception]]
        ] = {}

        # for action description objects which play semantic roles,
        # the SituationAction gives us the binding directly
        subcategorization_frame = action_description.frame
        for (role, action_object) in subcategorization_frame.roles_to_variables.items():
            # Regions can also fill certain semantic roles,
            # but Regions are always relative to objects,
            # so we can translate them after we have translated the action objects
            situation_object_bound_to_role = only(
                situation_action.argument_roles_to_fillers[role]
            )
            if isinstance(situation_object_bound_to_role, SituationObject):
                bindings[action_object] = self._objects_to_perceptions[
                    situation_object_bound_to_role
                ]
            elif isinstance(situation_object_bound_to_role, Region):
                bindings[action_object] = Region(
                    reference_object=self._objects_to_perceptions[
                        situation_object_bound_to_role.reference_object
                    ],
                    distance=situation_object_bound_to_role.distance,
                    direction=situation_object_bound_to_role.direction,
                )

        # but there are also action description objects
        # which don't fill semantic roles directly.
        # For these, we iterate through all objects we have perceived so far
        # to see if any satisfy all the constraints specified by the action.
        # We will use as a running example the object
        # corresponding to a person's hand used to move an object
        # for the action PUT.
        action_variables_from_non_frames: List[SituationObject] = []
        for condition_set in (
            action_description.enduring_conditions,
            action_description.preconditions,
            action_description.postconditions,
        ):
            for condition in condition_set:
                condition.accumulate_referenced_objects(action_variables_from_non_frames)
        if action_description.during:
            action_description.during.accumulate_referenced_objects(
                action_variables_from_non_frames
            )

        unbound_action_object_variables = immutableset(
            # We use immutableset to remove duplicates
            # while maintaining deterministic iteration order.
            action_variable
            for action_variable in action_variables_from_non_frames
            # not already mapped by a semantic role
            if action_variable not in bindings
            and isinstance(action_variable, SituationObject)
        )

        bindings.update(
            {
                unbound_action_object_variable: self._bind_action_object_variable(
                    situation_action, unbound_action_object_variable
                )
                for unbound_action_object_variable in unbound_action_object_variables
            }
        )

        return bindings

    def _bind_action_object_variable(
        self,
        situation_action: Action[OntologyNode, SituationObject],
        action_object_variable: SituationObject,
    ) -> Union[ObjectPerception, Region[ObjectPerception]]:
        """
        Binds an action object variable to an object that we have perceived.

        Currently this only pays attention to object properties in binding and not relationships.
        """
        explicit_binding = situation_action.auxiliary_variable_bindings.get(
            action_object_variable
        )
        if explicit_binding:
            if isinstance(explicit_binding, Region):
                return explicit_binding.copy_remapping_objects(
                    self._objects_to_perceptions
                )
            else:
                return self._objects_to_perceptions[explicit_binding]

        # we continue to use the hand from PUT
        # ( see _bind_action_objects_variables_to_perceived_objects )
        ontology = self._generator.ontology
        perceived_objects_matching_constraints = [
            object_perception
            for (
                object_perception,
                ontology_node,
            ) in self._object_perceptions_to_ontology_nodes.items()
            if ontology.is_subtype_of(ontology_node, action_object_variable.ontology_node)
            and ontology.has_all_properties(
                ontology_node, action_object_variable.properties
            )
        ]

        if len(perceived_objects_matching_constraints) == 1:
            return only(perceived_objects_matching_constraints)
        elif not perceived_objects_matching_constraints:
            raise RuntimeError(
                f"Can not find object with properties {action_object_variable} in order to bind "
                f"{action_object_variable}. All perceived objects are: {self._object_perceptions_to_ontology_nodes}"
            )
        else:
            distinct_property_sets_for_matching_object_types = set(
                ontology.properties_for_node(
                    self._object_perceptions_to_ontology_nodes[obj]
                )
                for obj in perceived_objects_matching_constraints
            )
            # if the found objects have identical properties, we choose one arbitrarily
            # e.g. a person with two hands
            if len(distinct_property_sets_for_matching_object_types) == 1:
                return perceived_objects_matching_constraints[0]
            else:
                raise RuntimeError(
                    f"Found multiple objects with properties {action_object_variable}: "
                    f"{perceived_objects_matching_constraints} when binding {action_object_variable}"
                )

    def _perceive_action_relations(
        self,
        conditions: ImmutableSet[Relation[SituationObject]],
        *,
        action_object_variables_to_object_perceptions: Mapping[
            SituationObject, Union[ObjectPerception, Region[ObjectPerception]]
        ],
    ) -> AbstractSet[Relation[ObjectPerception]]:
        """

        Args:
            already_known_relations: relations to automatically include in our returned output.
                                     This is to support putting thing perceived from an actions
                                     preconditions into its post-conditions as well.
        """
        relations = []

        for condition in conditions:  # each one is a SituationRelation
            # Generate perceptions for situation objects in the given condition.
            perception_1 = cast(
                ObjectPerception,
                action_object_variables_to_object_perceptions[condition.first_slot],
            )
            # the second slot of a relation can be a SituationObject or a Region,
            # so the mapping logic is more complicated.
            perception_2 = self._perceive_object_or_region_relation_filler(
                slot_filler=condition.second_slot,
                action_object_variables_to_object_perceptions=action_object_variables_to_object_perceptions,
            )

            relation_perception = Relation(
                relation_type=condition.relation_type,
                first_slot=perception_1,
                second_slot=perception_2,
            )

            if not condition.negated:
                relations.append(relation_perception)
            else:
                # Remove the relation from already known relations
                relations = [
                    relation
                    for relation in relations
                    if not (
                        relation.relation_type == condition.relation_type
                        and relation.first_slot == perception_1
                        and relation.second_slot == perception_2
                    )
                ]

        return immutableset(relations)

    def _perceive_object_or_region_relation_filler(
        self,
        slot_filler: Union[SituationObject, Region[SituationObject]],
        *,
        action_object_variables_to_object_perceptions: Mapping[
            SituationObject, Union[ObjectPerception, Region[ObjectPerception]]
        ],
    ) -> Union[ObjectPerception, Region[ObjectPerception]]:
        if isinstance(slot_filler, Region):
            object_mapping: Dict[SituationObject, ObjectPerception] = {}
            # regions are not a real possibility for lookup,
            # so mypy's complaints here are irrelevant
            object_mapping.update(self._objects_to_perceptions)  # type: ignore
            object_mapping.update(  # type: ignore
                action_object_variables_to_object_perceptions
            )

            return slot_filler.copy_remapping_objects(object_mapping)
        else:
            return action_object_variables_to_object_perceptions[slot_filler]

    def _perceive_property_assertions(self) -> None:
        for situation_object in self._situation.all_objects:
            # process explicitly and implicitly-specified properties
            all_object_properties: List[OntologyNode] = []
            # Explicit properties are stipulated by the user in the situation description.
            all_object_properties.extend(situation_object.properties)
            # Implicit properties are derived from what type of thing an object is,
            # e.g. that people are ANIMATE.
            all_object_properties.extend(
                self._generator.ontology.properties_for_node(
                    situation_object.ontology_node
                )
            )

            # Colors require special processing, so we strip them all out and then
            # add back only the (at most) single color we determined an object has.
            properties_to_perceive = [
                property_
                for property_ in all_object_properties
                if not self._generator.ontology.is_subtype_of(property_, COLOR)
            ]
            color = self._determine_color(situation_object)
            if color:
                properties_to_perceive.append(color)

            # If it is a liquid not inside a container, add two-dimensional property
            if LIQUID in GAILA_PHASE_1_ONTOLOGY.properties_for_node(
                situation_object.ontology_node
            ) and not any(
                r.first_slot == situation_object
                and r.relation_type == IN_REGION
                and isinstance(r.second_slot, Region)
                and HOLLOW
                in GAILA_PHASE_1_ONTOLOGY.properties_for_node(
                    r.second_slot.reference_object.ontology_node
                )
                and r.second_slot.distance == INTERIOR
                for r in self._situation.always_relations
            ):
                properties_to_perceive.append(TWO_DIMENSIONAL)

            # Focused Objects are in a special field of the Situation, we check if the situation_object
            # is a focused and apply the tag here if that is the case.
            if situation_object in self._situation.gazed_objects:
                properties_to_perceive.append(GAZED_AT)

            # We wrap an ImmutableSet around properties_to_perceive to remove duplicates
            # while still guaranteeing deterministic iteration order.
            for property_ in immutableset(properties_to_perceive):
                self._perceive_property(
                    self._generator.ontology.properties_for_node(property_),
                    self._objects_to_perceptions[situation_object],
                    property_,
                )

        # Properties derived from the role of the situation object in the action
        for action in self._situation.actions:
            action_description: ActionDescription = GAILA_PHASE_1_ONTOLOGY.required_action_description(
                action.action_type, action.argument_roles_to_fillers.keys()
            )
            for role in action_description.frame.semantic_roles:  # e.g. AGENT
                variable = action_description.frame.roles_to_variables[
                    role
                ]  # e.g. _PUT_AGENT
                fillers = action.argument_roles_to_fillers[role]  # e.g. {Mom}
                for property_ in action_description.asserted_properties[variable]:
                    for situation_or_region in fillers:
                        if isinstance(situation_or_region, SituationObject):
                            perception_of_object = self._objects_to_perceptions[
                                situation_or_region
                            ]
                        else:
                            # We are propagating properties asserted on regions to their
                            # reference objects.
                            # TODO: issue #263
                            perception_of_object = self._objects_to_perceptions[
                                situation_or_region.reference_object
                            ]
                        self._perceive_property(
                            self._generator.ontology.properties_for_node(property_),
                            perception_of_object,
                            property_,
                        )

    def _perceive_property(
        self,
        attributes_of_property: ImmutableSet[OntologyNode],
        perceived_object: ObjectPerception,
        property_: OntologyNode,
    ) -> None:
        # e.g. is this a perceivable property, binary property, color, etc...
        if PERCEIVABLE in attributes_of_property:
            # Convert the property (which as an OntologyNode object) into PropertyPerception object
            if BINARY in attributes_of_property:
                perceived_property: PropertyPerception = HasBinaryProperty(
                    perceived_object, property_
                )
            elif self._generator.ontology.is_subtype_of(property_, COLOR):
                # Sample an RGB value for the color property and generate perception for it
                if property_ in COLORS_TO_RGBS.keys():
                    color_options = COLORS_TO_RGBS[property_]
                    if color_options:
                        r, g, b = self._chooser.choice(color_options)
                        perceived_property = HasColor(
                            perceived_object, RgbColorPerception(r, g, b)
                        )
                    else:  # Handles the case of TRANSPARENT
                        perceived_property = HasBinaryProperty(
                            perceived_object, property_
                        )
                else:
                    raise RuntimeError(
                        f"Not sure how to generate perception for the unknown property {property_} "
                        f"which is marked as COLOR"
                    )
            else:
                raise RuntimeError(
                    f"Not sure how to generate perception for property {property_} "
                    f"which is marked as perceivable"
                )
            self._property_assertion_perceptions.append(perceived_property)

    def _determine_color(
        self, situation_object: SituationObject
    ) -> Optional[OntologyNode]:
        explicitly_specified_colors = immutableset(
            property_
            for property_ in situation_object.properties
            if self._generator.ontology.is_subtype_of(property_, COLOR)
        )

        prototypical_colors_for_object_type = immutableset(
            property_
            for property_ in self._generator.ontology.properties_for_node(
                situation_object.ontology_node
            )
            if self._generator.ontology.is_subtype_of(property_, COLOR)
        )

        if explicitly_specified_colors:
            # If any color is specified explicitly, then ignore any which are just prototypical
            # for the object type.
            if len(explicitly_specified_colors) == 1:
                return only(explicitly_specified_colors)
            else:
                raise RuntimeError("Cannot have multiple explicit colors on an object.")
        elif prototypical_colors_for_object_type:
            return self._chooser.choice(prototypical_colors_for_object_type)
        else:
            # We have no idea what color this is, so we currently don't perceive any color.
            # https://github.com/isi-vista/adam/issues/113
            return None

    def _perceive_objects(self) -> None:
        if not any(
            situation_object.ontology_node == GROUND
            for situation_object in self._situation.all_objects
        ):
            ground_schemata = only(self._generator.ontology.structural_schemata(GROUND))
            ground_observed = self._instantiate_object_schema(
                ground_schemata, situation_object=SituationObject(GROUND)
            )
            self._object_perceptions_to_ontology_nodes[ground_observed] = GROUND
        if not any(
            situation_object.ontology_node == LEARNER
            for situation_object in self._situation.all_objects
        ):
            learner_schemata = only(self._generator.ontology.structural_schemata(LEARNER))
            learner_observed = self._instantiate_object_schema(
                learner_schemata, situation_object=SituationObject(LEARNER)
            )
            self._object_perceptions_to_ontology_nodes[learner_observed] = LEARNER
        for situation_object in self._situation.all_objects:
            if not situation_object.ontology_node:
                raise RuntimeError(
                    "Don't yet know how to handle situation objects without "
                    "associated ontology nodes"
                )
            # these are the possible internal structures of objects of this type
            # that the ontology is aware of.
            object_schemata = self._generator.ontology.structural_schemata(
                situation_object.ontology_node
            )
            if not object_schemata:
                raise RuntimeError(f"No structural schema found for {situation_object}")
            if len(object_schemata) > 1:
                # TODO: add issue for this
                # https://github.com/isi-vista/adam/issues/87
                raise RuntimeError(
                    f"Support for objects with multiple structural schemata has not "
                    f"yet keep implemented."
                )

            perceived_object = self._instantiate_object_schema(
                only(object_schemata), situation_object=situation_object
            )
            self._object_perceptions_to_ontology_nodes.update(
                {perceived_object: situation_object.ontology_node}
            )

            self._object_perceptions_to_ontology_nodes[
                perceived_object
            ] = situation_object.ontology_node

    def _instantiate_object_schema(
        self,
        schema: ObjectStructuralSchema,
        *,
        # if the object being instantiated corresponds to an object
        # in the situation description, then this will track that object
        situation_object: Optional[SituationObject] = None,
    ) -> ObjectPerception:
        root_object_perception = ObjectPerception(
            debug_handle=self._object_handle_generator.subscripted_handle(schema),
            geon=schema.geon,
        )
        self._object_perceptions.append(root_object_perception)

        # for object perceptions which correspond to SituationObjects
        # (that is, typically, for objects which are not components of other objects)
        # we track the correspondence
        # to assist in translating SituationRelations and SituationActions.
        if situation_object:
            self._objects_to_perceptions[situation_object] = root_object_perception

        # recursively instantiate sub-components of this object
        sub_object_to_object_perception: ImmutableDict[
            SubObject, ObjectPerception
        ] = immutabledict(
            (sub_object, self._instantiate_object_schema(sub_object.schema))
            for sub_object in schema.sub_objects
        )
        for sub_object in schema.sub_objects:
            sub_object_perception = sub_object_to_object_perception[sub_object]
            self._object_perceptions.append(sub_object_perception)
            self._object_perceptions_to_ontology_nodes[
                sub_object_perception
            ] = sub_object.schema.ontology_node
            # every sub-component has an implicit partOf relationship to its parent object.
            self._relation_perceptions.append(
                Relation(PART_OF, root_object_perception, sub_object_perception)
            )

        # translate sub-object relations specified by the object's structural schema
        for sub_object_relation in schema.sub_object_relations:
            # TODO: right now we translate all situation relations directly to perceptual
            # relations without modification. This is not always the right thing.
            # See https://github.com/isi-vista/adam/issues/80 .
            arg1_perception = sub_object_to_object_perception[
                sub_object_relation.first_slot
            ]
            arg2_perception: Union[ObjectPerception, Region[ObjectPerception]]
            if isinstance(sub_object_relation.second_slot, SubObject):
                arg2_perception = sub_object_to_object_perception[
                    sub_object_relation.second_slot
                ]
            else:
                arg2 = sub_object_relation.second_slot
                arg2_perception = arg2.copy_remapping_objects(
                    sub_object_to_object_perception
                )
            self._relation_perceptions.append(
                Relation(
                    sub_object_relation.relation_type, arg1_perception, arg2_perception
                )
            )
        return root_object_perception

    def _compute_during(
        self, during_from_action_description: Optional[DuringAction[ObjectPerception]]
    ) -> Optional[DuringAction[ObjectPerception]]:
        during_from_situation = only(self._situation.actions).during

        if during_from_situation:
            remapped_during_from_situation = during_from_situation.copy_remapping_objects(
                self._objects_to_perceptions
            )
            if during_from_action_description:
                return remapped_during_from_situation.union(
                    during_from_action_description
                )
            else:
                return remapped_during_from_situation
        elif during_from_action_description:
            return during_from_action_description
        else:
            return None

    def _translate_path(
        self, path: SpatialPath[SituationObject]
    ) -> SpatialPath[ObjectPerception]:
        return path.copy_remapping_objects(self._objects_to_perceptions)

    def _perceive_relation(
        self, relation: Relation[SituationObject]
    ) -> Relation[ObjectPerception]:
        return relation.copy_remapping_objects(self._objects_to_perceptions)

    # TODO: We may need to rework this function to not use quadratic time but it works for now
    # https://github.com/isi-vista/adam/issues/215
    def _perceive_implicit_size(self):
        for situation_object in self._object_perceptions:
            # TODO: we should make a lookup map for this
            # Explicit relation types are tracked so that implicitly defined ones from the Ontology
            # do not overwrite them.
            explicit_relations_predicated_of_object = immutableset(
                relation
                for relation in self._relation_perceptions
                if relation.first_slot == situation_object
            )
            if situation_object not in self._object_perceptions_to_ontology_nodes:
                raise RuntimeError(
                    f"Unable to determine implicit relationship types for {situation_object}"
                    f"which doesn't have an entry in the Object Perceptions "
                    f"to Ontology Nodes dictionary."
                )
            relations_predicated_of_object_type = self._situation.ontology.subjects_to_relations[
                self._object_perceptions_to_ontology_nodes[situation_object]
            ]

            for other_object in self._object_perceptions:
                if not other_object == situation_object:
                    # Track the explicit relationship types we run into so we can ignore the
                    # implicit relationship if one exists.
                    if other_object not in self._object_perceptions_to_ontology_nodes:
                        raise RuntimeError(
                            f"Unable to determine implicit relationship types for {other_object}"
                            f" which doesn't have an entry in the Object Perceptions "
                            f"to Ontology Nodes dictionary."
                        )
                    explicit_relation_types = []
                    for explicit_relation in explicit_relations_predicated_of_object:
                        if explicit_relation.second_slot == other_object:
                            explicit_relation_types.append(
                                explicit_relation.relation_type
                            )
                    for implicit_relation in relations_predicated_of_object_type:
                        if (
                            implicit_relation.second_slot
                            == self._object_perceptions_to_ontology_nodes[other_object]
                        ):
                            if (
                                implicit_relation.relation_type
                                not in explicit_relation_types
                            ):
                                self._relation_perceptions.append(
                                    implicit_relation.copy_remapping_objects(
                                        {
                                            self._object_perceptions_to_ontology_nodes[
                                                other_object
                                            ]: other_object,
                                            self._object_perceptions_to_ontology_nodes[
                                                situation_object
                                            ]: situation_object,
                                        }
                                    )
                                )

    def _perceive_ground_relations(self):
        objects_to_relations = self._objects_to_relations()
        perceived_ground: ObjectPerception = None
        for object_ in self._object_perceptions:
            if self._object_perceptions_to_ontology_nodes[object_] == GROUND:
                perceived_ground = object_
                break
        for situation_object in self._situation.all_objects:
            if situation_object.ontology_node != GROUND:
                if self._objects_to_perceptions[situation_object] in objects_to_relations:
                    # TODO: Handle associating contacts ground so long as a pre-existing relation
                    #  doesn't define this. https://github.com/isi-vista/adam/issues/309
                    pass
                else:
                    self._relation_perceptions.append(
                        on(
                            self._objects_to_perceptions[situation_object],
                            perceived_ground,
                        )
                    )

    def _objects_to_relations(
        self
    ) -> ImmutableSetMultiDict[ObjectPerception, Relation[ObjectPerception]]:
        return immutablesetmultidict(
            (relation.first_slot, relation) for relation in self._relation_perceptions
        )


GAILA_PHASE_1_PERCEPTION_GENERATOR = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
    GAILA_PHASE_1_ONTOLOGY
)


@attrs(auto_exc=True, auto_attribs=True)
class TooManySpeakersException(RuntimeError):
    msg: str
