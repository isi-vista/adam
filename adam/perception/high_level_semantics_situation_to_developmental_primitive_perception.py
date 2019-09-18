from itertools import chain
from typing import AbstractSet, Dict, List, Mapping, Optional, Tuple, Union, cast

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableDict, ImmutableSet, immutabledict, immutableset
from more_itertools import only, quantify
from vistautils.preconditions import check_arg

from adam.ontology import OntologyNode, Region
from adam.ontology.action_description import ActionDescription
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    BINARY,
    COLOR,
    COLORS_TO_RGBS,
    GAILA_PHASE_1_ONTOLOGY,
    IS_SPEAKER,
    PART_OF,
    PERCEIVABLE,
)
from adam.ontology.structural_schema import ObjectStructuralSchema, SubObject
from adam.perception import PerceptualRepresentation, PerceptualRepresentationGenerator
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    HasBinaryProperty,
    HasColor,
    ObjectPerception,
    PropertyPerception,
    RgbColorPerception,
)
from adam.random_utils import SequenceChooser
from adam.relation import Relation
from adam.situation import SituationAction, SituationObject
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
        return _PerceptionGeneration(self, situation, chooser).do


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
        unsubscripted_handle = object_schema.parent_object.handle
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

    @property
    def do(self) -> PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]:
        self._sanity_check_situation()

        # The first step is to determine what objects are perceived.
        self._perceive_objects()
        # Next we perceive any regions referred to by the situation
        self._perceive_regions()
        # Then we perceive the properties those objects possess
        self._perceive_property_assertions()

        if not self._situation.actions:
            return PerceptualRepresentation.single_frame(
                DevelopmentalPrimitivePerceptionFrame(
                    perceived_objects=self._object_perceptions,
                    relations=self._relation_perceptions,
                    property_assertions=self._property_assertion_perceptions,
                )
            )

        # finally, if there are actions, we perceive the before and after states of the action
        before_relations, after_relations = self._perceive_action()
        first_frame = DevelopmentalPrimitivePerceptionFrame(
            perceived_objects=self._object_perceptions,
            relations=before_relations,
            property_assertions=self._property_assertion_perceptions,
        )
        second_frame = DevelopmentalPrimitivePerceptionFrame(
            perceived_objects=self._object_perceptions,
            relations=after_relations,
            property_assertions=self._property_assertion_perceptions,
        )

        return PerceptualRepresentation(frames=(first_frame, second_frame))

    def _sanity_check_situation(self) -> None:
        if (
            quantify(
                property_ == IS_SPEAKER
                for object_ in self._situation.objects
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
            (relation.second_slot for relation in self._situation.relations),
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
                region: Region(
                    reference_object=self._objects_to_perceptions[
                        region.reference_object
                    ],
                    distance=region.distance,
                    direction=region.direction,
                )
                for region in regions
            }
        )

    def _perceive_action(self) -> Tuple[AbstractSet[Relation[ObjectPerception]], ...]:
        # Extract relations from action
        situation_action = self._situation.actions[0]
        # TODO: Handle multiple actions
        if len(self._situation.actions) > 1:
            raise RuntimeError("Cannot handle multiple situation actions")

        # e.g: SituationAction(PUT, ((AGENT, mom),(THEME, ball),(DESTINATION, SituationRelation(
        # IN_REGION, ball, ON(TABLE)))))
        # Get description from PUT (PUT is action_type)
        action_description: ActionDescription = GAILA_PHASE_1_ONTOLOGY.action_to_description[
            situation_action.action_type
        ]

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

        return (
            immutableset(chain(enduring_relations, before_relations)),
            immutableset(chain(enduring_relations, after_relations)),
        )

    def _bind_action_objects_variables_to_perceived_objects(
        self, situation_action: SituationAction, action_description: ActionDescription
    ) -> Mapping[SituationObject, Union[Region[ObjectPerception], ObjectPerception]]:
        if len(action_description.frames) != 1:
            raise RuntimeError(
                "Currently we can only handle verbs with exactly one "
                "subcategorization frame"
            )
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
        subcategorization_frame = only(action_description.frames)
        for (role, action_object) in subcategorization_frame.roles_to_entities.items():
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
        unbound_action_object_variables = immutableset(
            # We use immutableset to remove duplicates
            # while maintaining deterministic iteration order.
            slot_filler
            for condition_set in (
                action_description.enduring_conditions,
                action_description.preconditions,
                action_description.postconditions,
            )
            for condition in condition_set
            for slot_filler in (condition.first_slot, condition.second_slot)
            # some slot fillers will be Regions;
            # we can translate these after the action objects have been translated.
            if (
                isinstance(slot_filler, SituationObject)
                # = not already mapped by a semantic role
                and slot_filler not in bindings
            )
        )

        bindings.update(
            {
                unbound_action_object_variable: self._bind_action_object_variable(
                    unbound_action_object_variable
                )
                for unbound_action_object_variable in unbound_action_object_variables
            }
        )

        return bindings

    def _bind_action_object_variable(
        self, action_object_variable: SituationObject
    ) -> ObjectPerception:
        """
        Binds an action object variable to an object that we have perceived.

        Currently this only pays attention to object properties in binding and not relationships.
        """
        # we continue to use the hand from PUT
        # ( see _bind_action_objects_variables_to_perceived_objects )
        ontology = self._generator.ontology
        perceived_objects_matching_constraints = [
            object_perception
            for (
                object_perception,
                ontology_node,
            ) in self._object_perceptions_to_ontology_nodes.items()
            if ontology.has_all_properties(
                ontology_node, action_object_variable.properties
            )
        ]
        if len(perceived_objects_matching_constraints) == 1:
            return only(perceived_objects_matching_constraints)
        elif not perceived_objects_matching_constraints:
            raise RuntimeError(
                f"Can not find object with properties {action_object_variable}"
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
                    f"{perceived_objects_matching_constraints}"
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
            # Region is not a real possibility here, but using this type lets us put the cast
            # in only one place.
            perceived_reference_object: Union[ObjectPerception, Region[ObjectPerception]]
            if slot_filler.reference_object in self._objects_to_perceptions:
                # this handles the case of a region that the user has explicitly bound
                # to something in the situation, like table in put(theme=book, goal=on(table))
                perceived_reference_object = self._objects_to_perceptions[
                    slot_filler.reference_object
                ]
            else:
                perceived_reference_object = action_object_variables_to_object_perceptions[
                    slot_filler.reference_object
                ]
            return Region(
                # this will be an ObjectPerception by construction of the maps
                reference_object=cast(ObjectPerception, perceived_reference_object),
                direction=slot_filler.direction,
                distance=slot_filler.distance,
            )
        else:
            return action_object_variables_to_object_perceptions[slot_filler]

    def _perceive_property_assertions(self) -> None:
        for situation_object in self._situation.objects:
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

            # We wrap an ImmutableSet around properties_to_perceive to remove duplicates
            # while still guaranteeing deterministic iteration order.
            for property_ in immutableset(properties_to_perceive):
                self._perceive_property(
                    self._generator.ontology.properties_for_node(property_),
                    self._objects_to_perceptions[situation_object],
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
        for situation_object in self._situation.objects:
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

            self._instantiate_object_schema(
                only(object_schemata), situation_object=situation_object
            )

    def _instantiate_object_schema(
        self,
        schema: ObjectStructuralSchema,
        *,
        # if the object being instantiated corresponds to an object
        # in the situation description, then this will track that object
        situation_object: Optional[SituationObject] = None,
    ) -> ObjectPerception:
        root_object_perception = ObjectPerception(
            debug_handle=self._object_handle_generator.subscripted_handle(schema)
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
            ] = sub_object.schema.parent_object
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
                arg2_perception = Region(
                    sub_object_to_object_perception[arg2.reference_object],
                    distance=arg2.distance,
                    direction=arg2.direction,
                )
            self._relation_perceptions.append(
                Relation(
                    sub_object_relation.relation_type, arg1_perception, arg2_perception
                )
            )
        return root_object_perception


GAILA_PHASE_1_PERCEPTION_GENERATOR = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
    GAILA_PHASE_1_ONTOLOGY
)


@attrs(auto_exc=True, auto_attribs=True)
class TooManySpeakersException(RuntimeError):
    msg: str
