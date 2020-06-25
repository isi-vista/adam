from enum import Enum, auto
from itertools import chain
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Union, cast

from more_itertools import only, quantify
from networkx import DiGraph

from adam.axes import AxesInfo, WORLD_AXES
from adam.axis import GeonAxis
from adam.geon import Geon
from adam.ontology import (
    BINARY,
    IN_REGION,
    IS_SPEAKER,
    IS_SUBSTANCE,
    OntologyNode,
    PERCEIVABLE,
)
from adam.ontology.action_description import ActionDescription, ActionDescriptionVariable
from adam.ontology.during import DuringAction
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import (
    ABOUT_THE_SAME_SIZE_AS_LEARNER,
    BABY,
    COLOR,
    COLORS_TO_RGBS,
    GAILA_PHASE_1_ONTOLOGY,
    GAZED_AT,
    GROUND,
    HOLLOW,
    LEARNER,
    LIQUID,
    PART_OF,
    SIZE_RELATIONS,
    TWO_DIMENSIONAL,
    on,
)
from adam.ontology.phase1_spatial_relations import (
    EXTERIOR_BUT_IN_CONTACT,
    INTERIOR,
    Region,
    SpatialPath,
)
from adam.ontology.phase2_ontology import GAILA_PHASE_2_ONTOLOGY
from adam.ontology.structural_schema import ObjectStructuralSchema, SubObject
from adam.perception import (
    GROUND_PERCEPTION,
    LEARNER_PERCEPTION,
    ObjectPerception,
    PerceptualRepresentation,
    PerceptualRepresentationGenerator,
    RegionPerception,
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
from adam.situation import Action, SituationObject, SituationRegion
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from attr import Factory, attrib, attrs
from attr.validators import deep_mapping, instance_of
from immutablecollections import (
    ImmutableDict,
    ImmutableSet,
    ImmutableSetMultiDict,
    immutabledict,
    immutableset,
    immutablesetmultidict,
)
from immutablecollections.converter_utils import _to_immutabledict
from vistautils.preconditions import check_arg


class ColorPerceptionMode(Enum):
    """
    Used as a field on `HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator`
    to indicate how colors should be perceived.
    """

    CONTINUOUS = auto()
    """
    Perceive colors are RGB triples
    """
    DISCRETE = auto()
    """
    Perceive colors as discrete categories.
    """


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

    color_perception_mode: ColorPerceptionMode = attrib(
        validator=instance_of(ColorPerceptionMode), default=ColorPerceptionMode.CONTINUOUS
    )

    def generate_perception(
        self,
        situation: HighLevelSemanticsSituation,
        chooser: SequenceChooser,
        *,
        include_ground=True,
    ) -> PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]:
        check_arg(
            situation.ontology == self.ontology,
            "Cannot generate perceptions "
            "for a situation with a mis-matched "
            "ontology.",
        )
        # all the work is done in a stateful _PerceptionGeneration object
        return _PerceptionGeneration(self, situation, chooser, include_ground).do()


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

    def subscripted_handle(self, ontology_node: OntologyNode) -> str:
        unsubscripted_handle = ontology_node.handle
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
    _include_ground = attrib(validator=instance_of(bool))
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
    _regions_to_perceptions: Dict[SituationRegion, RegionPerception] = attrib(
        init=False, default=Factory(dict)
    )
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
        # Once all the objects and relations are perceived, determine their colors.
        self._perceive_colors()

        # Handle implicit size relations
        self._perceive_size_relative_to_learner()
        # self._perceive_implicit_size()

        # for now, we assume that actions do not alter the relationship of objects axes
        # to the speaker, learner, and addressee
        axis_info = self._perceive_axis_info()

        # Other relations implied by actions will be handled during action translation below.

        if not self._situation.actions:
            if self._include_ground:
                self._relation_perceptions.extend(
                    self._perceive_ground_relations(self._relation_perceptions)
                )
            return PerceptualRepresentation.single_frame(
                DevelopmentalPrimitivePerceptionFrame(
                    perceived_objects=self._object_perceptions,
                    relations=self._relation_perceptions,
                    property_assertions=self._property_assertion_perceptions,
                    axis_info=axis_info,
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

        # Add ground for always perception if needed
        if self._include_ground:
            # Grabbing any "always" ground relations
            self._relation_perceptions.extend(
                self._perceive_ground_relations(
                    relations=chain(
                        self._relation_perceptions,
                        explicit_before_relations,
                        _action_perception.before_relations,
                        explicit_after_relations,
                        _action_perception.after_relations,
                    )
                )
            )
            # Due to the presence of an action need to specify implicit ground relations before and
            # after the action seperately.
            before_ground = self._perceive_ground_relations(
                relations=chain(
                    self._relation_perceptions,
                    explicit_before_relations,
                    _action_perception.before_relations,
                )
            )
            after_ground = self._perceive_ground_relations(
                relations=chain(
                    self._relation_perceptions,
                    explicit_after_relations,
                    _action_perception.after_relations,
                )
            )

        first_frame = DevelopmentalPrimitivePerceptionFrame(
            perceived_objects=self._object_perceptions,
            relations=[
                rel
                for rel in chain(
                    self._relation_perceptions,
                    explicit_before_relations,
                    _action_perception.before_relations,
                    before_ground,
                )
                if not rel.negated
            ],
            property_assertions=self._property_assertion_perceptions,
            axis_info=axis_info,
        )
        second_frame = DevelopmentalPrimitivePerceptionFrame(
            perceived_objects=self._object_perceptions,
            relations=[
                rel
                for rel in chain(
                    self._relation_perceptions,
                    explicit_after_relations,
                    _action_perception.after_relations,
                    after_ground,
                )
                if not rel.negated
            ],
            property_assertions=self._property_assertion_perceptions,
            axis_info=axis_info,
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

    def _perceive_objects(self) -> None:
        if (
            not any(
                situation_object.ontology_node == GROUND
                for situation_object in self._situation.all_objects
            )
            and self._include_ground
        ):
            self._perceive_object(
                SituationObject.instantiate_ontology_node(
                    GROUND, ontology=self._generator.ontology
                )
            )
        for situation_object in self._situation.all_objects:
            self._perceive_object(situation_object)

    def _perceive_object(self, situation_object: SituationObject) -> None:
        if not situation_object.ontology_node:
            raise RuntimeError(
                "Don't yet know how to handle situation objects without "
                "associated ontology nodes"
            )

        perceived_object: ObjectPerception

        if situation_object.ontology_node == GROUND:
            perceived_object = GROUND_PERCEPTION
        elif situation_object.ontology_node == LEARNER:
            perceived_object = LEARNER_PERCEPTION
        else:
            # These are the possible internal structures of objects of this type
            # that the ontology is aware of.
            object_schemata = self._generator.ontology.structural_schemata(
                situation_object.ontology_node
            )
            if len(object_schemata) > 1:
                # TODO: add issue for this
                # https://github.com/isi-vista/adam/issues/87
                raise RuntimeError(
                    f"Support for objects with multiple structural schemata has not "
                    f"yet been implemented."
                )
            if object_schemata:
                # We know the object's structure.
                # It might have complicated internal structure,
                # which we will recursively instantiate.
                perceived_object = self._instantiate_object_schema(
                    only(object_schemata), situation_object=situation_object
                ).instantiated_object
            else:
                if self._generator.ontology.has_property(
                    situation_object.ontology_node, IS_SUBSTANCE
                ):
                    # it is okay for a substance like MILK to lack any internal structure
                    perceived_object = ObjectPerception(
                        debug_handle=self._object_handle_generator.subscripted_handle(
                            situation_object.ontology_node
                        ),
                        geon=None,
                        axes=situation_object.axes,
                    )
                else:
                    raise RuntimeError(
                        f"No structural schema found for {situation_object}"
                    )
        self._object_perceptions.append(perceived_object)
        self._objects_to_perceptions[situation_object] = perceived_object
        self._object_perceptions_to_ontology_nodes[
            perceived_object
        ] = situation_object.ontology_node

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

    def _perceive_region(self, region: SituationRegion) -> RegionPerception:
        return region.copy_remapping_objects(self._objects_to_perceptions)

    def _perceive_property_assertions(self) -> None:
        # Situation objects require some special logic, so we need to be able to (1) check if a
        # perception came from a situation object, and (2) get the situation object if so.
        object_perceptions_to_situation_objects: ImmutableDict[
            ObjectPerception, SituationObject
        ] = immutabledict(
            [
                (self._objects_to_perceptions[situation_object], situation_object)
                for situation_object in self._situation.all_objects
            ]
        )
        for (
            object_perception,
            ontology_node,
        ) in self._object_perceptions_to_ontology_nodes.items():
            # process explicitly and implicitly-specified properties
            all_object_properties: List[OntologyNode] = []
            # Explicit properties are stipulated by the user in the situation description.
            if object_perception in object_perceptions_to_situation_objects:
                situation_object = object_perceptions_to_situation_objects[
                    object_perception
                ]
                all_object_properties.extend(situation_object.properties)
            # Implicit properties are derived from what type of thing an object is,
            # e.g. that people are ANIMATE.
            all_object_properties.extend(
                self._generator.ontology.properties_for_node(ontology_node)
            )

            # Colors require special processing, so we ignore them for now
            # and handle them in `_perceive_colors`.
            properties_to_perceive = [
                property_
                for property_ in all_object_properties
                if not self._generator.ontology.is_subtype_of(property_, COLOR)
            ]

            # If it is a liquid not inside a container, add two-dimensional property
            if (
                LIQUID
                in GAILA_PHASE_2_ONTOLOGY.properties_for_node(
                    ontology_node
                    # TODO: Handle non-explicit liquid objects (subobjects, etc.)
                )
                and (object_perception in object_perceptions_to_situation_objects)
                and not any(
                    r.first_slot
                    == object_perceptions_to_situation_objects[object_perception]
                    and r.relation_type == IN_REGION
                    and isinstance(r.second_slot, Region)
                    and HOLLOW
                    in GAILA_PHASE_2_ONTOLOGY.properties_for_node(
                        r.second_slot.reference_object.ontology_node
                    )
                    and r.second_slot.distance == INTERIOR
                    for r in self._situation.always_relations
                )
            ):
                properties_to_perceive.append(TWO_DIMENSIONAL)

            # Logic relevant only to situation_objects, such as gaze handling
            if object_perception in object_perceptions_to_situation_objects:
                # Focused Objects are in a special field of the Situation, we check if the situation_object
                # is a focused and apply the tag here if that is the case.
                situation_object = object_perceptions_to_situation_objects[
                    object_perception
                ]
                if situation_object in self._situation.gazed_objects:
                    properties_to_perceive.append(GAZED_AT)

            # We wrap an ImmutableSet around properties_to_perceive to remove duplicates
            # while still guaranteeing deterministic iteration order.
            for property_ in immutableset(properties_to_perceive):
                self._perceive_property(
                    self._generator.ontology.properties_for_node(property_),
                    object_perception,
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
                self._property_assertion_perceptions.append(perceived_property)
            elif self._generator.ontology.is_subtype_of(property_, COLOR):
                pass
            else:
                raise RuntimeError(
                    f"Not sure how to generate perception for property {property_} "
                    f"which is marked as perceivable"
                )

    def _perceive_relation(
        self, relation: Relation[SituationObject]
    ) -> Relation[ObjectPerception]:
        return relation.copy_remapping_objects(self._objects_to_perceptions)

    def _perceive_colors(self) -> None:
        # Create a graph representing each object and their sub-objects
        object_graph = DiGraph()
        root = ObjectPerception("root", axes=WORLD_AXES)
        object_graph.add_node(root)
        perceptions_to_situation_objects: Mapping[ObjectPerception, SituationObject] = {
            self._objects_to_perceptions[situation_object]: situation_object
            for situation_object in self._situation.all_objects
        }
        ontology_nodes_to_colors: MutableMapping[
            OntologyNode, Union[RgbColorPerception, OntologyNode]
        ] = {}
        visited = set()

        for situation_object in self._situation.all_objects:
            object_perception = self._objects_to_perceptions[situation_object]
            object_graph.add_node(object_perception)
            object_graph.add_edge(root, object_perception)

        for object_relation in self._relation_perceptions:
            if object_relation.relation_type == PART_OF:
                object_graph.add_edge(
                    object_relation.second_slot, object_relation.first_slot
                )

        def assert_color(
            object_perception, color: Union[RgbColorPerception, OntologyNode]
        ):
            """
            Append a color property to the list of property assertion perceptions
            """
            if isinstance(color, RgbColorPerception):
                self._property_assertion_perceptions.append(
                    HasColor(object_perception, color)
                )
            else:
                self._property_assertion_perceptions.append(
                    HasBinaryProperty(object_perception, color)
                )

        def handle_color_property(
            object_perception, color_node: OntologyNode
        ) -> Union[RgbColorPerception, OntologyNode]:
            """
            Convert a color property to an RgbColorPerception if needed
            before calling `assert_color`
            """
            if self._generator.color_perception_mode == ColorPerceptionMode.CONTINUOUS:
                if color_node in COLORS_TO_RGBS.keys():
                    color_options = COLORS_TO_RGBS[color_node]
                    if color_options:
                        r, g, b = self._chooser.choice(color_options)
                        rgb_color = RgbColorPerception(r, g, b)
                        assert_color(object_perception, rgb_color)
                        return rgb_color
                    else:  # Handles the case of TRANSPARENT
                        assert_color(object_perception, color_node)
                        return color_node
                else:
                    raise RuntimeError(
                        f"Not sure how to generate perception for the unknown property "
                        f"{color_node} "
                        f"which is marked as COLOR"
                    )
            elif self._generator.color_perception_mode == ColorPerceptionMode.DISCRETE:
                assert_color(object_perception, color_node)
                return color_node
            else:
                raise RuntimeError(
                    f"Unknown color perception mode {self._generator.color_perception_mode}"
                )

        def dfs_walk(node: ObjectPerception, inherited_color=None):
            visited.add(node)
            if not node == root:
                # Begin new component
                if object_graph.has_edge(root, node):
                    ontology_nodes_to_colors.clear()
                    inherited_color = None

                if node in perceptions_to_situation_objects:
                    node_situation_object = perceptions_to_situation_objects[node]
                    ontology_node = node_situation_object.ontology_node
                    explicitly_specified_colors: Optional[
                        ImmutableSet[OntologyNode]
                    ] = immutableset(
                        property_
                        for property_ in node_situation_object.properties
                        if self._generator.ontology.is_subtype_of(property_, COLOR)
                    )
                    prototypical_colors = immutableset(
                        property_
                        for property_ in self._generator.ontology.properties_for_node(
                            ontology_node
                        )
                        if self._generator.ontology.is_subtype_of(property_, COLOR)
                    )
                else:
                    ontology_node = self._object_perceptions_to_ontology_nodes[node]
                    # colors of sub-objects cannot be explicitly specified
                    explicitly_specified_colors = None
                    prototypical_colors = immutableset(
                        property_
                        for property_ in self._generator.ontology.properties_for_node(
                            ontology_node
                        )
                        if self._generator.ontology.is_subtype_of(property_, COLOR)
                    )

                # If a color is explicitly defined, use that color
                if explicitly_specified_colors:
                    if len(explicitly_specified_colors) == 1:
                        inherited_color = handle_color_property(
                            node, only(explicitly_specified_colors)
                        )
                    else:
                        raise RuntimeError(
                            "Cannot have multiple explicit colors on an object."
                        )
                # If an object's type already has a color associated with it,
                # assign that color to the object
                elif ontology_node in ontology_nodes_to_colors:
                    assert_color(node, ontology_nodes_to_colors[ontology_node])
                # If the object has a set of prototypical colors,
                # assign a random one to the object
                elif prototypical_colors:
                    color_choice = self._chooser.choice(prototypical_colors)
                    object_color = handle_color_property(node, color_choice)
                    ontology_nodes_to_colors[ontology_node] = object_color
                    inherited_color = object_color
                # Otherwise if an inherited color is defined, assign it to the sub-object
                elif inherited_color:
                    assert_color(node, inherited_color)
                # If we cannot determine an object's color,
                # we don't assign one
            elif not object_graph.successors(node):
                RuntimeError(f"Error while perceiving colors - {node} is not a tree")
            for successor in object_graph.successors(node):
                if successor not in visited:
                    dfs_walk(successor, inherited_color)

        dfs_walk(root)

    def _perceive_size_relative_to_learner(self) -> None:
        """
        When doing object recognition,
        size relations relative to the learner play a special role.
        Since relations with other objects are currently not examined
        by our object recognition algorithms,
        it is easier if we represent this as a property as well as a relation.
        """
        for perception, ontology_type in self._object_perceptions_to_ontology_nodes.items():
            size_relations = immutableset(
                relation
                for relation in self._situation.ontology.subjects_to_relations[
                    ontology_type
                ]
                if relation.relation_type in SIZE_RELATIONS
                and relation.second_slot == BABY
            )
            if size_relations:
                if len(size_relations) > 1:
                    raise RuntimeError(
                        f"Expected only one size relations for "
                        f"{ontology_type} but got {size_relations}"
                    )
                self._property_assertion_perceptions.append(
                    HasBinaryProperty(
                        perception,
                        only(size_relations).relation_type,
                    )
                )
            else:
                self._property_assertion_perceptions.append(
                    HasBinaryProperty(
                        perception,
                        ABOUT_THE_SAME_SIZE_AS_LEARNER,
                    )
                )

    def _perceive_axis_info(self) -> AxesInfo[ObjectPerception]:
        return self._situation.axis_info.copy_remapping_objects(
            self._objects_to_perceptions
        )

    def _perceive_ground_relations(
        self, relations: Iterable[Relation[ObjectPerception]]
    ) -> ImmutableSet[Relation[ObjectPerception]]:
        objects_to_relations = self._objects_to_relations(relations)
        ground_relations: List[Relation[ObjectPerception]] = []
        perceived_ground: Optional[ObjectPerception] = None
        for object_ in self._object_perceptions:
            if self._object_perceptions_to_ontology_nodes[object_] == GROUND:
                perceived_ground = object_
                break
        if not perceived_ground:
            raise RuntimeError("Couldn't find the ground.")

        for situation_object in self._situation.all_objects:
            if situation_object.ontology_node != GROUND:
                object_perception = self._objects_to_perceptions[situation_object]

                add_on_ground = True

                if object_perception in objects_to_relations:
                    # If this object is not on anything else, it should be on the ground,
                    # unless it's explicitly specified to be unsupported
                    for relation in objects_to_relations[object_perception]:
                        if relation.relation_type == IN_REGION and isinstance(
                            relation.second_slot, Region
                        ):
                            region = relation.second_slot
                            if (
                                relation.negated
                                and region.distance == EXTERIOR_BUT_IN_CONTACT
                                and region.reference_object == perceived_ground
                            ):
                                # Don't make something in contact with the ground
                                # if the situation explicitly says it isn't.
                                add_on_ground = False
                                self._relation_perceptions.remove(relation)
                            elif (
                                region.distance == EXTERIOR_BUT_IN_CONTACT
                                or region.distance == INTERIOR
                                or region.reference_object == perceived_ground
                            ) and not relation.negated:
                                # Anything else in contact with anything else is not on the ground.
                                # TODO: This is too lax:
                                # see https://github.com/isi-vista/adam/issues/597
                                # Also, don't duplicate explicit ground contact relations,
                                # and don't contradict distal/proximal relations with the ground.
                                add_on_ground = False

                if add_on_ground:
                    ground_relations.extend(on(object_perception, perceived_ground))

        return immutableset(ground_relations)

    def _objects_to_relations(
        self, relations: Iterable[Relation[ObjectPerception]]
    ) -> ImmutableSetMultiDict[ObjectPerception, Relation[ObjectPerception]]:
        return immutablesetmultidict(
            (relation.first_slot, relation) for relation in relations
        )

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

        return _PerceptionGeneration._ActionPerception(
            before_relations=immutableset(chain(enduring_relations, before_relations)),
            after_relations=immutableset(chain(enduring_relations, after_relations)),
            during_action=action_description.during.copy_remapping_objects(
                cast(
                    Mapping[ActionDescriptionVariable, ObjectPerception],
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
    ) -> Mapping[ActionDescriptionVariable, Union[RegionPerception, ObjectPerception]]:
        if any(
            len(fillers) > 1
            for fillers in situation_action.argument_roles_to_fillers.value_groups()
        ):
            raise RuntimeError("Cannot handle multiple fillers for an argument role yet.")

        bindings: Dict[
            ActionDescriptionVariable, Union[ObjectPerception, RegionPerception]
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
        action_variables_from_non_frames: List[ActionDescriptionVariable] = []
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
            and isinstance(action_variable, ActionDescriptionVariable)
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
        action_object_variable: ActionDescriptionVariable,
    ) -> Union[ObjectPerception, RegionPerception]:
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
        conditions: ImmutableSet[Relation[ActionDescriptionVariable]],
        *,
        action_object_variables_to_object_perceptions: Mapping[
            ActionDescriptionVariable, Union[ObjectPerception, RegionPerception]
        ],
    ) -> ImmutableSet[Relation[ObjectPerception]]:
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

            relations.append(
                Relation(
                    relation_type=condition.relation_type,
                    first_slot=perception_1,
                    second_slot=perception_2,
                    negated=condition.negated,
                )
            )
        return immutableset(relations)

    def _perceive_object_or_region_relation_filler(
        self,
        slot_filler: Union[ActionDescriptionVariable, Region[ActionDescriptionVariable]],
        *,
        action_object_variables_to_object_perceptions: Mapping[
            ActionDescriptionVariable, Union[ObjectPerception, RegionPerception]
        ],
    ) -> Union[ObjectPerception, RegionPerception]:
        if isinstance(slot_filler, Region):
            object_mapping: Dict[
                Union[SituationObject, ActionDescriptionVariable], ObjectPerception
            ] = {}
            # regions are not a real possibility for lookup,
            # so mypy's complaints here are irrelevant
            object_mapping.update(self._objects_to_perceptions)  # type: ignore
            object_mapping.update(  # type: ignore
                action_object_variables_to_object_perceptions  # type: ignore
            )

            return slot_filler.copy_remapping_objects(
                cast(Mapping[ActionDescriptionVariable, ObjectPerception], object_mapping)
            )
        else:
            return action_object_variables_to_object_perceptions[slot_filler]

    @attrs(frozen=True, slots=True)
    class _InstantiateObjectSchemaReturn:
        instantiated_object: ObjectPerception = attrib(
            validator=instance_of(ObjectPerception)
        )
        schema_axes_to_perceivable_axes: Mapping[GeonAxis, GeonAxis] = attrib(
            converter=_to_immutabledict,
            validator=deep_mapping(instance_of(GeonAxis), instance_of(GeonAxis)),
        )

    def _instantiate_object_schema(
        self,
        schema: ObjectStructuralSchema,
        *,
        # if the object being instantiated corresponds to an object
        # in the situation description, then this will track that object
        situation_object: Optional[SituationObject] = None,
    ) -> "_PerceptionGeneration._InstantiateObjectSchemaReturn":
        """
        Creates an `ObjectPerception` (and other associated structures) according to
        *schema*.

        Recall that an `ObjectStructuralSchema` is an abstract description of the shape
        of a general category of object (e.g. a chair in general).

        If we know this object corresponds to a top-level object in a `Situation`,
        the *situation_object* should be provided so we can give it special handling.

        This returns not only the created `ObjectPerception`,
        but also a mapping from the abstract axes of this object's schema
        to the concrete axes associated with the object itself.
        """

        # Compute a human-readable handle for this object, for debugging.
        # Use the debug handle from the situation object if it is available, as it is more
        # specific in the case of people (e.g. mom, dad, baby) than the debug handle
        # generated from the ObjectStructuralSchema
        if situation_object:
            debug_handle = self._object_handle_generator.subscripted_handle(
                situation_object.ontology_node
            )
        else:
            debug_handle = self._object_handle_generator.subscripted_handle(
                schema.ontology_node
            )

        # In the if-block below we are going to try to determine two things:
        #  (a) whether there is a perceivable geon (=shape) associated with this object
        #  (b) what the perceivable axes associated with this object are.
        #  (c) what the mapping between perceivable axes and schema axes is
        #
        # (c) requires a little explanation.
        # Abstract relations between objects specified by an object schema
        # (e.g. front chair legs are in front of back chair legs relative
        # to the front-back axis of a chair)
        # need to be translated to perceivable relations between object perceptions
        # (we do this towards the end of this method).
        # However, these relations may be specified in terms of abstract axes
        # used in the schema of this object or any of its immediate sub-objects.
        # We need to learn a mapping from these schema axes to perceivable axes
        # in order to translate relations later.
        top_level_schema_axes_to_perceivable_axes: Mapping[GeonAxis, GeonAxis]
        concrete_geon: Optional[Geon]

        if situation_object:
            # This object corresponds to a top-level object refered to by a situation
            # (e.g. a person, a table).
            if schema.geon:
                # This is an object with a known shape.
                # We copy that "abstract shape" to use as our "perceivable shape"
                # (they are currently the same type of object,
                # though it would be clearer if we distinguished them in the type system).
                # Normally when we copy a geon, its axes are copied to,
                # but in this case we want to make sure it inherit the axes
                # of the situation object itself in case the users referes to them
                # in the situation definition.
                concrete_geon = schema.geon.copy(
                    axis_mapping=situation_object.schema_axis_to_object_axis
                )
                top_level_schema_axes_to_perceivable_axes = (
                    situation_object.schema_axis_to_object_axis
                )
            else:
                # This is a top-level object, but it lacks a shape
                # (e.g. it is a substance like "water")
                concrete_geon = None
                # These also shouldn't have any axis relations,
                # so we don't track an axis mapping.
                top_level_schema_axes_to_perceivable_axes = immutabledict()
            axes = situation_object.axes
        else:
            # This object corresponds to a sub-object of some other situation object
            # (e.g. a person's arm, a table surface).
            if schema.geon:
                top_level_schema_axes_to_perceivable_axes = {}
                concrete_geon = schema.geon.copy(
                    output_axis_mapping=top_level_schema_axes_to_perceivable_axes
                )
                # When the geon was copied, its axes were already copied,
                # which is why we don't do it again.
                axes = concrete_geon.axes
            else:
                concrete_geon = None
                # There is no geon or situation object we need to match up with
                # so we just copy the axes directly.
                top_level_schema_axes_to_perceivable_axes = {
                    schema_axis: schema_axis.copy()
                    for schema_axis in schema.axes.all_axes
                }
                axes = schema.axes.remap_axes(top_level_schema_axes_to_perceivable_axes)

        # This is the actual perception of the object which we will return.
        root_object_perception = ObjectPerception(
            debug_handle=debug_handle, geon=concrete_geon, axes=axes
        )

        # Recursively instantiate sub-components of this object.
        # Because sub-object relations can refer to the axes of *immediate* sub-objects,
        # we need to track the relation between their schema and perceived axes
        # as well for sub-object relation translation.
        top_level_and_subobject_schema_axes_to_perceivable_axes: Dict[
            GeonAxis, GeonAxis
        ] = {}
        top_level_and_subobject_schema_axes_to_perceivable_axes.update(
            top_level_schema_axes_to_perceivable_axes
        )

        sub_object_to_object_perception: Dict[SubObject, ObjectPerception] = {}
        for sub_object in schema.sub_objects:
            instantiation_result = self._instantiate_object_schema(sub_object.schema)
            sub_object_to_object_perception[
                sub_object
            ] = instantiation_result.instantiated_object
            top_level_and_subobject_schema_axes_to_perceivable_axes.update(
                instantiation_result.schema_axes_to_perceivable_axes
            )

        for sub_object in schema.sub_objects:
            sub_object_perception = sub_object_to_object_perception[sub_object]
            self._object_perceptions.append(sub_object_perception)
            self._object_perceptions_to_ontology_nodes[
                sub_object_perception
            ] = sub_object.schema.ontology_node
            # every sub-component has an implicit partOf relationship to its parent object.
            self._relation_perceptions.append(
                Relation(PART_OF, sub_object_perception, root_object_perception)
            )

        # Translate sub-object relations specified by the object's structural schema.
        for sub_object_relation in schema.sub_object_relations:
            # TODO: right now we translate all situation relations directly to perceptual
            # relations without modification. This is not always the right thing.
            # See https://github.com/isi-vista/adam/issues/80 .
            arg1_perception = sub_object_to_object_perception[
                sub_object_relation.first_slot
            ]
            arg2_perception: Union[ObjectPerception, RegionPerception]
            if isinstance(sub_object_relation.second_slot, SubObject):
                arg2_perception = sub_object_to_object_perception[
                    sub_object_relation.second_slot
                ]
            else:
                arg2 = sub_object_relation.second_slot
                arg2_perception = arg2.copy_remapping_objects(
                    sub_object_to_object_perception,
                    axis_mapping=top_level_and_subobject_schema_axes_to_perceivable_axes,
                )
            self._relation_perceptions.append(
                Relation(
                    sub_object_relation.relation_type, arg1_perception, arg2_perception
                )
            )
        return _PerceptionGeneration._InstantiateObjectSchemaReturn(
            instantiated_object=root_object_perception,
            # Axes are only "visible" for relations one layer down,
            # which is why we return only the top-level axis mapping.
            schema_axes_to_perceivable_axes=top_level_schema_axes_to_perceivable_axes,
        )

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


GAILA_PHASE_1_PERCEPTION_GENERATOR = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
    GAILA_PHASE_1_ONTOLOGY
)
GAILA_PHASE_2_PERCEPTION_GENERATOR = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
    GAILA_PHASE_2_ONTOLOGY
)

GAILA_M6_PERCEPTION_GENERATOR = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
    GAILA_PHASE_1_ONTOLOGY, color_perception_mode=ColorPerceptionMode.DISCRETE
)
"""
This is the same as `GAILA_PHASE_1_PERCEPTION_GENERATOR`,
but it uses a discrete color representation.
After modifier learning is complete we will switch to the full perception generator.
"""


@attrs(auto_exc=True, auto_attribs=True)
class TooManySpeakersException(RuntimeError):
    msg: str
