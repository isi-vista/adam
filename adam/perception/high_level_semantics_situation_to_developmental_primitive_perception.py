from typing import Dict, List, Optional

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableDict, immutabledict
from more_itertools import only
from vistautils.preconditions import check_arg

from adam.ontology import ObjectStructuralSchema, Ontology, SubObject
from adam.ontology.phase1_ontology import PART_OF, PERCEIVABLE, BINARY, COLOR
from adam.perception import PerceptualRepresentation, PerceptualRepresentationGenerator
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    ObjectPerception,
    RelationPerception,
    PropertyPerception,
    HasBinaryProperty,
)
from adam.random_utils import SequenceChooser
from adam.situation import HighLevelSemanticsSituation, SituationObject


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
    _object_handle_generator: _ObjectHandleGenerator = attrib(
        init=False, default=Factory(_ObjectHandleGenerator)
    )
    """
    Used for tracking sub-scripts of objects.
    """
    _relation_perceptions: List[RelationPerception] = attrib(
        init=False, default=Factory(list)
    )
    r"""
    `RelationPerception`\ s perceived by the learner.
    """
    _property_assertion_perceptions: List[PropertyPerception] = attrib(
        init=False, default=Factory(list)
    )
    r"""
    `PropertyPerception`\ s perceived by the learner.
    """

    def do(self) -> PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]:
        # The first step is to determine what objects are perceived.
        self._perceive_objects()

        # TODO: translate property assertions
        # https://github.com/isi-vista/adam/issues/85
        self._perceive_property_assertions()

        # TODO: translate actions
        # https://github.com/isi-vista/adam/issues/86
        return PerceptualRepresentation.single_frame(
            DevelopmentalPrimitivePerceptionFrame(
                perceived_objects=self._object_perceptions,
                relations=self._relation_perceptions,
                property_assertions=self._property_assertion_perceptions,
            )
        )

    def _perceive_property_assertions(self) -> None:
        for situation_object in self._situation.objects:
            # Add the perceivable properties of each situation object into the perception
            object_properties_from_ontology = self._generator.ontology.properties_for_node(
                situation_object.ontology_node
            )
            for property_ in object_properties_from_ontology.union(
                situation_object.properties
            ):
                # for each property such as animate, sentient, etc
                attributes_of_property = self._generator.ontology.properties_for_node(
                    property_
                )
                # e.g. is this a perceivable property, binary property, color, etc...
                if PERCEIVABLE in attributes_of_property:
                    perceived_object = self._objects_to_perceptions[situation_object]
                    # Convert the property (which as an OntologyNode object) into PropertyPerception object
                    if BINARY in attributes_of_property:
                        perceived_property = HasBinaryProperty(
                            perceived_object, property_
                        )
                    elif COLOR in attributes_of_property:
                        # TODO: issue: generate perception of colors
                        raise RuntimeError("Fix this issue COLOR property")
                    else:
                        raise RuntimeError(
                            f"Not sure how to generate perception for property {property_} "
                            f"which is marked as perceivable"
                        )
                    self._property_assertion_perceptions.append(perceived_property)

    def _perceive_objects(self) -> None:
        for situation_object in self._situation.objects:
            if not situation_object.ontology_node:
                raise RuntimeError(
                    "Don't yet know how to handle situation objects without "
                    "associated ontology nodes"
                )
            # these are the possible internal structures of objects of this type
            # that the ontology is aware of.
            object_schemata = self._generator.ontology.structural_schemata[
                situation_object.ontology_node
            ]
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
            # every sub-component has an implicit partOf relationship to its parent object.
            self._relation_perceptions.append(
                RelationPerception(PART_OF, root_object_perception, sub_object_perception)
            )

        # translate sub-object relations specified by the object's strucural schema
        for sub_object_relation in schema.sub_object_relations:
            # TODO: right now we translate all situation relations directly to perceptual
            # relations without modification. This is not always the right thing.
            # See https://github.com/isi-vista/adam/issues/80 .
            self._relation_perceptions.append(
                RelationPerception(
                    sub_object_relation.relation_type,
                    sub_object_to_object_perception[sub_object_relation.arg1],
                    sub_object_to_object_perception[sub_object_relation.arg2],
                )
            )
        return root_object_perception
