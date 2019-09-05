from typing import Dict, List, Optional

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableDict, immutabledict
from more_itertools import only

from adam.ontology import ObjectStructuralSchema, Ontology, SubObject
from adam.ontology.phase1_ontology import PART_OF
from adam.perception import PerceptualRepresentation, PerceptualRepresentationGenerator
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
    ObjectPerception,
    RelationPerception,
)
from adam.random_utils import SequenceChooser
from adam.situation import HighLevelSemanticsSituation, SituationObject


@attrs(frozen=True, slots=True)
class HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
    PerceptualRepresentationGenerator[
        HighLevelSemanticsSituation, DevelopmentalPrimitivePerceptionFrame
    ]
):

    ontology: Ontology = attrib(validator=instance_of(Ontology))

    def generate_perception(
        self, situation: HighLevelSemanticsSituation, chooser: SequenceChooser
    ) -> PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]:
        return _PerceptionGeneration(self, situation, chooser).do()


@attrs(frozen=True, slots=True)
class _PerceptionGeneration:
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
    _object_perceptions: List[ObjectPerception] = attrib(
        init=False, default=Factory(list)
    )
    _object_handles_seen: List[str] = attrib(init=False, default=Factory(list))
    _relation_perceptions: List[RelationPerception] = attrib(
        init=False, default=Factory(list)
    )

    def do(self) -> PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]:
        self._map_objects()
        # TODO: property assertions
        # TODO: translate actions
        return PerceptualRepresentation.single_frame(
            DevelopmentalPrimitivePerceptionFrame(
                perceived_objects=self._object_perceptions,
                relations=self._relation_perceptions,
            )
        )

    def _map_objects(self) -> None:
        for situation_object in self._situation.objects:
            if not situation_object.ontology_node:
                raise RuntimeError(
                    "Don't yet know how to handle situation objects without "
                    "associated ontology nodes"
                )
            object_schemata = self._generator.ontology.structural_schemata[
                situation_object.ontology_node
            ]
            if not object_schemata:
                raise RuntimeError(f"No structural schema found for {situation_object}")
            if len(object_schemata) > 1:
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
        situation_object: Optional[SituationObject] = None,
    ) -> ObjectPerception:
        unindexed_handle = schema.parent_object.handle
        subscript = self._object_handles_seen.count(unindexed_handle)
        indexed_handle = f"{unindexed_handle}_{subscript}"
        root_object_perception = ObjectPerception(indexed_handle)
        self._object_handles_seen.append(unindexed_handle)
        self._object_perceptions.append(root_object_perception)
        if situation_object:
            self._objects_to_perceptions[situation_object] = root_object_perception
        sub_object_to_object_perception: ImmutableDict[
            SubObject, ObjectPerception
        ] = immutabledict(
            (sub_object, self._instantiate_object_schema(sub_object.schema))
            for sub_object in schema.sub_objects
        )
        for sub_object in schema.sub_objects:
            sub_object_perception = sub_object_to_object_perception[sub_object]
            self._object_perceptions.append(sub_object_perception)
            self._relation_perceptions.append(
                RelationPerception(PART_OF, root_object_perception, sub_object_perception)
            )
        for sub_object_relation in schema.sub_object_relations:
            # TODO: right now we translate all situation relations directly to perceptual
            # relations without modification. This is not always the right thing.
            self._relation_perceptions.append(
                RelationPerception(
                    sub_object_relation.relation_type,
                    sub_object_to_object_perception[sub_object_relation.arg1],
                    sub_object_to_object_perception[sub_object_relation.arg2],
                )
            )
        return root_object_perception
