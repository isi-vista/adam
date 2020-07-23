import pytest
from more_itertools import first, one

from adam.curriculum.curriculum_utils import PHASE1_CHOOSER_FACTORY, phase1_instances
from adam.language_specific.chinese.chinese_phase_1_lexicon import (
    GAILA_PHASE_1_CHINESE_LEXICON,
)
from adam.language_specific.english.english_language_generator import PREFER_DITRANSITIVE
from adam.learner import PerceptionSemanticAlignment
from adam.learner.integrated_learner import IntegratedTemplateLearner
from adam.learner.language_mode import LanguageMode
from adam.ontology.phase1_ontology import (
    AGENT,
    BABY,
    CHAIR,
    DAD,
    GAILA_PHASE_1_ONTOLOGY,
    GIVE,
    GOAL,
    PHASE_1_CURRICULUM_OBJECTS,
    THEME,
)
from adam.perception import PerceptualRepresentation
from adam.perception.high_level_semantics_situation_to_developmental_primitive_perception import (
    HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator,
)
from adam.perception.perception_graph import PerceptionGraph
from adam.random_utils import RandomChooser
from adam.situation import Action, SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates.phase1_templates import (
    Phase1SituationTemplate,
    object_variable,
    sampled,
)
from tests.learner import (
    LANGUAGE_MODE_TO_OBJECT_RECOGNIZER,
    LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER,
)


@pytest.mark.parametrize("object_type", PHASE_1_CURRICULUM_OBJECTS)
@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
def test_recognizes_ontology_objects(object_type, language_mode):
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY,
        salient_objects=[
            SituationObject.instantiate_ontology_node(
                ontology_node=object_type, ontology=GAILA_PHASE_1_ONTOLOGY
            )
        ],
    )
    perception_generator = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
        GAILA_PHASE_1_ONTOLOGY
    )
    perception = perception_generator.generate_perception(
        situation, chooser=RandomChooser.for_seed(0), include_ground=False
    )
    learner = IntegratedTemplateLearner(
        object_learner=LANGUAGE_MODE_TO_TEMPLATE_LEARNER_OBJECT_RECOGNIZER[language_mode],
        language_mode=language_mode,
    )
    descriptions = learner.describe(perception)
    assert descriptions
    if language_mode == LanguageMode.ENGLISH:
        assert object_type.handle in one(descriptions.items())[0].as_token_sequence()
    else:
        mappings = (
            GAILA_PHASE_1_CHINESE_LEXICON._ontology_node_to_word  # pylint:disable=protected-access
        )
        for k, v in mappings.items():
            if k.handle == object_type.handle:
                assert v.base_form in one(descriptions.items())[0].as_token_sequence()


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
def test_trivial_dynamic_situation_with_schemaless_object(language_mode):
    dad_situation_object = SituationObject.instantiate_ontology_node(
        ontology_node=DAD, ontology=GAILA_PHASE_1_ONTOLOGY
    )
    situation = HighLevelSemanticsSituation(
        ontology=GAILA_PHASE_1_ONTOLOGY, salient_objects=[dad_situation_object]
    )
    perception_generator = HighLevelSemanticsSituationToDevelopmentalPrimitivePerceptionGenerator(
        GAILA_PHASE_1_ONTOLOGY
    )
    # We explicitly exclude ground in perception generation

    # this generates a static perception...
    perception = perception_generator.generate_perception(
        situation, chooser=RandomChooser.for_seed(0), include_ground=False
    )

    # so we need to construct a dynamic one by hand from two identical scenes
    dynamic_perception = PerceptualRepresentation(
        frames=[perception.frames[0], perception.frames[0]]
    )

    perception_graph = PerceptionGraph.from_dynamic_perceptual_representation(
        dynamic_perception
    )
    perception_semantic_alignment = PerceptionSemanticAlignment.create_unaligned(
        perception_graph
    )
    (_, description_to_matched_semantic_node) = LANGUAGE_MODE_TO_OBJECT_RECOGNIZER[
        language_mode
    ].match_objects(perception_semantic_alignment)
    assert len(description_to_matched_semantic_node) == 1
    assert (
        language_mode == LanguageMode.ENGLISH
        and ("Dad",) in description_to_matched_semantic_node
    ) or (
        language_mode == LanguageMode.CHINESE
        and ("ba4 ba4",) in description_to_matched_semantic_node
    )


@pytest.mark.parametrize("language_mode", [LanguageMode.ENGLISH, LanguageMode.CHINESE])
def test_recognize_in_transfer_of_possession(language_mode):
    dad = object_variable("person_0", DAD)
    baby = object_variable("person_1", BABY)
    chair = object_variable("give_object_0", CHAIR)

    giving_template = Phase1SituationTemplate(
        "dad-transfer-of-possession",
        salient_object_variables=[dad, baby, chair],
        actions=[
            Action(
                GIVE,
                argument_roles_to_fillers=[(AGENT, dad), (GOAL, baby), (THEME, chair)],
            )
        ],
        syntax_hints=[PREFER_DITRANSITIVE],
    )

    (_, _, perception) = first(
        phase1_instances(
            "foo",
            sampled(
                giving_template,
                max_to_sample=1,
                chooser=PHASE1_CHOOSER_FACTORY(),
                ontology=GAILA_PHASE_1_ONTOLOGY,
            ),
        ).instances()
    )

    perception_graph = PerceptionGraph.from_dynamic_perceptual_representation(perception)
    perception_semantic_alignment = PerceptionSemanticAlignment.create_unaligned(
        perception_graph
    )
    (_, description_to_matched_semantic_node) = LANGUAGE_MODE_TO_OBJECT_RECOGNIZER[
        language_mode
    ].match_objects(perception_semantic_alignment)
    assert len(description_to_matched_semantic_node) == 4
    assert (
        language_mode == LanguageMode.ENGLISH
        and ("Dad",) in description_to_matched_semantic_node
    ) or (
        language_mode == LanguageMode.CHINESE
        and ("ba4 ba4",) in description_to_matched_semantic_node
    )
