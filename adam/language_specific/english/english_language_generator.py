import collections
from itertools import chain
from typing import Iterable, List, Mapping, MutableMapping, Union, cast

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutabledict, immutableset
from more_itertools import only
from networkx import DiGraph

from adam.language.dependency import (
    DependencyRole,
    DependencyTree,
    DependencyTreeLinearizer,
    DependencyTreeToken,
    LinearizedDependencyTree,
)
from adam.language.dependency.universal_dependencies import (
    ADJECTIVAL_MODIFIER,
    ADPOSITION,
    CASE_MARKING,
    DETERMINER,
    DETERMINER_ROLE,
    NOMINAL_MODIFIER,
    NOMINAL_MODIFIER_POSSESSIVE,
    NOMINAL_SUBJECT,
    NUMERAL,
    NUMERIC_MODIFIER,
    OBJECT,
    OBLIQUE_NOMINAL,
    PROPER_NOUN,
)
from adam.language.language_generator import LanguageGenerator
from adam.language.lexicon import LexiconEntry
from adam.language.ontology_dictionary import OntologyLexicon
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
    MASS_NOUN,
)
from adam.language_specific.english.english_syntax import (
    SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER,
)
from adam.ontology import IN_REGION, OntologyNode
from adam.ontology.phase1_ontology import (
    AGENT,
    COLOR,
    GOAL,
    GROUND,
    HAS,
    IS_ADDRESSEE,
    IS_SPEAKER,
    LEARNER,
    PATIENT,
    THEME,
)
from adam.ontology.phase1_spatial_relations import (
    DISTAL,
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_AXIS,
    INTERIOR,
    PROXIMAL,
    Region,
)
from adam.random_utils import SequenceChooser
from adam.relation import Relation
from adam.situation import Action, SituationObject
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation


@attrs(frozen=True, slots=True)
class SimpleRuleBasedEnglishLanguageGenerator(
    LanguageGenerator[HighLevelSemanticsSituation, LinearizedDependencyTree]
):
    r"""
    A simple rule-based approach for translating `HighLevelSemanticsSituation`\ s
    to English dependency trees.

    We currently only generate a single possible `LinearizedDependencyTree`
    for a given situation.
    """
    _ontology_lexicon: OntologyLexicon = attrib(
        validator=instance_of(OntologyLexicon), kw_only=True
    )
    """
    A mapping from nodes in our concept ontology to English words.
    """
    _dependency_tree_linearizer: DependencyTreeLinearizer = attrib(
        init=False, default=SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER, kw_only=True
    )
    """
    How to assign a word order to our dependency trees.
    
    This is hard-coded for now but may become flexible in the future.
    """

    def generate_language(
        self,
        situation: HighLevelSemanticsSituation,
        chooser: SequenceChooser,  # pylint:disable=unused-argument
    ) -> ImmutableSet[LinearizedDependencyTree]:
        return SimpleRuleBasedEnglishLanguageGenerator._Generation(
            self, situation
        ).generate()

    @attrs(frozen=True, slots=True)
    class _Generation:
        """
        This object encapsulates all the mutable state for an execution
        of `SimpleRuleBasedEnglishLanguageGenerator` on a single input.
        """

        # we need to keep this reference explicitly
        # because Python doesn't have real inner classes.
        generator: "SimpleRuleBasedEnglishLanguageGenerator" = attrib()
        # the situation being translated to language
        situation: HighLevelSemanticsSituation = attrib()
        # the dependency tree we are building
        dependency_graph: DiGraph = attrib(init=False, default=Factory(DiGraph))
        objects_to_dependency_nodes: MutableMapping[
            SituationObject, DependencyTreeToken
        ] = dict()
        """
        Don't access this directly;
        instead use `_noun_for_object`
        """
        object_counts: Mapping[OntologyNode, int] = attrib(init=False)
        """
        These are used to determine what quantifier to use when there are multiple objects.
        """

        def generate(self) -> ImmutableSet[LinearizedDependencyTree]:
            # The learner appears in a situation so they items may have spatial relations
            # with respect to it, but our language currently never refers to the learner itself.

            # We need to translate objects which appear in relations;
            # right now we only translate persistent relations to language
            # because it is unclear how to handle the others.
            if len(self.situation.actions) > 1:
                raise RuntimeError(
                    "Currently only situations with 0 or 1 actions are supported"
                )

            for object_ in self.situation.objects:
                if not self._only_translate_if_referenced(object_):
                    self._noun_for_object(object_)

            # We only translate those relations the user specifically calls out,
            # not the many "background" relations which are also true.
            for persisting_relation in self.situation.always_relations:
                self._translate_relation(persisting_relation)

            for action in self.situation.actions:
                self._translate_action_to_verb(action)

            return immutableset(
                [
                    self.generator._dependency_tree_linearizer.linearize(  # pylint:disable=protected-access
                        DependencyTree(self.dependency_graph)
                    )
                ]
            )

        def _noun_for_object(self, _object: SituationObject) -> DependencyTreeToken:
            if _object in self.objects_to_dependency_nodes:
                return self.objects_to_dependency_nodes[_object]

            count = self.object_counts[_object.ontology_node]
            # Eventually we will need to ensure that pluralized objects are
            # separated by their respective relations and actions
            # (e.g. in a situation where one box is on a table and one is below it,
            # don't output "two boxes")
            # https://github.com/isi-vista/adam/issues/129
            if not _object.ontology_node:
                raise RuntimeError(
                    f"Don't know how to handle objects which don't correspond to "
                    f"an ontology node currently: {_object}"
                )
            # TODO: we don't currently translate modifiers of nouns.
            # Issue: https://github.com/isi-vista/adam/issues/58
            lexicon_entry = self._unique_lexicon_entry(
                _object.ontology_node  # pylint:disable=protected-access
            )
            # Check if the situation object is the speaker
            if IS_SPEAKER in _object.properties:
                word_form = "I"
            elif IS_ADDRESSEE in _object.properties:
                word_form = "you"
            elif count > 1 and lexicon_entry.plural_form:
                word_form = lexicon_entry.plural_form
            else:
                word_form = lexicon_entry.base_form
            dependency_node = DependencyTreeToken(word_form, lexicon_entry.part_of_speech)

            self.dependency_graph.add_node(dependency_node)

            # add articles to things which are not proper nouns
            # ("a ball" but not "a Mom")
            if (dependency_node.part_of_speech != PROPER_NOUN) and (
                MASS_NOUN not in lexicon_entry.properties
            ):
                self.add_determiner(_object, count, dependency_node)

            # Begin work on translating modifiers of Nouns with Color
            for property_ in _object.properties:
                if self.situation.ontology.is_subtype_of(property_, COLOR):
                    color_lexicon_entry = self._unique_lexicon_entry(property_)
                    color_node = DependencyTreeToken(
                        color_lexicon_entry.base_form, color_lexicon_entry.part_of_speech
                    )
                    self.dependency_graph.add_edge(
                        color_node, dependency_node, role=ADJECTIVAL_MODIFIER
                    )

            return dependency_node

        def _only_translate_if_referenced(self, object_: SituationObject) -> bool:
            """
            Some special objects in the situation,
            like the ground, the speaker, and the addressee,
            should only be translated if referenced by an action or relation.
            """
            return (
                object_.ontology_node == GROUND
                or object_.ontology_node == LEARNER
                or IS_SPEAKER in object_.properties
                or IS_ADDRESSEE in object_.properties
            )

        def add_determiner(
            self,
            _object: SituationObject,
            count: int,
            noun_dependency_node: DependencyTreeToken,
        ) -> None:
            possession_relations = [
                relation
                for relation in self.situation.always_relations
                if relation.relation_type == HAS and relation.second_slot == _object
            ]
            if len(possession_relations) > 1:
                raise RuntimeError("Can not handle multiple possession relations")
            else:
                # e.g. it's always "the ground"
                if _object.ontology_node in ALWAYS_USE_THE_OBJECTS:
                    determiner_node = DependencyTreeToken("the", DETERMINER)
                    determiner_role = DETERMINER_ROLE
                # If speaker possesses the noun
                elif (
                    len(possession_relations) == 1
                    and IS_SPEAKER in possession_relations[0].first_slot.properties
                ):
                    determiner_node = DependencyTreeToken("my", DETERMINER)
                    determiner_role = NOMINAL_MODIFIER_POSSESSIVE
                # If addressee possess the noun
                elif (
                    len(possession_relations) == 1
                    and IS_ADDRESSEE in possession_relations[0].first_slot.properties
                ):
                    determiner_node = DependencyTreeToken("your", DETERMINER)
                    determiner_role = DETERMINER_ROLE
                # add 's if a non-agent possess the noun
                elif (
                    len(possession_relations) == 1
                    and self.situation.is_dynamic
                    and possession_relations[0].first_slot
                    not in only(self.situation.actions).argument_roles_to_fillers[AGENT]
                ):
                    determiner_node = self._noun_for_object(
                        possession_relations[0].first_slot
                    )
                    determiner_role = DETERMINER_ROLE
                    case_node = DependencyTreeToken("'s", DETERMINER)
                    case_role = CASE_MARKING
                    self.dependency_graph.add_edge(
                        case_node, determiner_node, role=case_role
                    )
                # otherwise do the normal determiner behavior
                elif count == 1:
                    determiner_node = DependencyTreeToken("a", DETERMINER)
                    determiner_role = DETERMINER_ROLE
                elif count == 2:
                    determiner_node = DependencyTreeToken("two", NUMERAL)
                    determiner_role = NUMERIC_MODIFIER
                # Currently, any number of objects greater than two is considered "many"
                else:
                    determiner_node = DependencyTreeToken("many", DETERMINER)
                    determiner_role = DETERMINER_ROLE
                self.dependency_graph.add_edge(
                    determiner_node, noun_dependency_node, role=determiner_role
                )

        def _translate_relation(self, relation: Relation[SituationObject]) -> None:
            if relation.relation_type == HAS:
                # 'has' is a special case.
                if self.situation.is_dynamic:
                    # already handled by noun translation code
                    pass
                else:
                    # otherwise, we realize it as the verb "has"
                    self._translate_relation_to_verb(relation)
            elif relation.relation_type == IN_REGION:
                self.dependency_graph.add_edge(
                    self.relation_to_prepositional_modifier(relation),
                    self._noun_for_object(relation.first_slot),
                    role=NOMINAL_MODIFIER,
                )
            else:
                raise RuntimeError(
                    f"Don't know how to translate relation " f"{relation} to English"
                )

        def _translate_action_to_verb(
            self, action: Action[OntologyNode, SituationObject]
        ) -> DependencyTreeToken:
            lexicon_entry = self._unique_lexicon_entry(action.action_type)
            # TODO: we don't currently handle verbal morphology.
            # https://github.com/isi-vista/adam/issues/60
            if len(self.situation.actions) > 1:
                raise RuntimeError(
                    "Currently only situations with 0 or 1 actions are supported"
                )
            elif any(
                property_
                in only(
                    only(self.situation.actions).argument_roles_to_fillers[AGENT]
                ).properties
                for property_ in [IS_SPEAKER, IS_ADDRESSEE]
            ):
                word_form = lexicon_entry.base_form
            elif lexicon_entry.verb_form_3SG_PRS:
                word_form = lexicon_entry.verb_form_3SG_PRS
            else:
                raise RuntimeError(
                    f"Verb has no 3SG present tense form: {lexicon_entry.base_form}"
                )
            verb_dependency_node = DependencyTreeToken(
                word_form, lexicon_entry.part_of_speech
            )
            self.dependency_graph.add_node(verb_dependency_node)

            for (argument_role, filler) in action.argument_roles_to_fillers.items():
                self._translate_verb_argument(
                    action, argument_role, filler, verb_dependency_node
                )

            for path_modifier in self._collect_action_modifiers(action):
                self.dependency_graph.add_edge(
                    path_modifier, verb_dependency_node, role=OBLIQUE_NOMINAL
                )

            return verb_dependency_node

        def _translate_verb_argument(
            self,
            action: Action[OntologyNode, SituationObject],
            argument_role: OntologyNode,
            filler: Union[SituationObject, Region[SituationObject]],
            verb_dependency_node: DependencyTreeToken,
        ):
            # TODO: to alternation
            # https://github.com/isi-vista/adam/issues/150
            if isinstance(filler, SituationObject):
                self.dependency_graph.add_edge(
                    self._noun_for_object(filler),
                    verb_dependency_node,
                    role=self._translate_argument_role(argument_role),
                )
            elif isinstance(filler, Region):
                if argument_role == GOAL:
                    if THEME not in action.argument_roles_to_fillers:
                        raise RuntimeError(
                            "Only know how to make English for a GOAL if"
                            "the verb has a THEME"
                        )

                    reference_object_dependency_node = self._noun_for_object(
                        filler.reference_object
                    )

                    self.dependency_graph.add_edge(
                        reference_object_dependency_node,
                        verb_dependency_node,
                        role=OBLIQUE_NOMINAL,
                    )

                    preposition_dependency_node = DependencyTreeToken(
                        self._preposition_for_region_as_goal(filler), ADPOSITION
                    )
                    self.dependency_graph.add_edge(
                        preposition_dependency_node,
                        reference_object_dependency_node,
                        role=CASE_MARKING,
                    )
                else:
                    raise RuntimeError(
                        "The only argument role we can currently handle regions as a filler "
                        "for is GOAL"
                    )
            else:
                raise RuntimeError(
                    f"Don't know how to handle {filler} as a filler of"
                    f" argument slot {argument_role} of verb "
                    f"{verb_dependency_node}"
                )

        # noinspection PyMethodMayBeStatic
        def _translate_argument_role(self, argument_role: OntologyNode) -> DependencyRole:
            return _ARGUMENT_ROLES_TO_DEPENDENCY_ROLES[argument_role]

        def _preposition_for_region_as_goal(self, region: Region[SituationObject]) -> str:
            """
            When a `Region` appears as the filler of the semantic role `GOAL`,
            determine what preposition to use to express it in English.
            """
            if region.distance == INTERIOR:
                return "in"
            elif (
                region.distance == EXTERIOR_BUT_IN_CONTACT
                and region.direction
                and region.direction.positive
                # TODO: put constraints on the axis
            ):
                return "on"
            else:
                raise RuntimeError(
                    f"Don't know how to translate {region} to a preposition yet"
                )

        def _collect_action_modifiers(
            self, action: Action[OntologyNode, SituationObject]
        ) -> Iterable[DependencyTreeToken]:
            """
            Collect adverbial and other modifiers of an action.

            For right now we only handle a subset of spatial modifiers
            which are realized as prepositions.
            """
            modifiers: List[DependencyTreeToken] = []

            if action.during:
                # so far we only handle IN_REGION relations which are asserted to hold
                # either continuously or at some point during an action
                for relation in chain(
                    action.during.at_some_point, action.during.continuously
                ):
                    if relation.relation_type == IN_REGION:
                        # the thing the relation is predicated of must be something plausibly
                        # moving, which for now is either..
                        fills_legal_argument_role = (
                            # the theme
                            relation.first_slot in action.argument_roles_to_fillers[THEME]
                            # or the agent or patient if there is no theme (e.g. jumps, falls)
                            or (
                                (
                                    relation.first_slot
                                    in action.argument_roles_to_fillers[AGENT]
                                    or relation.first_slot
                                    not in action.argument_roles_to_fillers[THEME]
                                )
                                and not action.argument_roles_to_fillers[THEME]
                            )
                        )
                        if fills_legal_argument_role:
                            modifiers.append(
                                self.relation_to_prepositional_modifier(relation)
                            )
                        else:
                            raise RuntimeError(
                                f"To translate a spatial relation as a verbal "
                                f"modifier, it must either be the theme or, if "
                                f"it is another filler, the theme must be absent:"
                                f" {relation} in {action} "
                            )
                    else:
                        raise RuntimeError(
                            f"Currently only know how to translate IN_REGION "
                            f"for relations which hold during an action: "
                            f"{relation} in {action}"
                        )

            return modifiers

        def relation_to_prepositional_modifier(self, relation) -> DependencyTreeToken:
            region = cast(Region[SituationObject], relation.second_slot)
            if (
                region.direction
                and region.direction.relative_to_axis == GRAVITATIONAL_AXIS
            ):
                if region.distance in (PROXIMAL, DISTAL):
                    if region.direction.positive:
                        preposition = "over"
                    else:
                        preposition = "under"
                elif (
                    region.distance == EXTERIOR_BUT_IN_CONTACT
                    and region.direction.positive
                ):
                    preposition = "on"
                else:
                    raise RuntimeError(
                        f"Don't know how to translate spatial " f"modifier: {relation}"
                    )
            else:
                raise RuntimeError(
                    f"Don't know how to translate spatial modifiers "
                    f"which are not relative to the gravitational "
                    f"axis: {relation}"
                )
            reference_object_node = self._noun_for_object(region.reference_object)
            self.dependency_graph.add_edge(
                DependencyTreeToken(preposition, ADPOSITION),
                reference_object_node,
                role=CASE_MARKING,
            )
            return reference_object_node

        def _translate_relation_to_verb(
            self, relation: Relation[SituationObject]
        ) -> DependencyTreeToken:
            lexicon_entry = self._unique_lexicon_entry(relation.relation_type)
            if any(
                property_ in relation.first_slot.properties
                for property_ in [IS_SPEAKER, IS_ADDRESSEE]
            ):
                word_form = lexicon_entry.base_form
            elif lexicon_entry.verb_form_3SG_PRS:
                word_form = lexicon_entry.verb_form_3SG_PRS
            else:
                raise RuntimeError(
                    f"Verb has no 3SG present tense form: {lexicon_entry.base_form}"
                )
            verb_dependency_node = DependencyTreeToken(
                word_form, lexicon_entry.part_of_speech
            )
            self.dependency_graph.add_node(verb_dependency_node)

            first_slot_dependency_node = self._noun_for_object(relation.first_slot)
            self.dependency_graph.add_edge(
                first_slot_dependency_node, verb_dependency_node, role=NOMINAL_SUBJECT
            )
            if isinstance(relation.second_slot, SituationObject):
                second_slot_dependency_node = self._noun_for_object(relation.second_slot)
            elif isinstance(relation.second_slot, Region):
                second_slot_dependency_node = self._noun_for_object(
                    relation.second_slot.reference_object
                )
            else:
                raise RuntimeError(
                    f"Unknown object type in relation {relation.second_slot}"
                )
            self.dependency_graph.add_edge(
                second_slot_dependency_node, verb_dependency_node, role=OBJECT
            )

            return verb_dependency_node

        def _unique_lexicon_entry(self, ontology_node: OntologyNode) -> LexiconEntry:
            lexicon_entries = self.generator._ontology_lexicon.words_for_node(  # pylint:disable=protected-access
                ontology_node
            )
            if lexicon_entries:
                if len(lexicon_entries) == 1:
                    return only(lexicon_entries)
                else:
                    raise RuntimeError(
                        f"We don't yet know how to deal with ontology nodes which "
                        f"could be realized by multiple lexical entries: "
                        f"{ontology_node} --> {lexicon_entries}. "
                        f"This is https://github.com/isi-vista/adam/issues/59 ."
                    )
            else:
                raise RuntimeError(f"No lexicon entry for ontology node {ontology_node}")

        @object_counts.default
        def _init_object_counts(self) -> Mapping[OntologyNode, int]:
            if not self.situation.actions:
                # For now, only apply quantifiers to object-only situations
                return collections.Counter(
                    [_object.ontology_node for _object in self.situation.objects]
                )
            else:
                return {
                    ontology_node: 1
                    for ontology_node in immutableset(
                        object_.ontology_node for object_ in self.situation.objects
                    )
                }


ALWAYS_USE_THE_OBJECTS = immutableset([GROUND])

# the relationship of argument roles to dependency roles
# is actually complex and verb-dependent.
# This is just a placeholder for a more sophisticated treatment.
_ARGUMENT_ROLES_TO_DEPENDENCY_ROLES: Mapping[
    OntologyNode, DependencyRole
] = immutabledict(
    (
        (AGENT, NOMINAL_SUBJECT),
        (PATIENT, OBJECT),
        (THEME, OBJECT),
        (GOAL, OBLIQUE_NOMINAL),
    )
)

GAILA_PHASE_1_LANGUAGE_GENERATOR = SimpleRuleBasedEnglishLanguageGenerator(
    ontology_lexicon=GAILA_PHASE_1_ENGLISH_LEXICON
)
