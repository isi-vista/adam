import collections
from itertools import chain
from typing import Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union, cast

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset, immutablesetmultidict
from more_itertools import first, only
from networkx import DiGraph

from adam.axes import FacingAddresseeAxis, GRAVITATIONAL_DOWN_TO_UP_AXIS
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
    ADVERB,
    ADVERBIAL_MODIFIER,
    CASE_POSSESSIVE,
    CASE_SPATIAL,
    DETERMINER,
    DETERMINER_ROLE,
    INDIRECT_OBJECT,
    NOMINAL_MODIFIER,
    NOMINAL_MODIFIER_POSSESSIVE,
    NOMINAL_SUBJECT,
    NUMERAL,
    NUMERIC_MODIFIER,
    OBJECT,
    OBLIQUE_NOMINAL,
    PROPER_NOUN,
    VERB,
    IS_ATTRIBUTE,
    OTHER,
    MARKER,
    ADJECTIVE,
)
from adam.language.language_generator import LanguageGenerator
from adam.language.lexicon import LexiconEntry
from adam.language.ontology_dictionary import OntologyLexicon
from adam.language_specific.english.english_phase_1_lexicon import (
    GAILA_PHASE_1_ENGLISH_LEXICON,
    I,
    ME,
    YOU,
)
from adam.language_specific.english.english_phase_2_lexicon import (
    GAILA_PHASE_2_ENGLISH_LEXICON,
)
from adam.language_specific import (
    FIRST_PERSON,
    SECOND_PERSON,
    ALLOWS_DITRANSITIVE,
    MASS_NOUN,
)
from adam.language_specific.english.english_syntax import (
    SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER,
)
from adam.ontology import IN_REGION, IS_ADDRESSEE, IS_SPEAKER, OntologyNode
from adam.ontology.phase1_ontology import (
    AGENT,
    COLOR,
    FALL,
    GOAL,
    GROUND,
    HAS,
    LEARNER,
    PATIENT,
    SIT,
    THEME,
    JUMP,
    FAST,
    SLOW,
    BIGGER_THAN,
    SMALLER_THAN,
)
from adam.ontology.phase1_spatial_relations import (
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_DOWN,
    INTERIOR,
    PROXIMAL,
    Region,
    TOWARD,
    GRAVITATIONAL_UP,
    SpatialPath,
    AWAY_FROM,
    TO,
    DISTAL,
)
from adam.random_utils import SequenceChooser
from adam.relation import Relation
from adam.situation import Action, SituationObject, SituationRegion
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
        ] = attrib(init=False, factory=dict)
        """
        Don't access this directly;
        instead use `_noun_for_object`
        """
        object_counts: Mapping[OntologyNode, int] = attrib(init=False)
        """
        These are used to determine what quantifier to use when there are multiple objects.
        """

        def generate(self) -> ImmutableSet[LinearizedDependencyTree]:
            try:
                return self._real_generate()
            except Exception as e:
                raise RuntimeError(
                    f"Error while generating English for situation " f"{self.situation}"
                ) from e

        def _real_generate(self) -> ImmutableSet[LinearizedDependencyTree]:
            # The learner appears in a situation so they items may have spatial relations
            # with respect to it, but our language currently never refers to the learner itself.

            # We need to translate objects which appear in relations;
            # right now we only translate persistent relations to language
            # because it is unclear how to handle the others.
            if len(self.situation.actions) > 1:
                raise RuntimeError(
                    "Currently only situations with 0 or 1 actions are supported"
                )

            # handle the special case of a static situation with only
            # multiple objects of the same type
            object_types_in_situation = set(
                object_.ontology_node for object_ in self.situation.salient_objects
            )

            action: Optional[Action[OntologyNode, SituationObject]]
            if self.situation.is_dynamic:
                # the situation contains an action, which we now translate.
                action = only(self.situation.actions)
                # mypy isn't smart enough to realized action can't be None here
                self._translate_action_to_verb(action)  # type: ignore

                # translate any leftover objects we didn't find while translating the action
                untranslated_objects = self.situation.salient_objects.difference(
                    self.objects_to_dependency_nodes.keys()
                )
                for untranslated_object in untranslated_objects:
                    self._noun_for_object(untranslated_object)
            else:
                # the situation is static (only objects and relations, no actions)
                action = None
                if len(object_types_in_situation) == 1:
                    # only one type of object is present
                    # e.g. three boxes
                    # doesn't matter which object we choose; they are all the same
                    first_object = first(self.situation.salient_objects)
                    self._noun_for_object(first_object)
                else:
                    # multiple objects of different types
                    for object_ in self.situation.salient_objects:
                        if not self._only_translate_if_referenced(object_):
                            self._noun_for_object(object_)

            # Always translate those relations the user specifically calls out,
            # not the many "background" relations which are also true.
            for persisting_relation in self.situation.always_relations:
                self._translate_relation(action, persisting_relation)

            return immutableset(
                [
                    self.generator._dependency_tree_linearizer.linearize(  # pylint:disable=protected-access
                        DependencyTree(self.dependency_graph)
                    )
                ]
            )

        def _noun_for_object(
            self,
            _object: SituationObject,
            *,
            syntactic_role_if_known: Optional[DependencyRole] = None,
        ) -> DependencyTreeToken:
            if _object in self.objects_to_dependency_nodes:
                return self.objects_to_dependency_nodes[_object]

            count = self.object_counts[_object.ontology_node]
            # Eventually we will need to ensure that pluralized objects are
            # separated by their respective relations and actions
            # (e.g. in a situation where one box is on a table and one is below it,
            # don't output "two boxes")
            # https://github.com/isi-vista/adam/issues/129

            # Check if the situation object is the speaker
            if IS_SPEAKER in _object.properties:
                if syntactic_role_if_known == NOMINAL_SUBJECT:
                    noun_lexicon_entry = I
                # For when HAS (which is a RELATION) is the verb and the speaker is the subject.
                # (This Special case is needed because HAS is a RELATION and not an ACTION in the
                # ontology, so HAS is never recognized as the NOMINAL_SUBJECT as this
                # determination normally occurs in translate_action_to_verb(), which only
                # processes ACTIONs)
                elif any(
                    relation
                    for relation in self.situation.always_relations
                    if (
                        relation.relation_type == HAS
                        and any(
                            property_ in relation.first_slot.properties
                            for property_ in [IS_SPEAKER]
                        )
                        and not self.situation.actions
                    )
                ):
                    noun_lexicon_entry = I
                else:
                    noun_lexicon_entry = ME
            elif IS_ADDRESSEE in _object.properties:
                # you works for both nominative and accusative
                noun_lexicon_entry = YOU
            else:
                noun_lexicon_entry = self._unique_lexicon_entry(
                    _object.ontology_node  # pylint:disable=protected-access
                )

            word_form = noun_lexicon_entry.base_form

            dependency_node = DependencyTreeToken(
                word_form,
                noun_lexicon_entry.part_of_speech,
                morphosyntactic_properties=noun_lexicon_entry.intrinsic_morphosyntactic_properties,
            )

            self.dependency_graph.add_node(dependency_node)

            self.add_determiner(
                _object, count, dependency_node, noun_lexicon_entry=noun_lexicon_entry
            )

            # IF plural, add a separate +s marker token
            if (
                count > 1
                and noun_lexicon_entry.plural_form
                and len(self.object_counts) == 1
            ):
                # If plural
                plural_marker_node = DependencyTreeToken("s", OTHER)  # PoS tag
                self.dependency_graph.add_node(plural_marker_node)
                self.dependency_graph.add_edge(
                    plural_marker_node, dependency_node, role=MARKER
                )

            if ATTRIBUTES_AS_X_IS_Y in self.situation.syntax_hints:
                self._translate_attribute_as_verb(_object, dependency_node)
            else:
                self._add_attributes(_object, dependency_node)

            self.objects_to_dependency_nodes[_object] = dependency_node
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
            *,
            noun_lexicon_entry: LexiconEntry,
        ) -> None:
            # not "the Mom"
            if (
                noun_dependency_node.part_of_speech == PROPER_NOUN
                # not "a sand"
                or MASS_NOUN in noun_lexicon_entry.properties
                # not "a you"
                or noun_lexicon_entry in (I, YOU, ME)
            ):
                return

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
                    case_role = CASE_POSSESSIVE
                    self.dependency_graph.add_edge(
                        case_node, determiner_node, role=case_role
                    )
                # only add one of the following quantifiers in situations with
                # multiples of the same object
                elif len(self.object_counts) == 1 and count > 1:
                    if count == 2:
                        determiner_node = DependencyTreeToken("two", NUMERAL)
                        determiner_role = NUMERIC_MODIFIER
                    # Currently, any number of objects greater than two is considered "many"
                    else:
                        determiner_node = DependencyTreeToken("many", DETERMINER)
                        determiner_role = DETERMINER_ROLE
                # otherwise do the normal determiner behavior
                else:
                    determiner_node = DependencyTreeToken("a", DETERMINER)
                    determiner_role = DETERMINER_ROLE
                self.dependency_graph.add_edge(
                    determiner_node, noun_dependency_node, role=determiner_role
                )

        def _add_attributes(
            self, _object: SituationObject, noun_dependency_node: DependencyTreeToken
        ) -> None:
            if IGNORE_COLORS not in self.situation.syntax_hints:
                # Begin work on translating modifiers of Nouns with Color
                for property_ in _object.properties:
                    if self.situation.ontology.is_subtype_of(property_, COLOR):
                        color_lexicon_entry = self._unique_lexicon_entry(property_)
                        color_node = DependencyTreeToken(
                            color_lexicon_entry.base_form,
                            color_lexicon_entry.part_of_speech,
                            color_lexicon_entry.intrinsic_morphosyntactic_properties,
                        )
                        self.dependency_graph.add_edge(
                            color_node, noun_dependency_node, role=ADJECTIVAL_MODIFIER
                        )

        def _translate_attribute_as_verb(
            self, _object: SituationObject, noun_dependency_node: DependencyTreeToken
        ) -> None:
            # We can use this only in static situations
            if self.situation.is_dynamic:
                raise RuntimeError("Cannot produce X is Y language in dynamic situations")
            # If we want x is y language, first get the attributes
            if IGNORE_COLORS in self.situation.syntax_hints:
                properties = [
                    property_
                    for property_ in _object.properties
                    if not self.situation.ontology.is_subtype_of(property_, COLOR)
                ]
            else:
                properties = [property_ for property_ in _object.properties]

            if len(properties) > 1:
                raise RuntimeError(
                    "Cannot handle X is Y language with multiple attributes"
                )
            elif not properties:
                raise RuntimeError("No available attributes (Y) for the X is Y language")

            attribute_lexicon_entry = self._unique_lexicon_entry(first(properties))
            node = DependencyTreeToken(
                attribute_lexicon_entry.base_form,
                attribute_lexicon_entry.part_of_speech,
                attribute_lexicon_entry.intrinsic_morphosyntactic_properties,
            )
            is_node = DependencyTreeToken("is", VERB, {})
            self.dependency_graph.add_edge(
                is_node, noun_dependency_node, role=IS_ATTRIBUTE
            )
            self.dependency_graph.add_edge(
                node, noun_dependency_node, role=NOMINAL_MODIFIER
            )

        def _translate_relation(
            self,
            action: Optional[Action[OntologyNode, SituationObject]],
            relation: Relation[SituationObject],
        ) -> None:
            if relation.relation_type == HAS:
                # 'has' is a special case.
                if self.situation.is_dynamic:
                    # already handled by noun translation code
                    pass
                else:
                    # otherwise, we realize it as the verb "has"
                    if IGNORE_HAS_AS_VERB in self.situation.syntax_hints:
                        # If the user explicitly states not to translate `has`
                        # to a verb, we don't want to do this
                        pass
                    else:
                        self._translate_relation_to_verb(relation)
            elif relation.relation_type == IN_REGION:
                prepositional_modifier = self.relation_to_prepositional_modifier(
                    action, relation
                )
                if prepositional_modifier:
                    self.dependency_graph.add_edge(
                        prepositional_modifier,
                        self._noun_for_object(relation.first_slot),
                        role=NOMINAL_MODIFIER,
                    )
            elif relation.relation_type == BIGGER_THAN:
                if (
                    relation.first_slot in self.situation.salient_objects
                    and isinstance(relation.second_slot, SituationObject)
                    and relation.second_slot.ontology_node == LEARNER
                ):
                    if USE_VERTICAL_MODIFIERS in self.situation.syntax_hints:
                        token = DependencyTreeToken("tall", ADJECTIVE)
                    else:
                        token = DependencyTreeToken("big", ADJECTIVE)
                    self.dependency_graph.add_node(token)
                    self.dependency_graph.add_edge(
                        token,
                        self._noun_for_object(relation.first_slot),
                        role=ADJECTIVAL_MODIFIER,
                    )
            elif relation.relation_type == SMALLER_THAN:
                if (
                    relation.first_slot in self.situation.salient_objects
                    and isinstance(relation.second_slot, SituationObject)
                    and relation.second_slot.ontology_node == LEARNER
                ):
                    if USE_VERTICAL_MODIFIERS in self.situation.syntax_hints:
                        token = DependencyTreeToken("short", ADJECTIVE)
                    else:
                        token = DependencyTreeToken("small", ADJECTIVE)
                    self.dependency_graph.add_node(token)
                    self.dependency_graph.add_edge(
                        token,
                        self._noun_for_object(relation.first_slot),
                        role=ADJECTIVAL_MODIFIER,
                    )
            else:
                raise RuntimeError(
                    f"Don't know how to translate relation " f"{relation} to English"
                )

        def _translate_action_to_verb(
            self, action: Action[OntologyNode, SituationObject]
        ) -> DependencyTreeToken:
            verb_lexical_entry = self._unique_lexicon_entry(action.action_type)

            # first, we map all the arguments to chunks of dependency tree
            syntactic_roles_to_argument_heads = immutablesetmultidict(
                self._translate_verb_argument(
                    action, verb_lexical_entry, argument_role, filler
                )
                for (argument_role, filler) in action.argument_roles_to_fillers.items()
                if not (
                    isinstance(filler, SituationObject)
                    and filler.ontology_node == LEARNER
                )
                and not (
                    argument_role == GOAL and IGNORE_GOAL in self.situation.syntax_hints
                )
                # Explicitly ignore relationships which are LEARNER objects from language generation
            )

            # determine the surface form of the verb,
            # which in English depends on the subject
            verb_word_form: str
            subject_heads = syntactic_roles_to_argument_heads[NOMINAL_SUBJECT]
            if subject_heads:
                if len(subject_heads) == 1:
                    subject_head = only(subject_heads)
                    if (
                        FIRST_PERSON in subject_head.morphosyntactic_properties
                        or SECOND_PERSON in subject_head.morphosyntactic_properties
                    ):
                        verb_word_form = verb_lexical_entry.base_form
                    elif verb_lexical_entry.verb_form_sg3_prs:
                        verb_word_form = verb_lexical_entry.verb_form_sg3_prs
                    else:
                        raise RuntimeError(
                            f"Verb has no 3SG present tense form: {verb_lexical_entry.base_form}"
                        )
                else:
                    raise RuntimeError(
                        f"Cannot currently handle multiple subject_heads: {action}; "
                        f"semantic role mapping: {syntactic_roles_to_argument_heads}"
                    )
            else:
                raise RuntimeError(
                    f"Cannot currently handle an absent subject: {action}; "
                    f"semantic role mapping: {syntactic_roles_to_argument_heads}"
                )

            # actually add the verb to the dependency tree
            verb_dependency_node = DependencyTreeToken(
                verb_word_form,
                verb_lexical_entry.part_of_speech,
                morphosyntactic_properties=verb_lexical_entry.intrinsic_morphosyntactic_properties,
            )
            self.dependency_graph.add_node(verb_dependency_node)

            # attach the arguments computed above to the verb's dependency tree node
            for (
                syntactic_role,
                argument_head,
            ) in syntactic_roles_to_argument_heads.items():
                self.dependency_graph.add_edge(
                    argument_head, verb_dependency_node, role=syntactic_role
                )

            # attach modifiers of the verbs (e.g. prepositions)
            for (modifier_role, path_modifier) in self._collect_action_modifiers(action):
                self.dependency_graph.add_edge(
                    path_modifier, verb_dependency_node, role=modifier_role
                )

            return verb_dependency_node

        def _translate_verb_argument(
            self,
            action: Action[OntologyNode, SituationObject],
            verb_lexical_entry: LexiconEntry,
            argument_role: OntologyNode,
            filler: Union[SituationObject, SituationRegion],
        ) -> Tuple[DependencyRole, DependencyTreeToken]:
            if isinstance(filler, SituationObject):
                syntactic_role = self._translate_argument_role(
                    action, verb_lexical_entry, argument_role
                )
                filler_noun = self._noun_for_object(
                    filler, syntactic_role_if_known=syntactic_role
                )
                # e.g. Mom gives a cookie *to a baby*
                if argument_role == GOAL and syntactic_role == OBLIQUE_NOMINAL:
                    preposition = self._determine_goal_preposition(
                        action, verb_lexical_entry, argument_role, filler
                    )
                    if preposition:
                        self.dependency_graph.add_edge(
                            preposition, filler_noun, role=CASE_SPATIAL
                        )
                return (syntactic_role, filler_noun)
            elif isinstance(filler, Region):
                if argument_role == GOAL:
                    if not self._get_moving_thing(action):
                        raise RuntimeError(
                            "Only know how to make English for a GOAL if"
                            "the verb has a THEME or AGENT"
                        )

                    reference_object_dependency_node = self._noun_for_object(
                        filler.reference_object
                    )

                    preposition_dependency_node = DependencyTreeToken(
                        self._preposition_for_region_as_goal(filler), ADPOSITION
                    )
                    self.dependency_graph.add_edge(
                        preposition_dependency_node,
                        reference_object_dependency_node,
                        role=CASE_SPATIAL,
                    )

                    return (OBLIQUE_NOMINAL, reference_object_dependency_node)
                else:
                    raise RuntimeError(
                        "The only argument role we can currently handle regions as a filler "
                        "for is GOAL"
                    )
            else:
                raise RuntimeError(
                    f"Don't know how to handle {filler} as a filler of"
                    f" argument slot {argument_role} of action "
                    f"{action}"
                )

        def _determine_goal_preposition(
            self,
            action: Action[OntologyNode, SituationObject],
            verb_lexical_entry: LexiconEntry,  # pylint:disable=unused-argument
            argument_role: OntologyNode,  # pylint:disable=unused-argument
            filler: Union[
                SituationObject, SituationRegion
            ],  # pylint:disable=unused-argument
        ) -> Optional[DependencyTreeToken]:
            moving_thing = self._get_moving_thing(action)

            if moving_thing:
                return DependencyTreeToken("to", ADPOSITION)
            else:
                return None

        def _get_moving_thing(self, action) -> Optional[SituationObject]:
            if THEME in action.argument_roles_to_fillers:
                return action.argument_roles_to_fillers[THEME]
            elif AGENT in action.argument_roles_to_fillers:
                return action.argument_roles_to_fillers[AGENT]
            else:
                return None

        # noinspection PyMethodMayBeStatic
        def _translate_argument_role(
            self,
            action: Action[OntologyNode, SituationObject],
            verb_lexical_entry: LexiconEntry,
            argument_role: OntologyNode,
        ) -> DependencyRole:
            if argument_role == AGENT:
                # Thomas reads the book.
                return NOMINAL_SUBJECT
            elif argument_role == PATIENT:
                # James smashes the Lego castle.
                return OBJECT
            elif argument_role == THEME:
                if AGENT in action.argument_roles_to_fillers:
                    # Beatrice rolls the ball.
                    return OBJECT
                else:
                    # the ball falls.
                    return NOMINAL_SUBJECT
            elif self.situation.ontology.is_subtype_of(argument_role, GOAL):
                if (
                    PREFER_DITRANSITIVE in self.situation.syntax_hints
                    and ALLOWS_DITRANSITIVE in verb_lexical_entry.properties
                ):
                    # Mom gives a baby a cookie
                    return INDIRECT_OBJECT
                else:
                    # Mom gives a cookie to a baby
                    # Dad puts a box on a table
                    return OBLIQUE_NOMINAL
            else:
                raise RuntimeError(
                    f"Do not know how to map argument role "
                    f"{argument_role} of {action} to a syntactic role."
                )

        def _preposition_for_region_as_goal(self, region: SituationRegion) -> str:
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
            elif region.distance == PROXIMAL and not region.direction:
                # See: https://github.com/isi-vista/adam/issues/836
                if USE_NEAR in self.situation.syntax_hints:
                    return "near"
                else:
                    return "to"
            elif region.direction == GRAVITATIONAL_UP:
                if USE_ABOVE_BELOW in self.situation.syntax_hints:
                    return "above"
                else:
                    return "over"
            elif region.direction == GRAVITATIONAL_DOWN:
                if USE_ABOVE_BELOW in self.situation.syntax_hints:
                    return "below"
                else:
                    return "under"
            elif region.distance == DISTAL and not region.direction:
                return "far from"
            elif region.direction and self.situation.axis_info:
                if not self.situation.axis_info.addressee:
                    raise RuntimeError(
                        f"Unable to translate region into a preposition because an addressee is lacking. "
                        f"Region: {region}\nSituation: {self.situation}"
                    )
                # HACK, from M3
                # see: https://github.com/isi-vista/adam/issues/573
                if isinstance(region.direction.relative_to_axis, FacingAddresseeAxis):
                    # "in front of" and "behind" is defined without a distance as you can accurate use the phrase
                    # regardless of distance example:
                    # "the teacher is in front of your laptop"
                    # (Assuming the laptop is near the back of class and the addressee is facing the front of the room)
                    # "your friend is in front of your laptop"
                    # (Assuming the friend is one row up in the classroom)
                    if region.direction.positive:
                        return "in front of"
                    else:
                        return "behind"
                elif (
                    region.direction.relative_to_axis != GRAVITATIONAL_DOWN_TO_UP_AXIS
                    and region.distance == PROXIMAL
                ):
                    return "beside"
                else:
                    raise RuntimeError(
                        f"Don't know how to translate {region} to a preposition yet"
                    )
            else:
                raise RuntimeError(
                    f"Don't know how to translate {region} to a preposition yet"
                )

        def _collect_action_modifiers(
            self, action: Action[OntologyNode, SituationObject]
        ) -> Iterable[Tuple[DependencyRole, DependencyTreeToken]]:
            """
            Collect adverbial and other modifiers of an action.

            For right now we only handle a subset of spatial modifiers
            which are realized as prepositions.
            """
            modifiers: List[Tuple[DependencyRole, DependencyTreeToken]] = []

            if USE_ADVERBIAL_PATH_MODIFIER in self.situation.syntax_hints:
                # up and down modifiers
                if action.during:
                    paths_involving_ground = immutableset(
                        path
                        for (_, path) in action.during.objects_to_paths.items()
                        if path.reference_object.ontology_node == GROUND
                    )
                    if paths_involving_ground:
                        # we just look at the first to determine the direction
                        first_path = first(paths_involving_ground)
                        if first_path.operator == TOWARD:
                            modifiers.append(
                                (ADVERBIAL_MODIFIER, DependencyTreeToken("down", ADVERB))
                            )
                        else:
                            modifiers.append(
                                (ADVERBIAL_MODIFIER, DependencyTreeToken("up", ADVERB))
                            )
                elif action.action_type == FALL or action.action_type == SIT:
                    # hack, awaiting https://github.com/isi-vista/adam/issues/239
                    modifiers.append(
                        (ADVERBIAL_MODIFIER, DependencyTreeToken("down", ADVERB))
                    )
                elif action.action_type == JUMP:
                    modifiers.append(
                        (ADVERBIAL_MODIFIER, DependencyTreeToken("up", ADVERB))
                    )
            if action.during:

                # so far we only handle IN_REGION relations which are asserted to hold
                # either continuously or at some point during an action
                for relation in chain(
                    action.during.at_some_point, action.during.continuously
                ):
                    self._translate_relation_to_action_modifier(
                        action, relation, modifiers
                    )
                if action.during.objects_to_paths:
                    for (
                        path_object,
                        spatial_path,
                    ) in action.during.objects_to_paths.items():
                        spatial_prepositonal_modifier = self._translate_spatial_path_to_action_modifier(
                            action, path_object, spatial_path
                        )
                        if spatial_prepositonal_modifier:
                            modifiers.append(
                                (OBLIQUE_NOMINAL, spatial_prepositonal_modifier)
                            )

                for (_, path) in action.during.objects_to_paths.items():
                    if FAST in path.properties:
                        modifiers.append(
                            (ADVERBIAL_MODIFIER, DependencyTreeToken("fast", ADVERB))
                        )
                    elif SLOW in path.properties:
                        modifiers.append(
                            (ADVERBIAL_MODIFIER, DependencyTreeToken("fast", ADVERB))
                        )

            for relation in self.situation.after_action_relations:
                self._translate_relation_to_action_modifier(action, relation, modifiers)

            return modifiers

        def _translate_relation_to_action_modifier(
            self,
            action: Action[OntologyNode, SituationObject],
            relation: Relation[SituationObject],
            modifiers,
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
                            relation.first_slot in action.argument_roles_to_fillers[AGENT]
                            or relation.first_slot
                            not in action.argument_roles_to_fillers[THEME]
                        )
                        and not action.argument_roles_to_fillers[THEME]
                    )
                )
                if fills_legal_argument_role:
                    prepositional_modifier = self.relation_to_prepositional_modifier(
                        action, relation
                    )
                    if prepositional_modifier:
                        modifiers.append((OBLIQUE_NOMINAL, prepositional_modifier))
                else:
                    # we don't want to translate relations of the agent (yet)
                    return
            else:
                raise RuntimeError(
                    f"Currently only know how to translate IN_REGION "
                    f"for relations which hold during an action: "
                    f"{relation} in {action}"
                )

        def relation_to_prepositional_modifier(
            self,
            action: Optional[Action[OntologyNode, SituationObject]],
            relation: Relation[SituationObject],
        ) -> Optional[DependencyTreeToken]:
            region = cast(SituationRegion, relation.second_slot)
            # don't talk about relations to non-salient objects
            if region.reference_object not in self.situation.salient_objects:
                return None

            if action:
                # If both arguments of the relation are core argument roles,
                # we assume the verb takes care of expressing their relationship.
                core_argument_fillers = immutableset(
                    chain(
                        action.argument_roles_to_fillers[AGENT],
                        action.argument_roles_to_fillers[PATIENT],
                        action.argument_roles_to_fillers[THEME],
                    )
                )

                if (
                    relation.first_slot in core_argument_fillers
                    and region.reference_object in core_argument_fillers
                ):
                    return None

            preposition: Optional[str] = None

            if region.distance == INTERIOR and relation.negated:
                preposition = "out of"

            elif region.distance == INTERIOR:
                preposition = "in"

            elif region.distance == PROXIMAL and not region.direction:
                # See: https://github.com/isi-vista/adam/issues/836
                if USE_NEAR in self.situation.syntax_hints:
                    preposition = "near"
                else:
                    preposition = "to"

            elif region.distance == DISTAL and not region.direction:
                preposition = "far from"

            elif region.direction:
                direction_axis = region.direction.relative_to_concrete_axis(
                    self.situation.axis_info
                )

                if region.distance == EXTERIOR_BUT_IN_CONTACT:
                    if region.direction.positive:
                        preposition = "on"
                else:
                    if direction_axis.aligned_to_gravitational:
                        if region.direction.positive:
                            if USE_ABOVE_BELOW in self.situation.syntax_hints:
                                preposition = "above"
                            else:
                                preposition = "over"
                        else:
                            if USE_ABOVE_BELOW in self.situation.syntax_hints:
                                preposition = "below"
                            else:
                                preposition = "under"
                    else:
                        # TODO: hack for M3; revisit in cleanup
                        # see: https://github.com/isi-vista/adam/issues/573
                        if isinstance(
                            region.direction.relative_to_axis, FacingAddresseeAxis
                        ):
                            if region.direction.positive:
                                preposition = "in front of"
                            else:
                                preposition = "behind"
                        elif region.distance == PROXIMAL:
                            preposition = "beside"

            if not preposition:
                raise RuntimeError(
                    f"Don't know how to translate spatial modifier {relation}"
                )

            reference_object_node = self._noun_for_object(region.reference_object)

            if self.dependency_graph.out_degree[reference_object_node]:
                # the reference node already has a syntactic connection to the tree,
                # so don't add another one
                return None
            else:
                self.dependency_graph.add_edge(
                    DependencyTreeToken(preposition, ADPOSITION),
                    reference_object_node,
                    role=CASE_SPATIAL,
                )
                return reference_object_node

        def _translate_spatial_path_to_action_modifier(
            self,
            action: Optional[Action[OntologyNode, SituationObject]],
            path_object: SituationObject,
            spatial_path: SpatialPath[SituationObject],
        ):
            # don't talk about spatial paths to non-salient objects
            if (
                spatial_path.reference_object
                and spatial_path.reference_object not in self.situation.salient_objects
            ):
                return None

            # Only derive spatial path if path object is salient
            if path_object not in self.situation.salient_objects:
                return None

            if action:
                # If both arguments of the relation are core argument roles,
                # we assume the verb takes care of expressing their relationship.
                core_argument_fillers = immutableset(
                    chain(
                        action.argument_roles_to_fillers[AGENT],
                        action.argument_roles_to_fillers[PATIENT],
                        action.argument_roles_to_fillers[THEME],
                    )
                )

                if (
                    path_object in core_argument_fillers
                    and spatial_path.reference_object in core_argument_fillers
                ):
                    return None

            preposition: Optional[str] = None

            if spatial_path.operator == TOWARD:
                preposition = "toward"
            elif spatial_path.operator == AWAY_FROM:
                preposition = "away_from"
            # TO is the default spatial path in most actions
            # so we let the preposition be handled by the GOAL
            # We also explicitly ignore the None operator
            # as spatial paths may just have properties
            # we are interested in asserting
            elif spatial_path.operator in [TO, None]:
                return None

            if not preposition:
                raise RuntimeError(
                    f"Don't know how to translate spatial path {spatial_path}"
                )

            if isinstance(spatial_path.reference_object, Region):
                reference_object_node = self._noun_for_object(
                    spatial_path.reference_object.reference_object
                )
            else:
                reference_object_node = self._noun_for_object(
                    spatial_path.reference_object
                )

            if self.dependency_graph.out_degree[reference_object_node]:
                # the reference node already has a syntactic connection to the tree,
                # so don't add another one
                return None
            else:
                self.dependency_graph.add_edge(
                    DependencyTreeToken(preposition, ADPOSITION),
                    reference_object_node,
                    role=CASE_SPATIAL,
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
            elif lexicon_entry.verb_form_sg3_prs:
                word_form = lexicon_entry.verb_form_sg3_prs
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
                    [_object.ontology_node for _object in self.situation.all_objects]
                )
            else:
                return {
                    ontology_node: 1
                    for ontology_node in immutableset(
                        # even though only salient objects have linguistic expression
                        # by default,
                        # we gather counts over all objects in the scene.
                        object_.ontology_node
                        for object_ in self.situation.all_objects
                    )
                }


ALWAYS_USE_THE_OBJECTS = immutableset([GROUND])


GAILA_PHASE_1_LANGUAGE_GENERATOR = SimpleRuleBasedEnglishLanguageGenerator(
    ontology_lexicon=GAILA_PHASE_1_ENGLISH_LEXICON
)
GAILA_PHASE_2_LANGUAGE_GENERATOR = SimpleRuleBasedEnglishLanguageGenerator(
    ontology_lexicon=GAILA_PHASE_2_ENGLISH_LEXICON
)

# these are "hints" situations can pass to the language generator
# to control its behavior
# See https://github.com/isi-vista/adam/issues/222
USE_ADVERBIAL_PATH_MODIFIER = "USE_ADVERBIAL_PATH_MODIFIER"
PREFER_DITRANSITIVE = "PREFER_DITRANSITIVE"
IGNORE_COLORS = "IGNORE_COLORS"
IGNORE_HAS_AS_VERB = "IGNORE_HAS_AS_VERB"
ATTRIBUTES_AS_X_IS_Y = "ATTRIBUTES_AS_X_IS_Y"
IGNORE_SIZE_ATTRIBUTE = "IGNORE_SIZE_ATTRIBUTE"
IGNORE_GOAL = "IGNORE_GOAL"
USE_VERTICAL_MODIFIERS = "USE_VERTICAL_MODIFIERS"
USE_ABOVE_BELOW = "USE_ABOVE_BELOW"
USE_NEAR = "USE_NEAR"
