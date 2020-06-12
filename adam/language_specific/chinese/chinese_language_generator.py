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
    PARTICLE,
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
    ADVERBIAL_CLAUSE_MODIFIER,
)
from adam.language.language_generator import LanguageGenerator
from adam.language.lexicon import LexiconEntry
from adam.language.ontology_dictionary import OntologyLexicon
from adam.language_specific.chinese.chinese_phase_1_lexicon import (
    GAILA_PHASE_1_CHINESE_LEXICON,
    ME,
    YOU,
)
from adam.language_specific import (
    FIRST_PERSON,
    SECOND_PERSON,
    ALLOWS_DITRANSITIVE,
    MASS_NOUN,
)
from adam.language_specific.chinese.chinese_syntax import (
    SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER,
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
)
from adam.ontology.phase1_spatial_relations import (
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_DOWN,
    INTERIOR,
    PROXIMAL,
    Region,
    TOWARD,
    GRAVITATIONAL_UP,
)
from adam.random_utils import SequenceChooser
from adam.relation import Relation
from adam.situation import Action, SituationObject, SituationRegion
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation


@attrs(frozen=True, slots=True)
class SimpleRuleBasedChineseLanguageGenerator(
    LanguageGenerator[HighLevelSemanticsSituation, LinearizedDependencyTree]
):
    """Simple rule-based approach for translating HighLevelSemanticSituations to
    Chinese dependency trees. We currently only generate a single possible linearized
    tree for a semantic situation."""

    # mapping from nodes in the concept ontology to Chinese words
    _ontology_lexicon: OntologyLexicon = attrib(
        validator=instance_of(OntologyLexicon), kw_only=True
    )

    # how to assign word order to the dependency trees
    _dependency_tree_linearizer: DependencyTreeLinearizer = attrib(
        init=False, default=SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER, kw_only=True
    )

    def generate_language(
        self, situation: HighLevelSemanticsSituation, chooser: SequenceChooser
    ) -> ImmutableSet[LinearizedDependencyTree]:
        """The function that actually generates the language"""
        # remove once done
        return SimpleRuleBasedChineseLanguageGenerator._Generation(
            self, situation
        ).generate()

    @attrs(frozen=True, slots=True)
    class _Generation:
        """This class encapsulates all the mutable state for an execution of the
           SimpleRuleBasedChineseLanguageGenerator on a single input"""

        # keep the reference to the parent because python doesn't have real inner classes
        generator: "SimpleRuleBasedChineseLanguageGenerator" = attrib()
        # the situation being translated into language
        situation: HighLevelSemanticsSituation = attrib()
        # the graph we are building
        dependency_graph: DiGraph = attrib(init=False, default=Factory(DiGraph))
        # stores a mapping of nouns for the objects in the situation
        objects_to_dependency_nodes: MutableMapping[
            SituationObject, DependencyTreeToken
        ] = attrib(init=False, factory=dict)
        # keep a mapping of object counts so we know what quantifiers to use when there are multiple objects
        object_counts: Mapping[OntologyNode, int] = attrib(init=False)

        def generate(self) -> ImmutableSet[LinearizedDependencyTree]:
            """The function that tries to generate language for a given representation"""
            try:
                return self._real_generate()
            except Exception as e:
                raise RuntimeError(
                    f"Error while generating Chinese for the situation {self.situation}"
                ) from e

        def _real_generate(self) -> ImmutableSet[LinearizedDependencyTree]:
            """The function that actually generates the language"""
            # we currently can't deal with situations with more than one action
            if len(self.situation.actions) > 1:
                raise RuntimeError(
                    "Currently only situations with 0 or 1 actions are supported"
                )

            # handle the special case of a static situation with only multiple objects of the same type
            object_types_in_situation = set(
                object_.ontology_node for object_ in self.situation.salient_objects
            )

            action: Optional[Action[OntologyNode, SituationObject]]
            # handle dynamic situations
            if self.situation.is_dynamic:
                # get the action
                action = only(self.situation.actions)
                self._translate_action_to_verb(action)  # type: ignore
            # handle static situations
            else:
                action = None
                # handle one type of object (there may be many of it)
                if len(object_types_in_situation) == 1:
                    first_object = first(self.situation.salient_objects)
                    self._noun_for_object(first_object)
                # multiple objects of different types
                else:
                    for object_ in self.situation.salient_objects:
                        if not self._only_translate_if_referenced(object_):
                            self._noun_for_object(object_)

            for persisting_relation in self.situation.always_relations:
                self._translate_relation(action, persisting_relation)

            return immutableset(
                [
                    self.generator._dependency_tree_linearizer.linearize(
                        DependencyTree(self.dependency_graph)
                    )
                ]
            )

        def _translate_action_to_verb(
            self, action: Action[OntologyNode, SituationObject]
        ) -> DependencyTreeToken:
            """Translate the situation's action to a VP"""
            verb_lexical_entry = self._unique_lexicon_entry(action.action_type)

            # map all the arguments to chunks of the dependency tree, ignoring LEARNER object from generation
            syntactic_roles_to_argument_heads = immutablesetmultidict(
                self._translate_verb_argument(
                    action, verb_lexical_entry, argument_role, filler
                )
                for (argument_role, filler) in action.argument_roles_to_fillers.items()
                if not (
                    isinstance(filler, SituationObject)
                    and filler.ontology_node == LEARNER
                )
            )

            # check that there is only one subject. We don't need to get a surface form here since Chinese doesn't conjugate
            subject_heads = syntactic_roles_to_argument_heads[NOMINAL_SUBJECT]
            if not len(subject_heads) == 1:
                raise RuntimeError(
                    f"Cannot handle multiple or zero subject heads for {action} and semantic role mapping {syntactic_roles_to_argument_heads}"
                )

            # actually add the verb to the dependency tree
            verb_dependency_node = DependencyTreeToken(
                verb_lexical_entry.base_form,
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

            # TODO: attach modifiers
            return verb_dependency_node

        def _translate_verb_argument(
            self,
            action: Action[OntologyNode, SituationObject],
            verb_lexical_entry: LexiconEntry,
            argument_role: OntologyNode,
            filler: Union[SituationObject, SituationRegion],
        ) -> Tuple[DependencyRole, DependencyTreeToken]:

            """Maps dependency roles to the corresponding node heads for verb arguments"""

            # deal with the case that this is an object in the situation
            if isinstance(filler, SituationObject):
                # get the syntactic role by using a helper function
                syntactic_role = self._translate_argument_role(
                    action, verb_lexical_entry, argument_role
                )
                # get the noun corresponding to the object
                filler_noun = self._noun_for_object(
                    filler, syntactic_role_if_known=syntactic_role
                )
                # this is the "ba" construction in Chinese where an object occurs with "ba" before
                # the verb if the verb isn't ditransitive but is "trying" to take 2 objects
                if argument_role == THEME and syntactic_role == OBLIQUE_NOMINAL:
                    preposition = DependencyTreeToken("ba3", ADPOSITION)
                    self.dependency_graph.add_edge(
                        preposition, filler_noun, role=CASE_SPATIAL
                    )
                    # raise NotImplementedError("Ditransitive prepositions not yet handled")
                return (syntactic_role, filler_noun)
            elif isinstance(filler, Region):
                # get the noun for the object
                reference_object_dependency_node = self._noun_for_object(
                    filler.reference_object
                )

                # this determines the coverb based on whether the situation is static or dynamic
                coverb: str = "dzai4"
                if self.situation.after_action_relations:
                    coverb = "dau4"
                self.dependency_graph.add_edge(
                    DependencyTreeToken(coverb, ADPOSITION),
                    reference_object_dependency_node,
                    role=CASE_SPATIAL,
                )

                # get the localiser as well
                localiser_dependency_node = DependencyTreeToken(
                    self._localiser_for_region_as_goal(filler), ADPOSITION
                )
                self.dependency_graph.add_edge(
                    localiser_dependency_node,
                    reference_object_dependency_node,
                    role=NOMINAL_MODIFIER,
                )
                # this is an adverbial clause modifier since it occurs post-verbally -- hack
                # (https://github.com/isi-vista/adam/issues/797)
                return (ADVERBIAL_CLAUSE_MODIFIER, reference_object_dependency_node)
            else:
                raise RuntimeError(
                    "The only argument role we can currently handle regions as a filler "
                    "for is GOAL"
                )

        def _localiser_for_region_as_goal(self, region: SituationRegion) -> str:
            """
            When a `Region` appears as the filler of the semantic role `GOAL`,
            determine what preposition to use to express it in English.
            """
            if region.distance == INTERIOR:
                return "nei4"
            elif (
                region.distance == EXTERIOR_BUT_IN_CONTACT
                and region.direction
                and region.direction.positive
                # TODO: put constraints on the axis
            ):
                return "shang4"
            elif region.distance == PROXIMAL and not region.direction:
                raise NotImplementedError("Haven't handled to in Chinese")
            elif region.direction == GRAVITATIONAL_UP:
                return "shang4 myan4"
            elif region.direction == GRAVITATIONAL_DOWN:
                return "sya4"
            # region.distance == DISTAL is not check as this does not define a specific preposition in scope for Phase 1
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
                        return "chyan2 myan4"
                    else:
                        return "hou4 myan4"
                elif (
                    region.direction.relative_to_axis != GRAVITATIONAL_DOWN_TO_UP_AXIS
                    and region.distance == PROXIMAL
                ):
                    return "pang2 byan1"
                else:
                    raise RuntimeError(
                        f"Don't know how to translate {region} to a preposition yet"
                    )
            else:
                raise RuntimeError(
                    f"Don't know how to translate {region} to a preposition yet"
                )

        def _translate_argument_role(
            self,
            action: Action[OntologyNode, SituationObject],
            verb_lexical_entry: LexiconEntry,
            argument_role: OntologyNode,
        ) -> DependencyRole:
            """Translate an argument role to a syntactic role so verb arguments can be joined to trees"""

            if argument_role == AGENT:
                # Thomas reads the book.
                return NOMINAL_SUBJECT
            elif argument_role == PATIENT:
                # James smashes the Lego castle.
                return OBJECT
            elif argument_role == THEME:
                if AGENT in action.argument_roles_to_fillers:
                    # if there's a theme and a goal but the verb isn't ditransitive, the theme becomes preverbial
                    if (
                        GOAL in action.argument_roles_to_fillers
                        and PREFER_DITRANSITIVE not in self.situation.syntax_hints
                    ):
                        return OBLIQUE_NOMINAL
                    # if there's no goal or the verb is ditransitive, it's still post verbal
                    else:
                        return OBJECT
                else:
                    # the theme can be the subject if there is not an agent in the action
                    return NOMINAL_SUBJECT
            elif self.situation.ontology.is_subtype_of(argument_role, GOAL):
                return INDIRECT_OBJECT
            else:
                raise RuntimeError(
                    f"Do not know how to map argument role "
                    f"{argument_role} of {action} to a syntactic role."
                )

        def _noun_for_object(
            self,
            _object: SituationObject,
            *,
            syntactic_role_if_known: Optional[DependencyRole] = None,
        ) -> DependencyTreeToken:
            """Get the noun for the object in a given situation"""

            # if we already have a mapping for a noun, we're done
            if _object in self.objects_to_dependency_nodes:
                return self.objects_to_dependency_nodes[_object]

            # the number of this object that is in the scene
            count = self.object_counts[_object.ontology_node]

            # make sure there is a corresponding ontology node
            if not _object.ontology_node:
                raise RuntimeError(
                    f"Don't know how to handle objects which don't correspond to "
                    f"an ontology node currently: {_object}"
                )

            # check if the situation object is the speaker
            if IS_SPEAKER in _object.properties:
                noun_lexicon_entry = ME

            # check if the situation object is the addressee
            elif IS_ADDRESSEE in _object.properties:
                noun_lexicon_entry = YOU

            # if not wo or ni, then just an object
            else:
                noun_lexicon_entry = self._unique_lexicon_entry(
                    _object.ontology_node  # pylint:disable=protected-access
                )

            # create a dependency node for the noun
            dependency_node = DependencyTreeToken(
                noun_lexicon_entry.base_form,
                noun_lexicon_entry.part_of_speech,
                morphosyntactic_properties=noun_lexicon_entry.intrinsic_morphosyntactic_properties,
            )
            self.dependency_graph.add_node(dependency_node)

            # add a classifier if necessary
            self.add_classifier(
                _object, count, dependency_node, noun_lexicon_entry=noun_lexicon_entry
            )

            # TODO: handle X_IS_Y; not implemented in syntax yet either (https://github.com/isi-vista/adam/issues/804)

            # if colour is specified it, add it as an adjectival modifier
            if IGNORE_COLORS not in self.situation.syntax_hints:
                for property_ in _object.properties:
                    if self.situation.ontology.is_subtype_of(property_, COLOR):
                        color_lexicon_entry = self._unique_lexicon_entry(property_)
                        color_node = DependencyTreeToken(
                            color_lexicon_entry.base_form,
                            color_lexicon_entry.part_of_speech,
                            color_lexicon_entry.intrinsic_morphosyntactic_properties,
                        )
                        self.dependency_graph.add_edge(
                            color_node, dependency_node, role=ADJECTIVAL_MODIFIER
                        )

            self.objects_to_dependency_nodes[_object] = dependency_node
            return dependency_node

        def add_classifier(
            self,
            _object: SituationObject,
            count: int,
            noun_dependency_node: DependencyTreeToken,
            *,
            noun_lexicon_entry: LexiconEntry,
        ) -> None:
            """Add a classifier to a given noun"""

            # get the current possession relations so we can add wo de or ni de
            possession_relations = [
                relation
                for relation in self.situation.always_relations
                if relation.relation_type == HAS and relation.second_slot == _object
            ]

            # we can only handle one possession relation at a time right now
            if len(possession_relations) > 1:
                raise RuntimeError("Cannot handle multiple possession relations")
            elif len(possession_relations) == 1:
                # handle the possession relation if there is one. We don't need to case on person in Chinese
                # since all possessives are expressed NP+de+NP
                possessor = self._noun_for_object(possession_relations[0].first_slot)
                de = DependencyTreeToken("de", PARTICLE)
                self.dependency_graph.add_edge(de, possessor, role=CASE_POSSESSIVE)
                self.dependency_graph.add_edge(
                    possessor, noun_dependency_node, role=NOMINAL_MODIFIER_POSSESSIVE
                )
            # if the count is one, we're done since we're not using yi CLF currently
            if count == 1:
                return
            #  https://github.com/isi-vista/adam/issues/782
            # TODO: get classifiers checked by a native speaker upon implementation
            elif count == 2:
                raise NotImplementedError(
                    "We don't know how to handle Chinese classifiers yet"
                )
            # if the count is many, we don't need a CLF
            else:
                many = DependencyTreeToken("hen3 dwo1", NUMERAL)
                self.dependency_graph.add_edge(
                    many, noun_dependency_node, role=NUMERIC_MODIFIER
                )

        def _translate_relation(
            self,
            action: Optional[Action[OntologyNode, SituationObject]],
            relation: Relation[SituationObject],
        ):
            """Translate relations that the user explicitly calls out"""
            if relation.relation_type == HAS:
                # if the situation is dynamic, then this will be handled within the NP
                if (
                    self.situation.is_dynamic
                    or IGNORE_HAS_AS_VERB in self.situation.syntax_hints
                ):
                    pass
                else:
                    raise NotImplementedError("We don't handle 'has' as a verb yet")
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
            else:
                raise RuntimeError(
                    f"We can't currently translate {relation} relation to Chinese"
                )

        def relation_to_prepositional_modifier(
            self,
            action: Optional[Action[OntologyNode, SituationObject]],
            relation: Relation[SituationObject],
        ) -> Optional[DependencyTreeToken]:
            """Translate a relation to a prepositional modifier"""

            region = cast(SituationRegion, relation.second_slot)
            # if the object in the relation is not salient, then we don't care about the relation
            if region.reference_object not in self.situation.salient_objects:
                return None
            # deal with actions with verbs
            if action:
                raise NotImplementedError("We don't handle prepositions in VP's yet")
            preposition: Optional[str] = None
            # inside/in
            if region.distance == INTERIOR:
                preposition = "nei4"
            # to/towards
            # TODO: to in Chinese is expressed differently than in English
            # https://github.com/isi-vista/adam/issues/805
            if region.distance == PROXIMAL and not region.direction:
                raise NotImplementedError(
                    "We have not finished implementing co-verbs in Chinese"
                )
            elif region.direction:
                direction_axis = region.direction.relative_to_concrete_axis(
                    self.situation.axis_info
                )
                # on & in contact
                if region.distance == EXTERIOR_BUT_IN_CONTACT:
                    if region.direction.positive:
                        preposition = "shang4"
                else:
                    if direction_axis.aligned_to_gravitational:
                        # TODO: check "over" mapping
                        # https://github.com/isi-vista/adam/issues/805
                        if region.direction.positive:
                            preposition = "shang4 myan4"
                        # under
                        else:
                            preposition = "sya4"
                    else:
                        if isinstance(
                            region.direction.relative_to_axis, FacingAddresseeAxis
                        ):
                            if region.direction.positive:
                                preposition = "chyan2 myan4"
                            else:
                                preposition = "hou4 myan4"
                        elif region.distance == PROXIMAL:
                            preposition = "pang2 byan1"
            if not preposition:
                raise RuntimeError(
                    f"Don't know how to handle {relation} as a preposition"
                )

            # get the noun for the OOP
            reference_object_node = self._noun_for_object(region.reference_object)

            # this means that the reference node is already in the graph
            if self.dependency_graph.out_degree[reference_object_node]:
                return None
            # if the reference node isn't already in the graph, add it
            else:
                self.dependency_graph.add_edge(
                    DependencyTreeToken(preposition, ADPOSITION),
                    reference_object_node,
                    role=NOMINAL_MODIFIER,
                )
                self.dependency_graph.add_edge(
                    DependencyTreeToken("dzai4", ADPOSITION),
                    reference_object_node,
                    role=CASE_SPATIAL,
                )
                return reference_object_node

        def _unique_lexicon_entry(self, ontology_node: OntologyNode) -> LexiconEntry:
            """Get a lexicon entry for a given ontology node"""
            # get the lexicon entries for the ontology node
            lexicon_entries = self.generator._ontology_lexicon.words_for_node(  # pylint:disable=protected-access
                ontology_node
            )
            # if there is one possible match, return it
            if lexicon_entries:
                if len(lexicon_entries) == 1:
                    return only(lexicon_entries)

                # if there's more than one possible match, we don't know how to handle this
                else:
                    raise RuntimeError(
                        f"We don't yet know how to deal with ontology nodes which "
                        f"could be realized by multiple lexical entries: "
                        f"{ontology_node} --> {lexicon_entries}. "
                        f"This is https://github.com/isi-vista/adam/issues/59 ."
                    )
            else:
                raise RuntimeError(f"No lexicon entry for ontology node {ontology_node}")

        def _only_translate_if_referenced(self, object_: SituationObject) -> bool:
            """Some objects like the ground or speaker shouldn't be addressed unless explicitly mentioned"""
            return (
                object_.ontology_node == GROUND
                or object_.ontology_node == LEARNER
                or IS_SPEAKER in object_.properties
                or IS_ADDRESSEE in object_.properties
            )

        # TODO: only counting salient object right now, this may need to be changed later or consider adding counts for both separately
        # https://github.com/isi-vista/adam/issues/802
        @object_counts.default
        def _init_object_counts(self) -> Mapping[OntologyNode, int]:
            """Default method for initializing the object counts"""
            if not self.situation.actions:
                # For now, only apply quantifiers to object-only situations
                return collections.Counter(
                    [_object.ontology_node for _object in self.situation.salient_objects]
                )
            else:
                return {
                    ontology_node: 1
                    for ontology_node in immutableset(
                        # even though only salient objects have linguistic expression
                        # by default,
                        # we gather counts over all objects in the scene.
                        object_.ontology_node
                        for object_ in self.situation.salient_objects
                    )
                }


# create an instance of the rule-based language generator for our current ontology
GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR = SimpleRuleBasedChineseLanguageGenerator(
    ontology_lexicon=GAILA_PHASE_1_CHINESE_LEXICON
)


# these are "hints" situations can pass to the language generator
# to control its behavior
# See https://github.com/isi-vista/adam/issues/222
# TODO: modify these to fit with Chinese syntax
USE_ADVERBIAL_PATH_MODIFIER = "USE_ADVERBIAL_PATH_MODIFIER"
PREFER_DITRANSITIVE = "PREFER_DITRANSITIVE"
IGNORE_COLORS = "IGNORE_COLORS"
IGNORE_HAS_AS_VERB = "IGNORE_HAS_AS_VERB"
ATTRIBUTES_AS_X_IS_Y = "ATTRIBUTES_AS_X_IS_Y"
