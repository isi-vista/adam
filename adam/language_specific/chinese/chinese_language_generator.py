import collections
from itertools import chain
from typing import Iterable, List, Mapping, MutableMapping, Optional, Tuple, Union, cast

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
    ADJECTIVE,
    ADPOSITION,
    ADVERB,
    ADVERBIAL_CLAUSE_MODIFIER,
    ADVERBIAL_MODIFIER,
    CASE_POSSESSIVE,
    CASE_SPATIAL,
    CLASSIFIER,
    INDIRECT_OBJECT,
    IS_ATTRIBUTE,
    NOMINAL_MODIFIER,
    NOMINAL_MODIFIER_POSSESSIVE,
    NOMINAL_SUBJECT,
    NOUN,
    NUMERAL,
    NUMERIC_MODIFIER,
    OBJECT,
    OBLIQUE_NOMINAL,
    PARTICLE,
    PROPER_NOUN,
    VERB,
)
from adam.language.language_generator import LanguageGenerator
from adam.language.lexicon import LexiconEntry
from adam.language.ontology_dictionary import OntologyLexicon
from adam.language_specific import ALLOWS_DITRANSITIVE
from adam.language_specific.chinese.chinese_phase_1_lexicon import (
    BABY,
    BIRD,
    DAD,
    DOG,
    GAILA_PHASE_1_CHINESE_LEXICON,
    GRAB,
    ME,
    MOM,
    RUN,
    SHOVE,
    TOSS,
    YOU,
)
from adam.language_specific.chinese.chinese_phase_2_lexicon import (
    GAILA_PHASE_2_CHINESE_LEXICON,
)
from adam.language_specific.chinese.chinese_syntax import (
    SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER,
)
from adam.ontology import IN_REGION, IS_ADDRESSEE, IS_SPEAKER, OntologyNode
from adam.ontology.phase1_ontology import (
    AGENT,
    BIGGER_THAN,
    COLOR,
    COME,
    FALL,
    FAST,
    GIVE,
    GO,
    GOAL,
    GOAL_MANIPULATOR,
    GROUND,
    HARD_FORCE,
    HAS,
    JUMP,
    LEARNER,
    PASS,
    PATIENT,
    PUSH,
    SIT,
    SLOW,
    SMALLER_THAN,
    SOFT_FORCE,
    TAKE,
    THEME,
    WALK,
)
from adam.ontology.phase1_spatial_relations import (
    AWAY_FROM,
    DISTAL,
    EXTERIOR_BUT_IN_CONTACT,
    GRAVITATIONAL_DOWN,
    GRAVITATIONAL_UP,
    INTERIOR,
    PROXIMAL,
    Region,
    SpatialPath,
    TO,
    TOWARD,
    VIA,
)
from adam.random_utils import SequenceChooser
from adam.relation import Relation
from adam.situation import Action, SituationObject, SituationRegion
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset, immutablesetmultidict


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
        """The function that actually generates the language for a given situation"""
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
            """The function that tries to generate language for a given representation and throws an error if this isn't possible"""
            try:
                return self._real_generate()
            except Exception as e:
                raise RuntimeError(
                    f"Error while generating Chinese for the situation {self.situation}"
                ) from e

        def _real_generate(self) -> ImmutableSet[LinearizedDependencyTree]:
            """The function that actually generates the language for the given situation"""
            # we currently can't deal with situations with more than one action
            if len(self.situation.actions) > 1:
                raise RuntimeError(
                    "Currently only situations with 0 or 1 actions are supported"
                )

            # get a set of the object types in the situation that can be used later for special cases
            object_types_in_situation = set(
                object_.ontology_node for object_ in self.situation.salient_objects
            )

            action: Optional[Action[OntologyNode, SituationObject]]

            # handle dynamic situations
            if self.situation.is_dynamic:
                # get the action and translate it to a verb
                action = only(self.situation.actions)
                self._translate_action_to_verb(action)  # type: ignore

            # handle static situations
            else:
                action = None
                # handle the case of only one type of object (there may be many of it)
                if len(object_types_in_situation) == 1:
                    first_object = first(self.situation.salient_objects)
                    self._noun_for_object(first_object)
                # handle the case of multiple objects of different types
                else:
                    for object_ in self.situation.salient_objects:
                        if not self._only_translate_if_referenced(object_):
                            self._noun_for_object(object_)
                # translate persisting relations for static situations -- these behave differently than those for dynamic situations in Chinese
                for persisting_relation in self.situation.always_relations:
                    self._translate_relation(action, persisting_relation)

            # linearize and return the generated tree
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
            """Translate the situation's action to a VP node in the tree"""

            # get the lexical entry corresponding to the verb for special cases
            special_lexical: bool = False
            if (
                action.action_type in [WALK, TAKE, PUSH, PASS]
                and action.during
                and action.during.objects_to_paths
            ):
                for (_, path) in action.during.objects_to_paths.items():
                    if HARD_FORCE in path.properties:
                        special_lexical = True
            if special_lexical:
                if action.action_type == WALK:
                    verb_lexical_entry = RUN
                elif action.action_type == TAKE:
                    verb_lexical_entry = GRAB
                elif action.action_type == PUSH:
                    verb_lexical_entry = SHOVE
                else:
                    verb_lexical_entry = TOSS

            else:
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
                and not (
                    argument_role == GOAL and IGNORE_GOAL in self.situation.syntax_hints
                )
            )

            # check that there is exactly one subject
            subject_heads = syntactic_roles_to_argument_heads[NOMINAL_SUBJECT]
            if not len(subject_heads) == 1:
                raise RuntimeError(
                    f"Cannot handle {len(subject_heads)} subject heads for {action} and semantic role mapping {syntactic_roles_to_argument_heads}"
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

            # handle modifiers, including path modifiers and always_relations, and attach them to the tree
            for (modifier_role, path_modifier) in self._collect_action_modifiers(action):
                self.dependency_graph.add_edge(
                    path_modifier, verb_dependency_node, role=modifier_role
                )

            return verb_dependency_node

        def _collect_action_modifiers(
            self, action: Action[OntologyNode, SituationObject]
        ) -> Iterable[Tuple[DependencyRole, DependencyTreeToken]]:
            """
            Collect adverbial and other modifiers of an action.

            For right now we only handle a subset of spatial modifiers which are realized as localisers.

            This also collects always_relations since it is helpful for Chinese to handle these in the context of the VP
            rather than globally since information about the VP will influence where the modifiers attach.
            """

            # the list of dependency-role -> dependencytoken mappings to be attached to the VP tree
            modifiers: List[Tuple[DependencyRole, DependencyTreeToken]] = []

            # if there are during action modifiers, collect them, parse them, and add them to the modifiers
            # we want to parse these after after_action relations since if there is an after-action relation that is also
            # a during relation, we want to be sure to translate it as having reached the goal
            if action.during:
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
                        spatial_modifier = self._translate_spatial_path(
                            action, path_object, spatial_path
                        )
                        if spatial_modifier:
                            modifiers.append(spatial_modifier)
                    # handle fast/slow
                    for (_, spatial_path) in action.during.objects_to_paths.items():
                        if FAST in spatial_path.properties:
                            modifiers.append(
                                (
                                    ADVERBIAL_MODIFIER,
                                    DependencyTreeToken("kwai4 su4", ADVERB),
                                )
                            )
                        elif SLOW in spatial_path.properties:
                            modifiers.append(
                                (
                                    ADVERBIAL_MODIFIER,
                                    DependencyTreeToken("man4 man", ADVERB),
                                )
                            )

            # if there are after action relations, collect them, parse them, and add them to the modifiers
            for relation in self.situation.after_action_relations:
                self._translate_relation_to_action_modifier(action, relation, modifiers)

            # if there are always relations, collect them, parse them, and add them to the modifiers
            for relation in self.situation.always_relations:
                self._translate_relation_to_action_modifier(action, relation, modifiers)

            # if there are adverbial path modifiers, collect them, parse them, and add them to the modifiers
            if USE_ADVERBIAL_PATH_MODIFIER in self.situation.syntax_hints:
                # adverbial modifiers we currently handle must occur during the action and be related to the ground; check if such modifiers exist
                if action.during:
                    paths_involving_ground = immutableset(
                        path
                        for (_, path) in action.during.objects_to_paths.items()
                        if (
                            isinstance(path.reference_object, SituationObject)
                            and path.reference_object.ontology_node == GROUND
                        )
                        or (
                            isinstance(path.reference_object, Region)
                            and path.reference_object.reference_object == GROUND
                        )
                    )
                    if paths_involving_ground:
                        # only consider the first one to determine direction
                        first_path = first(paths_involving_ground)
                        # the object is moving towards the ground, so down
                        if first_path.operator == TOWARD:
                            # Chinese handles adverbmods differently for other verbs than for qu and lai, so case on verb type here
                            if action.action_type == GO or action.action_type == COME:
                                modifiers.append(
                                    (
                                        ADVERBIAL_MODIFIER,
                                        DependencyTreeToken("sya4", ADVERB),
                                    )
                                )
                            # handle the case of regular verbs with a downwards adverb modifier
                            else:
                                modifiers.append(
                                    (
                                        ADVERBIAL_CLAUSE_MODIFIER,
                                        DependencyTreeToken("sya4 lai2", ADVERB),
                                    )
                                )
                        # the object is moving away from the ground, so up
                        else:
                            # Chinese handles adverbmods differently for other verbs than for qu and lai, so case on verb type here
                            if action.action_type == GO or action.action_type == COME:
                                modifiers.append(
                                    (
                                        ADVERBIAL_MODIFIER,
                                        DependencyTreeToken("shang4", ADVERB),
                                    )
                                )
                            else:
                                # handle the case of regular verbs with a upnwards adverb modifier
                                modifiers.append(
                                    (
                                        ADVERBIAL_CLAUSE_MODIFIER,
                                        DependencyTreeToken("chi3 lai2", ADVERB),
                                    )
                                )
                # falling or sitting is defaulted to down with advmod, but if specified otherwise, will be handled by first case
                elif action.action_type == FALL or action.action_type == SIT:
                    modifiers.append(
                        (
                            ADVERBIAL_CLAUSE_MODIFIER,
                            DependencyTreeToken("sya4 lai2", ADVERB),
                        )
                    )
                # jumping is defaulted to up with advmod, but if specified otherwise, will be handled by first case
                elif action.action_type == JUMP:
                    modifiers.append(
                        (
                            ADVERBIAL_CLAUSE_MODIFIER,
                            DependencyTreeToken("chi3 lai2", ADVERB),
                        )
                    )
            return modifiers

        def _translate_spatial_path(
            self,
            action: Optional[Action[OntologyNode, SituationObject]],
            path_object: SituationObject,
            spatial_path: SpatialPath[SituationObject],
        ):
            if spatial_path.reference_object and (
                (
                    isinstance(spatial_path.reference_object, SituationObject)
                    and spatial_path.reference_object
                    not in self.situation.salient_objects
                )
                or (
                    isinstance(spatial_path.reference_object, Region)
                    and spatial_path.reference_object.reference_object
                    not in self.situation.salient_objects
                )
            ):
                return None
            if path_object and (
                (
                    isinstance(path_object, SituationObject)
                    and path_object not in self.situation.salient_objects
                )
                or (
                    isinstance(path_object, Region)
                    and path_object.reference_object not in self.situation.salient_objects
                )
            ):
                return None

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
                preposition = "chau2"
            elif spatial_path.operator == AWAY_FROM:
                preposition = "li2"
            elif spatial_path.operator in [TO, VIA, None]:
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
                return None
            else:
                self.dependency_graph.add_edge(
                    DependencyTreeToken(preposition, ADPOSITION),
                    reference_object_node,
                    role=CASE_SPATIAL,
                )
                return (OBLIQUE_NOMINAL, reference_object_node)

        def _translate_relation_to_action_modifier(
            self,
            action: Action[OntologyNode, SituationObject],
            relation: Relation[SituationObject],
            modifiers,
        ):
            """Translates a relation such as a during_relation or always_relation to an action modifier for a VP"""

            # translate size relations. The salient objects have already been translated
            if relation.relation_type == BIGGER_THAN:
                if (
                    relation.first_slot in self.situation.salient_objects
                    and isinstance(relation.second_slot, SituationObject)
                    and relation.second_slot.ontology_node == LEARNER
                ):
                    # tall
                    if USE_VERTICAL_MODIFIERS in self.situation.syntax_hints:
                        token = DependencyTreeToken("gau1 da4", ADJECTIVE)
                    # big
                    else:
                        token = DependencyTreeToken("da4", ADJECTIVE)
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
                    # short
                    if USE_VERTICAL_MODIFIERS in self.situation.syntax_hints:
                        token = DependencyTreeToken("dwan3", ADJECTIVE)
                    # small
                    else:
                        token = DependencyTreeToken("syau3", ADJECTIVE)
                    self.dependency_graph.add_node(token)
                    self.dependency_graph.add_edge(
                        token,
                        self._noun_for_object(relation.first_slot),
                        role=ADJECTIVAL_MODIFIER,
                    )

            # we only translate in_region relations because have relations are translated elsewhere
            elif relation.relation_type == IN_REGION:
                # legal arguments include the theme or the agent or patient if there is no theme (intransitive verbs such as jump and fall)
                fills_legal_argument_role = relation.first_slot in action.argument_roles_to_fillers[
                    THEME
                ] or (
                    (
                        relation.first_slot in action.argument_roles_to_fillers[AGENT]
                        or relation.first_slot
                        not in action.argument_roles_to_fillers[THEME]
                    )
                    and not action.argument_roles_to_fillers[THEME]
                )
                # check if the relation includes legal arguments; if it doesn't we don't handle it
                if fills_legal_argument_role:
                    # cast the second slot in the relation to a region and get the noun for the reference object
                    region = cast(SituationRegion, relation.second_slot)
                    reference_object_node = self._noun_for_object(region.reference_object)
                    # we can only have one relation per object; this is an issue for cases such as having during and after action relations
                    # in the same VP. To solve this, we check if the reference object node is already in the modifiers and return if it is.
                    if any(m[1] == reference_object_node for m in modifiers):
                        return
                    # try to get the localiser phrase modifier for the given relation
                    localiser_modifier = self.relation_to_localiser_modifier(
                        action, relation
                    )
                    if localiser_modifier:
                        # an always relation presented as a localiser always occurs preverbially
                        # TODO: https://github.com/isi-vista/adam/issues/811 NP mods within VP's aren't currently handled
                        if relation in self.situation.always_relations:
                            modifiers.append((OBLIQUE_NOMINAL, localiser_modifier))
                        # all other relations indicate something about the path and so they occur post-verbially
                        else:
                            modifiers.append(
                                (ADVERBIAL_CLAUSE_MODIFIER, localiser_modifier)
                            )
                else:
                    # we don't want to translate relations of the agent (yet)
                    return

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
                # get the syntactic role
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
                    ba = DependencyTreeToken("ba3", ADPOSITION)
                    self.dependency_graph.add_edge(ba, filler_noun, role=CASE_SPATIAL)
                # when a person is coming or going towards a another person, we need zhao
                elif (
                    action
                    and (action.action_type == GO or action.action_type == COME)
                    and argument_role == GOAL
                    and (
                        IS_SPEAKER in filler.properties
                        or IS_ADDRESSEE in filler.properties
                    )
                ):
                    zhao = DependencyTreeToken("jau3", ADPOSITION)
                    self.dependency_graph.add_edge(zhao, filler_noun, role=CASE_SPATIAL)
                # deal with movement to a person as the goal
                elif (
                    action
                    and (argument_role == GOAL or argument_role == GOAL_MANIPULATOR)
                    and action.action_type not in [GIVE, GO, COME]
                    and (
                        IS_SPEAKER in filler.properties
                        or IS_ADDRESSEE in filler.properties
                        # hack to identify possible animate goals
                        or filler.ontology_node in [DAD, MOM, BABY, DOG, BIRD]
                    )
                ):
                    gei = DependencyTreeToken("gei3", ADPOSITION)
                    self.dependency_graph.add_edge(gei, filler_noun, role=CASE_SPATIAL)
                return (syntactic_role, filler_noun)
            # deal with the case that it's a region in the situation
            elif isinstance(filler, Region) and argument_role == GOAL:

                # get the noun for the object
                reference_object_dependency_node = self._noun_for_object(
                    filler.reference_object
                )

                # if this is the equivalent of "to" in English with a go/come verb, we just use the bare noun
                if (
                    (action.action_type == GO or action.action_type == COME)
                    and (filler.distance == PROXIMAL)
                    and (not filler.direction)
                    and (USE_NEAR not in self.situation.syntax_hints)
                ):
                    pass
                # in all other cases, we construct a localiser phrase to attach as an adverbial modifier
                else:
                    # this determines the coverb based on whether the situation is static or dynamic
                    coverb: str = "dzai4"
                    if self.situation.after_action_relations:
                        coverb = "dau4"
                    # handle people as goals
                    if (
                        action
                        and (argument_role == GOAL or argument_role == GOAL_MANIPULATOR)
                        and action.action_type not in [GIVE, GO, COME]
                        and (
                            IS_SPEAKER in filler.reference_object.properties
                            or IS_ADDRESSEE in filler.reference_object.properties
                            or filler.reference_object.ontology_node in [DAD, MOM, BABY]
                        )
                    ):
                        coverb = "gei3"
                    # notice we don't use gwo here since it indicates motion past a place and here, we handle goals
                    self.dependency_graph.add_edge(
                        DependencyTreeToken(coverb, ADPOSITION),
                        reference_object_dependency_node,
                        role=CASE_SPATIAL,
                    )
                    if coverb == "gei3":
                        return (
                            ADVERBIAL_CLAUSE_MODIFIER,
                            reference_object_dependency_node,
                        )

                    # get the localiser and add it to the noun as well
                    localiser_dependency_node = DependencyTreeToken(
                        self._localiser_for_region_as_goal(filler), ADPOSITION
                    )
                    self.dependency_graph.add_edge(
                        localiser_dependency_node,
                        reference_object_dependency_node,
                        role=NOMINAL_MODIFIER,
                    )
                # this is an adverbial clause modifier since it occurs post-verbally (https://github.com/isi-vista/adam/issues/797)
                return (ADVERBIAL_CLAUSE_MODIFIER, reference_object_dependency_node)
            else:
                raise RuntimeError(
                    "The only argument role we can currently handle regions as a filler "
                    "for is GOAL"
                )

        def _localiser_for_region_as_goal(self, region: SituationRegion) -> str:
            """
            When a `Region` appears as the filler of the semantic role `GOAL`,
            determine what localiser to use to express it in Chinese.
            """
            # in/inside
            if region.distance == INTERIOR:
                return "li3"
            # on (typically shang is used specifically for above & in-contact in Chinese)
            elif (
                region.distance == EXTERIOR_BUT_IN_CONTACT
                and region.direction
                and region.direction.positive
            ):
                return "shang4"
            # this is how we currently handle "to" in Chinese, but there's not a real equivalent
            elif region.distance == PROXIMAL and not region.direction:
                if USE_NEAR in self.situation.syntax_hints:
                    return "pang2 byan1"
                return "shang4"
            elif region.distance == DISTAL and not region.direction:
                return "ywan3 li2"
            # TODO: https://github.com/isi-vista/adam/issues/846 -- above/over distinction
            # over
            elif region.direction == GRAVITATIONAL_UP:
                # above
                if USE_ABOVE_BELOW in self.situation.syntax_hints:
                    return "sya4 fang1"
                return "shang4 myan4"
            # below or under -- no distinction in chinese
            elif region.direction == GRAVITATIONAL_DOWN:
                return "sya4 myan4"
            # region.distance == DISTAL is not check as this does not define a specific location in scope for Phase 1
            elif region.direction and self.situation.axis_info:
                if not self.situation.axis_info.addressee:
                    raise RuntimeError(
                        f"Unable to translate region into a localiser because an addressee is lacking. "
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
                    # in front of
                    if region.direction.positive:
                        return "chyan2 myan4"
                    # behind
                    else:
                        return "hou4 myan4"
                # next to/ beside
                elif (
                    region.direction.relative_to_axis != GRAVITATIONAL_DOWN_TO_UP_AXIS
                    and region.distance == PROXIMAL
                ):
                    return "pang2 byan1"
            else:
                raise RuntimeError(
                    f"Don't know how to translate {region} to a localiser yet"
                )

        def _translate_argument_role(
            self,
            action: Action[OntologyNode, SituationObject],
            verb_lexical_entry: LexiconEntry,
            argument_role: OntologyNode,
        ) -> DependencyRole:
            """Translate an argument role to a syntactic role so verb arguments can be joined to trees"""

            # Subject: e.g. THOMAS reads the book
            if argument_role == AGENT:
                return NOMINAL_SUBJECT
            # Patient: e.g. James smashes the LEGO CASTLE
            elif argument_role == PATIENT:
                return OBJECT
            elif argument_role == THEME:
                if AGENT in action.argument_roles_to_fillers:
                    # Chinese "ba" construction: if there's a theme and a goal (or after-action relation, which will likely
                    # be translated post-verbially but the verb isn't ditransitive, the theme becomes preverbial.
                    # The "ba" itself is added by the calling function
                    if (
                        (
                            GOAL in action.argument_roles_to_fillers
                            and IGNORE_GOAL not in self.situation.syntax_hints
                        )
                        or self.situation.after_action_relations
                        or (
                            action.during
                            and action.during.objects_to_paths
                            # hack since these modifiers don't change the position of the sentence but towards/away do
                            and any(
                                any(
                                    k not in [HARD_FORCE, SOFT_FORCE, FAST, SLOW]
                                    for k in v.properties
                                )
                                for k, v in action.during.objects_to_paths.items()
                            )
                        )
                        or USE_ADVERBIAL_PATH_MODIFIER in self.situation.syntax_hints
                    ) and (
                        PREFER_DITRANSITIVE not in self.situation.syntax_hints
                        or ALLOWS_DITRANSITIVE not in verb_lexical_entry.properties
                    ):
                        return OBLIQUE_NOMINAL
                    # if there's no goal or the verb is ditransitive and the syntax hints include using ditransitive, it's still post verbal
                    else:
                        return OBJECT
                else:
                    # the theme can be the subject if there is not an agent in the action (e.g. the ball falls)
                    return NOMINAL_SUBJECT
            # goals are translated to indirect objects
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
            # https://github.com/isi-vista/adam/issues/802 -- currently for Chinese we only count salient objects

            if _object.ontology_node not in self.object_counts:
                return
            else:
                count = self.object_counts[_object.ontology_node]

            # make sure there is a corresponding ontology node
            if not _object.ontology_node:
                raise RuntimeError(
                    f"Don't know how to handle objects which don't correspond to "
                    f"an ontology node currently: {_object}"
                )

            # check if the situation object is the speaker; if it is, return the firt person pronoun
            if IS_SPEAKER in _object.properties:
                noun_lexicon_entry = ME

            # check if the situation object is the addressee; if it is, return the second person pronoun
            elif IS_ADDRESSEE in _object.properties:
                noun_lexicon_entry = YOU

            # if not wo or ni, then just an object which we get the corresponding noun for
            else:
                noun_lexicon_entry = self._unique_lexicon_entry(
                    _object.ontology_node  # pylint:disable=protected-access
                )

            # create a dependency node for the noun and add it to the graph
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

            # handle sentences of the form 'the ball is red'
            if ATTRIBUTES_AS_X_IS_Y in self.situation.syntax_hints:
                # this can only be used in static situations
                if self.situation.is_dynamic:
                    raise RuntimeError(
                        "X is Y language can't be provided in dynamic situations"
                    )
                # get the attributes; exclude colours if that is a part of the syntax hints
                properties = [
                    property_
                    for property_ in _object.properties
                    if IGNORE_COLORS not in self.situation.syntax_hints
                    or not self.situation.ontology.is_subtype_of(property_, COLOR)
                ]
                if len(properties) != 1:
                    raise RuntimeError(
                        f"Cannot handle X is Y language with {len(properties)} attributes"
                    )
                attribute_lexicon_entry = self._unique_lexicon_entry(first(properties))
                node = DependencyTreeToken(
                    attribute_lexicon_entry.base_form,
                    attribute_lexicon_entry.part_of_speech,
                    attribute_lexicon_entry.intrinsic_morphosyntactic_properties,
                )
                is_node = DependencyTreeToken("shr4", VERB)
                self.dependency_graph.add_edge(
                    is_node, dependency_node, role=IS_ATTRIBUTE
                )
                self.dependency_graph.add_edge(
                    node, dependency_node, role=NOMINAL_MODIFIER
                )

            # if colour is specified, add it as an adjectival modifier
            elif IGNORE_COLORS not in self.situation.syntax_hints:
                for property_ in _object.properties:
                    if self.situation.ontology.is_subtype_of(property_, COLOR):
                        color_lexicon_entry = self._unique_lexicon_entry(property_)
                        # create a node for the colour and add it to the graph
                        color_node = DependencyTreeToken(
                            color_lexicon_entry.base_form,
                            color_lexicon_entry.part_of_speech,
                            color_lexicon_entry.intrinsic_morphosyntactic_properties,
                        )
                        self.dependency_graph.add_edge(
                            color_node, dependency_node, role=ADJECTIVAL_MODIFIER
                        )

            # add the node corresponding to the object to the mapping and return that node
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
            """Add a classifier, count, or possessive to a given noun"""

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
                # handle the possession relation if there is one
                possessor = None
                if IS_SPEAKER in possession_relations[0].first_slot.properties:
                    possessor = DependencyTreeToken("wo3 de", NOUN)
                elif IS_ADDRESSEE in possession_relations[0].first_slot.properties:
                    possessor = DependencyTreeToken("ni3 de", NOUN)
                # if the possessor is a 3rd person, check that "has" isn't the main verb
                elif (
                    IGNORE_HAS_AS_VERB not in self.situation.syntax_hints
                    and not self.situation.is_dynamic
                ):
                    return
                    # TODO: we currently return here since we can't handle one possessive node and one not for the third person (i.e. I have my ball, you have your ball
                    # but right now, we just have "Dad has a ball" since otherwise we'll end up adding "de" (equivalent of 's in English) to both instances (e.g. Dad's has Dad's ball)
                    # since the nodes aren't unique). https://github.com/isi-vista/adam/issues/55
                # handle the 3rd person possessor based on the relation
                elif (not self.situation.is_dynamic) or (
                    possession_relations[0].first_slot
                    not in only(self.situation.actions).argument_roles_to_fillers[AGENT]
                ):
                    possessor = self._noun_for_object(possession_relations[0].first_slot)
                    de = DependencyTreeToken("de", PARTICLE)
                    self.dependency_graph.add_edge(de, possessor, role=CASE_POSSESSIVE)
                # if there is a possessor, add "de" (the rough equivalent of 's in English) and add the resulting node to the tree
                if possessor:
                    self.dependency_graph.add_edge(
                        possessor, noun_dependency_node, role=NOMINAL_MODIFIER_POSSESSIVE
                    )
            # if the count is one, we're done since we're not using yi CLF currently
            # also, we don't count grounds or proper nouns
            if count == 0:
                raise RuntimeError(f"Invalid count for object {noun_lexicon_entry}")
            if (
                count == 1
                or _object.ontology_node == GROUND
                or PROPER_NOUN in _object.properties
            ):
                return
            elif count == 2:
                two = DependencyTreeToken("lyang3", NUMERAL)
                classifier = noun_lexicon_entry.counting_classifier
                if not classifier:
                    classifier = "ge4"
                self.dependency_graph.add_edge(
                    two, noun_dependency_node, role=NUMERIC_MODIFIER
                )
                self.dependency_graph.add_edge(
                    DependencyTreeToken(classifier, PARTICLE),
                    noun_dependency_node,
                    role=CLASSIFIER,
                )
            # if the count is many, we don't need a CLF, and we just use many (this will be checked by a native speaker in the next round of checks)
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
            """Translate relations that the user explicitly calls out, including possession and region"""

            if relation.relation_type == BIGGER_THAN:
                if (
                    relation.first_slot in self.situation.salient_objects
                    and isinstance(relation.second_slot, SituationObject)
                    and relation.second_slot.ontology_node == LEARNER
                ):
                    # tall
                    if USE_VERTICAL_MODIFIERS in self.situation.syntax_hints:
                        token = DependencyTreeToken("gau1 da4", ADJECTIVE)
                    # big
                    else:
                        token = DependencyTreeToken("da4", ADJECTIVE)
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
                    # short
                    if USE_VERTICAL_MODIFIERS in self.situation.syntax_hints:
                        token = DependencyTreeToken("dwan3", ADJECTIVE)
                    # small
                    else:
                        token = DependencyTreeToken("syau3", ADJECTIVE)
                    self.dependency_graph.add_node(token)
                    self.dependency_graph.add_edge(
                        token,
                        self._noun_for_object(relation.first_slot),
                        role=ADJECTIVAL_MODIFIER,
                    )

            # handle possession relations
            elif relation.relation_type == HAS:
                # if the situation is dynamic, then this will be handled within the NP
                if (
                    self.situation.is_dynamic
                    or IGNORE_HAS_AS_VERB in self.situation.syntax_hints
                ):
                    pass
                # otherwise, we translate "has" to a verb
                else:
                    self._translate_relation_to_verb(relation)
            # handle in_region relations
            elif relation.relation_type == IN_REGION:
                # make sure that a relation isn't translated twice, as is the case for over+under when we have a relation saying
                # "x is over y" and "y is under x" -- we only want to translate one of these, not both
                if self.dependency_graph.out_degree[
                    self._noun_for_object(relation.first_slot)
                ]:
                    return
                # get the localiser modifier
                localiser_modifier = self.relation_to_localiser_modifier(action, relation)
                if localiser_modifier:
                    # when there's not an action, we use the "de" construction (e.g. the table on the floor = the on the floor de table)
                    # this function is only called from within the NP generator, so we don't need to case on action
                    de = DependencyTreeToken("de", PARTICLE)
                    self.dependency_graph.add_edge(
                        de, localiser_modifier, role=CASE_POSSESSIVE
                    )
                    # it's a little odd to consider "de" as a possessive here, but it's syntactically equivalent
                    self.dependency_graph.add_edge(
                        localiser_modifier,
                        self._noun_for_object(relation.first_slot),
                        role=CASE_SPATIAL,
                    )

            else:
                raise RuntimeError(
                    f"We can't currently translate {relation} relation to Chinese"
                )

        def _translate_relation_to_verb(
            self, relation: Relation[SituationObject]
        ) -> DependencyTreeToken:
            """Translates a possession relation to a verb"""

            # get the lexicon entry, create a corresponding node, and add it to the tree
            lexicon_entry = self._unique_lexicon_entry(relation.relation_type)
            verb_dependency_node = DependencyTreeToken(
                lexicon_entry.base_form, lexicon_entry.part_of_speech
            )
            self.dependency_graph.add_node(verb_dependency_node)

            # get a noun for the first slot and add it to the tree as the subject
            first_slot_dependency_node = self._noun_for_object(relation.first_slot)
            self.dependency_graph.add_edge(
                first_slot_dependency_node, verb_dependency_node, role=NOMINAL_SUBJECT
            )

            # get a noun for the second object and add it to the tree
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

        def relation_to_localiser_modifier(
            self,
            action: Optional[Action[OntologyNode, SituationObject]],
            relation: Relation[SituationObject],
        ) -> Optional[DependencyTreeToken]:
            """Translate a relation to a localizer phrase modifier"""

            region = cast(SituationRegion, relation.second_slot)
            # if the object in the relation is not salient, then we don't care about the relation
            if region.reference_object not in self.situation.salient_objects:
                return None
            # deal with actions with verbs -- if the relationship is already represented by the verb, we're done
            if action:
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

            localiser: Optional[str] = None
            if region.distance == INTERIOR and not relation.negated:
                localiser = "li3"
            # to/towards -- this functions differntly in Chinese but this is the best approximation to handle it
            elif region.distance == PROXIMAL and not region.direction:
                if USE_NEAR in self.situation.syntax_hints:
                    localiser = "pang2 byan1"
                else:
                    localiser = "shang4"
            elif region.distance == DISTAL and not region.direction:
                localiser = "hen3 ywan3"
            elif region.direction:
                direction_axis = region.direction.relative_to_concrete_axis(
                    self.situation.axis_info
                )
                # on & in contact
                if region.distance == EXTERIOR_BUT_IN_CONTACT:
                    if region.direction.positive:
                        localiser = "shang4"
                else:
                    if direction_axis.aligned_to_gravitational:
                        # over
                        if region.direction.positive:
                            if USE_ABOVE_BELOW in self.situation.syntax_hints:
                                localiser = "shang4 fang1"
                            else:
                                localiser = "shang4 myan4"
                        # under
                        else:
                            localiser = "sya4 myan4"
                    else:
                        if isinstance(
                            region.direction.relative_to_axis, FacingAddresseeAxis
                        ):
                            # in front of
                            if region.direction.positive:
                                localiser = "chyan2 myan4"
                            # behind
                            else:
                                localiser = "hou4 myan4"
                        # beside
                        elif region.distance == PROXIMAL:
                            localiser = "pang2 byan1"

            # get the noun for the NP in the localiser phrase
            reference_object_node = self._noun_for_object(region.reference_object)

            # if there's no localiser, this is a relation we don't know how to handle
            if not localiser:
                # handle out, which is a coverb rather than a localiser
                if region.distance == INTERIOR and relation.negated:
                    coverb = "chu1"
                    self.dependency_graph.add_edge(
                        DependencyTreeToken(coverb, ADPOSITION),
                        reference_object_node,
                        role=CASE_SPATIAL,
                    )
                    return reference_object_node
                else:
                    raise RuntimeError(
                        f"Don't know how to handle {relation} as a localiser"
                    )

            # this means that the reference node is already in the graph, so we're done
            if self.dependency_graph.out_degree[reference_object_node]:
                return None
            # if the reference node isn't already in the graph, add it
            else:
                self.dependency_graph.add_edge(
                    DependencyTreeToken(localiser, ADPOSITION),
                    reference_object_node,
                    role=NOMINAL_MODIFIER,
                )
                # get the coverb that will be used in this phrase: https://github.com/isi-vista/adam/issues/796
                # zai is used by default
                coverb = "dzai4"
                # dao is used for after_action relations since this indicates that there was motion to a goal. If a relation
                # appears in both after and during, we translate it as after
                if (
                    relation in self.situation.after_action_relations
                    and relation not in self.situation.always_relations
                ):
                    coverb = "dau4"
                # gwo is used for during.at_some_point since it indicates motion past a point continuing afterwards (rather
                # than stopping at a goal as dao and zai do
                if (
                    action
                    and action.during
                    and relation in action.during.at_some_point
                    and relation not in self.situation.after_action_relations
                    and relation not in self.situation.always_relations
                ):
                    coverb = "gwo4"
                if region.distance == DISTAL and not region.direction:
                    coverb = "li2"
                # add the coverb to the localiser phrase and return it
                self.dependency_graph.add_edge(
                    DependencyTreeToken(coverb, ADPOSITION),
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
                        object_.ontology_node
                        for object_ in self.situation.salient_objects
                    )
                }


# create an instance of the rule-based language generator for our current ontology
GAILA_PHASE_1_CHINESE_LANGUAGE_GENERATOR = SimpleRuleBasedChineseLanguageGenerator(
    ontology_lexicon=GAILA_PHASE_1_CHINESE_LEXICON
)

GAILA_PHASE_2_CHINESE_LANGUAGE_GENERATOR = SimpleRuleBasedChineseLanguageGenerator(
    ontology_lexicon=GAILA_PHASE_2_CHINESE_LEXICON
)

# these are "hints" situations can pass to the language generator
# to control its behavior
# See https://github.com/isi-vista/adam/issues/222
USE_ADVERBIAL_PATH_MODIFIER = "USE_ADVERBIAL_PATH_MODIFIER"
PREFER_DITRANSITIVE = "PREFER_DITRANSITIVE"
IGNORE_COLORS = "IGNORE_COLORS"
IGNORE_HAS_AS_VERB = "IGNORE_HAS_AS_VERB"
ATTRIBUTES_AS_X_IS_Y = "ATTRIBUTES_AS_X_IS_Y"
USE_NEAR = "USE_NEAR"
IGNORE_GOAL = "IGNORE_GOAL"
USE_VERTICAL_MODIFIERS = "USE_VERTICAL_MODIFIERS"
USE_ABOVE_BELOW = "USE_ABOVE_BELOW"
