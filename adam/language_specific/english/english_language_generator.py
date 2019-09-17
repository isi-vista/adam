import collections
from typing import Mapping, MutableMapping, Union

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
    ADPOSITION,
    CASE_MARKING,
    DETERMINER,
    DETERMINER_ROLE,
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
from adam.ontology import OntologyNode, Region
from adam.ontology.phase1_ontology import AGENT, GOAL, LEARNER, PATIENT, THEME
from adam.ontology.phase1_spatial_relations import EXTERIOR_BUT_IN_CONTACT, INTERIOR
from adam.random_utils import SequenceChooser
from adam.situation import SituationAction, SituationNode, SituationObject
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

        def generate(self) -> ImmutableSet[LinearizedDependencyTree]:
            # The learner appears in a situation so they items may have spatial relations
            # with respect to it, but our language currently never refers to the learner itself.
            objects_to_translate = [
                object_
                for object_ in self.situation.objects
                if not object_.ontology_node == LEARNER
            ]

            node_counts: Mapping[OntologyNode, int]
            # Get number of objects of each type
            if not self.situation.actions:
                # For now, only apply quantifiers to object-only situations
                node_counts = collections.Counter(
                    [_object.ontology_node for _object in self.situation.objects]
                )
            else:
                node_counts = {
                    ontology_node: 1
                    for ontology_node in immutableset(
                        object_.ontology_node for object_ in objects_to_translate
                    )
                }

            # Use the counts to apply the appropriate quantifiers
            for _object in objects_to_translate:
                self._translate_object_to_noun(
                    _object, count=node_counts[_object.ontology_node]
                )

            if len(self.situation.actions) > 1:
                raise RuntimeError(
                    "Currently only situations with 0 or 1 actions are supported"
                )

            for action in self.situation.actions:
                self._translate_action_to_verb(action)

            return immutableset(
                [
                    self.generator._dependency_tree_linearizer.linearize(  # pylint:disable=protected-access
                        DependencyTree(self.dependency_graph)
                    )
                ]
            )

        def _translate_object_to_noun(
            self, _object: SituationObject, *, count: int = 1
        ) -> DependencyTreeToken:
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
            if count > 1 and lexicon_entry.plural_form:
                word_form = lexicon_entry.plural_form
            else:
                word_form = lexicon_entry.base_form
            dependency_node = DependencyTreeToken(word_form, lexicon_entry.part_of_speech)

            self.dependency_graph.add_node(dependency_node)
            # we remember what dependency node goes with this object
            # so that we can link to it when e.g. it appears
            # as an argument of a verb
            self.objects_to_dependency_nodes[_object] = dependency_node

            # add articles to things which are not proper nouns
            # ("a ball" but not "a Mom")
            if (dependency_node.part_of_speech != PROPER_NOUN) and (
                MASS_NOUN not in lexicon_entry.properties
            ):
                if count == 1:
                    quantifier_node = DependencyTreeToken("a", DETERMINER)
                    quantifier_role = DETERMINER_ROLE
                elif count == 2:
                    quantifier_node = DependencyTreeToken("two", NUMERAL)
                    quantifier_role = NUMERIC_MODIFIER
                # Currently, any number of objects greater than two is considered "many"
                else:
                    quantifier_node = DependencyTreeToken("many", DETERMINER)
                    quantifier_role = DETERMINER_ROLE
                self.dependency_graph.add_edge(
                    quantifier_node, dependency_node, role=quantifier_role
                )

            return dependency_node

        def _translate_action_to_verb(
            self, action: SituationAction
        ) -> DependencyTreeToken:
            lexicon_entry = self._unique_lexicon_entry(action.action_type)
            # TODO: we don't currently handle verbal morphology.
            # https://github.com/isi-vista/adam/issues/60
            if lexicon_entry.verb_form_3SG_PRS:
                verb_dependency_node = DependencyTreeToken(
                    lexicon_entry.verb_form_3SG_PRS, lexicon_entry.part_of_speech
                )
            else:
                raise RuntimeError(
                    f"Verb has no 3SG present tense form: {lexicon_entry.base_form}"
                )
            self.dependency_graph.add_node(verb_dependency_node)

            for (argument_role, filler) in action.argument_roles_to_fillers.items():
                self._translate_verb_argument(
                    action, argument_role, filler, verb_dependency_node
                )
            return verb_dependency_node

        def _translate_verb_argument(
            self,
            action: SituationAction,
            argument_role: OntologyNode,
            filler: Union[SituationNode, Region[SituationObject]],
            verb_dependency_node: DependencyTreeToken,
        ):
            # TODO: to alternation
            # https://github.com/isi-vista/adam/issues/150
            if isinstance(filler, SituationObject):
                filler_dependency_node = self.objects_to_dependency_nodes[filler]
                self.dependency_graph.add_edge(
                    filler_dependency_node,
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

                    reference_object_dependency_node = self.objects_to_dependency_nodes[
                        filler.reference_object
                    ]
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
