from typing import Dict, Mapping, MutableMapping

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutabledict, immutableset
from more_itertools import only
from networkx import DiGraph

from adam.language.dependency import (
    DependencyTree,
    DependencyTreeLinearizer,
    DependencyTreeToken,
    LinearizedDependencyTree,
    DependencyRole,
)
from adam.language.dependency.universal_dependencies import (
    ADJECTIVAL_MODIFIER,
    ADJECTIVE,
    DETERMINER,
    DETERMINER_ROLE,
    PROPER_NOUN,
    NOMINAL_SUBJECT,
    NUMERAL,
    NUMERIC_MODIFIER,
    OBJECT,
    OBLIQUE_NOMINAL,
    ADPOSITION,
    CASE_MARKING,
)
from adam.language.language_generator import LanguageGenerator
from adam.language.lexicon import LexiconEntry
from adam.language.ontology_dictionary import OntologyLexicon
from adam.language_specific.english.english_syntax import (
    SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER,
)
from adam.language_specific.english.english_phase_1_lexicon import MASS_NOUN
from adam.ontology import OntologyNode
from adam.ontology.phase1_ontology import AGENT, PATIENT, THEME, DESTINATION, ON
from adam.random_utils import SequenceChooser
from adam.situation import (
    HighLevelSemanticsSituation,
    SituationObject,
    SituationAction,
    SituationNode,
    SituationRelation,
)


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
            # For now, only apply quantifiers to object-only situations
            if self.situation.actions == ImmutableSet.empty():
                # Get number of objects of each type
                node_counts: Dict[OntologyNode, int] = dict()
                for _object in self.situation.objects:
                    try:
                        node_counts[_object.ontology_node] += 1
                    except KeyError:
                        node_counts.update({_object.ontology_node: 1})
                object_counts: Dict[SituationObject, int] = {
                    _object: node_counts[_object.ontology_node]
                    for _object in self.situation.objects
                }
                # Use the counts to apply the appropriate quantifiers
                for _object in object_counts:
                    self._translate_object_to_noun(_object, object_counts[_object])
            else:
                for _object in self.situation.objects:
                    self._translate_object_to_noun(_object)

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
            self, _object: SituationObject, count: int = 1
        ) -> DependencyTreeToken:
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
            if count > 1:
                dependency_node = DependencyTreeToken(
                    lexicon_entry.plural_form, lexicon_entry.part_of_speech
                )
            else:
                dependency_node = DependencyTreeToken(
                    lexicon_entry.base_form, lexicon_entry.part_of_speech
                )
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
                    determiner_node = DependencyTreeToken("a", DETERMINER)
                    self.dependency_graph.add_edge(
                        determiner_node, dependency_node, role=DETERMINER_ROLE
                    )
                elif count == 2:
                    numeral_node = DependencyTreeToken("two", NUMERAL)
                    self.dependency_graph.add_edge(
                        numeral_node, dependency_node, role=NUMERIC_MODIFIER
                    )
                # Currently, any number of objects greater than two is considered "many"
                else:
                    adjective_node = DependencyTreeToken("many", ADJECTIVE)
                    self.dependency_graph.add_edge(
                        adjective_node, dependency_node, role=ADJECTIVAL_MODIFIER
                    )

            return dependency_node

        def _translate_action_to_verb(
            self, action: SituationAction
        ) -> DependencyTreeToken:
            lexicon_entry = self._unique_lexicon_entry(action.action_type)
            # TODO: we don't currently handle verbal morphology.
            # https://github.com/isi-vista/adam/issues/60
            verb_dependency_node = DependencyTreeToken(
                lexicon_entry.verb_form_3SG_PRS, lexicon_entry.part_of_speech
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
            filler: SituationNode,
            verb_dependency_node: DependencyTreeToken,
        ):
            if isinstance(filler, SituationObject):
                filler_dependency_node = self.objects_to_dependency_nodes[filler]
                self.dependency_graph.add_edge(
                    filler_dependency_node,
                    verb_dependency_node,
                    role=self._translate_argument_role(argument_role),
                )
            elif isinstance(filler, SituationRelation):
                # TODO: this is a hack to handle prepositional arguments!
                # See https://github.com/isi-vista/adam/issues/61
                if filler.relation_type == ON:
                    thing_on_something_situation_node = filler.first_slot

                    if (
                        thing_on_something_situation_node
                        not in action.argument_roles_to_fillers[THEME]
                    ):
                        raise RuntimeError(
                            "We only know how to handle the on case if "
                            "the first slot of the on-relation matches the "
                            "verb's theme. See "
                            "https://github.com/isi-vista/adam/issues/61"
                        )

                    thing_it_is_on_dependency_node = self.objects_to_dependency_nodes[
                        filler.second_slot
                    ]
                    self.dependency_graph.add_edge(
                        thing_it_is_on_dependency_node,
                        verb_dependency_node,
                        role=OBLIQUE_NOMINAL,
                    )
                    on_dependency_node = DependencyTreeToken("on", ADPOSITION)
                    self.dependency_graph.add_edge(
                        on_dependency_node,
                        thing_it_is_on_dependency_node,
                        role=CASE_MARKING,
                    )

                else:
                    raise RuntimeError(
                        "The only relation we currently understand how to "
                        "handle is 'on'. See "
                        "https://github.com/isi-vista/adam/issues/61"
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
        (DESTINATION, OBLIQUE_NOMINAL),
    )
)
