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
)
from adam.language.dependency.universal_dependencies import (
    DETERMINER,
    DETERMINER_ROLE,
    PROPER_NOUN,
)
from adam.language.language_generator import LanguageGenerator
from adam.language.ontology_dictionary import OntologyLexicon
from adam.random_utils import SequenceChooser
from adam.situation import HighLevelSemanticsSituation, SituationObject


@attrs(frozen=True, slots=True)
class SimpleRuleBasedEnglishLanguageGenerator(
    LanguageGenerator[HighLevelSemanticsSituation, LinearizedDependencyTree]
):
    _ontology_lexicon: OntologyLexicon = attrib(
        validator=instance_of(OntologyLexicon), kw_only=True
    )
    _dependency_tree_linearizer: DependencyTreeLinearizer = attrib(
        validator=instance_of(DependencyTreeLinearizer), kw_only=True
    )

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
        generator: "SimpleRuleBasedEnglishLanguageGenerator" = attrib()
        situation: HighLevelSemanticsSituation = attrib()
        dependency_graph: DiGraph = attrib(init=False, default=Factory(DiGraph))

        def generate(self) -> ImmutableSet[LinearizedDependencyTree]:
            immutabledict(
                [
                    (_object, self._translate_object_to_noun(_object))
                    for _object in self.situation.objects
                ]
            )
            # TODO: currently only return a single interpretation
            return immutableset(
                [
                    self.generator._dependency_tree_linearizer.linearize(  # pylint:disable=protected-access
                        DependencyTree(self.dependency_graph)
                    )
                ]
            )

        def _translate_object_to_noun(
            self, _object: SituationObject
        ) -> DependencyTreeToken:
            if not _object.ontology_node:
                raise RuntimeError(
                    f"Don't know how to handle objects which don't correspond to "
                    f"an ontology node currently: {_object}"
                )
            # TODO: we don't currently translate modifiers to nominals
            lexicon_entries = self.generator._ontology_lexicon.words_for_node(  # pylint:disable=protected-access
                _object.ontology_node
            )
            if lexicon_entries:
                if len(lexicon_entries) == 1:
                    lexicon_entry = only(lexicon_entries)
                    # TODO: how to get proper noun?
                    dependency_node = DependencyTreeToken(
                        lexicon_entry.base_form, lexicon_entry.part_of_speech
                    )
                    self.dependency_graph.add_node(dependency_node)

                    # add articles to things which are not proper nouns (don't want "the Mom")
                    if dependency_node.pos_tag != PROPER_NOUN:
                        # TODO: address determiners other than "a" during language generation
                        determiner_node = DependencyTreeToken("a", DETERMINER)
                        self.dependency_graph.add_edge(
                            determiner_node, dependency_node, role=DETERMINER_ROLE
                        )

                    return dependency_node
                else:
                    raise RuntimeError(
                        f"We don't yet know how to deal with ontology nodes which "
                        f"could be realized by multiple lexical entries: "
                        f"{_object.ontology_node} --> {lexicon_entries}"
                    )
            else:
                raise RuntimeError(
                    f"No lexicon entry for ontology node {_object.ontology_node}"
                )
