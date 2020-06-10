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

"""Simple rule-based approach for translating HighLevelSemanticSituations to 
Chinese dependency trees. We currently only generate a single possible linearized
tree for a semantic situation."""


@attrs(frozen=True, slots=True)
class SimpleRuleBasedChineseLanguageGenerator(
    LanguageGenerator[HighLevelSemanticsSituation, LinearizedDependencyTree]
):
    # mapping from nodes in the concept ontology to Chinese words
    _ontology_lexicon: OntologyLexicon = attrib(
        validator=instance_of(OntologyLexicon), kw_only=True
    )

    # how to assign word order to the dependency trees
    _dependency_tree_linearizer: DependencyTreeLinearizer = attrib(
        init=False, default=SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER, kw_only=True
    )

    # the function that actually generates the language
    def generate_language(
        self, situation: HighLevelSemanticsSituation, chooser: SequenceChooser
    ) -> ImmutableSet[LinearizedDependencyTree]:
        # remove once done
        return SimpleRuleBasedChineseLanguageGenerator._Generation(
            self, situation
        ).generate()

    """This class encapsulates all the mutable state for an execution of the 
    SimpleRuleBasedChineseLanguageGenerator on a single input"""

    @attrs(frozen=True, slots=True)
    class _Generation:
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

        """The function that tries to generate language for a given representation"""

        def generate(self) -> ImmutableSet[LinearizedDependencyTree]:
            try:
                return self._real_generate()
            except Exception as e:
                raise RuntimeError(
                    "Error while generating Chinese for the situation {}".format(
                        self.situation
                    )
                ) from e

        """The function that actually generates the language"""

        def _real_generate(self) -> ImmutableSet[LinearizedDependencyTree]:
            # TODO: deal with situations with more than one action
            if len(self.situation.actions) > 1:
                raise RuntimeError(
                    "Currently only situations with 0 or 1 actions are supported"
                )

            # handle the special case of a static situation with only multiple objects of the same type
            object_types_in_situation = set(
                object_.ontology_node for object_ in self.situation.salient_objects
            )

            # handle dynamic situations
            if self.situation.is_dynamic:
                raise NotImplementedError
            # handle static situations
            else:
                # handle one type of object (there may be many of it)
                if len(object_types_in_situation) == 1:
                    first_object = first(self.situation.salient_objects)
                    self._noun_for_object(first_object)
                # multiple objects of different types
                else:
                    for object_ in self.situation.salient_objects:
                        if not self._only_translate_if_referenced(object_):
                            self._noun_for_object(object_)

            # TODO: handle persisting relations

            return immutableset(
                [
                    self.generator._dependency_tree_linearizer.linearize(
                        DependencyTree(self.dependency_graph)
                    )
                ]
            )

        """Get the noun for the object in a given situation"""

        def _noun_for_object(
            self,
            _object: SituationObject,
            *,
            syntactic_role_if_known: Optional[DependencyRole] = None,
        ) -> DependencyTreeToken:

            # if we already have a mapping for a noun, we're done
            if _object in self.objects_to_dependency_nodes:
                return self.objects_to_dependency_nodes[_object]

            # TODO: handle counts

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

            dependency_node = DependencyTreeToken(
                noun_lexicon_entry.base_form,
                noun_lexicon_entry.part_of_speech,
                morphosyntactic_properties=noun_lexicon_entry.intrinsic_morphosyntactic_properties,
            )
            # TODO: handle X_IS_Y and not IGNORE_COLOURS, deal with classifiers
            self.dependency_graph.add_node(dependency_node)
            return dependency_node

        """Get a lexicon entry for a given ontology node"""

        def _unique_lexicon_entry(self, ontology_node: OntologyNode) -> LexiconEntry:
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

        """functions from here on have not been touched yet and just have dummy definitions"""

        def _only_translate_if_referenced(self, object_: SituationObject) -> bool:
            raise NotImplementedError

        def add_classifier(
            self,
            _object: SituationObject,
            count: int,
            noun_dependency_node: DependencyTreeToken,
            *,
            noun_lexicon_entry: LexiconEntry,
        ) -> None:
            raise NotImplementedError


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
