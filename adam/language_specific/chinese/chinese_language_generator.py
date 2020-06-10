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
        raise NotImplementedError
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
