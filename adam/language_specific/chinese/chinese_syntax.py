"""Current implementation of Chinese syntax file; has been checked
by native speaker asked for grammaticality judgments."""

from typing import Tuple

from adam.language.dependency import (
    DependencyRole,
    HEAD,
    PartOfSpeechTag,
    RoleOrderDependencyTreeLinearizer,
)
from adam.language.dependency.universal_dependencies import (
    ADJECTIVAL_MODIFIER,
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
    NUMERIC_MODIFIER,
    OBJECT,
    OBLIQUE_NOMINAL,
    PROPER_NOUN,
    VERB,
)
from immutablecollections import ImmutableDict

_CHINESE_HEAD_TO_ROLE_ORDER: ImmutableDict[
    PartOfSpeechTag, Tuple[DependencyRole, ...]
] = [
    # Currently we treat locations associated with verbs as IO's since obliques occur before the verb.
    # https://github.com/isi-vista/adam/issues/797
    # TODO: find a better way to handle the above
    (
        VERB,
        (
            NOMINAL_SUBJECT,
            ADVERBIAL_MODIFIER,
            OBLIQUE_NOMINAL,
            HEAD,
            ADVERBIAL_CLAUSE_MODIFIER,
            INDIRECT_OBJECT,
            OBJECT,
        ),
    ),
    # according to https://web.stanford.edu/group/cslipublications/cslipublications/HPSG/2007/wang-liu.pdf,
    # the basic structure of a non-"de" NP is possessive, demonstrative, quantities, adjectives, nouns
    # classifiers only occur when an item is being counted. "De", the possessive particle, occurs at the end of
    # possessive NPs.
    (
        NOUN,
        (
            CASE_SPATIAL,
            NOMINAL_MODIFIER_POSSESSIVE,
            NUMERIC_MODIFIER,
            CLASSIFIER,
            ADJECTIVAL_MODIFIER,
            HEAD,
            IS_ATTRIBUTE,
            NOMINAL_MODIFIER,
            CASE_POSSESSIVE,
        ),
    ),
    # a similar structure applies for proper nouns, with the same localizer/preposition issue
    (
        PROPER_NOUN,
        (
            CASE_SPATIAL,
            NOMINAL_MODIFIER_POSSESSIVE,
            NUMERIC_MODIFIER,
            CLASSIFIER,
            ADJECTIVAL_MODIFIER,
            HEAD,
            IS_ATTRIBUTE,
            NOMINAL_MODIFIER,
            CASE_POSSESSIVE,
        ),
    ),
]

SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER = RoleOrderDependencyTreeLinearizer(
    _CHINESE_HEAD_TO_ROLE_ORDER
)
