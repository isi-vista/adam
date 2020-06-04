"""Draft Chinese syntax file; not all structures currently implemented.
Should be checked by a native speaker"""
from typing import Tuple

# import immutable dictionaries
from immutablecollections import ImmutableDict, immutabledict

# import some dependencies
from adam.language.dependency import (
    DependencyRole,
    HEAD,
    PartOfSpeechTag,
    RoleOrderDependencyTreeLinearizer,
)

# import some universal dependencies
from adam.language.dependency.universal_dependencies import (
    ADJECTIVAL_MODIFIER,
    ADVERBIAL_MODIFIER,
    CASE_POSSESSIVE,
    CASE_SPATIAL,
    DETERMINER_ROLE,
    INDIRECT_OBJECT,
    NOMINAL_MODIFIER,
    NOMINAL_MODIFIER_POSSESSIVE,
    NOMINAL_SUBJECT,
    NOUN,
    NUMERIC_MODIFIER,
    OBJECT,
    OBLIQUE_NOMINAL,
    PROPER_NOUN,
    VERB,
    IS_ATTRIBUTE,
)

_CHINESE_HEAD_TO_ROLE_ORDER: ImmutableDict[
    PartOfSpeechTag, Tuple[DependencyRole, ...]
] = [
    (VERB, (NOMINAL_SUBJECT, HEAD, INDIRECT_OBJECT, OBJECT)),
    (NOUN, (HEAD,)),
    (PROPER_NOUN, (HEAD,)),
]

SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER = RoleOrderDependencyTreeLinearizer(
    _CHINESE_HEAD_TO_ROLE_ORDER
)
