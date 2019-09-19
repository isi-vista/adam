from typing import Tuple

from immutablecollections import immutabledict, ImmutableDict

from adam.language.dependency import (
    HEAD,
    RoleOrderDependencyTreeLinearizer,
    PartOfSpeechTag,
    DependencyRole,
)
from adam.language.dependency.universal_dependencies import (
    VERB,
    OBLIQUE_NOMINAL,
    NOMINAL_SUBJECT,
    INDIRECT_OBJECT,
    OBJECT,
    NOUN,
    CASE_MARKING,
    DETERMINER_ROLE,
    NUMERIC_MODIFIER,
    ADJECTIVAL_MODIFIER,
    NOMINAL_MODIFIER_POSSESSIVE,
)

_ENGLISH_HEAD_TO_ROLE_ORDER: ImmutableDict[
    PartOfSpeechTag, Tuple[DependencyRole, ...]
] = immutabledict(
    [
        (VERB, (NOMINAL_SUBJECT, HEAD, INDIRECT_OBJECT, OBJECT, OBLIQUE_NOMINAL)),
        # At the moment we put CASE_MARKING first because in our current example
        # it corresponds to a preposition, but we will probably need to do something
        # more sophisticated later.
        (
            NOUN,
            (
                CASE_MARKING,
                NOMINAL_MODIFIER_POSSESSIVE,
                DETERMINER_ROLE,
                NUMERIC_MODIFIER,
                ADJECTIVAL_MODIFIER,
                HEAD,
            ),
        ),
    ]
)


SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER = RoleOrderDependencyTreeLinearizer(
    _ENGLISH_HEAD_TO_ROLE_ORDER
)
