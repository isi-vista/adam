from typing import Tuple

from immutablecollections import ImmutableDict, immutabledict

from adam.language.dependency import (
    DependencyRole,
    HEAD,
    MorphosyntacticProperty,
    PartOfSpeechTag,
    RoleOrderDependencyTreeLinearizer,
)
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
)

_ENGLISH_HEAD_TO_ROLE_ORDER: ImmutableDict[
    PartOfSpeechTag, Tuple[DependencyRole, ...]
] = immutabledict(
    [
        (
            VERB,
            (
                NOMINAL_SUBJECT,
                HEAD,
                INDIRECT_OBJECT,
                OBJECT,
                OBLIQUE_NOMINAL,
                ADVERBIAL_MODIFIER,
            ),
        ),
        # At the moment we put CASE_MARKING first because in our current example
        # it corresponds to a preposition, but we will probably need to do something
        # more sophisticated later.
        (
            NOUN,
            (
                CASE_SPATIAL,
                NOMINAL_MODIFIER_POSSESSIVE,
                DETERMINER_ROLE,
                NUMERIC_MODIFIER,
                ADJECTIVAL_MODIFIER,
                HEAD,
                # Right now all our nmods are prepositional phrases,
                # so this works, but we will need something more
                # sophisticated than this map eventually to handle
                # distinctions between noun modifier types.
                NOMINAL_MODIFIER,
                CASE_POSSESSIVE,
            ),
        ),
        (
            PROPER_NOUN,
            (CASE_SPATIAL, ADJECTIVAL_MODIFIER, HEAD, NOMINAL_MODIFIER, CASE_POSSESSIVE),
        ),
    ]
)


SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER = RoleOrderDependencyTreeLinearizer(
    _ENGLISH_HEAD_TO_ROLE_ORDER
)

FIRST_PERSON = MorphosyntacticProperty("1p")
SECOND_PERSON = MorphosyntacticProperty("1p")
THIRD_PERSON = MorphosyntacticProperty("1p")
