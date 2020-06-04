"""This file checks our Chinese syntax file to ensure that the generated word order is correct
Once complete, this should be checked by a native speaker"""
from networkx import DiGraph

from adam.language.dependency import DependencyTree, DependencyTreeToken
from adam.language.dependency.universal_dependencies import (
    ADPOSITION,
    CASE_SPATIAL,
    DETERMINER,
    DETERMINER_ROLE,
    NOMINAL_SUBJECT,
    NOUN,
    OBJECT,
    OBLIQUE_NOMINAL,
    VERB,
)
from adam.language_specific.chinese.chinese_syntax import (
    SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER,
)

"""Tests simple noun-verb to make sure that is in the correct order"""


def test_basic_noun():
    ball = DependencyTreeToken("chyou2", NOUN)
    go = DependencyTreeToken("dzou3", VERB)

    tree = DiGraph()
    tree.add_edge(ball, go, role=NOMINAL_SUBJECT)

    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("chyou2", "dzou3")
