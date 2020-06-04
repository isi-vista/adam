"""This file checks our Chinese syntax file to ensure that the generated word order is correct
Once complete, this should be checked by a native speaker"""
from networkx import DiGraph
import pytest
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
    INDIRECT_OBJECT,
    ADVERBIAL_MODIFIER,
    ADVERB,
)
from adam.language_specific.chinese.chinese_syntax import (
    SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER,
)

"""Just a noun"""


def test_basic_noun():
    truck = DependencyTreeToken("ka3 che1", NOUN)
    tree = DiGraph()
    tree.add_node(truck)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("ka3 che1",)


"""Tests simple noun-verb to make sure that is in the correct order"""


def test_basic_noun_verb():
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


"""Tests basic SVO order in Chinese"""


def test_direct_object():
    me = DependencyTreeToken("wo3", NOUN)
    drink = DependencyTreeToken("he1", VERB)
    juice = DependencyTreeToken("gwo3 jr1", NOUN)
    tree = DiGraph()
    tree.add_edge(me, drink, role=NOMINAL_SUBJECT)
    tree.add_edge(juice, drink, role=OBJECT)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("wo3", "he1", "gwo3 jr1")


"""Tests Chinese indirect objects with ditransitive verbs"""


def test_indirect_object():
    me = DependencyTreeToken("wo3", NOUN)
    give = DependencyTreeToken("gei3", VERB)
    you = DependencyTreeToken("ni3", NOUN)
    book = DependencyTreeToken("shu1", NOUN)
    tree = DiGraph()
    tree.add_edge(me, give, role=NOMINAL_SUBJECT)
    tree.add_edge(book, give, role=OBJECT)
    tree.add_edge(you, give, role=INDIRECT_OBJECT)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("wo3", "gei3", "ni3", "shu1")


"""Tests Chinese adverbial modifiers, which typically occur before the verb 
unless they are temporal modifiers"""


def test_adverb_mods():
    me = DependencyTreeToken("wo3", NOUN)
    drink = DependencyTreeToken("he1", VERB)
    slowly = DependencyTreeToken("man4 man de", ADVERB)
    tree = DiGraph()
    tree.add_edge(me, drink, role=NOMINAL_SUBJECT)
    tree.add_edge(slowly, drink, role=ADVERBIAL_MODIFIER)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("wo3", "man4 man de", "he1")
