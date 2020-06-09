"""This file checks our Chinese syntax file to ensure that the generated word order is correct
The structure and vocab have been verified separately by Chinese native speaker."""

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
    ADJECTIVAL_MODIFIER,
    ADJECTIVE,
    ADVERB,
    PARTICLE,
    CASE_POSSESSIVE,
    NOMINAL_MODIFIER_POSSESSIVE,
    NOMINAL_MODIFIER,
    NUMERAL,
    CLASSIFIER,
    NUMERIC_MODIFIER,
    PROPER_NOUN,
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


"""Noun-adjective pair"""


def test_adj_noun():
    truck = DependencyTreeToken("ka3 che1", NOUN)
    red = DependencyTreeToken("hung2 se4", ADJECTIVE)
    tree = DiGraph()
    tree.add_edge(red, truck, role=ADJECTIVAL_MODIFIER)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("hung2 se4", "ka3 che1")


"""Possessives: testing 'my red truck"""


def test_possessives():
    me = DependencyTreeToken("wo3", NOUN)
    de = DependencyTreeToken("de", PARTICLE)
    truck = DependencyTreeToken("ka3 che1", NOUN)
    red = DependencyTreeToken("hung2 se4", ADJECTIVE)
    tree = DiGraph()
    tree.add_edge(de, me, role=CASE_POSSESSIVE)
    tree.add_edge(red, truck, role=ADJECTIVAL_MODIFIER)
    tree.add_edge(me, truck, role=NOMINAL_MODIFIER_POSSESSIVE)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("wo3", "de", "hung2 se4", "ka3 che1")


"""Counting and classifiers in Chinese with '3 dogs'"""


def test_counting():
    three = DependencyTreeToken("san1", NUMERAL)
    clf = DependencyTreeToken("jr1", PARTICLE)
    dog = DependencyTreeToken("gou3", NOUN)
    tree = DiGraph()
    tree.add_edge(clf, dog, role=CLASSIFIER)
    tree.add_edge(three, dog, role=NUMERIC_MODIFIER)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("san1", "jr1", "gou3")


"""More complex noun phrase: my three red trucks"""


def test_my_3_red_trucks():
    me = DependencyTreeToken("wo3", NOUN)
    de = DependencyTreeToken("de", PARTICLE)
    truck = DependencyTreeToken("ka3 che1", NOUN)
    red = DependencyTreeToken("hung2 se4", ADJECTIVE)
    three = DependencyTreeToken("san1", NUMERAL)
    clf = DependencyTreeToken("lyang4", PARTICLE)
    tree = DiGraph()
    tree.add_edge(clf, truck, role=CLASSIFIER)
    tree.add_edge(three, truck, role=NUMERIC_MODIFIER)
    tree.add_edge(de, me, role=CASE_POSSESSIVE)
    tree.add_edge(red, truck, role=ADJECTIVAL_MODIFIER)
    tree.add_edge(me, truck, role=NOMINAL_MODIFIER_POSSESSIVE)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == (
        "wo3",
        "de",
        "san1",
        "lyang4",
        "hung2 se4",
        "ka3 che1",
    )


"""Test a localizer phrase with everything: next to mom's three red trucks"""


def test_long_localizer_phrase():
    de = DependencyTreeToken("de", PARTICLE)
    truck = DependencyTreeToken("ka3 che1", NOUN)
    red = DependencyTreeToken("hung2 se4", ADJECTIVE)
    three = DependencyTreeToken("san1", NUMERAL)
    clf = DependencyTreeToken("lyang4", PARTICLE)
    near = DependencyTreeToken("fu4, jin4", NOUN)
    at = DependencyTreeToken("dzai4", ADPOSITION)
    ma = DependencyTreeToken("ma1ma1", PROPER_NOUN)
    tree = DiGraph()
    tree.add_edge(at, truck, role=CASE_SPATIAL)
    tree.add_edge(near, truck, role=NOMINAL_MODIFIER)
    tree.add_edge(clf, truck, role=CLASSIFIER)
    tree.add_edge(three, truck, role=NUMERIC_MODIFIER)
    tree.add_edge(de, ma, role=CASE_POSSESSIVE)
    tree.add_edge(red, truck, role=ADJECTIVAL_MODIFIER)
    tree.add_edge(ma, truck, role=NOMINAL_MODIFIER_POSSESSIVE)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == (
        "dzai4",
        "ma1ma1",
        "de",
        "san1",
        "lyang4",
        "hung2 se4",
        "ka3 che1",
        "fu4, jin4",
    )


"""Testing for proper Nouns: 'mom's black dog"""


def test_proper_noun_possessive():
    ma = DependencyTreeToken("ma1", PROPER_NOUN)
    de = DependencyTreeToken("de", PARTICLE)
    dog = DependencyTreeToken("gou3", NOUN)
    black = DependencyTreeToken("hei1 se4", ADJECTIVE)
    tree = DiGraph()
    tree.add_edge(de, ma, role=CASE_POSSESSIVE)
    tree.add_edge(ma, dog, role=NOMINAL_MODIFIER_POSSESSIVE)
    tree.add_edge(black, dog, role=ADJECTIVAL_MODIFIER)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("ma1", "de", "hei1 se4", "gou3")


"""tests localizers and adjective for proper nouns: near mom"""


def test_proper_noun_modified():
    near = DependencyTreeToken("fu4, jin4", NOUN)
    at = DependencyTreeToken("dzai4", ADPOSITION)
    ma = DependencyTreeToken("ma1", PROPER_NOUN)
    tree = DiGraph()
    tree.add_edge(at, ma, role=CASE_SPATIAL)
    tree.add_edge(near, ma, role=NOMINAL_MODIFIER)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("dzai4", "ma1", "fu4, jin4")


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
    give = DependencyTreeToken("gei3", VERB)
    you = DependencyTreeToken("ni3", NOUN)
    book = DependencyTreeToken("shu1", NOUN)
    slowly = DependencyTreeToken("man4 man", ADVERB)
    tree = DiGraph()
    tree.add_edge(me, give, role=NOMINAL_SUBJECT)
    tree.add_edge(book, give, role=OBJECT)
    tree.add_edge(you, give, role=INDIRECT_OBJECT)
    tree.add_edge(slowly, give, role=ADVERBIAL_MODIFIER)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("wo3", "man4 man", "gei3", "ni3", "shu1")


"""Tests Chinese prepositional phrase occurring pre-verb (indicates place where verbal action took place)"""


def test_preverbial_prep():
    me = DependencyTreeToken("wo3", NOUN)
    at = DependencyTreeToken("dzai4", ADPOSITION)
    table = DependencyTreeToken("jwo1 dz", NOUN)
    on = DependencyTreeToken("shang4", NOUN)
    eat = DependencyTreeToken("chr1", VERB)
    tree = DiGraph()
    tree.add_edge(at, table, role=CASE_SPATIAL)
    tree.add_edge(on, table, role=NOMINAL_MODIFIER)
    tree.add_edge(me, eat, role=NOMINAL_SUBJECT)
    tree.add_edge(table, eat, role=OBLIQUE_NOMINAL)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("wo3", "dzai4", "jwo1 dz", "shang4", "chr1")


"""Tests Chinese coverbial phrase occuring after the verb (indicating something about the action)"""


def test_I_put_the_book_on_the_table():
    me = DependencyTreeToken("wo3", NOUN)
    book = DependencyTreeToken("shu1", NOUN)
    ba = DependencyTreeToken("ba3", PARTICLE)
    put = DependencyTreeToken("fang4", VERB)
    at = DependencyTreeToken("dzai4", ADPOSITION)
    table = DependencyTreeToken("jwo1 dz", NOUN)
    on = DependencyTreeToken("shang4", NOUN)
    tree = DiGraph()
    tree.add_edge(ba, book, role=CASE_SPATIAL)
    tree.add_edge(book, put, role=OBLIQUE_NOMINAL)
    tree.add_edge(me, put, role=NOMINAL_SUBJECT)
    tree.add_edge(at, table, role=CASE_SPATIAL)
    tree.add_edge(on, table, role=NOMINAL_MODIFIER)
    # TODO: this is a bit of a hack since I'm not sure this really counts as an IO, but I'm not sure what to classify it as
    #  since PP's occur before the verb in Chinese
    tree.add_edge(table, put, role=INDIRECT_OBJECT)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == (
        "wo3",
        "ba3",
        "shu1",
        "fang4",
        "dzai4",
        "jwo1 dz",
        "shang4",
    )


"""This test isn't too different from the one above but will be at the language generation stage"""


def test_I_push_the_book_along_the_table():
    me = DependencyTreeToken("wo3", NOUN)
    book = DependencyTreeToken("shu1", NOUN)
    ba = DependencyTreeToken("ba3", PARTICLE)
    push = DependencyTreeToken("twei1", VERB)
    at = DependencyTreeToken("dau4", ADPOSITION)
    table = DependencyTreeToken("jwo1 dz", NOUN)
    on = DependencyTreeToken("shang4", NOUN)
    tree = DiGraph()
    tree.add_edge(ba, book, role=CASE_SPATIAL)
    tree.add_edge(book, push, role=OBLIQUE_NOMINAL)
    tree.add_edge(me, push, role=NOMINAL_SUBJECT)
    tree.add_edge(at, table, role=CASE_SPATIAL)
    tree.add_edge(on, table, role=NOMINAL_MODIFIER)
    # TODO: this is a bit of a hack since I'm not sure this really counts as an IO, but I'm not sure what to classify it as
    tree.add_edge(table, push, role=INDIRECT_OBJECT)
    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_CHINESE_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == (
        "wo3",
        "ba3",
        "shu1",
        "twei1",
        "dau4",
        "jwo1 dz",
        "shang4",
    )
