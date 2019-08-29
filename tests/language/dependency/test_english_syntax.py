from networkx import DiGraph

from adam.language.dependency import DependencyTreeToken, DependencyTree
from adam.language.dependency.english_syntax import (
    SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER,
)
from adam.language.dependency.universal_dependencies import (
    DETERMINER,
    ADPOSITION,
    NOMINAL_SUBJECT,
    DETERMINER_ROLE,
    OBJECT,
    CASE_MARKING,
    OBLIQUE_NOMINAL,
    NOUN,
    VERB,
)


def test_mom_put_the_ball_on_the_table():
    mom = DependencyTreeToken("Mom", NOUN)
    put = DependencyTreeToken("put", VERB)
    the_0 = DependencyTreeToken("the", DETERMINER)
    ball = DependencyTreeToken("ball", NOUN)
    on = DependencyTreeToken("on", ADPOSITION)
    the_1 = DependencyTreeToken("the", DETERMINER)
    table = DependencyTreeToken("table", NOUN)

    tree = DiGraph()
    tree.add_edge(mom, put, role=NOMINAL_SUBJECT)
    tree.add_edge(the_0, ball, role=DETERMINER_ROLE)
    tree.add_edge(ball, put, role=OBJECT)
    tree.add_edge(on, table, role=CASE_MARKING)
    tree.add_edge(the_1, table, role=DETERMINER_ROLE)
    tree.add_edge(table, put, role=OBLIQUE_NOMINAL)

    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("Mom", "put", "the", "ball", "on", "the", "table")
