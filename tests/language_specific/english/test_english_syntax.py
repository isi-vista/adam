from networkx import DiGraph

from adam.language.dependency import DependencyTree, DependencyTreeToken
from adam.language.dependency.universal_dependencies import (ADPOSITION, CASE_SPATIAL, DETERMINER,
                                                             DETERMINER_ROLE, NOMINAL_SUBJECT, NOUN,
                                                             OBJECT, OBLIQUE_NOMINAL, VERB)
from adam.language_specific.english.english_syntax import (
    SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER,
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
    tree.add_edge(on, table, role=CASE_SPATIAL)
    tree.add_edge(the_1, table, role=DETERMINER_ROLE)
    tree.add_edge(table, put, role=OBLIQUE_NOMINAL)

    predicted_token_order = tuple(
        node.token
        for node in SIMPLE_ENGLISH_DEPENDENCY_TREE_LINEARIZER.linearize(
            DependencyTree(tree)
        ).surface_token_order
    )
    assert predicted_token_order == ("Mom", "put", "the", "ball", "on", "the", "table")
