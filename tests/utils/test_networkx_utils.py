from networkx import DiGraph

from adam.utils.networkx_utils import digraph_with_nodes_sorted_by


def test_digraph_with_nodes_sorted_by():
    initial_graph = DiGraph()

    initial_graph.add_node('a', foo='meep', bar=3)
    initial_graph.add_node('b', la='al')
    initial_graph.add_node('c')
    initial_graph.add_edge('a', 'b', some_attribute='something')
    initial_graph.add_edge('a', 'c', some_other_attribute='fred')

    def by_inverse_size_of_attributes(node_node_data_tuple) -> int:
        (_, node_data) = node_node_data_tuple
        return len(node_data)

    sorted_graph = digraph_with_nodes_sorted_by(initial_graph, by_inverse_size_of_attributes)

    assert list(sorted_graph.nodes(data=True)) == [('c', {}),
                                                   ('b', {'la': 'al'}),
                                                    ('a', {'foo': 'meep',
                                                           'bar': 3})]
    assert list(sorted_graph.edges(data=True)) == [
        ('a', 'b', {'some_attribute': 'something'}),
        ('a', 'c', {'some_other_attribute': 'fred'})
    ]