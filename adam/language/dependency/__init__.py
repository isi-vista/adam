"""
Representations for dependency trees
"""
from abc import ABC, abstractmethod
from typing import Iterable, Tuple

from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableDict, ImmutableSet, immutabledict, immutableset
from immutablecollections.converter_utils import _to_immutabledict, _to_tuple
from more_itertools import flatten
from networkx import DiGraph

from adam.language import LinguisticDescription


@attrs(frozen=True, slots=True)
class DependencyTree:
    """
    Note a `DependencyTree` is not a `LinguisticDescription`
    because it does not provide a surface token string,
    since the dependencies are unordered.

    You can pair a `DependencyTree` with a surface order
    to create an `LinearizedDependencyTree`
    """

    _graph: DiGraph = attrib(validator=instance_of(DiGraph))
    root: "DependencyTreeToken" = attrib(init=False)
    tokens: ImmutableSet["DependencyTreeToken"] = attrib(init=False)

    def modifiers(
        self, head: "DependencyTreeToken"
    ) -> ImmutableSet[Tuple["DependencyTreeToken", "DependencyRole"]]:
        return immutableset(
            (
                (source, role)
                for (source, target, role) in self._graph.in_edges(head, data="role")
            ),
            disable_order_check=True,
        )

    @root.default
    def _init_root(self) -> "DependencyTreeToken":
        roots = [
            node for node in self._graph.nodes() if self._graph.out_degree(node) == 0
        ]
        if len(roots) == 1:
            return roots[0]
        elif roots:
            raise RuntimeError(f"Dependency tree has multiple roots: {roots}")
        else:
            if self._graph:
                raise RuntimeError("Dependency tree has no roots")
            else:
                raise RuntimeError(
                    "Cannot initialize a dependency tree from an empty graph"
                )

    @tokens.default
    def _init_tokens(self) -> ImmutableSet["DependencyTreeToken"]:
        return immutableset(self._graph.nodes, disable_order_check=True)

    def __attrs_post_init__(self) -> None:
        bad_edges = [
            (source, target)
            for (source, target, role) in self._graph.edges(data="role")
            if role is None
        ]
        if bad_edges:
            raise RuntimeError(
                "Cannot construct a dependency tree with an edges without roles: "
                + ", ".join(f"({source}, {target}" for (source, target) in bad_edges)
            )


@attrs(frozen=True, slots=True)
class LinearizedDependencyTree(LinguisticDescription):
    dependency_tree: DependencyTree = attrib(validator=instance_of(DependencyTree))
    surface_token_order: Tuple["DependencyTreeToken", ...] = attrib(
        converter=_to_tuple, default=()
    )

    def as_token_sequence(self) -> Tuple[str, ...]:
        return tuple(node.token for node in self.surface_token_order)

    def __attrs_post_init__(self) -> None:
        surface_tokens = immutableset(self.surface_token_order)
        if self.dependency_tree.tokens != surface_tokens:
            raise RuntimeError(
                f"Cannot create an OrderedDependencyTree where the"
                f"dependency tree tokens do not match the surface form."
                f"Tree tokens: {self.dependency_tree.tokens}, "
                f"surface tokens: {surface_tokens}"
            )


@attrs(frozen=True, slots=True)
class PartOfSpeechTag:
    name: str = attrib(validator=instance_of(str))


@attrs(frozen=True, slots=True)
class DependencyTreeToken:
    token: str = attrib(validator=instance_of(str))
    pos_tag: PartOfSpeechTag = attrib(validator=instance_of(PartOfSpeechTag))


@attrs(frozen=True, slots=True)
class DependencyRole:
    name: str = attrib(validator=instance_of(str))


class DependencyTreeLinearizer(ABC):
    @abstractmethod
    def linearize(self, dependency_tree: DependencyTree) -> LinearizedDependencyTree:
        """
        Determine a surface word order for a `DependencyTree` .
        Args:
            dependency_tree: The `DependencyTree` to determine a surface word order for.

        Returns:
            A `LinearizedDependencyTree` pairing the input `DependencyTree` with a
            surface word order.
        """


HEAD = DependencyRole("head")


@attrs(frozen=True, slots=True)
class RoleOrderDependencyTreeLinearizer(DependencyTreeLinearizer):
    _head_pos_to_role_order: ImmutableDict[
        PartOfSpeechTag, Tuple[DependencyRole, ...]
    ] = attrib(converter=_to_immutabledict, default=immutabledict())

    def linearize(self, dependency_tree: DependencyTree) -> LinearizedDependencyTree:
        # TODO: handle tokens which correspond to dependency tree edges
        return LinearizedDependencyTree(
            dependency_tree, tuple(self._linearize(dependency_tree, dependency_tree.root))
        )

    def _linearize(
        self, dependency_tree: DependencyTree, head_node: DependencyTreeToken
    ) -> Iterable[DependencyTreeToken]:
        nodes_to_order = list(dependency_tree.modifiers(head_node))
        nodes_to_order.append((head_node, HEAD))

        if len(nodes_to_order) == 1:
            # the head has no modifiers, so there is nothing to order relative to the head
            return (head_node,)

        role_order = self._head_pos_to_role_order[head_node.pos_tag]

        # we need to consider the head node alongside its dependents because the head
        # will in general be positioned in the midst of them. We use the pseudo-dependency
        # HEAD to distinguish it.
        def position(node: Tuple[DependencyTreeToken, DependencyRole]) -> int:
            role = node[1]
            try:
                return role_order.index(role)
            except ValueError:
                raise RuntimeError(
                    f"Do not know how to order modifiers with role "
                    f"{role} relative to head of POS tag "
                    f"{head_node.pos_tag}. We know how to handle the "
                    f"following roles: {role_order}"
                )

        nodes_in_order = sorted(nodes_to_order, key=position)

        return flatten(
            self._linearize(dependency_tree, node)
            # don't recurse infinitely by trying to process the head word again
            if dependency != HEAD else (node,)
            for (node, dependency) in nodes_in_order
        )
