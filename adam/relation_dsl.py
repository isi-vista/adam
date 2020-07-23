"""
A simple domain-specific language for more compactly specifying relations.
"""
from typing import Any, Callable, Iterable, Tuple, TypeVar, Union

from more_itertools import flatten

from adam.ontology import IN_REGION, OntologyNode
from adam.ontology.phase1_spatial_relations import Direction, Distance, Region
from adam.relation import Relation, _ensure_iterable

_ObjectT = TypeVar("_ObjectT")


def make_dsl_relation(
    relation_type: OntologyNode
) -> Callable[
    [Union[_ObjectT, Iterable[_ObjectT]], Union[_ObjectT, Iterable[_ObjectT]]],
    Tuple[Relation[_ObjectT], ...],
]:
    r"""
    Make a function which, when given either single or groups
    of arguments for two slots of a `Relation`,
    generates `Relation`\ s of type *relation_type*
    for the cross-product of the arguments.

    See `adam.ontology.phase1_ontology` for many examples.
    """

    def dsl_relation_function(
        arg1s: Union[_ObjectT, Iterable[_ObjectT]],
        arg2s: Union[_ObjectT, Iterable[_ObjectT]],
    ) -> Tuple[Relation[_ObjectT], ...]:
        return tuple(
            Relation(relation_type, arg1, arg2)
            for arg1 in _ensure_iterable(arg1s)
            for arg2 in _ensure_iterable(arg2s)
        )

    return dsl_relation_function


def make_symetric_dsl_relation(
    relation_type: OntologyNode
) -> Callable[
    [Union[_ObjectT, Iterable[_ObjectT]], Union[_ObjectT, Iterable[_ObjectT]]],
    Tuple[Relation[_ObjectT], ...],
]:
    r"""
    Make a function which, when given either single or groups
    of arguments for two slots of a `Relation`,
    generates a symmetric `Relation`\ s of type *relation_type*
    for the cross-product of the arguments.

    See `adam.ontology.phase1_ontology` for many examples.
    """

    def dsl_symetric_function(
        arg1s: Union[_ObjectT, Iterable[_ObjectT]],
        arg2s: Union[_ObjectT, Iterable[_ObjectT]],
    ) -> Tuple[Relation[_ObjectT], ...]:
        arg1s = _ensure_iterable(arg1s)
        arg2s = _ensure_iterable(arg2s)
        return flatten(
            [
                tuple(
                    Relation(relation_type, arg1, arg2)
                    for arg1 in arg1s
                    for arg2 in arg2s
                ),
                tuple(
                    Relation(relation_type, arg2, arg1)
                    for arg2 in arg2s
                    for arg1 in arg1s
                ),
            ]
        )

    return dsl_symetric_function


def make_opposite_dsl_relation(
    relation_type: OntologyNode, *, opposite_type: OntologyNode
) -> Callable[
    [Union[_ObjectT, Iterable[_ObjectT]], Union[_ObjectT, Iterable[_ObjectT]]],
    Tuple[Relation[_ObjectT], ...],
]:
    r"""
    Make a function which, when given either single or groups
    of arguments for two slots of a `Relation`,
    generates a  `Relation`\ s of type *relation_type*
    and an inverse of type *opposite_type* for the for the
    reversed cross-product of the arguments

    See `adam.ontology.phase1_ontology` for many examples.
    """

    def dsl_opposite_function(
        arg1s: Union[_ObjectT, Iterable[_ObjectT]],
        arg2s: Union[_ObjectT, Iterable[_ObjectT]],
    ) -> Tuple[Relation[_ObjectT], ...]:
        arg1s = _ensure_iterable(arg1s)
        arg2s = _ensure_iterable(arg2s)
        return flatten(
            [
                tuple(
                    Relation(relation_type, arg1, arg2)
                    for arg1 in arg1s
                    for arg2 in arg2s
                ),
                tuple(
                    Relation(opposite_type, arg2, arg1)
                    for arg1 in arg1s
                    for arg2 in arg2s
                ),
            ]
        )

    return dsl_opposite_function


# Proper signature commented-out, see https://github.com/isi-vista/adam/issues/161
#
# def make_symmetric_dsl_region_relation(
#     region_factory: Callable[[ObjectT], Region[ObjectT]]
# ) -> Callable[
#     [Union[ObjectT, Iterable[ObjectT]], Union[ObjectT, Iterable[ObjectT]]],
#     Tuple[Relation[ObjectT], ...],
# ]:


def make_symmetric_dsl_region_relation(
    region_factory: Callable[..., "Region[_ObjectT]"]
) -> Callable[..., Tuple[Relation[Any], ...]]:
    def dsl_relation_function(
        arg1s: Union[_ObjectT, Iterable[_ObjectT]],
        arg2s: Union[_ObjectT, Iterable[_ObjectT]],
        **kw_args,
    ) -> Tuple["Relation[_ObjectT]", ...]:
        arg1s = _ensure_iterable(arg1s)
        arg2s = _ensure_iterable(arg2s)
        return flatten(
            [
                tuple(
                    Relation(IN_REGION, arg1, region_factory(arg2, **kw_args))
                    for arg1 in arg1s
                    for arg2 in arg2s
                ),
                tuple(
                    Relation(IN_REGION, arg2, region_factory(arg1, **kw_args))
                    for arg1 in arg1s
                    for arg2 in arg2s
                ),
            ]
        )

    return dsl_relation_function


# Proper signature commented-out, see https://github.com/isi-vista/adam/issues/161
#
# def make_opposite_dsl_region_relation(
#     region_factory: Callable[[ObjectT], Region[ObjectT]],
#     opposite_region_factory: Callable[[ObjectT], Region[ObjectT]],
# ) -> Callable[
#     [Union[ObjectT, Iterable[ObjectT]], Union[ObjectT, Iterable[ObjectT]]],
#     Tuple[Relation[ObjectT], ...],
# ]:


def make_opposite_dsl_region_relation(
    region_factory: Callable[..., "Region[Any]"],
    opposite_region_factory: Callable[..., "Region[Any]"],
) -> Callable[..., Tuple[Relation[Any], ...]]:
    def dsl_relation_function(
        arg1s: Union[_ObjectT, Iterable[_ObjectT]],
        arg2s: Union[_ObjectT, Iterable[_ObjectT]],
        **kw_args,
    ) -> Tuple[Relation[_ObjectT], ...]:
        arg1s = _ensure_iterable(arg1s)
        arg2s = _ensure_iterable(arg2s)
        return flatten(
            [
                tuple(
                    Relation(IN_REGION, arg1, region_factory(arg2, **kw_args))
                    for arg1 in arg1s
                    for arg2 in arg2s
                ),
                tuple(
                    Relation(IN_REGION, arg2, opposite_region_factory(arg1, **kw_args))
                    for arg1 in arg1s
                    for arg2 in arg2s
                ),
            ]
        )

    return dsl_relation_function


# This is currently used for over/under since English can't handle opposite relations with all salient objects ( see
# https://github.com/isi-vista/adam/issues/802)
def make_dsl_region_relation(
    region_factory: Callable[..., "Region[Any]"]
) -> Callable[..., Tuple[Relation[Any], ...]]:
    def dsl_relation_function(
        arg1s: Union[_ObjectT, Iterable[_ObjectT]],
        arg2s: Union[_ObjectT, Iterable[_ObjectT]],
        **kw_args,
    ) -> Tuple["Relation[_ObjectT]", ...]:
        return tuple(
            Relation(IN_REGION, arg1, region_factory(arg2, **kw_args))
            for arg1 in _ensure_iterable(arg1s)
            for arg2 in _ensure_iterable(arg2s)
        )

    return dsl_relation_function


def located(
    arg1s: Union[_ObjectT, Iterable[_ObjectT]],
    arg2s: Union[_ObjectT, Iterable[_ObjectT]],
    *,
    distance: Distance,
    # We need to do `Direction[Any]` because we can't infer the generic type
    # when a GeonAxis is used as the relative_to_axis.
    # Using Any here let's use isolate the type:ignores to this function.
    direction: Direction[Any],
) -> Tuple[Relation[_ObjectT]]:
    """
    All *arg1s* are located with at the given `Distance` and `Direction`
    with respect to *args2*.

    It is usually better to use more specialized relations derived using
    `make_opposite_dsl_region_relation`, etc.,
    but then can be useful when you need, for example, to refer to particular concrete axes.
    """
    arg1s = _ensure_iterable(arg1s)
    arg2s = _ensure_iterable(arg2s)

    return flatten(
        [
            tuple(
                Relation(
                    IN_REGION, arg1, Region(arg2, distance=distance, direction=direction)
                )
                for arg1 in arg1s
                for arg2 in arg2s
            ),
            tuple(
                Relation(
                    IN_REGION,
                    arg2,
                    Region(arg1, distance=distance, direction=direction.opposite()),
                )
                for arg1 in arg1s
                for arg2 in arg2s
            ),
        ]
    )


def negate(relations: Iterable[Relation[_ObjectT]]) -> Iterable[Relation[_ObjectT]]:
    return (relation.negated_copy() for relation in relations)
