from attr import attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from more_itertools import flatten
from typing import Callable, Generic, Iterable, Tuple, TypeVar, Union, Any
from vistautils.preconditions import check_arg

from adam.ontology import OntologyNode, IN_REGION, Region

ObjectT = TypeVar("ObjectT")
ObjectTOut = TypeVar("ObjectTOut", contravariant=True)


@attrs(frozen=True, repr=False)
class Relation(Generic[ObjectT]):
    r"""
    A relationship which holds between two objects
    or between a `SituationObject` and a Region.
    The latter case is allowed only for the special relation `IN_REGION` .

    This is used for relations in `Situation`\ s, `ObjectStructuralSchema`\ ta,
    perceptions, etc. since they have the same structure.
    """

    relation_type: OntologyNode = attrib(validator=instance_of(OntologyNode))
    first_slot: ObjectT = attrib()
    # for type ignore see
    # https://github.com/isi-vista/adam/issues/144
    second_slot: Union[ObjectT, Region[ObjectT]] = attrib()
    negated: bool = attrib(validator=instance_of(bool), default=False, kw_only=True)

    def negated_copy(self) -> "Relation[ObjectT]":
        # mypy doesn't know attrs classes provide evolve for copying
        return self.evolve(negated=not self.negated)  # type: ignore

    def __attrs_post_init__(self) -> None:
        check_arg(
            not isinstance(self.second_slot, Region) or self.relation_type == IN_REGION
        )

    def __repr__(self) -> str:
        negated_string = "-" if self.negated else ""
        return (
            f"{negated_string}{self.relation_type}({self.first_slot}, {self.second_slot})"
        )


# DSL to make writing object hierarchies easier


# commented out is what should be the true type signature, but it seems to confuse mypy
# def flatten_relations(
#     relation_collections: Iterable[Union[Relation[ObjectT], Iterable[Relation[ObjectT]]]]
# ) -> ImmutableSet[Relation[ObjectT]]:
def flatten_relations(
    relation_collections: Iterable[Union[Relation[Any], Iterable[Relation[Any]]]]
) -> ImmutableSet[Relation[Any]]:
    """
    Convenience method to enable writing sub-object relations
    in an `ObjectStructuralSchema` more easily.

    This method simply flattens collections of items in the input iterable.

    This is useful because it allows you to write methods for your relations which produce
    collections of relations as their output. This allows you to use such DSL-like methods to
    enforce constraints between the relations.

    Please see adam.ontology.phase1_ontology.PERSON_SCHEMA for an example of how this is useful.
    """
    return immutableset(flatten(_ensure_iterable(x) for x in relation_collections))


_OneOrMoreObjects = Union[ObjectT, Iterable[ObjectT]]

_T = TypeVar("_T")


def _ensure_iterable(x: Union[_T, Iterable[_T]]) -> Iterable[_T]:
    if isinstance(x, Iterable):
        return x
    else:
        return (x,)


def make_dsl_relation(
    relation_type: OntologyNode
) -> Callable[
    [Union[ObjectT, Iterable[ObjectT]], Union[ObjectT, Iterable[ObjectT]]],
    Tuple[Relation[ObjectT], ...],
]:
    r"""
    Make a function which, when given either single or groups
    of arguments for two slots of a `Relation`,
    generates `Relation`\ s of type *relation_type*
    for the cross-product of the arguments.

    See `adam.ontology.phase1_ontology` for many examples.
    """

    def dsl_relation_function(
        arg1s: Union[ObjectT, Iterable[ObjectT]], arg2s: Union[ObjectT, Iterable[ObjectT]]
    ) -> Tuple[Relation[ObjectT], ...]:
        return tuple(
            Relation(relation_type, arg1, arg2)
            for arg1 in _ensure_iterable(arg1s)
            for arg2 in _ensure_iterable(arg2s)
        )

    return dsl_relation_function


def make_symetric_dsl_relation(
    relation_type: OntologyNode
) -> Callable[
    [Union[ObjectT, Iterable[ObjectT]], Union[ObjectT, Iterable[ObjectT]]],
    Tuple[Relation[ObjectT], ...],
]:
    r"""
    Make a function which, when given either single or groups
    of arguments for two slots of a `Relation`,
    generates a symmetric `Relation`\ s of type *relation_type*
    for the cross-product of the arguments.

    See `adam.ontology.phase1_ontology` for many examples.
    """

    def dsl_symetric_function(
        arg1s: Union[ObjectT, Iterable[ObjectT]], arg2s: Union[ObjectT, Iterable[ObjectT]]
    ) -> Tuple[Relation[ObjectT], ...]:
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
    [Union[ObjectT, Iterable[ObjectT]], Union[ObjectT, Iterable[ObjectT]]],
    Tuple[Relation[ObjectT], ...],
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
        arg1s: Union[ObjectT, Iterable[ObjectT]], arg2s: Union[ObjectT, Iterable[ObjectT]]
    ) -> Tuple[Relation[ObjectT], ...]:
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
# def make_dsl_region_relation(
#     region_factory: Callable[[ObjectT], Region[ObjectT]]
# ) -> Callable[
#     [Union[ObjectT, Iterable[ObjectT]], Union[ObjectT, Iterable[ObjectT]]],
#     Tuple[Relation[ObjectT], ...],
# ]:
def make_dsl_region_relation(
    region_factory: Callable[[Any], Region[Any]]
) -> Callable[
    [Union[Any, Iterable[Any]], Union[Any, Iterable[Any]]], Tuple[Relation[Any], ...]
]:
    def dsl_relation_function(
        arg1s: Union[ObjectT, Iterable[ObjectT]], arg2s: Union[ObjectT, Iterable[ObjectT]]
    ) -> Tuple[Relation[ObjectT], ...]:
        return tuple(
            Relation(IN_REGION, arg1, region_factory(arg2))
            for arg1 in _ensure_iterable(arg1s)
            for arg2 in _ensure_iterable(arg2s)
        )

    return dsl_relation_function


# Proper signature commented-out, see https://github.com/isi-vista/adam/issues/161
#
# def make_symmetric_dsl_region_relation(
#     region_factory: Callable[[ObjectT], Region[ObjectT]]
# ) -> Callable[
#     [Union[ObjectT, Iterable[ObjectT]], Union[ObjectT, Iterable[ObjectT]]],
#     Tuple[Relation[ObjectT], ...],
# ]:
def make_symmetric_dsl_region_relation(
    region_factory: Callable[[ObjectT], Region[ObjectT]]
) -> Callable[
    [Union[Any, Iterable[Any]], Union[Any, Iterable[Any]]], Tuple[Relation[Any], ...]
]:
    def dsl_relation_function(
        arg1s: Union[ObjectT, Iterable[ObjectT]], arg2s: Union[ObjectT, Iterable[ObjectT]]
    ) -> Tuple[Relation[ObjectT], ...]:
        arg1s = _ensure_iterable(arg1s)
        arg2s = _ensure_iterable(arg2s)
        return flatten(
            [
                tuple(
                    Relation(IN_REGION, arg1, region_factory(arg2))
                    for arg1 in arg1s
                    for arg2 in arg2s
                ),
                tuple(
                    Relation(IN_REGION, arg2, region_factory(arg1))
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
    region_factory: Callable[[Any], Region[Any]],
    opposite_region_factory: Callable[[Any], Region[Any]],
) -> Callable[
    [Union[Any, Iterable[Any]], Union[Any, Iterable[Any]]], Tuple[Relation[Any], ...]
]:
    def dsl_relation_function(
        arg1s: Union[ObjectT, Iterable[ObjectT]], arg2s: Union[ObjectT, Iterable[ObjectT]]
    ) -> Tuple[Relation[ObjectT], ...]:
        arg1s = _ensure_iterable(arg1s)
        arg2s = _ensure_iterable(arg2s)
        return flatten(
            [
                tuple(
                    Relation(IN_REGION, arg1, region_factory(arg2))
                    for arg1 in arg1s
                    for arg2 in arg2s
                ),
                tuple(
                    Relation(IN_REGION, arg2, opposite_region_factory(arg1))
                    for arg1 in arg1s
                    for arg2 in arg2s
                ),
            ]
        )

    return dsl_relation_function


def negate(relations: Iterable[Relation[ObjectT]]) -> Iterable[Relation[ObjectT]]:
    return (relation.negated_copy() for relation in relations)
