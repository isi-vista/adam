from typing import Any, Generic, Iterable, List, Mapping, TYPE_CHECKING, TypeVar, Union

from more_itertools import flatten

from adam.ontology import IN_REGION, OntologyNode
from adam.remappable import CanRemapObjects
from attr import attrib, attrs, evolve
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from vistautils.preconditions import check_arg

if TYPE_CHECKING:
    from adam.ontology.phase1_spatial_relations import Region


_ObjectT = TypeVar("_ObjectT")
_NewObjectT = TypeVar("_NewObjectT")


@attrs(frozen=True, repr=False)
class Relation(Generic[_ObjectT]):
    r"""
    A relationship which holds between two objects
    or between a `SituationObject` and a Region.
    The latter case is allowed only for the special relation `IN_REGION` .

    This is used for relations in `Situation`\ s, `ObjectStructuralSchema`\ ta,
    perceptions, etc. since they have the same structure.
    """

    relation_type: OntologyNode = attrib(validator=instance_of(OntologyNode))
    first_slot: _ObjectT = attrib()
    # for type ignore see
    # https://github.com/isi-vista/adam/issues/144
    second_slot: Union[_ObjectT, "Region[_ObjectT]"] = attrib()
    negated: bool = attrib(validator=instance_of(bool), default=False, kw_only=True)

    def negated_copy(self) -> "Relation[_ObjectT]":
        # mypy doesn't know attrs classes provide evolve for copying
        return evolve(self, negated=not self.negated)  # type: ignore

    def copy_remapping_objects(
        self, object_mapping: Mapping[_ObjectT, _NewObjectT]
    ) -> "Relation[_NewObjectT]":
        translated_second_slot: Union[_NewObjectT, "Region[_NewObjectT]"]
        if isinstance(self.second_slot, CanRemapObjects):
            translated_second_slot = self.second_slot.copy_remapping_objects(  # type: ignore
                object_mapping
            )
        else:
            translated_second_slot = object_mapping[self.second_slot]
        return Relation(
            self.relation_type,
            first_slot=object_mapping[self.first_slot],
            second_slot=translated_second_slot,
            negated=self.negated,
        )

    def accumulate_referenced_objects(self, object_accumulator: List[_ObjectT]) -> None:
        r"""
        Adds all objects referenced by this `Relation`
        or any `Region`\ s it refers to to *object_accumulator*.
        """
        object_accumulator.append(self.first_slot)
        if isinstance(self.second_slot, CanRemapObjects):
            self.second_slot.accumulate_referenced_objects(object_accumulator)
        else:
            object_accumulator.append(self.second_slot)

    def __attrs_post_init__(self) -> None:
        check_arg(
            not isinstance(self.second_slot, CanRemapObjects)
            or self.relation_type == IN_REGION
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


_T = TypeVar("_T")


def _ensure_iterable(x: Union[_T, Iterable[_T]]) -> Iterable[_T]:
    if isinstance(x, Iterable):
        return x
    else:
        return (x,)
