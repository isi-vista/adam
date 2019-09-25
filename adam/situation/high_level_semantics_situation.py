from collections import Counter
from itertools import chain
from typing import Optional

from attr import attrib, attrs
from attr.validators import instance_of, optional
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset
from more_itertools import flatten
from vistautils.preconditions import check_arg

from adam.axes import AxesInfo
from adam.ontology import OntologyNode
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import is_recognized_particular
from adam.ontology.phase1_spatial_relations import Region
from adam.relation import Relation, flatten_relations
from adam.situation import Action, Situation, SituationObject


@attrs(slots=True, repr=False)
class HighLevelSemanticsSituation(Situation):
    """
    A human-friendly representation of `Situation`.
    """

    ontology: Ontology = attrib(validator=instance_of(Ontology))
    """
    What `Ontology` items from the objects, relations, and actions 
    in this `Situation` will come from.
    """
    salient_objects: ImmutableSet[SituationObject] = attrib(converter=_to_immutableset)
    axis_info: AxesInfo[SituationObject] = attrib(
        validator=instance_of(AxesInfo), kw_only=True
    )
    """
    The salient objects present in a `Situation`.  
    This will usually be the ones expressed in the linguistic form.
    """
    other_objects: ImmutableSet[SituationObject] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    r"""
    These are other objects appearing in the situation which are less important.
    For example, the cup holding a liquid being drunk.
    
    These typically correspond to auxiliary variables in `ActionDescription`\ s.
    """
    all_objects: ImmutableSet[SituationObject] = attrib(init=False)
    always_relations: ImmutableSet[Relation[SituationObject]] = attrib(
        converter=flatten_relations, default=immutableset(), kw_only=True
    )
    """
    The relations which hold in this `Situation`,
    both before and after any actions which occur.
    
    It is not necessary to state every relationship which holds in a situation.
    Rather this should contain the salient relationships
    which should be expressed in the linguistic description.
    
    Do not specify those relations here which are *implied* by any actions which occur.
    Those are handled automatically. 
    """
    before_action_relations: ImmutableSet[Relation[SituationObject]] = attrib(
        converter=flatten_relations, default=immutableset(), kw_only=True
    )
    """
    The relations which hold in this `Situation`,
    before, but not necessarily after, any actions which occur.
    
    It is not necessary to state every relationship which holds in a situation.
    Rather this should contain the salient relationships
    which should be expressed in the linguistic description.
    
    Do not specify those relations here which are *implied* by any actions which occur.
    Those are handled automatically. 
    """
    after_action_relations: ImmutableSet[Relation[SituationObject]] = attrib(
        converter=flatten_relations, default=immutableset(), kw_only=True
    )
    """
    The relations which hold in this `Situation`,
    after, but not necessarily before, any actions which occur.

    It is not necessary to state every relationship which holds in a situation.
    Rather this should contain the salient relationships
    which should be expressed in the linguistic description.

    Do not specify those relations here which are *implied* by any actions which occur.
    Those are handled automatically. 
    """
    actions: ImmutableSet[Action[OntologyNode, SituationObject]] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    """
    The actions occurring in this `Situation`
    """
    is_dynamic: bool = attrib(init=False)
    """
    Bool representing whether the situation has any actions, i.e is dynamic. 
    """
    gazed_objects: ImmutableSet[SituationObject] = attrib(
        converter=_to_immutableset, kw_only=True
    )
    r"""
    A set of `SituationObject` s which are the focus of the speaker. 
    Defaults to all semantic role fillers of situation actions.
    """
    syntax_hints: ImmutableSet[str] = attrib(
        converter=_to_immutableset, default=immutableset(), kw_only=True
    )
    """
    A temporary hack to allow control of language generation decisions
    using the situation template language.
    
    See https://github.com/isi-vista/adam/issues/222 .
    """
    from_template: Optional[str] = attrib(
        validator=optional(instance_of(str)), kw_only=True, default=None
    )

    def relation_always_holds(self, query_relation: Relation[SituationObject]) -> bool:
        # TODO: extend to handle transitive relations
        # https://github.com/isi-vista/adam/issues/195
        return query_relation in self.always_relations

    def __attrs_post_init__(self) -> None:
        check_arg(self.salient_objects, "A situation must contain at least one object")
        for relation in self.always_relations:
            if not isinstance(relation.first_slot, SituationObject) or not isinstance(
                relation.second_slot, (SituationObject, Region)
            ):
                raise RuntimeError(
                    f"Relation fillers for situations must be situation objects "
                    f"but got {relation}"
                )
        self.is_dynamic = len(self.actions) > 0
        for relation in self.always_relations:
            if (
                isinstance(relation.second_slot, Region)
                and relation.second_slot.reference_object not in self.all_objects
            ):
                raise RuntimeError(
                    f"Any object referred to by a region must be included in the "
                    f"set of situation objects but region {relation.second_slot}"
                    f" with situation objects {self.all_objects}"
                )
        for action in self.actions:
            for action_role_filler in flatten(
                action.argument_roles_to_fillers.value_groups()
            ):
                if (
                    isinstance(action_role_filler, SituationObject)
                    and action_role_filler not in self.all_objects
                ):
                    raise RuntimeError(
                        "Any object filling a semantic role must be included in the "
                        "set of situation objects."
                    )
                elif (
                    isinstance(action_role_filler, Region)
                    and action_role_filler.reference_object not in self.all_objects
                ):
                    raise RuntimeError(
                        "Any object referred to by a region must be included in the "
                        "set of situation objects."
                    )
        if not self.actions and (
            self.before_action_relations or self.after_action_relations
        ):
            raise RuntimeError(
                "Cannot specify relations to hold before or after actions "
                "if there are no actions"
            )

        # A situation cannot have multiple instances of the same recognized particular.
        # This blocks e.g. Dad gave Dad a house.
        recognized_particular_count = Counter(
            object_.ontology_node
            for object_ in self.all_objects
            if is_recognized_particular(self.ontology, object_.ontology_node)
        )
        for (recognized_particular, count) in recognized_particular_count.items():
            if count > 1:
                raise RuntimeError(
                    f"Cannot have two instances of a recognized particular in a "
                    f"situation, but got {count} instances of {recognized_particular}"
                    f" in {self}"
                )

    def __repr__(self) -> str:
        # TODO: the way we currently repr situations doesn't handle multiple nodes
        # of the same ontology type well.  We'd like to use subscripts (_0, _1)
        # to distinguish them, which requires pulling all the repr logic up to this
        # level and not delegating to the reprs of the sub-objects.
        # https://github.com/isi-vista/adam/issues/62
        lines = ["{"]
        lines.append("\tsalient objects:")
        lines.extend(f"\t\t{obj!r}" for obj in self.salient_objects)
        if self.other_objects:
            lines.append("\tother objects:")
            lines.extend(f"\t\t{obj!r}" for obj in self.other_objects)
        if self.always_relations:
            lines.append("\talways relations:")
            lines.extend(f"\t\t{relation!r}" for relation in self.always_relations)
        if self.before_action_relations:
            lines.append("\tbefore relations:")
            lines.extend(f"\t\t{relation!r}" for relation in self.before_action_relations)
        if self.after_action_relations:
            lines.append("\tafter relations:")
            lines.extend(f"\t\t{relation!r}" for relation in self.after_action_relations)
        if self.syntax_hints:
            lines.append("\tsyntax hints:")
            lines.extend(f"\t\t{self.syntax_hints}")
        if self.actions:
            lines.append("\tactions:")
            lines.extend(f"\t\t{action!r}" for action in self.actions)
        if self.from_template:
            lines.append(f"\tfrom template: {self.from_template}")
        lines.append("}")
        return "\n".join(lines)

    @gazed_objects.default
    def _determine_gazed_objects(self):
        return immutableset(
            object_
            for action in self.actions
            for (_, object_) in action.argument_roles_to_fillers.items()
        )

    @all_objects.default
    def _init_all_objects(self) -> ImmutableSet[SituationObject]:
        return immutableset(chain(self.salient_objects, self.other_objects))
