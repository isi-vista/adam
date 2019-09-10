r"""
Our strategy for `SituationTemplate`\ s in Phase 1 of ADAM.
"""
import random
from _random import Random
from itertools import product
from typing import Iterable, Sequence, TypeVar

from attr import Factory, attrib, attrs
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset

from adam.ontology import Ontology, OntologyNode, THING, ABSTRACT
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.ontology.selectors import ByHierarchyAndProperties, OntologyNodeSelector
from adam.random_utils import RandomChooser, SequenceChooser
from adam.situation import HighLevelSemanticsSituation
from adam.situation.templates import (
    SituationTemplate,
    SituationTemplateObject,
    SituationTemplateProcessor,
)


@attrs(frozen=True, slots=True)
class Phase1SituationTemplate(SituationTemplate):
    object_variables: ImmutableSet["TemplateObjectVariable"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )


@attrs(frozen=True, slots=True)
class Phase1SituationTemplateGenerator(
    SituationTemplateProcessor[Phase1SituationTemplate, HighLevelSemanticsSituation]
):
    # can be specified for testing purposes
    ontology: Ontology = attrib(default=GAILA_PHASE_1_ONTOLOGY)

    def generate_situations(
        self,
        template: Phase1SituationTemplate,
        *,
        chooser: SequenceChooser = Factory(
            RandomChooser.for_seed
        )  # pylint:disable=unused-argument
    ) -> Iterable[HighLevelSemanticsSituation]:
        # TODO: fix hard-coded rng
        rng = Random()
        rng.seed(0)

        object_var_to_options = {
            obj_var: shuffled(obj_var.node_selector.select_nodes(self.ontology), rng)
            for obj_var in template.object_variables
        }

        object_combinations = product(*object_var_to_options.values())

        for object_combination in object_combinations:
            yield HighLevelSemanticsSituation(
                ontology=self.ontology, objects=object_combination
            )


_T = TypeVar("_T")


def shuffled(items: Iterable[_T], rng: Random) -> Sequence[_T]:
    items_list = list(items)
    random.shuffle(items_list, rng.random)
    return items_list


# TODO: justify cmp=False in docstring
@attrs(frozen=True, slots=True, cmp=False)
class TemplateObjectVariable(SituationTemplateObject):
    node_selector: OntologyNodeSelector = attrib(
        validator=instance_of(OntologyNodeSelector)
    )


def object_variable(
    debug_handle: str,
    root_node: OntologyNode = THING,
    with_properties: Iterable[OntologyNode] = immutableset(),
):
    return TemplateObjectVariable(
        debug_handle,
        ByHierarchyAndProperties(
            descendents_of=root_node,
            required_properties=with_properties,
            banned_properties=immutableset([ABSTRACT]),
        ),
    )
