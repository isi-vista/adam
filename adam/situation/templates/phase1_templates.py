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

from adam.ontology import OntologyNode, THING, ABSTRACT
from adam.ontology.ontology import Ontology
from adam.ontology.phase1_ontology import GAILA_PHASE_1_ONTOLOGY
from adam.ontology.selectors import ByHierarchyAndProperties, OntologyNodeSelector
from adam.random_utils import RandomChooser, SequenceChooser
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation
from adam.situation.templates import (
    SituationTemplate,
    SituationTemplateObject,
    SituationTemplateProcessor,
)


@attrs(frozen=True, slots=True)
class Phase1SituationTemplate(SituationTemplate):
    r"""
    The `SituationTemplate` implementation used in Phase 1 of the ADAM project.

    Currently, this can only be a collection of `TemplateObjectVariable`\ s.

    `Phase1SituationTemplateGenerator` will translate these
    to a sequence `HighLevelSemanticsSituation`\ s corresponding
    to the Cartesian product of the possible values of the *object_variables*.

    Beware that this can be very large if the number of object variables
    or the number of possible values of the variables is even moderately large.
    """
    object_variables: ImmutableSet["TemplateObjectVariable"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )


@attrs(frozen=True, slots=True)
class Phase1SituationTemplateGenerator(
    SituationTemplateProcessor[Phase1SituationTemplate, HighLevelSemanticsSituation]
):
    r"""
    Generates `HighLevelSemanticsSituation`\ s from `Phase1SituationTemplate`\ s.
    """
    # can be set to something besides GAILA_PHASE_1_ONTOLOGY for testing purposes
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
            obj_var: _shuffled(obj_var.node_selector.select_nodes(self.ontology), rng)
            for obj_var in template.object_variables
        }

        object_combinations = product(*object_var_to_options.values())

        for object_combination in object_combinations:
            yield HighLevelSemanticsSituation(
                ontology=self.ontology, objects=object_combination
            )


_T = TypeVar("_T")


def _shuffled(items: Iterable[_T], rng: Random) -> Sequence[_T]:
    """
    Return the elements of *items* in shuffled order,
    using *rng* as the source of randomness.

    This should eventually get shifted to VistaUtils.
    """
    items_list = list(items)
    random.shuffle(items_list, rng.random)
    return items_list


# TODO: justify cmp=False in docstring
@attrs(frozen=True, slots=True, cmp=False)
class TemplateObjectVariable(SituationTemplateObject):
    r"""
    A variable in a `Phase1SituationTemplateGenerator`
    which could be filled by any object
    whose `OntologyNode` is selected by *node_selector*.

    We provide *object_variable* to make creating `TemplateObjectVariable`\ s more convenient.
    """

    node_selector: OntologyNodeSelector = attrib(
        validator=instance_of(OntologyNodeSelector)
    )


def object_variable(
    debug_handle: str,
    root_node: OntologyNode = THING,
    with_properties: Iterable[OntologyNode] = immutableset(),
):
    r"""
    Create a `TemplateObjectVariable` with the specified *debug_handle*
    which can be filled by any object whose `OntologyNode` is a descendant of
    (or is exactly) *root_node*
    and which possesses all properties in *with_properties*.
    """
    return TemplateObjectVariable(
        debug_handle,
        ByHierarchyAndProperties(
            descendents_of=root_node,
            required_properties=with_properties,
            banned_properties=immutableset([ABSTRACT]),
        ),
    )


GAILA_PHASE_1_TEMPLATE_GENERATOR = Phase1SituationTemplateGenerator()
