from attr import attrs, attrib
from attr.validators import instance_of
from immutablecollections import ImmutableSet, immutableset
from immutablecollections.converter_utils import _to_immutableset

from adam.ontology import OntologyNode, Ontology
from adam.ontology.phase1_ontology import RECOGNIZED_PARTICULAR
from adam.perception import PerceptualRepresentationFrame


@attrs(slots=True, frozen=True, repr=False)
class DevelopmentalPrimitivePerception(PerceptualRepresentationFrame):
    perceived_objects: ImmutableSet["DevelopmentalPrimitiveObject"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )
    property_assertions: ImmutableSet["DevelopmentalPrimitivePropertyAssertion"] = attrib(
        converter=_to_immutableset, default=immutableset()
    )


@attrs(slots=True, frozen=True, repr=False)
class DevelopmentalPrimitiveObject:
    debug_handle: str = attrib(validator=instance_of(str))

    def __repr__(self) -> str:
        return self.debug_handle


@attrs(slots=True, frozen=True, repr=False)
class DevelopmentalPrimitivePerceivableFlagProperty:
    debug_handle: str = attrib(validator=instance_of(str))

    def __repr__(self) -> str:
        return f"+{self.debug_handle}"


SENTIENT = DevelopmentalPrimitivePerceivableFlagProperty("sentient")


class DevelopmentalPrimitivePropertyAssertion:
    pass


@attrs(slots=True, frozen=True, repr=False)
class HasProperty(DevelopmentalPrimitivePropertyAssertion):
    perceived_object = attrib(validator=instance_of(DevelopmentalPrimitiveObject))
    property = attrib(
        validator=instance_of(DevelopmentalPrimitivePerceivableFlagProperty)
    )

    def __repr__(self) -> str:
        return f"hasProperty({self.perceived_object}, {self.property}"


@attrs(slots=True, frozen=True, repr=False)
class IsRecognizedParticular(DevelopmentalPrimitivePropertyAssertion):
    ontology: Ontology = attrib(validator=instance_of(Ontology))
    perceived_object: DevelopmentalPrimitiveObject = attrib(
        validator=instance_of(DevelopmentalPrimitiveObject)
    )
    particular_ontology_node: OntologyNode = attrib(validator=instance_of(OntologyNode))

    def __attrs_post_init__(self) -> None:
        if not self.ontology.has_all_properties(
            self.particular_ontology_node, [RECOGNIZED_PARTICULAR]
        ):
            raise RuntimeError(
                "The learner can only perceive the ontology node of an object "
                "if it is a recognized particular (e.g. Mom, Dad)"
            )

    def __repr__(self) -> str:
        return f"recognizedAs({self.perceived_object}, {self.particular_ontology_node})"


# CONSTANT
# Mom isa person
# Person has right leg
# Person has left leg
# Person is (size big)
#  [“is” means “has property”]
# Leg has foot
# Person has right arm
# Person has left arm
# Arm has hand
# Hand has finger1
# Hand has finger2
# Person is Sentient [etc]
# …
# [links from description above to Marr Geons]
# …
#
# Ball1 isa ball
# Ball1 is blue
# Ball1 is (size medium)
# Box1 isa box
# Box1 is green
# Box1 is (size medium)
# Mom bigger than Ball1
# BEFORE
# ((Right arm of Mom) and (Left arm of Mom))   supports Ball1
# [I’m assuming here that “supports” is a primitive.]
# [From Biederman:]
#
# Ball1 above Hand of right arm of Mom
# Ball1 contacts Hand of right arm of Mom
# Ball1 above Hand of left arm of Mom
# Ball1 contacts Hand of left arm of Mom
# Mom left of Box1
#
# AFTER
# Top of Box1 supports Ball1
# Ball1 above Box1
# Ball1 contacts Box1
# Mom left of Box1
# Mom beside Box1
#
# DELTA
# Path(Mom, * right of Box1, * beside Box1)
# Path(Hand of right arm of Mom,
#           Region(right of  head of Mom),
#            Region(below head of Mom)) [not quite right]
# Path(Ball1,
#         Region(above Hand of right arm of Mom),
#          Region( above Box1))
# SEMANTICS
# Put(Agent: Mom, Theme: Ball1, Goal(On(Box1)))
# Move(Agent:Mom,Theme: Ball1)
# Move(Theme: right arm of Mom and left arm of
#     Mom)
