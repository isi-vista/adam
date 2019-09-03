from attr import attrs, attrib
from attr.validators import instance_of

from adam.ontology import OntologyNode, Ontology
from adam.ontology.phase1_ontology import RECOGNIZED_PARTICULAR
from adam.perception import PerceptualRepresentationFrame


class DevelopmentalPrimitivePerception(PerceptualRepresentationFrame):
    pass


@attrs(slots=True, frozen=True, repr=False)
class DevelopmentalPrimitiveObject:
    debug_handle: str = attrib(validator=instance_of(str))

    def __repr__(self) -> str:
        return self.debug_handle


class IsRecognizedParticular:
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
