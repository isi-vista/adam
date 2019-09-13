import os
from typing import List

from attr import attrs

from adam.experiment import InstanceGroup
from adam.language.dependency import LinearizedDependencyTree
from adam.perception import PerceptualRepresentation
from adam.perception.developmental_primitive_perception import (
    DevelopmentalPrimitivePerceptionFrame,
)
from adam.situation.high_level_semantics_situation import HighLevelSemanticsSituation


@attrs(frozen=True, slots=True)
class CurriculumToHtml:
    """
    Class to turn an `InstanceGroup` into an html document
    """

    @staticmethod
    def generate(
        instances: List[
            InstanceGroup[
                HighLevelSemanticsSituation,
                LinearizedDependencyTree,
                DevelopmentalPrimitivePerceptionFrame,
            ]
        ],
        outputdestination: str,
        title: str = "Instance Group",
        overwrite: bool = False,
    ):
        r"""
        Static method to take a list of `InstanceGroup`\ s and turns each one into an indivual page
        
        Given a list of InstanceGroups and an output directory of *outputdestination* along with 
        a *title* for the pages the generator loops through each group and calls the internal 
        method to create HTML pages. *overwrite* indicates is previously existing HTML files should
        be overwritten or not
        """
        for x in range(len(instances)):  # pylint:disable=consider-using-enumerate
            CurriculumToHtml._generate(
                instance=instances[x],
                outputdestination=f"{outputdestination}{title}{x}.html",
                title=title + f" {x}",
                overwrite=overwrite,
            )

    @staticmethod
    def _generate(
        instance: InstanceGroup[
            HighLevelSemanticsSituation,
            LinearizedDependencyTree,
            DevelopmentalPrimitivePerceptionFrame,
        ],
        title: str,
        overwrite: bool,
        outputdestination: str,
    ):
        """
        Internal generation method for individual instance groups into HTML pages

        Given an `InstanceGroup` with a `HighLevelSemanticsSituation`,
        `LinearizedDependencyTree`, and `DevelopmentalPrimitivePerceptionFrame` this function
        creates an html page at the given *outputdestination* and *title*. If the file already
        exists and *overwrite* is set to False an error is raised in execution. Each page turns an
        instance group with each "instance" as an indiviudal section on the page.

        No returns

        """
        if os.path.isfile(outputdestination) and not overwrite:
            raise RuntimeError(f"Not able to create new HTML file in {outputdestination}")
        html = open(outputdestination, "w")
        html.write(f"<h1>{title} - {instance.name()}</h1>\n")
        instance_number = 0
        for inst in instance.instances():
            if not isinstance(inst[0], HighLevelSemanticsSituation):
                raise RuntimeError(
                    f"Expected the Situation to be HighLevelSemanticsSituation got {type(inst[0])}"
                )
            if not isinstance(inst[1], LinearizedDependencyTree):
                raise RuntimeError(
                    f"Expected the Lingustics to be LinearizedDependencyTree got {type(inst[1])}"
                )
            if not isinstance(inst[2], PerceptualRepresentation) and not isinstance(
                inst[2].frames, DevelopmentalPrimitivePerceptionFrame
            ):
                raise RuntimeError(
                    f"Expected the Perceptual Representation to contain DevelopmentalPrimitivePerceptionFrame got {type(inst[2].frames)}"
                )
            html.write(
                f'<table>\n<thead><tr><th colspan="3"><h2>Scene {instance_number}</h2>\n</th></tr></thead>'
                f'<tbody><tr><td><h3 id="situation{instance_number}">Situation Description</h3></td>'
                f'<td><h3 id="lingustics{instance_number}">Linguistic Descrption</h3></td>'
                f'<td><h3 id="perception{instance_number}">Perception Description</h3></td></tr>\n<tr>'
            )
            html.writelines(CurriculumToHtml._situationtext(inst[0]))
            html.writelines(CurriculumToHtml._linguistictext(inst[1]))
            html.writelines(CurriculumToHtml._perceptiontext(inst[2]))
            html.write("</tr></tbody>")
            instance_number = instance_number + 1
        html.close()

    @staticmethod
    def _situationtext(situation: HighLevelSemanticsSituation):
        """
        Converts a situation description into its sub-parts as a table entry

        Receiving a `HighLevelSemanticsSituation` the objects, actions, and relations are displayed
        in a table entry. Returns a List[str]

        """
        outputtext = [f"<td>\n", "<h4>Objects</h4>\n<ul>"]
        for obj in situation.objects:
            outputtext.append(f"<li>{obj.ontology_node.handle}</li>")
            # Reintroduce Properties as [] around the object
            # outputtext.append(f"<li>Properties:\n<ul>")
            # for prop in obj.properties:
            #    outputtext.append(f"<li>{prop.handle}</li>")
            # outputtext.append("</ul></li>")
        outputtext.append("</ul>")
        if not situation.actions:
            outputtext.append("<h4>Actions</h4>\n<ul>")
            for acts in situation.actions:
                outputtext.append(f"<li>{acts.action_type.handle}</li>")
            outputtext.append("</ul>")
        if not situation.relations:
            outputtext.append("<h4>Relations</h4>\n<ul>")
            for rel in situation.relations:
                outputtext.append(
                    f"<li>{rel.relation_type.handle}({rel.first_slot.ontology_node.handle},{rel.second_slot.ontology_node.handle})</li>"
                )
            outputtext.append("</ul>")
        outputtext.append("</td>")
        return outputtext

    @staticmethod
    def _perceptiontext(
        perception: PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]
    ):
        """
        Turns a perception into a list of items in the perceptions frames.

        Given a `PerceptualRepresentation[DevelopmentalPrimitivePerceptionFrame]` the information
        within is converted into a table entry with headings for the informaiton contained within
        the example. Returns a List[str]

        """
        outputtext = [f"<td>"]
        frame_number = 0
        for frame in perception.frames:
            outputtext.append(f"<h4>Frame {frame_number}</h4>")
            outputtext.append("<h5>Perceived Objects</h5>\n<ul>")
            for obj in frame.perceived_objects:
                outputtext.append(f"<li>{obj.debug_handle}</li>")
            outputtext.append("</ul>")
            if not frame.property_assertions:
                outputtext.append("<h5>Property Assertions</h5>\n<ul>")
                for prop in frame.property_assertions:
                    outputtext.append(f"<li>{prop}")
                outputtext.append("</ul>")
            if not frame.relations:
                outputtext.append("<h5>Relations</h5>\n<ul>")
                for rel in frame.relations:
                    outputtext.append(
                        f"<li>{rel.relation_type.handle}({rel.arg1},{rel.arg2})"
                    )
                outputtext.append("</ul>")
            frame_number = frame_number + 1
        outputtext.append("</td>")
        return outputtext

    @staticmethod
    def _linguistictext(linguistic: LinearizedDependencyTree):
        """
        Parses the Linguistic Description of a Linearized Dependency Tree into a table entry

        Takes a `LinearizedDependencyTree` which is turned into a token sequence and
        phrased as a sentence for display. Returns a List[str]
        """
        outputtext = [f"<td>", " ".join(linguistic.as_token_sequence()), "</td>"]
        return outputtext
